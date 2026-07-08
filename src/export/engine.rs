use std::sync::Arc;
use crate::export::job::ExportJob;
use crate::export::partitioner::SegmentPartitioner;
use crate::export::renderer::ExportRenderer;
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::video_encoder::{VideoEncoder, VideoEncoderBackend};
use crate::export::audio_encoder::AudioMuxEncoder;
use crate::export::muxer::Muxer;
use crate::export::progress::{progress_channel, ProgressReceiver, ProgressSender};
use crate::render::device::GpuDevice;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::io::ffi::avutil::AVRational;
use crate::timeline::store::TimelineStore;
use crate::timeline::source::SourceRegistry;
use crate::render::shader::registry::ShaderRegistry;
use crate::render::compute::ComputePipelineCache;
use crate::interop::capability::InteropCapability;
use crate::interop::cuda_context::CudaContext;
use crate::render::resource::ResourceId;

pub struct ExportEngine {
    device:      Arc<GpuDevice>,
    job:         Arc<ExportJob>,
    scheduler:   Arc<FrameScheduler>,
    timeline:    Arc<std::sync::RwLock<TimelineStore>>,
    sources:     Arc<std::sync::RwLock<SourceRegistry>>,
    capability:  InteropCapability,
    cuda_ctx:    Option<Arc<CudaContext>>,
    force_cpu:   bool,
}

impl ExportEngine {
    pub fn new(
        device:     Arc<GpuDevice>,
        job:        ExportJob,
        scheduler:  Arc<FrameScheduler>,
        timeline:   Arc<std::sync::RwLock<TimelineStore>>,
        sources:    Arc<std::sync::RwLock<SourceRegistry>>,
        capability: InteropCapability,
        cuda_ctx:   Option<Arc<CudaContext>>,
        force_cpu:  bool,
    ) -> Self {
        Self { device, job: Arc::new(job), scheduler, timeline, sources, capability, cuda_ctx, force_cpu }
    }

    pub fn start(
        self,
        shaders: Arc<ShaderRegistry>,
        compute_cache: Arc<ComputePipelineCache>,
    ) -> Result<ProgressReceiver, ExportError> {
        self.job.validate().map_err(ExportError::JobInvalid)?;

        let (prog_tx, prog_rx) = progress_channel(self.job.total_frames());

        // Select NVENC (GPU) or libx264/libx265 (CPU) encoder based on hardware
        // availability and user preference. force_cpu overrides auto-detection.
        let video_enc = if self.force_cpu {
            log::info!("[export] backend: FfmpegCpu (forced by user)");
            VideoEncoderBackend::FfmpegCpu(VideoEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?)
        } else {
            let backend = VideoEncoderBackend::select(
                &self.job,
                &self.capability,
                self.cuda_ctx.as_ref().map(|c| c.as_ref()),
                &self.device,
            ).map_err(ExportError::EncoderOpen)?;
            match &backend {
                VideoEncoderBackend::CudaNvenc { .. } => log::info!("[export] backend: CudaNvenc (NVENC)"),
                VideoEncoderBackend::FfmpegCpu(_)     => log::info!("[export] backend: FfmpegCpu (NVENC not available, fallback)"),
            }
            backend
        };
        let audio_enc = AudioMuxEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?;

        let enc_video_tb = AVRational { num: self.job.frame_rate.den as i32, den: self.job.frame_rate.num as i32 };
        let enc_audio_tb = AVRational { num: 1, den: 48000 };

        let muxer = Arc::new(Muxer::open(&self.job, &video_enc, &audio_enc, enc_video_tb, enc_audio_tb)
            .map_err(ExportError::MuxerOpen)?);

        let is_gpu = matches!(video_enc, VideoEncoderBackend::CudaNvenc { .. });
        let mut video_enc_opt = Some(video_enc);

        let backend = if is_gpu {
            let repack = crate::interop::encode_interop::Abgr10RepackNode::new(
                &self.device,
                ResourceId::FINAL_COLOR,
                ResourceId::FINAL_COLOR,
            );
            crate::export::renderer::ExportBackend::GpuNvenc {
                video_enc: video_enc_opt.take().unwrap(),
                muxer: Arc::clone(&muxer),
                repack,
            }
        } else {
            let readback = crate::export::readback::FrameReadback::new(&self.device, self.job.width, self.job.height)
                .map_err(|e| ExportError::EncoderOpen(crate::export::video_encoder::EncodeError::Interop(e)))?;
            crate::export::renderer::ExportBackend::Cpu { readback }
        };

        let queue = Arc::new(EncoderQueue::new());
        let timeline_clone = Arc::clone(&self.timeline);
        let sources_clone = Arc::clone(&self.sources);
        let segments = SegmentPartitioner::partition(&self.job);

        let job_clone     = Arc::clone(&self.job);
        let queue_clone   = Arc::clone(&queue);
        let device_clone  = Arc::clone(&self.device);
        let scheduler_clone = Arc::clone(&self.scheduler);
        let shaders_clone = Arc::clone(&shaders);
        let compute_cache_clone = Arc::clone(&compute_cache);

        // Spawn encoder thread (video) if FfmpegCpu is active
        if !is_gpu {
            let video_enc = video_enc_opt.take().unwrap();
            let prog_tx_enc = prog_tx.clone();
            let muxer_for_video = Arc::clone(&muxer);
            std::thread::Builder::new().name("ve-encoder".into()).spawn(move || {
                Self::encoder_thread(queue_clone, video_enc, muxer_for_video, prog_tx_enc);
            }).map_err(ExportError::ThreadSpawn)?;
        }

        // Spawn audio encode thread — decodes audio from source clips and muxes it.
        let job_clone2       = Arc::clone(&self.job);
        let muxer_for_audio  = Arc::clone(&muxer);
        let timeline_audio   = Arc::clone(&self.timeline);
        let sources_audio    = Arc::clone(&self.sources);
        std::thread::Builder::new().name("ve-audio-enc".into()).spawn(move || {
            Self::audio_thread(job_clone2, timeline_audio, sources_audio, audio_enc, muxer_for_audio);
        }).map_err(ExportError::ThreadSpawn)?;

        // Spawn render/dispatch thread
        let muxer_for_dispatch = Arc::clone(&muxer);
        std::thread::Builder::new().name("ve-export-dispatch".into()).spawn(move || {
            log::info!("[export] dispatch thread started, {} segment(s)", segments.len());

            // Keep a clone for the panic-recovery path (the inner `move` closure owns `queue`).
            let queue_err = Arc::clone(&queue);
            let is_gpu_thread = is_gpu;
            let muxer_for_dispatch_clone = Arc::clone(&muxer_for_dispatch);

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut renderer = ExportRenderer::new(
                    Arc::clone(&device_clone),
                    Arc::clone(&scheduler_clone),
                    Arc::clone(&job_clone),
                    Arc::clone(&timeline_clone),
                    Arc::clone(&sources_clone),
                    shaders_clone,
                    compute_cache_clone,
                    backend,
                );

                for (i, seg) in segments.iter().enumerate() {
                    log::info!("[export] starting segment {i}/{}", segments.len());
                    renderer.render_segment(seg, &queue, &prog_tx)
                        .expect(&format!("render_segment {i} failed"));
                    log::info!("[export] segment {i} complete");
                }
                queue.push(QueueItem::AllDone);
                log::info!("[export] dispatch thread finished — AllDone sent");

                if is_gpu_thread {
                    // Flush the video encoder (NVENC EOS flush)
                    if let crate::export::renderer::ExportBackend::GpuNvenc { ref mut video_enc, .. } = renderer.backend {
                        let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                            muxer_for_dispatch_clone.write_packet(pkt, true).unwrap();
                        };
                        video_enc.flush(&mut sink).unwrap();
                    }

                    // Spin-wait until audio thread finishes and drops its Arc<Muxer>
                    while Arc::strong_count(&muxer_for_dispatch_clone) > 1 {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    }
                    if let Ok(m) = Arc::try_unwrap(muxer_for_dispatch_clone) {
                        m.finalise().unwrap();
                    }
                }
            }));

            if let Err(e) = result {
                let msg = if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else {
                    "(unknown panic payload)".to_string()
                };
                log::error!("[export] dispatch thread PANICKED: {msg}");
                // Ensure the encoder thread unblocks
                queue_err.push(QueueItem::AllDone);
            }
        }).map_err(ExportError::ThreadSpawn)?;

        Ok(prog_rx)
    }

    fn encoder_thread(
        queue:         Arc<EncoderQueue>,
        mut video_enc: VideoEncoderBackend,
        muxer:         Arc<Muxer>,
        prog_tx:       ProgressSender,
    ) {
        use crate::export::progress::ExportPhase;
        let mut frames_encoded = 0;
        loop {
            match queue.pop() {
                Some(QueueItem::Frame(raw)) => {
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        muxer.write_packet(pkt, true).unwrap();
                    };
                    video_enc.encode_frame(&raw, &mut sink).unwrap();
                    frames_encoded += 1;
                    prog_tx.report(frames_encoded, ExportPhase::Encoding);
                }
                Some(QueueItem::SegmentDone { .. }) => {}
                Some(QueueItem::AllDone) | None => {
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        muxer.write_packet(pkt, true).unwrap();
                    };
                    video_enc.flush(&mut sink).unwrap();
                    prog_tx.report(frames_encoded, ExportPhase::Done);
                    break;
                }
            }
        }
        
        while Arc::strong_count(&muxer) > 1 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        if let Ok(m) = Arc::try_unwrap(muxer) {
            m.finalise().unwrap();
        }
    }

    /// Decode audio from all clips in the export range and encode into the muxer.
    fn audio_thread(
        job:      Arc<ExportJob>,
        timeline: Arc<std::sync::RwLock<TimelineStore>>,
        sources:  Arc<std::sync::RwLock<SourceRegistry>>,
        mut audio_enc: AudioMuxEncoder,
        muxer:    Arc<Muxer>,
    ) {
        use crate::io::demuxer::Demuxer;
        use crate::io::decoder::Decoder;
        use crate::io::ffi::avutil::{
            av_frame_alloc, av_frame_free,
            av_frame_get_data, av_frame_get_nb_samples,
        };
        use crate::audio::ffi::avresample::{
            swr_alloc_set_opts, swr_init, swr_free, swr_convert, swr_get_delay,
            AV_SAMPLE_FMT_FLTP, AV_CH_LAYOUT_STEREO,
        };
        use crate::io::ffi::avcodec::{
            avcodec_send_packet, avcodec_receive_frame,
            avcodec_ctx_get_sample_rate, avcodec_ctx_get_channel_layout,
            avcodec_ctx_get_channels, avcodec_ctx_get_sample_fmt,
        };
        use crate::io::ffi::avutil::AVERROR_EOF;

        // Collect clips with audio that overlap the export range, sorted by timeline position
        let clip_list: Vec<(crate::timeline::ids::SourceId, i64, i64, i64, f32)> = {
            let store = timeline.read().unwrap();
            let srcs  = sources.read().unwrap();
            let n = store.len();
            let mut clips = Vec::new();
            for i in 0..n {
                let t_in  = store.pts_in_at(i);
                let t_out = store.pts_out_at(i);
                let src   = store.source_id_at(i);
                let s_in  = store.source_in_at(i);
                let speed = store.speed_at(i);
                // Skip clips outside export range
                if t_out <= job.pts_in || t_in >= job.pts_out {
                    continue;
                }
                // Skip clips whose source has no audio
                if srcs.audio_info(src).is_err() {
                    continue;
                }
                clips.push((src, t_in, t_out, s_in, speed));
            }
            // Sort by timeline in-point
            clips.sort_by_key(|&(_, t_in, _, _, _)| t_in);
            clips
        };

        // Packet sink for muxing audio packets
        let mut mux_sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
            muxer.write_packet(pkt, false).unwrap();
        };

        let frame_size = audio_enc.frame_size();
        // Accumulation buffers in interleaved f32 (left/right alternating)
        let mut accum_left:  Vec<f32> = Vec::new();
        let mut accum_right: Vec<f32> = Vec::new();
        let mut enc_pts: i64 = 0; // in samples @ 48 kHz

        // Reusable out buffer for swr_convert (2 planes of `frame_size` f32)
        let swr_out_capacity = frame_size * 2 + 64;
        let mut swr_left:  Vec<f32> = vec![0.0; swr_out_capacity];
        let mut swr_right: Vec<f32> = vec![0.0; swr_out_capacity];

        for (src_id, clip_t_in, clip_t_out, src_material_in, speed) in clip_list {
            let path = {
                let srcs = sources.read().unwrap();
                match srcs.path(src_id) {
                    Some(p) => (*p).clone(),
                    None    => continue,
                }
            };

            let mut demuxer = match Demuxer::open(&path) {
                Ok(d)  => d,
                Err(_) => continue,
            };

            let audio_info = match demuxer.audio_stream().cloned() {
                Some(s) => s,
                None    => continue,
            };

            let mut decoder = match Decoder::open(&audio_info, audio_info.codecpar, false) {
                Ok(d)  => d,
                Err(_) => continue,
            };

            // Set up SwrContext for this source
            let (in_ch_layout, in_sample_fmt, mut in_sample_rate) = unsafe {
                let ctx = decoder.ctx();
                let sr   = avcodec_ctx_get_sample_rate(ctx);
                let mut cl = avcodec_ctx_get_channel_layout(ctx);
                let channels = avcodec_ctx_get_channels(ctx);
                let fmt  = avcodec_ctx_get_sample_fmt(ctx);
                if cl == 0 {
                    cl = if channels == 1 { 4 } else { 3 };
                }
                (cl as i64, fmt as i32, sr as i32)
            };

            // Adjust input sample rate to stretch/squash the audio according to speed
            in_sample_rate = (in_sample_rate as f32 * speed).round() as i32;

            let swr = unsafe {
                let s = swr_alloc_set_opts(
                    std::ptr::null_mut(),
                    AV_CH_LAYOUT_STEREO as i64, AV_SAMPLE_FMT_FLTP, 48_000,
                    in_ch_layout, in_sample_fmt, in_sample_rate,
                    0, std::ptr::null_mut(),
                );
                if s.is_null() { continue; }
                if swr_init(s) < 0 { continue; }
                s
            };

            let frame = unsafe { av_frame_alloc() };
            if frame.is_null() {
                unsafe { swr_free(&mut { swr }); }
                continue;
            }

            // Effective export overlap for this clip
            let eff_t_in  = clip_t_in.max(job.pts_in);
            let eff_t_out = clip_t_out.min(job.pts_out);

            // Translate effective timeline range to source material range (90 kHz project TB)
            let src_seek_pts   = src_material_in + crate::timeline::rational::speed_scale_pts(eff_t_in - clip_t_in, speed);
            let src_end_pts    = src_material_in + crate::timeline::rational::speed_scale_pts(eff_t_out - clip_t_in, speed);

            // Seek demuxer to just before the start of required audio
            let _ = demuxer.seek(src_seek_pts, job.project_tb);
            decoder.flush();

            // Convert PTS bounds to stream timebase
            let stream_seek_pts = job.project_tb.rescale_pts(src_seek_pts, audio_info.time_base);
            let stream_end_pts  = job.project_tb.rescale_pts(src_end_pts,  audio_info.time_base);

            // ALWAYS discard audio packets until we reach the target PTS
            if src_seek_pts > 0 {
                while let Ok(Some(pkt)) = demuxer.next_audio_packet() {
                    if pkt.pts != i64::MIN && pkt.pts >= stream_seek_pts {
                        // Found the first packet we need. We must send it to the decoder
                        // since we already read/consumed it.
                        unsafe {
                            let _ = avcodec_send_packet(decoder.ctx(), pkt.as_ptr());
                        }
                        break;
                    }
                }
            }

            // Decode all audio packets in the needed range
            'decode: loop {
                let pkt = match demuxer.next_audio_packet() {
                    Ok(Some(p)) => p,
                    _           => break,
                };

                if pkt.pts != i64::MIN && pkt.pts > stream_end_pts {
                    break;
                }

                unsafe {
                    let send_ret = avcodec_send_packet(decoder.ctx(), pkt.as_ptr());
                    if send_ret < 0 { break 'decode; }

                    loop {
                        let recv_ret = avcodec_receive_frame(decoder.ctx(), frame);
                        if recv_ret == crate::io::ffi::avutil::AVERROR_EAGAIN || recv_ret == AVERROR_EOF { break; }
                        if recv_ret < 0 { break; }

                        let nb = av_frame_get_nb_samples(frame) as usize;
                        if nb == 0 { continue; }

                        // How many output samples swr will produce
                        let out_max = (nb as i64 * 48_000 as i64 / in_sample_rate as i64 + 32) as usize;
                        if swr_left.len() < out_max {
                            swr_left.resize(out_max, 0.0);
                            swr_right.resize(out_max, 0.0);
                        }

                        let in_data = av_frame_get_data(frame) as *const *const u8;
                        let mut out_planes: [*mut u8; 2] = [
                            swr_left.as_mut_ptr()  as *mut u8,
                            swr_right.as_mut_ptr() as *mut u8,
                        ];
                        let converted = swr_convert(
                            swr,
                            out_planes.as_mut_ptr(),
                            out_max as i32,
                            in_data,
                            nb as i32,
                        );
                        if converted <= 0 { continue; }

                        accum_left.extend_from_slice(&swr_left[..converted as usize]);
                        accum_right.extend_from_slice(&swr_right[..converted as usize]);

                        // Flush full frames out of accumulator
                        while accum_left.len() >= frame_size {
                            let left_chunk:  Vec<f32> = accum_left.drain(..frame_size).collect();
                            let right_chunk: Vec<f32> = accum_right.drain(..frame_size).collect();
                            let _ = audio_enc.encode_pcm_chunk(&left_chunk, &right_chunk, enc_pts, &mut mux_sink);
                            enc_pts += frame_size as i64;
                        }
                    }
                }
            }


            // Flush swr delay for this clip
            unsafe {
                loop {
                    let delay = swr_get_delay(swr, 48_000);
                    if delay <= 0 { break; }
                    let out_max = (delay + 16) as usize;
                    if swr_left.len() < out_max { swr_left.resize(out_max, 0.0); swr_right.resize(out_max, 0.0); }
                    let mut out_planes: [*mut u8; 2] = [
                        swr_left.as_mut_ptr()  as *mut u8,
                        swr_right.as_mut_ptr() as *mut u8,
                    ];
                    let converted = swr_convert(swr, out_planes.as_mut_ptr(), out_max as i32, std::ptr::null(), 0);
                    if converted <= 0 { break; }
                    accum_left.extend_from_slice(&swr_left[..converted as usize]);
                    accum_right.extend_from_slice(&swr_right[..converted as usize]);
                    while accum_left.len() >= frame_size {
                        let l: Vec<f32> = accum_left.drain(..frame_size).collect();
                        let r: Vec<f32> = accum_right.drain(..frame_size).collect();
                        let _ = audio_enc.encode_pcm_chunk(&l, &r, enc_pts, &mut mux_sink);
                        enc_pts += frame_size as i64;
                    }
                }

                av_frame_free(&mut { frame });
                swr_free(&mut { swr });
            }
        }

        // Flush remaining partial frame with silence padding
        if !accum_left.is_empty() {
            accum_left.resize(frame_size, 0.0);
            accum_right.resize(frame_size, 0.0);
            let _ = audio_enc.encode_pcm_chunk(&accum_left, &accum_right, enc_pts, &mut mux_sink);
        }

        // Flush the encoder
        let _ = audio_enc.encode_all(&job, &mut mux_sink);
    }
}

#[derive(Debug)]
pub enum ExportError {
    JobInvalid(crate::export::job::JobError),
    EncoderOpen(crate::export::video_encoder::EncodeError),
    MuxerOpen(crate::export::muxer::MuxError),
    ThreadSpawn(std::io::Error),
}
