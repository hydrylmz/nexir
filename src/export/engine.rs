use crate::export::audio_encoder::AudioMuxEncoder;
use crate::export::job::ExportJob;
use crate::export::muxer::Muxer;
use crate::export::partitioner::SegmentPartitioner;
use crate::export::progress::{progress_channel, ExportPhase, ProgressReceiver, ProgressSender};
use crate::export::queue::{EncoderQueue, QueueItem};
use crate::export::renderer::ExportRenderer;
use crate::export::video_encoder::{VideoEncoder, VideoEncoderBackend};
use crate::interop::capability::InteropCapability;
use crate::interop::cuda_context::CudaContext;
use crate::io::ffi::avutil::AVRational;
use crate::render::compute::ComputePipelineCache;
use crate::render::device::GpuDevice;
use crate::render::shader::registry::ShaderRegistry;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::timeline::source::SourceRegistry;
use crate::timeline::store::TimelineStore;
use crate::timeline::track::TrackList;
use std::sync::Arc;

pub struct ExportEngine {
    device: Arc<GpuDevice>,
    job: Arc<ExportJob>,
    scheduler: Arc<FrameScheduler>,
    timeline: Arc<std::sync::RwLock<TimelineStore>>,
    tracks: Arc<std::sync::RwLock<TrackList>>,
    sources: Arc<std::sync::RwLock<SourceRegistry>>,
    capability: InteropCapability,
    cuda_ctx: Option<Arc<CudaContext>>,
    force_cpu: bool,
}

impl ExportEngine {
    pub fn new(
        device: Arc<GpuDevice>,
        job: ExportJob,
        scheduler: Arc<FrameScheduler>,
        timeline: Arc<std::sync::RwLock<TimelineStore>>,
        tracks: Arc<std::sync::RwLock<TrackList>>,
        sources: Arc<std::sync::RwLock<SourceRegistry>>,
        capability: InteropCapability,
        cuda_ctx: Option<Arc<CudaContext>>,
        force_cpu: bool,
    ) -> Self {
        Self {
            device,
            job: Arc::new(job),
            scheduler,
            timeline,
            tracks,
            sources,
            capability,
            cuda_ctx,
            force_cpu,
        }
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
            log::info!("[export] backend: FfmpegEncoder (forced by user)");
            VideoEncoderBackend::FfmpegEncoder(
                VideoEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?,
            )
        } else {
            let backend = VideoEncoderBackend::select(
                &self.job,
                &self.capability,
                self.cuda_ctx.as_ref(),
                &self.device,
            )
            .map_err(ExportError::EncoderOpen)?;
            match &backend {
                VideoEncoderBackend::CudaNvenc { .. } => {
                    log::info!("[export] backend: CudaNvenc (NVENC)")
                }
                VideoEncoderBackend::FfmpegEncoder(_) => log::info!(
                    "[export] backend: FfmpegEncoder (libavcodec; resolves \
                     h264_nvenc/hevc_nvenc first, so this is still a GPU encode \
                     when one opens — see the reason logged by select())"
                ),
            }
            backend
        };
        let audio_enc = AudioMuxEncoder::open(&self.job).map_err(ExportError::EncoderOpen)?;

        let enc_video_tb = AVRational {
            num: self.job.frame_rate.den as i32,
            den: self.job.frame_rate.num as i32,
        };
        let enc_audio_tb = AVRational { num: 1, den: 48000 };

        let muxer = Arc::new(
            Muxer::open(
                &self.job,
                &video_enc,
                &audio_enc,
                enc_video_tb,
                enc_audio_tb,
            )
            .map_err(ExportError::MuxerOpen)?,
        );

        let is_gpu = matches!(video_enc, VideoEncoderBackend::CudaNvenc { .. });
        let mut video_enc_opt = Some(video_enc);

        let backend = if is_gpu {
            // RGB → NV12 in our own shader.  Built from the JOB's output colour
            // description, which is the whole point: NVENC receives YUV that
            // already carries this matrix and performs no conversion of its own,
            // so `Muxer::open`'s tags are authoritative by construction rather
            // than a claim about what the driver happened to do.
            let nv12 = crate::interop::nv12_encode::Nv12EncodeNode::new(
                &self.device,
                self.job.output_color,
                self.job.width,
                self.job.height,
            );
            crate::export::renderer::ExportBackend::GpuNvenc {
                video_enc: video_enc_opt.take().unwrap(),
                muxer: Arc::clone(&muxer),
                nv12,
            }
        } else {
            let readback = crate::export::readback::FrameReadback::new(
                &self.device,
                self.job.width,
                self.job.height,
            )
            .map_err(|e| {
                ExportError::EncoderOpen(crate::export::video_encoder::EncodeError::Interop(e))
            })?;
            crate::export::renderer::ExportBackend::Cpu { readback }
        };

        let queue = Arc::new(EncoderQueue::new());
        let timeline_clone = Arc::clone(&self.timeline);
        let tracks_clone = Arc::clone(&self.tracks);
        let sources_clone = Arc::clone(&self.sources);
        let segments = SegmentPartitioner::partition(&self.job);

        let job_clone = Arc::clone(&self.job);
        let queue_clone = Arc::clone(&queue);
        let device_clone = Arc::clone(&self.device);
        let scheduler_clone = Arc::clone(&self.scheduler);
        let shaders_clone = Arc::clone(&shaders);
        let compute_cache_clone = Arc::clone(&compute_cache);

        // Spawn encoder thread (video) if FfmpegEncoder is active
        if !is_gpu {
            let video_enc = video_enc_opt.take().unwrap();
            let prog_tx_enc = prog_tx.clone();
            let muxer_for_video = Arc::clone(&muxer);
            std::thread::Builder::new()
                .name("ve-encoder".into())
                .spawn(move || {
                    Self::encoder_thread(queue_clone, video_enc, muxer_for_video, prog_tx_enc);
                })
                .map_err(ExportError::ThreadSpawn)?;
        }

        // Spawn audio encode thread — decodes audio from source clips and muxes it.
        let job_clone2 = Arc::clone(&self.job);
        let muxer_for_audio = Arc::clone(&muxer);
        let timeline_audio = Arc::clone(&self.timeline);
        let tracks_audio = Arc::clone(&self.tracks);
        let sources_audio = Arc::clone(&self.sources);
        let prog_tx_audio = prog_tx.clone();
        std::thread::Builder::new()
            .name("ve-audio-enc".into())
            .spawn(move || {
                Self::audio_thread(
                    job_clone2,
                    timeline_audio,
                    tracks_audio,
                    sources_audio,
                    audio_enc,
                    muxer_for_audio,
                    prog_tx_audio,
                );
            })
            .map_err(ExportError::ThreadSpawn)?;

        // Spawn render/dispatch thread
        let muxer_for_dispatch = Arc::clone(&muxer);
        let prog_tx_dispatch = prog_tx.clone();
        let total_frames_count = self.job.total_frames();
        std::thread::Builder::new()
            .name("ve-export-dispatch".into())
            .spawn(move || {
                log::info!(
                    "[export] dispatch thread started, {} segment(s)",
                    segments.len()
                );

                // Keep a clone for the panic-recovery path (the inner `move` closure owns `queue`).
                let queue_err = Arc::clone(&queue);
                let is_gpu_thread = is_gpu;
                let muxer_for_dispatch_clone = Arc::clone(&muxer_for_dispatch);

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || -> Result<(), String> {
                    let mut renderer = ExportRenderer::new(
                        Arc::clone(&device_clone),
                        Arc::clone(&scheduler_clone),
                        Arc::clone(&job_clone),
                        Arc::clone(&timeline_clone),
                        Arc::clone(&tracks_clone),
                        Arc::clone(&sources_clone),
                        shaders_clone,
                        compute_cache_clone,
                        backend,
                    );

                    for (i, seg) in segments.iter().enumerate() {
                        if prog_tx.control().is_cancelled() {
                            log::info!("[export] dispatch detected cancellation before segment {i}");
                            queue.push(QueueItem::AllDone);
                            prog_tx.report(renderer.frames_done, ExportPhase::Cancelled);
                            return Ok(());
                        }

                        log::info!("[export] starting segment {i}/{}", segments.len());
                        renderer
                            .render_segment(seg, &queue, &prog_tx)
                            .map_err(|e| format!("render_segment {i} failed: {e:?}"))?;
                        log::info!("[export] segment {i} complete");
                    }
                    queue.push(QueueItem::AllDone);
                    log::info!("[export] dispatch thread finished — AllDone sent");

                    if is_gpu_thread {
                        // Flush the video encoder (NVENC EOS flush)
                        if let crate::export::renderer::ExportBackend::GpuNvenc {
                            ref mut video_enc,
                            ..
                        } = renderer.backend
                        {
                            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                                if let Err(e) = muxer_for_dispatch_clone.write_packet(pkt, true) {
                                    log::error!("[export] NVENC flush mux write failed: {e:?}");
                                }
                            };
                            video_enc.flush(&mut sink)
                                .map_err(|e| format!("NVENC flush failed: {e:?}"))?;
                        }

                        // Thread-safe synchronous finalisation
                        if let Err(e) = muxer_for_dispatch_clone.finalise_sync() {
                            log::error!("[export] muxer finalise failed: {e:?}");
                        }
                    }
                    Ok(())
                }));

                match result {
                    Ok(Ok(())) => {
                        if is_gpu && !prog_tx_dispatch.control().is_cancelled() {
                            prog_tx_dispatch.report(total_frames_count, ExportPhase::Done);
                        }
                    }
                    Ok(Err(e)) => {
                        log::error!("[export] dispatch thread error: {e}");
                        prog_tx_dispatch.report(0, ExportPhase::Failed(e.clone()));
                        queue_err.push(QueueItem::AllDone);
                    }
                    Err(e) => {
                        let msg = if let Some(s) = e.downcast_ref::<String>() {
                            s.clone()
                        } else if let Some(s) = e.downcast_ref::<&str>() {
                            s.to_string()
                        } else {
                            "(unknown panic payload)".to_string()
                        };
                        log::error!("[export] dispatch thread PANICKED: {msg}");
                        prog_tx_dispatch.report(0, ExportPhase::Failed(msg));
                        queue_err.push(QueueItem::AllDone);
                    }
                }
            })
            .map_err(ExportError::ThreadSpawn)?;

        Ok(prog_rx)
    }

    fn encoder_thread(
        queue: Arc<EncoderQueue>,
        mut video_enc: VideoEncoderBackend,
        muxer: Arc<Muxer>,
        prog_tx: ProgressSender,
    ) {
        use crate::export::progress::ExportPhase;
        let mut frames_encoded = 0;

        // Whether the encode ran to completion, as opposed to being cancelled.
        //
        // `Done` is NOT reported inside the loop: the file has no trailer until
        // `finalise_sync` below, and mp4 keeps the `moov` atom until then, so a
        // caller that opens the path the moment it sees `Done` finds
        // "Invalid data found when processing input". That is exactly what
        // `tests::export_validation` does, and it only surfaced as a flake under
        // parallel load — the window is the few milliseconds between the two.
        // The NVENC dispatch thread in `start()` already finalises before
        // reporting; this is the CPU path being brought in line.
        let mut completed = false;

        let mut run = || -> Result<(), String> {
            loop {
                if prog_tx.control().is_cancelled() {
                    log::info!("[export] encoder thread cancelled");
                    prog_tx.report(frames_encoded, ExportPhase::Cancelled);
                    break;
                }
                while prog_tx.control().is_paused() {
                    if prog_tx.control().is_cancelled() {
                        prog_tx.report(frames_encoded, ExportPhase::Cancelled);
                        return Ok(());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                match queue.pop() {
                    Some(QueueItem::Frame(raw)) => {
                        let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                            if let Err(e) = muxer.write_packet(pkt, true) {
                                log::error!("[encoder] write_packet failed: {e:?}");
                            }
                        };
                        video_enc.encode_frame(&raw, &mut sink)
                            .map_err(|e| format!("encode_frame failed: {e:?}"))?;
                        frames_encoded += 1;
                        prog_tx.report(frames_encoded, ExportPhase::Encoding);
                    }
                    Some(QueueItem::SegmentDone { .. }) => {}
                    Some(QueueItem::AllDone) | None => {
                        if !prog_tx.control().is_cancelled() {
                            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                                if let Err(e) = muxer.write_packet(pkt, true) {
                                    log::error!("[encoder] flush write_packet failed: {e:?}");
                                }
                            };
                            video_enc.flush(&mut sink)
                                .map_err(|e| format!("flush failed: {e:?}"))?;
                            completed = true;
                        }
                        break;
                    }
                }
            }
            Ok(())
        };

        let outcome = run();

        // Finalise before reporting anything terminal, so `Done` means "the file
        // on disk is complete and readable".
        let finalise = muxer.finalise_sync();
        if let Err(e) = &finalise {
            log::error!("[export] encoder_thread: muxer finalise failed: {e:?}");
        }

        match outcome {
            Err(e) => {
                log::error!("[export] encoder_thread error: {e}");
                prog_tx.report(frames_encoded, ExportPhase::Failed(e));
            }
            // A trailer that failed to write leaves an unplayable file, so it is a
            // failed export rather than a `Done` with a warning in the log.
            Ok(()) if completed => match finalise {
                Ok(()) => prog_tx.report(frames_encoded, ExportPhase::Done),
                Err(e) => prog_tx.report(
                    frames_encoded,
                    ExportPhase::Failed(format!("muxer finalise failed: {e:?}")),
                ),
            },
            // Cancelled: the phase was already reported inside the loop.
            Ok(()) => {}
        }
    }

    /// Decode audio from all clips in the export range, apply per-clip DSP, and
    /// encode into the muxer. Respects track mute/solo, per-clip volume/pan/fades,
    /// and track gain/pan. Applies a soft limiter to each AAC frame before encoding.
    fn audio_thread(
        job: Arc<ExportJob>,
        timeline: Arc<std::sync::RwLock<TimelineStore>>,
        tracks: Arc<std::sync::RwLock<TrackList>>,
        sources: Arc<std::sync::RwLock<SourceRegistry>>,
        mut audio_enc: AudioMuxEncoder,
        muxer: Arc<Muxer>,
        prog_tx: ProgressSender,
    ) {
        use crate::audio::ffi::avresample::{
            swr_alloc_set_opts, swr_convert, swr_free, swr_get_delay, swr_init,
            AV_CH_LAYOUT_STEREO, AV_SAMPLE_FMT_FLTP,
        };
        use crate::io::decoder::Decoder;
        use crate::io::demuxer::Demuxer;
        use crate::io::ffi::avcodec::{
            avcodec_ctx_get_channel_layout, avcodec_ctx_get_channels, avcodec_ctx_get_sample_fmt,
            avcodec_ctx_get_sample_rate, avcodec_receive_frame, avcodec_send_packet,
        };
        use crate::io::ffi::avutil::AVERROR_EOF;
        use crate::io::ffi::avutil::{
            av_frame_alloc, av_frame_free, av_frame_get_data, av_frame_get_nb_samples,
        };

        // Per-clip DSP params collected from the timeline + track list
        #[allow(dead_code)]
        struct ClipDsp {
            src_id:      crate::timeline::ids::SourceId,
            clip_t_in:   i64,
            clip_t_out:  i64,
            src_mat_in:  i64,
            speed:       f32,
            volume:      f32,  // clip vol × track gain
            pan:         f32,  // clip pan + track pan, clamped
            fade_in_pts: i64,
            fade_out_pts: i64,
            muted:       bool,
        }

        // Collect clips with audio that overlap the export range, with track DSP
        let clip_list: Vec<ClipDsp> = {
            let store = timeline.read().unwrap();
            let srcs  = sources.read().unwrap();
            let trks  = tracks.read().unwrap();
            let any_soloed = trks.any_soloed();
            let n = store.len();
            let mut clips = Vec::new();
            for i in 0..n {
                let t_in  = store.pts_in_at(i);
                let t_out = store.pts_out_at(i);
                // Skip clips outside export range
                if t_out <= job.pts_in || t_in >= job.pts_out {
                    continue;
                }
                let src = store.source_id_at(i);
                // Skip clips whose source has no audio
                if srcs.audio_info(src).is_err() {
                    continue;
                }
                // Respect track mute/solo
                let track_id = store.track_id_at(i);
                let (track_gain, track_pan, track_active) =
                    if let Some(t) = trks.get(track_id) {
                        (t.gain, t.pan, t.is_active(any_soloed))
                    } else {
                        (1.0, 0.0, true)
                    };
                if !track_active {
                    continue;
                }
                let clip_muted = store.audio_muted_at(i);
                let clip_vol   = store.volume_at(i);
                let clip_pan   = store.pan_at(i);
                let combined_vol = clip_vol * track_gain;
                let combined_pan = (clip_pan + track_pan).clamp(-1.0, 1.0);
                clips.push(ClipDsp {
                    src_id:       src,
                    clip_t_in:    t_in,
                    clip_t_out:   t_out,
                    src_mat_in:   store.source_in_at(i),
                    speed:        store.speed_at(i),
                    volume:       combined_vol,
                    pan:          combined_pan,
                    fade_in_pts:  store.fade_in_pts_at(i),
                    fade_out_pts: store.fade_out_pts_at(i),
                    muted:        clip_muted,
                });
            }
            // Sort by timeline in-point
            clips.sort_by_key(|c| c.clip_t_in);
            clips
        };

        // Packet sink for muxing audio packets
        let mut mux_sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
            if let Err(e) = muxer.write_packet(pkt, false) {
                log::error!("[audio] write_packet failed: {e:?}");
            }
        };

        let frame_size = audio_enc.frame_size();
        // Accumulation buffers (planar f32)
        let mut accum_left:  Vec<f32> = Vec::new();
        let mut accum_right: Vec<f32> = Vec::new();
        let mut enc_pts: i64 = 0; // in samples @ 48 kHz

        // Reusable swr output buffer
        let swr_out_capacity = frame_size * 2 + 64;
        let mut swr_left:  Vec<f32> = vec![0.0; swr_out_capacity];
        let mut swr_right: Vec<f32> = vec![0.0; swr_out_capacity];

        for dsp in clip_list {
            if prog_tx.control().is_cancelled() {
                log::info!("[export] audio thread detected cancellation");
                return;
            }
            if dsp.muted {
                continue; // muted clips contribute silence — skip decoding
            }

            let path = {
                let srcs = sources.read().unwrap();
                match srcs.path(dsp.src_id) {
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
                let sr  = avcodec_ctx_get_sample_rate(ctx);
                let mut cl = avcodec_ctx_get_channel_layout(ctx);
                let channels = avcodec_ctx_get_channels(ctx);
                let fmt = avcodec_ctx_get_sample_fmt(ctx);
                if cl == 0 {
                    cl = if channels == 1 { 4 } else { 3 };
                }
                (cl as i64, fmt as i32, sr as i32)
            };

            // Adjust input sample rate to stretch/squash the audio according to speed
            in_sample_rate = (in_sample_rate as f32 * dsp.speed).round() as i32;

            let swr = unsafe {
                let s = swr_alloc_set_opts(
                    std::ptr::null_mut(),
                    AV_CH_LAYOUT_STEREO as i64,
                    AV_SAMPLE_FMT_FLTP,
                    48_000,
                    in_ch_layout,
                    in_sample_fmt,
                    in_sample_rate,
                    0,
                    std::ptr::null_mut(),
                );
                if s.is_null() {
                    continue;
                }
                if swr_init(s) < 0 {
                    continue;
                }
                s
            };

            let frame = unsafe { av_frame_alloc() };
            if frame.is_null() {
                unsafe { swr_free(&mut { swr }); }
                continue;
            }

            // Effective export overlap for this clip
            let eff_t_in  = dsp.clip_t_in.max(job.pts_in);
            let eff_t_out = dsp.clip_t_out.min(job.pts_out);

            // Translate effective timeline range to source material range (90 kHz project TB)
            let src_seek_pts = dsp.src_mat_in
                + crate::timeline::rational::speed_scale_pts(eff_t_in - dsp.clip_t_in, dsp.speed);
            let src_end_pts  = dsp.src_mat_in
                + crate::timeline::rational::speed_scale_pts(eff_t_out - dsp.clip_t_in, dsp.speed);

            // Seek demuxer to just before the start of required audio
            let _ = demuxer.seek(src_seek_pts, job.project_tb);
            decoder.flush();

            // Convert PTS bounds to stream timebase
            let stream_seek_pts = job.project_tb.rescale_pts(src_seek_pts, audio_info.time_base);
            let stream_end_pts  = job.project_tb.rescale_pts(src_end_pts,  audio_info.time_base);

            // Discard audio packets until we reach the target PTS
            if src_seek_pts > 0 {
                while let Ok(Some(pkt)) = demuxer.next_audio_packet() {
                    if pkt.pts != i64::MIN && pkt.pts >= stream_seek_pts {
                        unsafe { let _ = avcodec_send_packet(decoder.ctx(), pkt.as_ptr()); }
                        break;
                    }
                }
            }

            // Pre-compute constant-power pan gains for this clip
            let (pan_l, pan_r) = crate::audio::audio_mixer::constant_power_pan(dsp.pan);
            let gain_l = dsp.volume * pan_l;
            let gain_r = dsp.volume * pan_r;

            // Running sample counter for fade envelope
            let mut samples_decoded: u64 = 0;

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
                    if send_ret < 0 {
                        break 'decode;
                    }

                    loop {
                        let recv_ret = avcodec_receive_frame(decoder.ctx(), frame);
                        if recv_ret == crate::io::ffi::avutil::AVERROR_EAGAIN
                            || recv_ret == AVERROR_EOF
                        {
                            break;
                        }
                        if recv_ret < 0 {
                            break;
                        }

                        let nb = av_frame_get_nb_samples(frame) as usize;
                        if nb == 0 {
                            continue;
                        }

                        // How many output samples swr will produce
                        let out_max = (nb as i64 * 48_000_i64 / in_sample_rate as i64 + 32) as usize;
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
                        if converted <= 0 {
                            continue;
                        }

                        // Apply per-sample DSP: volume, pan, fade envelope
                        let converted = converted as usize;
                        for s in 0..converted {
                            let sample_offset = samples_decoded + s as u64;
                            // Map decoded sample index back to timeline PTS
                            let tl_pts = eff_t_in
                                + (sample_offset as f64 * 90_000.0 / 48_000.0) as i64;

                            let fade = crate::audio::audio_mixer::compute_fade_multiplier(
                                tl_pts,
                                dsp.clip_t_in,
                                dsp.clip_t_out,
                                dsp.fade_in_pts,
                                dsp.fade_out_pts,
                            );

                            swr_left[s]  *= gain_l * fade;
                            swr_right[s] *= gain_r * fade;
                        }
                        samples_decoded += converted as u64;

                        accum_left.extend_from_slice(&swr_left[..converted]);
                        accum_right.extend_from_slice(&swr_right[..converted]);

                        // Flush full encoder frames out of accumulator
                        while accum_left.len() >= frame_size {
                            let mut left_chunk: Vec<f32>  = accum_left.drain(..frame_size).collect();
                            let mut right_chunk: Vec<f32> = accum_right.drain(..frame_size).collect();
                            // Soft-limit the frame before encoding
                            crate::audio::audio_mixer::soft_limit_buffer(&mut left_chunk);
                            crate::audio::audio_mixer::soft_limit_buffer(&mut right_chunk);
                            let _ = audio_enc.encode_pcm_chunk(
                                &left_chunk,
                                &right_chunk,
                                enc_pts,
                                &mut mux_sink,
                            );
                            enc_pts += frame_size as i64;
                        }
                    }
                }
            }

            // Flush swr delay for this clip
            unsafe {
                loop {
                    let delay = swr_get_delay(swr, 48_000);
                    if delay <= 0 {
                        break;
                    }
                    let out_max = (delay + 16) as usize;
                    if swr_left.len() < out_max {
                        swr_left.resize(out_max, 0.0);
                        swr_right.resize(out_max, 0.0);
                    }
                    let mut out_planes: [*mut u8; 2] = [
                        swr_left.as_mut_ptr()  as *mut u8,
                        swr_right.as_mut_ptr() as *mut u8,
                    ];
                    let converted = swr_convert(
                        swr,
                        out_planes.as_mut_ptr(),
                        out_max as i32,
                        std::ptr::null(),
                        0,
                    );
                    if converted <= 0 {
                        break;
                    }
                    // Apply pan gain to flushed samples (fade already fully done by now)
                    for s in 0..converted as usize {
                        swr_left[s]  *= gain_l;
                        swr_right[s] *= gain_r;
                    }
                    accum_left.extend_from_slice(&swr_left[..converted as usize]);
                    accum_right.extend_from_slice(&swr_right[..converted as usize]);
                    while accum_left.len() >= frame_size {
                        let mut l: Vec<f32> = accum_left.drain(..frame_size).collect();
                        let mut r: Vec<f32> = accum_right.drain(..frame_size).collect();
                        crate::audio::audio_mixer::soft_limit_buffer(&mut l);
                        crate::audio::audio_mixer::soft_limit_buffer(&mut r);
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
            crate::audio::audio_mixer::soft_limit_buffer(&mut accum_left);
            crate::audio::audio_mixer::soft_limit_buffer(&mut accum_right);
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
