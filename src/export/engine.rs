use crate::audio::ffi::avresample::{swr_free, SwrContext};
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
use crate::io::ffi::avutil::{av_frame_free, AVFrame};
use crate::render::compute::ComputePipelineCache;
use crate::render::device::GpuDevice;
use crate::render::shader::registry::ShaderRegistry;
use crate::scheduler::frame_scheduler::FrameScheduler;
use crate::timeline::source::SourceRegistry;
use crate::timeline::store::TimelineStore;
use crate::timeline::track::TrackList;
use std::sync::Arc;

struct AudioDecodeResources {
    frame: *mut AVFrame,
    swr: *mut SwrContext,
}

impl AudioDecodeResources {
    fn new(frame: *mut AVFrame, swr: *mut SwrContext) -> Self {
        Self { frame, swr }
    }
}

impl Drop for AudioDecodeResources {
    fn drop(&mut self) {
        unsafe {
            if !self.frame.is_null() {
                av_frame_free(&mut self.frame);
            }
            if !self.swr.is_null() {
                swr_free(&mut self.swr);
            }
        }
    }
}

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

        // Spawn encoder thread (video) if FfmpegEncoder is active. The dispatch
        // coordinator joins this worker before finalising the shared muxer.
        let video_handle = if !is_gpu {
            let video_enc = video_enc_opt.take().unwrap();
            let muxer_for_video = Arc::clone(&muxer);
            let prog_tx_video = prog_tx.clone();
            Some(
                std::thread::Builder::new()
                    .name("ve-encoder".into())
                    .spawn(move || {
                        Self::encoder_thread(queue_clone, video_enc, muxer_for_video, prog_tx_video)
                    })
                    .map_err(ExportError::ThreadSpawn)?,
            )
        } else {
            None
        };

        // Spawn audio encode thread — decodes audio from source clips and muxes it.
        let job_clone2 = Arc::clone(&self.job);
        let muxer_for_audio = Arc::clone(&muxer);
        let timeline_audio = Arc::clone(&self.timeline);
        let tracks_audio = Arc::clone(&self.tracks);
        let sources_audio = Arc::clone(&self.sources);
        let prog_tx_audio = prog_tx.clone();
        let audio_handle = std::thread::Builder::new()
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
                )
            })
            .map_err(ExportError::ThreadSpawn)?;

        // Spawn render/dispatch thread. This coordinator owns completion: every
        // producer is stopped and joined before the muxer is finalized once.
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

                let render_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<ExportRenderer, String> {
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
                                log::info!(
                                    "[export] dispatch detected cancellation before segment {i}"
                                );
                                break;
                            }
                            log::info!("[export] starting segment {i}/{}", segments.len());
                            renderer
                                .render_segment(seg, &queue, &prog_tx)
                                .map_err(|e| format!("render_segment {i} failed: {e:?}"))?;
                            log::info!("[export] segment {i} complete");
                            if prog_tx.control().is_cancelled() {
                                break;
                            }
                        }
                        Ok(renderer)
                    },
                ));

                // The CPU encoder must always be released, including render errors,
                // cancellation, and panics. A failed worker may already be gone.
                queue.finish();

                let mut first_error = None;
                let mut renderer = match render_result {
                    Ok(Ok(renderer)) => Some(renderer),
                    Ok(Err(error)) => {
                        first_error = Some(error);
                        None
                    }
                    Err(payload) => {
                        first_error = Some(format!(
                            "dispatch thread panicked: {}",
                            Self::panic_message(payload)
                        ));
                        None
                    }
                };

                if is_gpu {
                    if let Some(renderer) = renderer.as_mut() {
                        if let crate::export::renderer::ExportBackend::GpuNvenc {
                            ref mut video_enc,
                            ..
                        } = renderer.backend
                        {
                            let mut mux_error = None;
                            let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                                if mux_error.is_none() {
                                    if let Err(e) = muxer_for_dispatch.write_packet(pkt, true) {
                                        mux_error =
                                            Some(format!("NVENC flush mux write failed: {e:?}"));
                                    }
                                }
                            };
                            if let Err(e) = video_enc.flush(&mut sink) {
                                first_error
                                    .get_or_insert_with(|| format!("NVENC flush failed: {e:?}"));
                            }
                            if let Some(e) = mux_error {
                                first_error.get_or_insert(e);
                            }
                        }
                    }
                } else if let Some(handle) = video_handle {
                    let video_result = match handle.join() {
                        Ok(result) => result,
                        Err(payload) => Err(format!(
                            "video worker panicked: {}",
                            Self::panic_message(payload)
                        )),
                    };
                    if let Err(e) = video_result {
                        first_error.get_or_insert(e);
                    }
                } else {
                    first_error.get_or_insert_with(|| "missing CPU video worker".to_string());
                }

                let audio_result = match audio_handle.join() {
                    Ok(result) => result,
                    Err(payload) => Err(format!(
                        "audio worker panicked: {}",
                        Self::panic_message(payload)
                    )),
                };
                if let Err(e) = audio_result {
                    first_error.get_or_insert(e);
                }

                let (video_packets, audio_packets) = muxer_for_dispatch.packet_stats();
                log::info!(
                    "[export] encoded packets: video={video_packets}, audio={audio_packets}"
                );
                if let Err(e) = muxer_for_dispatch.finalise_sync() {
                    first_error.get_or_insert_with(|| format!("muxer finalise failed: {e:?}"));
                }

                if let Some(error) = first_error {
                    log::error!("[export] dispatch thread error: {error}");
                    prog_tx_dispatch.report(0, ExportPhase::Failed(error));
                } else if prog_tx_dispatch.control().is_cancelled() {
                    let frames_done = renderer.as_ref().map_or(0, |r| r.frames_done);
                    prog_tx_dispatch.report(frames_done, ExportPhase::Cancelled);
                } else {
                    prog_tx_dispatch.report(total_frames_count, ExportPhase::Done);
                }
            })
            .map_err(ExportError::ThreadSpawn)?;

        Ok(prog_rx)
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "(unknown panic payload)".to_string()
        }
    }

    fn encoder_thread(
        queue: Arc<EncoderQueue>,
        mut video_enc: VideoEncoderBackend,
        muxer: Arc<Muxer>,
        prog_tx: ProgressSender,
    ) -> Result<(), String> {
        let mut frames_encoded = 0usize;
        loop {
            match queue.pop() {
                Some(QueueItem::Frame(raw)) => {
                    let mut mux_error = None;
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        if mux_error.is_none() {
                            if let Err(e) = muxer.write_packet(pkt, true) {
                                mux_error = Some(format!("video mux write failed: {e:?}"));
                            }
                        }
                    };
                    video_enc
                        .encode_frame(&raw, &mut sink)
                        .map_err(|e| format!("encode_frame failed: {e:?}"))?;
                    if let Some(e) = mux_error {
                        return Err(e);
                    }
                    frames_encoded += 1;
                    prog_tx.report(frames_encoded, ExportPhase::Encoding);
                }
                Some(QueueItem::SegmentDone { .. }) => {}
                Some(QueueItem::AllDone) | None => {
                    let mut mux_error = None;
                    let mut sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
                        if mux_error.is_none() {
                            if let Err(e) = muxer.write_packet(pkt, true) {
                                mux_error = Some(format!("video flush mux write failed: {e:?}"));
                            }
                        }
                    };
                    video_enc
                        .flush(&mut sink)
                        .map_err(|e| format!("video flush failed: {e:?}"))?;
                    if let Some(e) = mux_error {
                        return Err(e);
                    }
                    return Ok(());
                }
            }
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
    ) -> Result<(), String> {
        use crate::audio::ffi::avresample::{
            swr_alloc_set_opts, swr_convert, swr_get_delay, swr_init, AV_CH_LAYOUT_STEREO,
            AV_SAMPLE_FMT_FLTP,
        };
        use crate::io::decoder::Decoder;
        use crate::io::demuxer::{Demuxer, Packet};
        use crate::io::ffi::avcodec::{
            avcodec_ctx_get_channel_layout, avcodec_ctx_get_channels, avcodec_ctx_get_sample_fmt,
            avcodec_ctx_get_sample_rate, avcodec_receive_frame, avcodec_send_packet,
        };
        use crate::io::ffi::avutil::{
            av_frame_alloc, av_frame_get_data, av_frame_get_nb_samples, AVERROR_EAGAIN, AVERROR_EOF,
        };

        // Per-clip DSP params collected from the timeline + track list
        #[allow(dead_code)]
        struct ClipDsp {
            src_id: crate::timeline::ids::SourceId,
            clip_t_in: i64,
            clip_t_out: i64,
            src_mat_in: i64,
            speed: f32,
            volume: f32, // clip vol × track gain
            pan: f32,    // clip pan + track pan, clamped
            fade_in_pts: i64,
            fade_out_pts: i64,
            muted: bool,
        }

        // Collect clips with audio that overlap the export range, with track DSP
        let clip_list: Vec<ClipDsp> = {
            let store = timeline.read().unwrap();
            let srcs = sources.read().unwrap();
            let trks = tracks.read().unwrap();
            let any_soloed = trks.any_soloed();
            let n = store.len();
            let mut clips = Vec::new();
            for i in 0..n {
                let t_in = store.pts_in_at(i);
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
                let (track_gain, track_pan, track_active) = if let Some(t) = trks.get(track_id) {
                    (t.gain, t.pan, t.is_active(any_soloed))
                } else {
                    (1.0, 0.0, true)
                };
                if !track_active {
                    continue;
                }
                let clip_muted = store.audio_muted_at(i);
                let clip_vol = store.volume_at(i);
                let clip_pan = store.pan_at(i);
                let combined_vol = clip_vol * track_gain;
                let combined_pan = (clip_pan + track_pan).clamp(-1.0, 1.0);
                clips.push(ClipDsp {
                    src_id: src,
                    clip_t_in: t_in,
                    clip_t_out: t_out,
                    src_mat_in: store.source_in_at(i),
                    speed: store.speed_at(i),
                    volume: combined_vol,
                    pan: combined_pan,
                    fade_in_pts: store.fade_in_pts_at(i),
                    fade_out_pts: store.fade_out_pts_at(i),
                    muted: clip_muted,
                });
            }
            // Sort by timeline in-point
            clips.sort_by_key(|c| c.clip_t_in);
            clips
        };

        // Packet sink records the first mux failure so the worker can propagate it.
        let mut mux_error = None;
        let mut mux_sink = |pkt: *mut crate::io::ffi::avutil::AVPacket| {
            if mux_error.is_none() {
                if let Err(e) = muxer.write_packet(pkt, false) {
                    mux_error = Some(format!("audio mux write failed: {e:?}"));
                }
            }
        };

        let frame_size = audio_enc.frame_size();
        // One timeline-aligned program buffer. Clips sum into their timeline
        // positions; gaps remain zero and overlap does not extend the export.
        let sample_numer = (job.pts_out - job.pts_in) as i128 * 48_000;
        let sample_denom = job.project_tb.den as i128;
        let total_samples = ((sample_numer + sample_denom - 1) / sample_denom) as usize;
        let mut accum_left = vec![0.0f32; total_samples];
        let mut accum_right = vec![0.0f32; total_samples];

        // Reusable swr output buffer
        let swr_out_capacity = frame_size * 2 + 64;
        let mut swr_left: Vec<f32> = vec![0.0; swr_out_capacity];
        let mut swr_right: Vec<f32> = vec![0.0; swr_out_capacity];

        for dsp in clip_list {
            if prog_tx.control().is_cancelled() {
                log::info!("[export] audio thread detected cancellation");
                return Ok(());
            }
            if dsp.muted {
                continue; // muted clips contribute silence — skip decoding
            }

            let path = {
                let srcs = sources
                    .read()
                    .map_err(|_| "source registry lock poisoned".to_string())?;
                (*srcs
                    .path(dsp.src_id)
                    .ok_or_else(|| format!("audio source {:?} has no path", dsp.src_id))?)
                .clone()
            };

            let mut demuxer = Demuxer::open(&path)
                .map_err(|e| format!("failed to open audio source {}: {e:?}", path.display()))?;

            let audio_info = demuxer
                .audio_stream()
                .cloned()
                .ok_or_else(|| format!("audio source {} has no audio stream", path.display()))?;

            let mut decoder =
                Decoder::open(&audio_info, audio_info.codecpar, false).map_err(|e| {
                    format!("failed to open audio decoder for {}: {e:?}", path.display())
                })?;

            // Set up SwrContext for this source
            let (in_ch_layout, in_sample_fmt, mut in_sample_rate) = unsafe {
                let ctx = decoder.ctx();
                let sr = avcodec_ctx_get_sample_rate(ctx);
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

            let s = unsafe {
                swr_alloc_set_opts(
                    std::ptr::null_mut(),
                    AV_CH_LAYOUT_STEREO as i64,
                    AV_SAMPLE_FMT_FLTP,
                    48_000,
                    in_ch_layout,
                    in_sample_fmt,
                    in_sample_rate,
                    0,
                    std::ptr::null_mut(),
                )
            };
            if s.is_null() {
                return Err(format!(
                    "failed to allocate audio resampler for {}",
                    path.display()
                ));
            }
            if unsafe { swr_init(s) } < 0 {
                let _resources = AudioDecodeResources::new(std::ptr::null_mut(), s);
                return Err(format!(
                    "failed to initialize audio resampler for {}",
                    path.display()
                ));
            }
            let frame = unsafe { av_frame_alloc() };
            if frame.is_null() {
                let _resources = AudioDecodeResources::new(std::ptr::null_mut(), s);
                return Err("failed to allocate audio decode frame".to_string());
            }
            let resources = AudioDecodeResources::new(frame, s);
            let swr = resources.swr;

            // Effective export overlap for this clip
            let eff_t_in = dsp.clip_t_in.max(job.pts_in);
            let eff_t_out = dsp.clip_t_out.min(job.pts_out);

            if eff_t_out <= eff_t_in {
                continue;
            }

            // Translate effective timeline range to source material range (90 kHz project TB)
            let src_seek_pts = dsp.src_mat_in
                + crate::timeline::rational::speed_scale_pts(eff_t_in - dsp.clip_t_in, dsp.speed);
            let src_end_pts = dsp.src_mat_in
                + crate::timeline::rational::speed_scale_pts(eff_t_out - dsp.clip_t_in, dsp.speed);

            // Seek demuxer to just before the start of required audio. Keep the
            // first packet at or after the requested PTS so it is decoded exactly
            // once by the regular send/receive loop.
            demuxer
                .seek(src_seek_pts, job.project_tb)
                .map_err(|e| format!("audio seek failed for {}: {e:?}", path.display()))?;
            decoder.flush();

            let stream_seek_pts = job
                .project_tb
                .rescale_pts(src_seek_pts, audio_info.time_base);
            let stream_end_pts = job
                .project_tb
                .rescale_pts(src_end_pts, audio_info.time_base);
            let mut pending_packet: Option<Packet> = None;
            if src_seek_pts > 0 {
                loop {
                    match demuxer.next_audio_packet() {
                        Ok(Some(pkt)) => {
                            if pkt.pts == i64::MIN || pkt.pts >= stream_seek_pts {
                                pending_packet = Some(pkt);
                                break;
                            }
                        }
                        Ok(None) => break,
                        Err(e) => {
                            return Err(format!(
                                "audio demux failed for {}: {e:?}",
                                path.display()
                            ));
                        }
                    }
                }
            }

            // Running sample counter and the clip's sample offset in the program.
            let mut samples_decoded: u64 = 0;
            let clip_offset =
                (((eff_t_in - job.pts_in) as i128 * 48_000) / job.project_tb.den as i128) as usize;
            let clip_numer = (eff_t_out - eff_t_in) as i128 * 48_000;
            let clip_denom = job.project_tb.den as i128;
            let clip_sample_count = ((clip_numer + clip_denom - 1) / clip_denom) as usize;

            // Hard cap: stop decoding once the required timeline overlap is full.
            // Sources with missing packet PTS must not drain through source EOF.
            let max_audio_samples = clip_sample_count as u64;

            enum ReceiveState {
                NeedInput,
                Eof,
                Full,
            }

            let mut receive_frames = |samples_decoded: &mut u64| -> Result<ReceiveState, String> {
                unsafe {
                    loop {
                        let recv_ret = avcodec_receive_frame(decoder.ctx(), frame);
                        if recv_ret == AVERROR_EAGAIN {
                            return Ok(ReceiveState::NeedInput);
                        }
                        if recv_ret == AVERROR_EOF {
                            return Ok(ReceiveState::Eof);
                        }
                        if recv_ret < 0 {
                            return Err(format!(
                                "audio decode failed for {}: FFmpeg error {recv_ret}",
                                path.display()
                            ));
                        }

                        let nb = av_frame_get_nb_samples(frame) as usize;
                        if nb == 0 {
                            continue;
                        }
                        let out_max =
                            (nb as i64 * 48_000_i64 / in_sample_rate as i64 + 32) as usize;
                        if swr_left.len() < out_max {
                            swr_left.resize(out_max, 0.0);
                            swr_right.resize(out_max, 0.0);
                        }
                        let in_data = av_frame_get_data(frame) as *const *const u8;
                        let mut out_planes: [*mut u8; 2] = [
                            swr_left.as_mut_ptr() as *mut u8,
                            swr_right.as_mut_ptr() as *mut u8,
                        ];
                        let converted = swr_convert(
                            swr,
                            out_planes.as_mut_ptr(),
                            out_max as i32,
                            in_data,
                            nb as i32,
                        );
                        if converted < 0 {
                            return Err(format!(
                                "audio resample failed for {}: FFmpeg error {converted}",
                                path.display()
                            ));
                        }
                        if converted > 0 {
                            let converted = converted as usize;
                            let remaining =
                                clip_sample_count.saturating_sub(*samples_decoded as usize);
                            let mix_count = converted.min(remaining);
                            crate::audio::audio_mixer::AudioMixer::mix_clip_chunk(
                                &mut accum_left,
                                &mut accum_right,
                                &swr_left[..mix_count],
                                &swr_right[..mix_count],
                                clip_offset + *samples_decoded as usize,
                                dsp.clip_t_in,
                                dsp.clip_t_out,
                                dsp.fade_in_pts,
                                dsp.fade_out_pts,
                                dsp.volume,
                                dsp.pan,
                                false,
                                1.0,
                                0.0,
                                true,
                                job.pts_in,
                            );
                            *samples_decoded += mix_count as u64;
                            if *samples_decoded >= max_audio_samples {
                                return Ok(ReceiveState::Full);
                            }
                        }
                    }
                }
            };

            let mut reached_source_end = false;
            while samples_decoded < max_audio_samples {
                if prog_tx.control().is_cancelled() {
                    break;
                }
                let packet = if let Some(pkt) = pending_packet.take() {
                    Some(pkt)
                } else {
                    match demuxer.next_audio_packet() {
                        Ok(packet) => packet,
                        Err(e) => {
                            return Err(format!(
                                "audio demux failed for {}: {e:?}",
                                path.display()
                            ));
                        }
                    }
                };
                let Some(pkt) = packet else {
                    reached_source_end = true;
                    break;
                };
                if pkt.pts != i64::MIN && pkt.pts > stream_end_pts {
                    reached_source_end = true;
                    break;
                }

                loop {
                    let send_ret = unsafe { avcodec_send_packet(decoder.ctx(), pkt.as_ptr()) };
                    if send_ret == AVERROR_EAGAIN {
                        if matches!(receive_frames(&mut samples_decoded)?, ReceiveState::Full) {
                            break;
                        }
                        continue;
                    }
                    if send_ret < 0 {
                        return Err(format!(
                            "audio packet submit failed for {}: FFmpeg error {send_ret}",
                            path.display()
                        ));
                    }
                    break;
                }
                if matches!(receive_frames(&mut samples_decoded)?, ReceiveState::Full) {
                    break;
                }
            }

            // At real source/range EOF, drain delayed codec frames before flushing
            // the resampler. Cancellation skips extra work but still frees state.
            if reached_source_end
                && samples_decoded < max_audio_samples
                && !prog_tx.control().is_cancelled()
            {
                let send_ret = unsafe { avcodec_send_packet(decoder.ctx(), std::ptr::null()) };
                if send_ret < 0 && send_ret != AVERROR_EOF {
                    return Err(format!(
                        "audio decoder drain failed for {}: FFmpeg error {send_ret}",
                        path.display()
                    ));
                }
                loop {
                    match receive_frames(&mut samples_decoded)? {
                        ReceiveState::NeedInput | ReceiveState::Eof | ReceiveState::Full => break,
                    }
                }
            }

            drop(receive_frames);

            // Flush swr delay for this clip into the remaining timeline range.
            unsafe {
                while samples_decoded < max_audio_samples {
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
                        swr_left.as_mut_ptr() as *mut u8,
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
                    let remaining = clip_sample_count.saturating_sub(samples_decoded as usize);
                    let mix_count = (converted as usize).min(remaining);
                    crate::audio::audio_mixer::AudioMixer::mix_clip_chunk(
                        &mut accum_left,
                        &mut accum_right,
                        &swr_left[..mix_count],
                        &swr_right[..mix_count],
                        clip_offset + samples_decoded as usize,
                        dsp.clip_t_in,
                        dsp.clip_t_out,
                        dsp.fade_in_pts,
                        dsp.fade_out_pts,
                        dsp.volume,
                        dsp.pan,
                        false,
                        1.0,
                        0.0,
                        true,
                        job.pts_in,
                    );
                    samples_decoded += mix_count as u64;
                }

                drop(resources);
            }
        }

        crate::audio::audio_mixer::soft_limit_buffer(&mut accum_left);
        crate::audio::audio_mixer::soft_limit_buffer(&mut accum_right);

        // Encode the completed timeline program once, in consecutive AAC frames.
        // Only the final partial frame is padded; PTS stays in 48 kHz samples.
        let mut frame_left = vec![0.0f32; frame_size];
        let mut frame_right = vec![0.0f32; frame_size];
        let mut frame_start = 0usize;
        while frame_start < total_samples {
            if prog_tx.control().is_cancelled() {
                return Ok(());
            }
            let frame_end = (frame_start + frame_size).min(total_samples);
            let count = frame_end - frame_start;
            frame_left.fill(0.0);
            frame_right.fill(0.0);
            frame_left[..count].copy_from_slice(&accum_left[frame_start..frame_end]);
            frame_right[..count].copy_from_slice(&accum_right[frame_start..frame_end]);
            audio_enc
                .encode_pcm_chunk(&frame_left, &frame_right, frame_start as i64, &mut mux_sink)
                .map_err(|e| format!("audio encode failed: {e:?}"))?;
            frame_start += frame_size;
        }

        audio_enc
            .encode_all(&job, &mut mux_sink)
            .map_err(|e| format!("audio encoder flush failed: {e:?}"))?;
        drop(mux_sink);
        if let Some(e) = mux_error {
            return Err(e);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum ExportError {
    JobInvalid(crate::export::job::JobError),
    EncoderOpen(crate::export::video_encoder::EncodeError),
    MuxerOpen(crate::export::muxer::MuxError),
    ThreadSpawn(std::io::Error),
}
