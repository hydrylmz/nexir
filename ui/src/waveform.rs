// ui/src/waveform.rs
//
// Background audio peak extractor. Uses AudioDecoder infrastructure via demuxer FFI
// to decode an audio file and compute peak values for waveform display in the timeline.

use nexir::timeline::ids::SourceId;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct Waveform {
    /// Peaks at 100 Hz sample rate (one value per 10ms). Each is in range [0.0, 1.0].
    pub peaks: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct WaveformCache {
    cache: Arc<Mutex<HashMap<SourceId, Option<Arc<Waveform>>>>>,
}

impl WaveformCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the waveform if it's been computed. Returns `None` if still computing.
    pub fn get(&self, source_id: SourceId) -> Option<Arc<Waveform>> {
        self.cache
            .lock()
            .unwrap()
            .get(&source_id)
            .and_then(|v| v.clone())
    }

    /// Request waveform extraction for this source. Noop if already requested.
    pub fn request(&self, source_id: SourceId, path: PathBuf) {
        {
            let lock = self.cache.lock().unwrap();
            if lock.contains_key(&source_id) {
                return; // already requested or done
            }
        }
        // Mark as in-progress
        self.cache.lock().unwrap().insert(source_id, None);

        let cache_clone = Arc::clone(&self.cache);
        std::thread::spawn(move || {
            let result = generate_waveform(&path);
            let waveform = result.unwrap_or_else(|_| Waveform { peaks: vec![] });
            cache_clone
                .lock()
                .unwrap()
                .insert(source_id, Some(Arc::new(waveform)));
        });
    }
}

fn generate_waveform(path: &Path) -> Result<Waveform, String> {
    use nexir::audio::ffi::avresample::{
        AV_SAMPLE_FMT_FLTP, swr_alloc_set_opts, swr_convert, swr_free, swr_init,
    };
    use nexir::io::decoder::Decoder;
    use nexir::io::demuxer::Demuxer;
    use nexir::io::ffi::avcodec::{
        avcodec_ctx_get_channel_layout, avcodec_ctx_get_channels, avcodec_ctx_get_sample_fmt,
        avcodec_ctx_get_sample_rate, avcodec_receive_frame, avcodec_send_packet,
    };
    use nexir::io::ffi::avutil::{av_frame_alloc, av_frame_free, av_frame_get_nb_samples};

    let mut demuxer = Demuxer::open(path).map_err(|e| format!("{:?}", e))?;
    let audio_stream = demuxer
        .audio_stream()
        .cloned()
        .ok_or_else(|| "no audio stream".to_string())?;
    let _stream_tb = audio_stream.time_base;
    let decoder = Decoder::open(&audio_stream, audio_stream.codecpar, false)
        .map_err(|e| format!("{:?}", e))?;

    let (in_ch_layout, in_sample_fmt, in_sample_rate) = unsafe {
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

    // Resample to 4000 Hz (mono) for waveform extraction (cheap)
    const WAVEFORM_SR: i32 = 4000;
    let swr = unsafe {
        swr_alloc_set_opts(
            std::ptr::null_mut(),
            4, // AV_CH_LAYOUT_MONO
            AV_SAMPLE_FMT_FLTP,
            WAVEFORM_SR,
            in_ch_layout,
            in_sample_fmt,
            in_sample_rate,
            0,
            std::ptr::null_mut(),
        )
    };
    if swr.is_null() {
        return Err("swr_alloc failed".into());
    }
    unsafe {
        let r = swr_init(swr);
        if r < 0 {
            let mut p = swr;
            swr_free(&mut p);
            return Err(format!("swr_init failed: {}", r));
        }
    }

    let frame = unsafe { av_frame_alloc() };
    if frame.is_null() {
        unsafe {
            let mut p = swr;
            swr_free(&mut p);
        }
        return Err("av_frame_alloc failed".into());
    }

    let mut all_samples: Vec<f32> = Vec::with_capacity(1 << 18);

    unsafe {
        'outer: loop {
            match demuxer.next_audio_packet() {
                Ok(Some(pkt)) => {
                    if avcodec_send_packet(decoder.ctx(), pkt.as_ptr()) < 0 {
                        break;
                    }
                    loop {
                        let r = avcodec_receive_frame(decoder.ctx(), frame);
                        if r == nexir::io::ffi::avutil::AVERROR_EAGAIN
                            || r == nexir::io::ffi::avutil::AVERROR_EOF
                        {
                            break;
                        }
                        if r < 0 {
                            break 'outer;
                        }

                        let nb = av_frame_get_nb_samples(frame) as usize;
                        if nb == 0 {
                            continue;
                        }

                        let out_max =
                            (nb as i64 * WAVEFORM_SR as i64 / in_sample_rate as i64 + 4) as usize;
                        let mut mono_buf = vec![0.0f32; out_max];
                        let mut out_ptr = [mono_buf.as_mut_ptr() as *mut u8];
                        let in_data = (*frame).data.as_ptr() as *const *const u8;

                        let converted = swr_convert(
                            swr,
                            out_ptr.as_mut_ptr(),
                            out_max as i32,
                            in_data,
                            nb as i32,
                        );
                        if converted > 0 {
                            all_samples.extend(
                                mono_buf[..converted as usize].iter().map(|s| s.abs()),
                            );
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => break,
            }
        }
        av_frame_free(&mut { frame });
        let mut p = swr;
        swr_free(&mut p);
    }

    // Downsample to 100 Hz display rate (40 samples per peak at 4000 Hz)
    let chunk = (WAVEFORM_SR / 100) as usize;
    let mut peaks = Vec::with_capacity(all_samples.len() / chunk + 1);
    for c in all_samples.chunks(chunk) {
        let max = c.iter().copied().fold(0.0f32, f32::max);
        peaks.push(max);
    }

    Ok(Waveform { peaks })
}
