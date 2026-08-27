// src/audio/output_stream.rs

use crate::audio::ring_buffer::AudioRingBuffer;
use crate::sync::master_clock::MasterClock;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

/// Shared list of per-clip ring buffers that the CPAL callback mixes from.
pub type MixerBusList = Arc<Mutex<Vec<Arc<AudioRingBuffer>>>>;

pub struct AudioOutputStream {
    /// cpal stream handle. Kept alive as long as audio should play.
    /// Dropping this stops audio output.
    _stream: cpal::Stream,
    pub config: cpal::StreamConfig,
}

impl AudioOutputStream {
    /// Open the default audio output device and start streaming.
    pub fn open(
        mixer_bufs: MixerBusList,
        clock: Arc<MasterClock>,
    ) -> Result<Self, AudioStreamError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioStreamError::NoOutputDevice)?;

        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(48_000),
            buffer_size: cpal::BufferSize::Fixed(1024),
        };

        let mixer_cb = Arc::clone(&mixer_bufs);
        let clock_cb = Arc::clone(&clock);

        let stream = device
            .build_output_stream(
                &config,
                move |output: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    audio_callback(output, &mixer_cb, &clock_cb);
                },
                move |err| {
                    eprintln!("cpal stream error: {err}");
                },
                None,
            )
            .map_err(AudioStreamError::BuildStream)?;

        stream.play().map_err(AudioStreamError::Play)?;

        Ok(Self {
            _stream: stream,
            config,
        })
    }

    pub fn play(&self) -> Result<(), cpal::PlayStreamError> {
        self._stream.play()
    }

    pub fn pause(&self) -> Result<(), cpal::PauseStreamError> {
        self._stream.pause()
    }
}

/// Real-time audio callback: mixes all active per-clip ring buffers into `output`.
/// Runs on the CPAL audio thread — must never block or allocate.
fn audio_callback(output: &mut [f32], mixer_bufs: &MixerBusList, clock: &MasterClock) {
    let bufs = match mixer_bufs.try_lock() {
        Ok(guard) => guard,
        Err(_) => {
            output.fill(0.0);
            return;
        }
    };

    output.fill(0.0);
    clock.advance_samples(output.len() / 2);

    if bufs.is_empty() {
        return;
    }

    // Allocate a temporary scratch buffer on the stack (max 4096 samples = 1024 stereo frames).
    // Heap allocation is avoided — the CPAL buffer size is fixed at 1024 frames = 2048 samples.
    let mut scratch = [0.0f32; 2048];
    let n = output.len().min(scratch.len());

    for buf in bufs.iter() {
        let scratch_slice = &mut scratch[..n];
        scratch_slice.fill(0.0);
        buf.read(scratch_slice);
        for (out, &s) in output.iter_mut().zip(scratch_slice.iter()) {
            *out += s;
        }
    }

    // Soft limiter prevents digital clipping.
    crate::audio::audio_mixer::soft_limit_buffer(output);
}

#[derive(Debug)]
pub enum AudioStreamError {
    NoOutputDevice,
    UnsupportedConfig(cpal::SupportedStreamConfigsError),
    BuildStream(cpal::BuildStreamError),
    Play(cpal::PlayStreamError),
}
