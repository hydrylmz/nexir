// src/audio/output_stream.rs

use std::sync::Arc;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crate::audio::ring_buffer::AudioRingBuffer;
use crate::sync::master_clock::MasterClock;

pub struct AudioOutputStream {
    /// cpal stream handle. Kept alive as long as audio should play.
    /// Dropping this stops audio output.
    _stream: cpal::Stream,
    pub config: cpal::StreamConfig,
}

impl AudioOutputStream {
    /// Open the default audio output device and start streaming.
    pub fn open(
        ring:  Arc<AudioRingBuffer>,
        clock: Arc<MasterClock>,
    ) -> Result<Self, AudioStreamError> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or(AudioStreamError::NoOutputDevice)?;

        let config = cpal::StreamConfig {
            channels:    2,
            sample_rate: cpal::SampleRate(48_000),
            buffer_size: cpal::BufferSize::Fixed(1024),
        };

        let ring_cb  = Arc::clone(&ring);
        let clock_cb = Arc::clone(&clock);

        let stream = device.build_output_stream(
            &config,
            move |output: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                audio_callback(output, &ring_cb, &clock_cb);
            },
            move |err| { eprintln!("cpal stream error: {err}"); },
            None,
        ).map_err(AudioStreamError::BuildStream)?;

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

/// The real-time audio callback.
fn audio_callback(
    output: &mut [f32],
    ring:   &AudioRingBuffer,
    clock:  &MasterClock,
) {
    let read_elements = ring.read(output);
    clock.advance_samples(read_elements / 2);
}

#[derive(Debug)]
pub enum AudioStreamError {
    NoOutputDevice,
    UnsupportedConfig(cpal::SupportedStreamConfigsError),
    BuildStream(cpal::BuildStreamError),
    Play(cpal::PlayStreamError),
}
