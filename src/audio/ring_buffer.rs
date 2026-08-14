// src/audio/ring_buffer.rs

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// SPSC wait-free ring buffer for F32 interleaved audio samples.
/// Capacity is fixed at construction and must be a power of two
/// so that modulo reduces to a cheap bitwise AND.
pub struct AudioRingBuffer {
    /// Heap-allocated sample storage. Length = capacity (power of two).
    data: Box<[std::cell::UnsafeCell<f32>]>,
    /// Capacity — always a power of two.
    capacity: usize,
    /// Bitmask = capacity - 1 (used instead of %).
    mask: usize,
    /// Write position (samples written total, wraps implicitly via mask).
    /// Written only by the producer thread.
    write_pos: AtomicUsize,
    /// Read position (samples consumed total).
    /// Written only by the consumer thread.
    read_pos: AtomicUsize,
    /// Underrun counter
    underruns: AtomicUsize,
}

// SAFETY: AudioRingBuffer is explicitly designed for concurrent SPSC access.
unsafe impl Send for AudioRingBuffer {}
unsafe impl Sync for AudioRingBuffer {}

impl AudioRingBuffer {
    /// Allocate a ring buffer of `capacity_samples` F32 samples.
    pub fn new(capacity_samples: usize) -> Arc<Self> {
        assert!(
            capacity_samples.is_power_of_two(),
            "capacity must be power of two"
        );
        let mut vec = Vec::with_capacity(capacity_samples);
        for _ in 0..capacity_samples {
            vec.push(std::cell::UnsafeCell::new(0.0f32));
        }
        let data = vec.into_boxed_slice();
        Arc::new(Self {
            data,
            capacity: capacity_samples,
            mask: capacity_samples - 1,
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            underruns: AtomicUsize::new(0),
        })
    }

    /// Number of samples available to read (safe to call from any thread).
    pub fn available_read(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Number of samples the producer can write without overflowing.
    pub fn available_write(&self) -> usize {
        self.capacity - 1 - self.available_read()
    }

    /// Write samples into the ring buffer. Called from the PRODUCER thread only.
    pub fn write(&self, samples: &[f32]) -> usize {
        let n = samples.len().min(self.available_write());
        if n == 0 {
            return 0;
        }

        let w = self.write_pos.load(Ordering::Relaxed) & self.mask;
        let first_len = n.min(self.capacity - w);
        let second_len = n - first_len;

        // SAFETY: The struct encapsulates an unsafe pointer to `data` to avoid
        // taking &mut self, which isn't possible because `self` is shared between threads.
        // Wait, `data: Box<[f32]>` cannot be mutated through `&self` safely without unsafe!
        // The Lamport ring buffer design dictates that producer and consumer NEVER
        // access the same indices concurrently.
        unsafe {
            let data_ptr = self.data.as_ptr() as *mut std::cell::UnsafeCell<f32> as *mut f32;
            std::ptr::copy_nonoverlapping(samples.as_ptr(), data_ptr.add(w), first_len);
            if second_len > 0 {
                std::ptr::copy_nonoverlapping(
                    samples.as_ptr().add(first_len),
                    data_ptr,
                    second_len,
                );
            }
        }

        self.write_pos.fetch_add(n, Ordering::Release);
        n
    }

    /// Read samples from the ring buffer. Called from the CONSUMER thread only.
    pub fn read(&self, out: &mut [f32]) -> usize {
        let avail = self.available_read();
        let n = out.len().min(avail);
        if n == 0 {
            if !out.is_empty() {
                out.fill(0.0);
                self.underruns.fetch_add(1, Ordering::Relaxed);
            }
            return 0;
        }

        let r = self.read_pos.load(Ordering::Relaxed) & self.mask;
        let first_len = n.min(self.capacity - r);
        let second_len = n - first_len;

        unsafe {
            let data_ptr = self.data.as_ptr() as *const std::cell::UnsafeCell<f32> as *const f32;
            std::ptr::copy_nonoverlapping(data_ptr.add(r), out.as_mut_ptr(), first_len);
            if second_len > 0 {
                std::ptr::copy_nonoverlapping(
                    data_ptr,
                    out.as_mut_ptr().add(first_len),
                    second_len,
                );
            }
        }

        self.read_pos.fetch_add(n, Ordering::Release);

        if n < out.len() {
            out[n..].fill(0.0);
            self.underruns.fetch_add(1, Ordering::Relaxed);
        }

        n
    }

    /// Clear the ring buffer (called on seek to discard stale audio).
    pub fn clear(&self) {
        let w = self.write_pos.load(Ordering::Acquire);
        self.read_pos.store(w, Ordering::Release);
    }

    /// Underrun counter
    pub fn underrun_count(&self) -> usize {
        self.underruns.load(Ordering::Relaxed)
    }
}
