// src/render/gpu_timer.rs
//
// GPU-side timing via wgpu timestamp queries.
//
// WHY THIS EXISTS.  Every "GPU" number this project printed before it was CPU
// wall time around a submission plus a `poll(Wait)` — i.e. how long THIS THREAD
// spent blocked, which is an upper bound on GPU execution and can be dominated
// by scheduling that has nothing to do with the shaders.  The 4K benchmark row
// read `GPU wait 12.67 ms (41.8%)` and there was no way to tell how much of that
// the GPU was actually busy for, so "optimise the GPU" and "stop blocking the
// CPU" were indistinguishable proposals.
//
// A reading is `Option<f64>` for exactly the reason every `SystemMetrics` field
// is: `Some(0.0)` and `None` are different claims, and once both print as a
// number the reader cannot tell them apart.  A device without
// `Features::TIMESTAMP_QUERY` yields `None` forever and says so.
//
// WHAT IT DOES NOT DO.  Timestamps are written BETWEEN passes, never inside one:
// writing inside a pass needs `TIMESTAMP_QUERY_INSIDE_PASSES`, which this crate
// does not request.  Since every render-graph node opens and closes its own
// compute pass, between-node brackets already give per-node resolution.

use crate::render::device::GpuDevice;

/// Brackets GPU work with timestamp queries and reports the elapsed GPU time.
///
/// One timer owns one query set, so a timer must not be reused for a second
/// frame until the first frame's readings have been read — see
/// [`Self::reset`].  For a pipeline with several frames in flight, keep one
/// timer per slot rather than sharing one.
pub struct GpuTimer {
    inner: Option<Inner>,
}

struct Inner {
    query_set: wgpu::QuerySet,
    /// Resolve target.  `QUERY_RESOLVE | COPY_SRC`; wgpu forbids combining
    /// `QUERY_RESOLVE` with `MAP_READ`, hence the separate `readback` below.
    resolve: wgpu::Buffer,
    /// Mappable mirror of `resolve`.
    readback: wgpu::Buffer,
    /// Nanoseconds per raw tick, from `Queue::get_timestamp_period()`.
    period_ns: f32,
    /// Timestamps written into the set so far this frame.
    written: u32,
    capacity: u32,
    device: std::sync::Arc<wgpu::Device>,
}

impl GpuTimer {
    /// Build a timer with room for `pairs` begin/end brackets (2 queries each).
    ///
    /// Returns a disabled timer — one that measures nothing and reports `None` —
    /// when the device lacks `TIMESTAMP_QUERY` or `pairs` is zero.  Callers do
    /// not branch on support; they call `begin`/`end` unconditionally and get
    /// `None` back.
    pub fn new(device: &GpuDevice, pairs: u32) -> Self {
        if !device.has_timestamp_queries || pairs == 0 {
            return Self::disabled();
        }
        let count = pairs * 2;
        assert!(
            count <= wgpu::QUERY_SET_MAX_QUERIES,
            "GpuTimer: {count} queries exceeds QUERY_SET_MAX_QUERIES ({})",
            wgpu::QUERY_SET_MAX_QUERIES
        );

        let bytes = (count * wgpu::QUERY_SIZE) as u64;
        // Round up to the resolve alignment so the buffer is legal as a resolve
        // destination at offset 0 and stays legal if a caller ever resolves in
        // several chunks.
        let size = bytes.max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT);

        let query_set = device.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_timer_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            inner: Some(Inner {
                query_set,
                resolve,
                readback,
                period_ns: device.queue.get_timestamp_period(),
                written: 0,
                capacity: count,
                device: std::sync::Arc::clone(&device.device),
            }),
        }
    }

    /// A timer that measures nothing and says so.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    /// Whether this timer can produce readings at all.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Total queries this timer can hold (2 per bracket); 0 when disabled.
    pub fn capacity(&self) -> u32 {
        self.inner.as_ref().map_or(0, |i| i.capacity)
    }

    /// Timestamps written since the last [`Self::reset`].
    pub fn written(&self) -> u32 {
        self.inner.as_ref().map_or(0, |i| i.written)
    }

    /// Begin a fresh frame's worth of brackets.
    ///
    /// Must be called before re-using a timer, and only once the previous
    /// frame's readings have been read — the query set and readback buffer are
    /// overwritten from index 0.
    pub fn reset(&mut self) {
        if let Some(i) = &mut self.inner {
            i.written = 0;
        }
    }

    /// Open a bracket.  No-op when disabled or already full.
    pub fn begin(&mut self, enc: &mut wgpu::CommandEncoder) {
        self.mark(enc)
    }

    /// Close a bracket.  No-op when disabled or already full.
    pub fn end(&mut self, enc: &mut wgpu::CommandEncoder) {
        self.mark(enc)
    }

    fn mark(&mut self, enc: &mut wgpu::CommandEncoder) {
        if let Some(i) = &mut self.inner {
            if i.written < i.capacity {
                enc.write_timestamp(&i.query_set, i.written);
                i.written += 1;
            }
        }
    }

    /// Record the resolve and the copy-to-readback into the SAME encoder the
    /// timestamps went into, so the results are available as soon as that
    /// submission completes.
    ///
    /// Call once, after the last `end` of the frame.
    pub fn record_resolve(&self, enc: &mut wgpu::CommandEncoder) {
        if let Some(i) = &self.inner {
            if i.written == 0 {
                return;
            }
            enc.resolve_query_set(&i.query_set, 0..i.written, &i.resolve, 0);
            enc.copy_buffer_to_buffer(
                &i.resolve,
                0,
                &i.readback,
                0,
                (i.written * wgpu::QUERY_SIZE) as u64,
            );
        }
    }

    /// Elapsed GPU nanoseconds for every bracket recorded this frame, in order.
    ///
    /// One mapping for all of them: mapping is the expensive part, and reading
    /// brackets one at a time would map and unmap once per node.  The caller
    /// must have polled the submission carrying [`Self::record_resolve`] to
    /// completion first — otherwise this blocks until it has.
    ///
    /// `None` when disabled, when nothing was recorded, or when the mapping
    /// fails; never a zero standing in for a missing measurement.
    pub fn resolve_all(&self) -> Option<Vec<f64>> {
        let i = self.inner.as_ref()?;
        if i.written < 2 {
            return None;
        }
        let mapped_len = (i.written * wgpu::QUERY_SIZE) as u64;
        let slice = i.readback.slice(0..mapped_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Returns immediately when the submission has already completed, which
        // is the intended usage; it is a real wait only if the caller skipped
        // its own poll.
        i.device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;

        let ticks: Vec<u64> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        i.readback.unmap();

        let mut out = Vec::with_capacity(ticks.len() / 2);
        for pair in ticks.chunks_exact(2) {
            // Saturating rather than checked: a timestamp pair can come back
            // equal (work below the counter's resolution) or, on some drivers,
            // out of order across a queue boundary. Reporting 0 ns for "too fast
            // to measure" is honest; reporting a huge number from an underflow
            // is not.
            let delta = pair[1].saturating_sub(pair[0]);
            out.push(delta as f64 * i.period_ns as f64);
        }
        Some(out)
    }

    /// Elapsed GPU nanoseconds for one bracket.
    ///
    /// Convenience over [`Self::resolve_all`]; prefer that when reading several.
    pub fn resolve_pair(&self, pair: u32) -> Option<f64> {
        self.resolve_all()?.get(pair as usize).copied()
    }

    /// Convenience for the single-bracket case.
    pub fn resolve_last(&self) -> Option<f64> {
        self.resolve_pair(0)
    }

    /// The raw tick values of every timestamp recorded this frame, unpaired and
    /// unconverted.
    ///
    /// WHY RAW TICKS ARE EXPOSED.  [`Self::resolve_all`] can only measure spans
    /// *within* one timer, and the interesting question at 4K is a span BETWEEN
    /// two frames: how much GPU-timeline time passes between one frame's graph
    /// finishing and the next frame's graph starting. That window is where
    /// wgpu's `write_buffer` staging copies execute — `pending_writes.pre_submit`
    /// (wgpu-core-0.19.4 `device/queue.rs:230-242`) prepends them to the *next*
    /// `queue.submit`, so they land in a command buffer this crate never encodes
    /// and cannot bracket. Subtracting one frame's first tick from the previous
    /// frame's last tick measures it.
    ///
    /// Ticks from different `GpuTimer`s are comparable because they come from the
    /// same queue's counter — which is also the reason this must not be used
    /// across two different queues.
    pub fn resolve_raw_ticks(&self) -> Option<Vec<u64>> {
        let i = self.inner.as_ref()?;
        if i.written == 0 {
            return None;
        }
        let mapped_len = (i.written * wgpu::QUERY_SIZE) as u64;
        let slice = i.readback.slice(0..mapped_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        i.device.poll(wgpu::Maintain::Wait);
        rx.recv().ok()?.ok()?;
        let ticks: Vec<u64> = {
            let view = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        i.readback.unmap();
        Some(ticks)
    }

    /// Nanoseconds per raw tick, for converting spans built from
    /// [`Self::resolve_raw_ticks`]. `None` when this timer measures nothing.
    pub fn period_ns(&self) -> Option<f64> {
        self.inner.as_ref().map(|i| i.period_ns as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::device::GpuDevice;

    /// A disabled timer must yield `None`, not `Some(0.0)`.
    ///
    /// Same rule `SystemMetrics` enforces: a zero duration and "we could not
    /// measure" are different claims, and once both print as a number the reader
    /// cannot tell them apart.
    #[test]
    fn disabled_timer_yields_no_reading() {
        let timer = GpuTimer::disabled();
        assert!(!timer.is_enabled());
        assert_eq!(timer.capacity(), 0);
        assert!(timer.resolve_last().is_none());
        assert!(timer.resolve_all().is_none());
    }

    /// A timer that was never used must report nothing rather than zero.
    #[test]
    fn unrecorded_frame_yields_no_reading() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        let timer = GpuTimer::new(&device, 1);
        assert!(timer.resolve_all().is_none(), "no brackets recorded => no reading");
    }

    /// A real bracketed submission must report a positive, plausible duration.
    #[test]
    fn brackets_a_real_submission() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        if !device.has_timestamp_queries {
            eprintln!("SKIP: adapter lacks TIMESTAMP_QUERY");
            return;
        }

        let mut timer = GpuTimer::new(&device, 1);
        assert_eq!(timer.capacity(), 2);

        const BYTES: u64 = 64 << 20; // 64 MB: big enough to outrun counter noise.
        let src = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_test_src"),
            size: BYTES,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dst = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_timer_test_dst"),
            size: BYTES,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.begin_frame();
        timer.begin(&mut enc);
        enc.copy_buffer_to_buffer(&src, 0, &dst, 0, BYTES);
        timer.end(&mut enc);
        timer.record_resolve(&mut enc);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));

        let ns = timer
            .resolve_last()
            .expect("a device reporting TIMESTAMP_QUERY must produce a reading");
        assert!(ns > 0.0, "a 64 MB GPU copy must take measurable time, got {ns} ns");
        assert!(
            ns < 1.0e9,
            "1 s for a 64 MB copy means the tick->ns conversion is wrong: {ns} ns"
        );
    }

    /// Several brackets in one frame must come back in recording order and be
    /// individually addressable — the property per-node profiling depends on.
    #[test]
    fn multiple_brackets_resolve_independently() {
        let Ok(device) = pollster::block_on(GpuDevice::new_headless()) else {
            eprintln!("SKIP: no GPU on this machine");
            return;
        };
        if !device.has_timestamp_queries {
            eprintln!("SKIP: adapter lacks TIMESTAMP_QUERY");
            return;
        }

        // Two copies of very different sizes, so "the brackets are independent"
        // is observable rather than asserted: a timer that reported one figure
        // for both, or that mixed the pairs up, would not show a 16x size
        // difference between them.
        const SMALL: u64 = 1 << 20;
        const LARGE: u64 = 64 << 20;
        let mk = |size, usage, label| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let s_src = mk(SMALL, wgpu::BufferUsages::COPY_SRC, "s_src");
        let s_dst = mk(SMALL, wgpu::BufferUsages::COPY_DST, "s_dst");
        let l_src = mk(LARGE, wgpu::BufferUsages::COPY_SRC, "l_src");
        let l_dst = mk(LARGE, wgpu::BufferUsages::COPY_DST, "l_dst");

        let mut timer = GpuTimer::new(&device, 2);
        let mut enc = device.begin_frame();
        timer.begin(&mut enc);
        enc.copy_buffer_to_buffer(&s_src, 0, &s_dst, 0, SMALL);
        timer.end(&mut enc);
        timer.begin(&mut enc);
        enc.copy_buffer_to_buffer(&l_src, 0, &l_dst, 0, LARGE);
        timer.end(&mut enc);
        timer.record_resolve(&mut enc);
        let sid = device.submit(enc);
        device
            .device
            .poll(wgpu::Maintain::WaitForSubmissionIndex(sid));

        let all = timer.resolve_all().expect("two brackets were recorded");
        assert_eq!(all.len(), 2, "one reading per bracket");
        assert!(all.iter().all(|&ns| ns >= 0.0));
        assert!(
            all[1] > all[0],
            "the 64 MB copy must outlast the 1 MB copy; got small={} ns large={} ns \
             (equal or inverted means the pairs are being mixed up)",
            all[0], all[1]
        );

        // And `reset` must make the timer reusable from index 0.
        timer.reset();
        assert_eq!(timer.written(), 0);
    }
}
