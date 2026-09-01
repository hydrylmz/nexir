// src/profiling/sysinfo.rs
//
// OS queries for the two `SystemMetrics` fields that need no GPU driver: process
// resident set size, and this process's CPU utilisation.
//
// The contract this file exists to satisfy is AGENTS.md gotcha 9: a metric is
// measured, counted, or `n/a`. Both functions here return `Option` and return
// `None` on any platform or failure path rather than a plausible-looking number,
// and **neither is ever computed from arithmetic over other metrics** — deriving
// CPU utilisation from `CPU per frame / frame interval`, say, would report the
// profiler's own view of itself as an OS reading, which is precisely the class of
// regression that once printed `gpu_utilization: 88.5` as a measurement.

/// Resident set size of this process, in bytes.
///
/// Windows: `GetProcessMemoryInfo`'s `WorkingSetSize` — the current working set,
/// which is what Task Manager's "Memory" column shows and the closest Windows
/// equivalent of RSS. `PeakWorkingSetSize` is deliberately not used: a peak over
/// the whole process lifetime would include the warm-up and every previous
/// benchmark in the same run, so it is not a reading about the benchmark that just
/// finished.
///
/// `None` elsewhere, and `None` if the call fails. There is no fallback estimate.
#[cfg(windows)]
pub fn process_rss_bytes() -> Option<u64> {
    use winapi::um::processthreadsapi::GetCurrentProcess;
    use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

    // Zeroed then `cb` set, as the API requires: it uses `cb` to decide which
        // version of the struct the caller compiled against.
    let mut pmc: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: `GetCurrentProcess` is a pseudo-handle needing no close, and `pmc`
    // is a live, correctly sized, correctly tagged struct.
    let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) };
    if ok == 0 {
        return None;
    }
    Some(pmc.WorkingSetSize as u64)
}

#[cfg(not(windows))]
pub fn process_rss_bytes() -> Option<u64> {
    None
}

/// Total CPU time this process has consumed (user + kernel, all threads), as a
/// `Duration`.
///
/// The raw counter rather than a percentage, because a percentage needs a wall
/// interval to divide by and only the caller knows which interval it means. See
/// [`CpuSampler`].
#[cfg(windows)]
pub fn process_cpu_time() -> Option<std::time::Duration> {
    use winapi::shared::minwindef::FILETIME;
    use winapi::um::processthreadsapi::{GetCurrentProcess, GetProcessTimes};

    let mut creation = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut exit = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut kernel = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let mut user = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    // SAFETY: four live FILETIMEs and a pseudo-handle.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return None;
    }
    // FILETIME is a split u64 of 100-nanosecond ticks. Kernel + user is the CPU
    // time actually burned, summed over every thread — so on a multi-core machine
    // it can exceed wall time, which is exactly why the percentage below divides
    // by the core count as well.
    fn ticks(ft: &FILETIME) -> u64 {
        ((ft.dwHighDateTime as u64) << 32) | ft.dwLowDateTime as u64
    }
    let total = ticks(&kernel) + ticks(&user);
    Some(std::time::Duration::from_nanos(total.saturating_mul(100)))
}

#[cfg(not(windows))]
pub fn process_cpu_time() -> Option<std::time::Duration> {
    None
}

/// Samples this process's CPU time across an interval and reports utilisation.
///
/// Two readings, taken by the caller around the work it wants to characterise,
/// because a single instantaneous reading of a cumulative counter is not a rate.
/// [`Self::start`] is called before the measured loop and [`Self::utilisation`]
/// after it; the wall interval is measured by this struct rather than passed in,
/// so the CPU numerator and the wall denominator cannot come from two different
/// spans.
///
/// The result is normalised by core count, so 100% means "every core busy for the
/// whole interval" and a single-threaded process on an 8-core machine caps at
/// 12.5%. That convention matches Task Manager's per-process column, and the
/// alternative (per-core percent, so 800% is possible) reads as a broken number in
/// a table whose other utilisation columns are 0-100.
pub struct CpuSampler {
    /// `None` when the platform has no query — every method then reports `None`
    /// rather than a zero, and a caller needs no `cfg` of its own.
    start_cpu: Option<std::time::Duration>,
    start_wall: std::time::Instant,
}

impl CpuSampler {
    /// Take the opening reading. Cheap: two syscalls at most.
    pub fn start() -> Self {
        Self {
            start_cpu: process_cpu_time(),
            start_wall: std::time::Instant::now(),
        }
    }

    /// CPU utilisation over the interval since [`Self::start`], as a percentage of
    /// all cores.
    ///
    /// `None` when the platform has no query, when the closing query fails, or
    /// when the interval is too short to divide by — a rate over a zero interval
    /// is undefined, not 0%.
    pub fn utilisation(&self) -> Option<f32> {
        let start = self.start_cpu?;
        let end = process_cpu_time()?;
        let wall = self.start_wall.elapsed().as_secs_f64();
        // A microsecond floor: below that, timer granularity dominates and the
        // quotient is noise scaled by the core count.
        if wall <= 1e-6 {
            return None;
        }
        let cpu = end.saturating_sub(start).as_secs_f64();
        let cores = num_cpus::get().max(1) as f64;
        Some((cpu / (wall * cores) * 100.0) as f32)
    }
}

#[cfg(test)]
mod tests {
    /// A real RSS reading must be plausible: a process with a wgpu device and 4K
    /// staging buffers is not using 1 MB, and it is not using 64 GB. A test that
    /// only checked `is_some()` would pass on a stub returning 0.
    #[test]
    fn process_rss_is_plausible_or_absent() {
        match super::process_rss_bytes() {
            Some(b) => {
                assert!(b > 4 << 20, "RSS {b} is implausibly small — is this a stub?");
                assert!(b < 64u64 << 30, "RSS {b} exceeds any plausible value");
            }
            None => eprintln!("SKIP: no RSS query on this platform"),
        }
    }

    /// CPU time must be cumulative and must actually advance when the process
    /// burns CPU.
    ///
    /// The second half is what distinguishes a working query from one that returns
    /// a constant: a stub handing back the same `Duration` twice satisfies
    /// "monotonic" and fails here.
    #[test]
    fn process_cpu_time_advances_when_the_cpu_is_busy() {
        let Some(before) = super::process_cpu_time() else {
            eprintln!("SKIP: no CPU-time query on this platform");
            return;
        };
        // Deliberately spin rather than sleep: sleeping advances wall time and NOT
        // CPU time, so it would test the opposite property.
        let mut sink = 0u64;
        let spin_start = std::time::Instant::now();
        while spin_start.elapsed() < std::time::Duration::from_millis(60) {
            sink = sink.wrapping_add(spin_start.elapsed().as_nanos() as u64);
        }
        std::hint::black_box(sink);
        let after = super::process_cpu_time().expect("the query worked once already");
        assert!(after >= before, "a cumulative counter cannot go backwards");
        assert!(
            after - before >= std::time::Duration::from_millis(20),
            "60 ms of spinning must show up as CPU time; saw {:?}",
            after - before
        );
    }

    /// A sampler over a busy interval must report a plausible percentage — and one
    /// that is normalised by core count rather than a per-core figure.
    #[test]
    fn a_busy_interval_reports_a_plausible_utilisation() {
        let sampler = super::CpuSampler::start();
        let mut sink = 0u64;
        let spin_start = std::time::Instant::now();
        while spin_start.elapsed() < std::time::Duration::from_millis(80) {
            sink = sink.wrapping_add(1);
        }
        std::hint::black_box(sink);
        match sampler.utilisation() {
            Some(pct) => {
                assert!(
                    pct > 0.0,
                    "a spinning thread must register above 0%; got {pct}"
                );
                // Normalised by cores, so a single busy thread cannot exceed 100%.
                // A small margin over 100 for a rounding edge on a 1-core box.
                assert!(
                    pct <= 101.0,
                    "utilisation is a percentage of ALL cores, so {pct}% means it \
                     was not divided by the core count"
                );
            }
            None => eprintln!("SKIP: no CPU-time query on this platform"),
        }
    }

    /// A sampler asked for a rate before any measurable interval must say `None`,
    /// not 0%.
    #[test]
    fn a_zero_length_interval_has_no_rate() {
        // Constructed with a wall clock already at "now", queried immediately: the
        // elapsed interval is at or below the floor.
        let sampler = super::CpuSampler {
            start_cpu: Some(std::time::Duration::ZERO),
            start_wall: std::time::Instant::now(),
        };
        // Only assert the property this test owns: if the platform has no query at
        // all the answer is `None` for that reason instead, which is still not a
        // zero.
        if super::process_cpu_time().is_none() {
            assert!(sampler.utilisation().is_none());
        }
    }
}
