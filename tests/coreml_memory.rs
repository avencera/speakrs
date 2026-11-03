//! Opt-in check that CoreML memory stays bounded on long files
//!
//! CoreML keeps a set of GPU buffers for every concurrent prediction a model has served, so
//! the number of batched segmentation predictions in flight sets the peak footprint. With
//! one in flight per core, a 10-minute file peaked near 14 GiB on an M4 Pro; three in flight
//! keep it near 7 GiB, most of which is three batch-64 segmentation sets (about 1 GiB each),
//! the 111-window chunk embedding model's set (about 1.9 GiB) and the loaded models
//!
//! The test reads the whole process's peak footprint, so it lives in its own test binary
//! and runs alone:
//!
//! ```sh
//! cargo test --release --features coreml --test coreml_memory -- --ignored
//! ```
#![cfg(all(target_os = "macos", feature = "online", feature = "coreml"))]

use speakrs::OwnedDiarizationPipeline;
use speakrs::inference::ExecutionMode;

mod support;

use support::{fixture_path, load_wav_samples};

/// Audio length that gives segmentation more batch-64 batches than the eight workers a
/// many-core Mac would otherwise run at once
const LONG_AUDIO_SECONDS: usize = 600;

/// Peak the long file must stay under, between the 7.2 GiB this test measured with three
/// predictions in flight and the 12.5-12.9 GiB it measured with eight on an M4 Pro
const MAX_PEAK_FOOTPRINT_BYTES: u64 = 9 << 30;

/// Leading fields of `struct rusage_info_v4` from `<sys/resource.h>`
#[repr(C)]
#[derive(Default)]
struct RusageInfoV4 {
    uuid: [u8; 16],
    user_time: u64,
    system_time: u64,
    pkg_idle_wkups: u64,
    interrupt_wkups: u64,
    pageins: u64,
    wired_size: u64,
    resident_size: u64,
    phys_footprint: u64,
    proc_start_abstime: u64,
    proc_exit_abstime: u64,
    child_user_time: u64,
    child_system_time: u64,
    child_pkg_idle_wkups: u64,
    child_interrupt_wkups: u64,
    child_pageins: u64,
    child_elapsed_abstime: u64,
    diskio_bytesread: u64,
    diskio_byteswritten: u64,
    cpu_time_qos: [u64; 7],
    billed_system_time: u64,
    serviced_system_time: u64,
    logical_writes: u64,
    lifetime_max_phys_footprint: u64,
    // the kernel writes the remaining v4 fields here
    rest: [u64; 16],
}

const RUSAGE_INFO_V4: i32 = 4;

unsafe extern "C" {
    fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut RusageInfoV4) -> i32;
}

/// Highest physical footprint of this process so far, the number `/usr/bin/time -l` reports
/// as "peak memory footprint"
fn lifetime_peak_footprint() -> u64 {
    let mut info = RusageInfoV4::default();
    let pid = i32::try_from(std::process::id()).unwrap();

    // SAFETY: `info` is at least as large as `rusage_info_v4`, which the kernel fills in
    let status = unsafe { proc_pid_rusage(pid, RUSAGE_INFO_V4, &mut info) };
    assert_eq!(status, 0, "proc_pid_rusage failed");
    info.lifetime_max_phys_footprint
}

/// Requires models deployed to HF (`cargo xtask models deploy`)
#[test]
#[ignore]
fn coreml_footprint_stays_bounded_on_long_audio() {
    let (clip, sample_rate) = load_wav_samples(&fixture_path("test.wav"));
    assert_eq!(sample_rate, 16_000);
    let long_audio: Vec<f32> = clip
        .iter()
        .copied()
        .cycle()
        .take(LONG_AUDIO_SECONDS * 16_000)
        .collect();

    let mut pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::CoreMl).unwrap();
    let result = pipeline.run(&long_audio).unwrap();
    assert!(result.rttm("long").contains("SPEAKER long 1"));

    let peak = lifetime_peak_footprint();
    eprintln!(
        "peak footprint {} MiB on {LONG_AUDIO_SECONDS} s of audio",
        peak >> 20
    );
    assert!(
        peak < MAX_PEAK_FOOTPRINT_BYTES,
        "peak footprint {} MiB on {LONG_AUDIO_SECONDS} s of audio, limit {} MiB",
        peak >> 20,
        MAX_PEAK_FOOTPRINT_BYTES >> 20,
    );
}
