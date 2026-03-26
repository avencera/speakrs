use std::time::Duration;

use color_eyre::eyre::Result;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_build_xtask, features_for_mode};
use crate::cmd::{find_fluidaudio, project_root, run_cmd, wav_duration_seconds};
use crate::fluidaudio;
mod compare;
mod der;
mod jobs;
mod report;
mod runner;
mod selection;
mod types;

pub use compare::compare;
pub use der::{DerArgs, der};
pub use jobs::{
    BenchmarkJobConfig, BenchmarkJobResult, GpuBenchmarkSuiteConfig, ProgressUpdate, gpu_impls,
    run_benchmark_job, run_gpu_benchmark_suite, run_speakrs_gpu, validate_gpu_impls,
};
pub use report::{DerResultsWriter, format_eta, now_stamp};
pub(crate) use selection::discover_files;
pub(crate) use types::{BatchCommandRunner, PREFLIGHT_TIMEOUT, PyannoteRsFileRunner};
pub use types::{
    BenchmarkMetadata, DerAccumulation, DerImplResult, DerImplStatus, ImplType, PyannoteBatchSizes,
};

use runner::{CommandSpec, capture_benchmark_cmd};

pub fn run(
    source: &str,
    python_device: &str,
    runs: u32,
    warmups: u32,
    rust_mode: &str,
) -> Result<()> {
    let prepared_audio = prepare_audio(source)?;
    let wav = prepared_audio.wav_path();
    let features = features_for_mode(rust_mode);

    println!();
    println!("=== Building Rust binary ===");
    cargo_build_xtask(&features)?;

    let root = project_root();
    let binary = root.join("target/release/xtask");
    let audio_seconds = wav_duration_seconds(wav)?;

    println!();
    println!("=== Benchmark ===");

    let rust_command = CommandSpec::new(&binary)
        .arg("diarize")
        .arg("--mode")
        .arg(rust_mode)
        .arg(wav)
        .current_dir(&root);
    let rust_result = bench_tool("Rust", &rust_command, runs, warmups)?;

    let python_command = CommandSpec::new("uv")
        .arg("run")
        .arg("--project")
        .arg("scripts/pyannote-bench")
        .arg("python")
        .arg("scripts/pyannote-bench/diarize.py")
        .arg("--device")
        .arg(python_device)
        .arg(wav)
        .current_dir(&root);
    let python_result = bench_tool("Python", &python_command, runs, warmups)?;

    println!(
        "Audio duration: {:.2} minutes ({audio_seconds:.1}s)",
        audio_seconds / 60.0
    );
    println!("Runs: {runs} measured, {warmups} warmup");
    println!("Rust mode: {rust_mode}");
    println!("Python device: {python_device}");
    println!();
    println!("Name\tmean_s\tmin_s\tspeed");
    for (name, times) in [("Rust", &rust_result), ("Python", &python_result)] {
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        println!("{name}\t{mean:.3}\t{min:.3}\t{:.2}x", audio_seconds / mean);
    }

    Ok(())
}

fn bench_tool(name: &str, command: &CommandSpec, runs: u32, warmups: u32) -> Result<Vec<f64>> {
    for _ in 0..warmups {
        run_once(command)?;
    }

    let mut times = Vec::with_capacity(runs as usize);
    for _ in 0..runs {
        let elapsed = run_once(command)?;
        times.push(elapsed);
        print!("  {name}: {elapsed:.2}s\r");
    }
    println!("  {name}: done ({runs} runs)");
    Ok(times)
}

fn run_once(command: &CommandSpec) -> Result<f64> {
    let mut benchmark_command = command.build_command();
    let output = capture_benchmark_cmd(&mut benchmark_command, Duration::from_secs(30 * 60))?;
    Ok(output.elapsed_seconds)
}
