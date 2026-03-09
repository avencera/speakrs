use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::Result;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_run, features_for_mode};
use crate::cmd::{project_root, run_cmd, tee_cmd};
use crate::compare_rttm::compare_rttm_files;

pub fn run(source: &str, python_device: &str, rust_mode: &str) -> Result<()> {
    let (wav, tmp) = prepare_audio(source)?;
    let features = features_for_mode(rust_mode);
    let wav_str = wav.to_string_lossy();

    let rust_rttm = tmp.path().join("rust.rttm");
    let python_rttm = tmp.path().join("python.rttm");

    println!();
    println!("=== speakrs (Rust) ===");
    cargo_run(
        "diarize",
        &features,
        &["--mode", rust_mode, &wav_str],
        Some(&rust_rttm),
    )?;

    println!();
    println!("=== pyannote (Python) ===");
    tee_cmd(
        Command::new("uv")
            .args([
                "run",
                "scripts/diarize_pyannote.py",
                "--device",
                python_device,
            ])
            .arg(wav_str.as_ref())
            .current_dir(project_root()),
        &python_rttm,
    )?;

    println!();
    println!("=== Comparison ===");
    compare_rttm_files(&rust_rttm, &python_rttm)?;

    Ok(())
}

pub fn rttm(a: &Path, b: &Path) -> Result<()> {
    compare_rttm_files(a, b)
}

pub fn accuracy(source: &str, rust_mode: &str) -> Result<()> {
    let (wav, tmp) = prepare_audio(source)?;
    let features = features_for_mode(rust_mode);
    let wav_str = wav.to_string_lossy();

    let rust_rttm = tmp.path().join("rust.rttm");
    let python_cpu_rttm = tmp.path().join("python_cpu.rttm");
    let fluidaudio_json = tmp.path().join("fluidaudio.json");
    let fluidaudio_rttm = tmp.path().join("fluidaudio.rttm");

    println!();
    println!("=== speakrs (Rust) ===");
    cargo_run(
        "diarize",
        &features,
        &["--mode", rust_mode, &wav_str],
        Some(&rust_rttm),
    )?;

    println!();
    println!("=== pyannote (Python CPU) ===");
    let python_cpu_str = python_cpu_rttm.to_string_lossy().to_string();
    run_cmd(
        Command::new("uv")
            .args([
                "run",
                "scripts/diarize_pyannote.py",
                "--device",
                "cpu",
                "--output",
                &python_cpu_str,
            ])
            .arg(wav_str.as_ref())
            .current_dir(project_root()),
    )?;
    print!("{}", std::fs::read_to_string(&python_cpu_rttm)?);

    println!();
    println!("=== FluidAudio ===");
    let fluidaudio_path = find_fluidaudio();
    let fluidaudio_json_str = fluidaudio_json.to_string_lossy().to_string();
    let fluidaudio_rttm_str = fluidaudio_rttm.to_string_lossy().to_string();

    run_cmd(
        Command::new("swift")
            .args(["run", "--package-path"])
            .arg(&fluidaudio_path)
            .args(["fluidaudiocli", "process"])
            .arg(wav_str.as_ref())
            .args(["--mode", "offline", "--output"])
            .arg(&fluidaudio_json_str),
    )?;
    run_cmd(
        Command::new("uv")
            .args([
                "run",
                "scripts/fluidaudio_json_to_rttm.py",
                &fluidaudio_json_str,
                "--output",
                &fluidaudio_rttm_str,
            ])
            .current_dir(project_root()),
    )?;
    print!("{}", std::fs::read_to_string(&fluidaudio_rttm)?);

    println!();
    println!("=== speakrs vs pyannote CPU ===");
    compare_rttm_files(&rust_rttm, &python_cpu_rttm)?;

    println!();
    println!("=== FluidAudio vs pyannote CPU ===");
    compare_rttm_files(&fluidaudio_rttm, &python_cpu_rttm)?;

    Ok(())
}

fn find_fluidaudio() -> PathBuf {
    if let Ok(path) = std::env::var("FLUIDAUDIO_PATH") {
        return PathBuf::from(path);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/cmd/repos/github.com/FluidInference/FluidAudio")
}
