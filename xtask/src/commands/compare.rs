use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_run, features_for_mode};
use crate::cmd::{find_fluidaudio, project_root, run_cmd, tee_cmd};
use crate::compare_rttm::compare_rttm_files;
use crate::fluidaudio;

pub fn run(source: &str, python_device: &str, rust_mode: &str) -> Result<()> {
    let (wav, temp_dir) = prepare_audio(source)?.into_parts();
    let features = features_for_mode(rust_mode);
    let wav_str = wav.to_string_lossy();

    let rust_rttm = temp_dir.path().join("rust.rttm");
    let python_rttm = temp_dir.path().join("python.rttm");

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
                "--project",
                "scripts/pyannote-bench",
                "python",
                "scripts/pyannote-bench/diarize.py",
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
    let (wav, temp_dir) = prepare_audio(source)?.into_parts();
    let features = features_for_mode(rust_mode);
    let wav_str = wav.to_string_lossy();

    let rust_rttm = temp_dir.path().join("rust.rttm");
    let python_cpu_rttm = temp_dir.path().join("python_cpu.rttm");
    let fluidaudio_json = temp_dir.path().join("fluidaudio.json");
    let fluidaudio_rttm = temp_dir.path().join("fluidaudio.rttm");

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
                "--project",
                "scripts/pyannote-bench",
                "python",
                "scripts/pyannote-bench/diarize.py",
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
    let fluidaudio_path = find_fluidaudio().ok_or_else(|| {
        color_eyre::eyre::eyre!(
            "FluidAudio not found. Set FLUIDAUDIO_PATH or clone to ~/.cache/cmd/repos/github.com/FluidInference/FluidAudio"
        )
    })?;
    run_cmd(
        Command::new("swift")
            .args(["run", "--package-path"])
            .arg(&fluidaudio_path)
            .args(["fluidaudiocli", "process"])
            .arg(wav_str.as_ref())
            .args(["--mode", "offline", "--output"])
            .arg(&fluidaudio_json),
    )?;
    let rttm_text = fluidaudio::json_to_rttm(&fluidaudio_json, "file1")?;
    std::fs::write(&fluidaudio_rttm, &rttm_text)?;
    print!("{rttm_text}");

    println!();
    println!("=== speakrs vs pyannote CPU ===");
    compare_rttm_files(&rust_rttm, &python_cpu_rttm)?;

    println!();
    println!("=== FluidAudio vs pyannote CPU ===");
    compare_rttm_files(&fluidaudio_rttm, &python_cpu_rttm)?;

    Ok(())
}
