use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::Result;

use crate::audio::prepare_audio;
use crate::cargo::{cargo_build, features_for_mode};
use crate::cmd::{project_root, run_cmd};
use crate::python::uv_run;

pub fn run(
    source: &str,
    python_device: &str,
    runs: u32,
    warmups: u32,
    rust_mode: &str,
) -> Result<()> {
    let (wav, _tmp) = prepare_audio(source)?;
    let features = features_for_mode(rust_mode);
    let wav_str = wav.to_string_lossy();

    println!();
    println!("=== Building Rust binary ===");
    cargo_build("diarize", &features)?;

    println!();
    println!("=== Benchmark ===");
    uv_run(&[
        "scripts/benchmark_diarization.py",
        &wav_str,
        "--rust-binary",
        "target/release/diarize",
        "--rust-mode",
        rust_mode,
        "--python-script",
        "scripts/diarize_pyannote.py",
        "--python-device",
        python_device,
        "--runs",
        &runs.to_string(),
        "--warmups",
        &warmups.to_string(),
    ])
}

pub fn compare(source: &str, runs: u32, warmups: u32) -> Result<()> {
    let (wav, _tmp) = prepare_audio(source)?;
    let wav_str = wav.to_string_lossy();
    let root = project_root();

    println!();
    println!("=== Building binaries ===");
    cargo_build("diarize", &["native-coreml".to_string()])?;
    run_cmd(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root.join("scripts/pyannote_rs_bench")),
    )?;

    println!();
    println!("=== Running benchmarks ===");
    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    ensure_pyannote_rs_emb_model(&emb_model)?;

    let mut args = vec![
        "scripts/benchmark_comparison.py".to_string(),
        wav_str.to_string(),
        "--speakrs-binary".to_string(),
        "target/release/diarize".to_string(),
        "--pyannote-rs-binary".to_string(),
        "scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs".to_string(),
        "--pyannote-rs-seg-model".to_string(),
        seg_model.to_string_lossy().to_string(),
        "--pyannote-rs-emb-model".to_string(),
        emb_model.to_string_lossy().to_string(),
        "--python-script".to_string(),
        "scripts/diarize_pyannote.py".to_string(),
        "--warmups".to_string(),
        warmups.to_string(),
        "--runs".to_string(),
        runs.to_string(),
    ];
    append_fluidaudio_args(&root, &mut args);

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    uv_run(&arg_refs)
}

pub fn der(max_files: u32, max_minutes: u32) -> Result<()> {
    let root = project_root();
    let vox_dir = root.join("fixtures/voxconverse");
    ensure_voxconverse(&vox_dir)?;

    println!("=== Building binaries ===");
    cargo_build("diarize", &["native-coreml".to_string()])?;
    run_cmd(
        Command::new("cargo")
            .args(["build", "--release"])
            .current_dir(root.join("scripts/pyannote_rs_bench")),
    )?;

    let models_dir = root.join("fixtures/models");
    let seg_model = models_dir.join("segmentation-3.0.onnx");
    let emb_model = models_dir.join("wespeaker_en_voxceleb_CAM++.onnx");
    ensure_pyannote_rs_emb_model(&emb_model)?;

    println!();
    println!("=== Running DER benchmark ===");

    let mut args = vec![
        "scripts/benchmark_der.py".to_string(),
        vox_dir.to_string_lossy().to_string(),
        "--speakrs-binary".to_string(),
        "target/release/diarize".to_string(),
        "--pyannote-rs-binary".to_string(),
        "scripts/pyannote_rs_bench/target/release/diarize-pyannote-rs".to_string(),
        "--pyannote-rs-seg-model".to_string(),
        seg_model.to_string_lossy().to_string(),
        "--pyannote-rs-emb-model".to_string(),
        emb_model.to_string_lossy().to_string(),
        "--python-script".to_string(),
        "scripts/diarize_pyannote.py".to_string(),
        "--max-files".to_string(),
        max_files.to_string(),
        "--max-minutes".to_string(),
        max_minutes.to_string(),
    ];
    append_fluidaudio_args(&root, &mut args);

    // set SPEAKRS_MODELS_DIR for the benchmark script
    // SAFETY: single-threaded CLI, no other threads reading env vars
    unsafe { std::env::set_var("SPEAKRS_MODELS_DIR", models_dir) };

    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    uv_run(&arg_refs)
}

fn ensure_pyannote_rs_emb_model(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    println!("Downloading pyannote-rs embedding model...");
    run_cmd(
        Command::new("curl")
            .args(["-L", "-o"])
            .arg(path)
            .arg("https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM++.onnx"),
    )
}

fn ensure_voxconverse(vox_dir: &Path) -> Result<()> {
    let wav_dir = vox_dir.join("wav");
    let rttm_dir = vox_dir.join("rttm");

    if !wav_dir.is_dir() {
        println!("=== Downloading VoxConverse dev WAVs ===");
        fs::create_dir_all(vox_dir)?;
        let zip_path = vox_dir.join("voxconverse_dev_wav.zip");
        run_cmd(Command::new("curl").args(["-L", "-o"]).arg(&zip_path).arg(
            "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
        ))?;
        run_cmd(
            Command::new("unzip")
                .args(["-q"])
                .arg(&zip_path)
                .arg("-d")
                .arg(vox_dir),
        )?;
        // the zip may extract to audio/ instead of wav/
        let audio_dir = vox_dir.join("audio");
        if audio_dir.is_dir() && !wav_dir.is_dir() {
            fs::rename(&audio_dir, &wav_dir)?;
        }
        let _ = fs::remove_file(&zip_path);
    }

    if !rttm_dir.is_dir() {
        println!("=== Downloading VoxConverse ground truth RTTMs ===");
        let tmp_clone = std::env::temp_dir().join("voxconverse-clone");
        let _ = fs::remove_dir_all(&tmp_clone);
        run_cmd(
            Command::new("git")
                .args([
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/joonson/voxconverse",
                ])
                .arg(&tmp_clone),
        )?;
        fs::create_dir_all(&rttm_dir)?;
        for entry in fs::read_dir(tmp_clone.join("dev"))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "rttm") {
                fs::copy(&path, rttm_dir.join(entry.file_name()))?;
            }
        }
        let _ = fs::remove_dir_all(&tmp_clone);
    }

    Ok(())
}

fn find_fluidaudio() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FLUIDAUDIO_PATH") {
        let p = PathBuf::from(path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let default = PathBuf::from(home).join(".cache/cmd/repos/github.com/FluidInference/FluidAudio");
    if default.is_dir() {
        Some(default)
    } else {
        None
    }
}

fn append_fluidaudio_args(root: &Path, args: &mut Vec<String>) {
    match find_fluidaudio() {
        Some(path) => {
            args.extend([
                "--fluidaudio-path".to_string(),
                path.to_string_lossy().to_string(),
                "--fluidaudio-rttm-script".to_string(),
                root.join("scripts/fluidaudio_json_to_rttm.py")
                    .to_string_lossy()
                    .to_string(),
            ]);
        }
        None => {
            println!("Note: FluidAudio not found, skipping");
        }
    }
}
