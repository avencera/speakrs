use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use clap::ValueEnum;
use color_eyre::eyre::{Result, bail, ensure};
use speakrs::clustering::plda::PldaTransform;
use speakrs::inference::ExecutionMode;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{FAST_SEGMENTATION_STEP_SECONDS, SEGMENTATION_STEP_SECONDS, diarize};

use crate::wav;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DiarizeMode {
    Cpu,
    Coreml,
    #[value(name = "coreml-fast")]
    CoremlFast,
    Cuda,
    #[value(name = "pyannote-cpu")]
    PyannoteCpu,
    #[value(name = "pyannote-mps")]
    PyannoteMps,
    #[value(name = "pyannote-cuda")]
    PyannoteCuda,
}

impl DiarizeMode {
    fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Cpu => ExecutionMode::Cpu,
            Self::Coreml => ExecutionMode::CoreMl,
            Self::CoremlFast => ExecutionMode::CoreMlFast,
            Self::Cuda => ExecutionMode::Cuda,
            Self::PyannoteCpu | Self::PyannoteMps | Self::PyannoteCuda => unreachable!(),
        }
    }

    fn step_seconds(self) -> f64 {
        match self {
            Self::CoremlFast => FAST_SEGMENTATION_STEP_SECONDS,
            _ => SEGMENTATION_STEP_SECONDS,
        }
    }

    fn pyannote_device(self) -> Option<&'static str> {
        match self {
            Self::PyannoteCpu => Some("cpu"),
            Self::PyannoteMps => Some("mps"),
            Self::PyannoteCuda => Some("cuda"),
            _ => None,
        }
    }
}

pub fn run(mode: DiarizeMode, wav_files: Vec<PathBuf>) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    ensure!(!wav_files.is_empty(), "no WAV files specified");

    if let Some(device) = mode.pyannote_device() {
        let wav_path = wav_files[0].to_string_lossy();
        let output = run_pyannote_sidecar(device, &wav_path)?;
        print!("{output}");
        return Ok(());
    }

    let execution_mode = mode.execution_mode();
    let models_dir = resolve_models_dir();

    let step = mode.step_seconds();
    let mut seg_model = SegmentationModel::with_mode(
        models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
        step as f32,
        execution_mode,
    )?;
    let mut emb_model = EmbeddingModel::with_mode(
        models_dir
            .join("wespeaker-voxceleb-resnet34.onnx")
            .to_str()
            .unwrap(),
        execution_mode,
    )?;
    let plda = PldaTransform::from_dir(&models_dir)?;

    let total = wav_files.len();
    let mut cumulative = 0.0f64;

    for (i, wav_path) in wav_files.iter().enumerate() {
        let file_id = wav_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file1".to_string());

        let (samples, sr) = wav::load_wav_samples(&wav_path.to_string_lossy())?;
        ensure!(sr == 16000, "expected 16kHz WAV, got {sr}Hz");

        let start = Instant::now();
        let result = diarize(&mut seg_model, &mut emb_model, &plda, &samples, &file_id)?;
        let elapsed = start.elapsed().as_secs_f64();
        cumulative += elapsed;

        let avg = cumulative / (i + 1) as f64;
        let remaining = (total - i - 1) as f64 * avg;
        let eta = format_eta(remaining);
        let now = chrono::Local::now().format("%H:%M:%S");
        eprintln!(
            "  [{}/{}] {file_id}: {elapsed:.1}s (ETA {eta}) [{now}]",
            i + 1,
            total
        );
        print!("{}", result.rttm);
    }
    Ok(())
}

fn resolve_models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SPEAKRS_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models")
        .to_path_buf()
}

fn run_pyannote_sidecar(device: &str, wav_path: &str) -> Result<String> {
    let script_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/diarize_pyannote.py");
    let output = Command::new("uv")
        .arg("run")
        .arg(script_path)
        .arg("--device")
        .arg(device)
        .arg(wav_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("pyannote sidecar exited with {}: {stderr}", output.status);
    }

    Ok(String::from_utf8(output.stdout)?)
}

fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.0}s")
    } else {
        let mins = (seconds / 60.0).floor() as u64;
        let secs = (seconds % 60.0).round() as u64;
        format!("{mins}m {secs:02}s")
    }
}
