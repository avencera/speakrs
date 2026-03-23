use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::Instant;

use color_eyre::eyre::{Result, bail, ensure};
use speakrs::inference::ExecutionMode;
use speakrs::inference::embedding::EmbeddingModel;
use speakrs::inference::segmentation::SegmentationModel;
use speakrs::pipeline::{
    DiarizationPipeline, FAST_SEGMENTATION_STEP_SECONDS, SEGMENTATION_STEP_SECONDS,
};

use crate::wav;

#[derive(Debug, Clone, Copy)]
pub enum DiarizeMode {
    Speakrs(SpeakrsMode),
    Pyannote(PyannoteDevice),
}

#[derive(Debug, Clone, Copy)]
pub enum SpeakrsMode {
    Cpu,
    Coreml,
    CoremlFast,
    Cuda,
    CudaFast,
}

#[derive(Debug, Clone, Copy)]
pub enum PyannoteDevice {
    Cpu,
    Mps,
    Cuda,
}

impl SpeakrsMode {
    fn execution_mode(self) -> ExecutionMode {
        match self {
            Self::Cpu => ExecutionMode::Cpu,
            Self::Coreml => ExecutionMode::CoreMl,
            Self::CoremlFast => ExecutionMode::CoreMlFast,
            Self::Cuda => ExecutionMode::Cuda,
            Self::CudaFast => ExecutionMode::CudaFast,
        }
    }

    fn step_seconds(self) -> f64 {
        match self {
            Self::CoremlFast | Self::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
            _ => SEGMENTATION_STEP_SECONDS,
        }
    }
}

impl PyannoteDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Mps => "mps",
            Self::Cuda => "cuda",
        }
    }
}

impl FromStr for DiarizeMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(Self::Speakrs(SpeakrsMode::Cpu)),
            "coreml" => Ok(Self::Speakrs(SpeakrsMode::Coreml)),
            "coreml-fast" => Ok(Self::Speakrs(SpeakrsMode::CoremlFast)),
            "cuda" => Ok(Self::Speakrs(SpeakrsMode::Cuda)),
            "cuda-fast" => Ok(Self::Speakrs(SpeakrsMode::CudaFast)),
            "pyannote-cpu" => Ok(Self::Pyannote(PyannoteDevice::Cpu)),
            "pyannote-mps" => Ok(Self::Pyannote(PyannoteDevice::Mps)),
            "pyannote-cuda" => Ok(Self::Pyannote(PyannoteDevice::Cuda)),
            _ => Err(format!(
                "unknown mode '{s}', expected one of: cpu, coreml, coreml-fast, cuda, cuda-fast, pyannote-cpu, pyannote-mps, pyannote-cuda"
            )),
        }
    }
}

impl fmt::Display for DiarizeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Speakrs(SpeakrsMode::Cpu) => write!(f, "cpu"),
            Self::Speakrs(SpeakrsMode::Coreml) => write!(f, "coreml"),
            Self::Speakrs(SpeakrsMode::CoremlFast) => write!(f, "coreml-fast"),
            Self::Speakrs(SpeakrsMode::Cuda) => write!(f, "cuda"),
            Self::Speakrs(SpeakrsMode::CudaFast) => write!(f, "cuda-fast"),
            Self::Pyannote(PyannoteDevice::Cpu) => write!(f, "pyannote-cpu"),
            Self::Pyannote(PyannoteDevice::Mps) => write!(f, "pyannote-mps"),
            Self::Pyannote(PyannoteDevice::Cuda) => write!(f, "pyannote-cuda"),
        }
    }
}

pub fn run(mode: DiarizeMode, wav_files: Vec<PathBuf>) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    ensure!(!wav_files.is_empty(), "no WAV files specified");

    match mode {
        DiarizeMode::Pyannote(device) => {
            let wav_path = wav_files[0].to_string_lossy();
            let output = run_pyannote_sidecar(device.as_str(), &wav_path)?;
            print!("{output}");
        }
        DiarizeMode::Speakrs(speakrs_mode) => {
            let execution_mode = speakrs_mode.execution_mode();
            let models_dir = resolve_models_dir();

            let step = speakrs_mode.step_seconds();
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
            let mut pipeline =
                DiarizationPipeline::new(&mut seg_model, &mut emb_model, &models_dir)?;

            let total = wav_files.len();
            let mut cumulative = 0.0f64;
            let profile_file_timing = std::env::var_os("SPEAKRS_PROFILE_FILE_TIMING").is_some();

            for (i, wav_path) in wav_files.iter().enumerate() {
                let file_id = wav_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "file1".to_string());

                let file_start = Instant::now();
                let load_start = Instant::now();
                let (samples, sr) = wav::load_wav_samples(&wav_path.to_string_lossy())?;
                let load_elapsed = load_start.elapsed();
                ensure!(sr == 16000, "expected 16kHz WAV, got {sr}Hz");

                let start = Instant::now();
                let result = pipeline.run_with_file_id(&samples, &file_id)?;
                let elapsed = start.elapsed().as_secs_f64();
                let drop_start = Instant::now();
                let speakrs::pipeline::DiarizationResult { rttm, .. } = result;
                let drop_elapsed = drop_start.elapsed();
                cumulative += elapsed;

                let avg = cumulative / (i + 1) as f64;
                let remaining = (total - i - 1) as f64 * avg;
                let eta = format_eta(remaining);
                let total_elapsed = format_eta(cumulative);
                let now = chrono::Local::now().format("%H:%M:%S");
                eprintln!(
                    "  [{}/{}] {file_id}: {elapsed:.1}s (elapsed {total_elapsed}, ETA {eta}) [{now}]",
                    i + 1,
                    total
                );
                let output_start = Instant::now();
                print!("{rttm}");
                let output_elapsed = output_start.elapsed();
                if profile_file_timing {
                    eprintln!(
                        "    file timing {file_id}: load_ms={} pipeline_ms={} drop_ms={} output_ms={} total_ms={}",
                        load_elapsed.as_millis(),
                        (elapsed * 1000.0).round() as u64,
                        drop_elapsed.as_millis(),
                        output_elapsed.as_millis(),
                        file_start.elapsed().as_millis(),
                    );
                }
            }
        }
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
