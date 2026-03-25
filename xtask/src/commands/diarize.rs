use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::Instant;

use color_eyre::eyre::{Result, bail, ensure};
use speakrs::inference::CoreMlComputeUnits;
use speakrs::inference::ExecutionMode;
use speakrs::inference::{EmbeddingModel, SegmentationModel};
use speakrs::pipeline::{
    COREML_SEGMENTATION_STEP_SECONDS, CUDA_SEGMENTATION_STEP_SECONDS, DiarizationPipeline,
    FAST_SEGMENTATION_STEP_SECONDS, RuntimeConfig, SEGMENTATION_STEP_SECONDS,
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
            Self::Coreml => COREML_SEGMENTATION_STEP_SECONDS,
            Self::Cuda => CUDA_SEGMENTATION_STEP_SECONDS,
            Self::Cpu => SEGMENTATION_STEP_SECONDS,
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

pub fn run(
    mode: DiarizeMode,
    models_dir: Option<PathBuf>,
    chunk_emb_workers: usize,
    chunk_emb_compute_units: &str,
    wav_files: Vec<PathBuf>,
) -> Result<()> {
    let command_start = Instant::now();

    ensure!(!wav_files.is_empty(), "no WAV files specified");

    match mode {
        DiarizeMode::Pyannote(device) => {
            let wav_path = wav_files[0].to_string_lossy();
            let output = run_pyannote_sidecar(device.as_str(), &wav_path)?;
            print!("{output}");
        }
        DiarizeMode::Speakrs(speakrs_mode) => {
            let execution_mode = speakrs_mode.execution_mode();

            let compute_units = match chunk_emb_compute_units {
                "ane" | "cpu-and-neural-engine" => CoreMlComputeUnits::CpuAndNeuralEngine,
                _ => CoreMlComputeUnits::All,
            };
            let runtime_config = RuntimeConfig {
                chunk_emb_workers,
                #[cfg(feature = "coreml")]
                chunk_emb_compute_units: compute_units,
            };
            if chunk_emb_workers > 1 || compute_units != CoreMlComputeUnits::All {
                eprintln!(
                    "runtime config: workers={chunk_emb_workers} compute_units={chunk_emb_compute_units}"
                );
            }

            let models_dir_start = Instant::now();
            let models_dir = models_dir.unwrap_or_else(default_models_dir);
            let models_dir_elapsed = models_dir_start.elapsed();

            let step = speakrs_mode.step_seconds();
            let seg_model_start = Instant::now();
            let mut seg_model = SegmentationModel::with_mode(
                models_dir.join("segmentation-3.0.onnx").to_str().unwrap(),
                step as f32,
                execution_mode,
            )?;
            let seg_model_elapsed = seg_model_start.elapsed();

            let emb_model_start = Instant::now();
            let mut emb_model = EmbeddingModel::with_mode_and_config(
                models_dir
                    .join("wespeaker-voxceleb-resnet34.onnx")
                    .to_str()
                    .unwrap(),
                execution_mode,
                &runtime_config,
            )?;
            let emb_model_elapsed = emb_model_start.elapsed();

            let pipeline_start = Instant::now();
            let mut pipeline =
                DiarizationPipeline::new(&mut seg_model, &mut emb_model, &models_dir)?;
            let pipeline_elapsed = pipeline_start.elapsed();

            let loop_start = Instant::now();
            tracing::trace!(
                models_dir_ms = models_dir_elapsed.as_millis(),
                seg_model_ms = seg_model_elapsed.as_millis(),
                emb_model_ms = emb_model_elapsed.as_millis(),
                pipeline_ms = pipeline_elapsed.as_millis(),
                startup_ms = loop_start.duration_since(command_start).as_millis(),
                "Startup timing",
            );

            let total = wav_files.len();

            // load all audio files
            let load_start = Instant::now();
            let mut audio_data: Vec<(String, Vec<f32>)> = Vec::with_capacity(total);
            for wav_path in &wav_files {
                let file_id = wav_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "file1".to_string());
                let (samples, sr) = wav::load_wav_samples(&wav_path.to_string_lossy())?;
                ensure!(sr == 16000, "expected 16kHz WAV, got {sr}Hz");
                audio_data.push((file_id, samples));
            }
            tracing::trace!(
                load_ms = load_start.elapsed().as_millis(),
                files = total,
                "All audio loaded",
            );

            // build batch inputs
            let batch_inputs: Vec<speakrs::pipeline::BatchInput<'_>> = audio_data
                .iter()
                .map(|(file_id, samples)| speakrs::pipeline::BatchInput {
                    audio: samples,
                    file_id,
                })
                .collect();

            // run batch
            let batch_start = Instant::now();
            let results = pipeline.run_batch(&batch_inputs)?;
            let batch_elapsed = batch_start.elapsed();

            // output results in order
            for (i, result) in results.iter().enumerate() {
                let file_id = &audio_data[i].0;
                let audio_secs = audio_data[i].1.len() as f64 / 16_000.0;
                let per_file = batch_elapsed.as_secs_f64() / (i + 1) as f64;
                let eta = format_eta((total - i - 1) as f64 * per_file);
                let elapsed = format_eta(batch_elapsed.as_secs_f64());
                let now = chrono::Local::now().format("%H:%M:%S");
                eprintln!(
                    "  [{}/{}] {file_id}: {audio_secs:.0}s audio (elapsed {elapsed}, ETA {eta}) [{now}]",
                    i + 1,
                    total,
                );
                print!("{}", result.rttm(file_id));
            }

            eprintln!(
                "  Total: {:.1}s for {} files",
                batch_elapsed.as_secs_f64(),
                total,
            );

            tracing::trace!(
                startup_ms = loop_start.duration_since(command_start).as_millis(),
                loop_ms = loop_start.elapsed().as_millis(),
                total_ms = command_start.elapsed().as_millis(),
                "Command timing",
            );
        }
    }

    Ok(())
}

fn default_models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("fixtures/models")
        .to_path_buf()
}

fn run_pyannote_sidecar(device: &str, wav_path: &str) -> Result<String> {
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/pyannote-bench");
    let output = Command::new("uv")
        .arg("run")
        .arg("--project")
        .arg(&project_path)
        .arg("python")
        .arg(project_path.join("diarize.py"))
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
