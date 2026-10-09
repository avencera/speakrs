#[cfg(feature = "cpu")]
mod cpu;
use std::path::Path;
#[cfg(any(feature = "migraphx", feature = "coreml"))]
use std::path::PathBuf;

use ndarray::Array2;

#[cfg(feature = "coreml")]
use crate::inference::CoreMlError;
use crate::inference::{ExecutionMode, InferenceBackend, InferenceError, ModelLoadError};
use crate::pipeline::RuntimeConfig;

#[cfg(feature = "_cuda")]
mod cuda;
#[cfg(feature = "coreml")]
mod native;
#[cfg(feature = "migraphx")]
mod onnx;
#[cfg(feature = "coreml")]
mod parallel;
mod run;
mod tensor;

#[cfg(feature = "cpu")]
use cpu::CpuSegmentationBackend;
#[cfg(feature = "_cuda")]
use cuda::CudaSegmentationBackend;
#[cfg(feature = "coreml")]
use native::CoreMlSegmentation;
#[cfg(feature = "migraphx")]
use onnx::OrtSegmentation;
pub(crate) use tensor::{InvalidWindowGeometry, WindowSpec, segmentation_window_count};

/// Errors that can occur during segmentation inference
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SegmentationError {
    /// The inference backend failed
    #[error(transparent)]
    Inference(#[from] InferenceError),
    /// Streaming channel was closed before all windows were sent
    #[error("receiver disconnected")]
    Disconnected(#[from] crossbeam_channel::SendError<Array2<f32>>),
    /// Internal segmentation invariant was violated
    #[error("{context}: {message}")]
    Invariant {
        /// Which step failed
        context: &'static str,
        /// Invariant failure details
        message: String,
    },
    /// Model output was missing or had an unexpected shape
    #[error("{context}: {message}")]
    MalformedOutput {
        /// Which output extraction step failed
        context: &'static str,
        /// Output validation details
        message: String,
    },
    /// Background worker panicked
    #[error("{worker} thread panicked")]
    WorkerPanic {
        /// Worker or thread name
        worker: String,
    },
}

#[cfg(feature = "_ort")]
impl From<ort::Error> for SegmentationError {
    fn from(error: ort::Error) -> Self {
        Self::Inference(error.into())
    }
}

#[cfg(feature = "coreml")]
impl From<CoreMlError> for SegmentationError {
    fn from(error: CoreMlError) -> Self {
        Self::Inference(error.into())
    }
}

// seg models exported with EnumeratedShapes for batch 1-32 and b64
#[cfg(any(feature = "migraphx", feature = "coreml", feature = "_cuda"))]
const PRIMARY_BATCH_SIZE: usize = 32;
#[cfg(feature = "coreml")]
const LARGE_BATCH_SIZE: usize = 64;

/// Sliding-window segmentation model (pyannote segmentation-3.0)
pub struct SegmentationModel {
    mode: ExecutionMode,
    backend: SegmentationBackend,
    window_spec: WindowSpec,
    sample_rate: usize,
}

/// Sessions for the one runtime chosen from the execution mode at load time
enum SegmentationBackend {
    #[cfg(feature = "cpu")]
    Cpu(Box<CpuSegmentationBackend>),
    #[cfg(feature = "migraphx")]
    Ort(OrtSegmentation),
    #[cfg(feature = "coreml")]
    CoreMl(Box<CoreMlSegmentation>),
    // boxed because the CUDA backend carries its session and staging inline
    #[cfg(feature = "_cuda")]
    Cuda(Box<CudaSegmentationBackend>),
}

impl SegmentationModel {
    /// Load native segmentation-3.0 weights for CPU inference
    ///
    /// Requires the `cpu` feature
    pub fn new(model_path: impl AsRef<Path>, step_duration: f32) -> Result<Self, ModelLoadError> {
        Self::with_mode(model_path, step_duration, ExecutionMode::Cpu)
    }

    /// Load a segmentation-3.0 model with the requested execution mode
    ///
    /// `model_path` names the base `segmentation-3.0.onnx` file. CoreML modes load the
    /// compiled bundles next to it. CPU and CUDA modes load
    /// `segmentation-3.0.safetensors`; they do not read the ONNX file. CPU also accepts
    /// that native filename directly and rejects other model families
    pub fn with_mode(
        model_path: impl AsRef<Path>,
        step_duration: f32,
        mode: ExecutionMode,
    ) -> Result<Self, ModelLoadError> {
        Self::with_mode_and_config(model_path, step_duration, mode, &RuntimeConfig::default())
    }

    /// Load a segmentation-3.0 model with the requested execution mode and runtime config
    ///
    /// The runtime config selects the CUDA modes' precision, LSTM algorithm and CUDA
    /// graphs; other modes ignore it
    pub fn with_mode_and_config(
        model_path: impl AsRef<Path>,
        step_duration: f32,
        mode: ExecutionMode,
        #[cfg_attr(not(feature = "_cuda"), allow(unused_variables))] config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let backend = mode.backend()?;

        let model_path = model_path.as_ref();
        let sample_rate = 16000;
        let window_spec = WindowSpec::from_seconds(10.0, step_duration, sample_rate).map_err(
            |error: InvalidWindowGeometry| ModelLoadError::InvalidConfiguration {
                message: error.message,
            },
        )?;
        let window_samples = window_spec.window_samples();

        let backend = match backend {
            #[cfg(feature = "cpu")]
            InferenceBackend::Cpu => SegmentationBackend::Cpu(Box::new(
                CpuSegmentationBackend::load(model_path, window_samples)?,
            )),
            #[cfg(feature = "migraphx")]
            InferenceBackend::Ort(provider) => SegmentationBackend::Ort(OrtSegmentation::load(
                model_path,
                provider,
                window_samples,
            )?),
            #[cfg(feature = "coreml")]
            InferenceBackend::CoreMl => SegmentationBackend::CoreMl(Box::new(
                CoreMlSegmentation::load(model_path, mode, window_samples)?,
            )),
            #[cfg(feature = "_cuda")]
            InferenceBackend::Cuda => SegmentationBackend::Cuda(Box::new(
                CudaSegmentationBackend::load(model_path, mode, window_samples, config)?,
            )),
        };

        Ok(Self {
            mode,
            backend,
            window_spec,
            sample_rate,
        })
    }

    #[cfg_attr(not(feature = "coreml"), allow(dead_code))]
    pub(crate) fn window_count(&self, audio_samples: usize) -> usize {
        segmentation_window_count(audio_samples, self.window_spec())
    }

    /// Audio sample rate in Hz (16000)
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Number of audio samples per sliding window
    pub fn window_samples(&self) -> usize {
        self.window_spec.window_samples()
    }

    /// Number of audio samples the window advances each step
    pub fn step_samples(&self) -> usize {
        self.window_spec.step_samples()
    }

    pub(crate) fn window_spec(&self) -> WindowSpec {
        self.window_spec
    }

    /// Step size in seconds
    pub fn step_seconds(&self) -> f64 {
        self.window_spec.step_samples() as f64 / self.sample_rate as f64
    }

    /// Execution mode this model was loaded with
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Create a handle that can run on another thread and owns new scratch buffers
    ///
    /// ORT session weights and arenas are shared, and each inference call locks only the
    /// session that it uses. A CUDA handle gets its own stream and device copy of the
    /// weights, because CUDA state is used by one thread at a time
    #[cfg(all(
        any(feature = "cpu", feature = "migraphx", feature = "_cuda"),
        not(feature = "coreml")
    ))]
    pub(crate) fn clone_shared(&self) -> Result<Self, InferenceError> {
        let backend = match &self.backend {
            #[cfg(feature = "cpu")]
            SegmentationBackend::Cpu(backend) => SegmentationBackend::Cpu(backend.clone()),
            #[cfg(feature = "migraphx")]
            SegmentationBackend::Ort(backend) => {
                SegmentationBackend::Ort(backend.clone_shared(self.window_samples()))
            }
            #[cfg(feature = "_cuda")]
            SegmentationBackend::Cuda(backend) => {
                SegmentationBackend::Cuda(Box::new(backend.reload()?))
            }
        };

        Ok(Self {
            mode: self.mode,
            backend,
            window_spec: self.window_spec,
            sample_rate: self.sample_rate,
        })
    }

    #[cfg(feature = "coreml")]
    fn coreml_backend(&self) -> Option<&CoreMlSegmentation> {
        match &self.backend {
            SegmentationBackend::CoreMl(backend) => Some(backend),
            #[cfg(feature = "cpu")]
            SegmentationBackend::Cpu(_) => None,
            #[cfg(feature = "migraphx")]
            SegmentationBackend::Ort(_) => None,
            #[cfg(feature = "_cuda")]
            SegmentationBackend::Cuda(_) => None,
        }
    }
}

#[cfg(any(feature = "migraphx", feature = "coreml"))]
fn batched_model_path(model_path: &Path, batch_size: usize) -> Option<PathBuf> {
    let path = model_path;
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".onnx")?;
    Some(path.with_file_name(format!("{stem}-b{batch_size}.onnx")))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::SegmentationModel;
    use crate::inference::{ExecutionMode, ModelLoadError};

    #[test]
    fn with_mode_rejects_invalid_step_before_loading() {
        for step in [0.0, -0.5, f32::NAN, f32::INFINITY] {
            let error =
                match SegmentationModel::with_mode("/nonexistent.onnx", step, ExecutionMode::Cpu) {
                    Ok(_) => panic!("expected invalid configuration for step={step}"),
                    Err(error) => error,
                };
            match error {
                ModelLoadError::InvalidConfiguration { message } => {
                    assert!(
                        message.contains("step duration"),
                        "unexpected message for step={step}: {message}"
                    );
                }
                other => panic!("expected invalid configuration for step={step}, got {other}"),
            }
        }
    }
}
