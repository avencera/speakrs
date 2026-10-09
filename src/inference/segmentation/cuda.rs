use std::path::{Path, PathBuf};

use ndarray::Array2;
use tracing::debug;

use crate::inference::cuda::{
    CudaError, CudaRuntime, CudaSegmentation, CudaSession, SafetensorsFile, SegmentationOptions,
};
use crate::inference::{ExecutionMode, InferenceError, ModelLoadError};
use crate::pipeline::RuntimeConfig;

use super::SegmentationError;
use crate::inference::cuda::implementation::policy::RecipeMode;

/// Native CUDA segmentation plus private input staging for one model handle
pub(super) struct CudaSegmentationBackend {
    session: CudaSession<SegmentationState>,
    // the reload inputs are only read by `clone_shared`, which CoreML builds do not have
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    weights: PathBuf,
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    options: SegmentationOptions,
    #[cfg_attr(feature = "coreml", allow(dead_code))]
    recipe_mode: RecipeMode,
    window_samples: usize,
    /// `[batch, window_samples]` zero-padded windows for the next upload
    staging: Vec<f32>,
}

/// The device model; a session keeps it with the runtime it was built on
struct SegmentationState(CudaSegmentation);

// SAFETY: `SegmentationState` is not `Send` only because of cuDNN state: the
// convolution and RNN descriptors and cudarc's cuDNN handle they share, and the
// captured CUDA graphs. All of it was created on the session's runtime and is reachable
// only through this session, which is neither `Clone` nor `Sync` and hands out no
// references to it, so moving the session moves every owner of that state at once and
// no two threads ever use it concurrently. cuDNN handles and descriptors, and CUDA
// graphs, may be used from any host thread as long as calls are not concurrent, and
// `CudaSession::run` and its `Drop` make the runtime's context current on the thread
// that uses or frees them
unsafe impl Send for CudaSession<SegmentationState> {}

impl CudaSegmentationBackend {
    /// Loads `segmentation-3.0.safetensors` next to the base ONNX path
    pub(super) fn load(
        model_path: &Path,
        mode: ExecutionMode,
        window_samples: usize,
        config: &RuntimeConfig,
    ) -> Result<Self, ModelLoadError> {
        let weights = model_path.with_extension("safetensors");
        if !weights.is_file() {
            return Err(ModelLoadError::MissingCudaWeights {
                mode,
                path: weights,
            });
        }

        let options = SegmentationOptions {
            math: config.cuda_segmentation_math,
            #[cfg(feature = "_cuda-libraries")]
            lstm_algo: config.cuda_lstm_algorithm,
            cuda_graph: config.cuda_graphs.enabled(),
        };
        let recipe_mode =
            RecipeMode::new(config.cuda_segmentation_math, config.cuda_embedding_math);
        let (session, options) = open_session(&weights, options, window_samples, recipe_mode)?;

        Ok(Self {
            session,
            weights,
            options,
            recipe_mode,
            window_samples,
            staging: Vec::new(),
        })
    }

    /// A handle with its own runtime, stream and device copy of the same weights
    #[cfg(not(feature = "coreml"))]
    pub(super) fn reload(&self) -> Result<Self, InferenceError> {
        let (session, options) = open_session(
            &self.weights,
            self.options,
            self.window_samples,
            self.recipe_mode,
        )?;
        Ok(Self {
            session,
            weights: self.weights.clone(),
            options,
            recipe_mode: self.recipe_mode,
            window_samples: self.window_samples,
            staging: Vec::new(),
        })
    }

    pub(super) fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        let mut outputs = self.run_batch(&[window])?;
        outputs.pop().ok_or(SegmentationError::MalformedOutput {
            context: "cuda segmentation window output",
            message: "no output for the window".to_owned(),
        })
    }

    /// Runs one batch of up to `window_samples`-long windows, zero padding shorter ones
    pub(super) fn run_batch(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let samples = self.window_samples;
        self.staging.clear();
        self.staging.resize(windows.len() * samples, 0.0);
        for (row, window) in self.staging.chunks_exact_mut(samples).zip(windows) {
            let len = window.len().min(samples);
            row[..len].copy_from_slice(&window[..len]);
        }

        let batch = windows.len();
        let staging = &self.staging;
        let output = self
            .session
            .run(|runtime, state| state.0.run(runtime, batch, staging))
            .map_err(InferenceError::from)?;
        logit_rows(output, batch)
    }

    /// [`Self::run_batch`] for windows cut out of one stretch of audio: row `i` is
    /// `span[starts[i]..]`, clipped to the window and zero padded
    pub(super) fn run_span(
        &mut self,
        span: &[f32],
        starts: &[usize],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let samples = self.window_samples;
        let output = self
            .session
            .run(|runtime, state| state.0.run_span(runtime, samples, span, starts))
            .map_err(InferenceError::from)?;
        logit_rows(output, starts.len())
    }
}

/// Splits `[batch, frames, 7]` logits into one `[frames, 7]` array per window
fn logit_rows(output: Vec<f32>, batch: usize) -> Result<Vec<Array2<f32>>, SegmentationError> {
    let stride = output.len().checked_div(batch).unwrap_or(0);
    let frames = stride / CudaSegmentation::CLASSES;
    output
        .chunks_exact(stride.max(1))
        .map(|row| {
            Array2::from_shape_vec((frames, CudaSegmentation::CLASSES), row.to_vec()).map_err(
                |error| SegmentationError::MalformedOutput {
                    context: "cuda segmentation batch output",
                    message: format!("invalid output shape: {error}"),
                },
            )
        })
        .collect()
}

/// Opens the driver runtime and constructs only selected model state
fn open_session(
    weights: &Path,
    options: SegmentationOptions,
    window_samples: usize,
    recipe_mode: RecipeMode,
) -> Result<(CudaSession<SegmentationState>, SegmentationOptions), CudaError> {
    let file = SafetensorsFile::open(weights)?;
    let session = CudaSession::new(
        CudaRuntime::new(0)?.with_recipe_mode(recipe_mode),
        |runtime| {
            let mut model = CudaSegmentation::new(runtime, &file, options)?;
            // construct production classes before load returns, outside graph capture
            for batch in [1, 32] {
                model.workspace(runtime, batch, window_samples)?;
            }
            debug!(capability = %runtime.compute_capability(), ptx_tier = %model.kernel_tier(), ?options, "Loaded CUDA segmentation");
            Ok(SegmentationState(model))
        },
    )?;
    Ok((session, options))
}
