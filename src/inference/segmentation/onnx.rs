use std::path::Path;
use std::time::Instant;

use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::TensorRef;

use crate::inference::{
    InferenceError, ModelLoadError, OrtProvider, SharedSession, ensure_ort_ready,
};

use super::tensor::{first_output, output_shape3};
use super::{PRIMARY_BATCH_SIZE, SegmentationError, batched_model_path};

/// ONNX Runtime segmentation sessions plus private input staging
pub(super) struct OrtSegmentation {
    session: SharedSession,
    primary_batched_session: Option<SharedSession>,
    input_buffer: Array3<f32>,
    primary_batch_input_buffer: Array3<f32>,
}

impl OrtSegmentation {
    pub(super) fn load(
        model_path: &Path,
        provider: OrtProvider,
        window_samples: usize,
    ) -> Result<Self, ModelLoadError> {
        ensure_ort_ready()?;

        let session_start = Instant::now();
        let session = SharedSession::new(build_session(model_path, provider)?);
        let session_elapsed = session_start.elapsed();

        let batched_start = Instant::now();
        let primary_batched_session = batched_model_path(model_path, PRIMARY_BATCH_SIZE)
            .filter(|path| path.exists())
            .map(|path| build_session(&path, provider).map(SharedSession::new))
            .transpose()?;
        let batched_elapsed = batched_start.elapsed();

        tracing::trace!(
            ort_single_ms = session_elapsed.as_millis(),
            ort_batched_ms = batched_elapsed.as_millis(),
            total_ms = (session_elapsed + batched_elapsed).as_millis(),
            "Segmentation model init",
        );

        Ok(Self::with_sessions(
            session,
            primary_batched_session,
            window_samples,
        ))
    }

    fn with_sessions(
        session: SharedSession,
        primary_batched_session: Option<SharedSession>,
        window_samples: usize,
    ) -> Self {
        Self {
            session,
            primary_batched_session,
            input_buffer: Array3::zeros((1, 1, window_samples)),
            primary_batch_input_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, window_samples)),
        }
    }

    /// Share sessions with a new handle that owns fresh scratch buffers
    #[cfg(not(feature = "coreml"))]
    pub(super) fn clone_shared(&self, window_samples: usize) -> Self {
        Self::with_sessions(
            self.session.clone(),
            self.primary_batched_session.clone(),
            window_samples,
        )
    }

    pub(super) fn has_batched(&self) -> bool {
        self.primary_batched_session.is_some()
    }

    pub(super) fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        self.input_buffer.fill(0.0);
        self.input_buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_tensor = TensorRef::from_array_view(self.input_buffer.view())?;

        let mut session = self.session.lock()?;
        let outputs = session.run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation window output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (_batch, frames, classes) = output_shape3(shape, "segmentation window output")?;

        Array2::from_shape_vec((frames, classes), data.to_vec()).map_err(|error| {
            SegmentationError::MalformedOutput {
                context: "segmentation window output",
                message: format!("invalid output shape: {error}"),
            }
        })
    }

    pub(super) fn run_batch(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        self.primary_batch_input_buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            self.primary_batch_input_buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_tensor = TensorRef::from_array_view(self.primary_batch_input_buffer.view())?;

        let mut session = self
            .primary_batched_session
            .as_ref()
            .ok_or(InferenceError::ModelUnavailable {
                model: "batched segmentation session",
            })?
            .lock()?;
        let outputs = session.run(ort::inputs![input_tensor])?;
        let output = first_output(outputs.values(), "segmentation batch output")?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        let (batch, frames, classes) = output_shape3(shape, "segmentation batch output")?;
        let stride = frames * classes;
        let expected_len =
            batch
                .checked_mul(stride)
                .ok_or_else(|| SegmentationError::MalformedOutput {
                    context: "segmentation batch output",
                    message: format!("output shape {shape} exceeded addressable memory"),
                })?;
        if data.len() != expected_len {
            return Err(SegmentationError::MalformedOutput {
                context: "segmentation batch output",
                message: format!(
                    "shape {shape} expected {expected_len} values, got {}",
                    data.len()
                ),
            });
        }

        (0..batch)
            .map(|batch_idx| {
                let start = batch_idx * stride;
                Array2::from_shape_vec((frames, classes), data[start..start + stride].to_vec())
                    .map_err(|error| SegmentationError::MalformedOutput {
                        context: "segmentation batch output",
                        message: format!("invalid output shape: {error}"),
                    })
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

fn build_session(model_path: &Path, provider: OrtProvider) -> Result<Session, ort::Error> {
    let builder = Session::builder()?
        .with_independent_thread_pool()?
        .with_intra_threads(available_threads().min(6))?
        .with_inter_threads(1)?
        .with_memory_pattern(true)?;
    let mut builder = provider.apply(builder)?;
    builder.commit_from_file(model_path)
}

fn available_threads() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
}
