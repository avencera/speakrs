#![cfg(feature = "coreml")]

use std::path::{Path, PathBuf};

use ndarray::Array2;
use objc2_core_ml::MLComputeUnits;
use tracing::info;

use crate::inference::ExecutionMode;
use crate::inference::coreml::{
    CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel, coreml_model_path,
    coreml_w8a16_model_path,
};

use super::{LARGE_BATCH_SIZE, PRIMARY_BATCH_SIZE, SegmentationModel, batched_model_path};

impl SegmentationModel {
    pub(super) fn select_parallel_native_model(
        &self,
        total_windows: usize,
    ) -> Option<(&SharedCoreMlModel, usize)> {
        let min_batch_windows = PRIMARY_BATCH_SIZE * 6;
        if total_windows < min_batch_windows {
            return self.native_session.as_ref().map(|model| (model, 1));
        }

        self.native_large_batched_session
            .as_ref()
            .map(|model| (model, LARGE_BATCH_SIZE))
            .or_else(|| {
                self.native_batched_session
                    .as_ref()
                    .map(|model| (model, PRIMARY_BATCH_SIZE))
            })
            .or_else(|| self.native_session.as_ref().map(|model| (model, 1)))
    }

    fn resolve_coreml_path(model_path: &Path, mode: ExecutionMode) -> Option<PathBuf> {
        match mode {
            ExecutionMode::CoreMlFast => Some(coreml_w8a16_model_path(model_path)),
            ExecutionMode::CoreMl => Some(coreml_model_path(model_path)),
            _ => None,
        }
    }

    fn compute_units_for_mode(_mode: ExecutionMode) -> MLComputeUnits {
        CoreMlModel::default_compute_units()
    }

    fn resolve_batched_coreml_path(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<PathBuf> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }

        let batched_onnx = batched_model_path(model_path, batch_size)?;
        Self::resolve_coreml_path(&batched_onnx, mode)
    }

    fn load_native_coreml_model(
        coreml_path: &Path,
        mode: ExecutionMode,
        missing_message: &str,
        load_error_message: &str,
    ) -> Option<SharedCoreMlModel> {
        if !coreml_path.exists() {
            if !missing_message.is_empty() {
                tracing::warn!(path = %coreml_path.display(), "{missing_message}");
            }
            return None;
        }

        match SharedCoreMlModel::load(
            coreml_path,
            Self::compute_units_for_mode(mode),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(err) => {
                tracing::warn!("{load_error_message}: {err}");
                None
            }
        }
    }

    pub(super) fn load_native_coreml(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_coreml_path(model_path, mode)?;
        Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "Native CoreML segmentation model not found, falling back to ORT CPU",
            "Failed to load native CoreML segmentation",
        )
    }

    pub(super) fn load_native_coreml_batched(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_batched_coreml_path(model_path, mode, PRIMARY_BATCH_SIZE)?;
        Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "",
            "Failed to load native CoreML batched segmentation",
        )
    }

    pub(super) fn load_native_coreml_large_batched(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        let coreml_path = Self::resolve_batched_coreml_path(model_path, mode, LARGE_BATCH_SIZE)?;
        let model = Self::load_native_coreml_model(
            &coreml_path,
            mode,
            "",
            "Failed to load b64 segmentation",
        )?;
        info!("Loaded b64 segmentation model");
        Some(model)
    }

    pub(super) fn run_native_single(
        native: &SharedCoreMlModel,
        window: &[f32],
        buffer: &mut ndarray::Array3<f32>,
        cached_shape: &CachedInputShape,
    ) -> Result<Array2<f32>, ort::Error> {
        buffer.fill(0.0);
        buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_data = buffer.as_slice().ok_or_else(|| {
            ort::Error::new("native segmentation single input was not contiguous")
        })?;

        let (data, out_shape) = native
            .predict_cached(&[(cached_shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let frames = out_shape[1];
        let classes = out_shape[2];
        Array2::from_shape_vec((frames, classes), data).map_err(|error| {
            ort::Error::new(format!("native segmentation single output shape: {error}"))
        })
    }

    pub(super) fn run_native_batch(
        native: &SharedCoreMlModel,
        windows: &[&[f32]],
        buffer: &mut ndarray::Array3<f32>,
        cached_shape: &CachedInputShape,
    ) -> Result<Vec<Array2<f32>>, ort::Error> {
        buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_data = buffer
            .as_slice()
            .ok_or_else(|| ort::Error::new("native segmentation batch input was not contiguous"))?;

        let (data, out_shape) = native
            .predict_cached(&[(cached_shape, input_data)])
            .map_err(|e| ort::Error::new(e.to_string()))?;

        let batch = out_shape[0];
        let frames = out_shape[1];
        let classes = out_shape[2];

        (0..batch)
            .map(|batch_idx| {
                let start = batch_idx * frames * classes;
                let end = start + frames * classes;
                Array2::from_shape_vec((frames, classes), data[start..end].to_vec()).map_err(
                    |error| {
                        ort::Error::new(format!("native segmentation batch output shape: {error}"))
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()
    }
}
