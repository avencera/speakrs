#![cfg(feature = "coreml")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use ndarray::{Array2, Array3};
use tracing::info;

use crate::inference::coreml::{
    CachedInputShape, CoreMlModel, SharedCoreMlModel, coreml_model_path,
};
use crate::inference::{CoreMlSegmentationBatch, ExecutionMode, InferenceError, ModelLoadError};

use super::{LARGE_BATCH_SIZE, PRIMARY_BATCH_SIZE, SegmentationError, batched_model_path};

/// Compiled CoreML bundle paths derived from the base `segmentation-3.0.onnx` path
///
/// Only the file stem is used, so the ONNX file itself does not need to exist
struct SegmentationAssetPaths {
    single: PathBuf,
    batched: PathBuf,
    large_batched: PathBuf,
}

impl SegmentationAssetPaths {
    fn resolve(model_path: &Path, mode: ExecutionMode) -> Result<Self, ModelLoadError> {
        let batched_path = |batch_size: usize| {
            batched_model_path(model_path, batch_size)
                .map(|onnx_path| coreml_model_path(&onnx_path))
                .ok_or_else(|| ModelLoadError::MissingNativeAsset {
                    mode,
                    path: model_path.to_path_buf(),
                })
        };

        Ok(Self {
            single: coreml_model_path(model_path),
            batched: batched_path(PRIMARY_BATCH_SIZE)?,
            large_batched: batched_path(LARGE_BATCH_SIZE)?,
        })
    }

    /// Checks every bundle `batch` loads before loading any, so a missing one fails fast
    fn require(
        &self,
        batch: CoreMlSegmentationBatch,
        mode: ExecutionMode,
    ) -> Result<(), ModelLoadError> {
        let required: &[&Path] = match batch {
            CoreMlSegmentationBatch::Single => &[&self.single],
            CoreMlSegmentationBatch::Batch32 => &[&self.single, &self.batched],
            CoreMlSegmentationBatch::Batch64 => &[&self.single, &self.batched, &self.large_batched],
        };

        for path in required {
            require_native_asset(path, mode)?;
        }
        Ok(())
    }
}

fn require_native_asset(path: &Path, mode: ExecutionMode) -> Result<(), ModelLoadError> {
    if path.exists() {
        Ok(())
    } else {
        Err(ModelLoadError::MissingNativeAsset {
            mode,
            path: path.to_path_buf(),
        })
    }
}

fn load_native_model(
    coreml_path: &Path,
    mode: ExecutionMode,
    load_error_message: &str,
) -> Result<SharedCoreMlModel, ModelLoadError> {
    require_native_asset(coreml_path, mode)?;

    SharedCoreMlModel::load(coreml_path, CoreMlModel::default_compute_units(), "output").map_err(
        |err| ModelLoadError::NativeAssetLoad {
            mode,
            path: coreml_path.to_path_buf(),
            message: format!("{load_error_message}: {err}"),
        },
    )
}

/// Batch-32 model plus private input staging for the sequential path
///
/// The parallel path also uses it, for batch-32 runs and to warm-start batch-64 runs
struct PrimaryBatchModel {
    model: SharedCoreMlModel,
    cached_input_shape: CachedInputShape,
    input_buffer: Array3<f32>,
}

impl PrimaryBatchModel {
    fn load(
        path: &Path,
        mode: ExecutionMode,
        window_samples: usize,
    ) -> Result<Self, ModelLoadError> {
        Ok(Self {
            model: load_native_model(
                path,
                mode,
                "Failed to load native CoreML batched segmentation",
            )?,
            cached_input_shape: CachedInputShape::new(
                "input",
                &[PRIMARY_BATCH_SIZE, 1, window_samples],
            ),
            input_buffer: Array3::zeros((PRIMARY_BATCH_SIZE, 1, window_samples)),
        })
    }
}

/// Batched models loaded for the configured [`CoreMlSegmentationBatch`]
enum BatchedModels {
    /// [`CoreMlSegmentationBatch::Single`] loads no batched model
    None,
    Primary(PrimaryBatchModel),
    Large {
        primary: PrimaryBatchModel,
        large: SharedCoreMlModel,
    },
}

impl BatchedModels {
    fn load(
        paths: &SegmentationAssetPaths,
        mode: ExecutionMode,
        window_samples: usize,
        batch: CoreMlSegmentationBatch,
    ) -> Result<Self, ModelLoadError> {
        let load_primary = || PrimaryBatchModel::load(&paths.batched, mode, window_samples);

        match batch {
            CoreMlSegmentationBatch::Single => Ok(Self::None),
            CoreMlSegmentationBatch::Batch32 => Ok(Self::Primary(load_primary()?)),
            CoreMlSegmentationBatch::Batch64 => {
                let primary = load_primary()?;
                let large = load_native_model(
                    &paths.large_batched,
                    mode,
                    "Failed to load b64 segmentation",
                )?;
                info!("Loaded b64 segmentation model");
                Ok(Self::Large { primary, large })
            }
        }
    }

    fn primary_mut(&mut self) -> Option<&mut PrimaryBatchModel> {
        match self {
            Self::None => None,
            Self::Primary(primary) | Self::Large { primary, .. } => Some(primary),
        }
    }
}

/// Model the parallel streaming path runs, chosen from the recording length and batch limit
#[derive(Clone, Copy)]
pub(super) enum ParallelModel<'a> {
    Single(&'a SharedCoreMlModel),
    Batched {
        model: &'a SharedCoreMlModel,
        batch_size: usize,
        /// Batch-32 model that runs the leading windows of a batch-64 run
        warm_start: Option<&'a SharedCoreMlModel>,
    },
}

/// Native CoreML segmentation models plus private input staging
pub(super) struct CoreMlSegmentation {
    single: SharedCoreMlModel,
    batched: BatchedModels,
    cached_single_input_shape: CachedInputShape,
    input_buffer: Array3<f32>,
}

impl CoreMlSegmentation {
    pub(super) fn load(
        model_path: &Path,
        mode: ExecutionMode,
        window_samples: usize,
        batch: CoreMlSegmentationBatch,
    ) -> Result<Self, ModelLoadError> {
        let paths = SegmentationAssetPaths::resolve(model_path, mode)?;
        paths.require(batch, mode)?;

        let single_start = Instant::now();
        let single = load_native_model(
            &paths.single,
            mode,
            "Failed to load native CoreML segmentation",
        )?;
        let single_elapsed = single_start.elapsed();

        let batched_start = Instant::now();
        let batched = BatchedModels::load(&paths, mode, window_samples, batch)?;
        let batched_elapsed = batched_start.elapsed();

        tracing::trace!(
            max_batch = batch.size(),
            native_single_ms = single_elapsed.as_millis(),
            native_batched_ms = batched_elapsed.as_millis(),
            total_ms = (single_elapsed + batched_elapsed).as_millis(),
            "Segmentation model init",
        );

        Ok(Self {
            single,
            batched,
            cached_single_input_shape: CachedInputShape::new("input", &[1, 1, window_samples]),
            input_buffer: Array3::zeros((1, 1, window_samples)),
        })
    }

    /// Whether the sequential path can run batch-32 predictions
    pub(super) fn has_batched(&self) -> bool {
        !matches!(self.batched, BatchedModels::None)
    }

    /// Model for the parallel streaming path
    pub(super) fn select_parallel_model(&self, total_windows: usize) -> ParallelModel<'_> {
        let min_batch_windows = PRIMARY_BATCH_SIZE * 6;
        if total_windows < min_batch_windows {
            return ParallelModel::Single(&self.single);
        }

        match &self.batched {
            BatchedModels::None => ParallelModel::Single(&self.single),
            BatchedModels::Primary(primary) => ParallelModel::Batched {
                model: &primary.model,
                batch_size: PRIMARY_BATCH_SIZE,
                warm_start: None,
            },
            BatchedModels::Large { primary, large } => ParallelModel::Batched {
                model: large,
                batch_size: LARGE_BATCH_SIZE,
                warm_start: Some(&primary.model),
            },
        }
    }

    pub(super) fn run_window(&mut self, window: &[f32]) -> Result<Array2<f32>, SegmentationError> {
        let buffer = &mut self.input_buffer;
        buffer.fill(0.0);
        buffer
            .slice_mut(ndarray::s![0, 0, ..window.len()])
            .assign(&ndarray::ArrayView1::from(window));
        let input_data = buffer
            .as_slice()
            .ok_or(InferenceError::NonContiguousBuffer {
                context: "native segmentation single input",
            })?;

        let tensor = self
            .single
            .predict_cached(&[(&self.cached_single_input_shape, input_data)])?;
        let (data, frames, classes) = tensor
            .rank3_hw("native segmentation single output")
            .map_err(InferenceError::from)?;
        Array2::from_shape_vec((frames, classes), data).map_err(|source| {
            InferenceError::OutputArray {
                context: "native segmentation single output",
                source,
            }
            .into()
        })
    }

    pub(super) fn run_batch(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Array2<f32>>, SegmentationError> {
        let primary = self
            .batched
            .primary_mut()
            .ok_or_else(|| SegmentationError::Invariant {
                context: "native segmentation batch",
                message: "batched predictions need a batched model, which the single-window batch limit does not load".to_owned(),
            })?;
        let buffer = &mut primary.input_buffer;
        buffer.fill(0.0);
        for (batch_idx, window) in windows.iter().enumerate() {
            buffer
                .slice_mut(ndarray::s![batch_idx, 0, ..window.len()])
                .assign(&ndarray::ArrayView1::from(*window));
        }
        let input_data = buffer
            .as_slice()
            .ok_or(InferenceError::NonContiguousBuffer {
                context: "native segmentation batch input",
            })?;

        let tensor = primary
            .model
            .predict_cached(&[(&primary.cached_input_shape, input_data)])?;
        let (batch, frames, classes) = tensor
            .try_rank3("native segmentation batch output")
            .map_err(InferenceError::from)?;
        let data = tensor.into_data();

        (0..batch)
            .map(|batch_idx| {
                let start = batch_idx * frames * classes;
                let end = start + frames * classes;
                Array2::from_shape_vec((frames, classes), data[start..end].to_vec()).map_err(
                    |source| {
                        InferenceError::OutputArray {
                            context: "native segmentation batch output",
                            source,
                        }
                        .into()
                    },
                )
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(prefix: &str) -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("speakrs-{prefix}-{unique}"));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn load_native_coreml_errors_when_compiled_bundle_is_invalid() {
        let dir = TestDir::new("seg-coreml-invalid");
        // the ONNX file is never read in CoreML modes, so it is absent here
        let model_path = dir.path().join("segmentation-3.0.onnx");

        let compiled_path = dir.path().join("segmentation-3.0.mlmodelc");
        fs::create_dir_all(compiled_path.join("weights")).unwrap();
        fs::create_dir_all(compiled_path.join("analytics")).unwrap();
        fs::write(compiled_path.join("model.mil"), b"invalid").unwrap();
        fs::write(compiled_path.join("coremldata.bin"), b"invalid").unwrap();
        fs::write(compiled_path.join("weights/weight.bin"), b"invalid").unwrap();
        fs::write(compiled_path.join("analytics/coremldata.bin"), b"invalid").unwrap();

        let paths = SegmentationAssetPaths::resolve(&model_path, ExecutionMode::CoreMl).unwrap();
        assert_eq!(paths.single, compiled_path);
        let error = match load_native_model(&paths.single, ExecutionMode::CoreMl, "test load") {
            Ok(_) => panic!("invalid compiled bundle should error"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            ModelLoadError::NativeAssetLoad {
                mode: ExecutionMode::CoreMl,
                ..
            }
        ));
    }

    #[test]
    fn load_reports_missing_native_bundle_without_onnx_file() {
        let dir = TestDir::new("seg-coreml-missing");
        let model_path = dir.path().join("segmentation-3.0.onnx");

        let error = match CoreMlSegmentation::load(
            &model_path,
            ExecutionMode::CoreMl,
            160_000,
            CoreMlSegmentationBatch::default(),
        ) {
            Ok(_) => panic!("missing bundles should error"),
            Err(error) => error,
        };

        match error {
            ModelLoadError::MissingNativeAsset { mode, path } => {
                assert_eq!(mode, ExecutionMode::CoreMl);
                assert_eq!(path, dir.path().join("segmentation-3.0.mlmodelc"));
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
