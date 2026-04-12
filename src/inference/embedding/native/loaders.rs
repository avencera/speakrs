#![cfg(feature = "coreml")]

use std::path::Path;
use std::sync::Arc;

use objc2_core_ml::MLComputeUnits;

use crate::inference::coreml::{CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel};
use crate::inference::{ExecutionMode, ModelLoadError};

use super::super::{
    CHUNK_SPEAKER_BATCH_SIZE, ChunkEmbeddingSession, ChunkSessionSpec, EmbeddingModel,
    FBANK_FEATURES, MASK_FRAMES, fp32_coreml_path, split_fbank_batched_model_path,
    split_fbank_model_path, split_tail_model_path,
};

fn load_shared_or_warn(
    path: &Path,
    mode: ExecutionMode,
    compute_units: MLComputeUnits,
    error_context: &str,
) -> Result<SharedCoreMlModel, ModelLoadError> {
    EmbeddingModel::require_native_asset(path.to_path_buf(), mode)?;
    SharedCoreMlModel::load(path, compute_units, "output", GpuPrecision::Low).map_err(|error| {
        ModelLoadError::NativeAssetLoad {
            mode,
            path: path.to_path_buf(),
            message: format!("{error_context}: {error}"),
        }
    })
}

impl EmbeddingModel {
    fn require_native_asset(
        path: std::path::PathBuf,
        mode: ExecutionMode,
    ) -> Result<(), ModelLoadError> {
        if path.exists() {
            Ok(())
        } else {
            Err(ModelLoadError::MissingNativeAsset { mode, path })
        }
    }

    pub(in crate::inference::embedding) fn validate_native_coreml_assets(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<(), ModelLoadError> {
        if !mode.is_coreml() {
            return Ok(());
        }

        Self::require_native_asset(fp32_coreml_path(&split_fbank_model_path(model_path)), mode)?;
        Self::require_native_asset(
            fp32_coreml_path(&split_fbank_batched_model_path(model_path)),
            mode,
        )?;
        Self::require_native_asset(
            model_path.with_file_name("wespeaker-fbank-30s.mlmodelc"),
            mode,
        )?;
        Self::require_native_asset(
            fp32_coreml_path(&split_tail_model_path(model_path, 1)),
            mode,
        )?;
        Self::require_native_asset(
            fp32_coreml_path(&split_tail_model_path(model_path, CHUNK_SPEAKER_BATCH_SIZE)),
            mode,
        )?;
        Self::require_native_asset(
            fp32_coreml_path(&model_path.with_file_name("wespeaker-multimask-tail-b32.onnx")),
            mode,
        )?;

        for (step_resnet, num_windows, _, _) in Self::chunk_session_config(mode) {
            let stem = format!("wespeaker-chunk-emb-s{step_resnet}-w{num_windows}.mlmodelc");
            Self::require_native_asset(model_path.with_file_name(stem), mode)?;
        }

        Ok(())
    }

    pub(in crate::inference::embedding) fn load_native_tail(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Result<Option<CoreMlModel>, ModelLoadError> {
        let compute_units = match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
                CoreMlModel::default_compute_units()
            }
            _ => return Ok(None),
        };
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        let coreml_path = fp32_coreml_path(&tail_onnx);
        Self::require_native_asset(coreml_path.clone(), mode)?;
        let model = CoreMlModel::load(&coreml_path, compute_units, "output", GpuPrecision::Low)
            .map_err(|error| ModelLoadError::NativeAssetLoad {
                mode,
                path: coreml_path,
                message: format!(
                    "Failed to load native CoreML tail (batch_size={batch_size}): {error}"
                ),
            })?;
        Ok(Some(model))
    }

    pub(in crate::inference::embedding) fn has_native_tail_model(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> bool {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {}
            _ => return false,
        }
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        fp32_coreml_path(&tail_onnx).exists()
    }

    pub(in crate::inference::embedding) fn load_native_fbank(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Result<Option<SharedCoreMlModel>, ModelLoadError> {
        if !mode.is_coreml() {
            return Ok(None);
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        let coreml_path = fp32_coreml_path(&fbank_onnx);
        load_shared_or_warn(
            &coreml_path,
            mode,
            CoreMlModel::default_compute_units(),
            &format!("Failed to load native CoreML fbank (batch_size={batch_size})"),
        )
        .map(Some)
    }

    pub(in crate::inference::embedding) fn has_native_fbank_model(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> bool {
        if !mode.is_coreml() {
            return false;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        fp32_coreml_path(&fbank_onnx).exists()
    }

    pub(in crate::inference::embedding) fn load_native_fbank_30s(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Option<SharedCoreMlModel>, ModelLoadError> {
        if !mode.is_coreml() {
            return Ok(None);
        }
        let coreml_path = model_path.with_file_name("wespeaker-fbank-30s.mlmodelc");
        let model = load_shared_or_warn(
            &coreml_path,
            mode,
            MLComputeUnits::CPUAndNeuralEngine,
            "Failed to load 30s fbank model",
        )?;
        tracing::info!("Loaded 30s fbank model (CPUAndNeuralEngine)");
        Ok(Some(model))
    }

    pub(in crate::inference::embedding) fn load_native_multi_mask(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Option<SharedCoreMlModel>, ModelLoadError> {
        if !mode.is_coreml() {
            return Ok(None);
        }
        let onnx_path = model_path.with_file_name("wespeaker-multimask-tail-b32.onnx");
        let coreml_path = fp32_coreml_path(&onnx_path);
        load_shared_or_warn(
            &coreml_path,
            mode,
            CoreMlModel::default_compute_units(),
            "Failed to load native CoreML multi-mask",
        )
        .map(Some)
    }

    pub(in crate::inference::embedding) fn has_native_multi_mask_model(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> bool {
        if !mode.is_coreml() {
            return false;
        }
        let onnx_path = model_path.with_file_name("wespeaker-multimask-tail-b32.onnx");
        fp32_coreml_path(&onnx_path).exists()
    }

    fn chunk_session_config(mode: ExecutionMode) -> &'static [(usize, usize, usize, usize)] {
        match mode {
            ExecutionMode::CoreMlFast => &[
                (25, 11, 3000, 33),
                (25, 16, 4000, 48),
                (25, 21, 5000, 63),
                (25, 26, 6000, 78),
                (25, 36, 8000, 108),
                (25, 46, 10000, 138),
                (25, 56, 12000, 168),
            ],
            _ => &[
                (12, 22, 3016, 66),
                (12, 37, 4456, 111),
                (12, 53, 5992, 159),
                (12, 84, 8968, 252),
                (12, 116, 12040, 348),
            ],
        }
    }

    pub(in crate::inference::embedding) fn chunk_session_specs(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Vec<ChunkSessionSpec> {
        if !mode.is_coreml() {
            return Vec::new();
        }

        Self::chunk_session_config(mode)
            .iter()
            .filter_map(|&(step_resnet, num_windows, fbank_frames, num_masks)| {
                let stem = format!("wespeaker-chunk-emb-s{step_resnet}-w{num_windows}");
                let w8a16_path = model_path.with_file_name(format!("{stem}-w8a16.mlmodelc"));
                let fp32_path = model_path.with_file_name(format!("{stem}.mlmodelc"));

                let coreml_path = if fp32_path.exists() {
                    fp32_path
                } else if w8a16_path.exists() {
                    w8a16_path
                } else {
                    return None;
                };

                Some(ChunkSessionSpec {
                    coreml_path,
                    num_windows,
                    fbank_frames,
                    num_masks,
                })
            })
            .collect()
    }

    pub(in crate::inference::embedding) fn load_chunk_session(
        spec: &ChunkSessionSpec,
        compute_units: MLComputeUnits,
    ) -> Result<ChunkEmbeddingSession, crate::inference::coreml::CoreMlError> {
        let model = SharedCoreMlModel::load(
            &spec.coreml_path,
            compute_units,
            "output",
            GpuPrecision::Low,
        )?;
        Ok(ChunkEmbeddingSession {
            model: Arc::new(model),
            num_windows: spec.num_windows,
            fbank_frames: spec.fbank_frames,
            num_masks: spec.num_masks,
            cached_fbank_shape: Arc::new(CachedInputShape::new(
                "fbank",
                &[1, spec.fbank_frames, FBANK_FEATURES],
            )),
            cached_masks_shape: Arc::new(CachedInputShape::new(
                "masks",
                &[spec.num_masks, MASK_FRAMES],
            )),
        })
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

        fn write_invalid_mlmodelc(&self, name: &str) {
            let bundle = self.path().join(name);
            fs::create_dir_all(bundle.join("weights")).unwrap();
            fs::create_dir_all(bundle.join("analytics")).unwrap();
            fs::write(bundle.join("model.mil"), b"invalid").unwrap();
            fs::write(bundle.join("coremldata.bin"), b"invalid").unwrap();
            fs::write(bundle.join("weights/weight.bin"), b"invalid").unwrap();
            fs::write(bundle.join("analytics/coremldata.bin"), b"invalid").unwrap();
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn load_native_fbank_errors_when_bundle_is_invalid() {
        let dir = TestDir::new("emb-fbank-invalid");
        let model_path = dir.path().join("wespeaker-voxceleb-resnet34.onnx");
        fs::write(&model_path, b"placeholder").unwrap();
        dir.write_invalid_mlmodelc("wespeaker-fbank.mlmodelc");

        let error = match EmbeddingModel::load_native_fbank(&model_path, ExecutionMode::CoreMl, 1) {
            Ok(_) => panic!("invalid fbank bundle should error"),
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
    fn load_native_tail_errors_when_bundle_is_invalid() {
        let dir = TestDir::new("emb-tail-invalid");
        let model_path = dir.path().join("wespeaker-voxceleb-resnet34.onnx");
        fs::write(&model_path, b"placeholder").unwrap();
        dir.write_invalid_mlmodelc("wespeaker-voxceleb-resnet34-tail.mlmodelc");

        let error = match EmbeddingModel::load_native_tail(&model_path, ExecutionMode::CoreMl, 1) {
            Ok(_) => panic!("invalid tail bundle should error"),
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
}
