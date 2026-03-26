#![cfg(feature = "coreml")]

use std::path::Path;
use std::sync::Arc;

use objc2_core_ml::MLComputeUnits;

use crate::inference::ExecutionMode;
use crate::inference::coreml::{CachedInputShape, CoreMlModel, GpuPrecision, SharedCoreMlModel};

use super::super::{
    ChunkEmbeddingSession, ChunkSessionSpec, EmbeddingModel, FBANK_FEATURES, MASK_FRAMES,
    fp32_coreml_path, split_fbank_batched_model_path, split_fbank_model_path,
    split_tail_model_path,
};

impl EmbeddingModel {
    pub(in crate::inference::embedding) fn load_native_tail(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> Option<CoreMlModel> {
        let compute_units = match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
                CoreMlModel::default_compute_units()
            }
            _ => return None,
        };
        let tail_onnx = split_tail_model_path(model_path, batch_size);
        let coreml_path = fp32_coreml_path(&tail_onnx);
        if !coreml_path.exists() {
            if batch_size == 1 {
                tracing::warn!(
                    path = %coreml_path.display(),
                    "Native CoreML tail model not found, falling back to ORT CPU",
                );
            }
            return None;
        }
        match CoreMlModel::load(&coreml_path, compute_units, "output", GpuPrecision::Low) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML tail: {e}");
                None
            }
        }
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
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let fbank_onnx = if batch_size == 1 {
            split_fbank_model_path(model_path)
        } else {
            split_fbank_batched_model_path(model_path)
        };
        let coreml_path = fp32_coreml_path(&fbank_onnx);
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!(batch_size, "Failed to load native CoreML fbank: {e}");
                None
            }
        }
    }

    pub(in crate::inference::embedding) fn has_native_fbank_model(
        model_path: &Path,
        mode: ExecutionMode,
        batch_size: usize,
    ) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
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
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let coreml_path = model_path.with_file_name("wespeaker-fbank-30s.mlmodelc");
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            MLComputeUnits::CPUAndNeuralEngine,
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => {
                tracing::info!("Loaded 30s fbank model (CPUAndNeuralEngine)");
                Some(model)
            }
            Err(e) => {
                tracing::warn!("Failed to load 30s fbank model: {e}");
                None
            }
        }
    }

    pub(in crate::inference::embedding) fn load_native_multi_mask(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Option<SharedCoreMlModel> {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
            return None;
        }
        let onnx_path = model_path.with_file_name("wespeaker-multimask-tail-b32.onnx");
        let coreml_path = fp32_coreml_path(&onnx_path);
        if !coreml_path.exists() {
            return None;
        }
        match SharedCoreMlModel::load(
            &coreml_path,
            CoreMlModel::default_compute_units(),
            "output",
            GpuPrecision::Low,
        ) {
            Ok(model) => Some(model),
            Err(e) => {
                tracing::warn!("Failed to load native CoreML multi-mask: {e}");
                None
            }
        }
    }

    pub(in crate::inference::embedding) fn has_native_multi_mask_model(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> bool {
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
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
        if !matches!(mode, ExecutionMode::CoreMl | ExecutionMode::CoreMlFast) {
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
