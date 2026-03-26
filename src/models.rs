use std::path::{Path, PathBuf};

#[cfg(feature = "online")]
use crate::inference::ExecutionMode;

const SEGMENTATION_ONNX: &str = "segmentation-3.0.onnx";
const EMBEDDING_ONNX: &str = "wespeaker-voxceleb-resnet34.onnx";

/// Resolved model paths for the speakrs pipeline
///
/// Captures the three root paths needed by [`SegmentationModel`], [`EmbeddingModel`],
/// and `PldaTransform`. Variant models (batched, CoreML, split) are derived
/// internally by each model constructor from the base ONNX path.
///
/// [`SegmentationModel`]: crate::inference::segmentation::SegmentationModel
/// [`EmbeddingModel`]: crate::inference::embedding::EmbeddingModel
#[derive(Debug, Clone)]
pub struct ModelBundle {
    segmentation_onnx: PathBuf,
    embedding_onnx: PathBuf,
    plda_dir: PathBuf,
}

impl ModelBundle {
    /// Resolve paths from a local directory containing all model files
    pub fn from_dir(models_dir: impl Into<PathBuf>) -> Self {
        let dir = models_dir.into();
        Self {
            segmentation_onnx: dir.join(SEGMENTATION_ONNX),
            embedding_onnx: dir.join(EMBEDDING_ONNX),
            plda_dir: dir,
        }
    }

    /// Download models from HuggingFace and resolve paths
    #[cfg(feature = "online")]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, hf_hub::api::sync::ApiError> {
        let manager = ModelManager::new()?;
        let dir = manager.ensure(mode)?;
        Ok(Self::from_dir(dir))
    }

    /// Base ONNX path for the segmentation model
    pub fn segmentation_path(&self) -> &Path {
        &self.segmentation_onnx
    }

    /// Base ONNX path for the embedding model
    pub fn embedding_path(&self) -> &Path {
        &self.embedding_onnx
    }

    /// Directory containing PLDA parameter files
    pub fn plda_dir(&self) -> &Path {
        &self.plda_dir
    }
}

#[cfg(feature = "online")]
const HF_REPO: &str = "avencera/speakrs-models";

/// Manages downloading and caching speakrs ONNX models from HuggingFace
#[cfg(feature = "online")]
pub struct ModelManager {
    repo: hf_hub::api::sync::ApiRepo,
}

#[cfg(feature = "online")]
impl ModelManager {
    /// Create a manager using the default HuggingFace cache directory
    pub fn new() -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(HF_REPO.to_string());
        Ok(Self { repo })
    }

    /// Create a manager with a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api =
            hf_hub::api::sync::ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        let repo = api.model(HF_REPO.to_string());
        Ok(Self { repo })
    }

    /// Download a single file, returns path to cached copy
    pub fn get(&self, filename: impl AsRef<str>) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        self.repo.get(filename.as_ref())
    }

    /// Ensure all files for a mode are downloaded, return base models dir
    pub fn ensure(&self, mode: ExecutionMode) -> Result<PathBuf, hf_hub::api::sync::ApiError> {
        let files = required_files(mode);
        for file in &files {
            self.repo.get(file)?;
        }
        // all files land in the same snapshot dir
        let first = self.repo.get(&files[0])?;
        let Some(parent) = first.parent() else {
            return Ok(first);
        };
        Ok(parent.to_path_buf())
    }
}

#[cfg(feature = "online")]
const PLDA_FILES: &[&str] = &[
    "plda_lda.npy",
    "plda_tr.npy",
    "plda_mu.npy",
    "plda_psi.npy",
    "plda_mean1.npy",
    "plda_mean2.npy",
    "wespeaker-voxceleb-resnet34.min_num_samples.txt",
];

#[cfg(feature = "online")]
const ONNX_FILES: &[&str] = &[
    "segmentation-3.0.onnx",
    "wespeaker-voxceleb-resnet34.onnx",
    "wespeaker-voxceleb-resnet34.onnx.data",
];

#[cfg(feature = "online")]
fn mlmodelc_files(name: &str) -> Vec<String> {
    vec![
        format!("{name}/model.mil"),
        format!("{name}/coremldata.bin"),
        format!("{name}/weights/weight.bin"),
        format!("{name}/analytics/coremldata.bin"),
    ]
}

#[cfg(feature = "online")]
fn required_files(mode: ExecutionMode) -> Vec<String> {
    let mut files: Vec<String> = PLDA_FILES.iter().map(|s| s.to_string()).collect();

    match mode {
        ExecutionMode::Cpu => {
            files.extend(ONNX_FILES.iter().map(|s| s.to_string()));
        }
        ExecutionMode::Cuda | ExecutionMode::CudaFast => {
            files.extend(ONNX_FILES.iter().map(|s| s.to_string()));
            // split models for multi-mask embedding (CPU fbank + GPU multi-mask)
            files.push("wespeaker-fbank.onnx".to_string());
            files.push("wespeaker-fbank-b32.onnx".to_string());
            files.push("wespeaker-multimask-tail.onnx".to_string());
            files.push("wespeaker-multimask-tail-b32.onnx".to_string());
            // batched seg/emb models
            files.push("segmentation-3.0-b32.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-b64.onnx".to_string());
        }
        ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
            // CoreML modes still need the ONNX segmentation model for the constructor
            files.push("segmentation-3.0.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34.onnx.data".to_string());
            // b32 batched ONNX for segmentation
            files.push("segmentation-3.0-b32.onnx".to_string());
            // split ONNX models for embedding
            files.push("wespeaker-fbank.onnx".to_string());
            files.push("wespeaker-fbank-b32.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail-b3.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail-b32.onnx".to_string());
            // FP32 CoreML models
            files.extend(mlmodelc_files("segmentation-3.0-b32.mlmodelc"));
            files.extend(mlmodelc_files("segmentation-3.0.mlmodelc"));
            files.extend(mlmodelc_files("wespeaker-fbank-b32.mlmodelc"));
            files.extend(mlmodelc_files("wespeaker-fbank.mlmodelc"));
            files.extend(mlmodelc_files(
                "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
            ));
            files.extend(mlmodelc_files(
                "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc",
            ));
            files.extend(mlmodelc_files("wespeaker-voxceleb-resnet34-tail.mlmodelc"));
        }
    }

    files
}
