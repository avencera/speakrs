use std::path::{Path, PathBuf};

#[cfg(feature = "online")]
use crate::inference::ExecutionMode;
#[cfg(feature = "online")]
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
#[cfg(feature = "online")]
use hf_hub::{Repo, RepoType};

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
    #[cfg_attr(docsrs, doc(cfg(feature = "online")))]
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
#[cfg(feature = "online")]
const HF_REVISION: &str = "a785ebdbe6313868088c36c93d9efa71c470bd34";

#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelRepositoryId(&'static str);

#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ModelRevision(&'static str);

#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PinnedModelRepository {
    id: ModelRepositoryId,
    revision: ModelRevision,
}

#[cfg(feature = "online")]
impl PinnedModelRepository {
    const fn speakrs_models() -> Self {
        Self {
            id: ModelRepositoryId(HF_REPO),
            revision: ModelRevision(HF_REVISION),
        }
    }

    fn into_hf_repo(self) -> Repo {
        Repo::with_revision(
            self.id.0.to_owned(),
            RepoType::Model,
            self.revision.0.to_owned(),
        )
    }

    fn bind(self, api: &Api) -> ApiRepo {
        api.repo(self.into_hf_repo())
    }
}

#[cfg(feature = "online")]
const PINNED_MODEL_REPOSITORY: PinnedModelRepository = PinnedModelRepository::speakrs_models();

/// Manages downloading and caching speakrs ONNX models from HuggingFace
#[cfg(feature = "online")]
#[cfg_attr(docsrs, doc(cfg(feature = "online")))]
pub struct ModelManager {
    repo: ApiRepo,
}

#[cfg(feature = "online")]
impl ModelManager {
    /// Create a manager using the default HuggingFace cache directory
    pub fn new() -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = Api::new()?;
        Ok(Self::from_api(api))
    }

    /// Create a manager with a custom cache directory
    pub fn with_cache_dir(cache_dir: PathBuf) -> Result<Self, hf_hub::api::sync::ApiError> {
        let api = ApiBuilder::from_cache(hf_hub::Cache::new(cache_dir)).build()?;
        Ok(Self::from_api(api))
    }

    fn from_api(api: Api) -> Self {
        let repo = PINNED_MODEL_REPOSITORY.bind(&api);
        Self { repo }
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
fn extend_mlmodelc_files(files: &mut Vec<String>, names: &[&str]) {
    for name in names {
        files.extend(mlmodelc_files(name));
    }
}

#[cfg(feature = "online")]
const COREML_COMMON_MODEL_STEMS: &[&str] = &[
    "segmentation-3.0.mlmodelc",
    "segmentation-3.0-b32.mlmodelc",
    "segmentation-3.0-b64.mlmodelc",
    "wespeaker-fbank.mlmodelc",
    "wespeaker-fbank-b32.mlmodelc",
    "wespeaker-fbank-30s.mlmodelc",
    "wespeaker-multimask-tail-b32.mlmodelc",
    "wespeaker-voxceleb-resnet34-tail.mlmodelc",
    "wespeaker-voxceleb-resnet34-tail-b3.mlmodelc",
    "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc",
];

#[cfg(feature = "online")]
const COREML_CHUNK_MODEL_STEMS: &[&str] = &[
    "wespeaker-chunk-emb-p1s-w21.mlmodelc",
    "wespeaker-chunk-emb-p1s-w36.mlmodelc",
    "wespeaker-chunk-emb-p1s-w51.mlmodelc",
    "wespeaker-chunk-emb-p1s-w81.mlmodelc",
    "wespeaker-chunk-emb-p1s-w111.mlmodelc",
];

#[cfg(feature = "online")]
const COREML_FAST_SEGMENTATION_MODEL_STEMS: &[&str] = &[
    "segmentation-3.0-w8a16.mlmodelc",
    "segmentation-3.0-b32-w8a16.mlmodelc",
    "segmentation-3.0-b64-w8a16.mlmodelc",
];

#[cfg(feature = "online")]
const COREML_FAST_CHUNK_MODEL_STEMS: &[&str] = &[
    "wespeaker-chunk-emb-s25-w11.mlmodelc",
    "wespeaker-chunk-emb-s25-w16.mlmodelc",
    "wespeaker-chunk-emb-s25-w21.mlmodelc",
    "wespeaker-chunk-emb-s25-w26.mlmodelc",
    "wespeaker-chunk-emb-s25-w36.mlmodelc",
    "wespeaker-chunk-emb-s25-w46.mlmodelc",
    "wespeaker-chunk-emb-s25-w56.mlmodelc",
];

#[cfg(feature = "online")]
fn required_files(mode: ExecutionMode) -> Vec<String> {
    let mut files: Vec<String> = PLDA_FILES.iter().map(|s| s.to_string()).collect();

    match mode {
        ExecutionMode::Cpu => {
            files.extend(ONNX_FILES.iter().map(|s| s.to_string()));
        }
        ExecutionMode::Cuda | ExecutionMode::CudaFast | ExecutionMode::MiGraphX => {
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
        ExecutionMode::CoreMl => {
            // native CoreML modes still need the ONNX segmentation model for the constructor
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
            extend_mlmodelc_files(&mut files, COREML_COMMON_MODEL_STEMS);
            extend_mlmodelc_files(&mut files, COREML_CHUNK_MODEL_STEMS);
        }
        ExecutionMode::CoreMlFast => {
            files.push("segmentation-3.0.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34.onnx.data".to_string());
            files.push("segmentation-3.0-b32.onnx".to_string());
            files.push("wespeaker-fbank.onnx".to_string());
            files.push("wespeaker-fbank-b32.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail-b3.onnx".to_string());
            files.push("wespeaker-voxceleb-resnet34-tail-b32.onnx".to_string());
            extend_mlmodelc_files(&mut files, COREML_COMMON_MODEL_STEMS);
            extend_mlmodelc_files(&mut files, COREML_FAST_SEGMENTATION_MODEL_STEMS);
            extend_mlmodelc_files(&mut files, COREML_FAST_CHUNK_MODEL_STEMS);
        }
    }

    files
}

#[cfg(all(test, feature = "online"))]
mod tests {
    use super::*;

    const MODEL_FILENAME: &str = "segmentation-3.0.onnx";
    const EXPECTED_MODEL_URL: &str = "https://huggingface.co/avencera/speakrs-models/resolve/a785ebdbe6313868088c36c93d9efa71c470bd34/segmentation-3.0.onnx";

    #[test]
    fn pinned_repository_selects_model_card_revision() {
        let repo = PINNED_MODEL_REPOSITORY.into_hf_repo();
        assert_eq!(repo.folder_name(), "models--avencera--speakrs-models");
        assert_eq!(repo.revision(), HF_REVISION);

        let api = ApiBuilder::new().with_progress(false).build().unwrap();
        let api_repo = api.repo(repo);
        assert_eq!(api_repo.url(MODEL_FILENAME), EXPECTED_MODEL_URL);
    }

    #[test]
    fn default_and_custom_cache_managers_share_pinned_repository() {
        let default_manager = ModelManager::new().unwrap();
        let custom_cache = std::env::temp_dir().join(format!(
            "speakrs-pinned-model-manager-{}",
            std::process::id()
        ));
        let custom_manager = ModelManager::with_cache_dir(custom_cache).unwrap();

        assert_eq!(default_manager.repo.url(MODEL_FILENAME), EXPECTED_MODEL_URL);
        assert_eq!(custom_manager.repo.url(MODEL_FILENAME), EXPECTED_MODEL_URL);
    }

    #[test]
    fn coreml_required_files_include_chunk_fast_path_assets() {
        let files = required_files(ExecutionMode::CoreMl);
        assert!(files.contains(&"segmentation-3.0-b64.mlmodelc/model.mil".to_string()));
        assert!(files.contains(&"wespeaker-fbank-30s.mlmodelc/model.mil".to_string()));
        assert!(files.contains(&"wespeaker-multimask-tail-b32.mlmodelc/model.mil".to_string()));
        assert!(files.contains(&"wespeaker-chunk-emb-p1s-w111.mlmodelc/model.mil".to_string()));
    }

    #[test]
    fn coreml_fast_required_files_include_fast_assets() {
        let files = required_files(ExecutionMode::CoreMlFast);
        assert!(files.contains(&"segmentation-3.0-w8a16.mlmodelc/model.mil".to_string()));
        assert!(files.contains(&"segmentation-3.0-b64-w8a16.mlmodelc/model.mil".to_string()));
        assert!(files.contains(&"wespeaker-chunk-emb-s25-w56.mlmodelc/model.mil".to_string()));
    }

    #[test]
    fn migraphx_required_files_include_accelerated_onnx_assets() {
        let files = required_files(ExecutionMode::MiGraphX);
        assert!(files.contains(&"segmentation-3.0-b32.onnx".to_string()));
        assert!(files.contains(&"wespeaker-fbank.onnx".to_string()));
        assert!(files.contains(&"wespeaker-fbank-b32.onnx".to_string()));
        assert!(files.contains(&"wespeaker-multimask-tail.onnx".to_string()));
        assert!(files.contains(&"wespeaker-multimask-tail-b32.onnx".to_string()));
        assert!(files.contains(&"wespeaker-voxceleb-resnet34-b64.onnx".to_string()));
    }
}
