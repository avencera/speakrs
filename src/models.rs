use std::path::{Path, PathBuf};

#[cfg(feature = "online")]
use crate::inference::ExecutionMode;
#[cfg(feature = "online")]
use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};
#[cfg(feature = "online")]
use hf_hub::{Repo, RepoType};

const SEGMENTATION_ONNX: &str = "segmentation-3.0.onnx";
const EMBEDDING_ONNX: &str = "wespeaker-voxceleb-resnet34.onnx";
const EMBEDDING_MIN_SAMPLES: &str = "wespeaker-voxceleb-resnet34.min_num_samples.txt";

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
    /// Resolve and validate paths from a local directory containing all model files
    pub fn from_dir(
        models_dir: impl Into<PathBuf>,
    ) -> Result<Self, crate::inference::ModelLoadError> {
        let dir = models_dir.into();
        crate::inference::embedding::read_min_num_samples(&dir.join(EMBEDDING_MIN_SAMPLES))?;

        Ok(Self {
            segmentation_onnx: dir.join(SEGMENTATION_ONNX),
            embedding_onnx: dir.join(EMBEDDING_ONNX),
            plda_dir: dir,
        })
    }

    /// Download models from HuggingFace and resolve paths
    #[cfg(feature = "online")]
    #[cfg_attr(docsrs, doc(cfg(feature = "online")))]
    pub fn from_pretrained(mode: ExecutionMode) -> Result<Self, crate::inference::ModelLoadError> {
        let manager = ModelManager::new()?;
        let dir = manager.ensure(mode)?;
        Self::from_dir(dir)
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
// CI downloads fixtures from this same revision via SPEAKRS_MODEL_FIXTURE_REV
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
    EMBEDDING_MIN_SAMPLES,
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

/// Model family owned by the asset catalog
#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelFamily {
    Segmentation,
    Embedding,
    Filterbank,
    EmbeddingTail,
    MultiMaskTail,
    ChunkEmbedding,
    Plda,
    Metadata,
}

/// Backend that consumes a catalog asset
#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelBackend {
    Onnx,
    CoreMl,
}

/// Weight precision recorded for a catalog asset
#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelPrecision {
    Fp32,
    W8A16,
}

/// Whether a catalog asset must be downloaded for a mode
#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssetRequirement {
    Required,
    Optional,
}

/// One remote model file or compiled CoreML bundle stem
#[cfg(feature = "online")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ModelAsset {
    file_name: &'static str,
    family: ModelFamily,
    backend: ModelBackend,
    precision: ModelPrecision,
    batch: Option<u32>,
    requirement: AssetRequirement,
}

#[cfg(feature = "online")]
impl ModelAsset {
    const fn new(
        file_name: &'static str,
        family: ModelFamily,
        backend: ModelBackend,
        precision: ModelPrecision,
        batch: Option<u32>,
        requirement: AssetRequirement,
    ) -> Self {
        Self {
            file_name,
            family,
            backend,
            precision,
            batch,
            requirement,
        }
    }

    /// Remote file name or compiled-bundle directory
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn file_name(self) -> &'static str {
        self.file_name
    }

    /// Model family this asset belongs to
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn family(self) -> ModelFamily {
        self.family
    }

    /// Backend that loads this asset
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn backend(self) -> ModelBackend {
        self.backend
    }

    /// Recorded weight precision
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn precision(self) -> ModelPrecision {
        self.precision
    }

    /// Batch dimension when the asset is a fixed-shape model
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn batch(self) -> Option<u32> {
        self.batch
    }

    /// Whether this asset must be present for the selected mode
    pub(crate) fn requirement(self) -> AssetRequirement {
        self.requirement
    }

    fn remote_paths(self) -> Vec<String> {
        if self.file_name.ends_with(".mlmodelc") {
            mlmodelc_files(self.file_name)
        } else {
            vec![self.file_name.to_string()]
        }
    }
}

#[cfg(feature = "online")]
fn family_for_stem(name: &str) -> ModelFamily {
    if name.starts_with("segmentation") {
        ModelFamily::Segmentation
    } else if name.contains("chunk-emb") {
        ModelFamily::ChunkEmbedding
    } else if name.contains("fbank") {
        ModelFamily::Filterbank
    } else if name.contains("multimask") {
        ModelFamily::MultiMaskTail
    } else if name.contains("tail") {
        ModelFamily::EmbeddingTail
    } else {
        ModelFamily::Embedding
    }
}

#[cfg(feature = "online")]
fn batch_from_name(name: &str) -> Option<u32> {
    let stem = name
        .trim_end_matches(".mlmodelc")
        .trim_end_matches(".onnx")
        .trim_end_matches(".onnx.data");
    stem.rsplit_once("-b")
        .and_then(|(_, batch)| batch.parse().ok())
}

#[cfg(feature = "online")]
fn onnx_asset(name: &'static str, family: ModelFamily) -> ModelAsset {
    ModelAsset::new(
        name,
        family,
        ModelBackend::Onnx,
        ModelPrecision::Fp32,
        batch_from_name(name),
        AssetRequirement::Required,
    )
}

#[cfg(feature = "online")]
fn catalog_assets(mode: ExecutionMode) -> Vec<ModelAsset> {
    let mut assets: Vec<ModelAsset> = PLDA_FILES
        .iter()
        .map(|name| {
            let family = if name.ends_with(".txt") {
                ModelFamily::Metadata
            } else {
                ModelFamily::Plda
            };
            ModelAsset::new(
                name,
                family,
                ModelBackend::Onnx,
                ModelPrecision::Fp32,
                None,
                AssetRequirement::Required,
            )
        })
        .collect();

    match mode {
        ExecutionMode::Cpu => {
            assets.extend(ONNX_FILES.iter().copied().map(|name| {
                onnx_asset(
                    name,
                    if name.starts_with("segmentation") {
                        ModelFamily::Segmentation
                    } else {
                        ModelFamily::Embedding
                    },
                )
            }));
        }
        ExecutionMode::Cuda | ExecutionMode::CudaFast | ExecutionMode::MiGraphX => {
            assets.push(onnx_asset(
                "segmentation-3.0.onnx",
                ModelFamily::Segmentation,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34.onnx",
                ModelFamily::Embedding,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34.onnx.data",
                ModelFamily::Embedding,
            ));
            assets.push(onnx_asset("wespeaker-fbank.onnx", ModelFamily::Filterbank));
            assets.push(onnx_asset(
                "wespeaker-fbank-b32.onnx",
                ModelFamily::Filterbank,
            ));
            assets.push(onnx_asset(
                "wespeaker-multimask-tail.onnx",
                ModelFamily::MultiMaskTail,
            ));
            assets.push(onnx_asset(
                "wespeaker-multimask-tail-b32.onnx",
                ModelFamily::MultiMaskTail,
            ));
            assets.push(onnx_asset(
                "segmentation-3.0-b32.onnx",
                ModelFamily::Segmentation,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34-b64.onnx",
                ModelFamily::Embedding,
            ));
        }
        ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => {
            assets.push(onnx_asset(
                "segmentation-3.0.onnx",
                ModelFamily::Segmentation,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34.onnx",
                ModelFamily::Embedding,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34.onnx.data",
                ModelFamily::Embedding,
            ));
            assets.push(onnx_asset(
                "segmentation-3.0-b32.onnx",
                ModelFamily::Segmentation,
            ));
            assets.push(onnx_asset("wespeaker-fbank.onnx", ModelFamily::Filterbank));
            assets.push(onnx_asset(
                "wespeaker-fbank-b32.onnx",
                ModelFamily::Filterbank,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34-tail.onnx",
                ModelFamily::EmbeddingTail,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34-tail-b3.onnx",
                ModelFamily::EmbeddingTail,
            ));
            assets.push(onnx_asset(
                "wespeaker-voxceleb-resnet34-tail-b32.onnx",
                ModelFamily::EmbeddingTail,
            ));
            assets.extend(COREML_COMMON_MODEL_STEMS.iter().map(|&name| {
                ModelAsset::new(
                    name,
                    family_for_stem(name),
                    ModelBackend::CoreMl,
                    ModelPrecision::Fp32,
                    batch_from_name(name),
                    AssetRequirement::Required,
                )
            }));
            if matches!(mode, ExecutionMode::CoreMl) {
                assets.extend(COREML_CHUNK_MODEL_STEMS.iter().map(|&name| {
                    ModelAsset::new(
                        name,
                        ModelFamily::ChunkEmbedding,
                        ModelBackend::CoreMl,
                        ModelPrecision::Fp32,
                        batch_from_name(name),
                        AssetRequirement::Required,
                    )
                }));
            }
            if matches!(mode, ExecutionMode::CoreMlFast) {
                assets.extend(COREML_FAST_SEGMENTATION_MODEL_STEMS.iter().map(|&name| {
                    ModelAsset::new(
                        name,
                        ModelFamily::Segmentation,
                        ModelBackend::CoreMl,
                        ModelPrecision::W8A16,
                        batch_from_name(name),
                        AssetRequirement::Required,
                    )
                }));
                assets.extend(COREML_FAST_CHUNK_MODEL_STEMS.iter().map(|&name| {
                    ModelAsset::new(
                        name,
                        ModelFamily::ChunkEmbedding,
                        ModelBackend::CoreMl,
                        ModelPrecision::Fp32,
                        batch_from_name(name),
                        AssetRequirement::Required,
                    )
                }));
            }
            assets.push(ModelAsset::new(
                "wespeaker-voxceleb-resnet34-tail-b64.mlmodelc",
                ModelFamily::EmbeddingTail,
                ModelBackend::CoreMl,
                ModelPrecision::Fp32,
                Some(64),
                AssetRequirement::Optional,
            ));
        }
    }

    assets
}

#[cfg(feature = "online")]
fn required_files(mode: ExecutionMode) -> Vec<String> {
    catalog_assets(mode)
        .into_iter()
        .filter(|asset| asset.requirement() == AssetRequirement::Required)
        .flat_map(ModelAsset::remote_paths)
        .collect()
}

#[cfg(test)]
mod local_tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::{EMBEDDING_MIN_SAMPLES, ModelBundle};

    static NEXT_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn scratch_directory(test_name: &str) -> std::path::PathBuf {
        let id = NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "speakrs-model-bundle-{}-{test_name}-{id}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn local_bundle_requires_embedding_metadata() {
        let directory = scratch_directory("missing-metadata");
        let error = ModelBundle::from_dir(&directory).unwrap_err();

        assert!(error.to_string().contains("missing embedding metadata"));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn local_bundle_validates_embedding_metadata_before_construction() {
        let directory = scratch_directory("metadata-validation");
        let metadata = directory.join(EMBEDDING_MIN_SAMPLES);
        fs::write(&metadata, "0\n").unwrap();
        let error = ModelBundle::from_dir(&directory).unwrap_err();
        assert!(error.to_string().contains("must be greater than zero"));

        fs::write(metadata, "400\n").unwrap();
        ModelBundle::from_dir(&directory).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }
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
    fn catalog_records_batch_32_coreml_tail() {
        let tail = ModelAsset::new(
            "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc",
            ModelFamily::EmbeddingTail,
            ModelBackend::CoreMl,
            ModelPrecision::Fp32,
            Some(32),
            AssetRequirement::Required,
        );
        assert_eq!(
            tail.file_name(),
            "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc"
        );
        assert_eq!(tail.family(), ModelFamily::EmbeddingTail);
        assert_eq!(tail.backend(), ModelBackend::CoreMl);
        assert_eq!(tail.precision(), ModelPrecision::Fp32);
        assert_eq!(tail.batch(), Some(32));
        assert_eq!(
            tail.remote_paths()[0],
            "wespeaker-voxceleb-resnet34-tail-b32.mlmodelc/model.mil"
        );
    }

    #[test]
    fn cpu_required_files_include_plda_metadata_and_onnx() {
        let files = required_files(ExecutionMode::Cpu);
        for name in PLDA_FILES.iter().chain(ONNX_FILES) {
            assert!(
                files.contains(&name.to_string()),
                "cpu catalog missing {name}"
            );
        }
    }

    #[test]
    fn coreml_catalog_records_optional_batch_64_tail() {
        let optional = catalog_assets(ExecutionMode::CoreMl)
            .into_iter()
            .find(|asset| asset.batch() == Some(64) && asset.family() == ModelFamily::EmbeddingTail)
            .expect("optional batch-64 CoreML tail");
        assert_eq!(optional.requirement(), AssetRequirement::Optional);
        let files = required_files(ExecutionMode::CoreMl);
        assert!(!files.iter().any(|path| path.contains("tail-b64")));
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
