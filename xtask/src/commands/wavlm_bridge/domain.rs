use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use clap::ValueEnum;
use color_eyre::eyre::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

pub use speakrs::imported_segmentation::Sha256Digest;
use speakrs::pipeline::{
    EmbeddingAvailability, EmbeddingFailureReason as LibraryEmbeddingFailureReason,
    EmbeddingReceipt, EmbeddingStageEntry as LibraryEmbeddingStageEntry, EmbeddingStageSnapshot,
    InactiveEmbeddingReason as LibraryInactiveEmbeddingReason, PipelineGeometry,
};

pub const SPEC_SCHEMA_VERSION: u32 = 1;
pub const VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const RUN_SCHEMA_VERSION: u32 = 1;
pub const SYSTEM_SCHEMA_VERSION: u32 = 1;
pub const REPORT_SCHEMA_VERSION: u32 = 1;
pub const SCORE_SCHEMA_VERSION: u32 = 2;
pub const RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const CACHE_SCHEMA_VERSION: u32 = 2;
pub const EMBEDDING_CACHE_SCHEMA_VERSION: u32 = 2;
pub const EMBEDDING_STAGE_SCHEMA_VERSION: u32 = 2;
pub const MAX_EMBEDDING_STAGE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_EMBEDDING_STAGE_VALUES: usize = 64 * 1024 * 1024;
pub const MAX_EMBEDDING_STAGE_ENTRIES: usize = 4 * 1024 * 1024;
const MAX_EMBEDDING_STAGE_MASK_BYTES: usize = MAX_EMBEDDING_STAGE_VALUES.div_ceil(8);
const IMPORTED_EMBEDDING_WIDTH: usize = 256;

/// Execution mode admitted by the bridge command
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum BridgeMode {
    /// ONNX Runtime CPU execution
    Cpu,
    /// ONNX Runtime CUDA execution when the CUDA feature is compiled
    #[cfg(feature = "cuda")]
    #[serde(rename = "cuda")]
    #[value(name = "cuda")]
    Cuda,
}

impl BridgeMode {
    pub fn to_execution_mode(self) -> Result<speakrs::ExecutionMode> {
        match self {
            Self::Cpu => Ok(speakrs::ExecutionMode::Cpu),
            #[cfg(feature = "cuda")]
            Self::Cuda => Ok(speakrs::ExecutionMode::Cuda),
        }
    }
}

/// Fixed scorer identity shared by all three report inputs
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScorerIdentity {
    pub implementation: String,
    pub version: String,
    pub collar_seconds: f64,
    pub overlap: OverlapPolicy,
    pub speaker_count: SpeakerCountPolicy,
    pub config_sha256: Sha256Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OverlapPolicy {
    Included,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerCountPolicy {
    Automatic,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JoinIdentity {
    pub membership_sha256: Sha256Digest,
    pub audio_manifest_sha256: Sha256Digest,
    pub reference_manifest_sha256: Sha256Digest,
    pub uem_manifest_sha256: Sha256Digest,
    pub scorer: ScorerIdentity,
    pub aggregation: AggregationIdentity,
    pub qualification: QualificationIdentity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AggregationIdentity {
    pub id: String,
    pub revision: String,
    pub equal_domain_average: bool,
    pub pooled_secondary: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationIdentity {
    pub id: String,
    pub revision: String,
    pub sentinel_membership_sha256: Option<Sha256Digest>,
    pub monitor_membership_sha256: Option<Sha256Digest>,
    pub diagnostic_membership_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MembershipIdentity {
    pub id: String,
    pub sha256: Sha256Digest,
    pub role: MembershipRole,
    pub recording_ids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MembershipRole {
    FixedProbe,
    Sentinel,
    Monitor,
    Diagnostic,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub mode: BridgeMode,
    pub models_sha256: Sha256Digest,
    pub embedding_model: EmbeddingModelIdentity,
    pub plda: ModelIdentity,
    pub precision: Precision,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingModelIdentity {
    pub id: String,
    pub revision: String,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub sidecar_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub id: String,
    pub revision: String,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    Float32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactInput {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BundleInput {
    pub path: PathBuf,
    pub bundle_id: Sha256Digest,
    pub manifest_sha256: Sha256Digest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordingSpec {
    pub id: String,
    pub source: String,
    pub domain: String,
    pub parent_group: String,
    pub audio: ArtifactInput,
    pub bundle: BundleInput,
    pub reference: ArtifactInput,
    pub uem: ArtifactInput,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeSpec {
    pub id: String,
    pub revision: String,
    pub seed: u64,
    pub decoder: RecipeIdentity,
    pub embedding: RecipeIdentity,
    pub clustering: ClusteringSpec,
    pub reconstruction: ReconstructionSpec,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeIdentity {
    pub id: String,
    pub revision: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusteringSpec {
    pub ahc_threshold: f32,
    pub speaker_keep_threshold: f64,
    pub vbx_fa: f64,
    pub vbx_fb: f64,
    pub vbx_max_iters: usize,
    pub vbx_epsilon: f64,
    pub vbx_initialization: VbxInitialization,
    pub clean_frame_duration_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VbxInitialization {
    Hard,
    Uniform,
    Smoothed { scale: f64 },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionSpec {
    pub id: String,
    pub revision: String,
    pub activity: ActivitySpec,
    pub merge_gap_seconds: f64,
    pub method: ReconstructionMethod,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivitySpec {
    pub min_active_frames: usize,
    pub max_inactive_gap_frames: usize,
    pub pad_before_frames: usize,
    pub pad_after_frames: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReconstructionMethod {
    Standard,
    Smoothed { epsilon: f32 },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeSpec {
    pub schema_version: u32,
    pub experiment_id: String,
    pub membership: MembershipIdentity,
    pub join: JoinIdentity,
    pub runtime: RuntimeIdentity,
    pub recordings: Vec<RecordingSpec>,
    pub recipes: Vec<RecipeSpec>,
}

impl BridgeSpec {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .wrap_err_with(|| format!("failed to read bridge spec {}", path.display()))?;
        let spec: Self = serde_json::from_slice(&bytes)
            .wrap_err_with(|| format!("invalid bridge spec {}", path.display()))?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SPEC_SCHEMA_VERSION,
            "unsupported bridge spec schema {}; expected {SPEC_SCHEMA_VERSION}",
            self.schema_version
        );
        validate_id("experiment_id", &self.experiment_id)?;
        validate_id("membership.id", &self.membership.id)?;
        ensure!(!self.recordings.is_empty(), "bridge spec has no recordings");
        ensure!(!self.recipes.is_empty(), "bridge spec has no recipes");
        ensure!(
            self.join.membership_sha256 == self.membership.sha256,
            "join membership digest does not match membership identity"
        );
        validate_scorer(&self.join.scorer)?;
        validate_model_path(
            "runtime.embedding_model.path",
            &self.runtime.embedding_model.path,
        )?;
        validate_model_path("runtime.plda.path", &self.runtime.plda.path)?;
        validate_id(
            "runtime.embedding_model.id",
            &self.runtime.embedding_model.id,
        )?;
        validate_id(
            "runtime.embedding_model.revision",
            &self.runtime.embedding_model.revision,
        )?;
        validate_id("runtime.plda.id", &self.runtime.plda.id)?;
        validate_id("runtime.plda.revision", &self.runtime.plda.revision)?;
        validate_id("join.aggregation.id", &self.join.aggregation.id)?;
        validate_id("join.aggregation.revision", &self.join.aggregation.revision)?;
        validate_id("join.qualification.id", &self.join.qualification.id)?;
        validate_id(
            "join.qualification.revision",
            &self.join.qualification.revision,
        )?;
        let mut recording_ids = BTreeSet::new();
        for recording in &self.recordings {
            validate_recording_shape(recording)?;
            ensure!(
                recording_ids.insert(recording.id.clone()),
                "duplicate recording id {}",
                recording.id
            );
        }
        ensure!(
            self.membership.recording_ids
                == self
                    .recordings
                    .iter()
                    .map(|r| r.id.clone())
                    .collect::<Vec<_>>(),
            "membership recording_ids must exactly match recordings in order"
        );
        let mut recipe_ids = BTreeSet::new();
        for recipe in &self.recipes {
            validate_recipe(recipe)?;
            ensure!(
                recipe_ids.insert(recipe.id.clone()),
                "duplicate recipe id {}",
                recipe.id
            );
        }
        Ok(())
    }

    pub fn recipe(&self, id: &str) -> Result<&RecipeSpec> {
        self.recipes
            .iter()
            .find(|recipe| recipe.id == id)
            .ok_or_else(|| color_eyre::eyre::eyre!("unknown recipe {id}"))
    }
}

impl RecipeSpec {
    pub fn pipeline_config(&self, mode: speakrs::ExecutionMode) -> Result<speakrs::PipelineConfig> {
        let ahc = speakrs::AhcConfig::new(self.clustering.ahc_threshold)?;
        let initialization = match self.clustering.vbx_initialization {
            VbxInitialization::Hard => speakrs::ResponsibilityInitialization::Hard,
            VbxInitialization::Uniform => speakrs::ResponsibilityInitialization::Uniform,
            VbxInitialization::Smoothed { scale } => {
                speakrs::ResponsibilityInitialization::Smoothed(scale)
            }
        };
        let vbx = speakrs::VbxConfig::new(
            self.clustering.vbx_fa,
            self.clustering.vbx_fb,
            self.clustering.vbx_max_iters,
            self.clustering.vbx_epsilon,
            initialization,
        )?;
        let clustering = speakrs::ClusteringConfig::new(
            ahc,
            self.clustering.speaker_keep_threshold,
            speakrs::ClusteringBackend::GaussianVbx(vbx),
        )?;
        let clean_frame_duration = speakrs::pipeline::CleanFrameDuration::new(
            self.clustering.clean_frame_duration_seconds,
        )?;
        let method = match self.reconstruction.method {
            ReconstructionMethod::Standard => speakrs::pipeline::ReconstructMethod::Standard,
            ReconstructionMethod::Smoothed { epsilon } => {
                speakrs::pipeline::ReconstructMethod::Smoothed { epsilon }
            }
        };
        let _ = mode;
        Ok(speakrs::PipelineConfig {
            activity: speakrs::ActivityCleanup::new(
                self.reconstruction.activity.min_active_frames,
                self.reconstruction.activity.max_inactive_gap_frames,
                self.reconstruction.activity.pad_before_frames,
                self.reconstruction.activity.pad_after_frames,
            ),
            clustering,
            clean_frame_duration,
            merge_gap: self.reconstruction.merge_gap_seconds,
            reconstruct_method: method,
        })
    }
}

fn validate_scorer(scorer: &ScorerIdentity) -> Result<()> {
    ensure!(
        scorer.implementation == "pyannote.metrics",
        "B3 requires scorer implementation pyannote.metrics"
    );
    ensure!(
        scorer.version == "4.0.0",
        "B3 requires pyannote.metrics version 4.0.0"
    );
    ensure!(scorer.collar_seconds == 0.0, "B3 requires zero collar");
    ensure!(
        scorer.overlap == OverlapPolicy::Included,
        "B3 requires overlap included"
    );
    ensure!(
        scorer.speaker_count == SpeakerCountPolicy::Automatic,
        "B3 requires automatic speaker count"
    );
    Ok(())
}

fn validate_recording_shape(recording: &RecordingSpec) -> Result<()> {
    validate_id("recording.id", &recording.id)?;
    ensure!(
        !recording.source.trim().is_empty(),
        "recording source is empty"
    );
    ensure!(
        !recording.domain.trim().is_empty(),
        "recording domain is empty"
    );
    ensure!(
        !recording.parent_group.trim().is_empty(),
        "recording parent_group is empty"
    );
    validate_relative_or_absolute_path("audio.path", &recording.audio.path)?;
    validate_relative_or_absolute_path("bundle.path", &recording.bundle.path)?;
    validate_relative_or_absolute_path("reference.path", &recording.reference.path)?;
    validate_relative_or_absolute_path("uem.path", &recording.uem.path)?;
    Ok(())
}

fn validate_recipe(recipe: &RecipeSpec) -> Result<()> {
    validate_id("recipe.id", &recipe.id)?;
    validate_id("recipe.revision", &recipe.revision)?;
    validate_id("recipe.decoder.id", &recipe.decoder.id)?;
    validate_id("recipe.decoder.revision", &recipe.decoder.revision)?;
    validate_id("recipe.embedding.id", &recipe.embedding.id)?;
    validate_id("recipe.embedding.revision", &recipe.embedding.revision)?;
    validate_id("recipe.reconstruction.id", &recipe.reconstruction.id)?;
    validate_id(
        "recipe.reconstruction.revision",
        &recipe.reconstruction.revision,
    )?;
    ensure!(
        recipe.clustering.ahc_threshold.is_finite() && recipe.clustering.ahc_threshold >= 0.0,
        "recipe {} has invalid AHC threshold",
        recipe.id
    );
    ensure!(
        recipe.clustering.speaker_keep_threshold.is_finite()
            && recipe.clustering.speaker_keep_threshold >= 0.0,
        "recipe {} has invalid speaker keep threshold",
        recipe.id
    );
    ensure!(
        recipe.clustering.vbx_fa.is_finite() && recipe.clustering.vbx_fa > 0.0,
        "recipe {} has invalid VBx FA",
        recipe.id
    );
    ensure!(
        recipe.clustering.vbx_fb.is_finite() && recipe.clustering.vbx_fb > 0.0,
        "recipe {} has invalid VBx FB",
        recipe.id
    );
    ensure!(
        recipe.clustering.vbx_max_iters > 0,
        "recipe {} has zero VBx iterations",
        recipe.id
    );
    ensure!(
        recipe.clustering.vbx_epsilon.is_finite() && recipe.clustering.vbx_epsilon >= 0.0,
        "recipe {} has invalid VBx epsilon",
        recipe.id
    );
    if let VbxInitialization::Smoothed { scale } = recipe.clustering.vbx_initialization {
        ensure!(
            scale.is_finite() && scale > 0.0,
            "recipe {} has invalid VBx smoothing scale",
            recipe.id
        );
    }
    ensure!(
        recipe.clustering.clean_frame_duration_seconds.is_finite()
            && recipe.clustering.clean_frame_duration_seconds > 0.0,
        "recipe {} has invalid clean-frame duration",
        recipe.id
    );
    ensure!(
        recipe.reconstruction.merge_gap_seconds.is_finite()
            && recipe.reconstruction.merge_gap_seconds >= 0.0,
        "recipe {} has invalid merge gap",
        recipe.id
    );
    if let ReconstructionMethod::Smoothed { epsilon } = recipe.reconstruction.method {
        ensure!(
            epsilon.is_finite() && epsilon >= 0.0,
            "recipe {} has invalid reconstruction epsilon",
            recipe.id
        );
    }
    Ok(())
}

fn validate_id(field: &str, value: &str) -> Result<()> {
    ensure!(!value.is_empty(), "{field} cannot be empty");
    ensure!(
        value.trim() == value,
        "{field} cannot contain surrounding whitespace"
    );
    ensure!(
        value != "." && value != ".." && !value.contains('/') && !value.contains('\\'),
        "{field} must be a single safe identifier: {value}"
    );
    Ok(())
}

fn validate_relative_or_absolute_path(field: &str, path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{field} cannot be empty");
    ensure!(
        !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir)),
        "{field} cannot contain parent traversal: {}",
        path.display()
    );
    Ok(())
}

fn validate_model_path(field: &str, path: &Path) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{field} cannot be empty");
    ensure!(
        !path.is_absolute(),
        "{field} must be relative to --models-dir"
    );
    ensure!(
        !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir)),
        "{field} cannot contain parent traversal: {}",
        path.display()
    );
    Ok(())
}

pub fn digest_bytes(bytes: &[u8]) -> Sha256Digest {
    Sha256Digest::digest(bytes)
}

pub fn digest_file(path: &Path) -> Result<Sha256Digest> {
    let bytes = fs::read(path).wrap_err_with(|| format!("failed to read {}", path.display()))?;
    Ok(digest_bytes(&bytes))
}

pub fn digest_tree(root: &Path) -> Result<Sha256Digest> {
    ensure_directory(root, "digest root")?;
    let mut files = Vec::new();
    collect_files(root, root, &mut files)?;
    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut digest = Sha256::new();
    for (relative, path) in files {
        let bytes = fs::read(&path)
            .wrap_err_with(|| format!("failed to read model file {}", path.display()))?;
        update_length_prefixed(&mut digest, relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(finalize_digest(digest))
}

fn collect_files(root: &Path, current: &Path, files: &mut Vec<(String, PathBuf)>) -> Result<()> {
    for entry in fs::read_dir(current)
        .wrap_err_with(|| format!("failed to read directory {}", current.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "symlinks are not allowed: {}",
            path.display()
        );
        if metadata.is_dir() {
            collect_files(root, &path, files)?;
        } else if metadata.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| color_eyre::eyre::eyre!("model file escaped root"))?
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, path));
        } else {
            bail!("unsupported model directory member: {}", path.display());
        }
    }
    Ok(())
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn finalize_digest(digest: Sha256) -> Sha256Digest {
    Sha256Digest::new(format!("{:x}", digest.finalize()))
        .expect("SHA-256 formatting always produces a canonical digest")
}

pub fn canonical_json_digest<T: Serialize>(value: &T) -> Result<Sha256Digest> {
    let bytes = serde_json::to_vec(value)?;
    Ok(digest_bytes(&bytes))
}

pub fn ensure_regular_file(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .wrap_err_with(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "{label} must be a regular file: {}",
        path.display()
    );
    Ok(())
}

pub fn ensure_directory(path: &Path, label: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .wrap_err_with(|| format!("failed to inspect {label} {}", path.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "{label} must be a real directory: {}",
        path.display()
    );
    Ok(())
}

pub fn make_tree_read_only(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "published tree contains a symlink: {}",
        root.display()
    );
    let mut permissions = metadata.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(root, permissions)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(root)? {
            make_tree_read_only(&entry?.path())?;
        }
    }
    Ok(())
}

/// A durable reference to an immutable published file
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRef {
    pub relative_path: PathBuf,
    pub sha256: Sha256Digest,
    pub bytes: u64,
}

/// A file that is either present and hashed or explicitly unavailable
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ArtifactState {
    Available { artifact: ArtifactRef },
    Unavailable { reason: AvailabilityReason },
}

/// Closed reasons for a deliberately absent measurement or artifact
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AvailabilityReason {
    NotMeasured,
    NotCalculated,
    MissingInput,
    InferenceFailed { detail: String },
    InvalidCache { detail: String },
    NoUsableEmbedding,
    Other { detail: String },
}

/// A measurement that was observed or remains unavailable
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum MeasurementState<T> {
    Available { value: T },
    Unavailable { reason: AvailabilityReason },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Bundle,
    Decode,
    Embedding,
    Clustering,
    Reconstruction,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageDependency {
    pub name: StageDependencyName,
    pub value: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StageDependencyName {
    Recording,
    Audio,
    Bundle,
    Decoder,
    Embedding,
    Implementation,
    Model,
    Runtime,
    Geometry,
    Clustering,
    Reconstruction,
    Seed,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptRef {
    pub stage: StageKind,
    pub cache_key: Sha256Digest,
    pub receipt_sha256: Sha256Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RationalReceipt {
    pub numerator: i64,
    pub denominator: i64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrameGridReceipt {
    pub frame_count: u32,
    pub origin: RationalReceipt,
    pub step: RationalReceipt,
    pub support: RationalReceipt,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkExtentReceipt {
    pub index: usize,
    pub start_samples: usize,
    pub valid_samples: usize,
    pub padding_samples: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryReceipt {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub window_samples: usize,
    pub step_samples: usize,
    pub chunks: Vec<ChunkExtentReceipt>,
    pub frame_grid: FrameGridReceipt,
    pub aggregate_grid: FrameGridReceipt,
    pub start_frames: Vec<usize>,
    pub output_frames: usize,
    pub output_extent_start_samples: u64,
    pub output_extent_end_samples: u64,
    pub output_extent_policy: speakrs::imported_segmentation::OutputExtentPolicy,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AvailabilityCounts {
    pub chunks: usize,
    pub local_slots: usize,
    pub available: usize,
    pub inactive: usize,
    pub inference_failed: usize,
    pub clean_mask: usize,
    pub full_mask_fallback: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StageReceipt {
    pub schema_version: u32,
    pub stage: StageKind,
    pub cache_key: Sha256Digest,
    pub parents: Vec<ReceiptRef>,
    pub dependencies: Vec<StageDependency>,
    pub geometry: GeometryReceipt,
    pub availability: AvailabilityCounts,
    pub outputs: Vec<ArtifactRef>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptDocument {
    pub receipt: StageReceipt,
    pub receipt_sha256: Sha256Digest,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInactiveReason {
    NoActivity,
    InsufficientActivity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingFailureReason {
    ModelExecution,
    InvalidOutput,
    LegacyUnavailable,
}

/// One validated embedding vector serialized as standard padded base64
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EncodedEmbeddingVector([f32; IMPORTED_EMBEDDING_WIDTH]);

impl EncodedEmbeddingVector {
    pub(crate) fn try_from_values(values: &[f32]) -> Result<Self> {
        ensure!(
            values.len() == IMPORTED_EMBEDDING_WIDTH,
            "embedding vector has width {}, expected {IMPORTED_EMBEDDING_WIDTH}",
            values.len()
        );
        ensure!(
            values.iter().all(|value| value.is_finite()),
            "embedding vector contains non-finite values"
        );
        Ok(Self(
            values
                .try_into()
                .expect("embedding vector width was checked"),
        ))
    }

    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.0
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.0.iter().all(|value| value.is_finite()),
            "embedding vector contains non-finite values"
        );
        Ok(())
    }
}

impl Serialize for EncodedEmbeddingVector {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if let Err(error) = self.validate() {
            return Err(serde::ser::Error::custom(error.to_string()));
        }
        let bytes = self
            .0
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        serializer.serialize_str(&STANDARD.encode(bytes))
    }
}

impl<'de> Deserialize<'de> for EncodedEmbeddingVector {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let bytes = decode_canonical_base64(
            &value,
            IMPORTED_EMBEDDING_WIDTH * std::mem::size_of::<f32>(),
            "embedding vector",
        )
        .map_err(|error| serde::de::Error::custom(error.to_string()))?;
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        Self::try_from_values(&values).map_err(|error| serde::de::Error::custom(error.to_string()))
    }
}

/// Packed segmentation bits with bounded canonical base64 serialization
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PackedSegmentationMask(Vec<u8>);

impl PackedSegmentationMask {
    pub(crate) fn try_from_values(shape: [usize; 3], values: &[f32]) -> Result<Self> {
        let value_count = checked_segmentation_value_count(shape)?;
        ensure!(
            values.len() == value_count,
            "embedding-stage segmentation values {} do not match shape {shape:?}",
            values.len()
        );
        ensure!(
            value_count <= MAX_EMBEDDING_STAGE_VALUES,
            "embedding-stage segmentation values exceed bound"
        );
        let byte_count = packed_mask_byte_count(value_count)?;
        let mut bytes = vec![0_u8; byte_count];
        for (index, value) in values.iter().enumerate() {
            ensure!(
                value.is_finite() && (*value == 0.0 || *value == 1.0),
                "embedding-stage masks are not finite binary values"
            );
            if *value == 1.0 {
                bytes[index / 8] |= 1 << (index % 8);
            }
        }
        Ok(Self(bytes))
    }

    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self(Vec::new())
    }
}

impl Serialize for PackedSegmentationMask {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        if self.0.len() > MAX_EMBEDDING_STAGE_MASK_BYTES {
            return Err(serde::ser::Error::custom(
                "embedding-stage segmentation mask exceeds the decoded size bound",
            ));
        }
        serializer.serialize_str(&STANDARD.encode(&self.0))
    }
}

impl<'de> Deserialize<'de> for PackedSegmentationMask {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        let bytes = decode_bounded_canonical_base64(
            &value,
            MAX_EMBEDDING_STAGE_MASK_BYTES,
            "embedding-stage segmentation mask",
        )
        .map_err(|error| serde::de::Error::custom(error.to_string()))?;
        Ok(Self(bytes))
    }
}

/// One serialized embedding-stage slot with availability-specific data
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum EmbeddingStageEntry {
    Available { values: Box<EncodedEmbeddingVector> },
    Inactive { reason: EmbeddingInactiveReason },
    InferenceFailed { reason: EmbeddingFailureReason },
}

impl TryFrom<&LibraryEmbeddingStageEntry> for EmbeddingStageEntry {
    type Error = color_eyre::eyre::Report;

    fn try_from(entry: &LibraryEmbeddingStageEntry) -> std::result::Result<Self, Self::Error> {
        match entry.availability() {
            EmbeddingAvailability::Available => {
                let values = entry
                    .values()
                    .ok_or_else(|| color_eyre::eyre::eyre!("available embedding has no vector"))?;
                Ok(Self::Available {
                    values: Box::new(EncodedEmbeddingVector::try_from_values(values)?),
                })
            }
            EmbeddingAvailability::Inactive { reason } => {
                ensure!(
                    entry.values().is_none(),
                    "unavailable embedding has a vector"
                );
                Ok(Self::Inactive {
                    reason: match reason {
                        LibraryInactiveEmbeddingReason::NoActivity => {
                            EmbeddingInactiveReason::NoActivity
                        }
                        LibraryInactiveEmbeddingReason::InsufficientActivity => {
                            EmbeddingInactiveReason::InsufficientActivity
                        }
                    },
                })
            }
            EmbeddingAvailability::InferenceFailed { reason } => {
                ensure!(
                    entry.values().is_none(),
                    "unavailable embedding has a vector"
                );
                Ok(Self::InferenceFailed {
                    reason: match reason {
                        LibraryEmbeddingFailureReason::ModelExecution => {
                            EmbeddingFailureReason::ModelExecution
                        }
                        LibraryEmbeddingFailureReason::InvalidOutput => {
                            EmbeddingFailureReason::InvalidOutput
                        }
                        LibraryEmbeddingFailureReason::LegacyUnavailable => {
                            EmbeddingFailureReason::LegacyUnavailable
                        }
                    },
                })
            }
        }
    }
}

impl EmbeddingStageEntry {
    fn validate(&self) -> Result<()> {
        if let Self::Available { values } = self {
            values.validate()?;
        }
        Ok(())
    }

    pub(crate) fn to_library(&self) -> LibraryEmbeddingStageEntry {
        match self {
            Self::Available { values } => LibraryEmbeddingStageEntry::new(
                EmbeddingAvailability::Available,
                Some(values.as_slice().to_vec()),
            ),
            Self::Inactive { reason } => LibraryEmbeddingStageEntry::new(
                EmbeddingAvailability::Inactive {
                    reason: match reason {
                        EmbeddingInactiveReason::NoActivity => {
                            LibraryInactiveEmbeddingReason::NoActivity
                        }
                        EmbeddingInactiveReason::InsufficientActivity => {
                            LibraryInactiveEmbeddingReason::InsufficientActivity
                        }
                    },
                },
                None,
            ),
            Self::InferenceFailed { reason } => LibraryEmbeddingStageEntry::new(
                EmbeddingAvailability::InferenceFailed {
                    reason: match reason {
                        EmbeddingFailureReason::ModelExecution => {
                            LibraryEmbeddingFailureReason::ModelExecution
                        }
                        EmbeddingFailureReason::InvalidOutput => {
                            LibraryEmbeddingFailureReason::InvalidOutput
                        }
                        EmbeddingFailureReason::LegacyUnavailable => {
                            LibraryEmbeddingFailureReason::LegacyUnavailable
                        }
                    },
                },
                None,
            ),
        }
    }
}

/// Immutable serialized output of bundle decode and per-speaker embedding
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EmbeddingStageDocument {
    pub schema_version: u32,
    pub recording_id: String,
    pub stage_key: Sha256Digest,
    pub geometry: GeometryReceipt,
    pub segmentation_shape: [usize; 3],
    /// Row-major mask bits with the least-significant bit first in each byte
    pub segmentation_values: PackedSegmentationMask,
    pub entries: Vec<EmbeddingStageEntry>,
    pub embedding_receipt: AvailabilityCounts,
}

impl EmbeddingStageDocument {
    pub(crate) fn decode_segmentation_values(&self) -> Result<Vec<f32>> {
        let value_count = checked_segmentation_value_count(self.segmentation_shape)?;
        let bytes = self.decode_segmentation_bytes()?;
        Ok((0..value_count)
            .map(|index| {
                if bytes[index / 8] & (1 << (index % 8)) == 0 {
                    0.0
                } else {
                    1.0
                }
            })
            .collect())
    }

    fn decode_segmentation_bytes(&self) -> Result<Vec<u8>> {
        let value_count = checked_segmentation_value_count(self.segmentation_shape)?;
        ensure!(
            value_count <= MAX_EMBEDDING_STAGE_VALUES,
            "embedding-stage segmentation values exceed bound"
        );
        let bytes = self.segmentation_values.0.as_slice();
        ensure!(
            bytes.len() == packed_mask_byte_count(value_count)?,
            "embedding-stage segmentation mask byte count does not match shape"
        );
        ensure_mask_padding_is_zero(bytes, value_count)?;
        Ok(bytes.to_owned())
    }

    pub(crate) fn to_library_snapshot(
        &self,
        geometry: PipelineGeometry,
    ) -> Result<EmbeddingStageSnapshot> {
        self.validate()?;
        let entries = self
            .entries
            .iter()
            .map(EmbeddingStageEntry::to_library)
            .collect::<Vec<_>>();
        EmbeddingStageSnapshot::from_flat_parts(
            geometry,
            self.segmentation_shape,
            self.decode_segmentation_values()?,
            entries,
            EmbeddingReceipt {
                clean_mask_count: self.embedding_receipt.clean_mask,
                full_mask_fallback_count: self.embedding_receipt.full_mask_fallback,
                inactive_count: self.embedding_receipt.inactive,
                inference_failure_count: self.embedding_receipt.inference_failed,
            },
        )
        .map_err(Into::into)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == EMBEDDING_STAGE_SCHEMA_VERSION,
            "unsupported embedding-stage schema {}",
            self.schema_version
        );
        ensure!(
            !self.recording_id.is_empty(),
            "embedding-stage recording id is empty"
        );
        let values = checked_segmentation_value_count(self.segmentation_shape)?;
        ensure!(
            values <= MAX_EMBEDDING_STAGE_VALUES,
            "embedding-stage segmentation values exceed bound"
        );
        self.decode_segmentation_bytes()?;
        let entries = self.segmentation_shape[0]
            .checked_mul(self.segmentation_shape[2])
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding-stage entry shape overflow"))?;
        ensure!(
            entries == self.entries.len() && entries <= MAX_EMBEDDING_STAGE_ENTRIES,
            "embedding-stage entries do not match shape or exceed bound"
        );
        let mut available = 0;
        let mut inactive = 0;
        let mut failed = 0;
        for entry in &self.entries {
            entry.validate()?;
            match entry {
                EmbeddingStageEntry::Available { .. } => available += 1,
                EmbeddingStageEntry::Inactive { .. } => inactive += 1,
                EmbeddingStageEntry::InferenceFailed { .. } => failed += 1,
            }
        }
        ensure!(
            available == self.embedding_receipt.available,
            "embedding receipt available count does not match entries"
        );
        ensure!(
            inactive == self.embedding_receipt.inactive,
            "embedding receipt inactive count does not match entries"
        );
        ensure!(
            failed == self.embedding_receipt.inference_failed,
            "embedding receipt failure count does not match entries"
        );
        ensure!(
            self.embedding_receipt.chunks == self.segmentation_shape[0]
                && self.embedding_receipt.local_slots == self.segmentation_shape[2],
            "embedding receipt shape does not match entries"
        );
        ensure!(
            self.embedding_receipt
                .clean_mask
                .saturating_add(self.embedding_receipt.full_mask_fallback)
                == entries.saturating_sub(inactive),
            "embedding receipt mask counts do not match eligible entries"
        );
        Ok(())
    }
}

fn checked_segmentation_value_count(shape: [usize; 3]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, extent| {
        ensure!(
            *extent <= MAX_EMBEDDING_STAGE_VALUES,
            "embedding-stage shape extent is too large"
        );
        count
            .checked_mul(*extent)
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding-stage shape overflow"))
    })
}

fn packed_mask_byte_count(value_count: usize) -> Result<usize> {
    value_count
        .checked_add(7)
        .map(|count| count / 8)
        .ok_or_else(|| color_eyre::eyre::eyre!("embedding-stage mask byte count overflow"))
}

fn ensure_mask_padding_is_zero(bytes: &[u8], value_count: usize) -> Result<()> {
    if let Some(last) = bytes.last() {
        let used_bits = value_count % 8;
        if used_bits != 0 {
            ensure!(
                last & !((1_u8 << used_bits) - 1) == 0,
                "embedding-stage segmentation mask has nonzero padding bits"
            );
        }
    }
    Ok(())
}

fn decode_canonical_base64(value: &str, expected_bytes: usize, label: &str) -> Result<Vec<u8>> {
    let expected_encoded_bytes = encoded_base64_len(expected_bytes)?;
    ensure!(
        value.len() == expected_encoded_bytes,
        "{label} has {} encoded bytes, expected {expected_encoded_bytes}",
        value.len()
    );
    let decoded = STANDARD
        .decode(value)
        .wrap_err_with(|| format!("{label} is not valid standard base64"))?;
    ensure!(
        decoded.len() == expected_bytes,
        "{label} decodes to {} bytes, expected {expected_bytes}",
        decoded.len()
    );
    ensure!(
        STANDARD.encode(&decoded) == value,
        "{label} is not canonical standard-padded base64"
    );
    Ok(decoded)
}

fn decode_bounded_canonical_base64(
    value: &str,
    maximum_bytes: usize,
    label: &str,
) -> Result<Vec<u8>> {
    ensure!(
        value.len() <= encoded_base64_len(maximum_bytes)?,
        "{label} exceeds the encoded size bound"
    );
    let decoded = STANDARD
        .decode(value)
        .wrap_err_with(|| format!("{label} is not valid standard base64"))?;
    ensure!(
        decoded.len() <= maximum_bytes,
        "{label} exceeds the decoded size bound"
    );
    ensure!(
        STANDARD.encode(&decoded) == value,
        "{label} is not canonical standard-padded base64"
    );
    Ok(decoded)
}

fn encoded_base64_len(decoded_bytes: usize) -> Result<usize> {
    decoded_bytes
        .checked_add(2)
        .and_then(|bytes| bytes.checked_div(3))
        .and_then(|groups| groups.checked_mul(4))
        .ok_or_else(|| color_eyre::eyre::eyre!("base64 encoded length overflow"))
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimedSegment {
    pub start_seconds: f64,
    pub duration_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerTrack {
    pub speaker_id: String,
    pub segments: Vec<TimedSegment>,
    pub total_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerTracks {
    pub schema_version: u32,
    pub recording_id: String,
    pub geometry: GeometryReceipt,
    pub tracks: Vec<SpeakerTrack>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SystemKind {
    UnchangedSpeakrs,
    FrozenPythonWavlm,
    HybridWavlmSpeakrs,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SystemRecord {
    pub recording_id: String,
    pub source: String,
    pub domain: String,
    pub parent_group: String,
    pub audio_sha256: Sha256Digest,
    pub reference_sha256: Sha256Digest,
    pub uem_sha256: Sha256Digest,
    pub hypothesis: ArtifactState,
    pub speaker_tracks: ArtifactState,
    pub runtime_seconds: MeasurementState<f64>,
    pub peak_memory_bytes: MeasurementState<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreDocument {
    pub schema_version: u32,
    pub system: SystemKind,
    pub join: JoinIdentity,
    pub per_record: Vec<RecordScore>,
    pub source_rows: Vec<AggregateScoreRow>,
    pub domain_rows: Vec<AggregateScoreRow>,
    pub parent_rows: Vec<AggregateScoreRow>,
    pub hierarchical_equal_domain: Option<AggregateScoreRow>,
    pub pooled: Option<AggregateScoreRow>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecordScore {
    pub recipe_id: String,
    pub recording_id: String,
    pub hypothesis_sha256: Sha256Digest,
    pub miss_seconds: f64,
    pub false_alarm_seconds: f64,
    pub confusion_seconds: f64,
    pub reference_speaker_seconds: f64,
    pub der: Option<f64>,
    pub jer: Option<f64>,
    pub reference_speaker_count: usize,
    pub predicted_speaker_count: usize,
    pub fragmentation: Option<f64>,
    pub short_speaker_retention: Option<f64>,
    pub mixed_mask_fallback: Option<f64>,
    pub runtime_seconds: MeasurementState<f64>,
    pub peak_memory_bytes: MeasurementState<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AggregateScoreRow {
    pub key: String,
    pub recording_ids: Vec<String>,
    pub reference_speaker_seconds: Option<f64>,
    pub der: Option<f64>,
    pub miss: Option<f64>,
    pub false_alarm: Option<f64>,
    pub confusion: Option<f64>,
    pub jer: Option<f64>,
    pub uncertainty: Option<f64>,
}

impl ScoreDocument {
    pub(crate) fn validate_for(&self, spec: &BridgeSpec) -> Result<()> {
        ensure!(
            self.schema_version == SCORE_SCHEMA_VERSION,
            "unsupported score schema {}; expected {SCORE_SCHEMA_VERSION}",
            self.schema_version
        );
        let expected_ids = spec
            .recordings
            .iter()
            .map(|recording| recording.id.clone())
            .collect::<BTreeSet<_>>();
        validate_record_scores(&expected_ids, &self.per_record)?;
        let scores_by_id = self
            .per_record
            .iter()
            .map(|score| (score.recording_id.as_str(), score))
            .collect::<BTreeMap<_, _>>();
        validate_aggregate_rows(
            "source",
            &self.source_rows,
            &grouped_recordings(spec, GroupField::Source),
            &scores_by_id,
        )?;
        validate_aggregate_rows(
            "domain",
            &self.domain_rows,
            &grouped_recordings(spec, GroupField::Domain),
            &scores_by_id,
        )?;
        validate_aggregate_rows(
            "parent",
            &self.parent_rows,
            &grouped_recordings(spec, GroupField::Parent),
            &scores_by_id,
        )?;
        let equal_domain = self
            .hierarchical_equal_domain
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("missing equal-domain aggregate"))?;
        ensure!(
            equal_domain.key == "equal_domain",
            "equal-domain aggregate has wrong key"
        );
        let all_components = DerComponents::from_records(&expected_ids, &scores_by_id)?;
        let equal_domain_der = if spec.join.aggregation.equal_domain_average {
            mean_required_metric(&self.domain_rows, |row| row.der, "domain DER")?
        } else {
            all_components.der("equal-domain DER")?
        };
        validate_aggregate_row(
            "equal-domain",
            equal_domain,
            &expected_ids,
            all_components,
            equal_domain_der,
        )?;
        let pooled = self
            .pooled
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("missing pooled aggregate"))?;
        ensure!(pooled.key == "pooled", "pooled aggregate has wrong key");
        validate_aggregate_row(
            "pooled",
            pooled,
            &expected_ids,
            all_components,
            all_components.der("pooled DER")?,
        )?;
        Ok(())
    }
}

fn validate_record_scores(expected: &BTreeSet<String>, scores: &[RecordScore]) -> Result<()> {
    ensure!(
        scores.len() == expected.len(),
        "score record count does not match membership"
    );
    let mut seen = BTreeSet::new();
    let mut recipe_ids = BTreeSet::new();
    for score in scores {
        validate_id("score.recipe_id", &score.recipe_id)?;
        recipe_ids.insert(&score.recipe_id);
        ensure!(
            expected.contains(&score.recording_id),
            "score row is outside membership: {}",
            score.recording_id
        );
        ensure!(
            seen.insert(&score.recording_id),
            "duplicate score row {}",
            score.recording_id
        );
        validate_nonnegative(score.miss_seconds, "miss_seconds")?;
        validate_nonnegative(score.false_alarm_seconds, "false_alarm_seconds")?;
        validate_nonnegative(score.confusion_seconds, "confusion_seconds")?;
        ensure!(
            score.reference_speaker_seconds.is_finite() && score.reference_speaker_seconds > 0.0,
            "record {} has an invalid DER denominator",
            score.recording_id
        );
        ensure!(
            score.reference_speaker_count > 0,
            "record {} has no reference speakers",
            score.recording_id
        );
        let der = required_metric(score.der, "der")?;
        let expected_der = checked_der(
            score.miss_seconds,
            score.false_alarm_seconds,
            score.confusion_seconds,
            score.reference_speaker_seconds,
            &format!("record {} DER", score.recording_id),
        )?;
        ensure_metric_matches(
            &format!("record {} DER", score.recording_id),
            der,
            expected_der,
        )?;
        required_metric(score.jer, "jer")?;
        required_metric(score.fragmentation, "fragmentation")?;
        let retention = required_metric(score.short_speaker_retention, "short_speaker_retention")?;
        ensure!(retention <= 1.0, "short-speaker retention is above one");
        let fallback = required_metric(score.mixed_mask_fallback, "mixed_mask_fallback")?;
        ensure!(fallback <= 1.0, "mixed-mask fallback is above one");
        validate_runtime(&score.runtime_seconds, &score.recording_id)?;
        validate_peak_memory(&score.peak_memory_bytes, &score.recording_id)?;
    }
    ensure!(
        seen.len() == expected.len(),
        "score membership is incomplete"
    );
    ensure!(
        recipe_ids.len() == 1,
        "score rows do not contain a recipe identity"
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum GroupField {
    Source,
    Domain,
    Parent,
}

fn grouped_recordings(spec: &BridgeSpec, field: GroupField) -> BTreeMap<String, BTreeSet<String>> {
    let mut groups = BTreeMap::new();
    for recording in &spec.recordings {
        let key = match field {
            GroupField::Source => &recording.source,
            GroupField::Domain => &recording.domain,
            GroupField::Parent => &recording.parent_group,
        };
        groups
            .entry(key.clone())
            .or_insert_with(BTreeSet::new)
            .insert(recording.id.clone());
    }
    groups
}

fn validate_aggregate_rows(
    label: &str,
    rows: &[AggregateScoreRow],
    expected: &BTreeMap<String, BTreeSet<String>>,
    scores_by_id: &BTreeMap<&str, &RecordScore>,
) -> Result<()> {
    ensure!(
        rows.len() == expected.len(),
        "{label} aggregate coverage is incomplete"
    );
    let mut seen = BTreeSet::new();
    for row in rows {
        let expected_ids = expected
            .get(&row.key)
            .ok_or_else(|| color_eyre::eyre::eyre!("unknown {label} aggregate {}", row.key))?;
        ensure!(
            seen.insert(&row.key),
            "duplicate {label} aggregate {}",
            row.key
        );
        let components = DerComponents::from_records(expected_ids, scores_by_id)?;
        validate_aggregate_row(
            label,
            row,
            expected_ids,
            components,
            components.der(&format!("{label} aggregate {} DER", row.key))?,
        )?;
    }
    ensure!(
        seen.len() == expected.len(),
        "{label} aggregate coverage is incomplete"
    );
    Ok(())
}

fn validate_aggregate_row(
    label: &str,
    row: &AggregateScoreRow,
    expected_ids: &BTreeSet<String>,
    expected_components: DerComponents,
    expected_der: f64,
) -> Result<()> {
    ensure!(!row.key.is_empty(), "{label} aggregate key is empty");
    ensure!(
        row.recording_ids.len() == expected_ids.len(),
        "{label} aggregate {} has incomplete recording coverage",
        row.key
    );
    let actual_ids = row.recording_ids.iter().collect::<BTreeSet<_>>();
    ensure!(
        actual_ids.len() == row.recording_ids.len()
            && actual_ids.iter().all(|id| expected_ids.contains(*id)),
        "{label} aggregate {} has invalid recording coverage",
        row.key
    );
    let reference =
        required_positive_metric(row.reference_speaker_seconds, "reference_speaker_seconds")?;
    let der = required_metric(row.der, "der")?;
    let miss = required_metric(row.miss, "miss")?;
    let false_alarm = required_metric(row.false_alarm, "false_alarm")?;
    let confusion = required_metric(row.confusion, "confusion")?;
    required_metric(row.jer, "jer")?;
    required_metric(row.uncertainty, "uncertainty")?;

    ensure_metric_matches(
        &format!("{label} aggregate {} reference speaker seconds", row.key),
        reference,
        expected_components.reference_speaker_seconds,
    )?;
    ensure_metric_matches(
        &format!("{label} aggregate {} miss", row.key),
        miss,
        expected_components.miss,
    )?;
    ensure_metric_matches(
        &format!("{label} aggregate {} false alarm", row.key),
        false_alarm,
        expected_components.false_alarm,
    )?;
    ensure_metric_matches(
        &format!("{label} aggregate {} confusion", row.key),
        confusion,
        expected_components.confusion,
    )?;
    ensure_metric_matches(
        &format!("{label} aggregate {} DER", row.key),
        der,
        expected_der,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
struct DerComponents {
    reference_speaker_seconds: f64,
    miss: f64,
    false_alarm: f64,
    confusion: f64,
}

impl DerComponents {
    fn from_records(
        recording_ids: &BTreeSet<String>,
        scores_by_id: &BTreeMap<&str, &RecordScore>,
    ) -> Result<Self> {
        let mut components = Self {
            reference_speaker_seconds: 0.0,
            miss: 0.0,
            false_alarm: 0.0,
            confusion: 0.0,
        };
        for recording_id in recording_ids {
            let score = scores_by_id.get(recording_id.as_str()).ok_or_else(|| {
                color_eyre::eyre::eyre!("aggregate references missing record {recording_id}")
            })?;
            components.reference_speaker_seconds = checked_sum(
                components.reference_speaker_seconds,
                score.reference_speaker_seconds,
                "aggregate reference speaker seconds",
            )?;
            components.miss = checked_sum(components.miss, score.miss_seconds, "aggregate miss")?;
            components.false_alarm = checked_sum(
                components.false_alarm,
                score.false_alarm_seconds,
                "aggregate false alarm",
            )?;
            components.confusion = checked_sum(
                components.confusion,
                score.confusion_seconds,
                "aggregate confusion",
            )?;
        }

        Ok(components)
    }

    fn der(self, label: &str) -> Result<f64> {
        checked_der(
            self.miss,
            self.false_alarm,
            self.confusion,
            self.reference_speaker_seconds,
            label,
        )
    }
}

fn mean_required_metric(
    rows: &[AggregateScoreRow],
    value: impl Fn(&AggregateScoreRow) -> Option<f64>,
    field: &str,
) -> Result<f64> {
    ensure!(!rows.is_empty(), "cannot average empty {field} rows");
    let sum = rows.iter().try_fold(0.0, |sum, row| {
        let value = required_metric(value(row), field)?;
        checked_sum(sum, value, &format!("{field} sum"))
    })?;
    let mean = sum / rows.len() as f64;
    ensure!(mean.is_finite(), "{field} mean is not finite");

    Ok(mean)
}

fn ensure_metric_matches(label: &str, actual: f64, expected: f64) -> Result<()> {
    const ABSOLUTE_TOLERANCE: f64 = 1e-9;
    const RELATIVE_TOLERANCE: f64 = 1e-9;

    ensure!(expected.is_finite(), "computed {label} is not finite");
    let tolerance = ABSOLUTE_TOLERANCE.max(RELATIVE_TOLERANCE * expected.abs());
    ensure!(
        (actual - expected).abs() <= tolerance,
        "{label} is {actual}, expected {expected} within {tolerance}"
    );
    Ok(())
}

fn checked_der(
    miss: f64,
    false_alarm: f64,
    confusion: f64,
    reference_speaker_seconds: f64,
    label: &str,
) -> Result<f64> {
    let errors = checked_sum(miss, false_alarm, &format!("{label} error sum"))?;
    let errors = checked_sum(errors, confusion, &format!("{label} error sum"))?;
    let der = errors / reference_speaker_seconds;
    ensure!(der.is_finite(), "computed {label} is not finite");

    Ok(der)
}

fn checked_sum(left: f64, right: f64, label: &str) -> Result<f64> {
    let sum = left + right;
    ensure!(sum.is_finite(), "computed {label} is not finite");

    Ok(sum)
}

fn required_metric(value: Option<f64>, field: &str) -> Result<f64> {
    let value = value.ok_or_else(|| color_eyre::eyre::eyre!("missing {field}"))?;
    validate_nonnegative(value, field)?;
    Ok(value)
}

fn required_positive_metric(value: Option<f64>, field: &str) -> Result<f64> {
    let value = required_metric(value, field)?;
    ensure!(value > 0.0, "{field} must be positive");
    Ok(value)
}

fn validate_nonnegative(value: f64, field: &str) -> Result<()> {
    ensure!(
        value.is_finite() && value >= 0.0,
        "{field} must be finite and nonnegative"
    );
    Ok(())
}

pub(crate) fn validate_runtime(value: &MeasurementState<f64>, recording_id: &str) -> Result<()> {
    match value {
        MeasurementState::Available { value } => {
            ensure!(
                value.is_finite() && *value >= 0.0,
                "record {recording_id} has invalid runtime evidence"
            );
            Ok(())
        }
        MeasurementState::Unavailable { .. } => {
            bail!("record {recording_id} has missing runtime evidence")
        }
    }
}

pub(crate) fn validate_peak_memory(
    value: &MeasurementState<u64>,
    recording_id: &str,
) -> Result<()> {
    match value {
        MeasurementState::Available { value } => {
            ensure!(
                *value > 0,
                "record {recording_id} has invalid peak-memory evidence"
            );
            Ok(())
        }
        MeasurementState::Unavailable { .. } => {
            bail!("record {recording_id} has missing peak-memory evidence")
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ScoreDocumentState {
    Available { document: Box<ScoreDocument> },
    Unavailable { reason: AvailabilityReason },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SystemManifest {
    pub schema_version: u32,
    pub experiment_id: String,
    pub system: SystemKind,
    pub join: JoinIdentity,
    pub recipe_id: String,
    pub records: Vec<SystemRecord>,
    pub score_document: ScoreDocumentState,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecording {
    pub recording_id: String,
    pub recipe_id: String,
    pub cache_key: Sha256Digest,
    pub cache_reused: bool,
    pub embedding_cache_reused: bool,
    pub stage_receipts: Vec<ArtifactRef>,
    pub hypothesis: ArtifactRef,
    pub speaker_tracks: ArtifactRef,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunDocument {
    pub schema_version: u32,
    pub experiment_id: String,
    pub spec_sha256: Sha256Digest,
    pub runtime: RuntimeIdentity,
    pub recipes: Vec<String>,
    pub records: Vec<RunRecording>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationDocument {
    pub schema_version: u32,
    pub bundle_path: PathBuf,
    pub bundle_id: Sha256Digest,
    pub manifest_sha256: Sha256Digest,
    pub audio_path: PathBuf,
    pub sample_rate: u32,
    pub sample_count: usize,
    pub waveform_sha256: Sha256Digest,
    pub geometry: GeometryReceipt,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JoinedRecord {
    pub recording_id: String,
    pub source: String,
    pub domain: String,
    pub parent_group: String,
    pub systems: Vec<JoinedSystemRecord>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JoinedSystemRecord {
    pub system: SystemKind,
    pub hypothesis: ArtifactState,
    pub speaker_tracks: ArtifactState,
    pub runtime_seconds: MeasurementState<f64>,
    pub peak_memory_bytes: MeasurementState<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReportStatus {
    Complete,
    Incomplete { missing: Vec<ReportMissing> },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReportMissing {
    pub system: SystemKind,
    pub reason: AvailabilityReason,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReportDocument {
    pub schema_version: u32,
    pub experiment_id: String,
    pub join: JoinIdentity,
    pub systems: Vec<SystemManifest>,
    pub records: Vec<JoinedRecord>,
    pub status: ReportStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score_spec() -> BridgeSpec {
        let membership_sha256 = digest_bytes(b"membership");
        let join = JoinIdentity {
            membership_sha256: membership_sha256.clone(),
            audio_manifest_sha256: digest_bytes(b"audio"),
            reference_manifest_sha256: digest_bytes(b"reference"),
            uem_manifest_sha256: digest_bytes(b"uem"),
            scorer: ScorerIdentity {
                implementation: "pyannote.metrics".into(),
                version: "4.0.0".into(),
                collar_seconds: 0.0,
                overlap: OverlapPolicy::Included,
                speaker_count: SpeakerCountPolicy::Automatic,
                config_sha256: digest_bytes(b"scorer"),
            },
            aggregation: AggregationIdentity {
                id: "aggregation".into(),
                revision: "v1".into(),
                equal_domain_average: true,
                pooled_secondary: true,
            },
            qualification: QualificationIdentity {
                id: "qualification".into(),
                revision: "v1".into(),
                sentinel_membership_sha256: None,
                monitor_membership_sha256: None,
                diagnostic_membership_sha256: None,
            },
        };
        BridgeSpec {
            schema_version: SPEC_SCHEMA_VERSION,
            experiment_id: "experiment".into(),
            membership: MembershipIdentity {
                id: "membership".into(),
                sha256: membership_sha256,
                role: MembershipRole::FixedProbe,
                recording_ids: vec!["r1".into(), "r2".into()],
            },
            join,
            runtime: RuntimeIdentity {
                mode: BridgeMode::Cpu,
                models_sha256: digest_bytes(b"models"),
                embedding_model: EmbeddingModelIdentity {
                    id: "embedding".into(),
                    revision: "v1".into(),
                    path: "embedding.onnx".into(),
                    sha256: digest_bytes(b"embedding"),
                    sidecar_sha256: digest_bytes(b"sidecar"),
                },
                plda: ModelIdentity {
                    id: "plda".into(),
                    revision: "v1".into(),
                    path: "plda".into(),
                    sha256: digest_bytes(b"plda"),
                },
                precision: Precision::Float32,
            },
            recordings: vec![
                RecordingSpec {
                    id: "r1".into(),
                    source: "source-a".into(),
                    domain: "domain-a".into(),
                    parent_group: "parent-a".into(),
                    audio: ArtifactInput {
                        path: "r1.wav".into(),
                        sha256: digest_bytes(b"r1-audio"),
                    },
                    bundle: BundleInput {
                        path: "r1-bundle".into(),
                        bundle_id: digest_bytes(b"r1-bundle"),
                        manifest_sha256: digest_bytes(b"r1-manifest"),
                    },
                    reference: ArtifactInput {
                        path: "r1.rttm".into(),
                        sha256: digest_bytes(b"r1-reference"),
                    },
                    uem: ArtifactInput {
                        path: "r1.uem".into(),
                        sha256: digest_bytes(b"r1-uem"),
                    },
                },
                RecordingSpec {
                    id: "r2".into(),
                    source: "source-b".into(),
                    domain: "domain-b".into(),
                    parent_group: "parent-b".into(),
                    audio: ArtifactInput {
                        path: "r2.wav".into(),
                        sha256: digest_bytes(b"r2-audio"),
                    },
                    bundle: BundleInput {
                        path: "r2-bundle".into(),
                        bundle_id: digest_bytes(b"r2-bundle"),
                        manifest_sha256: digest_bytes(b"r2-manifest"),
                    },
                    reference: ArtifactInput {
                        path: "r2.rttm".into(),
                        sha256: digest_bytes(b"r2-reference"),
                    },
                    uem: ArtifactInput {
                        path: "r2.uem".into(),
                        sha256: digest_bytes(b"r2-uem"),
                    },
                },
            ],
            recipes: Vec::new(),
        }
    }

    fn record_score(recording_id: &str) -> RecordScore {
        RecordScore {
            recipe_id: "recipe".into(),
            recording_id: recording_id.into(),
            hypothesis_sha256: digest_bytes(recording_id.as_bytes()),
            miss_seconds: 1.0,
            false_alarm_seconds: 2.0,
            confusion_seconds: 3.0,
            reference_speaker_seconds: 100.0,
            der: Some(0.06),
            jer: Some(0.1),
            reference_speaker_count: 2,
            predicted_speaker_count: 2,
            fragmentation: Some(0.2),
            short_speaker_retention: Some(0.8),
            mixed_mask_fallback: Some(0.1),
            runtime_seconds: MeasurementState::Available { value: 1.0 },
            peak_memory_bytes: MeasurementState::Available { value: 1_024 },
        }
    }

    fn aggregate(key: &str, recording_id: &str) -> AggregateScoreRow {
        AggregateScoreRow {
            key: key.into(),
            recording_ids: vec![recording_id.into()],
            reference_speaker_seconds: Some(100.0),
            der: Some(0.06),
            miss: Some(1.0),
            false_alarm: Some(2.0),
            confusion: Some(3.0),
            jer: Some(0.1),
            uncertainty: Some(0.01),
        }
    }

    fn valid_score_document() -> (BridgeSpec, ScoreDocument) {
        let spec = score_spec();
        let document = ScoreDocument {
            schema_version: SCORE_SCHEMA_VERSION,
            system: SystemKind::HybridWavlmSpeakrs,
            join: spec.join.clone(),
            per_record: vec![record_score("r1"), record_score("r2")],
            source_rows: vec![aggregate("source-a", "r1"), aggregate("source-b", "r2")],
            domain_rows: vec![aggregate("domain-a", "r1"), aggregate("domain-b", "r2")],
            parent_rows: vec![aggregate("parent-a", "r1"), aggregate("parent-b", "r2")],
            hierarchical_equal_domain: Some(AggregateScoreRow {
                key: "equal_domain".into(),
                recording_ids: vec!["r1".into(), "r2".into()],
                reference_speaker_seconds: Some(200.0),
                der: Some(0.06),
                miss: Some(2.0),
                false_alarm: Some(4.0),
                confusion: Some(6.0),
                jer: Some(0.1),
                uncertainty: Some(0.01),
            }),
            pooled: Some(AggregateScoreRow {
                key: "pooled".into(),
                recording_ids: vec!["r1".into(), "r2".into()],
                reference_speaker_seconds: Some(200.0),
                der: Some(0.06),
                miss: Some(2.0),
                false_alarm: Some(4.0),
                confusion: Some(6.0),
                jer: Some(0.1),
                uncertainty: Some(0.01),
            }),
        };
        (spec, document)
    }

    fn valid_embedding_stage_document() -> EmbeddingStageDocument {
        EmbeddingStageDocument {
            schema_version: EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: digest_bytes(b"embedding-stage"),
            geometry: GeometryReceipt {
                sample_rate: 16_000,
                sample_count: 0,
                window_samples: 1,
                step_samples: 1,
                chunks: Vec::new(),
                frame_grid: FrameGridReceipt {
                    frame_count: 0,
                    origin: RationalReceipt {
                        numerator: 0,
                        denominator: 1,
                    },
                    step: RationalReceipt {
                        numerator: 1,
                        denominator: 1,
                    },
                    support: RationalReceipt {
                        numerator: 1,
                        denominator: 1,
                    },
                },
                aggregate_grid: FrameGridReceipt {
                    frame_count: 0,
                    origin: RationalReceipt {
                        numerator: 0,
                        denominator: 1,
                    },
                    step: RationalReceipt {
                        numerator: 1,
                        denominator: 1,
                    },
                    support: RationalReceipt {
                        numerator: 1,
                        denominator: 1,
                    },
                },
                start_frames: Vec::new(),
                output_frames: 0,
                output_extent_start_samples: 0,
                output_extent_end_samples: 0,
                output_extent_policy:
                    speakrs::imported_segmentation::OutputExtentPolicy::AggregateGrid,
            },
            segmentation_shape: [1, 1, 1],
            segmentation_values: PackedSegmentationMask::try_from_values([1, 1, 1], &[1.0])
                .unwrap(),
            entries: vec![EmbeddingStageEntry::Available {
                values: Box::new(
                    EncodedEmbeddingVector::try_from_values(&[0.0; IMPORTED_EMBEDDING_WIDTH])
                        .unwrap(),
                ),
            }],
            embedding_receipt: AvailabilityCounts {
                chunks: 1,
                local_slots: 1,
                available: 1,
                clean_mask: 1,
                ..AvailabilityCounts::default()
            },
        }
    }

    #[test]
    fn bridge_mode_has_stable_json_names() {
        assert_eq!(serde_json::to_string(&BridgeMode::Cpu).unwrap(), "\"cpu\"");
    }

    #[test]
    fn model_tree_digest_is_order_independent() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("b"), b"b").unwrap();
        fs::write(directory.path().join("a"), b"a").unwrap();
        let first = digest_tree(directory.path()).unwrap();
        fs::remove_file(directory.path().join("a")).unwrap();
        fs::write(directory.path().join("a"), b"a").unwrap();
        assert_eq!(first, digest_tree(directory.path()).unwrap());
    }

    #[test]
    fn model_tree_digest_matches_cross_language_vector() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();
        fs::write(directory.path().join("a.txt"), b"alpha").unwrap();
        fs::write(directory.path().join("nested/b.bin"), [0, 1, 2, 255]).unwrap();

        // this vector matches the Python bridge tree digest contract
        assert_eq!(
            digest_tree(directory.path()).unwrap().as_str(),
            "48c74f4785da606a9f2147aa1d62640e8d2f9008b8101281983344624ca64e18"
        );
    }

    #[test]
    fn previous_score_schema_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.schema_version = SCORE_SCHEMA_VERSION - 1;
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn empty_score_aggregate_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.source_rows.clear();
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn nonfinite_score_metric_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.per_record[0].der = Some(f64::NAN);
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn record_der_must_match_its_additive_components() {
        let (spec, mut document) = valid_score_document();
        document.per_record[0].der = Some(0.0);

        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn record_der_rejects_overflowing_arithmetic() {
        let (spec, mut document) = valid_score_document();
        document.per_record[0].miss_seconds = 1e308;
        document.per_record[0].false_alarm_seconds = 1e308;
        document.per_record[0].confusion_seconds = 0.0;
        document.per_record[0].der = Some(1.0);

        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn aggregate_components_must_match_their_member_records() {
        let (spec, mut document) = valid_score_document();
        document.source_rows[0].miss = Some(0.0);

        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn aggregate_components_reject_overflowing_totals() {
        let (spec, mut document) = valid_score_document();
        for score in &mut document.per_record {
            score.miss_seconds = 1e308;
            score.false_alarm_seconds = 0.0;
            score.confusion_seconds = 0.0;
            score.reference_speaker_seconds = 1e308;
            score.der = Some(1.0);
        }
        for rows in [
            &mut document.source_rows,
            &mut document.domain_rows,
            &mut document.parent_rows,
        ] {
            for row in rows {
                row.reference_speaker_seconds = Some(1e308);
                row.der = Some(1.0);
                row.miss = Some(1e308);
                row.false_alarm = Some(0.0);
                row.confusion = Some(0.0);
            }
        }

        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn equal_domain_der_must_use_equal_domain_weighting() {
        let (spec, mut document) = valid_score_document();
        document.per_record[1].reference_speaker_seconds = 200.0;
        document.per_record[1].der = Some(0.03);
        for rows in [
            &mut document.source_rows,
            &mut document.domain_rows,
            &mut document.parent_rows,
        ] {
            rows[1].reference_speaker_seconds = Some(200.0);
            rows[1].der = Some(0.03);
        }
        let equal_domain = document.hierarchical_equal_domain.as_mut().unwrap();
        equal_domain.reference_speaker_seconds = Some(300.0);
        equal_domain.der = Some(0.045);
        let pooled = document.pooled.as_mut().unwrap();
        pooled.reference_speaker_seconds = Some(300.0);
        pooled.der = Some(0.04);

        assert!(document.validate_for(&spec).is_ok());

        document.hierarchical_equal_domain.as_mut().unwrap().der = Some(0.04);
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn score_rows_require_one_recipe_identity() {
        let (spec, mut document) = valid_score_document();
        document.per_record[1].recipe_id = "other".into();
        assert!(document.validate_for(&spec).is_err());

        let (spec, mut document) = valid_score_document();
        document.per_record[0].recipe_id.clear();
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn missing_score_component_or_denominator_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.source_rows[0].confusion = None;
        assert!(document.validate_for(&spec).is_err());

        let (spec, mut document) = valid_score_document();
        document.domain_rows[0].reference_speaker_seconds = None;
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn missing_peak_memory_evidence_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.per_record[0].peak_memory_bytes = MeasurementState::Unavailable {
            reason: AvailabilityReason::NotMeasured,
        };
        assert!(document.validate_for(&spec).is_err());
    }

    #[test]
    fn embedding_snapshot_rejects_truncated_masks() {
        let mut document = valid_embedding_stage_document();
        document.segmentation_values = PackedSegmentationMask::empty();
        assert!(document.validate().is_err());
    }

    #[test]
    fn embedding_snapshot_round_trips_packed_masks_and_exact_vector_bits() {
        let mut document = valid_embedding_stage_document();
        let shape = [1, 9, 1];
        let mask_values = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        document.segmentation_shape = shape;
        document.segmentation_values =
            PackedSegmentationMask::try_from_values(shape, &mask_values).unwrap();
        assert_eq!(
            serde_json::to_string(&document.segmentation_values).unwrap(),
            "\"gQE=\""
        );
        let vector = (0..IMPORTED_EMBEDDING_WIDTH)
            .map(|index| f32::from_bits(0x3f80_0000 + index as u32))
            .collect::<Vec<_>>();
        document.entries[0] = EmbeddingStageEntry::Available {
            values: Box::new(EncodedEmbeddingVector::try_from_values(&vector).unwrap()),
        };

        document.validate().unwrap();
        assert_eq!(document.decode_segmentation_values().unwrap(), mask_values);
        let decoded = match &document.entries[0] {
            EmbeddingStageEntry::Available { values } => values
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            _ => panic!("test entry is not available"),
        };
        assert_eq!(
            decoded,
            vector.into_iter().map(f32::to_bits).collect::<Vec<_>>()
        );
    }

    #[test]
    fn embedding_snapshot_serializes_compactly() {
        let document = valid_embedding_stage_document();
        let bytes = serde_json::to_vec(&document).unwrap();
        assert!(!bytes.contains(&b'\n'));
        assert!(
            serde_json::from_slice::<EmbeddingStageDocument>(&bytes)
                .unwrap()
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn embedding_snapshot_rejects_mask_padding_bits() {
        let mut document = valid_embedding_stage_document();
        document.segmentation_values = PackedSegmentationMask(vec![0b0000_0011]);
        let error = document.validate().unwrap_err();
        assert!(error.to_string().contains("padding bits"));
    }

    #[test]
    fn embedding_snapshot_rejects_noncanonical_base64() {
        let encoded = serde_json::to_string("/x==").unwrap();
        assert!(serde_json::from_str::<PackedSegmentationMask>(&encoded).is_err());
    }

    #[test]
    fn embedding_snapshot_rejects_nonfinite_embedding_bits() {
        let mut bytes = vec![0_u8; IMPORTED_EMBEDDING_WIDTH * std::mem::size_of::<f32>()];
        bytes[..std::mem::size_of::<f32>()].copy_from_slice(&f32::NAN.to_le_bytes());
        let encoded = serde_json::to_string(&STANDARD.encode(bytes)).unwrap();
        assert!(serde_json::from_str::<EncodedEmbeddingVector>(&encoded).is_err());
    }

    #[test]
    fn embedding_snapshot_rejects_wrong_vector_width() {
        let values: Vec<f32> = vec![0.0; IMPORTED_EMBEDDING_WIDTH - 1];
        let encoded = STANDARD.encode(
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        assert!(
            serde_json::from_str::<EncodedEmbeddingVector>(
                &serde_json::to_string(&encoded).unwrap()
            )
            .is_err()
        );
    }

    #[test]
    fn embedding_stage_entry_rejects_impossible_json_shapes() {
        assert!(serde_json::from_str::<EmbeddingStageEntry>(r#"{"state":"available"}"#).is_err());
        assert!(
            serde_json::from_str::<EmbeddingStageEntry>(
                r#"{"state":"inactive","reason":"no_activity","values":"AA=="}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<EmbeddingStageEntry>(
                r#"{"state":"inactive","reason":"no_activity","extra":true}"#
            )
            .is_err()
        );
    }

    #[test]
    fn embedding_snapshot_rejects_available_count_mismatch() {
        let mut document = valid_embedding_stage_document();
        document.embedding_receipt.available = 0;
        assert!(document.validate().is_err());
    }
}
