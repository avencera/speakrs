use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use clap::ValueEnum;
use color_eyre::eyre::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as ShaDigest, Sha256};

pub use speakrs::imported_segmentation::Sha256Digest;

pub const SPEC_SCHEMA_VERSION: u32 = 1;
pub const VALIDATION_SCHEMA_VERSION: u32 = 1;
pub const RUN_SCHEMA_VERSION: u32 = 1;
pub const SYSTEM_SCHEMA_VERSION: u32 = 1;
pub const REPORT_SCHEMA_VERSION: u32 = 1;
pub const SCORE_SCHEMA_VERSION: u32 = 1;
pub const RECEIPT_SCHEMA_VERSION: u32 = 1;
pub const CACHE_SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_CACHE_SCHEMA_VERSION: u32 = 1;
pub const EMBEDDING_STAGE_SCHEMA_VERSION: u32 = 1;
pub const MAX_EMBEDDING_STAGE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_EMBEDDING_STAGE_VALUES: usize = 64 * 1024 * 1024;
pub const MAX_EMBEDDING_STAGE_ENTRIES: usize = 4 * 1024 * 1024;
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
    Ok(digest_bytes(&digest.finalize()))
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
    Model,
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

/// A serialized typed availability state for one embedding slot
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
pub enum EmbeddingSlotAvailability {
    Available,
    Inactive { reason: EmbeddingInactiveReason },
    InferenceFailed { reason: EmbeddingFailureReason },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInactiveReason {
    NoActivity,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingFailureReason {
    ModelExecution,
    InvalidOutput,
    LegacyUnavailable,
}

/// One serialized embedding-stage slot without a sentinel vector
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingStageEntry {
    pub availability: EmbeddingSlotAvailability,
    pub values: Option<Vec<f32>>,
}

/// Immutable serialized output of bundle decode and per-speaker embedding
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingStageDocument {
    pub schema_version: u32,
    pub recording_id: String,
    pub stage_key: Sha256Digest,
    pub geometry: GeometryReceipt,
    pub segmentation_shape: [usize; 3],
    pub segmentation_values: Vec<f32>,
    pub entries: Vec<EmbeddingStageEntry>,
    pub embedding_receipt: AvailabilityCounts,
}

impl EmbeddingStageDocument {
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
        let values = self
            .segmentation_shape
            .iter()
            .try_fold(1usize, |count, extent| {
                ensure!(
                    *extent <= MAX_EMBEDDING_STAGE_VALUES,
                    "embedding-stage shape extent is too large"
                );
                count
                    .checked_mul(*extent)
                    .ok_or_else(|| color_eyre::eyre::eyre!("embedding-stage shape overflow"))
            })?;
        ensure!(
            values == self.segmentation_values.len(),
            "embedding-stage segmentation shape does not match values"
        );
        ensure!(
            values <= MAX_EMBEDDING_STAGE_VALUES,
            "embedding-stage segmentation values exceed bound"
        );
        let entries = self.segmentation_shape[0]
            .checked_mul(self.segmentation_shape[2])
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding-stage entry shape overflow"))?;
        ensure!(
            entries == self.entries.len() && entries <= MAX_EMBEDDING_STAGE_ENTRIES,
            "embedding-stage entries do not match shape or exceed bound"
        );
        ensure!(
            self.segmentation_values
                .iter()
                .all(|value| value.is_finite() && (*value == 0.0 || *value == 1.0)),
            "embedding-stage masks are not finite binary values"
        );
        let mut available = 0;
        let mut inactive = 0;
        let mut failed = 0;
        for entry in &self.entries {
            match (&entry.availability, &entry.values) {
                (EmbeddingSlotAvailability::Available, Some(values)) => {
                    available += 1;
                    ensure!(
                        !values.is_empty(),
                        "available embedding has an empty vector"
                    );
                    ensure!(
                        values.len() == IMPORTED_EMBEDDING_WIDTH
                            && values.iter().all(|value| value.is_finite()),
                        "available embedding has invalid values"
                    );
                }
                (EmbeddingSlotAvailability::Available, None) => {
                    bail!("available embedding has no vector")
                }
                (EmbeddingSlotAvailability::Inactive { .. }, None) => inactive += 1,
                (EmbeddingSlotAvailability::InferenceFailed { .. }, None) => failed += 1,
                (_, Some(_)) => bail!("unavailable embedding has a vector"),
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
    pub recording_id: String,
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
        validate_aggregate_rows(
            "source",
            &self.source_rows,
            &grouped_recordings(spec, GroupField::Source),
        )?;
        validate_aggregate_rows(
            "domain",
            &self.domain_rows,
            &grouped_recordings(spec, GroupField::Domain),
        )?;
        validate_aggregate_rows(
            "parent",
            &self.parent_rows,
            &grouped_recordings(spec, GroupField::Parent),
        )?;
        let equal_domain = self
            .hierarchical_equal_domain
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("missing equal-domain aggregate"))?;
        ensure!(
            equal_domain.key == "equal_domain",
            "equal-domain aggregate has wrong key"
        );
        validate_aggregate_row("equal-domain", equal_domain, &expected_ids)?;
        let pooled = self
            .pooled
            .as_ref()
            .ok_or_else(|| color_eyre::eyre::eyre!("missing pooled aggregate"))?;
        ensure!(pooled.key == "pooled", "pooled aggregate has wrong key");
        validate_aggregate_row("pooled", pooled, &expected_ids)?;
        Ok(())
    }
}

fn validate_record_scores(expected: &BTreeSet<String>, scores: &[RecordScore]) -> Result<()> {
    ensure!(
        scores.len() == expected.len(),
        "score record count does not match membership"
    );
    let mut seen = BTreeSet::new();
    for score in scores {
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
        required_metric(score.der, "der")?;
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
        validate_aggregate_row(label, row, expected_ids)?;
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
    required_positive_metric(row.reference_speaker_seconds, "reference_speaker_seconds")?;
    required_metric(row.der, "der")?;
    required_metric(row.miss, "miss")?;
    required_metric(row.false_alarm, "false_alarm")?;
    required_metric(row.confusion, "confusion")?;
    required_metric(row.jer, "jer")?;
    required_metric(row.uncertainty, "uncertainty")?;
    Ok(())
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
            recording_id: recording_id.into(),
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
            segmentation_values: vec![1.0],
            entries: vec![EmbeddingStageEntry {
                availability: EmbeddingSlotAvailability::Available,
                values: Some(vec![0.0; IMPORTED_EMBEDDING_WIDTH]),
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
    fn score_schema_zero_is_rejected() {
        let (spec, mut document) = valid_score_document();
        document.schema_version = 0;
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
    fn embedding_snapshot_rejects_nonbinary_masks() {
        let mut document = valid_embedding_stage_document();
        document.segmentation_values[0] = 0.5;
        assert!(document.validate().is_err());
    }

    #[test]
    fn embedding_snapshot_rejects_wrong_vector_width() {
        let mut document = valid_embedding_stage_document();
        document.entries[0].values = Some(vec![0.0; IMPORTED_EMBEDDING_WIDTH - 1]);
        assert!(document.validate().is_err());
    }

    #[test]
    fn embedding_snapshot_rejects_available_count_mismatch() {
        let mut document = valid_embedding_stage_document();
        document.embedding_receipt.available = 0;
        assert!(document.validate().is_err());
    }
}
