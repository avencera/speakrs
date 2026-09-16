//! Strict WavLM segmentation bundle contracts and bounded shard loading
//!
//! The bundle is an immutable hand-off from the Python model runner.  This
//! module validates its meaning and bytes without constructing a model
//! session.  The manifest is intentionally explicit: timing, powerset class
//! order, score representation, and every shard inventory item are part of
//! the contract

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Manifest format version admitted by this module
pub const FORMAT_VERSION: u32 = 1;

/// Stable schema identifier for imported segmentation bundles
pub const SCHEMA_ID: &str = "speakrs.segmentation_bundle";

/// Maximum manifest byte length admitted before JSON parsing
pub const MAX_MANIFEST_BYTES: usize = 1_048_576;

/// Maximum NPY header byte length admitted before tensor allocation
pub const MAX_NPY_HEADER_BYTES: usize = 16_384;

/// Maximum byte length of one uncompressed NPY shard
pub const MAX_SHARD_BYTES: u64 = 512 * 1024 * 1024;

/// Maximum number of float32 tensor elements in one imported bundle
pub const MAX_TENSOR_ELEMENTS: u64 = 64 * 1024 * 1024;

/// Maximum number of chunks in one imported bundle
pub const MAX_CHUNKS: u64 = 1_000_000;

/// Maximum number of frames in one chunk tensor
pub const MAX_FRAMES: u64 = 100_000;

/// Maximum number of powerset classes in one chunk tensor
pub const MAX_CLASSES: u64 = 4_096;

/// Maximum odd median window width admitted for imported score filtering
pub const MAX_MEDIAN_FILTER_WIDTH: usize = 255;

/// Maximum UTF-8 byte length of a relative tensor member path
pub const MAX_MEMBER_PATH_BYTES: usize = 512;

/// Absolute tolerance for probability bounds and normalization checks
pub const SCORE_PROBABILITY_TOLERANCE: f64 = 1.0e-4;

/// Absolute tolerance for log-probability log-sum-exp checks
pub const SCORE_LOG_PROBABILITY_TOLERANCE: f64 = 1.0e-4;

/// Maximum byte length of a free-form identity label
pub const MAX_IDENTITY_TEXT_BYTES: usize = 512;

/// Error returned when a segmentation manifest, shard, or publication is
/// malformed or inconsistent
#[derive(Debug, Error)]
pub enum SegmentationBundleError {
    /// A typed contract invariant was not satisfied
    #[error("invalid segmentation bundle: {0}")]
    Invalid(String),

    /// A filesystem operation failed while reading a bundle
    #[error("segmentation bundle I/O error: {0}")]
    Io(#[from] io::Error),

    /// Manifest JSON could not be decoded into its strict typed model
    #[error("segmentation manifest JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// A lowercase full SHA-256 digest
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Sha256Digest(String);

impl Sha256Digest {
    /// Creates a digest after checking its canonical lowercase form
    pub fn new(value: impl Into<String>) -> Result<Self, SegmentationBundleError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(SegmentationBundleError::Invalid(
                "SHA-256 digests must be 64 lowercase hexadecimal characters".to_owned(),
            ));
        }
        Ok(Self(value))
    }

    /// Computes a canonical lowercase digest from bytes
    pub fn digest(bytes: &[u8]) -> Self {
        let value = format!("{:x}", Sha256::digest(bytes));
        Self(value)
    }

    /// Returns the digest text
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Sha256Digest {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Sha256Digest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl TryFrom<&str> for Sha256Digest {
    type Error = SegmentationBundleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<String> for Sha256Digest {
    type Error = SegmentationBundleError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl Serialize for Sha256Digest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Sha256Digest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DigestVisitor;

        impl Visitor<'_> for DigestVisitor {
            type Value = Sha256Digest;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a lowercase 64-character SHA-256 digest")
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Sha256Digest::new(value).map_err(|error| E::custom(error.to_string()))
            }
        }

        deserializer.deserialize_str(DigestVisitor)
    }
}

/// Stable identity for one model, source, producer, or environment snapshot
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityReference {
    /// Stable identity label
    pub id: String,
    /// Source or artifact revision
    pub revision: String,
    /// Full content or snapshot digest
    pub sha256: Sha256Digest,
}

/// Identity of one named model component
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentIdentity {
    /// Component identity label
    pub id: String,
    /// Component name from the complete model graph
    pub name: String,
    /// Component revision
    pub revision: String,
    /// Full component digest
    pub sha256: Sha256Digest,
}

/// Floating-point precision used for the exported score tensor
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// Little-endian float32 values
    Float32,
}

/// All identities needed to reproduce the segmentation scores
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BundleIdentity {
    /// Content identity for the immutable bundle record
    pub bundle_id: Sha256Digest,
    /// Complete component identities in producer-declared order
    pub components: Vec<ComponentIdentity>,
    /// Configuration identity for the complete segmentation graph
    pub config: IdentityReference,
    /// Runtime environment snapshot identity
    pub environment: IdentityReference,
    /// Digest of the canonical manifest identity projection
    pub manifest_digest: Sha256Digest,
    /// Complete model artifact identity
    pub model: IdentityReference,
    /// Exported tensor precision
    pub precision: Precision,
    /// Producer implementation and source snapshot identity
    pub producer: IdentityReference,
    /// Source repository or package snapshot identity
    pub source: IdentityReference,
}

/// Channel selection policy used to produce canonical audio
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSelection {
    /// Select the first channel exactly
    First,
}

/// Downmix operation used to produce canonical mono audio
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Downmix {
    /// No mix is applied after selecting the first channel
    None,
}

/// Identity of the audio resampling operation
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResamplingIdentity {
    /// Algorithm or identity operation name
    pub algorithm: String,
    /// Stable operation identity
    pub id: String,
    /// Operation revision
    pub revision: String,
    /// Full operation configuration digest
    pub sha256: Sha256Digest,
}

/// Canonical waveform identity and recording lineage
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AudioIdentity {
    /// Exact channel selection operation
    pub channel_selection: ChannelSelection,
    /// Number of channels after canonicalization
    pub channels: u8,
    /// Exact downmix operation
    pub downmix: Downmix,
    /// Original recording identity
    pub original_recording_id: String,
    /// Optional parent recording identity
    pub parent_recording_id: Option<String>,
    /// Resampling operation identity
    pub resampling: ResamplingIdentity,
    /// Number of canonical mono samples
    pub sample_count: u64,
    /// Canonical waveform sample rate
    pub sample_rate: u32,
    /// SHA-256 digest of canonical waveform values
    pub waveform_sha256: Sha256Digest,
}

/// Closed representation of score values in each class vector
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreRepresentation {
    /// Unnormalized finite model logits
    Logits,
    /// Non-negative normalized probabilities
    Probabilities,
    /// Log probabilities with explicit negative infinity for zero mass
    LogProbabilities,
}

/// Explicit tie behavior for powerset hard decoding
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgmaxTie {
    /// Keep the first class when equal maxima occur
    First,
    /// Keep the last class when equal maxima occur
    Last,
}

/// Powerset head shape and complete class order
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentationHead {
    /// Hard-decoder tie policy
    pub argmax_tie: ArgmaxTie,
    /// Ordered mapping from class columns to local slot subsets
    pub class_to_slot_subsets: Vec<Vec<u32>>,
    /// Number of local speaker slots
    pub local_slots: u32,
    /// Maximum overlap represented by the head
    pub max_overlap: u32,
    /// Representation of each raw class score vector
    pub score_representation: ScoreRepresentation,
}

/// A reduced rational sample coordinate
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RationalSample {
    /// Positive reduced denominator
    pub denominator: i64,
    /// Signed numerator
    pub numerator: i64,
}

/// One frame grid, including exact support and origin
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrameGrid {
    /// Number of output frames for one padded window
    pub frame_count: u32,
    /// First frame origin in samples
    pub origin: RationalSample,
    /// Frame-to-frame sample step
    pub step: RationalSample,
    /// Receptive-field support in samples
    pub support: RationalSample,
}

/// Explicit chunk placement and valid/padded sample counts
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkGeometry {
    /// Zero-based chunk index
    pub index: u64,
    /// Number of right-padding samples
    pub padding_samples: u64,
    /// Chunk start in canonical audio samples
    pub start_samples: u64,
    /// Number of samples copied from canonical audio
    pub valid_samples: u64,
}

/// Output extent policy for frame-to-audio conversion
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputExtentPolicy {
    /// Preserve the extent of the explicit aggregate frame grid
    AggregateGrid,
    /// Expose a normalized view clipped to the canonical audio extent
    AudioExtent,
}

/// Canonical output sample extent
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OutputExtent {
    /// Exclusive end sample
    pub end_samples: u64,
    /// Inclusive start sample
    pub start_samples: u64,
}

/// Fixed-step window planning policy
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WindowPlanningKind {
    /// Regular fixed-step planning with explicit padded tail behavior
    RegularFixedStepV1,
}

/// Tail handling for the final planned window
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TailPolicy {
    /// Emit a final window and right-pad it to the declared length
    PadFinal,
}

/// Window and frame geometry for all imported chunks
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentationGeometry {
    /// Aggregation/output frame grid
    pub aggregate_grid: FrameGrid,
    /// Explicit ordered chunk placements
    pub chunks: Vec<ChunkGeometry>,
    /// Receptive-field frame grid
    pub frame_grid: FrameGrid,
    /// Canonical output extent
    pub output_extent: OutputExtent,
    /// Boundary policy for output extent
    pub output_extent_policy: OutputExtentPolicy,
    /// Pinned window planning rule
    pub window_planning: WindowPlanning,
    /// Number of samples in every padded input window
    pub window_samples: u64,
}

/// Window start planning parameters
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WindowPlanning {
    /// Closed planning rule
    pub kind: WindowPlanningKind,
    /// Step between planned window starts in samples
    pub step_samples: u64,
    /// Final-window tail behavior
    pub tail_policy: TailPolicy,
}

/// Exact score stage represented by v1 shards
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreStage {
    /// Joint powerset scores before decode, filtering, count, and embedding
    JointScoresPreDecode,
}

/// One bounded NPY shard inventory entry
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorShard {
    /// Exact file byte length
    pub bytes: u64,
    /// Exclusive chunk range end
    pub chunk_end: u64,
    /// Inclusive chunk range start
    pub chunk_start: u64,
    /// Safe relative path from the bundle root
    pub path: String,
    /// Full NPY file digest
    pub sha256: Sha256Digest,
    /// Rank-three tensor shape `[chunks, frames, classes]`
    pub shape: [u64; 3],
}

/// Shard inventory and score-stage identity
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorInventory {
    /// Exact stage represented by every shard
    pub score_stage: ScoreStage,
    /// Shards in chunk-range order
    pub shards: Vec<TensorShard>,
}

/// Median filter boundary behavior
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FilterBoundary {
    /// Reflect values at the signal boundary
    Reflect,
}

/// Decoder recipe identity and score interpretation
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecoderPolicy {
    /// Explicit hard-decoder tie policy
    pub argmax_tie: ArgmaxTie,
    /// Stable decoder recipe identity
    pub id: String,
    /// Score representation consumed by the decoder
    pub representation: ScoreRepresentation,
    /// Decoder recipe revision
    pub revision: String,
}

/// Optional median filtering recipe identity
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FilterPolicy {
    /// Boundary behavior when filtering is enabled
    pub boundary: FilterBoundary,
    /// Whether filtering is part of the declared reference recipe
    pub enabled: bool,
    /// Stable filter recipe identity
    pub id: String,
    /// Filter recipe revision
    pub revision: String,
    /// Odd median window width
    pub width: u32,
}

/// Count aggregation rounding policy
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CountRounding {
    /// Round half-way values to the nearest even integer
    NearestEven,
}

/// Speaker-count aggregation recipe identity
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CountPolicy {
    /// Stable count recipe identity
    pub id: String,
    /// Count recipe revision
    pub revision: String,
    /// Explicit rounding behavior
    pub rounding: CountRounding,
}

/// Mask interpolation policy for the qualified embedding path
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskInterpolation {
    /// PyTorch-compatible nearest interpolation
    Nearest,
}

/// Frontend identity admitted by the fixed embedding wrapper
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingFrontend {
    /// WeSpeaker 16 kHz filterbank frontend
    #[serde(rename = "wespeaker_fbank_v1")]
    WeSpeakerFbankV1,
}

/// Pooling identity admitted by the fixed embedding wrapper
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPooling {
    /// Masked mean and standard-deviation statistics pooling
    MaskedStatsPoolV1,
}

/// Numeric precision admitted by the fixed embedding wrapper
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPrecision {
    /// Float32 inputs, weights, and outputs
    Float32,
}

/// Embedding mask selection and resize recipe identity
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingPolicy {
    /// Exact fixed embedding model identity
    pub embedding_model: IdentityReference,
    /// SHA-256 digest of the exact embedding metadata sidecar bytes
    pub embedding_sidecar_sha256: Sha256Digest,
    /// Fixed embedding frontend identity
    pub frontend: EmbeddingFrontend,
    /// Stable embedding recipe identity
    pub id: String,
    /// Mask interpolation operation
    pub mask_interpolation: MaskInterpolation,
    /// Minimum selected samples for clean-mask eligibility
    pub min_num_samples: u64,
    /// Compatible PLDA artifact identity
    pub plda: IdentityReference,
    /// Fixed embedding pooling identity
    pub pooling: EmbeddingPooling,
    /// Fixed embedding numerical precision
    pub precision: EmbeddingPrecision,
    /// Embedding recipe revision
    pub revision: String,
    /// Internal pooling/ResNet frame target after mask interpolation
    pub target_frames: u32,
}

/// Reconstruction boundary behavior
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionBoundary {
    /// Preserve explicit chunk/grid boundaries
    Explicit,
}

/// Reconstruction recipe identity
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReconstructionPolicy {
    /// Boundary behavior for reconstructed tracks
    pub boundary: ReconstructionBoundary,
    /// Output extent used by reconstruction
    pub extent_policy: OutputExtentPolicy,
    /// Stable reconstruction recipe identity
    pub id: String,
    /// Reconstruction recipe revision
    pub revision: String,
}

/// Separately identified downstream recipes bound to the raw score meaning
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentationPolicy {
    /// Count aggregation recipe
    pub count: CountPolicy,
    /// Powerset decode recipe
    pub decoder: DecoderPolicy,
    /// Embedding mask and resize recipe
    pub embedding: EmbeddingPolicy,
    /// Optional median filter recipe
    pub filter: FilterPolicy,
    /// Reconstruction recipe
    pub reconstruction: ReconstructionPolicy,
}

/// Complete strict v1 segmentation manifest
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SegmentationManifest {
    /// Canonical waveform identity
    pub audio: AudioIdentity,
    /// Closed manifest format version
    pub format_version: u32,
    /// Explicit timing and chunk geometry
    pub geometry: SegmentationGeometry,
    /// Powerset head shape and class mapping
    pub head: SegmentationHead,
    /// Model, producer, source, and digest identities
    pub identity: BundleIdentity,
    /// Bound decoder and downstream recipes
    pub policy: SegmentationPolicy,
    /// Closed schema identifier
    pub schema_id: String,
    /// Shard inventory and score stage
    pub tensors: TensorInventory,
}

/// A decoded and validated NPY score shard
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedTensorShard {
    /// Relative member path from the bundle root
    path: String,
    /// Inclusive chunk range start
    chunk_start: u64,
    /// Exclusive chunk range end
    chunk_end: u64,
    /// Rank-three tensor shape `[chunks, frames, classes]`
    shape: [u64; 3],
    /// C-order float32 values
    values: Vec<f32>,
}

impl LoadedTensorShard {
    /// Return the relative member path from the bundle root
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Return the inclusive chunk range start
    pub const fn chunk_start(&self) -> u64 {
        self.chunk_start
    }

    /// Return the exclusive chunk range end
    pub const fn chunk_end(&self) -> u64 {
        self.chunk_end
    }

    /// Return the rank-three tensor shape `[chunks, frames, classes]`
    pub const fn shape(&self) -> [u64; 3] {
        self.shape
    }

    /// Return the admitted C-order float32 values
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// A complete immutable segmentation bundle with loaded score shards
#[derive(Clone, Debug, PartialEq)]
pub struct ImportedSegmentationBundle {
    /// Canonical path of the published bundle directory
    root: PathBuf,
    /// Strict validated manifest
    manifest: SegmentationManifest,
    /// Loaded score shards in manifest order
    shards: Vec<LoadedTensorShard>,
}

impl ImportedSegmentationBundle {
    /// Return the canonical path of the published bundle directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Return the strict validated manifest
    pub const fn manifest(&self) -> &SegmentationManifest {
        &self.manifest
    }

    /// Return loaded score shards in manifest order
    pub fn shards(&self) -> &[LoadedTensorShard] {
        &self.shards
    }
}

/// Public name for a validated immutable imported segmentation bundle
pub type SegmentationBundle = ImportedSegmentationBundle;

pub(crate) fn regular_fixed_step_starts(
    sample_count: u64,
    window_samples: u64,
    step_samples: u64,
) -> Result<Vec<u64>, SegmentationBundleError> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }
    let last = if sample_count <= window_samples {
        0
    } else {
        let remainder = sample_count - window_samples;
        remainder
            .checked_add(step_samples - 1)
            .ok_or_else(|| SegmentationBundleError::Invalid("window start overflow".to_owned()))?
            / step_samples
            * step_samples
    };
    let count = last / step_samples + 1;
    if count > MAX_CHUNKS {
        return Err(SegmentationBundleError::Invalid(
            "planned chunk count exceeds the admission bound".to_owned(),
        ));
    }
    (0..count)
        .map(|index| {
            index
                .checked_mul(step_samples)
                .ok_or_else(|| SegmentationBundleError::Invalid("window start overflow".to_owned()))
        })
        .collect()
}

mod admission;
mod canonical;
mod validation;

pub use admission::load_imported_segmentation_bundle;
pub use canonical::{canonical_bundle_id, canonical_manifest_digest, parse_manifest};
