use std::fs;
use std::path::{Path, PathBuf};

pub use crate::imported_segmentation::Sha256Digest;
use serde::{Deserialize, Serialize};

/// The fixed input and output geometry of a masked embedding wrapper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmbeddingInputGeometry {
    sample_rate: usize,
    window_samples: usize,
    mask_frames: usize,
    embedding_width: usize,
}

impl EmbeddingInputGeometry {
    /// Build a geometry contract from validated positive dimensions
    pub const fn new(
        sample_rate: usize,
        window_samples: usize,
        mask_frames: usize,
        embedding_width: usize,
    ) -> Result<Self, EmbeddingGeometryError> {
        if sample_rate == 0 {
            return Err(EmbeddingGeometryError::ZeroDimension {
                field: "sample_rate",
            });
        }
        if window_samples == 0 {
            return Err(EmbeddingGeometryError::ZeroDimension {
                field: "window_samples",
            });
        }
        if mask_frames == 0 {
            return Err(EmbeddingGeometryError::ZeroDimension {
                field: "mask_frames",
            });
        }
        if embedding_width == 0 {
            return Err(EmbeddingGeometryError::ZeroDimension {
                field: "embedding_width",
            });
        }

        Ok(Self {
            sample_rate,
            window_samples,
            mask_frames,
            embedding_width,
        })
    }

    /// Return the waveform sample rate required by the wrapper
    pub const fn sample_rate(self) -> usize {
        self.sample_rate
    }

    /// Return the exact waveform window length in samples
    pub const fn window_samples(self) -> usize {
        self.window_samples
    }

    /// Return the exact mask length in frames
    pub const fn mask_frames(self) -> usize {
        self.mask_frames
    }

    /// Return the embedding vector width
    pub const fn embedding_width(self) -> usize {
        self.embedding_width
    }

    /// Return whether this is the pinned legacy WeSpeaker geometry
    pub const fn is_legacy(self) -> bool {
        self.sample_rate == 16_000
            && self.window_samples == LEGACY_WINDOW_SAMPLES
            && self.mask_frames == LEGACY_MASK_FRAMES
            && self.embedding_width == LEGACY_EMBEDDING_WIDTH
    }
}

/// Geometry values used by the existing optimized WeSpeaker assets
pub const LEGACY_WINDOW_SAMPLES: usize = 160_000;
/// Mask frame count used by the existing optimized WeSpeaker assets
pub const LEGACY_MASK_FRAMES: usize = 589;
/// Embedding width used by the existing optimized WeSpeaker assets
pub const LEGACY_EMBEDDING_WIDTH: usize = 256;
/// ResNet and statistics-pooling frame target used by the optimized assets
pub(crate) const LEGACY_POOLING_FRAMES: usize = 125;
const REFERENCE_WINDOW_SAMPLES: usize = 128_000;
const REFERENCE_MASK_FRAMES: usize = 399;
const REFERENCE_RESNET_FRAMES: usize = 100;

/// Stable identity for the fixed embedding wrapper accepted by the imported path
pub const EMBEDDING_ARTIFACT_ID: &str = "wespeaker-voxceleb-resnet34-fixed";
/// Revision of the fixed embedding wrapper accepted by the imported path
pub const EMBEDDING_ARTIFACT_REVISION: &str = "b2-fixed";

/// A closed runtime capability admitted by an embedding model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRuntimeCapability {
    /// One masked waveform can be embedded at a time
    PerSpeakerMasked,
    /// The legacy fused primary batch layout is admitted
    LegacyFusedBatch,
    /// The legacy split filterbank and tail layout is admitted
    LegacySplit,
    /// The legacy multi-mask tail layout is admitted
    LegacyMultiMask,
    /// Legacy native chunk embedding layouts are admitted
    LegacyChunk,
}

/// Runtime profile selected from the primary wrapper geometry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingRuntimeProfile {
    /// A fixed-shape wrapper with only the checked per-speaker path
    GenericPerSpeakerMasked,
    /// The exact geometry required by the legacy optimized assets
    LegacyOptimized,
}

impl EmbeddingRuntimeProfile {
    /// Return the capabilities admitted by this profile
    pub const fn supports(self, capability: EmbeddingRuntimeCapability) -> bool {
        match self {
            Self::GenericPerSpeakerMasked => {
                matches!(capability, EmbeddingRuntimeCapability::PerSpeakerMasked)
            }
            Self::LegacyOptimized => true,
        }
    }

    /// Return whether this profile admits legacy optimized assets
    pub const fn is_legacy(self) -> bool {
        matches!(self, Self::LegacyOptimized)
    }
}

/// Typed runtime capabilities resolved for one loaded embedding model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EmbeddingRuntimeCapabilities {
    profile: EmbeddingRuntimeProfile,
}

impl EmbeddingRuntimeCapabilities {
    /// Resolve capabilities from an already selected runtime profile
    pub const fn for_profile(profile: EmbeddingRuntimeProfile) -> Self {
        Self { profile }
    }

    /// Resolve a runtime profile from the primary wrapper geometry
    pub const fn for_geometry(geometry: EmbeddingInputGeometry) -> Self {
        let profile = if geometry.is_legacy() {
            EmbeddingRuntimeProfile::LegacyOptimized
        } else {
            EmbeddingRuntimeProfile::GenericPerSpeakerMasked
        };
        Self::for_profile(profile)
    }

    /// Return the closed runtime profile
    pub const fn profile(self) -> EmbeddingRuntimeProfile {
        self.profile
    }

    /// Return whether the requested capability is admitted
    pub const fn supports(self, capability: EmbeddingRuntimeCapability) -> bool {
        self.profile.supports(capability)
    }

    /// Return whether one masked waveform per speaker is admitted
    pub const fn supports_per_speaker_masked(self) -> bool {
        self.supports(EmbeddingRuntimeCapability::PerSpeakerMasked)
    }

    /// Return whether legacy optimized layouts are admitted
    pub const fn supports_legacy_optimized(self) -> bool {
        self.profile.is_legacy()
    }
}

/// Errors raised while constructing or validating embedding geometry
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddingGeometryError {
    /// A required geometry dimension was zero
    #[error("embedding geometry field `{field}` must be greater than zero")]
    ZeroDimension {
        /// The invalid field name
        field: &'static str,
    },
    /// A required ONNX dimension was dynamic
    #[error("primary embedding tensor `{tensor}` dimension {axis} is dynamic")]
    DynamicDimension {
        /// The tensor input or output name
        tensor: String,
        /// The dynamic dimension index
        axis: usize,
    },
    /// A required ONNX dimension was negative or zero
    #[error(
        "primary embedding tensor `{tensor}` dimension {axis} must be greater than zero, got {value}"
    )]
    InvalidDimension {
        /// The tensor input or output name
        tensor: String,
        /// The invalid dimension index
        axis: usize,
        /// The invalid dimension value
        value: i64,
    },
    /// A primary tensor had the wrong rank or fixed dimensions
    #[error("primary embedding tensor `{tensor}` expected shape {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The tensor input or output name
        tensor: String,
        /// The required shape, with `None` for a derived dimension
        expected: Vec<Option<usize>>,
        /// The shape reported by ONNX Runtime
        actual: Vec<i64>,
    },
    /// A primary tensor was not a tensor value
    #[error("primary embedding value `{tensor}` is not a tensor")]
    NotATensor {
        /// The tensor input or output name
        tensor: String,
    },
    /// A primary tensor did not use float32 values
    #[error("primary embedding tensor `{tensor}` must use float32 values, got {actual}")]
    TypeMismatch {
        /// The tensor input or output name
        tensor: String,
        /// The reported ONNX Runtime element type
        actual: String,
    },
    /// A required input was absent
    #[error("primary embedding session is missing required input `{name}`")]
    MissingInput {
        /// The required input name
        name: &'static str,
    },
    /// The primary session had an unsupported input inventory
    #[error(
        "primary embedding session expected exactly waveform and weights inputs, got {actual:?}"
    )]
    UnexpectedInputs {
        /// The names reported by ONNX Runtime
        actual: Vec<String>,
    },
    /// The primary session had no output or more than one output
    #[error("primary embedding session expected exactly one output, got {count}")]
    UnexpectedOutputs {
        /// The number of reported outputs
        count: usize,
    },
}

/// An input or output tensor shape used by the pure primary-session validator
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrimaryTensorShape {
    pub(crate) name: String,
    pub(crate) shape: Vec<i64>,
}

impl PrimaryTensorShape {
    pub(crate) fn new(name: impl Into<String>, shape: impl IntoIterator<Item = i64>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into_iter().collect(),
        }
    }
}

/// Validate the fixed dimensions of the primary masked wrapper
pub(crate) fn geometry_from_primary_shapes(
    waveform: &PrimaryTensorShape,
    weights: &PrimaryTensorShape,
    output: &PrimaryTensorShape,
) -> Result<EmbeddingInputGeometry, EmbeddingGeometryError> {
    validate_fixed_rank_shape(waveform, &[Some(1), Some(1), None])?;
    validate_fixed_rank_shape(weights, &[Some(1), None])?;
    validate_fixed_rank_shape(output, &[Some(1), None])?;

    let window_samples = positive_dimension(&waveform.name, &waveform.shape, 2)?;
    let mask_frames = positive_dimension(&weights.name, &weights.shape, 1)?;
    let embedding_width = positive_dimension(&output.name, &output.shape, 1)?;

    EmbeddingInputGeometry::new(16_000, window_samples, mask_frames, embedding_width)
}

fn validate_fixed_rank_shape(
    tensor: &PrimaryTensorShape,
    expected: &[Option<usize>],
) -> Result<(), EmbeddingGeometryError> {
    if tensor.shape.len() != expected.len() {
        return Err(EmbeddingGeometryError::ShapeMismatch {
            tensor: tensor.name.clone(),
            expected: expected.to_vec(),
            actual: tensor.shape.clone(),
        });
    }

    for (axis, (&value, expected_value)) in tensor.shape.iter().zip(expected).enumerate() {
        if value == -1 {
            return Err(EmbeddingGeometryError::DynamicDimension {
                tensor: tensor.name.clone(),
                axis,
            });
        }
        if value <= 0 {
            return Err(EmbeddingGeometryError::InvalidDimension {
                tensor: tensor.name.clone(),
                axis,
                value,
            });
        }
        if let Some(expected_value) = expected_value
            && value != *expected_value as i64
        {
            return Err(EmbeddingGeometryError::ShapeMismatch {
                tensor: tensor.name.clone(),
                expected: expected.to_vec(),
                actual: tensor.shape.clone(),
            });
        }
    }

    Ok(())
}

fn positive_dimension(
    tensor: &str,
    shape: &[i64],
    axis: usize,
) -> Result<usize, EmbeddingGeometryError> {
    let value = shape[axis];
    usize::try_from(value).map_err(|_| EmbeddingGeometryError::InvalidDimension {
        tensor: tensor.to_owned(),
        axis,
        value,
    })
}

/// Typed errors raised before an embedding wrapper receives malformed input
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddingInputError {
    /// Audio exceeds the fixed wrapper window
    #[error("embedding audio has {actual} samples but the fixed window accepts at most {expected}")]
    AudioTooLong {
        /// The fixed wrapper window length
        expected: usize,
        /// The supplied audio length
        actual: usize,
    },
    /// A mask does not match the wrapper mask grid
    #[error("embedding mask has {actual} frames but the wrapper requires exactly {expected}")]
    MaskLengthMismatch {
        /// The exact wrapper mask length
        expected: usize,
        /// The supplied mask length
        actual: usize,
    },
    /// A clean mask does not match the primary mask grid
    #[error("embedding clean mask has {actual} frames but the wrapper requires exactly {expected}")]
    CleanMaskLengthMismatch {
        /// The exact wrapper mask length
        expected: usize,
        /// The supplied clean mask length
        actual: usize,
    },
}

/// Validate an audio length against a fixed embedding window
pub(crate) fn validate_audio_length(
    window_samples: usize,
    audio_len: usize,
) -> Result<(), EmbeddingInputError> {
    if audio_len > window_samples {
        return Err(EmbeddingInputError::AudioTooLong {
            expected: window_samples,
            actual: audio_len,
        });
    }
    Ok(())
}

/// Validate a mask length against a fixed embedding mask grid
pub(crate) fn validate_mask_length(
    mask_frames: usize,
    mask_len: usize,
) -> Result<(), EmbeddingInputError> {
    if mask_len != mask_frames {
        return Err(EmbeddingInputError::MaskLengthMismatch {
            expected: mask_frames,
            actual: mask_len,
        });
    }
    Ok(())
}

/// Compute the strict clean-mask activity threshold in mask frames
pub(crate) fn clean_mask_threshold(
    mask_len: usize,
    window_samples: usize,
    min_num_samples: usize,
) -> Option<usize> {
    if window_samples == 0 {
        return None;
    }
    mask_len
        .checked_mul(min_num_samples)
        .map(|numerator| numerator.div_ceil(window_samples))
}

/// Frontend identity admitted by the fixed-shape wrapper sidecar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingFrontend {
    /// WeSpeaker 16 kHz filterbank frontend
    #[serde(rename = "wespeaker_fbank_v1")]
    WeSpeakerFbankV1,
}

/// Pooling identity admitted by the fixed-shape wrapper sidecar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingPooling {
    /// Masked mean and standard-deviation statistics pooling
    #[serde(rename = "masked_stats_pool_v1")]
    MaskedStatsPoolV1,
}

/// Mask interpolation identity admitted by the fixed-shape wrapper sidecar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingMaskInterpolation {
    /// PyTorch nearest-neighbor interpolation
    #[serde(rename = "nearest")]
    Nearest,
}

/// Numeric precision identity admitted by the fixed-shape wrapper sidecar
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EmbeddingPrecision {
    /// Float32 inputs, weights, and outputs
    #[serde(rename = "float32")]
    Float32,
}

/// Strict metadata sidecar for a non-legacy fixed-shape embedding wrapper
///
/// The file is adjacent to the model and uses the name
/// `<model-stem>.embedding.json`, for example
/// `wespeaker-fixed.onnx` and `wespeaker-fixed.embedding.json`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingArtifactMetadata {
    /// Sidecar schema version
    pub schema_version: u32,
    /// SHA-256 digest of the exact ONNX file bytes
    pub onnx_sha256: Sha256Digest,
    /// Audio sample rate in Hz
    pub sample_rate: usize,
    /// Fixed waveform window length in samples
    pub window_samples: usize,
    /// Fixed segmentation mask length in frames
    pub mask_frames: usize,
    /// Embedding vector width
    pub embedding_width: usize,
    /// Frontend identity
    pub frontend: EmbeddingFrontend,
    /// Pooling identity
    pub pooling: EmbeddingPooling,
    /// Mask interpolation identity
    pub interpolation: EmbeddingMaskInterpolation,
    /// Number of ResNet frames produced by the frontend
    pub resnet_frames: usize,
    /// Numeric precision identity
    pub precision: EmbeddingPrecision,
    /// Minimum audio samples used by clean-mask selection
    pub min_num_samples: usize,
}

impl EmbeddingArtifactMetadata {
    /// Build a sidecar record for a validated fixed-shape wrapper
    pub fn new(
        geometry: EmbeddingInputGeometry,
        onnx_sha256: Sha256Digest,
        resnet_frames: usize,
        min_num_samples: usize,
    ) -> Result<Self, EmbeddingMetadataError> {
        if resnet_frames == 0 {
            return Err(EmbeddingMetadataError::InvalidField {
                field: "resnet_frames",
                message: "must be greater than zero".to_owned(),
            });
        }
        if min_num_samples == 0 {
            return Err(EmbeddingMetadataError::InvalidField {
                field: "min_num_samples",
                message: "must be greater than zero".to_owned(),
            });
        }
        Ok(Self {
            schema_version: 1,
            onnx_sha256,
            sample_rate: geometry.sample_rate(),
            window_samples: geometry.window_samples(),
            mask_frames: geometry.mask_frames(),
            embedding_width: geometry.embedding_width(),
            frontend: EmbeddingFrontend::WeSpeakerFbankV1,
            pooling: EmbeddingPooling::MaskedStatsPoolV1,
            interpolation: EmbeddingMaskInterpolation::Nearest,
            resnet_frames,
            precision: EmbeddingPrecision::Float32,
            min_num_samples,
        })
    }

    /// Parse strict JSON sidecar bytes
    pub fn from_json(bytes: &[u8]) -> Result<Self, EmbeddingMetadataError> {
        serde_json::from_slice(bytes).map_err(|error| EmbeddingMetadataError::Malformed {
            message: error.to_string(),
        })
    }

    /// Return the adjacent sidecar path for an ONNX model
    pub fn path_for(model_path: impl AsRef<Path>) -> PathBuf {
        let model_path = model_path.as_ref();
        model_path.with_extension("embedding.json")
    }

    /// Read and parse the adjacent sidecar
    pub fn read_for(model_path: impl AsRef<Path>) -> Result<Self, EmbeddingMetadataError> {
        let (path, bytes) = Self::read_sidecar_bytes(model_path)?;
        Self::from_json(&bytes).map_err(|error| EmbeddingMetadataError::Malformed {
            message: format!("{}: {error}", path.display()),
        })
    }

    /// Read, validate, and retain the exact model and sidecar identity
    pub fn read_verified_for(
        model_path: impl AsRef<Path>,
        geometry: EmbeddingInputGeometry,
    ) -> Result<VerifiedEmbeddingArtifact, EmbeddingMetadataError> {
        let model_path = model_path.as_ref();
        let (path, bytes) = Self::read_sidecar_bytes(model_path)?;
        let metadata =
            Self::from_json(&bytes).map_err(|error| EmbeddingMetadataError::Malformed {
                message: format!("{}: {error}", path.display()),
            })?;
        metadata.validate_against(model_path, geometry)?;
        Ok(VerifiedEmbeddingArtifact {
            metadata,
            sidecar_sha256: Sha256Digest::digest(&bytes),
        })
    }

    fn read_sidecar_bytes(
        model_path: impl AsRef<Path>,
    ) -> Result<(PathBuf, Vec<u8>), EmbeddingMetadataError> {
        let path = Self::path_for(model_path);
        let bytes = fs::read(&path).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                EmbeddingMetadataError::Missing { path: path.clone() }
            } else {
                EmbeddingMetadataError::Unreadable {
                    path: path.clone(),
                    message: error.to_string(),
                }
            }
        })?;
        Ok((path, bytes))
    }

    /// Validate every sidecar identity against the loaded primary wrapper
    pub fn validate_against(
        &self,
        model_path: impl AsRef<Path>,
        geometry: EmbeddingInputGeometry,
    ) -> Result<(), EmbeddingMetadataError> {
        if self.schema_version != 1 {
            return Err(EmbeddingMetadataError::UnsupportedSchema {
                expected: 1,
                actual: self.schema_version,
            });
        }

        let sidecar_geometry = EmbeddingInputGeometry::new(
            self.sample_rate,
            self.window_samples,
            self.mask_frames,
            self.embedding_width,
        )
        .map_err(|error| EmbeddingMetadataError::InvalidField {
            field: "geometry",
            message: error.to_string(),
        })?;
        if sidecar_geometry != geometry {
            return Err(EmbeddingMetadataError::ShapeMismatch {
                expected: geometry,
                actual: sidecar_geometry,
            });
        }

        if self.resnet_frames == 0 {
            return Err(EmbeddingMetadataError::InvalidField {
                field: "resnet_frames",
                message: "must be greater than zero".to_owned(),
            });
        }
        if geometry.window_samples() == REFERENCE_WINDOW_SAMPLES
            && geometry.mask_frames() == REFERENCE_MASK_FRAMES
            && self.resnet_frames != REFERENCE_RESNET_FRAMES
        {
            return Err(EmbeddingMetadataError::InvalidField {
                field: "resnet_frames",
                message: format!(
                    "the {REFERENCE_WINDOW_SAMPLES}-sample/{REFERENCE_MASK_FRAMES}-frame wrapper requires {REFERENCE_RESNET_FRAMES} ResNet frames"
                ),
            });
        }
        if self.min_num_samples == 0 {
            return Err(EmbeddingMetadataError::InvalidField {
                field: "min_num_samples",
                message: "must be greater than zero".to_owned(),
            });
        }

        let expected_hash =
            Sha256Digest::digest(&fs::read(model_path.as_ref()).map_err(|error| {
                EmbeddingMetadataError::Unreadable {
                    path: model_path.as_ref().to_path_buf(),
                    message: error.to_string(),
                }
            })?);
        if self.onnx_sha256 != expected_hash {
            return Err(EmbeddingMetadataError::HashMismatch {
                expected: expected_hash,
                actual: self.onnx_sha256.clone(),
            });
        }

        Ok(())
    }
}

/// Verified identity receipt for one fixed-shape embedding wrapper
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEmbeddingArtifact {
    metadata: EmbeddingArtifactMetadata,
    sidecar_sha256: Sha256Digest,
}

impl VerifiedEmbeddingArtifact {
    /// Return the fixed embedding artifact identity
    pub const fn id(&self) -> &'static str {
        EMBEDDING_ARTIFACT_ID
    }

    /// Return the fixed embedding artifact revision
    pub const fn revision(&self) -> &'static str {
        EMBEDDING_ARTIFACT_REVISION
    }

    /// Return the parsed sidecar metadata
    pub fn metadata(&self) -> &EmbeddingArtifactMetadata {
        &self.metadata
    }

    /// Return the exact model file digest declared and verified by the sidecar
    pub fn model_sha256(&self) -> &Sha256Digest {
        &self.metadata.onnx_sha256
    }

    /// Return the digest of the exact sidecar bytes that were read
    pub fn sidecar_sha256(&self) -> &Sha256Digest {
        &self.sidecar_sha256
    }
}

/// Errors raised while reading or validating a fixed-shape embedding sidecar
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EmbeddingMetadataError {
    /// The adjacent sidecar is missing
    #[error("missing embedding sidecar `{path}`")]
    Missing {
        /// The expected sidecar path
        path: PathBuf,
    },
    /// The adjacent sidecar could not be read
    #[error("unreadable embedding sidecar `{path}`: {message}")]
    Unreadable {
        /// The sidecar path
        path: PathBuf,
        /// The read error
        message: String,
    },
    /// Sidecar JSON was malformed or contained unknown fields
    #[error("malformed embedding sidecar: {message}")]
    Malformed {
        /// The parse error
        message: String,
    },
    /// The sidecar schema version is unsupported
    #[error("embedding sidecar schema version {actual} is unsupported; expected {expected}")]
    UnsupportedSchema {
        /// The supported schema version
        expected: u32,
        /// The supplied schema version
        actual: u32,
    },
    /// A sidecar field is invalid
    #[error("invalid embedding sidecar field `{field}`: {message}")]
    InvalidField {
        /// The invalid field name
        field: &'static str,
        /// The validation reason
        message: String,
    },
    /// Sidecar geometry does not match the primary session
    #[error("embedding sidecar geometry mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The primary session geometry
        expected: EmbeddingInputGeometry,
        /// The sidecar geometry
        actual: EmbeddingInputGeometry,
    },
    /// Sidecar hash does not match the ONNX bytes
    #[error("embedding sidecar ONNX SHA-256 mismatch: expected {expected}, got {actual}")]
    HashMismatch {
        /// The digest computed from the ONNX file
        expected: Sha256Digest,
        /// The digest declared by the sidecar
        actual: Sha256Digest,
    },
}

impl From<EmbeddingGeometryError> for crate::inference::ModelLoadError {
    fn from(error: EmbeddingGeometryError) -> Self {
        Self::InvalidConfiguration {
            message: error.to_string(),
        }
    }
}

impl From<EmbeddingMetadataError> for crate::inference::ModelLoadError {
    fn from(error: EmbeddingMetadataError) -> Self {
        Self::InvalidConfiguration {
            message: error.to_string(),
        }
    }
}

impl From<EmbeddingInputError> for ort::Error {
    fn from(error: EmbeddingInputError) -> Self {
        ort::Error::new(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> EmbeddingInputGeometry {
        EmbeddingInputGeometry::new(16_000, 128_000, 399, 256).unwrap()
    }

    #[test]
    fn geometry_validator_accepts_fixed_primary_shapes() {
        let actual = geometry_from_primary_shapes(
            &PrimaryTensorShape::new("waveform", [1, 1, 128_000]),
            &PrimaryTensorShape::new("weights", [1, 399]),
            &PrimaryTensorShape::new("output", [1, 256]),
        )
        .unwrap();

        assert_eq!(actual, geometry());
        assert!(!actual.is_legacy());
    }

    #[test]
    fn geometry_validator_rejects_dynamic_and_invalid_shapes() {
        let dynamic = geometry_from_primary_shapes(
            &PrimaryTensorShape::new("waveform", [1, 1, -1]),
            &PrimaryTensorShape::new("weights", [1, 399]),
            &PrimaryTensorShape::new("output", [1, 256]),
        )
        .unwrap_err();
        assert!(matches!(
            dynamic,
            EmbeddingGeometryError::DynamicDimension { .. }
        ));

        let rank = geometry_from_primary_shapes(
            &PrimaryTensorShape::new("waveform", [1, 128_000]),
            &PrimaryTensorShape::new("weights", [1, 399]),
            &PrimaryTensorShape::new("output", [1, 256]),
        )
        .unwrap_err();
        assert!(matches!(rank, EmbeddingGeometryError::ShapeMismatch { .. }));

        let zero = geometry_from_primary_shapes(
            &PrimaryTensorShape::new("waveform", [1, 1, 0]),
            &PrimaryTensorShape::new("weights", [1, 399]),
            &PrimaryTensorShape::new("output", [1, 256]),
        )
        .unwrap_err();
        assert!(matches!(
            zero,
            EmbeddingGeometryError::InvalidDimension { .. }
        ));
    }

    #[test]
    fn runtime_profile_only_admits_optimized_paths_for_legacy_geometry() {
        let legacy = EmbeddingInputGeometry::new(16_000, 160_000, 589, 256).unwrap();
        let modern = geometry();
        let legacy_caps = EmbeddingRuntimeCapabilities::for_geometry(legacy);
        let modern_caps = EmbeddingRuntimeCapabilities::for_geometry(modern);

        assert!(legacy_caps.supports(EmbeddingRuntimeCapability::LegacySplit));
        assert!(legacy_caps.supports(EmbeddingRuntimeCapability::LegacyMultiMask));
        assert!(legacy_caps.supports(EmbeddingRuntimeCapability::LegacyChunk));
        assert!(!modern_caps.supports(EmbeddingRuntimeCapability::LegacySplit));
        assert!(!modern_caps.supports(EmbeddingRuntimeCapability::LegacyMultiMask));
        assert!(!modern_caps.supports(EmbeddingRuntimeCapability::LegacyChunk));
        assert!(modern_caps.supports_per_speaker_masked());
    }

    #[test]
    fn pooling_target_is_distinct_from_wrapper_mask_frames() {
        let modern = geometry();
        let metadata =
            EmbeddingArtifactMetadata::new(modern, Sha256Digest::digest(b"model"), 100, 4_000)
                .unwrap();

        assert_eq!(modern.mask_frames(), 399);
        assert_eq!(metadata.resnet_frames, 100);
        assert_eq!(LEGACY_MASK_FRAMES, 589);
        assert_eq!(LEGACY_POOLING_FRAMES, 125);
    }

    #[test]
    fn audio_and_mask_validation_rejects_only_overlong_or_wrong_masks() {
        assert!(validate_audio_length(128_000, 128_000).is_ok());
        assert!(matches!(
            validate_audio_length(128_000, 128_001),
            Err(EmbeddingInputError::AudioTooLong { .. })
        ));
        assert!(validate_mask_length(399, 399).is_ok());
        assert!(matches!(
            validate_mask_length(399, 398),
            Err(EmbeddingInputError::MaskLengthMismatch { .. })
        ));
    }

    #[test]
    fn clean_threshold_is_strict_and_uses_the_fixed_window() {
        assert_eq!(clean_mask_threshold(399, 128_000, 1_000), Some(4));
        assert_eq!(clean_mask_threshold(399, 128_000, 1_001), Some(4));
        assert_eq!(clean_mask_threshold(399, 128_000, 1_282), Some(4));
        assert_eq!(clean_mask_threshold(399, 128_000, 1_284), Some(5));
        assert_eq!(clean_mask_threshold(399, 0, 1), None);
    }

    #[test]
    fn metadata_round_trips_and_rejects_unknown_fields() {
        let hash = Sha256Digest::digest(b"model");
        let metadata =
            EmbeddingArtifactMetadata::new(geometry(), hash.clone(), 100, 4_000).unwrap();
        let bytes = serde_json::to_vec(&metadata).unwrap();
        assert_eq!(
            EmbeddingArtifactMetadata::from_json(&bytes).unwrap(),
            metadata
        );

        let mut value: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&bytes).unwrap();
        value.insert("unexpected".to_owned(), serde_json::Value::Bool(true));
        let bytes = serde_json::to_vec(&value).unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::from_json(&bytes),
            Err(EmbeddingMetadataError::Malformed { .. })
        ));

        let mut noncanonical = serde_json::to_value(&metadata).unwrap();
        noncanonical["onnx_sha256"] =
            serde_json::Value::String(metadata.onnx_sha256.as_str().to_ascii_uppercase());
        let bytes = serde_json::to_vec(&noncanonical).unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::from_json(&bytes),
            Err(EmbeddingMetadataError::Malformed { .. })
        ));
    }

    #[test]
    fn metadata_validation_rejects_missing_hash_shape_and_identity_changes() {
        let root = std::env::temp_dir().join(format!(
            "speakrs-embedding-sidecar-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("wrapper.onnx");
        fs::write(&model_path, b"model").unwrap();
        let metadata_path = EmbeddingArtifactMetadata::path_for(&model_path);
        let hash = Sha256Digest::digest(b"model");
        let metadata =
            EmbeddingArtifactMetadata::new(geometry(), hash.clone(), 100, 4_000).unwrap();

        assert!(matches!(
            EmbeddingArtifactMetadata::read_for(&model_path),
            Err(EmbeddingMetadataError::Missing { .. })
        ));

        let wrong_hash = EmbeddingArtifactMetadata::new(
            geometry(),
            Sha256Digest::digest(b"different"),
            100,
            4_000,
        )
        .unwrap();
        fs::write(&metadata_path, serde_json::to_vec(&wrong_hash).unwrap()).unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::read_for(&model_path)
                .unwrap()
                .validate_against(&model_path, geometry()),
            Err(EmbeddingMetadataError::HashMismatch { .. })
        ));

        let wrong_shape = EmbeddingArtifactMetadata::new(
            EmbeddingInputGeometry::new(16_000, 128_001, 399, 256).unwrap(),
            hash,
            100,
            4_000,
        )
        .unwrap();
        fs::write(&metadata_path, serde_json::to_vec(&wrong_shape).unwrap()).unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::read_for(&model_path)
                .unwrap()
                .validate_against(&model_path, geometry()),
            Err(EmbeddingMetadataError::ShapeMismatch { .. })
        ));

        let mut identity = serde_json::to_value(&metadata).unwrap();
        identity["frontend"] = serde_json::Value::String("other_frontend".to_owned());
        fs::write(&metadata_path, serde_json::to_vec(&identity).unwrap()).unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::read_for(&model_path),
            Err(EmbeddingMetadataError::Malformed { .. })
        ));

        let mut wrong_resnet_frames = metadata;
        wrong_resnet_frames.resnet_frames = 101;
        fs::write(
            &metadata_path,
            serde_json::to_vec(&wrong_resnet_frames).unwrap(),
        )
        .unwrap();
        assert!(matches!(
            EmbeddingArtifactMetadata::read_for(&model_path)
                .unwrap()
                .validate_against(&model_path, geometry()),
            Err(EmbeddingMetadataError::InvalidField {
                field: "resnet_frames",
                ..
            })
        ));

        let _ = fs::remove_dir_all(root);
    }
}
