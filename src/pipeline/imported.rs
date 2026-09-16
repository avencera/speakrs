use std::path::Path;

use ndarray::{Array1, s};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::clustering::plda::{PldaError, PldaTransform};
use crate::imported_segmentation::{
    MaskInterpolation, SegmentationBundle, SegmentationBundleError, SegmentationManifest,
    Sha256Digest,
};
use crate::inference::embedding::{
    EmbeddingFrontend, EmbeddingInputGeometry, EmbeddingMaskInterpolation, EmbeddingModel,
    EmbeddingPooling, EmbeddingPrecision, should_use_clean_mask_slice,
};

use super::config::{MIN_SPEAKER_ACTIVITY, PipelineConfig};
use super::imported_decoder::{ImportedDecodeError, ImportedSegmentationDecoder};
use super::types::{
    DecodedSegmentations, EmbeddingAvailability, EmbeddingMaskChoice, EmbeddingStageSnapshot,
    InactiveEmbeddingReason, InferenceArtifacts, PipelineError, PipelineGeometry,
    TypedChunkEmbeddings, TypedEmbedding,
};

/// Errors raised before or during an imported segmentation run
#[derive(Debug, Error)]
pub enum ImportedPipelineError {
    /// The immutable segmentation bundle is malformed
    #[error(transparent)]
    Bundle(#[from] SegmentationBundleError),
    /// Imported score decoding failed
    #[error(transparent)]
    Decode(#[from] ImportedDecodeError),
    /// Imported timing geometry is invalid
    #[error(transparent)]
    Geometry(#[from] super::PipelineGeometryError),
    /// Existing clustering or reconstruction failed after imported inference
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    /// The supplied waveform has the wrong canonical sample count
    #[error("canonical waveform sample count mismatch: expected {expected}, got {actual}")]
    WaveformSampleCount {
        /// Manifest sample count
        expected: u64,
        /// Supplied sample count
        actual: usize,
    },
    /// The supplied waveform bytes do not match the bundle identity
    #[error("canonical waveform SHA-256 mismatch: expected {expected}, got {actual}")]
    WaveformHashMismatch {
        /// Manifest waveform digest
        expected: Sha256Digest,
        /// Digest computed from little-endian f32 bytes
        actual: Sha256Digest,
    },
    /// A supplied waveform contains a non-finite sample
    #[error("canonical waveform sample {index} is not finite")]
    WaveformNonFinite {
        /// Zero-based sample index
        index: usize,
    },
    /// The embedding model and imported policy do not have the same geometry
    #[error("imported embedding contract mismatch for {field}: expected {expected}, got {actual}")]
    EmbeddingContractMismatch {
        /// Contract field that differed
        field: &'static str,
        /// Required value
        expected: String,
        /// Model value
        actual: String,
    },
    /// The embedding runtime could not execute one active slot
    #[error(
        "embedding execution failed for chunk {chunk_index}, local speaker {speaker_index}: {source}"
    )]
    EmbeddingExecution {
        /// Zero-based chunk index
        chunk_index: usize,
        /// Zero-based local-speaker index
        speaker_index: usize,
        /// Runtime execution error
        #[source]
        source: ort::Error,
    },
    /// The embedding runtime returned a vector with the wrong width
    #[error(
        "embedding output width mismatch for chunk {chunk_index}, local speaker {speaker_index}: expected {expected_width}, got {actual_width}"
    )]
    EmbeddingOutputWidthMismatch {
        /// Zero-based chunk index
        chunk_index: usize,
        /// Zero-based local-speaker index
        speaker_index: usize,
        /// Required embedding width
        expected_width: usize,
        /// Observed embedding width
        actual_width: usize,
    },
    /// The embedding runtime returned a non-finite value
    #[error(
        "embedding output is non-finite at value {value_index} for chunk {chunk_index}, local speaker {speaker_index}"
    )]
    EmbeddingOutputNonFinite {
        /// Zero-based chunk index
        chunk_index: usize,
        /// Zero-based local-speaker index
        speaker_index: usize,
        /// Zero-based vector index
        value_index: usize,
    },
    /// A restored embedding snapshot contains failed inference slots
    #[error("embedding snapshot contains {count} failed inference slots")]
    FailedEmbeddingSnapshot {
        /// Number of failed slots
        count: usize,
    },
}

/// Which mask was selected for one embedding attempt
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum ImportedMaskSelection {
    /// The clean mask had activity above the strict threshold
    Clean,
    /// The clean mask was too short, so the full mask was selected
    Full,
}

/// Checked imported segmentation and embedding pipeline
///
/// This owner contains a segmentation bundle, an embedding model, and PLDA
/// parameters. It never constructs or loads a segmentation model session
pub struct ImportedDiarizationPipeline<'a> {
    bundle: SegmentationBundle,
    geometry: PipelineGeometry,
    decoder: ImportedSegmentationDecoder,
    emb_model: &'a mut EmbeddingModel,
    plda: PldaTransform,
}

impl<'a> ImportedDiarizationPipeline<'a> {
    /// Build an imported pipeline and load PLDA parameters from a model directory
    pub fn new(
        bundle: SegmentationBundle,
        emb_model: &'a mut EmbeddingModel,
        models_dir: &Path,
    ) -> Result<Self, ImportedPipelineError> {
        let plda = PldaTransform::from_imported_artifact(
            models_dir,
            &bundle.manifest().policy.embedding.plda,
        )
        .map_err(map_plda_error)?;
        Self::from_parts(bundle, emb_model, plda)
    }

    /// Build an imported pipeline from already loaded embedding and PLDA values
    pub(crate) fn from_parts(
        bundle: SegmentationBundle,
        emb_model: &'a mut EmbeddingModel,
        plda: PldaTransform,
    ) -> Result<Self, ImportedPipelineError> {
        bundle.manifest().validate()?;
        let geometry =
            PipelineGeometry::from_imported(&bundle.manifest().audio, &bundle.manifest().geometry)?;
        let decoder = ImportedSegmentationDecoder::from_manifest(bundle.manifest())?;
        validate_embedding_contract(bundle.manifest(), emb_model)?;
        validate_plda_contract(bundle.manifest(), &plda)?;
        Ok(Self {
            bundle,
            geometry,
            decoder,
            emb_model,
            plda,
        })
    }

    /// Return the immutable manifest bound to this pipeline
    pub fn manifest(&self) -> &SegmentationManifest {
        self.bundle.manifest()
    }

    /// Return the checked geometry derived from the bundle and audio identity
    pub fn geometry(&self) -> &PipelineGeometry {
        &self.geometry
    }

    /// Validate a canonical mono 16 kHz waveform against the bundle identity
    pub fn validate_waveform(&self, audio: &[f32]) -> Result<(), ImportedPipelineError> {
        validate_waveform_identity(audio, self.bundle.manifest())
    }

    /// Decode the bundle and extract typed per-speaker inference artifacts
    pub fn run_inference(
        &mut self,
        audio: &[f32],
    ) -> Result<InferenceArtifacts, ImportedPipelineError> {
        self.validate_waveform(audio)?;
        let segmentations = self.decoder.decode(&self.bundle)?;
        let outcomes = self.extract_embeddings(audio, &segmentations)?;
        InferenceArtifacts::try_new_from_outcomes(
            self.geometry.clone(),
            segmentations,
            outcomes,
            #[cfg(feature = "_metrics")]
            None,
        )
        .map_err(ImportedPipelineError::from)
    }

    /// Run segmentation and embedding inference into a reusable typed snapshot
    pub fn run_embedding_stage(
        &mut self,
        audio: &[f32],
    ) -> Result<EmbeddingStageSnapshot, ImportedPipelineError> {
        self.run_inference(audio)
            .map(EmbeddingStageSnapshot::from_artifacts)
    }

    /// Finish clustering and reconstruction from a checked embedding-stage snapshot
    pub fn finish_embedding_stage(
        &self,
        snapshot: EmbeddingStageSnapshot,
        config: &PipelineConfig,
    ) -> Result<super::DiarizationResult, ImportedPipelineError> {
        ensure_successful_embedding_snapshot(&snapshot)?;
        if snapshot.geometry() != &self.geometry {
            return Err(ImportedPipelineError::Pipeline(PipelineError::Invariant(
                "embedding snapshot geometry does not match the imported bundle".to_owned(),
            )));
        }
        let shape = snapshot.segmentation_shape();
        if shape[2] != self.bundle.manifest().head.local_slots as usize {
            return Err(ImportedPipelineError::Pipeline(PipelineError::Invariant(
                "embedding snapshot local-slot count does not match the imported bundle".to_owned(),
            )));
        }
        let artifacts = snapshot.into_artifacts()?;
        super::post_inference(artifacts, config, &self.plda).map_err(ImportedPipelineError::from)
    }

    /// Run imported inference followed by post-inference with an explicit config
    pub fn run(
        &mut self,
        audio: &[f32],
        config: &PipelineConfig,
    ) -> Result<super::DiarizationResult, ImportedPipelineError> {
        let snapshot = self.run_embedding_stage(audio)?;
        self.finish_embedding_stage(snapshot, config)
    }

    fn extract_embeddings(
        &mut self,
        audio: &[f32],
        segmentations: &DecodedSegmentations,
    ) -> Result<TypedChunkEmbeddings, ImportedPipelineError> {
        let chunks = segmentations.shape()[0];
        let frames = segmentations.shape()[1];
        let speakers = segmentations.shape()[2];
        let width = self.emb_model.embedding_width();
        let mut outcomes = TypedChunkEmbeddings::new(chunks, speakers, width);

        for chunk_idx in 0..chunks {
            let chunk_audio = self.geometry.chunk_audio(audio, chunk_idx);
            let chunk = segmentations.slice(s![chunk_idx, .., ..]);
            let clean_masks = clean_overlap_masks(&chunk);
            for speaker_idx in 0..speakers {
                let mask = chunk.column(speaker_idx).to_vec();
                let clean_mask = clean_masks.column(speaker_idx).to_vec();
                if let Some(reason) = inactive_embedding_reason(&mask) {
                    outcomes.push(TypedEmbedding {
                        availability: EmbeddingAvailability::Inactive { reason },
                        values: None,
                        mask_choice: None,
                    });
                    continue;
                }

                let selection = choose_mask(
                    &mask,
                    &clean_mask,
                    self.emb_model.window_samples(),
                    self.emb_model.min_num_samples(),
                    self.emb_model.pooling_frames(),
                );
                let choice = match selection {
                    ImportedMaskSelection::Clean => EmbeddingMaskChoice::Clean,
                    ImportedMaskSelection::Full => EmbeddingMaskChoice::Full,
                };
                let result = self
                    .emb_model
                    .embed_masked(chunk_audio, &mask, Some(&clean_mask));
                outcomes.push(admit_embedding_result(
                    result,
                    width,
                    chunk_idx,
                    speaker_idx,
                    choice,
                )?);
            }
        }

        debug_assert_eq!(
            frames,
            self.manifest().geometry.frame_grid.frame_count as usize
        );
        Ok(outcomes)
    }
}

fn inactive_embedding_reason(mask: &[f32]) -> Option<InactiveEmbeddingReason> {
    let activity = mask.iter().copied().sum::<f32>();
    if activity == 0.0 {
        return Some(InactiveEmbeddingReason::NoActivity);
    }
    if activity < MIN_SPEAKER_ACTIVITY {
        return Some(InactiveEmbeddingReason::InsufficientActivity);
    }

    None
}

fn admit_embedding_result(
    result: Result<Array1<f32>, ort::Error>,
    expected_width: usize,
    chunk_index: usize,
    speaker_index: usize,
    mask_choice: EmbeddingMaskChoice,
) -> Result<TypedEmbedding, ImportedPipelineError> {
    let vector = result.map_err(|source| ImportedPipelineError::EmbeddingExecution {
        chunk_index,
        speaker_index,
        source,
    })?;
    if vector.len() != expected_width {
        return Err(ImportedPipelineError::EmbeddingOutputWidthMismatch {
            chunk_index,
            speaker_index,
            expected_width,
            actual_width: vector.len(),
        });
    }
    if let Some(value_index) = vector.iter().position(|value| !value.is_finite()) {
        return Err(ImportedPipelineError::EmbeddingOutputNonFinite {
            chunk_index,
            speaker_index,
            value_index,
        });
    }

    Ok(TypedEmbedding {
        availability: EmbeddingAvailability::Available,
        values: Some(vector.to_vec()),
        mask_choice: Some(mask_choice),
    })
}

fn ensure_successful_embedding_snapshot(
    snapshot: &EmbeddingStageSnapshot,
) -> Result<(), ImportedPipelineError> {
    let count = snapshot.embedding_receipt().inference_failure_count;
    if count > 0 {
        return Err(ImportedPipelineError::FailedEmbeddingSnapshot { count });
    }

    Ok(())
}

/// Validate embedding input, pooling, output, and minimum-activity contracts
pub(crate) fn validate_imported_embedding_geometry(
    manifest: &SegmentationManifest,
    model_geometry: EmbeddingInputGeometry,
    pooling_frames: usize,
    min_num_samples: usize,
) -> Result<(), ImportedPipelineError> {
    let expected_sample_rate = usize::try_from(manifest.audio.sample_rate).map_err(|_| {
        ImportedPipelineError::EmbeddingContractMismatch {
            field: "sample_rate",
            expected: manifest.audio.sample_rate.to_string(),
            actual: model_geometry.sample_rate().to_string(),
        }
    })?;
    check_embedding_field(
        "sample_rate",
        expected_sample_rate,
        model_geometry.sample_rate(),
    )?;
    check_embedding_field(
        "window_samples",
        usize::try_from(manifest.geometry.window_samples).map_err(|_| {
            ImportedPipelineError::EmbeddingContractMismatch {
                field: "window_samples",
                expected: manifest.geometry.window_samples.to_string(),
                actual: model_geometry.window_samples().to_string(),
            }
        })?,
        model_geometry.window_samples(),
    )?;
    check_embedding_field(
        "mask_frames",
        usize::try_from(manifest.geometry.frame_grid.frame_count).map_err(|_| {
            ImportedPipelineError::EmbeddingContractMismatch {
                field: "mask_frames",
                expected: manifest.geometry.frame_grid.frame_count.to_string(),
                actual: model_geometry.mask_frames().to_string(),
            }
        })?,
        model_geometry.mask_frames(),
    )?;
    check_embedding_field(
        "pooling_frames",
        usize::try_from(manifest.policy.embedding.target_frames).map_err(|_| {
            ImportedPipelineError::EmbeddingContractMismatch {
                field: "pooling_frames",
                expected: manifest.policy.embedding.target_frames.to_string(),
                actual: pooling_frames.to_string(),
            }
        })?,
        pooling_frames,
    )?;
    check_embedding_field(
        "embedding_width",
        crate::inference::embedding::EMBEDDING_WIDTH,
        model_geometry.embedding_width(),
    )?;
    check_embedding_field(
        "min_num_samples",
        usize::try_from(manifest.policy.embedding.min_num_samples).map_err(|_| {
            ImportedPipelineError::EmbeddingContractMismatch {
                field: "min_num_samples",
                expected: manifest.policy.embedding.min_num_samples.to_string(),
                actual: min_num_samples.to_string(),
            }
        })?,
        min_num_samples,
    )?;
    if !matches!(
        manifest.policy.embedding.mask_interpolation,
        MaskInterpolation::Nearest
    ) {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "mask_interpolation",
            expected: "nearest".to_owned(),
            actual: format!("{:?}", manifest.policy.embedding.mask_interpolation),
        });
    }
    Ok(())
}

fn validate_embedding_contract(
    manifest: &SegmentationManifest,
    emb_model: &EmbeddingModel,
) -> Result<(), ImportedPipelineError> {
    if !emb_model.capabilities().supports_per_speaker_masked() {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "runtime_capability",
            expected: "per-speaker masked".to_owned(),
            actual: format!("{:?}", emb_model.capabilities().profile()),
        });
    }
    validate_imported_embedding_geometry(
        manifest,
        emb_model.input_geometry(),
        emb_model.pooling_frames(),
        emb_model.min_num_samples(),
    )?;
    let artifact = emb_model.artifact_metadata().ok_or_else(|| {
        ImportedPipelineError::EmbeddingContractMismatch {
            field: "embedding_artifact",
            expected: "verified fixed-shape artifact".to_owned(),
            actual: "legacy or unavailable artifact metadata".to_owned(),
        }
    })?;
    let policy = &manifest.policy.embedding;
    check_embedding_identity(
        "embedding_model.id",
        policy.embedding_model.id.as_str(),
        artifact.id(),
    )?;
    check_embedding_identity(
        "embedding_model.revision",
        policy.embedding_model.revision.as_str(),
        artifact.revision(),
    )?;
    check_embedding_identity(
        "embedding_model.sha256",
        policy.embedding_model.sha256.as_str(),
        artifact.model_sha256().as_str(),
    )?;
    check_embedding_identity(
        "embedding_sidecar_sha256",
        policy.embedding_sidecar_sha256.as_str(),
        artifact.sidecar_sha256().as_str(),
    )?;
    if !matches!(
        policy.frontend,
        crate::imported_segmentation::EmbeddingFrontend::WeSpeakerFbankV1
    ) || artifact.metadata().frontend != EmbeddingFrontend::WeSpeakerFbankV1
    {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "frontend",
            expected: "wespeaker_fbank_v1".to_owned(),
            actual: format!("{:?}", artifact.metadata().frontend),
        });
    }
    if !matches!(
        policy.pooling,
        crate::imported_segmentation::EmbeddingPooling::MaskedStatsPoolV1
    ) || artifact.metadata().pooling != EmbeddingPooling::MaskedStatsPoolV1
    {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "pooling",
            expected: "masked_stats_pool_v1".to_owned(),
            actual: format!("{:?}", artifact.metadata().pooling),
        });
    }
    if policy.mask_interpolation != MaskInterpolation::Nearest
        || artifact.metadata().interpolation != EmbeddingMaskInterpolation::Nearest
    {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "mask_interpolation",
            expected: "nearest".to_owned(),
            actual: format!("{:?}", artifact.metadata().interpolation),
        });
    }
    if !matches!(
        policy.precision,
        crate::imported_segmentation::EmbeddingPrecision::Float32
    ) || artifact.metadata().precision != EmbeddingPrecision::Float32
    {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "precision",
            expected: "float32".to_owned(),
            actual: format!("{:?}", artifact.metadata().precision),
        });
    }
    if policy.target_frames as usize != artifact.metadata().resnet_frames {
        return Err(ImportedPipelineError::EmbeddingContractMismatch {
            field: "target_frames",
            expected: policy.target_frames.to_string(),
            actual: artifact.metadata().resnet_frames.to_string(),
        });
    }
    Ok(())
}

fn validate_plda_contract(
    manifest: &SegmentationManifest,
    plda: &PldaTransform,
) -> Result<(), ImportedPipelineError> {
    let policy = &manifest.policy.embedding.plda;
    let receipt = plda.receipt();
    check_embedding_identity("plda.id", policy.id.as_str(), receipt.id())?;
    check_embedding_identity(
        "plda.revision",
        policy.revision.as_str(),
        receipt.revision(),
    )?;
    check_embedding_identity(
        "plda.sha256",
        policy.sha256.as_str(),
        receipt.sha256().as_str(),
    )
}

fn map_plda_error(error: PldaError) -> ImportedPipelineError {
    match error {
        PldaError::ArtifactIdentityMismatch {
            field,
            expected,
            actual,
        } => ImportedPipelineError::EmbeddingContractMismatch {
            field,
            expected,
            actual,
        },
        error => ImportedPipelineError::Pipeline(PipelineError::Plda(error)),
    }
}

fn check_embedding_identity(
    field: &'static str,
    expected: &str,
    actual: &str,
) -> Result<(), ImportedPipelineError> {
    if expected == actual {
        return Ok(());
    }
    Err(ImportedPipelineError::EmbeddingContractMismatch {
        field,
        expected: expected.to_owned(),
        actual: actual.to_owned(),
    })
}

fn check_embedding_field(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ImportedPipelineError> {
    if expected == actual {
        return Ok(());
    }
    Err(ImportedPipelineError::EmbeddingContractMismatch {
        field,
        expected: expected.to_string(),
        actual: actual.to_string(),
    })
}

/// Compute the exact SHA-256 identity over little-endian f32 sample bytes
pub fn canonical_waveform_digest(audio: &[f32]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    for sample in audio {
        hasher.update(sample.to_le_bytes());
    }
    let digest = hasher.finalize();
    let text = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Sha256Digest::new(text).expect("SHA-256 formatting always produces a canonical digest")
}

/// Validate sample count and exact sample-bit identity against a manifest
pub fn validate_waveform_identity(
    audio: &[f32],
    manifest: &SegmentationManifest,
) -> Result<(), ImportedPipelineError> {
    let sample_count =
        u64::try_from(audio.len()).map_err(|_| ImportedPipelineError::WaveformSampleCount {
            expected: manifest.audio.sample_count,
            actual: audio.len(),
        })?;
    if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
        return Err(ImportedPipelineError::WaveformNonFinite { index });
    }
    if sample_count != manifest.audio.sample_count {
        return Err(ImportedPipelineError::WaveformSampleCount {
            expected: manifest.audio.sample_count,
            actual: audio.len(),
        });
    }
    let actual = canonical_waveform_digest(audio);
    if actual != manifest.audio.waveform_sha256 {
        return Err(ImportedPipelineError::WaveformHashMismatch {
            expected: manifest.audio.waveform_sha256.clone(),
            actual,
        });
    }
    Ok(())
}

/// Select a clean or full mask with the model-owned strict threshold
pub(crate) fn choose_mask(
    mask: &[f32],
    clean_mask: &[f32],
    window_samples: usize,
    min_num_samples: usize,
    pooling_frames: usize,
) -> ImportedMaskSelection {
    debug_assert_eq!(mask.len(), clean_mask.len());
    if should_use_clean_mask_slice(clean_mask, window_samples, min_num_samples, pooling_frames) {
        ImportedMaskSelection::Clean
    } else {
        ImportedMaskSelection::Full
    }
}

fn clean_overlap_masks(segmentations: &ndarray::ArrayView2<'_, f32>) -> ndarray::Array2<f32> {
    let mut clean = ndarray::Array2::<f32>::zeros(segmentations.raw_dim());
    for frame_idx in 0..segmentations.nrows() {
        let overlap = segmentations.row(frame_idx).iter().copied().sum::<f32>() >= 2.0;
        if !overlap {
            clean
                .slice_mut(s![frame_idx, ..])
                .assign(&segmentations.slice(s![frame_idx, ..]));
        }
    }
    clean
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::pipeline::{EmbeddingReceipt, EmbeddingStageEntry};

    #[test]
    fn waveform_digest_distinguishes_negative_zero() {
        assert_ne!(
            canonical_waveform_digest(&[0.0]),
            canonical_waveform_digest(&[-0.0])
        );
    }

    #[test]
    fn waveform_validation_rejects_bitwise_hash_mismatch() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 1;
        manifest.audio.waveform_sha256 = canonical_waveform_digest(&[0.0]);
        let error = validate_waveform_identity(&[-0.0], &manifest).unwrap_err();
        assert!(matches!(
            error,
            ImportedPipelineError::WaveformHashMismatch { .. }
        ));
    }

    #[test]
    fn waveform_validation_rejects_non_finite_samples_before_hashing() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 1;
        let error = validate_waveform_identity(&[f32::NAN], &manifest).unwrap_err();
        assert!(matches!(
            error,
            ImportedPipelineError::WaveformNonFinite { index: 0 }
        ));
    }

    #[test]
    fn clean_mask_falls_back_at_the_strict_threshold() {
        let mut below_threshold = [0.0; 399];
        below_threshold[..4].fill(1.0);
        let mut above_threshold = [0.0; 399];
        above_threshold[..6].fill(1.0);

        assert_eq!(
            choose_mask(&[1.0; 399], &below_threshold, 128_000, 1_284, 100),
            ImportedMaskSelection::Full
        );
        assert_eq!(
            choose_mask(&[1.0; 399], &above_threshold, 128_000, 1_284, 100,),
            ImportedMaskSelection::Clean
        );
    }

    #[test]
    fn clean_mask_falls_back_when_pooling_resize_drops_all_activity() {
        let full = [1.0; 399];
        let mut clean = [0.0; 399];
        clean[176..179].fill(1.0);

        assert_eq!(
            choose_mask(&full, &clean, 128_000, 400, 100),
            ImportedMaskSelection::Full
        );

        clean[175] = 1.0;
        assert_eq!(
            choose_mask(&full, &clean, 128_000, 400, 100),
            ImportedMaskSelection::Clean
        );
    }

    #[test]
    fn embedding_activity_must_reach_the_pipeline_threshold() {
        assert_eq!(
            inactive_embedding_reason(&[0.0; 10]),
            Some(InactiveEmbeddingReason::NoActivity)
        );
        assert_eq!(
            inactive_embedding_reason(&[1.0, 0.0, 0.0]),
            Some(InactiveEmbeddingReason::InsufficientActivity)
        );
        assert_eq!(inactive_embedding_reason(&[1.0; 10]), None);
    }

    #[test]
    fn overlap_clean_mask_zeroes_every_slot_in_overlapping_frames() {
        let segmentations = array![[1.0, 1.0], [1.0, 0.0]];
        let clean = clean_overlap_masks(&segmentations.view());
        assert_eq!(clean, array![[0.0, 0.0], [1.0, 0.0]]);
    }

    #[test]
    fn embedding_execution_error_is_fatal() {
        let error = admit_embedding_result(
            Err(ort::Error::new("execution failed")),
            256,
            2,
            1,
            EmbeddingMaskChoice::Full,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            ImportedPipelineError::EmbeddingExecution {
                chunk_index: 2,
                speaker_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn invalid_embedding_output_is_fatal() {
        let wrong_width =
            admit_embedding_result(Ok(Array1::zeros(255)), 256, 2, 1, EmbeddingMaskChoice::Full)
                .unwrap_err();
        assert!(matches!(
            wrong_width,
            ImportedPipelineError::EmbeddingOutputWidthMismatch {
                expected_width: 256,
                actual_width: 255,
                ..
            }
        ));

        let mut values = Array1::zeros(256);
        values[17] = f32::NAN;
        let non_finite =
            admit_embedding_result(Ok(values), 256, 2, 1, EmbeddingMaskChoice::Full).unwrap_err();
        assert!(matches!(
            non_finite,
            ImportedPipelineError::EmbeddingOutputNonFinite {
                value_index: 17,
                ..
            }
        ));
    }

    #[test]
    fn failed_embedding_snapshot_cannot_enter_post_inference() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 160_000).unwrap();
        let shape = [1, geometry.frame_grid().frame_count as usize, 1];
        let snapshot = EmbeddingStageSnapshot::from_flat_parts(
            geometry,
            shape,
            vec![1.0; shape.iter().product()],
            vec![EmbeddingStageEntry::new(
                EmbeddingAvailability::InferenceFailed {
                    reason: crate::pipeline::EmbeddingFailureReason::ModelExecution,
                },
                None,
            )],
            EmbeddingReceipt {
                full_mask_fallback_count: 1,
                inference_failure_count: 1,
                ..EmbeddingReceipt::default()
            },
        )
        .unwrap();

        assert!(matches!(
            ensure_successful_embedding_snapshot(&snapshot),
            Err(ImportedPipelineError::FailedEmbeddingSnapshot { count: 1 })
        ));
    }

    #[test]
    fn embedding_contract_rejects_window_pooling_and_minimum_mismatches() {
        let manifest = SegmentationManifest::from_json(include_bytes!(
            "../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        let geometry = EmbeddingInputGeometry::new(16_000, 128_000, 399, 256).unwrap();
        assert!(validate_imported_embedding_geometry(&manifest, geometry, 100, 1).is_ok());
        assert!(matches!(
            validate_imported_embedding_geometry(
                &manifest,
                EmbeddingInputGeometry::new(16_000, 160_000, 589, 256).unwrap(),
                125,
                1,
            ),
            Err(ImportedPipelineError::EmbeddingContractMismatch {
                field: "window_samples",
                ..
            })
        ));
        assert!(matches!(
            validate_imported_embedding_geometry(&manifest, geometry, 99, 1),
            Err(ImportedPipelineError::EmbeddingContractMismatch {
                field: "pooling_frames",
                ..
            })
        ));
        assert!(matches!(
            validate_imported_embedding_geometry(&manifest, geometry, 100, 2),
            Err(ImportedPipelineError::EmbeddingContractMismatch {
                field: "min_num_samples",
                ..
            })
        ));
    }

    #[test]
    fn imported_policy_rejects_stale_plda_digest() {
        let manifest = SegmentationManifest::from_json(include_bytes!(
            "../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        let models_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/models");
        let plda = PldaTransform::from_dir(&models_dir).unwrap();
        assert!(matches!(
            validate_plda_contract(&manifest, &plda),
            Err(ImportedPipelineError::EmbeddingContractMismatch {
                field: "plda.sha256",
                ..
            })
        ));
    }
}
