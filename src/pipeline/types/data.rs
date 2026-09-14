use std::ops::Deref;

use ndarray::{Array2, Array3, s};

use crate::powerset::PowersetMapping;

use super::{FrameTiming, PipelineGeometry};

pub(crate) struct PendingEmbedding<'a> {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub audio: &'a [f32],
    pub mask: Vec<f32>,
    pub clean_mask: Vec<f32>,
}

pub(crate) struct PendingSplitEmbedding {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub fbank_idx: usize,
    pub weights: Vec<f32>,
}

/// Decoded powerset segmentations per chunk, shape (chunks, frames, speakers)
#[derive(Debug, Clone)]
pub struct DecodedSegmentations(pub Array3<f32>);

impl Deref for DecodedSegmentations {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Speaker embeddings per chunk, shape (chunks, speakers, embedding_dim)
#[derive(Debug, Clone)]
pub struct ChunkEmbeddings(pub Array3<f32>);

impl Deref for ChunkEmbeddings {
    type Target = Array3<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Reason an embedding slot has no usable vector because its mask is inactive
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InactiveEmbeddingReason {
    /// The decoded speaker mask contains no active frames
    NoActivity,
}

/// Closed reason why a model-backed embedding is unavailable
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EmbeddingFailureReason {
    /// The embedding runtime returned an execution error
    ModelExecution,
    /// The runtime returned a non-finite vector or the wrong width
    InvalidOutput,
    /// A legacy path supplied no typed availability state
    LegacyUnavailable,
}

/// Availability state for one chunk and local-speaker embedding slot
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum EmbeddingAvailability {
    /// A finite embedding vector was produced
    Available,
    /// The slot was not eligible for inference
    Inactive {
        /// Why the slot was not eligible
        reason: InactiveEmbeddingReason,
    },
    /// Inference ran but did not produce a usable vector
    InferenceFailed {
        /// Closed reason for the unavailable vector
        reason: EmbeddingFailureReason,
    },
}

/// Checked availability grid aligned with the chunk and local-speaker axes
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EmbeddingAvailabilityGrid {
    chunks: usize,
    speakers: usize,
    entries: Vec<EmbeddingAvailability>,
}

impl EmbeddingAvailabilityGrid {
    /// Return the number of chunk rows
    pub const fn chunks(&self) -> usize {
        self.chunks
    }

    /// Return the number of local-speaker columns
    pub const fn speakers(&self) -> usize {
        self.speakers
    }

    /// Return the availability for one chunk and local-speaker pair
    pub fn get(&self, chunk: usize, speaker: usize) -> Option<&EmbeddingAvailability> {
        if chunk >= self.chunks || speaker >= self.speakers {
            return None;
        }
        let index = chunk.checked_mul(self.speakers)?.checked_add(speaker)?;
        self.entries.get(index)
    }

    /// Iterate over entries in row-major chunk and local-speaker order
    pub fn iter(&self) -> impl Iterator<Item = &EmbeddingAvailability> {
        self.entries.iter()
    }

    fn from_embeddings(embeddings: &Array3<f32>) -> Self {
        let chunks = embeddings.shape()[0];
        let speakers = embeddings.shape()[1];
        let entries = (0..chunks)
            .flat_map(|chunk| {
                (0..speakers).map(move |speaker| {
                    let values = embeddings.slice(s![chunk, speaker, ..]);
                    if values.iter().all(|value| value.is_finite()) {
                        EmbeddingAvailability::Available
                    } else {
                        EmbeddingAvailability::InferenceFailed {
                            reason: EmbeddingFailureReason::LegacyUnavailable,
                        }
                    }
                })
            })
            .collect();
        Self {
            chunks,
            speakers,
            entries,
        }
    }
}

/// Counts recorded while selecting clean or full masks for embeddings
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct EmbeddingReceipt {
    /// Embeddings attempted with a clean, non-overlap mask
    pub clean_mask_count: usize,
    /// Embeddings that used the full mask after clean-mask fallback
    pub full_mask_fallback_count: usize,
    /// Slots skipped because their decoded mask was inactive
    pub inactive_count: usize,
    /// Slots whose model call failed or returned invalid values
    pub inference_failure_count: usize,
}

impl EmbeddingReceipt {
    fn from_availability(availability: &EmbeddingAvailabilityGrid) -> Self {
        let mut receipt = Self::default();
        for state in availability.iter() {
            match state {
                EmbeddingAvailability::Available => {}
                EmbeddingAvailability::Inactive { .. } => receipt.inactive_count += 1,
                EmbeddingAvailability::InferenceFailed { .. } => {
                    receipt.inference_failure_count += 1;
                }
            }
        }
        receipt
    }
}

/// One typed embedding-stage slot with an explicit availability state
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingStageEntry {
    availability: EmbeddingAvailability,
    values: Option<Vec<f32>>,
}

impl EmbeddingStageEntry {
    /// Create one slot for a checked embedding-stage snapshot
    pub fn new(availability: EmbeddingAvailability, values: Option<Vec<f32>>) -> Self {
        Self {
            availability,
            values,
        }
    }

    /// Return the typed availability state for this slot
    pub const fn availability(&self) -> &EmbeddingAvailability {
        &self.availability
    }

    /// Return the vector for an available slot, or no vector for an unavailable slot
    pub fn values(&self) -> Option<&[f32]> {
        self.values.as_deref()
    }
}

/// Immutable, checked output of the imported segmentation and embedding stages
#[derive(Clone, Debug)]
pub struct EmbeddingStageSnapshot {
    geometry: PipelineGeometry,
    segmentations: DecodedSegmentations,
    entries: Vec<EmbeddingStageEntry>,
    embedding_receipt: EmbeddingReceipt,
}

impl EmbeddingStageSnapshot {
    /// Build a snapshot from flat decoded masks and typed embedding slots
    pub fn from_flat_parts(
        geometry: PipelineGeometry,
        segmentation_shape: [usize; 3],
        segmentation_values: Vec<f32>,
        entries: Vec<EmbeddingStageEntry>,
        embedding_receipt: EmbeddingReceipt,
    ) -> Result<Self, super::PipelineError> {
        let segmentation_len = segmentation_shape
            .iter()
            .try_fold(1usize, |length, extent| length.checked_mul(*extent))
            .ok_or_else(|| {
                super::PipelineError::Invariant("segmentation shape overflow".to_owned())
            })?;
        if segmentation_values.len() != segmentation_len {
            return Err(super::PipelineError::Invariant(format!(
                "segmentation values {} do not match shape {segmentation_shape:?}",
                segmentation_values.len()
            )));
        }
        let entry_count = segmentation_shape[0]
            .checked_mul(segmentation_shape[2])
            .ok_or_else(|| {
                super::PipelineError::Invariant("embedding shape overflow".to_owned())
            })?;
        if entries.len() != entry_count {
            return Err(super::PipelineError::Invariant(format!(
                "embedding entries {} do not match shape {entry_count}",
                entries.len()
            )));
        }
        if segmentation_values
            .iter()
            .any(|value| !value.is_finite() || (*value != 0.0 && *value != 1.0))
        {
            return Err(super::PipelineError::Invariant(
                "decoded segmentation values are not finite binary masks".to_owned(),
            ));
        }
        let segmentations = DecodedSegmentations(
            Array3::from_shape_vec(segmentation_shape, segmentation_values).map_err(|error| {
                super::PipelineError::Invariant(format!("invalid segmentation shape: {error}"))
            })?,
        );
        Self::from_parts(geometry, segmentations, entries, embedding_receipt)
    }

    pub(crate) fn from_artifacts(artifacts: InferenceArtifacts) -> Self {
        let InferenceArtifacts {
            geometry,
            segmentations,
            embeddings,
            embedding_availability,
            embedding_receipt,
            #[cfg(feature = "_metrics")]
                stage_timings: _,
        } = artifacts;
        let entries = embedding_availability
            .iter()
            .enumerate()
            .map(|(entry_index, availability)| {
                let speakers = embeddings.shape()[1];
                let chunk = entry_index / speakers;
                let speaker = entry_index % speakers;
                let values = matches!(availability, EmbeddingAvailability::Available).then(|| {
                    embeddings
                        .slice(s![chunk, speaker, ..])
                        .iter()
                        .copied()
                        .collect()
                });
                EmbeddingStageEntry::new(availability.clone(), values)
            })
            .collect();
        Self {
            geometry,
            segmentations,
            entries,
            embedding_receipt,
        }
    }

    fn from_parts(
        geometry: PipelineGeometry,
        segmentations: DecodedSegmentations,
        entries: Vec<EmbeddingStageEntry>,
        embedding_receipt: EmbeddingReceipt,
    ) -> Result<Self, super::PipelineError> {
        let chunks = segmentations.shape()[0];
        let speakers = segmentations.shape()[2];
        if segmentations.shape()[1] != geometry.frame_grid().frame_count as usize {
            return Err(super::PipelineError::Invariant(
                "segmentation frame count does not match geometry".to_owned(),
            ));
        }
        let width = crate::inference::embedding::EMBEDDING_WIDTH;
        let mut values = Array3::from_elem((chunks, speakers, width), f32::NAN);
        let mut availability = Vec::with_capacity(entries.len());
        let mut inactive_count = 0;
        let mut inference_failure_count = 0;
        for (entry_index, entry) in entries.iter().enumerate() {
            match entry.availability() {
                EmbeddingAvailability::Available => {
                    let vector = entry.values().ok_or_else(|| {
                        super::PipelineError::Invariant(format!(
                            "available embedding {entry_index} has no vector"
                        ))
                    })?;
                    if vector.len() != width || vector.iter().any(|value| !value.is_finite()) {
                        return Err(super::PipelineError::Invariant(format!(
                            "available embedding {entry_index} has invalid values"
                        )));
                    }
                    let chunk = entry_index / speakers;
                    let speaker = entry_index % speakers;
                    values
                        .slice_mut(s![chunk, speaker, ..])
                        .assign(&ndarray::ArrayView1::from(vector));
                }
                EmbeddingAvailability::Inactive { .. } => {
                    if entry.values().is_some() {
                        return Err(super::PipelineError::Invariant(format!(
                            "inactive embedding {entry_index} unexpectedly has a vector"
                        )));
                    }
                    inactive_count += 1;
                }
                EmbeddingAvailability::InferenceFailed { .. } => {
                    if entry.values().is_some() {
                        return Err(super::PipelineError::Invariant(format!(
                            "failed embedding {entry_index} unexpectedly has a vector"
                        )));
                    }
                    inference_failure_count += 1;
                }
            }
            availability.push(entry.availability().clone());
        }
        if embedding_receipt.inactive_count != inactive_count
            || embedding_receipt.inference_failure_count != inference_failure_count
        {
            return Err(super::PipelineError::Invariant(
                "embedding receipt counts do not match availability".to_owned(),
            ));
        }
        if embedding_receipt
            .clean_mask_count
            .saturating_add(embedding_receipt.full_mask_fallback_count)
            != chunks
                .saturating_mul(speakers)
                .saturating_sub(inactive_count)
        {
            return Err(super::PipelineError::Invariant(
                "embedding mask receipt counts do not match eligible slots".to_owned(),
            ));
        }
        let artifacts = InferenceArtifacts::try_new_checked(
            geometry.clone(),
            segmentations.clone(),
            ChunkEmbeddings(values),
            EmbeddingAvailabilityGrid {
                chunks,
                speakers,
                entries: availability,
            },
            embedding_receipt,
            #[cfg(feature = "_metrics")]
            None,
        )?;
        let InferenceArtifacts {
            geometry,
            segmentations,
            embedding_availability,
            embedding_receipt,
            ..
        } = artifacts;
        debug_assert_eq!(embedding_availability.iter().count(), entries.len());
        Ok(Self {
            geometry,
            segmentations,
            entries,
            embedding_receipt,
        })
    }

    pub(crate) fn into_artifacts(self) -> Result<InferenceArtifacts, super::PipelineError> {
        let Self {
            geometry,
            segmentations,
            entries,
            embedding_receipt,
        } = self;
        let chunks = segmentations.shape()[0];
        let speakers = segmentations.shape()[2];
        let width = crate::inference::embedding::EMBEDDING_WIDTH;
        let mut values = Array3::from_elem((chunks, speakers, width), f32::NAN);
        let mut availability = Vec::with_capacity(entries.len());
        for (entry_index, entry) in entries.into_iter().enumerate() {
            if let EmbeddingAvailability::Available = entry.availability {
                let vector = entry.values.ok_or_else(|| {
                    super::PipelineError::Invariant(format!(
                        "available embedding {entry_index} has no vector"
                    ))
                })?;
                let chunk = entry_index / speakers;
                let speaker = entry_index % speakers;
                values
                    .slice_mut(s![chunk, speaker, ..])
                    .assign(&ndarray::ArrayView1::from(vector.as_slice()));
            }
            availability.push(entry.availability);
        }
        InferenceArtifacts::try_new_checked(
            geometry,
            segmentations,
            ChunkEmbeddings(values),
            EmbeddingAvailabilityGrid {
                chunks,
                speakers,
                entries: availability,
            },
            embedding_receipt,
            #[cfg(feature = "_metrics")]
            None,
        )
    }

    /// Return the geometry bound to the snapshot
    pub const fn geometry(&self) -> &PipelineGeometry {
        &self.geometry
    }

    /// Return the decoded segmentation tensor shape
    pub fn segmentation_shape(&self) -> [usize; 3] {
        let shape = self.segmentations.shape();
        [shape[0], shape[1], shape[2]]
    }

    /// Return decoded segmentation values in row-major order
    pub fn segmentation_values(&self) -> Vec<f32> {
        self.segmentations.iter().copied().collect()
    }

    /// Return typed embedding slots in row-major chunk and local-speaker order
    pub fn entries(&self) -> &[EmbeddingStageEntry] {
        &self.entries
    }

    /// Return embedding-stage counts needed to reconstruct inference artifacts
    pub const fn embedding_receipt(&self) -> EmbeddingReceipt {
        self.embedding_receipt
    }
}

#[derive(Debug)]
pub(crate) struct TypedChunkEmbeddings {
    pub(crate) chunks: usize,
    pub(crate) speakers: usize,
    pub(crate) embedding_width: usize,
    pub(crate) entries: Vec<TypedEmbedding>,
}

#[derive(Debug)]
pub(crate) struct TypedEmbedding {
    pub(crate) availability: EmbeddingAvailability,
    pub(crate) values: Option<Vec<f32>>,
    pub(crate) mask_choice: Option<EmbeddingMaskChoice>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum EmbeddingMaskChoice {
    Clean,
    Full,
}

impl TypedChunkEmbeddings {
    pub(crate) fn new(chunks: usize, speakers: usize, embedding_width: usize) -> Self {
        let entry_count = chunks.saturating_mul(speakers);
        Self {
            chunks,
            speakers,
            embedding_width,
            entries: Vec::with_capacity(entry_count.min(1024)),
        }
    }

    pub(crate) fn push(&mut self, embedding: TypedEmbedding) {
        self.entries.push(embedding);
    }
}

/// Number of active speakers per chunk
#[derive(Debug, Clone)]
pub struct SpeakerCountTrack(pub Vec<usize>);

impl Deref for SpeakerCountTrack {
    type Target = Vec<usize>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Cluster assignments per chunk-speaker pair, shape (chunks, speakers)
///
/// Values are cluster IDs (-1 for unassigned)
#[derive(Debug, Clone)]
pub struct ChunkSpeakerClusters(pub Array2<i32>);

impl Deref for ChunkSpeakerClusters {
    type Target = Array2<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Frame-level binary speaker activations with their exact output timing
#[derive(Debug, Clone)]
pub struct DiscreteDiarization {
    activations: Array2<f32>,
    timing: FrameTiming,
}

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.activations
    }
}

impl DiscreteDiarization {
    /// Create activations checked against a pipeline geometry
    pub fn try_new(
        activations: Array2<f32>,
        geometry: &PipelineGeometry,
    ) -> Result<Self, super::PipelineError> {
        if activations.nrows() != geometry.output_frames() {
            return Err(super::PipelineError::Invariant(format!(
                "activation frames {} do not match geometry output frames {}",
                activations.nrows(),
                geometry.output_frames()
            )));
        }
        let timing = geometry.frame_timing()?;
        Ok(Self {
            activations,
            timing,
        })
    }

    pub(crate) fn with_timing(activations: Array2<f32>, timing: FrameTiming) -> Self {
        Self {
            activations,
            timing,
        }
    }

    pub(crate) fn map_activations(
        &self,
        activations: Array2<f32>,
    ) -> Result<Self, super::PipelineError> {
        if activations.raw_dim() != self.activations.raw_dim() {
            return Err(super::PipelineError::Invariant(
                "activation transform changed the diarization shape".to_owned(),
            ));
        }
        Ok(Self::with_timing(activations, self.timing))
    }

    /// Return the exact frame timing owned by these activations
    pub const fn timing(&self) -> FrameTiming {
        self.timing
    }

    /// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
    pub fn make_exclusive(&mut self) {
        crate::reconstruct::make_exclusive(&mut self.activations);
    }

    /// Convert frame activations to time-stamped speaker segments using owned timing
    pub fn to_segments(&self) -> Vec<crate::segment::Segment> {
        crate::segment::to_segments_with_timing(&self.activations, self.timing)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FrameActivations(pub(crate) Array2<f32>);

impl Deref for FrameActivations {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub(crate) struct RawSegmentationWindows(pub Vec<Array2<f32>>);

impl RawSegmentationWindows {
    pub(crate) fn decode(
        self,
        powerset: &PowersetMapping,
    ) -> Result<DecodedSegmentations, crate::powerset::PowersetDecodeError> {
        let mut windows = self.0.into_iter();
        let Some(first_window) = windows.next() else {
            return Ok(DecodedSegmentations(Array3::zeros((0, 0, 0))));
        };

        let num_windows = windows.len() + 1;
        let first = powerset.hard_decode(&first_window)?;
        let mut stacked = Array3::<f32>::zeros((num_windows, first.nrows(), first.ncols()));
        stacked.slice_mut(s![0, .., ..]).assign(&first);

        for (window_idx, window) in windows.enumerate() {
            let decoded = powerset.hard_decode(&window)?;
            stacked
                .slice_mut(s![window_idx + 1, .., ..])
                .assign(&decoded);
        }

        Ok(DecodedSegmentations(stacked))
    }
}

/// Input for batch diarization
pub struct BatchInput<'a> {
    /// Mono 16kHz audio samples
    pub audio: &'a [f32],
    /// Identifier used in RTTM output lines
    pub file_id: &'a str,
}

/// Intermediate results from segmentation and embedding inference
#[derive(Clone)]
pub struct InferenceArtifacts {
    pub(crate) geometry: PipelineGeometry,
    pub(crate) segmentations: DecodedSegmentations,
    pub(crate) embeddings: ChunkEmbeddings,
    pub(crate) embedding_availability: EmbeddingAvailabilityGrid,
    pub(crate) embedding_receipt: EmbeddingReceipt,
    #[cfg(feature = "_metrics")]
    pub(crate) stage_timings: Option<InferenceStageTimings>,
}

impl InferenceArtifacts {
    /// Checked artifacts with matching chunk, speaker, embedding, and geometry extents
    pub(crate) fn try_new(
        geometry: PipelineGeometry,
        segmentations: DecodedSegmentations,
        embeddings: ChunkEmbeddings,
        #[cfg(feature = "_metrics")] stage_timings: Option<InferenceStageTimings>,
    ) -> Result<Self, super::PipelineError> {
        let availability = EmbeddingAvailabilityGrid::from_embeddings(&embeddings.0);
        let receipt = EmbeddingReceipt::from_availability(&availability);
        Self::try_new_checked(
            geometry,
            segmentations,
            embeddings,
            availability,
            receipt,
            #[cfg(feature = "_metrics")]
            stage_timings,
        )
    }

    pub(crate) fn try_new_from_outcomes(
        geometry: PipelineGeometry,
        segmentations: DecodedSegmentations,
        typed: TypedChunkEmbeddings,
        #[cfg(feature = "_metrics")] stage_timings: Option<InferenceStageTimings>,
    ) -> Result<Self, super::PipelineError> {
        let chunks = segmentations.0.shape()[0];
        let speakers = segmentations.0.shape()[2];
        if typed.chunks != chunks || typed.speakers != speakers {
            return Err(super::PipelineError::Invariant(format!(
                "typed embedding shape {}x{} does not match segmentation shape {}x{}",
                typed.chunks, typed.speakers, chunks, speakers
            )));
        }
        let expected_entries = chunks.checked_mul(speakers).ok_or_else(|| {
            super::PipelineError::Invariant("typed embedding shape overflow".to_owned())
        })?;
        if typed.entries.len() != expected_entries {
            return Err(super::PipelineError::Invariant(format!(
                "typed embedding entries {} do not match shape {expected_entries}",
                typed.entries.len()
            )));
        }

        let mut embeddings =
            Array3::<f32>::from_elem((chunks, speakers, typed.embedding_width), f32::NAN);
        let mut availability = Vec::with_capacity(expected_entries);
        let mut receipt = EmbeddingReceipt::default();
        for (entry_index, entry) in typed.entries.into_iter().enumerate() {
            let TypedEmbedding {
                availability: state,
                values,
                mask_choice,
            } = entry;
            if let Some(choice) = mask_choice {
                match choice {
                    EmbeddingMaskChoice::Clean => receipt.clean_mask_count += 1,
                    EmbeddingMaskChoice::Full => receipt.full_mask_fallback_count += 1,
                }
            }
            match &state {
                EmbeddingAvailability::Available => {
                    let values = values.ok_or_else(|| {
                        super::PipelineError::Invariant(format!(
                            "available embedding {entry_index} has no vector"
                        ))
                    })?;
                    if values.len() != typed.embedding_width
                        || values.iter().any(|value| !value.is_finite())
                    {
                        return Err(super::PipelineError::Invariant(format!(
                            "available embedding {entry_index} has invalid values"
                        )));
                    }
                    let chunk = entry_index / speakers;
                    let speaker = entry_index % speakers;
                    embeddings
                        .slice_mut(s![chunk, speaker, ..])
                        .assign(&ndarray::ArrayView1::from(values.as_slice()));
                }
                EmbeddingAvailability::Inactive { .. } => {
                    receipt.inactive_count += 1;
                    if values.is_some() {
                        return Err(super::PipelineError::Invariant(format!(
                            "inactive embedding {entry_index} unexpectedly has a vector"
                        )));
                    }
                }
                EmbeddingAvailability::InferenceFailed { .. } => {
                    receipt.inference_failure_count += 1;
                    if values.is_some() {
                        return Err(super::PipelineError::Invariant(format!(
                            "failed embedding {entry_index} unexpectedly has a vector"
                        )));
                    }
                }
            }
            availability.push(state);
        }

        Self::try_new_checked(
            geometry,
            segmentations,
            ChunkEmbeddings(embeddings),
            EmbeddingAvailabilityGrid {
                chunks,
                speakers,
                entries: availability,
            },
            receipt,
            #[cfg(feature = "_metrics")]
            stage_timings,
        )
    }

    fn try_new_checked(
        geometry: PipelineGeometry,
        segmentations: DecodedSegmentations,
        embeddings: ChunkEmbeddings,
        embedding_availability: EmbeddingAvailabilityGrid,
        embedding_receipt: EmbeddingReceipt,
        #[cfg(feature = "_metrics")] stage_timings: Option<InferenceStageTimings>,
    ) -> Result<Self, super::PipelineError> {
        let chunks = segmentations.0.shape()[0];
        let speakers = segmentations.0.shape()[2];
        if embeddings.0.shape()[0] != chunks {
            return Err(super::PipelineError::Invariant(format!(
                "embedding chunks {} do not match segmentation chunks {chunks}",
                embeddings.0.shape()[0]
            )));
        }
        if geometry.start_frames().len() != chunks {
            return Err(super::PipelineError::Invariant(format!(
                "geometry chunks {} do not match segmentation chunks {chunks}",
                geometry.start_frames().len()
            )));
        }
        if chunks > 0 && segmentations.0.shape()[1] != geometry.frame_grid().frame_count as usize {
            return Err(super::PipelineError::Invariant(format!(
                "segmentation frames {} do not match geometry frames {}",
                segmentations.0.shape()[1],
                geometry.frame_grid().frame_count
            )));
        }
        if embeddings.0.shape()[1] != speakers {
            return Err(super::PipelineError::Invariant(format!(
                "embedding speakers {} do not match segmentation speakers {speakers}",
                embeddings.0.shape()[1]
            )));
        }
        if embeddings.0.shape()[2] != crate::inference::embedding::EMBEDDING_WIDTH {
            return Err(super::PipelineError::Invariant(format!(
                "embedding width {} is invalid",
                embeddings.0.shape()[2]
            )));
        }
        if embedding_availability.chunks != chunks || embedding_availability.speakers != speakers {
            return Err(super::PipelineError::Invariant(
                "embedding availability shape does not match segmentations".to_owned(),
            ));
        }
        Ok(Self {
            geometry,
            segmentations,
            embeddings,
            embedding_availability,
            embedding_receipt,
            #[cfg(feature = "_metrics")]
            stage_timings,
        })
    }

    pub(crate) fn empty_with_geometry(geometry: PipelineGeometry) -> Self {
        Self {
            geometry,
            segmentations: DecodedSegmentations(ndarray::Array3::zeros((0, 0, 0))),
            embeddings: ChunkEmbeddings(ndarray::Array3::zeros((0, 0, 0))),
            embedding_availability: EmbeddingAvailabilityGrid {
                chunks: 0,
                speakers: 0,
                entries: Vec::new(),
            },
            embedding_receipt: EmbeddingReceipt::default(),
            #[cfg(feature = "_metrics")]
            stage_timings: None,
        }
    }

    /// Return the checked geometry used by every inference stage
    pub fn geometry(&self) -> &PipelineGeometry {
        &self.geometry
    }

    /// Return typed availability for every chunk and local-speaker slot
    pub fn embedding_availability(&self) -> &EmbeddingAvailabilityGrid {
        &self.embedding_availability
    }

    /// Return clean-mask, fallback, inactive, and failure counts
    pub const fn embedding_receipt(&self) -> EmbeddingReceipt {
        self.embedding_receipt
    }

    /// Return detailed chunk-inference timings when metrics are enabled and available
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub const fn stage_timings(&self) -> Option<InferenceStageTimings> {
        self.stage_timings
    }

    /// Return decoded segmentation windows for metrics experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub const fn segmentations(&self) -> &DecodedSegmentations {
        &self.segmentations
    }

    /// Return speaker embeddings for metrics experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub const fn embeddings(&self) -> &ChunkEmbeddings {
        &self.embeddings
    }

    /// Count embeddings that satisfy the requested clean-frame duration
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub fn usable_training_embedding_count(
        &self,
        clean_frame_duration: super::super::CleanFrameDuration,
    ) -> usize {
        self.embeddings
            .training_set(&self.segmentations, &self.geometry, clean_frame_duration)
            .0
            .nrows()
    }
}

/// Detailed timings from the native chunk inference path
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Clone, Copy, Debug)]
pub struct InferenceStageTimings {
    /// Time spent by the segmentation worker, which can overlap embedding work
    pub segmentation_seconds: f64,
    /// Time from embedding orchestration start to completion
    pub embedding_seconds: f64,
    /// Sum of Core ML chunk-embedding prediction call durations
    pub prediction_seconds: f64,
    /// Sum of filterbank preparation durations across preparation workers
    pub filterbank_preparation_seconds: f64,
    /// Sum of speaker-mask preparation durations
    pub mask_preparation_seconds: f64,
    /// Complete concurrent inference wall time
    pub total_seconds: f64,
    /// Number of embedding chunks submitted
    pub chunk_count: usize,
    /// Whether preparation and prediction used the pipelined path
    pub pipelined: bool,
}

/// Complete output from a diarization run
pub struct DiarizationResult {
    /// Decoded segmentations from the powerset model
    pub segmentations: DecodedSegmentations,
    /// Speaker embeddings extracted from each chunk
    pub embeddings: ChunkEmbeddings,
    /// Typed availability for each chunk and local-speaker embedding slot
    pub embedding_availability: EmbeddingAvailabilityGrid,
    /// Clean-mask, fallback, inactive, and failure counts
    pub embedding_receipt: EmbeddingReceipt,
    /// Number of active speakers per chunk
    pub speaker_count: SpeakerCountTrack,
    /// Cluster assignment for each chunk-speaker pair
    pub hard_clusters: ChunkSpeakerClusters,
    /// Frame-level binary speaker activations after reconstruction
    pub discrete_diarization: DiscreteDiarization,
    /// Merged speaker segments (time-stamped speaker turns)
    pub segments: Vec<crate::segment::Segment>,
    pub(crate) geometry: PipelineGeometry,
}

impl DiarizationResult {
    /// Return the checked geometry used to create this result
    pub fn geometry(&self) -> &PipelineGeometry {
        &self.geometry
    }

    /// Render RTTM output with the given file identifier
    pub fn rttm(&self, file_id: &str) -> String {
        crate::segment::to_rttm(&self.segments, file_id)
    }
}

pub(crate) enum InferencePath {
    Sequential,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EmbeddingPath {
    Masked,
    Split,
    MultiMask,
}

#[cfg(test)]
mod tests {
    use super::super::layout::PipelineGeometry;
    use super::*;
    use crate::imported_segmentation::{ChunkGeometry, SegmentationManifest};
    use crate::inference::embedding::EMBEDDING_WIDTH;
    use ndarray::{Array2, Array3};

    #[test]
    fn try_new_rejects_mismatched_chunk_counts() {
        let layout = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 176_000).unwrap();
        let segmentations = DecodedSegmentations(Array3::zeros((2, 589, 3)));
        let embeddings = ChunkEmbeddings(Array3::zeros((1, 3, EMBEDDING_WIDTH)));
        assert!(
            InferenceArtifacts::try_new(
                layout,
                segmentations,
                embeddings,
                #[cfg(feature = "_metrics")]
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn try_new_rejects_invalid_embedding_width() {
        let layout = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 160_000).unwrap();
        let segmentations = DecodedSegmentations(Array3::zeros((1, 589, 3)));
        let embeddings = ChunkEmbeddings(Array3::zeros((1, 3, 8)));
        assert!(
            InferenceArtifacts::try_new(
                layout,
                segmentations,
                embeddings,
                #[cfg(feature = "_metrics")]
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn try_new_accepts_matching_extents() {
        let layout = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 160_000).unwrap();
        let segmentations = DecodedSegmentations(Array3::zeros((1, 589, 3)));
        let embeddings = ChunkEmbeddings(Array3::zeros((1, 3, EMBEDDING_WIDTH)));
        assert!(
            InferenceArtifacts::try_new(
                layout,
                segmentations,
                embeddings,
                #[cfg(feature = "_metrics")]
                None,
            )
            .is_ok()
        );
    }

    #[test]
    fn embedding_stage_snapshot_round_trips_typed_slots() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 160_000).unwrap();
        let shape = [1, geometry.frame_grid().frame_count as usize, 3];
        let entries = vec![
            EmbeddingStageEntry::new(
                EmbeddingAvailability::Available,
                Some(vec![1.0; crate::inference::embedding::EMBEDDING_WIDTH]),
            ),
            EmbeddingStageEntry::new(
                EmbeddingAvailability::Inactive {
                    reason: InactiveEmbeddingReason::NoActivity,
                },
                None,
            ),
            EmbeddingStageEntry::new(
                EmbeddingAvailability::InferenceFailed {
                    reason: EmbeddingFailureReason::ModelExecution,
                },
                None,
            ),
        ];
        let snapshot = EmbeddingStageSnapshot::from_flat_parts(
            geometry,
            shape,
            vec![0.0; shape[0] * shape[1] * shape[2]],
            entries,
            EmbeddingReceipt {
                clean_mask_count: 1,
                full_mask_fallback_count: 1,
                inactive_count: 1,
                inference_failure_count: 1,
            },
        )
        .unwrap();
        assert_eq!(snapshot.segmentation_shape(), shape);
        assert!(snapshot.entries()[1].values().is_none());
        assert!(snapshot.into_artifacts().is_ok());
    }

    #[test]
    fn try_new_admits_exact_aligned_tail_artifacts() {
        let layout = PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 192_000).unwrap();
        let segmentations = DecodedSegmentations(Array3::zeros((4, 589, 3)));
        let embeddings = ChunkEmbeddings(Array3::zeros((4, 3, EMBEDDING_WIDTH)));

        let artifacts = InferenceArtifacts::try_new(
            layout,
            segmentations,
            embeddings,
            #[cfg(feature = "_metrics")]
            None,
        )
        .unwrap();

        assert_eq!(artifacts.geometry.chunk_count(), 4);
        assert_eq!(artifacts.segmentations.0.shape()[0], 4);
    }

    #[test]
    fn empty_artifacts_have_zero_chunks() {
        let artifacts = InferenceArtifacts::empty_with_geometry(
            PipelineGeometry::from_legacy(16_000, 160_000, 16_000, 0).unwrap(),
        );
        assert_eq!(artifacts.segmentations.0.shape()[0], 0);
        assert_eq!(artifacts.embeddings.0.shape()[0], 0);
        assert!(artifacts.geometry.start_frames().is_empty());
    }

    #[test]
    fn imported_diarization_uses_first_and_last_exact_frame_timing() {
        let mut manifest = SegmentationManifest::from_json(include_bytes!(
            "../../../fixtures/wavlm_bridge/manifest.json"
        ))
        .unwrap();
        manifest.audio.sample_count = 1;
        manifest.geometry.chunks = vec![ChunkGeometry {
            index: 0,
            padding_samples: 127_999,
            start_samples: 0,
            valid_samples: 1,
        }];
        manifest.geometry.output_extent.end_samples = 128_400;
        let geometry =
            PipelineGeometry::from_imported(&manifest.audio, &manifest.geometry).unwrap();

        let mut activations = Array2::<f32>::zeros((geometry.output_frames(), 1));
        activations[[0, 0]] = 1.0;
        activations[[geometry.output_frames() - 1, 0]] = 1.0;
        let diarization = DiscreteDiarization::try_new(activations, &geometry).unwrap();

        let segments = diarization.to_segments();

        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 200.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[0].end - 520.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[1].start - 128_200.0 / 16_000.0).abs() < 1e-12);
        assert!((segments[1].end - 128_200.0 / 16_000.0).abs() < 1e-12);
    }

    #[test]
    fn typed_unavailable_embeddings_become_nan_only_at_the_artifact_boundary() {
        let geometry = PipelineGeometry::from_legacy(16_000, 160_000, 160_000, 160_000).unwrap();
        let segmentations = DecodedSegmentations(Array3::zeros((1, 589, 2)));
        let mut typed = TypedChunkEmbeddings::new(1, 2, EMBEDDING_WIDTH);
        typed.push(TypedEmbedding {
            availability: EmbeddingAvailability::Inactive {
                reason: InactiveEmbeddingReason::NoActivity,
            },
            values: None,
            mask_choice: None,
        });
        typed.push(TypedEmbedding {
            availability: EmbeddingAvailability::InferenceFailed {
                reason: EmbeddingFailureReason::InvalidOutput,
            },
            values: None,
            mask_choice: Some(EmbeddingMaskChoice::Full),
        });
        let artifacts = InferenceArtifacts::try_new_from_outcomes(
            geometry,
            segmentations,
            typed,
            #[cfg(feature = "_metrics")]
            None,
        )
        .unwrap();

        assert!(artifacts.embeddings.0.iter().all(|value| value.is_nan()));
        assert!(matches!(
            artifacts.embedding_availability.get(0, 0),
            Some(EmbeddingAvailability::Inactive { .. })
        ));
        assert!(matches!(
            artifacts.embedding_availability.get(0, 1),
            Some(EmbeddingAvailability::InferenceFailed {
                reason: EmbeddingFailureReason::InvalidOutput
            })
        ));
        assert_eq!(artifacts.embedding_receipt.full_mask_fallback_count, 1);
        assert_eq!(artifacts.embedding_receipt.inactive_count, 1);
        assert_eq!(artifacts.embedding_receipt.inference_failure_count, 1);
    }
}
