use std::ops::Deref;

use ndarray::{Array2, Array3, s};

use crate::powerset::PowersetMapping;

use super::ChunkLayout;

pub(in crate::pipeline) struct PendingEmbedding<'a> {
    pub chunk_idx: usize,
    pub speaker_idx: usize,
    pub audio: &'a [f32],
    pub mask: Vec<f32>,
    pub clean_mask: Vec<f32>,
}

pub(in crate::pipeline) struct PendingSplitEmbedding {
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

/// Frame-level binary speaker activations, shape (frames, speakers)
#[derive(Debug, Clone)]
pub struct DiscreteDiarization(pub Array2<f32>);

impl Deref for DiscreteDiarization {
    type Target = Array2<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DiscreteDiarization {
    /// Zero out all but the highest-scoring speaker in each frame, making activations exclusive
    pub fn make_exclusive(&mut self) {
        crate::reconstruct::make_exclusive(&mut self.0);
    }

    /// Convert frame activations to time-stamped speaker segments using default frame timing.
    pub fn to_segments(&self) -> Vec<crate::segment::Segment> {
        self.to_segments_with(
            crate::pipeline::FRAME_STEP_SECONDS,
            crate::pipeline::FRAME_DURATION_SECONDS,
        )
    }

    /// Convert frame activations to time-stamped speaker segments using custom frame timing.
    pub fn to_segments_with(
        &self,
        frame_step_seconds: f64,
        frame_duration_seconds: f64,
    ) -> Vec<crate::segment::Segment> {
        crate::segment::to_segments(&self.0, frame_step_seconds, frame_duration_seconds)
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

pub(in crate::pipeline) struct RawSegmentationWindows(pub Vec<Array2<f32>>);

impl RawSegmentationWindows {
    pub(in crate::pipeline) fn decode(
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
    pub(in crate::pipeline) layout: ChunkLayout,
    pub(in crate::pipeline) segmentations: DecodedSegmentations,
    pub(in crate::pipeline) embeddings: ChunkEmbeddings,
    #[cfg(feature = "_metrics")]
    pub(in crate::pipeline) stage_timings: Option<InferenceStageTimings>,
}

impl InferenceArtifacts {
    /// Checked artifacts with matching chunk, speaker, embedding, and layout extents
    pub(in crate::pipeline) fn try_new(
        layout: ChunkLayout,
        segmentations: DecodedSegmentations,
        embeddings: ChunkEmbeddings,
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
        if layout.start_frames.len() != chunks {
            return Err(super::PipelineError::Invariant(format!(
                "layout chunks {} do not match segmentation chunks {chunks}",
                layout.start_frames.len()
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
        Ok(Self {
            layout,
            segmentations,
            embeddings,
            #[cfg(feature = "_metrics")]
            stage_timings,
        })
    }

    pub(in crate::pipeline) fn empty_with_layout(layout: ChunkLayout) -> Self {
        Self {
            layout: layout.with_num_chunks(0),
            segmentations: DecodedSegmentations(ndarray::Array3::zeros((0, 0, 0))),
            embeddings: ChunkEmbeddings(ndarray::Array3::zeros((0, 0, 0))),
            #[cfg(feature = "_metrics")]
            stage_timings: None,
        }
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
            .training_set(&self.segmentations, clean_frame_duration)
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
    /// Number of active speakers per chunk
    pub speaker_count: SpeakerCountTrack,
    /// Cluster assignment for each chunk-speaker pair
    pub hard_clusters: ChunkSpeakerClusters,
    /// Frame-level binary speaker activations after reconstruction
    pub discrete_diarization: DiscreteDiarization,
    /// Merged speaker segments (time-stamped speaker turns)
    pub segments: Vec<crate::segment::Segment>,
}

impl DiarizationResult {
    /// Render RTTM output with the given file identifier
    pub fn rttm(&self, file_id: &str) -> String {
        crate::segment::to_rttm(&self.segments, file_id)
    }
}

pub(in crate::pipeline) enum InferencePath {
    Sequential,
    Concurrent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::pipeline) enum EmbeddingPath {
    Masked,
    Split,
    MultiMask,
}

#[cfg(test)]
mod tests {
    use super::super::layout::ChunkLayout;
    use super::*;
    use crate::inference::embedding::EMBEDDING_WIDTH;
    use ndarray::Array3;

    #[test]
    fn try_new_rejects_mismatched_chunk_counts() {
        let layout = ChunkLayout::new(1.0, 16_000, 160_000, 2);
        let segmentations = DecodedSegmentations(Array3::zeros((2, 4, 3)));
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
        let layout = ChunkLayout::new(1.0, 16_000, 160_000, 1);
        let segmentations = DecodedSegmentations(Array3::zeros((1, 4, 3)));
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
        let layout = ChunkLayout::new(1.0, 16_000, 160_000, 1);
        let segmentations = DecodedSegmentations(Array3::zeros((1, 4, 3)));
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
    fn empty_artifacts_have_zero_chunks() {
        let artifacts =
            InferenceArtifacts::empty_with_layout(ChunkLayout::new(1.0, 16_000, 160_000, 4));
        assert_eq!(artifacts.segmentations.0.shape()[0], 0);
        assert_eq!(artifacts.embeddings.0.shape()[0], 0);
        assert!(artifacts.layout.start_frames.is_empty());
    }
}
