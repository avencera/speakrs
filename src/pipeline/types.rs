mod data;
mod error;
mod extract;
mod layout;

pub(crate) use data::FrameActivations;
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
pub use data::InferenceStageTimings;
#[cfg(all(feature = "_metrics", feature = "coreml"))]
pub(crate) use data::InferenceStageTimings as InternalInferenceStageTimings;
pub use data::{
    BatchInput, ChunkEmbeddings, ChunkSpeakerClusters, DecodedSegmentations, DiarizationResult,
    DiscreteDiarization, EmbeddingAvailability, EmbeddingAvailabilityGrid, EmbeddingFailureReason,
    EmbeddingReceipt, EmbeddingStageEntry, EmbeddingStageSnapshot, InactiveEmbeddingReason,
    InferenceArtifacts, SpeakerCountTrack,
};
pub(crate) use data::{
    EmbeddingMaskChoice, EmbeddingPath, InferencePath, PendingEmbedding, PendingSplitEmbedding,
    RawSegmentationWindows, TypedChunkEmbeddings, TypedEmbedding,
};
pub use error::PipelineError;
pub(crate) use extract::{Array3Writer, EmbeddingStorage, flush_masked, flush_split};
pub(crate) use layout::chunk_audio_raw;
pub use layout::{ChunkExtent, FrameTiming, PipelineGeometry, PipelineGeometryError};
#[cfg(test)]
pub(crate) use layout::{chunk_start_frames, total_output_frames};
