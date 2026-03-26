mod data;
mod error;
mod extract;
mod layout;

pub(crate) use data::FrameActivations;
pub use data::{
    BatchInput, ChunkEmbeddings, ChunkSpeakerClusters, DecodedSegmentations, DiarizationResult,
    DiscreteDiarization, InferenceArtifacts, SpeakerCountTrack,
};
pub(super) use data::{
    EmbeddingPath, InferencePath, PendingEmbedding, PendingSplitEmbedding, RawSegmentationWindows,
};
pub use error::PipelineError;
pub(super) use extract::{Array3Writer, EmbeddingStorage, flush_masked, flush_split};
pub(super) use layout::{ChunkLayout, chunk_audio_raw};
#[cfg(test)]
pub(super) use layout::{chunk_start_frames, total_output_frames};
