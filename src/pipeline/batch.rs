use super::{DiarizationResult, PipelineError};

/// One owned, mono 16 kHz input for streamed batch processing
pub struct OwnedBatchInput {
    /// Audio samples in file order
    pub audio: Vec<f32>,
    /// Identifier used for RTTM output
    pub file_id: String,
}

/// One batch result paired with its original input metadata
pub struct BatchOutput {
    /// Identifier of the input file
    pub file_id: String,
    /// Duration of the input audio in seconds
    pub audio_secs: f64,
    /// Complete diarization output
    pub result: DiarizationResult,
}

/// An input producer or diarization failure in a streamed batch
#[derive(Debug, thiserror::Error)]
pub enum BatchStreamError<E> {
    /// The input producer failed
    #[error("{0}")]
    Input(E),
    /// Inference, clustering, or reconstruction failed
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
}
