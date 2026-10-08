use std::thread::{Scope, ScopedJoinHandle};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::clustering::plda::PldaTransform;

use super::{DiarizationResult, InferenceArtifacts, PipelineConfig, PipelineError, post_inference};

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

struct PostJob {
    artifacts: InferenceArtifacts,
    file_id: String,
    audio_secs: f64,
    start: Instant,
    inference_ms: u128,
    result: Sender<Result<BatchOutput, PipelineError>>,
}

/// Owns one post-inference worker and at most one pending file
pub(super) struct BatchPostProcessor<'scope> {
    jobs: Option<Sender<PostJob>>,
    pending: Option<Receiver<Result<BatchOutput, PipelineError>>>,
    worker: Option<ScopedJoinHandle<'scope, ()>>,
    outputs: Vec<BatchOutput>,
}

impl<'scope> BatchPostProcessor<'scope> {
    pub(super) fn new<'env>(
        scope: &'scope Scope<'scope, 'env>,
        plda: &'env PldaTransform,
        config: &'env PipelineConfig,
    ) -> Result<Self, PipelineError> {
        let (tx, rx) = bounded::<PostJob>(1);
        let worker = std::thread::Builder::new()
            .name("batch-post-inference".to_owned())
            .spawn_scoped(scope, move || {
                for job in rx {
                    let span = tracing::debug_span!("batch_post_inference", file_id = %job.file_id);
                    let _entered = span.enter();
                    let post_start = Instant::now();
                    let result = post_inference(job.artifacts, config, plda).map(|result| {
                        tracing::trace!(
                            target: "speakrs::timing",
                            file_id = %job.file_id,
                            inference_ms = job.inference_ms,
                            post_ms = post_start.elapsed().as_millis(),
                            total_ms = job.start.elapsed().as_millis(),
                            audio_secs = job.audio_secs,
                            "Pipeline complete",
                        );
                        BatchOutput {
                            file_id: job.file_id,
                            audio_secs: job.audio_secs,
                            result,
                        }
                    });
                    let failed = result.is_err();
                    if job.result.send(result).is_err() || failed {
                        break;
                    }
                }
            })
            .map_err(|source| PipelineError::WorkerSpawn {
                worker: "batch post-inference",
                source,
            })?;

        Ok(Self {
            jobs: Some(tx),
            pending: None,
            worker: Some(worker),
            outputs: Vec::new(),
        })
    }

    pub(super) fn collect_ready(&mut self) -> Result<(), PipelineError> {
        let Some(rx) = &self.pending else {
            return Ok(());
        };

        match rx.try_recv() {
            Ok(result) => {
                self.pending.take();
                self.outputs.push(result?);
                Ok(())
            }
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(()),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.pending.take();
                Err(Self::worker_panic())
            }
        }
    }

    pub(super) fn collect(&mut self) -> Result<(), PipelineError> {
        let Some(rx) = self.pending.take() else {
            return Ok(());
        };
        self.outputs
            .push(rx.recv().map_err(|_| Self::worker_panic())??);
        Ok(())
    }

    pub(super) fn submit(
        &mut self,
        artifacts: InferenceArtifacts,
        file_id: String,
        audio_secs: f64,
        start: Instant,
        inference_ms: u128,
    ) -> Result<(), PipelineError> {
        // collecting before submission bounds host artifacts to one pending file
        self.collect()?;
        let (tx, rx) = bounded(1);
        self.jobs
            .as_ref()
            .ok_or_else(Self::worker_panic)?
            .send(PostJob {
                artifacts,
                file_id,
                audio_secs,
                start,
                inference_ms,
                result: tx,
            })
            .map_err(|_| Self::worker_panic())?;
        self.pending = Some(rx);
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<Vec<BatchOutput>, PipelineError> {
        let result = self.collect();
        self.jobs.take();
        let joined = self.worker.take().map_or(Ok(()), |worker| {
            worker.join().map_err(|_| Self::worker_panic())
        });
        result?;
        joined?;
        Ok(std::mem::take(&mut self.outputs))
    }

    fn worker_panic() -> PipelineError {
        PipelineError::WorkerPanic {
            worker: "batch post-inference".to_owned(),
        }
    }
}

impl Drop for BatchPostProcessor<'_> {
    fn drop(&mut self) {
        // close before joining, including unwinding and early input failures
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
