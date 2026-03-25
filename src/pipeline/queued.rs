use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

use super::{
    BatchInput, DiarizationResult, OwnedDiarizationPipeline, PipelineConfig, PipelineError,
};

// compile-time Send assertion
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert() {
        _assert_send::<OwnedDiarizationPipeline>();
    }
};

/// Monotonically increasing job identifier assigned by the queue on push
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueuedDiarizationJobId(u64);

/// A diarization request that owns its audio buffer
pub struct QueuedDiarizationRequest {
    file_id: String,
    audio: Vec<f32>,
}

impl QueuedDiarizationRequest {
    pub fn new(file_id: impl Into<String>, audio: Vec<f32>) -> Self {
        Self {
            file_id: file_id.into(),
            audio,
        }
    }
}

/// Result from a queued diarization job
///
/// Per-job failures are surfaced here without stopping the worker
pub struct QueuedDiarizationResult {
    pub job_id: QueuedDiarizationJobId,
    pub file_id: String,
    pub result: Result<DiarizationResult, PipelineError>,
}

#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("queue worker has shut down")]
    WorkerGone,
    #[error("worker thread panicked: {0}")]
    WorkerPanicked(String),
}

struct WorkerRequest {
    job_id: QueuedDiarizationJobId,
    file_id: String,
    audio: Vec<f32>,
}

/// Background-processing pipeline that accepts files incrementally via push/push_batch
///
/// The worker thread drains queued requests into batches and processes them via
/// `run_batch_with_config`, preserving cross-file batch optimizations (chunk embedding,
/// priority-pull) within each worker pass.
///
/// ```no_run
/// # use speakrs::pipeline::*;
/// # use speakrs::inference::ExecutionMode;
/// let pipeline = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cpu)?;
/// let queue = pipeline.into_queued()?;
///
/// queue.push(QueuedDiarizationRequest::new("file1", audio_samples))?;
/// let result = queue.recv()?;
/// println!("{}", result.result.unwrap().rttm);
///
/// queue.finish()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct QueuedDiarizationPipeline {
    request_tx: Option<Sender<WorkerRequest>>,
    result_rx: Receiver<QueuedDiarizationResult>,
    worker: Option<JoinHandle<()>>,
    next_job_id: AtomicU64,
}

impl QueuedDiarizationPipeline {
    pub(super) fn new(
        pipeline: OwnedDiarizationPipeline,
        config: PipelineConfig,
    ) -> Result<Self, QueueError> {
        let (request_tx, request_rx) = crossbeam_channel::bounded::<WorkerRequest>(64);
        let (result_tx, result_rx) = crossbeam_channel::bounded::<QueuedDiarizationResult>(64);

        let worker = std::thread::Builder::new()
            .name("speakrs-queue-worker".into())
            .spawn(move || worker_loop(pipeline, config, request_rx, result_tx))
            .map_err(|e| QueueError::WorkerPanicked(e.to_string()))?;

        Ok(Self {
            request_tx: Some(request_tx),
            result_rx,
            worker: Some(worker),
            next_job_id: AtomicU64::new(0),
        })
    }

    /// Submit a single file for background diarization
    pub fn push(
        &self,
        request: QueuedDiarizationRequest,
    ) -> Result<QueuedDiarizationJobId, QueueError> {
        let job_id = QueuedDiarizationJobId(self.next_job_id.fetch_add(1, Ordering::Relaxed));
        let tx = self.request_tx.as_ref().ok_or(QueueError::WorkerGone)?;

        tx.send(WorkerRequest {
            job_id,
            file_id: request.file_id,
            audio: request.audio,
        })
        .map_err(|_| QueueError::WorkerGone)?;

        Ok(job_id)
    }

    /// Submit multiple files at once, guaranteeing they land in the same worker batch
    pub fn push_batch(
        &self,
        requests: Vec<QueuedDiarizationRequest>,
    ) -> Result<Vec<QueuedDiarizationJobId>, QueueError> {
        let tx = self.request_tx.as_ref().ok_or(QueueError::WorkerGone)?;
        let mut job_ids = Vec::with_capacity(requests.len());

        for request in requests {
            let job_id = QueuedDiarizationJobId(self.next_job_id.fetch_add(1, Ordering::Relaxed));
            tx.send(WorkerRequest {
                job_id,
                file_id: request.file_id,
                audio: request.audio,
            })
            .map_err(|_| QueueError::WorkerGone)?;
            job_ids.push(job_id);
        }

        Ok(job_ids)
    }

    /// Block until the next result is available
    pub fn recv(&self) -> Result<QueuedDiarizationResult, QueueError> {
        self.result_rx.recv().map_err(|_| QueueError::WorkerGone)
    }

    /// Return a result if one is ready, or None if the worker is still processing
    pub fn try_recv(&self) -> Result<Option<QueuedDiarizationResult>, QueueError> {
        match self.result_rx.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => Err(QueueError::WorkerGone),
        }
    }

    /// Close the queue and wait for the worker to finish processing all submitted jobs
    ///
    /// Callers should drain results via `recv` before calling this
    pub fn finish(mut self) -> Result<(), QueueError> {
        // close request channel so worker exits after draining
        self.request_tx.take();

        if let Some(handle) = self.worker.take() {
            handle
                .join()
                .map_err(|e| QueueError::WorkerPanicked(format!("{e:?}")))?;
        }

        Ok(())
    }
}

fn worker_loop(
    mut pipeline: OwnedDiarizationPipeline,
    config: PipelineConfig,
    request_rx: Receiver<WorkerRequest>,
    result_tx: Sender<QueuedDiarizationResult>,
) {
    while let Ok(first) = request_rx.recv() {
        // drain all currently queued requests into a batch
        let mut batch = vec![first];
        while let Ok(req) = request_rx.try_recv() {
            batch.push(req);
        }

        let results = process_batch(&mut pipeline, &batch, &config);
        for result in results {
            if result_tx.send(result).is_err() {
                return;
            }
        }
    }
}

fn process_batch(
    pipeline: &mut OwnedDiarizationPipeline,
    batch: &[WorkerRequest],
    config: &PipelineConfig,
) -> Vec<QueuedDiarizationResult> {
    let inputs: Vec<BatchInput<'_>> = batch
        .iter()
        .map(|r| BatchInput {
            audio: &r.audio,
            file_id: &r.file_id,
        })
        .collect();

    match pipeline.run_batch_with_config(&inputs, config) {
        Ok(results) => batch
            .iter()
            .zip(results)
            .map(|(req, result)| QueuedDiarizationResult {
                job_id: req.job_id,
                file_id: req.file_id.clone(),
                result: Ok(result),
            })
            .collect(),
        Err(_) => {
            // batch failed — run individually to isolate which file(s) broke
            batch
                .iter()
                .map(|req| QueuedDiarizationResult {
                    job_id: req.job_id,
                    file_id: req.file_id.clone(),
                    result: pipeline.run_with_config(&req.audio, &req.file_id, config),
                })
                .collect()
        }
    }
}
