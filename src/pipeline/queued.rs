use std::any::Any;
use std::sync::Arc;
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
    /// Create a request with a file identifier and 16 kHz mono f32 audio samples
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
    /// The job identifier returned by [`QueueSender::push`]
    pub job_id: QueuedDiarizationJobId,
    /// The file identifier from the original request
    pub file_id: String,
    /// Diarization result, or an error if this file failed
    pub result: Result<DiarizationResult, PipelineError>,
}

/// Errors from the queued diarization pipeline
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    /// The queue has finished processing all submitted jobs
    #[error("queue has finished processing all submitted jobs")]
    Closed,
    /// The background worker has shut down or was never started
    #[error("queue worker has shut down")]
    WorkerGone,
    /// The background worker thread could not be started
    #[error("failed to start queue worker: {0}")]
    WorkerStart(#[source] std::io::Error),
    /// The background worker thread panicked
    #[error("worker thread panicked: {0}")]
    WorkerPanicked(String),
}

impl QueueError {
    fn format_worker_panic(err: Box<dyn Any + Send + 'static>) -> Self {
        Self::WorkerPanicked(panic_payload_message(err))
    }
}

struct WorkerRequest {
    job_id: QueuedDiarizationJobId,
    file_id: String,
    audio: Vec<f32>,
}

/// Background queue sender for incremental diarization requests
///
/// The worker thread drains queued requests into batches and processes them via
/// `run_batch_with_config`, preserving cross-file batch optimizations (chunk embedding,
/// Priority-pull) within each worker pass
///
/// ```no_run
/// # use speakrs::pipeline::*;
/// # use speakrs::inference::ExecutionMode;
/// let (tx, rx) = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cpu)?.into_queued()?;
///
/// let audio1: Vec<f32> = vec![]; // 16 kHz mono samples
/// let audio2: Vec<f32> = vec![];
/// tx.push(QueuedDiarizationRequest::new("file1", audio1))?;
/// tx.push(QueuedDiarizationRequest::new("file2", audio2))?;
/// drop(tx);
///
/// for result in rx {
///     let result = result?;
///     let diarization = result.result?;
///     println!("{}", diarization.rttm(&result.file_id));
/// }
/// # Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
/// ```
#[derive(Clone)]
pub struct QueueSender {
    request_tx: Sender<WorkerRequest>,
    next_job_id: Arc<AtomicU64>,
}

impl QueueSender {
    pub(super) fn new(
        pipeline: OwnedDiarizationPipeline,
        config: PipelineConfig,
    ) -> Result<(Self, QueueReceiver), QueueError> {
        let (request_tx, request_rx) = crossbeam_channel::bounded::<WorkerRequest>(64);
        let (result_tx, result_rx) = crossbeam_channel::bounded::<QueuedDiarizationResult>(64);

        let worker = std::thread::Builder::new()
            .name("speakrs-queue-worker".into())
            .spawn(move || worker_loop(pipeline, config, request_rx, result_tx))
            .map_err(QueueError::WorkerStart)?;

        Ok((
            Self {
                request_tx,
                next_job_id: Arc::new(AtomicU64::new(0)),
            },
            QueueReceiver {
                result_rx,
                worker: Some(worker),
                state: QueueReceiverState::Running,
            },
        ))
    }

    /// Submit a single file for background diarization
    pub fn push(
        &self,
        request: QueuedDiarizationRequest,
    ) -> Result<QueuedDiarizationJobId, QueueError> {
        let job_id = QueuedDiarizationJobId(self.next_job_id.fetch_add(1, Ordering::Relaxed));

        self.request_tx
            .send(WorkerRequest {
                job_id,
                file_id: request.file_id,
                audio: request.audio,
            })
            .map_err(|_| QueueError::WorkerGone)?;

        Ok(job_id)
    }
}

#[derive(Debug)]
enum QueueReceiverState {
    Running,
    Closed,
    WorkerPanicked(String),
}

/// Background queue receiver for diarization results
///
/// `recv` and `try_recv` require mutable access so the receiver can join the worker once
/// and transition into a terminal state without interior mutability
pub struct QueueReceiver {
    result_rx: Receiver<QueuedDiarizationResult>,
    worker: Option<JoinHandle<()>>,
    state: QueueReceiverState,
}

impl QueueReceiver {
    /// Block until the next result is available
    ///
    /// Returns [`QueueError::Closed`] after the worker has finished and all queued results
    /// have been drained
    pub fn recv(&mut self) -> Result<QueuedDiarizationResult, QueueError> {
        if !matches!(self.state, QueueReceiverState::Running) {
            return Err(self.terminal_error());
        }

        match self.result_rx.recv() {
            Ok(result) => Ok(result),
            Err(_) => Err(self.join_terminal_worker()),
        }
    }

    /// Return a result if one is ready, or `None` if the worker is still processing
    ///
    /// Returns [`QueueError::Closed`] after the worker has finished and all queued results
    /// have been drained
    pub fn try_recv(&mut self) -> Result<Option<QueuedDiarizationResult>, QueueError> {
        if !matches!(self.state, QueueReceiverState::Running) {
            return Err(self.terminal_error());
        }

        match self.result_rx.try_recv() {
            Ok(result) => Ok(Some(result)),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => Err(self.join_terminal_worker()),
        }
    }

    fn join_terminal_worker(&mut self) -> QueueError {
        match join_worker(self.worker.take()) {
            Ok(()) => {
                self.state = QueueReceiverState::Closed;
                QueueError::Closed
            }
            Err(QueueError::WorkerPanicked(message)) => {
                self.state = QueueReceiverState::WorkerPanicked(message.clone());
                QueueError::WorkerPanicked(message)
            }
            Err(err) => unreachable!("unexpected terminal queue error: {err}"),
        }
    }

    fn terminal_error(&self) -> QueueError {
        match &self.state {
            QueueReceiverState::Running => unreachable!("running receiver has no terminal error"),
            QueueReceiverState::Closed => QueueError::Closed,
            QueueReceiverState::WorkerPanicked(message) => {
                QueueError::WorkerPanicked(message.clone())
            }
        }
    }
}

/// Iterator that drains results from a [`QueueReceiver`]
///
/// Created by calling `.into_iter()` on a [`QueueReceiver`]
/// Yields queued results until the worker has finished processing all queued jobs
/// If the worker panics after sending partial results, the iterator yields one terminal error
pub struct QueueReceiverIter {
    receiver: QueueReceiver,
    yielded_terminal_error: bool,
}

impl Iterator for QueueReceiverIter {
    type Item = Result<QueuedDiarizationResult, QueueError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.yielded_terminal_error {
            return None;
        }

        match self.receiver.recv() {
            Ok(result) => Some(Ok(result)),
            Err(QueueError::Closed) => None,
            Err(err) => {
                self.yielded_terminal_error = true;
                Some(Err(err))
            }
        }
    }
}

impl IntoIterator for QueueReceiver {
    type Item = Result<QueuedDiarizationResult, QueueError>;
    type IntoIter = QueueReceiverIter;

    fn into_iter(self) -> Self::IntoIter {
        QueueReceiverIter {
            receiver: self,
            yielded_terminal_error: false,
        }
    }
}

fn join_worker(worker: Option<JoinHandle<()>>) -> Result<(), QueueError> {
    if let Some(handle) = worker {
        handle.join().map_err(QueueError::format_worker_panic)?;
    }

    Ok(())
}

fn panic_payload_message(err: Box<dyn Any + Send + 'static>) -> String {
    match err.downcast::<String>() {
        Ok(message) => *message,
        Err(err) => match err.downcast::<&'static str>() {
            Ok(message) => (*message).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

fn worker_loop(
    mut pipeline: OwnedDiarizationPipeline,
    config: PipelineConfig,
    request_rx: Receiver<WorkerRequest>,
    result_tx: Sender<QueuedDiarizationResult>,
) {
    while let Ok(first) = request_rx.recv() {
        // drain all currently queued requests into one batch
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
            // the batch failed, so retry each file individually to isolate failures
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_reports_clean_close_after_worker_exit() {
        let (result_tx, result_rx) = crossbeam_channel::bounded(1);
        drop(result_tx);

        let worker = std::thread::spawn(|| {});
        let mut receiver = QueueReceiver {
            result_rx,
            worker: Some(worker),
            state: QueueReceiverState::Running,
        };

        assert!(matches!(receiver.recv(), Err(QueueError::Closed)));
        assert!(matches!(receiver.try_recv(), Err(QueueError::Closed)));
    }

    #[test]
    fn receiver_reports_worker_panic() {
        let (result_tx, result_rx) = crossbeam_channel::bounded(1);
        drop(result_tx);

        let worker = std::thread::spawn(|| panic!("worker exploded"));
        let mut receiver = QueueReceiver {
            result_rx,
            worker: Some(worker),
            state: QueueReceiverState::Running,
        };

        assert!(
            matches!(receiver.recv(), Err(QueueError::WorkerPanicked(message)) if message.contains("worker exploded"))
        );
        assert!(
            matches!(receiver.try_recv(), Err(QueueError::WorkerPanicked(message)) if message.contains("worker exploded"))
        );
    }

    #[test]
    fn iterator_yields_terminal_worker_panic_once() {
        let (result_tx, result_rx) = crossbeam_channel::bounded(1);
        drop(result_tx);

        let worker = std::thread::spawn(|| panic!("iterator panic"));
        let receiver = QueueReceiver {
            result_rx,
            worker: Some(worker),
            state: QueueReceiverState::Running,
        };
        let mut iter = receiver.into_iter();

        assert!(
            matches!(iter.next(), Some(Err(QueueError::WorkerPanicked(message))) if message.contains("iterator panic"))
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn sender_reports_worker_gone_after_request_channel_closes() {
        let (request_tx, request_rx) = crossbeam_channel::bounded::<WorkerRequest>(1);
        drop(request_rx);

        let sender = QueueSender {
            request_tx,
            next_job_id: Arc::new(AtomicU64::new(0)),
        };

        assert!(matches!(
            sender.push(QueuedDiarizationRequest::new("file", Vec::new())),
            Err(QueueError::WorkerGone)
        ));
    }
}
