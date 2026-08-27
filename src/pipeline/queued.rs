use std::any::Any;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, TrySendError};

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

/// Monotonically increasing job identifier assigned by the in-process queue
///
/// IDs are local to one sender lineage and are not durable across process restarts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueuedDiarizationJobId(u64);

/// Construction options for a background diarization queue
///
/// The local queue is in-process only. Capacity bounds the request channel and
/// is not durable across process restarts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueConfig {
    /// Maximum number of in-flight requests the worker channel will hold
    pub capacity: usize,
}

impl QueueConfig {
    /// Default request-channel capacity used by [`Self::default`]
    pub const DEFAULT_CAPACITY: usize = 64;

    /// Create a queue config with the given request-channel capacity
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::InvalidCapacity`] when `capacity` is 0
    pub fn new(capacity: usize) -> Result<Self, QueueError> {
        let config = Self { capacity };
        config.validate()?;
        Ok(config)
    }

    /// Return [`QueueError::InvalidCapacity`] when capacity is 0
    pub(crate) fn validate(self) -> Result<(), QueueError> {
        if self.capacity == 0 {
            Err(QueueError::InvalidCapacity)
        } else {
            Ok(())
        }
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            capacity: Self::DEFAULT_CAPACITY,
        }
    }
}

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

    /// File identifier carried with this request
    pub fn file_id(&self) -> &str {
        &self.file_id
    }
}

impl fmt::Debug for QueuedDiarizationRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueuedDiarizationRequest")
            .field("file_id", &self.file_id)
            .field("samples", &self.audio.len())
            .finish()
    }
}

/// Result from a queued diarization job
///
/// Per-job failures are surfaced here without stopping the worker
pub struct QueuedDiarizationResult {
    /// The job identifier returned by [`QueueSender::try_push`]
    pub job_id: QueuedDiarizationJobId,
    /// The file identifier from the original request
    pub file_id: String,
    /// Diarization result, or an error if this file failed
    pub result: Result<DiarizationResult, PipelineError>,
}

/// Errors from the queued diarization pipeline
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum QueueError {
    /// The queue has finished processing all submitted jobs
    #[error("queue has finished processing all submitted jobs")]
    Closed,
    /// The background worker has shut down or was never started
    #[error("queue worker has shut down")]
    WorkerGone,
    /// The request channel is at capacity
    ///
    /// Contains the rejected request so the caller can retry
    #[error("queue is full")]
    Full(QueuedDiarizationRequest),
    /// Queue capacity must be greater than zero
    #[error("queue capacity must be greater than zero")]
    InvalidCapacity,
    /// The background worker thread could not be started
    #[error("failed to start queue worker: {0}")]
    WorkerStart(#[source] std::io::Error),
    /// The background worker thread panicked
    #[error("worker thread panicked: {0}")]
    WorkerPanicked(String),
    /// The receiver reached an unexpected terminal queue error
    #[error("queue reached terminal error: {0}")]
    Terminal(String),
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
/// `run_batch_with_config`, preserving cross-file batch optimizations within
/// each worker pass
///
/// Admission should use [`Self::try_push`], which never blocks. [`Self::push`]
/// is a non-blocking alias kept for existing callers
///
/// ```no_run
/// # use speakrs::pipeline::*;
/// # use speakrs::inference::ExecutionMode;
/// let (tx, rx) = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cpu)?.into_queued()?;
///
/// let audio1: Vec<f32> = vec![]; // 16 kHz mono samples
/// let audio2: Vec<f32> = vec![];
/// tx.try_push(QueuedDiarizationRequest::new("file1", audio1))?;
/// tx.try_push(QueuedDiarizationRequest::new("file2", audio2))?;
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
    capacity: usize,
}

impl QueueSender {
    /// Start a background worker with the given pipeline and queue config
    pub(crate) fn new(
        pipeline: OwnedDiarizationPipeline,
        config: PipelineConfig,
        queue: QueueConfig,
    ) -> Result<(Self, QueueReceiver), QueueError> {
        queue.validate()?;

        let (request_tx, request_rx) = crossbeam_channel::bounded::<WorkerRequest>(queue.capacity);
        let (result_tx, result_rx) =
            crossbeam_channel::bounded::<QueuedDiarizationResult>(queue.capacity);

        let worker = std::thread::Builder::new()
            .name("speakrs-queue-worker".into())
            .spawn(move || worker_loop(pipeline, config, request_rx, result_tx))
            .map_err(QueueError::WorkerStart)?;

        Ok((
            Self {
                request_tx,
                next_job_id: Arc::new(AtomicU64::new(0)),
                capacity: queue.capacity,
            },
            QueueReceiver {
                result_rx,
                worker: Some(worker),
                state: QueueReceiverState::Running,
            },
        ))
    }

    /// Maximum number of requests the worker channel will hold
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Submit a single file for background diarization
    ///
    /// This is a non-blocking alias of [`Self::try_push`]. It returns
    /// [`QueueError::Full`] when the request channel is at capacity instead of
    /// waiting for a slot
    pub fn push(
        &self,
        request: QueuedDiarizationRequest,
    ) -> Result<QueuedDiarizationJobId, QueueError> {
        self.try_push(request)
    }

    /// Submit a single file for background diarization without blocking
    ///
    /// Returns [`QueueError::Full`] immediately when the request channel is at
    /// capacity. The rejected request is returned so the caller can retry
    pub fn try_push(
        &self,
        request: QueuedDiarizationRequest,
    ) -> Result<QueuedDiarizationJobId, QueueError> {
        let job_id = QueuedDiarizationJobId(self.next_job_id.fetch_add(1, Ordering::Relaxed));

        match self.request_tx.try_send(WorkerRequest {
            job_id,
            file_id: request.file_id,
            audio: request.audio,
        }) {
            Ok(()) => Ok(job_id),

            Err(TrySendError::Full(rejected)) => Err(QueueError::Full(QueuedDiarizationRequest {
                file_id: rejected.file_id,
                audio: rejected.audio,
            })),

            Err(TrySendError::Disconnected(_)) => Err(QueueError::WorkerGone),
        }
    }
}

#[derive(Debug)]
enum QueueReceiverState {
    Running,
    Closed,
    WorkerPanicked(String),
    Terminal(String),
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
            Err(err) => {
                let message = err.to_string();
                self.state = QueueReceiverState::Terminal(message);
                err
            }
        }
    }

    fn terminal_error(&self) -> QueueError {
        match &self.state {
            QueueReceiverState::Running => unreachable!("running receiver has no terminal error"),
            QueueReceiverState::Closed => QueueError::Closed,
            QueueReceiverState::WorkerPanicked(message) => {
                QueueError::WorkerPanicked(message.clone())
            }
            QueueReceiverState::Terminal(message) => QueueError::Terminal(message.clone()),
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
        let batch = drain_batch(first, &request_rx);

        let results = process_batch(&mut pipeline, &batch, &config);
        if !send_results(&result_tx, results) {
            return;
        }
    }
}

fn drain_batch(first: WorkerRequest, request_rx: &Receiver<WorkerRequest>) -> Vec<WorkerRequest> {
    // drain all currently queued requests into one batch
    let mut batch = vec![first];
    while let Ok(req) = request_rx.try_recv() {
        batch.push(req);
    }
    batch
}

fn send_results(
    result_tx: &Sender<QueuedDiarizationResult>,
    results: Vec<QueuedDiarizationResult>,
) -> bool {
    results
        .into_iter()
        .all(|result| result_tx.send(result).is_ok())
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

    let Ok(results) = pipeline.run_batch_with_config(&inputs, config) else {
        return process_individually(pipeline, batch, config);
    };

    batch
        .iter()
        .zip(results)
        .map(|(req, result)| queued_result(req, Ok(result)))
        .collect()
}

fn process_individually(
    pipeline: &mut OwnedDiarizationPipeline,
    batch: &[WorkerRequest],
    config: &PipelineConfig,
) -> Vec<QueuedDiarizationResult> {
    // the batch failed, so retry each file individually to isolate failures
    batch
        .iter()
        .map(|req| {
            queued_result(
                req,
                pipeline.run_with_config(&req.audio, &req.file_id, config),
            )
        })
        .collect()
}

fn queued_result(
    request: &WorkerRequest,
    result: Result<DiarizationResult, PipelineError>,
) -> QueuedDiarizationResult {
    QueuedDiarizationResult {
        job_id: request.job_id,
        file_id: request.file_id.clone(),
        result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl QueueSender {
        fn from_request_tx(request_tx: Sender<WorkerRequest>) -> Self {
            let capacity = request_tx
                .capacity()
                .expect("test senders use a bounded channel");

            Self {
                request_tx,
                next_job_id: Arc::new(AtomicU64::new(0)),
                capacity,
            }
        }
    }

    #[test]
    fn queue_config_rejects_zero_capacity() {
        assert!(matches!(
            QueueConfig::new(0),
            Err(QueueError::InvalidCapacity)
        ));
        assert!(matches!(
            QueueConfig { capacity: 0 }.validate(),
            Err(QueueError::InvalidCapacity)
        ));
    }

    #[test]
    fn queue_config_default_capacity_is_64() {
        assert_eq!(
            QueueConfig::default().capacity,
            QueueConfig::DEFAULT_CAPACITY
        );
        assert_eq!(QueueConfig::DEFAULT_CAPACITY, 64);
        assert_eq!(QueueConfig::new(2).unwrap().capacity, 2);
    }

    #[test]
    fn sender_reports_configured_capacity() {
        let (request_tx, _request_rx) = crossbeam_channel::bounded::<WorkerRequest>(2);
        let sender = QueueSender::from_request_tx(request_tx);
        assert_eq!(sender.capacity(), 2);
    }

    #[test]
    fn try_push_returns_full_when_channel_has_no_consumer() {
        let (request_tx, _request_rx) = crossbeam_channel::bounded::<WorkerRequest>(1);
        let sender = QueueSender::from_request_tx(request_tx);

        sender
            .try_push(QueuedDiarizationRequest::new("first", Vec::new()))
            .unwrap();

        let err = sender
            .try_push(QueuedDiarizationRequest::new("second", vec![1.0]))
            .unwrap_err();

        match err {
            QueueError::Full(request) => assert_eq!(request.file_id(), "second"),
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn try_push_succeeds_after_a_slot_is_freed() {
        let (request_tx, request_rx) = crossbeam_channel::bounded::<WorkerRequest>(1);
        let sender = QueueSender::from_request_tx(request_tx);

        sender
            .try_push(QueuedDiarizationRequest::new("first", Vec::new()))
            .unwrap();

        let rejected = match sender.try_push(QueuedDiarizationRequest::new("second", vec![1.0])) {
            Err(QueueError::Full(request)) => request,
            other => panic!("expected Full, got {other:?}"),
        };

        assert_eq!(rejected.file_id(), "second");
        let _drained = request_rx.recv().unwrap();
        sender.try_push(rejected).unwrap();
    }

    #[test]
    fn push_is_non_blocking_and_returns_full() {
        let (request_tx, _request_rx) = crossbeam_channel::bounded::<WorkerRequest>(1);
        let sender = QueueSender::from_request_tx(request_tx);

        sender
            .push(QueuedDiarizationRequest::new("first", Vec::new()))
            .unwrap();

        assert!(matches!(
            sender.push(QueuedDiarizationRequest::new("second", Vec::new())),
            Err(QueueError::Full(_))
        ));
    }

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

        let sender = QueueSender::from_request_tx(request_tx);

        assert!(matches!(
            sender.push(QueuedDiarizationRequest::new("file", Vec::new())),
            Err(QueueError::WorkerGone)
        ));
    }

    #[test]
    fn receiver_repeats_unexpected_terminal_error_without_panicking() {
        let (_result_tx, result_rx) = crossbeam_channel::bounded(1);
        let mut receiver = QueueReceiver {
            result_rx,
            worker: None,
            state: QueueReceiverState::Terminal("future terminal error".to_owned()),
        };

        assert!(
            matches!(receiver.recv(), Err(QueueError::Terminal(message)) if message == "future terminal error")
        );
        assert!(
            matches!(receiver.try_recv(), Err(QueueError::Terminal(message)) if message == "future terminal error")
        );
    }
}
