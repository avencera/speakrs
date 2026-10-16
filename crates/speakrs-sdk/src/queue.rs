//! A pipeline on a background worker, fed by a bounded request queue

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, MutexGuard, PoisonError, TryLockError};
use std::time::Duration;

use speakrs::ExecutionMode;
use speakrs::pipeline::{QueueError, QueueReceiver, QueueSender, QueuedDiarizationRequest};

use crate::audio::{SDK_SAMPLE_RATE_HZ, decode_file_to_mono_16khz};
use crate::error::{SdkError, SdkErrorCategory};
use crate::pipeline::{Diarization, to_diarization};

/// Error code for a push rejected because the queue is at capacity
pub const QUEUE_FULL: &str = "queue_full";
/// Error code for a push after [`SdkQueue::close`], or a receive after the last result
pub const QUEUE_CLOSED: &str = "queue_closed";

/// One finished queue job
#[derive(Debug, Clone, PartialEq)]
pub struct QueueItem {
    /// Id returned by the push that submitted the job
    pub job_id: u64,
    /// File id given at push time
    pub file_id: String,
    /// The job's result; a failed job does not stop the queue
    pub result: Result<Diarization, SdkError>,
}

/// A pipeline running on a background worker, fed by a bounded request queue
///
/// Every method takes `&self`, so pushes and receives can run on different threads
pub struct SdkQueue {
    sender: Mutex<Option<QueueSender>>,
    receiver: Mutex<QueueReceiver>,
    // results carry no audio, so each job's duration is recorded at push time
    durations: Mutex<HashMap<u64, f64>>,
    mode: ExecutionMode,
    model_revision: Option<String>,
}

impl SdkQueue {
    pub(crate) fn new(
        sender: QueueSender,
        receiver: QueueReceiver,
        mode: ExecutionMode,
        model_revision: Option<String>,
    ) -> Self {
        Self {
            sender: Mutex::new(Some(sender)),
            receiver: Mutex::new(receiver),
            durations: Mutex::new(HashMap::new()),
            mode,
            model_revision,
        }
    }

    /// Submit 16 kHz mono samples without blocking
    ///
    /// Fails with code [`QUEUE_FULL`] when the queue is at capacity
    pub fn push_samples(&self, file_id: String, samples: Vec<f32>) -> Result<u64, SdkError> {
        let duration = samples.len() as f64 / f64::from(SDK_SAMPLE_RATE_HZ);
        // holding the map across the push keeps a fast worker's result from arriving
        // before its duration is recorded
        let mut durations = lock(&self.durations);
        let sender = lock(&self.sender);
        let sender = sender
            .as_ref()
            .ok_or_else(|| closed_error("the queue was closed"))?;
        let job_id = sender
            .try_push(QueuedDiarizationRequest::new(file_id, samples))
            .map_err(Self::map_error)?
            .as_u64();
        durations.insert(job_id, duration);
        Ok(job_id)
    }

    /// Decode an audio file on the calling thread, then submit it without blocking
    pub fn push_file(&self, file_id: String, path: &Path) -> Result<u64, SdkError> {
        let audio = decode_file_to_mono_16khz(path).map_err(SdkError::from_audio_error)?;
        self.push_samples(file_id, audio.samples)
    }

    /// Stop accepting pushes; receives drain the remaining results, then fail with
    /// code [`QUEUE_CLOSED`]
    pub fn close(&self) {
        lock(&self.sender).take();
    }

    /// Wait for the next finished job
    pub fn recv(&self) -> Result<QueueItem, SdkError> {
        let result = lock(&self.receiver).recv().map_err(Self::map_error)?;
        Ok(self.item(result))
    }

    /// Wait up to `timeout` for the next finished job
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Option<QueueItem>, SdkError> {
        let result = lock(&self.receiver)
            .recv_timeout(timeout)
            .map_err(Self::map_error)?;
        Ok(result.map(|result| self.item(result)))
    }

    /// Return a finished job if one is ready, without blocking
    ///
    /// Returns `None` while another receive holds the receiver, since that receive
    /// takes the next result anyway
    pub fn try_recv(&self) -> Result<Option<QueueItem>, SdkError> {
        // waiting for the lock would block an async caller's event loop for up to a
        // concurrent `recv_timeout` slice
        let mut receiver = match self.receiver.try_lock() {
            Ok(receiver) => receiver,
            Err(TryLockError::WouldBlock) => return Ok(None),
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        };
        let result = receiver.try_recv().map_err(Self::map_error)?;
        drop(receiver);
        Ok(result.map(|result| self.item(result)))
    }

    pub(crate) fn map_error(error: QueueError) -> SdkError {
        match error {
            QueueError::Full(_) => SdkError::with_code(
                SdkErrorCategory::Queue,
                QUEUE_FULL,
                "the queue is full; receive results before pushing more",
            ),
            QueueError::Closed | QueueError::WorkerGone => {
                closed_error("the queue has no more results")
            }
            QueueError::InvalidCapacity => SdkError::configuration(error.to_string()),
            other => SdkError::new(SdkErrorCategory::Queue, other.to_string()),
        }
    }

    fn item(&self, result: speakrs::pipeline::QueuedDiarizationResult) -> QueueItem {
        let job_id = result.job_id.as_u64();
        let duration = lock(&self.durations).remove(&job_id).unwrap_or_default();
        let file_id = result.file_id;
        let result = result
            .result
            .map(|result| {
                to_diarization(
                    result,
                    file_id.clone(),
                    duration,
                    self.mode,
                    self.model_revision.clone(),
                    None,
                )
            })
            .map_err(SdkError::from_pipeline_error);

        QueueItem {
            job_id,
            file_id,
            result,
        }
    }
}

fn closed_error(message: &str) -> SdkError {
    SdkError::with_code(SdkErrorCategory::Queue, QUEUE_CLOSED, message)
}

// a panic while holding one of these locks leaves plain data behind, so it stays usable
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}
