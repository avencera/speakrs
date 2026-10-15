use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender, bounded};
use speakrs::pipeline::OwnedBatchInput;

use crate::audio::decode_file_to_mono_16khz;
use crate::cancel::CancelToken;
use crate::error::{SdkError, SdkErrorCategory};
use crate::pipeline::{ProgressCallback, ProgressEvent};

/// Decoded files kept ahead of the one in inference; each holds a whole file in memory
const LOOKAHEAD: usize = 3;
const MAX_WORKERS: usize = 4;

type DecodeResult = Result<Vec<f32>, SdkError>;

struct DecodeJob {
    path: PathBuf,
    result: Sender<DecodeResult>,
}

enum PendingDecode {
    Worker {
        file_id: String,
        result: Receiver<DecodeResult>,
    },
    // pipes and devices are read in turn so a writer is never connected early
    InOrder {
        file_id: String,
        path: PathBuf,
    },
}

/// Ordered stream of decoded batch inputs with a small decode pool ahead of it
///
/// The pipeline consumes this iterator on its own thread, so cancellation and
/// progress are checked there between files
pub(crate) struct DecodePool<'a> {
    files: std::vec::IntoIter<(String, PathBuf)>,
    jobs: Option<Sender<DecodeJob>>,
    pending: VecDeque<PendingDecode>,
    workers: Vec<JoinHandle<()>>,
    stop: Arc<AtomicBool>,
    cancel_token: Option<CancelToken>,
    progress: Option<&'a ProgressCallback<'a>>,
    next_index: usize,
}

impl<'a> DecodePool<'a> {
    pub(crate) fn new(
        files: Vec<(String, PathBuf)>,
        cancel_token: Option<CancelToken>,
        progress: Option<&'a ProgressCallback<'a>>,
    ) -> Result<Self, SdkError> {
        let worker_count = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(MAX_WORKERS)
            .min(files.len());
        let (jobs, job_rx) = bounded::<DecodeJob>(LOOKAHEAD);
        let mut pool = Self {
            files: files.into_iter(),
            jobs: Some(jobs),
            pending: VecDeque::with_capacity(LOOKAHEAD + 1),
            workers: Vec::with_capacity(worker_count),
            stop: Arc::new(AtomicBool::new(false)),
            cancel_token,
            progress,
            next_index: 0,
        };

        for index in 0..worker_count {
            let job_rx = job_rx.clone();
            let stop = Arc::clone(&pool.stop);
            let worker = std::thread::Builder::new()
                .name(format!("speakrs-decode-{index}"))
                .spawn(move || {
                    for job in job_rx {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        // each job has its own result slot, so a dropped pool never blocks a worker
                        let _ = job.result.send(decode(&job.path));
                    }
                })
                .map_err(|err| {
                    SdkError::new(
                        SdkErrorCategory::Internal,
                        format!("could not start an audio decode thread: {err}"),
                    )
                })?;
            pool.workers.push(worker);
        }

        for _ in 0..LOOKAHEAD {
            pool.schedule();
        }

        Ok(pool)
    }

    fn schedule(&mut self) {
        let Some((file_id, path)) = self.files.next() else {
            return;
        };

        let is_regular_file = std::fs::metadata(&path).is_ok_and(|metadata| metadata.is_file());
        if is_regular_file && let Some(jobs) = &self.jobs {
            let (result, result_rx) = bounded(1);
            if jobs
                .send(DecodeJob {
                    path: path.clone(),
                    result,
                })
                .is_ok()
            {
                self.pending.push_back(PendingDecode::Worker {
                    file_id,
                    result: result_rx,
                });
                return;
            }
        }

        self.pending
            .push_back(PendingDecode::InOrder { file_id, path });
    }
}

impl Iterator for DecodePool<'_> {
    type Item = Result<OwnedBatchInput, SdkError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Err(err) = CancelToken::check(self.cancel_token.as_ref()) {
            return Some(Err(err));
        }

        let pending = self.pending.pop_front()?;
        self.schedule();

        let (file_id, audio) = match pending {
            PendingDecode::Worker { file_id, result } => {
                let audio = result.recv().unwrap_or_else(|_| {
                    Err(SdkError::new(
                        SdkErrorCategory::Internal,
                        "an audio decode thread stopped unexpectedly",
                    ))
                });
                (file_id, audio)
            }
            PendingDecode::InOrder { file_id, path } => (file_id, decode(&path)),
        };

        if let Some(progress) = self.progress {
            progress(ProgressEvent::BatchFile {
                index: self.next_index,
                file_id: file_id.clone(),
            });
        }
        self.next_index += 1;

        // the callback may have cancelled, so this file must not reach inference
        if let Err(err) = CancelToken::check(self.cancel_token.as_ref()) {
            return Some(Err(err));
        }

        Some(audio.map(|audio| OwnedBatchInput { audio, file_id }))
    }
}

impl Drop for DecodePool<'_> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.jobs.take();
        self.pending.clear();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn decode(path: &std::path::Path) -> DecodeResult {
    decode_file_to_mono_16khz(path)
        .map(|audio| audio.samples)
        .map_err(SdkError::from_audio_error)
}
