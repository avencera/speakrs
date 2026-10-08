use std::collections::VecDeque;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use color_eyre::eyre::{Result, ensure, eyre};
use crossbeam_channel::{Receiver, Sender, bounded};
use speakrs::pipeline::OwnedBatchInput;

use super::read_wav_samples;

const LOOKAHEAD: usize = 3;

// only this owner can create a speculative input from an opened regular file
struct RegularInput(File);

impl RegularInput {
    #[cfg(unix)]
    fn open(path: &Path) -> std::io::Result<Option<Self>> {
        use std::fs::OpenOptions;
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;

        // open without waiting for a FIFO writer, then check the actual handle, not its path
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)?;
        if !file.metadata()?.is_file() {
            return Ok(None);
        }

        let descriptor = file.as_raw_fd();
        // safety: the owned file keeps this descriptor valid throughout both fcntl calls
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
        if flags == -1 {
            return Err(std::io::Error::last_os_error());
        }

        // safety: F_SETFL takes an integer flag value and the descriptor remains owned
        if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags & !libc::O_NONBLOCK) } == -1 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Some(Self(file)))
    }

    #[cfg(not(unix))]
    fn open(_path: &Path) -> std::io::Result<Option<Self>> {
        // without a nonblocking probe, retain the serial path for all inputs
        Ok(None)
    }
}

struct DecodeJob {
    path: PathBuf,
    input: RegularInput,
    result: Sender<Result<OwnedBatchInput>>,
}

enum PendingDecode {
    Worker(Receiver<Result<OwnedBatchInput>>),
    InOrder(PathBuf),
}

/// A small decode pool with at most three files ahead of the consumer
pub struct WavBatchDecoder {
    paths: std::vec::IntoIter<PathBuf>,
    jobs: Option<Sender<DecodeJob>>,
    pending: VecDeque<PendingDecode>,
    cancelled: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl WavBatchDecoder {
    /// Start decoding, keeping results and failures in input order
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let worker_count = thread::available_parallelism()
            .map_or(1, usize::from)
            .min(4)
            .min(paths.len());
        let (tx, rx) = bounded::<DecodeJob>(LOOKAHEAD);
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut decoder = Self {
            paths: paths.into_iter(),
            jobs: Some(tx),
            pending: VecDeque::with_capacity(LOOKAHEAD),
            cancelled,
            workers: Vec::with_capacity(worker_count),
        };

        for index in 0..worker_count {
            let rx = rx.clone();
            let cancelled = Arc::clone(&decoder.cancelled);
            decoder.workers.push(
                thread::Builder::new()
                    .name(format!("audio-decode-{index}"))
                    .spawn(move || {
                        for job in rx {
                            if cancelled.load(Ordering::Relaxed) {
                                break;
                            }

                            // each job has one result slot, so dropping a consumer cannot block it
                            let _ =
                                job.result
                                    .send(Self::decode(job.path, job.input.0, &cancelled));
                        }
                    })?,
            );
        }

        for _ in 0..LOOKAHEAD {
            decoder.schedule()?;
        }

        Ok(decoder)
    }

    fn schedule(&mut self) -> Result<()> {
        let Some(path) = self.paths.next() else {
            return Ok(());
        };

        // failed probes and non-regular inputs retain their original in-order open behavior
        let input = match RegularInput::open(&path) {
            Ok(Some(input)) => input,
            Ok(None) | Err(_) => {
                self.pending.push_back(PendingDecode::InOrder(path));
                return Ok(());
            }
        };

        let (tx, rx) = bounded(1);
        self.jobs
            .as_ref()
            .ok_or_else(|| eyre!("audio decode pool closed"))?
            .send(DecodeJob {
                path,
                input,
                result: tx,
            })
            .map_err(|_| eyre!("audio decode worker stopped"))?;
        self.pending.push_back(PendingDecode::Worker(rx));
        Ok(())
    }

    fn decode(path: PathBuf, file: File, cancelled: &AtomicBool) -> Result<OwnedBatchInput> {
        let file_id = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file1".to_owned());
        let span = tracing::debug_span!("audio_decode", %file_id);
        let _entered = span.enter();
        let start = Instant::now();
        let (audio, sample_rate) =
            read_wav_samples(BufReader::new(file), || cancelled.load(Ordering::Relaxed))?;
        ensure!(
            sample_rate == 16_000,
            "expected 16kHz WAV, got {sample_rate}Hz"
        );
        tracing::trace!(load_ms = start.elapsed().as_millis(), %file_id, "Audio decoded");
        Ok(OwnedBatchInput { audio, file_id })
    }
}

impl Iterator for WavBatchDecoder {
    type Item = Result<OwnedBatchInput>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.pending.pop_front()? {
            PendingDecode::Worker(rx) => rx
                .recv()
                .unwrap_or_else(|_| Err(eyre!("audio decode worker panicked"))),
            PendingDecode::InOrder(path) => File::open(&path)
                .map_err(Into::into)
                .and_then(|file| Self::decode(path, file, &self.cancelled)),
        };
        if result.is_err() {
            self.cancelled.store(true, Ordering::Relaxed);
            self.pending.clear();
            return Some(result);
        }

        if let Err(error) = self.schedule() {
            return Some(Err(error));
        }

        Some(result)
    }
}

impl Drop for WavBatchDecoder {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        self.jobs.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
