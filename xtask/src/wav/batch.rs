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

#[cfg(unix)]
#[derive(Debug)]
struct InputChangedDuringLookahead(PathBuf);

#[cfg(unix)]
impl std::fmt::Display for InputChangedDuringLookahead {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "input changed from a regular file to a non-regular file during lookahead: {}",
            self.0.display()
        )
    }
}

#[cfg(unix)]
impl std::error::Error for InputChangedDuringLookahead {}

enum ProbedInput {
    Regular(RegularInput),
    #[cfg(unix)]
    Changed(InputChangedDuringLookahead),
}

impl ProbedInput {
    #[cfg(unix)]
    fn open(path: &Path) -> std::io::Result<Option<Self>> {
        // opening a known FIFO would connect its writer before the input's turn
        if !std::fs::metadata(path)?.is_file() {
            return Ok(None);
        }

        Self::open_candidate(path).map(Some)
    }

    #[cfg(unix)]
    fn open_candidate(path: &Path) -> std::io::Result<Self> {
        use std::fs::OpenOptions;
        use std::os::fd::AsRawFd;
        use std::os::unix::fs::OpenOptionsExt;

        // a path can change after stat, so this open must not wait for a writer
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)?;

        // only the opened handle can prove that a speculative read is safe for a worker
        if !file.metadata()?.is_file() {
            return Ok(Self::Changed(InputChangedDuringLookahead(path.to_owned())));
        }

        let descriptor = file.as_raw_fd();
        // safety: the owned file keeps this descriptor valid throughout both fcntl calls
        let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
        if flags == -1 {
            return Err(std::io::Error::last_os_error());
        }

        // safety: F_SETFL takes integer flags and the descriptor remains owned
        if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags & !libc::O_NONBLOCK) } == -1 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self::Regular(RegularInput(file)))
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
    #[cfg(unix)]
    Changed(InputChangedDuringLookahead),
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

        let probe = ProbedInput::open(&path);
        self.schedule_probed(path, probe)
    }

    fn schedule_probed(
        &mut self,
        path: PathBuf,
        probe: std::io::Result<Option<ProbedInput>>,
    ) -> Result<()> {
        // known non-regular inputs and failed probes keep their original in-order open behavior
        let input = match probe {
            Ok(Some(ProbedInput::Regular(input))) => input,
            #[cfg(unix)]
            Ok(Some(ProbedInput::Changed(error))) => {
                self.pending.push_back(PendingDecode::Changed(error));
                return Ok(());
            }
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
            #[cfg(unix)]
            PendingDecode::Changed(error) => Err(error.into()),
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{LOOKAHEAD, WavBatchDecoder};

    fn write_wav(path: &Path, sample: i16, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        writer.write_sample(sample).unwrap();
        writer.finalize().unwrap();
    }

    #[test]
    fn parallel_decode_is_bounded_and_in_input_order() {
        let directory = tempfile::tempdir().unwrap();
        let paths: Vec<_> = (0..8)
            .map(|index| {
                let path = directory.path().join(format!("file-{index}.wav"));
                write_wav(&path, index * 1000, 16_000);
                path
            })
            .collect();
        let mut decoder = WavBatchDecoder::new(paths).unwrap();
        assert_eq!(decoder.pending.len(), LOOKAHEAD);

        for index in 0..8 {
            let file = decoder.next().unwrap().unwrap();
            assert_eq!(file.file_id, format!("file-{index}"));
            assert_eq!(file.audio, vec![(index * 1000) as f32 / 32768.0]);
            assert!(decoder.pending.len() <= LOOKAHEAD);
        }

        assert!(decoder.next().is_none());
    }

    #[test]
    fn decode_error_is_ordered_and_stops_the_stream() {
        let directory = tempfile::tempdir().unwrap();
        let valid = directory.path().join("valid.wav");
        let wrong_rate = directory.path().join("wrong-rate.wav");
        let missing = directory.path().join("missing.wav");
        write_wav(&valid, 1234, 16_000);
        write_wav(&wrong_rate, 1, 8000);
        let mut decoder = WavBatchDecoder::new(vec![valid, wrong_rate, missing]).unwrap();
        assert_eq!(
            decoder.next().unwrap().unwrap().audio,
            vec![1234.0 / 32768.0]
        );
        let error = decoder.next().unwrap().err().unwrap();
        assert_eq!(error.to_string(), "expected 16kHz WAV, got 8000Hz");
        assert!(decoder.next().is_none());
    }

    #[test]
    fn dropping_a_full_decoder_joins_workers() {
        let directory = tempfile::tempdir().unwrap();
        let paths: Vec<_> = (0..8)
            .map(|index| {
                let path = directory.path().join(format!("file-{index}.wav"));
                write_wav(&path, index, 16_000);
                path
            })
            .collect();
        let mut decoder = WavBatchDecoder::new(paths).unwrap();
        assert_eq!(decoder.next().unwrap().unwrap().audio, vec![0.0]);
        drop(decoder);
    }

    #[cfg(unix)]
    mod unix {
        use std::ffi::CString;
        use std::fs::{File, OpenOptions};
        use std::io::Write;
        use std::os::unix::ffi::OsStrExt;
        use std::path::Path;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        use super::{WavBatchDecoder, write_wav};

        fn create_fifo(path: &Path) {
            let name = CString::new(path.as_os_str().as_bytes()).unwrap();
            // safety: the path is a valid NUL-terminated string held through this call
            assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        }

        fn controlled_fifo(path: &Path) -> File {
            create_fifo(path);
            // keep both ends open so a premature reader blocks until the test supplies a header
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .unwrap()
        }

        #[test]
        fn fifo_writer_without_extra_reader_keeps_samples_and_order() {
            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.wav");
            let source = directory.path().join("source.wav");
            let fifo = directory.path().join("pipe.wav");
            write_wav(&first, 1000, 16_000);
            write_wav(&source, -2000, 16_000);
            let bytes = std::fs::read(source).unwrap();
            create_fifo(&fifo);

            let (started_tx, started_rx) = mpsc::channel();
            let (opened_tx, opened_rx) = mpsc::channel();
            let (write_tx, write_rx) = mpsc::channel();
            let (writer_tx, writer_rx) = mpsc::channel();
            let writer_path = fifo.clone();
            let producer = thread::spawn(move || {
                started_tx.send(()).unwrap();
                let mut writer = OpenOptions::new().write(true).open(&writer_path).unwrap();
                opened_tx.send(()).unwrap();
                write_rx.recv().unwrap();
                let result = writer.write_all(&bytes);
                let needs_release = result.is_err();
                writer_tx.send(result).unwrap();
                drop(writer);
                if needs_release {
                    // release the regressed in-order reopen without adding an extra reader
                    OpenOptions::new()
                        .write(true)
                        .open(writer_path)
                        .unwrap()
                        .write_all(&bytes)
                        .unwrap();
                }
            });
            started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            assert!(opened_rx.recv_timeout(Duration::from_millis(100)).is_err());

            let (first_tx, first_rx) = mpsc::channel();
            let (consume_tx, consume_rx) = mpsc::channel();
            let (finished_tx, finished_rx) = mpsc::channel();
            let consumer = thread::spawn(move || {
                let mut decoder = WavBatchDecoder::new(vec![first, fifo]).unwrap();
                let first = decoder.next().unwrap().unwrap();
                first_tx.send(first).unwrap();
                consume_rx.recv().unwrap();
                let second = decoder.next().unwrap().unwrap();
                assert!(decoder.next().is_none());
                drop(decoder);
                finished_tx.send(second).unwrap();
            });
            let first = first_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            write_tx.send(()).unwrap();
            // a known FIFO must not release its writer during lookahead
            let premature_write = writer_rx.recv_timeout(Duration::from_millis(100));
            consume_tx.send(()).unwrap();
            let writer_result = premature_write
                .unwrap_or_else(|_| writer_rx.recv_timeout(Duration::from_secs(1)).unwrap());
            let second = finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
            producer.join().unwrap();
            consumer.join().unwrap();

            assert!(
                writer_result.is_ok(),
                "FIFO producer failed: {writer_result:?}"
            );
            assert_eq!(first.file_id, "first");
            assert_eq!(first.audio, vec![1000.0 / 32768.0]);
            assert_eq!(second.file_id, "pipe");
            assert_eq!(second.audio, vec![-2000.0 / 32768.0]);
        }

        #[test]
        fn earlier_error_joins_workers_without_waiting_for_later_fifo() {
            let directory = tempfile::tempdir().unwrap();
            let wrong_rate = directory.path().join("wrong-rate.wav");
            let fifo = directory.path().join("blocked.wav");
            let valid = directory.path().join("valid.wav");
            write_wav(&wrong_rate, 1, 8000);
            write_wav(&valid, 1234, 16_000);
            let mut control = controlled_fifo(&fifo);
            let release_bytes = std::fs::read(valid).unwrap();
            let (finished_tx, finished_rx) = mpsc::channel();
            let consumer = thread::spawn(move || {
                let mut decoder = WavBatchDecoder::new(vec![wrong_rate, fifo]).unwrap();
                assert!(!decoder.workers.is_empty());
                let error = decoder.next().unwrap().err().unwrap();
                assert!(decoder.next().is_none());
                // dropping joins every decode worker before the completion message can be sent
                drop(decoder);
                finished_tx.send(error.to_string()).unwrap();
            });
            let result = finished_rx.recv_timeout(Duration::from_secs(1));
            // release a regressed speculative reader before assertions, so no thread is left blocked
            control.write_all(&release_bytes).unwrap();
            drop(control);
            consumer.join().unwrap();
            assert_eq!(
                result.expect("earlier error must return while the FIFO header is held"),
                "expected 16kHz WAV, got 8000Hz"
            );
        }

        #[test]
        fn fifo_is_decoded_at_its_turn_without_changing_samples_or_order() {
            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.wav");
            let source = directory.path().join("source.wav");
            let fifo = directory.path().join("pipe.wav");
            write_wav(&first, 1000, 16_000);
            write_wav(&source, -2000, 16_000);
            let mut control = controlled_fifo(&fifo);
            let mut decoder = WavBatchDecoder::new(vec![first, fifo]).unwrap();
            let first = decoder.next().unwrap().unwrap();
            assert_eq!(first.file_id, "first");
            assert_eq!(first.audio, vec![1000.0 / 32768.0]);
            control.write_all(&std::fs::read(source).unwrap()).unwrap();
            let second = decoder.next().unwrap().unwrap();
            assert_eq!(second.file_id, "pipe");
            assert_eq!(second.audio, vec![-2000.0 / 32768.0]);
            assert!(decoder.next().is_none());
            drop(decoder);
        }

        #[test]
        fn opened_regular_input_survives_path_replacement_with_fifo() {
            use std::os::fd::AsRawFd;
            use std::os::unix::fs::symlink;
            use std::sync::atomic::AtomicBool;

            use super::super::ProbedInput;

            let directory = tempfile::tempdir().unwrap();
            let regular = directory.path().join("regular.wav");
            let fifo = directory.path().join("fifo.wav");
            let input_path = directory.path().join("input.wav");
            write_wav(&regular, 1234, 16_000);
            let _control = controlled_fifo(&fifo);
            symlink(&regular, &input_path).unwrap();
            let Some(ProbedInput::Regular(input)) = ProbedInput::open(&input_path).unwrap() else {
                panic!("expected an opened regular input");
            };

            // safety: the file owns this valid descriptor while fcntl reads its flags
            let flags = unsafe { libc::fcntl(input.0.as_raw_fd(), libc::F_GETFL) };
            assert_ne!(flags, -1);
            assert_eq!(flags & libc::O_NONBLOCK, 0);
            std::fs::remove_file(&input_path).unwrap();
            symlink(&fifo, &input_path).unwrap();
            assert!(ProbedInput::open(&input_path).unwrap().is_none());
            let file =
                WavBatchDecoder::decode(input_path, input.0, &AtomicBool::new(false)).unwrap();
            assert_eq!(file.file_id, "input");
            assert_eq!(file.audio, vec![1234.0 / 32768.0]);
        }

        #[test]
        fn stat_open_race_returns_an_ordered_typed_error_without_waiting() {
            use std::sync::atomic::Ordering;

            use super::super::{InputChangedDuringLookahead, ProbedInput};

            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.wav");
            let path = directory.path().join("raced.wav");
            let later = directory.path().join("later.wav");
            write_wav(&first, 1000, 16_000);
            write_wav(&path, 3000, 16_000);
            write_wav(&later, -2000, 16_000);
            // exercise the open phase after the candidate changes between stat and open
            assert!(std::fs::metadata(&path).unwrap().is_file());
            std::fs::remove_file(&path).unwrap();
            create_fifo(&path);
            let (finished_tx, finished_rx) = mpsc::channel();
            let expected_path = path.clone();
            let consumer = thread::spawn(move || {
                let probe = ProbedInput::open_candidate(&path).map(Some);
                let mut decoder = WavBatchDecoder::new(vec![first]).unwrap();
                decoder.schedule_probed(path, probe).unwrap();
                decoder
                    .schedule_probed(later.clone(), ProbedInput::open(&later))
                    .unwrap();
                let first = decoder.next().unwrap().unwrap();
                let error = decoder.next().unwrap().err().unwrap();
                assert!(decoder.cancelled.load(Ordering::Relaxed));
                assert!(decoder.next().is_none());
                drop(decoder);
                // an input error must not affect the next decoder's output
                let mut next = WavBatchDecoder::new(vec![later]).unwrap();
                let next = next.next().unwrap().unwrap();
                finished_tx.send((first, error, next)).unwrap();
            });
            let (first, error, next) = finished_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("a raced FIFO without a writer must return an error, not wait");
            consumer.join().unwrap();

            assert_eq!(first.file_id, "first");
            assert_eq!(first.audio, vec![1000.0 / 32768.0]);
            let changed = error.downcast_ref::<InputChangedDuringLookahead>().unwrap();
            assert_eq!(changed.0, expected_path);
            assert_eq!(
                error.to_string(),
                format!(
                    "input changed from a regular file to a non-regular file during lookahead: {}",
                    expected_path.display()
                )
            );
            assert_eq!(next.file_id, "later");
            assert_eq!(next.audio, vec![-2000.0 / 32768.0]);
        }

        #[test]
        fn dropping_a_decoder_with_a_raced_fifo_does_not_wait_for_a_writer() {
            use super::super::{PendingDecode, ProbedInput};

            let directory = tempfile::tempdir().unwrap();
            let first = directory.path().join("first.wav");
            let path = directory.path().join("raced.wav");
            write_wav(&first, 1234, 16_000);
            create_fifo(&path);
            let (finished_tx, finished_rx) = mpsc::channel();
            let consumer = thread::spawn(move || {
                let probe = ProbedInput::open_candidate(&path).map(Some);
                let mut decoder = WavBatchDecoder::new(vec![first]).unwrap();
                decoder.schedule_probed(path, probe).unwrap();
                assert!(matches!(
                    decoder.pending.back(),
                    Some(PendingDecode::Changed(_))
                ));
                let first = decoder.next().unwrap().unwrap();
                // leave the ordered error unconsumed and join all workers before signalling
                drop(decoder);
                finished_tx.send(first).unwrap();
            });
            let first = finished_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("dropping a raced FIFO must not wait for a writer");
            consumer.join().unwrap();
            assert_eq!(first.file_id, "first");
            assert_eq!(first.audio, vec![1234.0 / 32768.0]);
        }
    }
}
