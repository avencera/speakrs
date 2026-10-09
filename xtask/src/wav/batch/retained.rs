use std::fs::{File, FileType};
use std::io::{self, Read, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileTypeExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const WAIT_INTERVAL: Duration = Duration::from_millis(10);

pub(super) fn set_nonblocking(file: &File, enabled: bool) -> io::Result<()> {
    let descriptor = file.as_raw_fd();
    // safety: the owned file keeps this descriptor valid throughout both fcntl calls
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags == -1 {
        return Err(io::Error::last_os_error());
    }

    let flags = if enabled {
        flags | libc::O_NONBLOCK
    } else {
        flags & !libc::O_NONBLOCK
    };

    // safety: F_SETFL takes an integer flag value and the descriptor remains owned
    if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags) } == -1 {
        return Err(io::Error::last_os_error());
    }

    Ok(())
}

/// An opened non-regular input kept connected until its in-order turn
pub(super) struct RetainedInput {
    file: File,
    state: ReadState,
}

impl RetainedInput {
    pub(super) fn new(file: File, file_type: FileType) -> Self {
        Self {
            file,
            state: if file_type.is_fifo() {
                ReadState::WaitingForWriter
            } else {
                ReadState::Reading
            },
        }
    }

    pub(super) fn into_reader(self, cancelled: &AtomicBool) -> io::Result<RetainedReader<'_>> {
        // only consumption uses nonblocking reads, so waits remain cancellable even mid-header
        set_nonblocking(&self.file, true)?;
        Ok(RetainedReader {
            file: self.file,
            cancelled,
            state: self.state,
        })
    }
}

enum ReadState {
    WaitingForWriter,
    Reading,
}

pub(super) struct RetainedReader<'a> {
    file: File,
    cancelled: &'a AtomicBool,
    state: ReadState,
}

impl Read for RetainedReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }

        loop {
            if self.cancelled.load(Ordering::Relaxed) {
                // read_exact retries Interrupted, so cancellation must use a terminal error kind
                return Err(io::Error::other("audio decode cancelled"));
            }

            match self.file.read(buffer) {
                // an unopened FIFO writer looks like EOF; retain the connection until data arrives
                Ok(0) if matches!(self.state, ReadState::WaitingForWriter) => {}
                Ok(count) => {
                    self.state = ReadState::Reading;
                    return Ok(count);
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }

            // bounded waits also work on systems where poll does not report FIFO EOF
            thread::sleep(WAIT_INTERVAL);
        }
    }
}

impl Seek for RetainedReader<'_> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.file.seek(position)
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::fs::OpenOptions;
    use std::io::{Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::{RetainedInput, set_nonblocking};

    fn retained_fifo() -> (tempfile::TempDir, PathBuf, RetainedInput) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pipe.wav");
        let name = CString::new(path.as_os_str().as_bytes()).unwrap();
        // safety: the path is a valid NUL-terminated string held through this call
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(&path)
            .unwrap();

        let file_type = file.metadata().unwrap().file_type();
        set_nonblocking(&file, false).unwrap();
        (directory, path, RetainedInput::new(file, file_type))
    }

    #[test]
    fn retained_fifo_waits_for_first_writer_and_preserves_eof() {
        let (_directory, path, input) = retained_fifo();
        // safety: the pending input owns the descriptor throughout this flags check
        let flags = unsafe { libc::fcntl(input.file.as_raw_fd(), libc::F_GETFL) };
        assert_ne!(flags, -1);
        assert_eq!(flags & libc::O_NONBLOCK, 0);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let consumer = thread::spawn(move || {
            let cancelled = AtomicBool::new(false);
            let mut reader = input.into_reader(&cancelled).unwrap();
            let mut bytes = Vec::new();
            started_tx.send(()).unwrap();
            let result = reader.read_to_end(&mut bytes);
            finished_tx.send((result, bytes)).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            finished_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
        let mut writer = OpenOptions::new().write(true).open(path).unwrap();
        writer.write_all(b"retained samples").unwrap();
        drop(writer);
        let (result, bytes) = finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        consumer.join().unwrap();
        assert_eq!(result.unwrap(), 16);
        assert_eq!(bytes, b"retained samples");
    }

    #[test]
    fn retained_fifo_cancels_before_a_writer_connects() {
        let (_directory, _path, input) = retained_fifo();
        let cancelled = Arc::new(AtomicBool::new(false));
        let reader_cancelled = Arc::clone(&cancelled);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let consumer = thread::spawn(move || {
            let mut reader = input.into_reader(&reader_cancelled).unwrap();
            started_tx.send(()).unwrap();
            let result = reader.read_exact(&mut [0u8; 4]);
            finished_tx.send(result).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            finished_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
        cancelled.store(true, Ordering::Relaxed);
        let error = finished_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap_err();
        consumer.join().unwrap();
        assert_eq!(error.to_string(), "audio decode cancelled");
        assert_eq!(error.kind(), std::io::ErrorKind::Other);
    }

    #[test]
    fn retained_fifo_cancels_a_partial_read_with_a_connected_writer() {
        let (_directory, path, input) = retained_fifo();
        let mut writer = OpenOptions::new().write(true).open(path).unwrap();
        writer.write_all(&[1, 2]).unwrap();
        let cancelled = Arc::new(AtomicBool::new(false));
        let reader_cancelled = Arc::clone(&cancelled);
        let (started_tx, started_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let consumer = thread::spawn(move || {
            let mut reader = input.into_reader(&reader_cancelled).unwrap();
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes[..2]).unwrap();
            started_tx.send(()).unwrap();
            let result = reader.read_exact(&mut bytes[2..]);
            finished_tx.send((result, bytes)).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            finished_rx
                .recv_timeout(Duration::from_millis(100))
                .is_err()
        );
        cancelled.store(true, Ordering::Relaxed);
        let (result, bytes) = finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        consumer.join().unwrap();
        drop(writer);
        assert_eq!(bytes[..2], [1, 2]);
        assert_eq!(result.unwrap_err().to_string(), "audio decode cancelled");
    }

    #[test]
    fn non_fifo_eof_is_not_a_writer_wait() {
        let file = std::fs::File::open("/dev/null").unwrap();
        let file_type = file.metadata().unwrap().file_type();
        let input = RetainedInput::new(file, file_type);
        let cancelled = AtomicBool::new(false);
        let mut reader = input.into_reader(&cancelled).unwrap();
        assert_eq!(reader.read(&mut [0u8; 4]).unwrap(), 0);
    }
}
