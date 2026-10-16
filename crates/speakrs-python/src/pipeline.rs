//! The pipeline and queue behind `speakrs.Pipeline` and `speakrs.Queue`

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use pyo3::buffer::PyUntypedBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use speakrs_sdk::speakrs::ExecutionMode;
use speakrs_sdk::speakrs::pipeline::QueueConfig;
use speakrs_sdk::{
    CancelToken, ProgressEvent, RunOptions, SdkError, SdkPipeline, SdkQueue, mode_name,
};

use crate::config::{PyPipelineConfig, PyRuntimeConfig};
use crate::models::PyPreparedModels;
use crate::results::{PyDiarizationResult, PyProgressEvent, PyQueueResult};
use crate::{config_err, lock, mode_from, py_bool, sdk_err};

// how long a blocking `recv` waits before checking for Ctrl+C
const RECV_SIGNAL_POLL: Duration = Duration::from_millis(100);

thread_local! {
    // set while a progress callback runs, so a callback that calls back into a
    // pipeline fails instead of waiting forever on the lock its caller holds
    static IN_PROGRESS_CALLBACK: Cell<bool> = const { Cell::new(false) };
}

/// Cancels a running diarization from any thread
///
/// The check runs between stages and between batch files; a stage that has
/// already started finishes first.
#[pyclass(name = "CancelToken", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
pub(crate) struct PyCancelToken(pub(crate) CancelToken);

#[pymethods]
impl PyCancelToken {
    #[new]
    fn new() -> Self {
        Self(CancelToken::new())
    }

    /// Request cancellation
    fn cancel(&self) {
        self.0.cancel();
    }

    /// Whether cancellation was requested
    #[getter]
    fn cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    /// A token cancelled along with this one, whose own cancellation leaves this
    /// one untouched
    fn child(&self) -> Self {
        Self(self.0.child())
    }

    fn __repr__(&self) -> String {
        format!("CancelToken(cancelled={})", py_bool(self.0.is_cancelled()))
    }
}

/// 16 kHz mono samples, borrowed from a float32 buffer when possible
///
/// The borrow outlives the GIL, so the caller must not write the buffer during the
/// call, which the Python docs state. Copying instead was measured at about 4 ms
/// per 8.7 minutes of audio on a Linux server, 2-3% of a CUDA-fast run on an H100,
/// and a multi-threaded copy was slower still
enum Samples {
    // checked at extraction: 1-D, native-endian float32, C-contiguous, aligned, non-empty
    Borrowed(PyUntypedBuffer),
    Owned(Vec<f32>),
}

impl Samples {
    fn extract(samples: &Bound<'_, PyAny>) -> PyResult<Self> {
        let Ok(buffer) = PyUntypedBuffer::get(samples) else {
            return samples.extract::<Vec<f32>>().map(Self::Owned).map_err(|_| {
                PyTypeError::new_err("samples must be a float32 buffer or a sequence of floats")
            });
        };

        // the format is checked here because PyO3 0.30's own float32 check accepts
        // the non-native byte order and rejects an explicit native one
        let format = buffer.format().to_bytes();
        let Some(byte_order) = float32_byte_order(format).filter(|_| buffer.item_size() == 4)
        else {
            return Err(PyTypeError::new_err(format!(
                "samples must be float32, got buffer format '{}'; convert with numpy.asarray(samples, dtype=numpy.float32)",
                String::from_utf8_lossy(format)
            )));
        };
        if buffer.dimensions() != 1 {
            return Err(PyValueError::new_err(format!(
                "samples must be a 1-D array of mono audio, got {} dimensions",
                buffer.dimensions()
            )));
        }

        let aligned = buffer.buf_ptr().cast::<f32>().is_aligned();
        if byte_order == ByteOrder::Native
            && buffer.is_c_contiguous()
            && aligned
            && buffer.item_count() > 0
        {
            return Ok(Self::Borrowed(buffer));
        }

        // strided, unaligned, or byte-swapped buffers cannot be borrowed as a slice
        let copied = (0..buffer.item_count())
            .map(|index| {
                // SAFETY: `index` is within the 1-D shape and each item is 4 bytes
                let bits = unsafe { buffer.get_ptr(&[index]).cast::<u32>().read_unaligned() };
                match byte_order {
                    ByteOrder::Native => f32::from_bits(bits),
                    ByteOrder::Swapped => f32::from_bits(bits.swap_bytes()),
                }
            })
            .collect();
        Ok(Self::Owned(copied))
    }

    fn as_slice(&self) -> &[f32] {
        match self {
            // SAFETY: extraction checked the layout, and the buffer export keeps the
            // memory alive until `self` drops; a caller that mutates the array during
            // the call races with this read, as with any extension that releases the GIL
            Self::Borrowed(buffer) => unsafe {
                std::slice::from_raw_parts(buffer.buf_ptr().cast::<f32>(), buffer.item_count())
            },
            Self::Owned(samples) => samples,
        }
    }

    fn into_vec(self) -> Vec<f32> {
        match self {
            Self::Owned(samples) => samples,
            borrowed => borrowed.as_slice().to_vec(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    Native,
    Swapped,
}

/// Byte order of a struct-module format string for one float32, or `None` for
/// any other type
fn float32_byte_order(format: &[u8]) -> Option<ByteOrder> {
    let (prefix, code) = match format {
        [code] => (b'@', *code),
        [prefix, code] => (*prefix, *code),
        _ => return None,
    };
    if code != b'f' {
        return None;
    }

    let little = match prefix {
        b'@' | b'=' => return Some(ByteOrder::Native),
        b'<' => true,
        b'>' | b'!' => false,
        _ => return None,
    };
    if little == cfg!(target_endian = "little") {
        Some(ByteOrder::Native)
    } else {
        Some(ByteOrder::Swapped)
    }
}

/// Collects the first exception a Python progress callback raises
///
/// That exception, or a Ctrl+C seen at an event, also cancels the run at the
/// next stage or file boundary
pub(crate) struct ProgressSink {
    callback: Option<Py<PyAny>>,
    cancel: CancelToken,
    error: Mutex<Option<PyErr>>,
}

impl ProgressSink {
    pub(crate) fn new(callback: Option<Py<PyAny>>, cancel: CancelToken) -> Self {
        Self {
            callback,
            cancel,
            error: Mutex::new(None),
        }
    }

    pub(crate) fn send(&self, event: ProgressEvent) {
        Python::attach(|py| {
            // Ctrl+C sets a flag that only Python code observes, so it is polled here
            if let Err(error) = py.check_signals() {
                self.record(error);
            }

            if let Some(callback) = &self.callback {
                // restore rather than clear, so a nested callback, such as one passed
                // to `prepare()` from inside this one, keeps the outer guard set; the
                // return value drops inside the guard because its finalizer is Python
                let outer = IN_PROGRESS_CALLBACK.replace(true);
                let called = callback.call1(py, (PyProgressEvent(event),)).map(drop);
                IN_PROGRESS_CALLBACK.set(outer);
                if let Err(error) = called {
                    self.record(error);
                }
            }
        });
    }

    fn record(&self, error: PyErr) {
        self.cancel.cancel();
        lock(&self.error).get_or_insert(error);
    }

    /// Prefer a callback's exception over the result, so it is never lost
    pub(crate) fn finish<T>(self, result: Result<T, SdkError>) -> PyResult<T> {
        match self
            .error
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
        {
            Some(error) => Err(error),
            None => result.map_err(sdk_err),
        }
    }
}

/// Native pipeline; `speakrs.Pipeline` wraps it
#[pyclass(name = "_Pipeline", module = "speakrs._native", frozen)]
pub(crate) struct PyPipeline {
    // `None` after `into_queue`
    inner: Mutex<Option<SdkPipeline>>,
    mode: ExecutionMode,
}

impl PyPipeline {
    /// Run `f` with the GIL released and the pipeline locked
    ///
    /// Locking after release keeps a second Python thread that waits for the
    /// pipeline from holding the GIL a progress callback needs
    fn with_pipeline<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut SdkPipeline) -> Result<T, SdkError> + Send,
    ) -> Result<T, SdkError> {
        reject_reentry()?;
        py.detach(|| {
            let mut inner = lock(&self.inner);
            let pipeline = inner.as_mut().ok_or_else(|| {
                SdkError::configuration("the pipeline was moved into a queue with into_queue()")
            })?;
            f(pipeline)
        })
    }
}

#[pymethods]
impl PyPipeline {
    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.mode)
    }

    #[getter]
    fn pipeline_config(&self, py: Python<'_>) -> PyResult<PyPipelineConfig> {
        self.with_pipeline(py, |pipeline| Ok(pipeline.pipeline_config()))
            .map(PyPipelineConfig)
            .map_err(sdk_err)
    }

    #[pyo3(signature = (samples, file_id, pipeline_config=None, cancel_token=None, progress=None))]
    fn diarize_samples(
        &self,
        py: Python<'_>,
        samples: &Bound<'_, PyAny>,
        file_id: String,
        pipeline_config: Option<PyPipelineConfig>,
        cancel_token: Option<PyCancelToken>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<PyDiarizationResult> {
        let samples = Samples::extract(samples)?;
        let (token, sink) = cancellable(cancel_token, progress);
        let result = self.with_pipeline(py, |pipeline| {
            let callback = |event| sink.send(event);
            pipeline.diarize_samples(
                samples.as_slice(),
                &file_id,
                RunOptions {
                    pipeline_config: pipeline_config.as_ref().map(|config| &config.0),
                    cancel_token: Some(&token),
                    progress: Some(&callback),
                },
            )
        });
        sink.finish(result).map(PyDiarizationResult)
    }

    #[pyo3(signature = (path, file_id, pipeline_config=None, cancel_token=None, progress=None))]
    fn diarize_file(
        &self,
        py: Python<'_>,
        path: PathBuf,
        file_id: String,
        pipeline_config: Option<PyPipelineConfig>,
        cancel_token: Option<PyCancelToken>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<PyDiarizationResult> {
        let (token, sink) = cancellable(cancel_token, progress);
        let result = self.with_pipeline(py, |pipeline| {
            let callback = |event| sink.send(event);
            pipeline.diarize_file(
                &path,
                &file_id,
                RunOptions {
                    pipeline_config: pipeline_config.as_ref().map(|config| &config.0),
                    cancel_token: Some(&token),
                    progress: Some(&callback),
                },
            )
        });
        sink.finish(result).map(PyDiarizationResult)
    }

    #[pyo3(signature = (files, pipeline_config=None, cancel_token=None, progress=None))]
    fn diarize_files(
        &self,
        py: Python<'_>,
        files: Vec<(String, PathBuf)>,
        pipeline_config: Option<PyPipelineConfig>,
        cancel_token: Option<PyCancelToken>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<Vec<PyDiarizationResult>> {
        let (token, sink) = cancellable(cancel_token, progress);
        let result = self.with_pipeline(py, |pipeline| {
            let callback = |event| sink.send(event);
            pipeline.diarize_files(
                files,
                RunOptions {
                    pipeline_config: pipeline_config.as_ref().map(|config| &config.0),
                    cancel_token: Some(&token),
                    progress: Some(&callback),
                },
            )
        });
        sink.finish(result)
            .map(|results| results.into_iter().map(PyDiarizationResult).collect())
    }

    #[pyo3(name = "into_queue", signature = (pipeline_config=None, capacity=None))]
    fn make_queue(
        &self,
        py: Python<'_>,
        pipeline_config: Option<PyPipelineConfig>,
        capacity: Option<usize>,
    ) -> PyResult<PyQueue> {
        reject_reentry().map_err(sdk_err)?;
        // validate before taking the pipeline, so a bad capacity leaves it usable
        let queue_config = match capacity {
            Some(capacity) => QueueConfig::new(capacity).map_err(config_err)?,
            None => QueueConfig::default(),
        };
        py.detach(|| {
            let pipeline = lock(&self.inner).take().ok_or_else(|| {
                SdkError::configuration("the pipeline was already moved into a queue")
            })?;
            pipeline.into_queue(pipeline_config.map(|config| config.0), queue_config)
        })
        .map(PyQueue)
        .map_err(sdk_err)
    }
}

/// Native queue; `speakrs.Queue` wraps it
#[pyclass(name = "_Queue", module = "speakrs._native", frozen)]
pub(crate) struct PyQueue(pub(crate) SdkQueue);

#[pymethods]
impl PyQueue {
    fn push_samples(
        &self,
        py: Python<'_>,
        file_id: String,
        samples: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        let samples = Samples::extract(samples)?;
        py.detach(|| self.0.push_samples(file_id, samples.into_vec()))
            .map_err(sdk_err)
    }

    fn push_file(&self, py: Python<'_>, file_id: String, path: PathBuf) -> PyResult<u64> {
        py.detach(|| self.0.push_file(file_id, &path))
            .map_err(sdk_err)
    }

    fn close(&self) {
        self.0.close();
    }

    fn recv(&self, py: Python<'_>) -> PyResult<PyQueueResult> {
        // wait in slices so Ctrl+C can interrupt a wait on an open, idle queue
        loop {
            if let Some(item) = py
                .detach(|| self.0.recv_timeout(RECV_SIGNAL_POLL))
                .map_err(sdk_err)?
            {
                return Ok(PyQueueResult(item));
            }
            py.check_signals()?;
        }
    }

    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<PyQueueResult>> {
        py.detach(|| self.0.try_recv())
            .map(|item| item.map(PyQueueResult))
            .map_err(sdk_err)
    }
}

/// Fail instead of waiting forever when a progress callback calls back into a
/// pipeline whose lock its caller holds
fn reject_reentry() -> Result<(), SdkError> {
    if IN_PROGRESS_CALLBACK.get() {
        return Err(SdkError::configuration(
            "a progress callback cannot call a pipeline; the running call holds it",
        ));
    }

    Ok(())
}

/// The caller's cancel token, or a private one, wired to a progress sink
///
/// Every call gets its own token so Ctrl+C and a raising callback stop the run
/// early. It is a child of the caller's token, so cancelling the call never
/// cancels a token the caller shares with other calls
fn cancellable(
    cancel_token: Option<PyCancelToken>,
    progress: Option<Py<PyAny>>,
) -> (CancelToken, ProgressSink) {
    let token = cancel_token.map_or_else(CancelToken::new, |token| token.0.child());
    let sink = ProgressSink::new(progress, token.clone());
    (token, sink)
}

#[pyfunction]
pub(crate) fn _build_pipeline(
    py: Python<'_>,
    prepared: PyPreparedModels,
    mode: &str,
    pipeline_config: Option<PyPipelineConfig>,
    runtime_config: Option<PyRuntimeConfig>,
) -> PyResult<PyPipeline> {
    let mode = mode_from(mode)?;
    let runtime = runtime_config
        .map(|config| config.0.to_runtime_config())
        .transpose()
        .map_err(sdk_err)?;
    py.detach(|| {
        SdkPipeline::from_prepared(
            &prepared.0,
            mode,
            pipeline_config.map(|config| config.0),
            runtime,
        )
    })
    .map(|pipeline| PyPipeline {
        inner: Mutex::new(Some(pipeline)),
        mode,
    })
    .map_err(sdk_err)
}
