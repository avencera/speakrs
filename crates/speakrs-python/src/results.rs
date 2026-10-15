//! Result and event types returned to Python

use pyo3::prelude::*;
use speakrs_sdk::speakrs::segment::Segment;
use speakrs_sdk::{Diarization, ProgressEvent, QueueItem, TimingStats, mode_name};

use crate::{py_bool, py_opt_str, sdk_err};

/// One speaker turn
#[pyclass(name = "Segment", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PySegment(pub(crate) Segment);

#[pymethods]
impl PySegment {
    #[new]
    fn new(start: f64, end: f64, speaker: String) -> Self {
        Self(Segment {
            start,
            end,
            speaker,
        })
    }

    /// Start time in seconds
    #[getter]
    fn start(&self) -> f64 {
        self.0.start
    }

    /// End time in seconds
    #[getter]
    fn end(&self) -> f64 {
        self.0.end
    }

    /// Speaker label such as `SPEAKER_00`
    #[getter]
    fn speaker(&self) -> &str {
        &self.0.speaker
    }

    /// `end - start` in seconds
    #[getter]
    fn duration(&self) -> f64 {
        self.0.end - self.0.start
    }

    fn __repr__(&self) -> String {
        format!(
            "Segment(start={:.3}, end={:.3}, speaker='{}')",
            self.0.start, self.0.end, self.0.speaker
        )
    }
}

/// Wall-clock time per stage of a single-file call, in milliseconds
#[pyclass(name = "TimingStats", module = "speakrs", frozen)]
#[derive(Clone)]
pub(crate) struct PyTimingStats(pub(crate) TimingStats);

#[pymethods]
impl PyTimingStats {
    /// Decoding the audio file; zero for in-memory samples
    #[getter]
    fn audio_decode_ms(&self) -> f64 {
        self.0.audio_decode_ms
    }

    /// Resampling to 16 kHz; zero when the audio was already 16 kHz
    #[getter]
    fn audio_resample_ms(&self) -> f64 {
        self.0.audio_resample_ms
    }

    /// Inference, clustering, and reconstruction
    #[getter]
    fn pipeline_ms(&self) -> f64 {
        self.0.pipeline_ms
    }

    /// The whole call
    #[getter]
    fn total_ms(&self) -> f64 {
        self.0.total_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "TimingStats(audio_decode_ms={:.1}, audio_resample_ms={:.1}, pipeline_ms={:.1}, total_ms={:.1})",
            self.0.audio_decode_ms, self.0.audio_resample_ms, self.0.pipeline_ms, self.0.total_ms
        )
    }
}

/// Speaker turns for one input
#[pyclass(name = "DiarizationResult", module = "speakrs", frozen)]
#[derive(Clone)]
pub(crate) struct PyDiarizationResult(pub(crate) Diarization);

#[pymethods]
impl PyDiarizationResult {
    /// Identifier used in RTTM output
    #[getter]
    fn file_id(&self) -> &str {
        &self.0.file_id
    }

    /// Input duration in seconds
    #[getter]
    fn duration(&self) -> f64 {
        self.0.duration
    }

    /// Speaker turns in time order; overlapping speech gives overlapping segments
    #[getter]
    fn segments(&self) -> Vec<PySegment> {
        self.0.segments.iter().cloned().map(PySegment).collect()
    }

    /// Speaker turns with at most one speaker at a time; overlaps go to the speaker
    /// with the highest activation
    #[getter]
    fn exclusive_segments(&self) -> Vec<PySegment> {
        self.0
            .exclusive_segments
            .iter()
            .cloned()
            .map(PySegment)
            .collect()
    }

    /// Sorted, distinct speaker labels
    #[getter]
    fn speakers(&self) -> Vec<String> {
        let mut speakers: Vec<String> = self
            .0
            .segments
            .iter()
            .map(|segment| segment.speaker.clone())
            .collect();
        speakers.sort();
        speakers.dedup();
        speakers
    }

    /// Execution mode that produced the result, such as `"CPU"`
    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.0.mode)
    }

    /// Model revision, or `None` for an unverified local model directory
    #[getter]
    fn model_revision(&self) -> Option<&str> {
        self.0.model_revision.as_deref()
    }

    /// Stage timings for single-file calls; `None` for batch and queue results
    #[getter]
    fn timing(&self) -> Option<PyTimingStats> {
        self.0.timing.map(PyTimingStats)
    }

    /// Render RTTM, one `SPEAKER` line per segment
    ///
    /// `file_id` overrides the result's file id; `exclusive=True` renders
    /// `exclusive_segments`.
    #[pyo3(signature = (file_id=None, *, exclusive=false))]
    fn rttm(&self, file_id: Option<&str>, exclusive: bool) -> String {
        self.0.rttm(file_id, exclusive)
    }

    fn __len__(&self) -> usize {
        self.0.segments.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DiarizationResult(file_id='{}', duration={:.3}, segments={}, speakers={}, mode='{}')",
            self.0.file_id,
            self.0.duration,
            self.0.segments.len(),
            self.speakers().len(),
            mode_name(self.0.mode)
        )
    }
}

/// A finished queue job
#[pyclass(name = "QueueResult", module = "speakrs", frozen)]
pub(crate) struct PyQueueResult(pub(crate) QueueItem);

#[pymethods]
impl PyQueueResult {
    /// Id returned by the push that submitted the job
    #[getter]
    fn job_id(&self) -> u64 {
        self.0.job_id
    }

    /// File id given at push time
    #[getter]
    fn file_id(&self) -> &str {
        &self.0.file_id
    }

    /// Whether the job succeeded
    #[getter]
    fn ok(&self) -> bool {
        self.0.result.is_ok()
    }

    /// The result, or `None` if the job failed
    #[getter]
    fn result(&self) -> Option<PyDiarizationResult> {
        self.0
            .result
            .as_ref()
            .ok()
            .cloned()
            .map(PyDiarizationResult)
    }

    /// The `SpeakrsError` the job failed with, or `None`
    #[getter]
    fn error(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0
            .result
            .as_ref()
            .err()
            .map(|error| sdk_err(error.clone()).into_value(py).into_any())
    }

    /// Return the result, or raise the job's `SpeakrsError`
    fn unwrap(&self) -> PyResult<PyDiarizationResult> {
        self.0
            .result
            .clone()
            .map(PyDiarizationResult)
            .map_err(sdk_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "QueueResult(job_id={}, file_id='{}', ok={})",
            self.0.job_id,
            self.0.file_id,
            py_bool(self.0.result.is_ok())
        )
    }
}

/// A progress report passed to a `progress` callback
#[pyclass(name = "ProgressEvent", module = "speakrs", frozen)]
pub(crate) struct PyProgressEvent(pub(crate) ProgressEvent);

#[pymethods]
impl PyProgressEvent {
    /// One of the `ProgressKind` values, such as `"decoding_audio"`
    #[getter]
    fn kind(&self) -> &'static str {
        match self.0 {
            ProgressEvent::DownloadingModel { .. } => "downloading_model",
            ProgressEvent::DecodingAudio => "decoding_audio",
            ProgressEvent::ResamplingAudio => "resampling_audio",
            ProgressEvent::RunningPipeline => "running_pipeline",
            ProgressEvent::BatchFile { .. } => "batch_file",
            ProgressEvent::Completed => "completed",
        }
    }

    /// Model file for `downloading_model`, file id for `batch_file`, else `None`
    #[getter]
    fn name(&self) -> Option<&str> {
        match &self.0 {
            ProgressEvent::DownloadingModel { file } => Some(file),
            ProgressEvent::BatchFile { file_id, .. } => Some(file_id),
            _ => None,
        }
    }

    /// Zero-based input position for `batch_file`, else `None`
    #[getter]
    fn index(&self) -> Option<usize> {
        match self.0 {
            ProgressEvent::BatchFile { index, .. } => Some(index),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressEvent(kind='{}', name={}, index={})",
            self.kind(),
            py_opt_str(self.name()),
            self.index()
                .map_or("None".to_string(), |index| index.to_string())
        )
    }
}
