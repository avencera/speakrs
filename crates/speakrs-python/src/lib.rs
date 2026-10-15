//! The `speakrs._native` extension module
//!
//! The Python package in `python/speakrs` wraps the pipeline and queue classes and
//! re-exports everything else from here unchanged

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use speakrs_sdk::speakrs::ExecutionMode;
use speakrs_sdk::speakrs::pipeline::{
    ActivityCleanup, AhcConfig, ClusteringBackend, ClusteringConfig, PipelineConfig,
    ReconstructMethod, ResponsibilityInitialization, VbxConfig,
};
use speakrs_sdk::speakrs::segment::Segment;
use speakrs_sdk::{
    CacheEntry, CancelToken, Diarization, ModelManifest, ModelManifestEntry, ModelManifestError,
    ModelPrepareError, ModelStore, PrepareModelsOptions, PreparedModels, ProgressEvent, QueueItem,
    RunOptions, RuntimeOptions, SdkError, SdkErrorCategory, SdkPipeline, SdkQueue, TimingStats,
    mode_name, parse_mode,
};

pyo3::import_exception!(speakrs._errors, SpeakrsError);

fn sdk_err(error: SdkError) -> PyErr {
    SpeakrsError::new_err((error.category.code(), error.code, error.message))
}

fn category_err(category: SdkErrorCategory, message: impl ToString) -> PyErr {
    sdk_err(SdkError::new(category, message.to_string()))
}

fn config_err(error: impl ToString) -> PyErr {
    category_err(SdkErrorCategory::Configuration, error)
}

fn prepare_sdk_err(error: ModelPrepareError) -> SdkError {
    match error {
        ModelPrepareError::Manifest(error) => {
            SdkError::new(SdkErrorCategory::ModelManifest, error.to_string())
        }
        other => SdkError::new(SdkErrorCategory::ModelPrepare, other.to_string()),
    }
}

fn prepare_err(error: ModelPrepareError) -> PyErr {
    sdk_err(prepare_sdk_err(error))
}

fn manifest_err(error: ModelManifestError) -> PyErr {
    category_err(SdkErrorCategory::ModelManifest, error)
}

// ClusteringConfig has no PartialEq because a backend may hold non-comparable state
fn clustering_eq(lhs: &ClusteringConfig, rhs: &ClusteringConfig) -> bool {
    let same_backend = match (lhs.backend(), rhs.backend()) {
        (ClusteringBackend::GaussianVbx(lhs), ClusteringBackend::GaussianVbx(rhs)) => lhs == rhs,
        #[allow(unreachable_patterns)]
        _ => false,
    };
    same_backend
        && lhs.ahc() == rhs.ahc()
        && lhs.speaker_keep_threshold() == rhs.speaker_keep_threshold()
}

fn mode_from(name: &str) -> PyResult<ExecutionMode> {
    parse_mode(name).map_err(sdk_err)
}

// a panic while holding one of these locks leaves plain data behind, so it stays usable
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Cancels a running diarization from any thread
///
/// The check runs between stages and between batch files; a stage that has
/// already started finishes first.
#[pyclass(name = "CancelToken", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
struct PyCancelToken(CancelToken);

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

    fn __repr__(&self) -> String {
        format!("CancelToken(cancelled={})", py_bool(self.0.is_cancelled()))
    }
}

/// Frame-count cleanup applied to each speaker's activity after clustering
///
/// Frames are 16.875 ms apart. All zeros, the default for full-precision modes,
/// leaves activity unchanged.
#[pyclass(name = "ActivityCleanup", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
struct PyActivityCleanup(ActivityCleanup);

#[pymethods]
impl PyActivityCleanup {
    #[new]
    #[pyo3(signature = (min_active_frames=0, max_inactive_gap_frames=0, pad_before_frames=0, pad_after_frames=0))]
    fn new(
        min_active_frames: usize,
        max_inactive_gap_frames: usize,
        pad_before_frames: usize,
        pad_after_frames: usize,
    ) -> Self {
        Self(ActivityCleanup::new(
            min_active_frames,
            max_inactive_gap_frames,
            pad_before_frames,
            pad_after_frames,
        ))
    }

    /// Shortest active run kept, in frames
    #[getter]
    fn min_active_frames(&self) -> usize {
        self.0.min_active_frames()
    }

    /// Longest inactive gap filled between two active runs, in frames
    #[getter]
    fn max_inactive_gap_frames(&self) -> usize {
        self.0.max_inactive_gap_frames()
    }

    /// Frames added before each active run
    #[getter]
    fn pad_before_frames(&self) -> usize {
        self.0.pad_before_frames()
    }

    /// Frames added after each active run
    #[getter]
    fn pad_after_frames(&self) -> usize {
        self.0.pad_after_frames()
    }

    fn __repr__(&self) -> String {
        format!(
            "ActivityCleanup(min_active_frames={}, max_inactive_gap_frames={}, pad_before_frames={}, pad_after_frames={})",
            self.0.min_active_frames(),
            self.0.max_inactive_gap_frames(),
            self.0.pad_before_frames(),
            self.0.pad_after_frames()
        )
    }
}

/// Agglomerative clustering that seeds VBx
#[pyclass(name = "AhcConfig", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
struct PyAhcConfig(AhcConfig);

#[pymethods]
impl PyAhcConfig {
    /// Raises `SpeakrsError` (category `configuration`) for a negative or
    /// non-finite threshold
    #[new]
    #[pyo3(signature = (threshold=None))]
    fn new(threshold: Option<f32>) -> PyResult<Self> {
        let config = match threshold {
            Some(threshold) => AhcConfig::new(threshold).map_err(config_err)?,
            None => AhcConfig::default(),
        };
        Ok(Self(config))
    }

    /// Distance threshold for merging clusters
    #[getter]
    fn threshold(&self) -> f32 {
        self.0.threshold()
    }

    fn __repr__(&self) -> String {
        format!("AhcConfig(threshold={:?})", self.0.threshold())
    }
}

/// Variational Bayes (VBx) speaker clustering
#[pyclass(name = "VbxConfig", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
struct PyVbxConfig(VbxConfig);

#[pymethods]
impl PyVbxConfig {
    /// Omitted values keep the defaults. `init_smoothing` below zero starts from
    /// hard AHC assignments, zero from uniform responsibilities, and above zero
    /// from softmax-smoothed ones.
    ///
    /// Raises `SpeakrsError` (category `configuration`) for out-of-range values
    #[new]
    #[pyo3(signature = (fa=None, fb=None, max_iters=None, epsilon=None, init_smoothing=None))]
    fn new(
        fa: Option<f64>,
        fb: Option<f64>,
        max_iters: Option<usize>,
        epsilon: Option<f64>,
        init_smoothing: Option<f64>,
    ) -> PyResult<Self> {
        let defaults = VbxConfig::default();
        let initialization = match init_smoothing {
            Some(value) => {
                ResponsibilityInitialization::from_smoothing(value).map_err(config_err)?
            }
            None => defaults.initialization(),
        };
        VbxConfig::new(
            fa.unwrap_or(defaults.fa()),
            fb.unwrap_or(defaults.fb()),
            max_iters.unwrap_or(defaults.max_iters()),
            epsilon.unwrap_or(defaults.epsilon()),
            initialization,
        )
        .map(Self)
        .map_err(config_err)
    }

    /// Speaker-model scale
    #[getter]
    fn fa(&self) -> f64 {
        self.0.fa()
    }

    /// Between-speaker scale
    #[getter]
    fn fb(&self) -> f64 {
        self.0.fb()
    }

    /// Maximum iterations
    #[getter]
    fn max_iters(&self) -> usize {
        self.0.max_iters()
    }

    /// Convergence threshold on the evidence lower bound
    #[getter]
    fn epsilon(&self) -> f64 {
        self.0.epsilon()
    }

    /// Responsibility initialization: -1.0 hard, 0.0 uniform, > 0 smoothed
    #[getter]
    fn init_smoothing(&self) -> f64 {
        match self.0.initialization() {
            ResponsibilityInitialization::Hard => -1.0,
            ResponsibilityInitialization::Uniform => 0.0,
            ResponsibilityInitialization::Smoothed(value) => value,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VbxConfig(fa={:?}, fb={:?}, max_iters={}, epsilon={:?}, init_smoothing={:?})",
            self.0.fa(),
            self.0.fb(),
            self.0.max_iters(),
            self.0.epsilon(),
            self.init_smoothing()
        )
    }
}

/// AHC, VBx, and the speaker-keep threshold
#[pyclass(name = "ClusteringConfig", module = "speakrs", frozen)]
#[derive(Clone)]
struct PyClusteringConfig(ClusteringConfig);

#[pymethods]
impl PyClusteringConfig {
    /// Raises `SpeakrsError` (category `configuration`) for a negative or
    /// non-finite `speaker_keep_threshold`
    #[new]
    #[pyo3(signature = (ahc=None, vbx=None, speaker_keep_threshold=None))]
    fn new(
        ahc: Option<PyRef<'_, PyAhcConfig>>,
        vbx: Option<PyRef<'_, PyVbxConfig>>,
        speaker_keep_threshold: Option<f64>,
    ) -> PyResult<Self> {
        let defaults = ClusteringConfig::default();
        ClusteringConfig::new(
            ahc.map_or(defaults.ahc(), |ahc| ahc.0),
            speaker_keep_threshold.unwrap_or(defaults.speaker_keep_threshold()),
            ClusteringBackend::GaussianVbx(vbx.map_or_else(VbxConfig::default, |vbx| vbx.0)),
        )
        .map(Self)
        .map_err(config_err)
    }

    /// Agglomerative clustering settings
    #[getter]
    fn ahc(&self) -> PyAhcConfig {
        PyAhcConfig(self.0.ahc())
    }

    /// VBx settings
    #[getter]
    fn vbx(&self) -> PyResult<PyVbxConfig> {
        match self.0.backend() {
            ClusteringBackend::GaussianVbx(vbx) => Ok(PyVbxConfig(vbx)),
            #[allow(unreachable_patterns)]
            _ => Err(category_err(
                SdkErrorCategory::Internal,
                "the clustering backend is not VBx",
            )),
        }
    }

    /// Minimum speaker activity weight that keeps a speaker
    #[getter]
    fn speaker_keep_threshold(&self) -> f64 {
        self.0.speaker_keep_threshold()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<Self>()
            .is_ok_and(|other| clustering_eq(&self.0, &other.get().0))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "ClusteringConfig(ahc={}, vbx={}, speaker_keep_threshold={:?})",
            self.ahc().__repr__(),
            self.vbx()?.__repr__(),
            self.0.speaker_keep_threshold()
        ))
    }
}

/// Post-inference settings: activity cleanup, clustering, segment merging, and
/// reconstruction
///
/// `PipelineConfig.for_mode(mode)` gives the defaults a pipeline uses for that mode.
#[pyclass(name = "PipelineConfig", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
struct PyPipelineConfig(PipelineConfig);

#[pymethods]
impl PyPipelineConfig {
    /// Omitted values keep the full-precision defaults. `reconstruct_method` is
    /// `"smoothed"` (default) or `"standard"`; `smoothing_epsilon` only applies to
    /// `"smoothed"`.
    #[new]
    #[pyo3(signature = (activity=None, clustering=None, merge_gap=None, reconstruct_method=None, smoothing_epsilon=None))]
    fn new(
        activity: Option<PyRef<'_, PyActivityCleanup>>,
        clustering: Option<PyRef<'_, PyClusteringConfig>>,
        merge_gap: Option<f64>,
        reconstruct_method: Option<&str>,
        smoothing_epsilon: Option<f32>,
    ) -> PyResult<Self> {
        let defaults = PipelineConfig::default();
        let default_epsilon = match defaults.reconstruct_method {
            ReconstructMethod::Smoothed { epsilon } => epsilon,
            _ => 0.1,
        };
        let reconstruct_method = match (reconstruct_method.unwrap_or("smoothed"), smoothing_epsilon)
        {
            ("smoothed", epsilon) => ReconstructMethod::Smoothed {
                epsilon: epsilon.unwrap_or(default_epsilon),
            },
            ("standard", None) => ReconstructMethod::Standard,
            ("standard", Some(_)) => {
                return Err(config_err(
                    "smoothing_epsilon only applies to reconstruct_method=\"smoothed\"",
                ));
            }
            (other, _) => {
                return Err(config_err(format!(
                    "unknown reconstruct_method `{other}`; expected smoothed or standard"
                )));
            }
        };
        let merge_gap = merge_gap.unwrap_or(defaults.merge_gap);
        if !(merge_gap.is_finite() && merge_gap >= 0.0) {
            return Err(config_err(format!(
                "merge_gap must be finite and non-negative, got {merge_gap}"
            )));
        }

        let mut config = defaults;
        if let Some(activity) = activity {
            config.activity = activity.0;
        }
        if let Some(clustering) = clustering {
            config.clustering = clustering.0;
        }
        config.merge_gap = merge_gap;
        config.reconstruct_method = reconstruct_method;
        Ok(Self(config))
    }

    /// The defaults a pipeline uses for `mode`
    #[staticmethod]
    fn for_mode(mode: &str) -> PyResult<Self> {
        Ok(Self(PipelineConfig::for_mode(mode_from(mode)?)))
    }

    /// Activity cleanup after clustering
    #[getter]
    fn activity(&self) -> PyActivityCleanup {
        PyActivityCleanup(self.0.activity)
    }

    /// Clustering settings
    #[getter]
    fn clustering(&self) -> PyClusteringConfig {
        PyClusteringConfig(self.0.clustering)
    }

    /// Largest gap in seconds between two segments of one speaker that is merged
    #[getter]
    fn merge_gap(&self) -> f64 {
        self.0.merge_gap
    }

    /// `"smoothed"` or `"standard"`
    #[getter]
    fn reconstruct_method(&self) -> &'static str {
        match self.0.reconstruct_method {
            ReconstructMethod::Standard => "standard",
            _ => "smoothed",
        }
    }

    /// Score difference below which smoothing keeps the previous speaker, or `None`
    /// for standard reconstruction
    #[getter]
    fn smoothing_epsilon(&self) -> Option<f32> {
        match self.0.reconstruct_method {
            ReconstructMethod::Smoothed { epsilon } => Some(epsilon),
            _ => None,
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Self>().is_ok_and(|other| {
            let other = &other.get().0;
            self.0.activity == other.activity
                && clustering_eq(&self.0.clustering, &other.clustering)
                && self.0.merge_gap == other.merge_gap
                && self.0.reconstruct_method == other.reconstruct_method
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let reconstruct = match self.smoothing_epsilon() {
            Some(epsilon) => {
                format!("reconstruct_method='smoothed', smoothing_epsilon={epsilon:?}")
            }
            None => "reconstruct_method='standard'".to_string(),
        };
        Ok(format!(
            "PipelineConfig(activity={}, clustering={}, merge_gap={:?}, {reconstruct})",
            self.activity().__repr__(),
            self.clustering().__repr__()?,
            self.0.merge_gap
        ))
    }
}

/// Backend-specific runtime settings
///
/// Each option only exists in builds with its backend; setting one the build
/// lacks raises `SpeakrsError` (category `unsupported_runtime`). Reading an
/// option the build lacks returns `None`.
#[pyclass(name = "RuntimeConfig", module = "speakrs", frozen, from_py_object, eq)]
#[derive(Clone, PartialEq)]
struct PyRuntimeConfig(RuntimeOptions);

#[pymethods]
impl PyRuntimeConfig {
    #[new]
    #[pyo3(signature = (*, coreml_compute_units=None, cuda_segmentation_math=None, cuda_embedding_math=None, cuda_graphs=None))]
    fn new(
        coreml_compute_units: Option<String>,
        cuda_segmentation_math: Option<String>,
        cuda_embedding_math: Option<String>,
        cuda_graphs: Option<bool>,
    ) -> PyResult<Self> {
        RuntimeOptions {
            coreml_compute_units,
            cuda_segmentation_math,
            cuda_embedding_math,
            cuda_graphs,
        }
        .resolved()
        .map(Self)
        .map_err(sdk_err)
    }

    /// CoreML compute units: `"all"`, `"cpu_and_neural_engine"`, or `"cpu_only"`
    #[getter]
    fn coreml_compute_units(&self) -> Option<String> {
        self.0.coreml_compute_units.clone()
    }

    /// CUDA segmentation precision, `"fp32"` (default) or `"tf32"`
    #[getter]
    fn cuda_segmentation_math(&self) -> Option<String> {
        self.0.cuda_segmentation_math.clone()
    }

    /// CUDA embedding precision, `"tf32"` (default) or `"fp32"`
    #[getter]
    fn cuda_embedding_math(&self) -> Option<String> {
        self.0.cuda_embedding_math.clone()
    }

    /// Whether CUDA modes replay captured CUDA graphs
    #[getter]
    fn cuda_graphs(&self) -> Option<bool> {
        self.0.cuda_graphs
    }

    fn __repr__(&self) -> String {
        format!(
            "RuntimeConfig(coreml_compute_units={}, cuda_segmentation_math={}, cuda_embedding_math={}, cuda_graphs={})",
            py_opt_str(self.0.coreml_compute_units.as_deref()),
            py_opt_str(self.0.cuda_segmentation_math.as_deref()),
            py_opt_str(self.0.cuda_embedding_math.as_deref()),
            self.0.cuda_graphs.map_or("None", py_bool)
        )
    }
}

/// One speaker turn
#[pyclass(name = "Segment", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
struct PySegment(Segment);

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
struct PyTimingStats(TimingStats);

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
struct PyDiarizationResult(Diarization);

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
struct PyQueueResult(QueueItem);

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
struct PyProgressEvent(ProgressEvent);

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

/// Size and SHA-256 digest of one model file
#[pyclass(name = "ModelManifestEntry", module = "speakrs", frozen)]
#[derive(Clone)]
struct PyModelManifestEntry(ModelManifestEntry);

#[pymethods]
impl PyModelManifestEntry {
    /// Path relative to the model directory
    #[getter]
    fn path(&self) -> &str {
        &self.0.path
    }

    /// File size in bytes
    #[getter]
    fn size_bytes(&self) -> u64 {
        self.0.size_bytes
    }

    /// Lowercase hex SHA-256 digest
    #[getter]
    fn sha256(&self) -> &str {
        &self.0.sha256
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelManifestEntry(path='{}', size_bytes={})",
            self.0.path, self.0.size_bytes
        )
    }
}

/// Sizes and digests of the files one mode loads
///
/// Save it with `to_json()` and pass it to `prepare(manifest=...)` later to check
/// that a model directory has not changed.
#[pyclass(name = "ModelManifest", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
struct PyModelManifest(ModelManifest);

#[pymethods]
impl PyModelManifest {
    /// Parse a manifest written by `to_json()`
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        ModelManifest::from_json(json)
            .map(Self)
            .map_err(manifest_err)
    }

    /// Serialize as JSON
    fn to_json(&self) -> String {
        self.0.to_json()
    }

    /// Repository the files came from
    #[getter]
    fn repository(&self) -> &str {
        &self.0.repository
    }

    /// Repository revision the files came from
    #[getter]
    fn revision(&self) -> &str {
        &self.0.revision
    }

    /// One entry per file, sorted by path
    #[getter]
    fn files(&self) -> Vec<PyModelManifestEntry> {
        self.0
            .files
            .iter()
            .cloned()
            .map(PyModelManifestEntry)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelManifest(repository='{}', revision='{}', files={})",
            self.0.repository,
            self.0.revision,
            self.0.files.len()
        )
    }
}

/// A model directory that holds every file one mode loads
#[pyclass(name = "PreparedModels", module = "speakrs", frozen, from_py_object)]
#[derive(Clone)]
struct PyPreparedModels(PreparedModels);

#[pymethods]
impl PyPreparedModels {
    /// Directory the pipeline loads from
    #[getter]
    fn model_dir(&self) -> PathBuf {
        self.0.model_dir.clone()
    }

    /// Mode the directory was prepared for
    #[getter]
    fn mode(&self) -> &'static str {
        mode_name(self.0.mode)
    }

    /// Model revision, or `None` for an unverified local directory
    #[getter]
    fn revision(&self) -> Option<&str> {
        self.0.revision.as_deref()
    }

    /// Manifest the files were verified against, if one was given
    #[getter]
    fn manifest(&self) -> Option<PyModelManifest> {
        self.0.manifest.clone().map(PyModelManifest)
    }

    fn __repr__(&self) -> String {
        format!(
            "PreparedModels(model_dir='{}', mode='{}', revision={})",
            self.0.model_dir.display(),
            mode_name(self.0.mode),
            py_opt_str(self.0.revision.as_deref())
        )
    }
}

/// One cached revision of the model repository
#[pyclass(name = "CacheEntry", module = "speakrs", frozen)]
#[derive(Clone)]
struct PyCacheEntry(CacheEntry);

#[pymethods]
impl PyCacheEntry {
    /// Hugging Face repository id
    #[getter]
    fn repository(&self) -> &str {
        &self.0.repository
    }

    /// Revision (commit hash)
    #[getter]
    fn revision(&self) -> &str {
        &self.0.revision
    }

    /// Snapshot directory
    #[getter]
    fn path(&self) -> PathBuf {
        self.0.path.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CacheEntry(revision='{}', path='{}')",
            self.0.revision,
            self.0.path.display()
        )
    }
}

/// 16 kHz mono samples, borrowed from a float32 buffer when possible
enum Samples {
    Borrowed(PyBuffer<f32>),
    Owned(Vec<f32>),
}

impl Samples {
    fn extract(samples: &Bound<'_, PyAny>) -> PyResult<Self> {
        match PyBuffer::<f32>::get(samples) {
            Ok(buffer) => {
                if buffer.dimensions() != 1 {
                    return Err(PyValueError::new_err(format!(
                        "samples must be a 1-D array of mono audio, got {} dimensions",
                        buffer.dimensions()
                    )));
                }
                if buffer.as_slice_ptr().is_some() {
                    Ok(Self::Borrowed(buffer))
                } else {
                    // strided views cannot be borrowed as one slice
                    Ok(Self::Owned(buffer.to_vec(samples.py())?))
                }
            }
            Err(error) => {
                if let Ok(buffer) = pyo3::buffer::PyUntypedBuffer::get(samples) {
                    return Err(PyTypeError::new_err(format!(
                        "samples must be float32, got buffer format '{}'; convert with numpy.asarray(samples, dtype=numpy.float32)",
                        buffer.format().to_string_lossy()
                    )));
                }
                samples.extract::<Vec<f32>>().map(Self::Owned).map_err(|_| {
                    PyTypeError::new_err(format!(
                        "samples must be a float32 buffer or a sequence of floats ({error})"
                    ))
                })
            }
        }
    }

    fn as_slice(&self) -> &[f32] {
        match self {
            // SAFETY: the buffer export keeps the memory alive and C-contiguous until
            // `self` drops; a caller that mutates the array during the call races
            // with this read, as with any extension that releases the GIL
            Self::Borrowed(buffer) => unsafe {
                buffer
                    .as_slice_ptr()
                    .expect("checked C-contiguous at extraction")
                    .as_ref()
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

/// Collects the first exception a Python progress callback raises
struct ProgressSink {
    callback: Option<Py<PyAny>>,
    cancel_on_signal: Option<CancelToken>,
    error: Mutex<Option<PyErr>>,
}

impl ProgressSink {
    fn new(callback: Option<Py<PyAny>>, cancel_on_signal: Option<CancelToken>) -> Self {
        Self {
            callback,
            cancel_on_signal,
            error: Mutex::new(None),
        }
    }

    fn is_active(&self) -> bool {
        self.callback.is_some() || self.cancel_on_signal.is_some()
    }

    fn send(&self, event: ProgressEvent) {
        Python::attach(|py| {
            // Ctrl+C sets a flag that only Python code observes, so batch calls poll it
            // here and cancel at the next file boundary
            if let Some(token) = &self.cancel_on_signal
                && let Err(error) = py.check_signals()
            {
                token.cancel();
                self.record(error);
            }

            if let Some(callback) = &self.callback
                && let Err(error) = callback.call1(py, (PyProgressEvent(event),))
            {
                self.record(error);
            }
        });
    }

    fn record(&self, error: PyErr) {
        lock(&self.error).get_or_insert(error);
    }

    /// Prefer a callback's exception over the result, so it is never lost
    fn finish<T>(self, result: Result<T, SdkError>) -> PyResult<T> {
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
struct PyPipeline {
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
        let sink = ProgressSink::new(progress, None);
        let result = self.with_pipeline(py, |pipeline| {
            let callback = |event| sink.send(event);
            pipeline.diarize_samples(
                samples.as_slice(),
                &file_id,
                RunOptions {
                    pipeline_config: pipeline_config.as_ref().map(|config| &config.0),
                    cancel_token: cancel_token.as_ref().map(|token| &token.0),
                    progress: sink.is_active().then_some(&callback),
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
        let sink = ProgressSink::new(progress, None);
        let result = self.with_pipeline(py, |pipeline| {
            let callback = |event| sink.send(event);
            pipeline.diarize_file(
                &path,
                &file_id,
                RunOptions {
                    pipeline_config: pipeline_config.as_ref().map(|config| &config.0),
                    cancel_token: cancel_token.as_ref().map(|token| &token.0),
                    progress: sink.is_active().then_some(&callback),
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
        let token = cancel_token.map_or_else(CancelToken::new, |token| token.0);
        let sink = ProgressSink::new(progress, Some(token.clone()));
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
        py.detach(|| {
            let pipeline = lock(&self.inner).take().ok_or_else(|| {
                SdkError::configuration("the pipeline was already moved into a queue")
            })?;
            pipeline.into_queue(pipeline_config.map(|config| config.0), capacity)
        })
        .map(PyQueue)
        .map_err(sdk_err)
    }
}

/// Native queue; `speakrs.Queue` wraps it
#[pyclass(name = "_Queue", module = "speakrs._native", frozen)]
struct PyQueue(SdkQueue);

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
        py.detach(|| self.0.recv())
            .map(PyQueueResult)
            .map_err(sdk_err)
    }

    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<PyQueueResult>> {
        py.detach(|| self.0.try_recv())
            .map(|item| item.map(PyQueueResult))
            .map_err(sdk_err)
    }
}

#[pyfunction]
fn _build_pipeline(
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

#[pyfunction]
#[pyo3(signature = (mode, cache_dir=None, model_dir=None, manifest=None, progress=None))]
fn _prepare(
    py: Python<'_>,
    mode: &str,
    cache_dir: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    manifest: Option<PyModelManifest>,
    progress: Option<Py<PyAny>>,
) -> PyResult<PyPreparedModels> {
    let mut options = PrepareModelsOptions::new(mode_from(mode)?);
    options.model_dir = model_dir;
    options.manifest = manifest.map(|manifest| manifest.0);
    let store = cache_dir.map_or_else(ModelStore::default, ModelStore::new);
    let sink = ProgressSink::new(progress, None);
    let result = py.detach(|| {
        let on_download = |file: &str| {
            if sink.is_active() {
                sink.send(ProgressEvent::DownloadingModel { file: file.into() });
            }
        };
        store
            .prepare(options, &on_download)
            .map_err(prepare_sdk_err)
    });
    sink.finish(result).map(PyPreparedModels)
}

/// Hash every file `mode` loads from `model_dir`
#[pyfunction]
#[pyo3(signature = (model_dir, mode="CPU"))]
fn generate_manifest(py: Python<'_>, model_dir: PathBuf, mode: &str) -> PyResult<PyModelManifest> {
    let mode = mode_from(mode)?;
    py.detach(|| ModelManifest::generate(&model_dir, mode))
        .map(PyModelManifest)
        .map_err(manifest_err)
}

/// Files `mode` loads, relative to the model directory
#[pyfunction]
#[pyo3(signature = (mode="CPU"))]
fn required_model_files(mode: &str) -> PyResult<Vec<String>> {
    Ok(speakrs_sdk::required_model_files(mode_from(mode)?))
}

/// The model cache: `$SPEAKRS_CACHE_DIR`, else `$HF_HOME/hub`, else
/// `~/.cache/huggingface/hub`
#[pyfunction]
fn default_cache_dir() -> PathBuf {
    ModelStore::default_cache_dir()
}

/// Cached revisions of the model repository
#[pyfunction]
#[pyo3(signature = (cache_dir=None))]
fn list_cache(cache_dir: Option<PathBuf>) -> PyResult<Vec<PyCacheEntry>> {
    cache_dir
        .map_or_else(ModelStore::default, ModelStore::new)
        .list_cache()
        .map(|entries| entries.into_iter().map(PyCacheEntry).collect())
        .map_err(prepare_err)
}

/// Delete one cached revision and the files no other revision shares
///
/// Returns `False` when the revision was not cached.
#[pyfunction]
#[pyo3(signature = (revision, cache_dir=None))]
fn cleanup_revision(revision: &str, cache_dir: Option<PathBuf>) -> PyResult<bool> {
    cache_dir
        .map_or_else(ModelStore::default, ModelStore::new)
        .cleanup_revision(revision)
        .map_err(prepare_err)
}

/// Execution modes compiled into this build, such as `["CPU", "CoreML", "CoreMLFast"]`
///
/// A CUDA mode can still fail at load time without an NVIDIA driver or a
/// supported GPU.
#[pyfunction]
fn available_modes() -> Vec<&'static str> {
    speakrs_sdk::available_modes()
        .into_iter()
        .map(mode_name)
        .collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "DEFAULT_MODEL_REPOSITORY",
        speakrs_sdk::DEFAULT_MODEL_REPOSITORY,
    )?;
    m.add(
        "DEFAULT_MODEL_REVISION",
        speakrs_sdk::DEFAULT_MODEL_REVISION,
    )?;
    m.add_class::<PyCancelToken>()?;
    m.add_class::<PyActivityCleanup>()?;
    m.add_class::<PyAhcConfig>()?;
    m.add_class::<PyVbxConfig>()?;
    m.add_class::<PyClusteringConfig>()?;
    m.add_class::<PyPipelineConfig>()?;
    m.add_class::<PyRuntimeConfig>()?;
    m.add_class::<PySegment>()?;
    m.add_class::<PyTimingStats>()?;
    m.add_class::<PyDiarizationResult>()?;
    m.add_class::<PyQueueResult>()?;
    m.add_class::<PyProgressEvent>()?;
    m.add_class::<PyModelManifestEntry>()?;
    m.add_class::<PyModelManifest>()?;
    m.add_class::<PyPreparedModels>()?;
    m.add_class::<PyCacheEntry>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyQueue>()?;
    m.add_function(wrap_pyfunction!(_build_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(_prepare, m)?)?;
    m.add_function(wrap_pyfunction!(generate_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(required_model_files, m)?)?;
    m.add_function(wrap_pyfunction!(default_cache_dir, m)?)?;
    m.add_function(wrap_pyfunction!(list_cache, m)?)?;
    m.add_function(wrap_pyfunction!(cleanup_revision, m)?)?;
    m.add_function(wrap_pyfunction!(available_modes, m)?)?;
    Ok(())
}

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn py_opt_str(value: Option<&str>) -> String {
    value.map_or_else(|| "None".to_string(), |value| format!("'{value}'"))
}
