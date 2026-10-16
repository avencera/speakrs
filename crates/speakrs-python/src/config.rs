//! Configuration value objects

use pyo3::prelude::*;
use speakrs_sdk::speakrs::pipeline::{
    ActivityCleanup, AhcConfig, ClusteringBackend, ClusteringConfig, PipelineConfig,
    ReconstructMethod, ResponsibilityInitialization, VbxConfig,
};
use speakrs_sdk::{RuntimeOptions, SdkErrorCategory};

use crate::{category_err, config_err, mode_from, py_bool, py_opt_str, sdk_err};

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

/// Frame-count cleanup applied to each speaker's activity after clustering
///
/// Frames are 16.875 ms apart. All zeros, the default for full-precision modes,
/// leaves activity unchanged.
#[pyclass(name = "ActivityCleanup", module = "speakrs", frozen, eq)]
#[derive(Clone, PartialEq)]
pub(crate) struct PyActivityCleanup(pub(crate) ActivityCleanup);

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
pub(crate) struct PyAhcConfig(pub(crate) AhcConfig);

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
pub(crate) struct PyVbxConfig(pub(crate) VbxConfig);

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
pub(crate) struct PyClusteringConfig(pub(crate) ClusteringConfig);

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
pub(crate) struct PyPipelineConfig(pub(crate) PipelineConfig);

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
            ("smoothed", epsilon) => {
                let epsilon = epsilon.unwrap_or(default_epsilon);
                // the core keeps the previous speaker while scores differ by less than
                // epsilon, so NaN or a negative value would silently disable smoothing
                // and infinity would never switch speakers
                if !(epsilon.is_finite() && epsilon >= 0.0) {
                    return Err(config_err(format!(
                        "smoothing_epsilon must be finite and non-negative, got {epsilon}"
                    )));
                }
                ReconstructMethod::Smoothed { epsilon }
            }
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
pub(crate) struct PyRuntimeConfig(pub(crate) RuntimeOptions);

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
