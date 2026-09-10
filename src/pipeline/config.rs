#[cfg(any(feature = "coreml", feature = "_metrics"))]
use crate::inference::CoreMlComputeUnits;
use crate::inference::ExecutionMode;
#[cfg(feature = "_metrics")]
use crate::pipeline::SphereVbxPfConfig;
use crate::pipeline::{AhcConfig, BinarizeConfig, VbxConfig};

/// Speaker clustering model and its valid configuration
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ClusteringBackend {
    /// PLDA-transformed Gaussian variational Bayes clustering
    GaussianVbx(VbxConfig),
    /// Parameter-free spherical variational Bayes clustering
    SphereVbxPf(SphereVbxPfConfig),
}

/// How to map cluster assignments back to per-frame speaker activations
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ReconstructMethod {
    /// Standard top-K selection (pyannote-compatible)
    Standard,
    /// Temporal smoothing. If scores are within epsilon, keep the previous speaker.
    Smoothed {
        /// Score difference below which the previous frame's speaker is preferred
        epsilon: f32,
    },
}

/// Tunable parameters for the diarization pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Hysteresis binarization and min-duration filtering
    pub binarize: BinarizeConfig,
    /// Agglomerative hierarchical clustering settings
    pub ahc: AhcConfig,
    /// Variational Bayes HMM clustering settings
    pub vbx: VbxConfig,
    /// Optional clustering backend used by the internal experiment harness
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    #[doc(hidden)]
    pub experimental_clustering: Option<ClusteringBackend>,
    /// Minimum single-speaker activity used to select clustering embeddings in experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub clean_frame_duration: CleanFrameDuration,
    /// Maximum gap in seconds between segments to merge into one
    pub merge_gap: f64,
    /// Minimum speaker activity weight to keep a speaker in output
    pub speaker_keep_threshold: f64,
    /// Strategy for mapping clusters back to frame activations
    pub reconstruct_method: ReconstructMethod,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            binarize: BinarizeConfig::default(),
            ahc: AhcConfig::default(),
            vbx: VbxConfig::default(),
            #[cfg(feature = "_metrics")]
            experimental_clustering: None,
            #[cfg(feature = "_metrics")]
            clean_frame_duration: CleanFrameDuration::default(),
            merge_gap: 0.0,
            speaker_keep_threshold: 1e-7,
            reconstruct_method: ReconstructMethod::Smoothed { epsilon: 0.1 },
        }
    }
}

/// Minimum single-speaker activity required for a clustering embedding
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct CleanFrameDuration(f64);

impl CleanFrameDuration {
    /// Create a positive finite duration in seconds
    pub fn new(seconds: f64) -> Result<Self, CleanFrameDurationError> {
        if seconds.is_finite() && seconds > 0.0 {
            Ok(Self(seconds))
        } else {
            Err(CleanFrameDurationError(seconds))
        }
    }

    /// Return the duration in seconds
    pub const fn seconds(self) -> f64 {
        self.0
    }

    pub(crate) fn minimum_frames(self) -> f32 {
        (self.0 / FRAME_STEP_SECONDS).floor() as f32
    }
}

impl Default for CleanFrameDuration {
    fn default() -> Self {
        Self(2.0)
    }
}

/// Error returned for a non-positive or non-finite clean-frame duration
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CleanFrameDurationError(f64);

impl std::fmt::Display for CleanFrameDurationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "clean-frame duration must be finite and greater than zero, got {}",
            self.0
        )
    }
}

impl std::error::Error for CleanFrameDurationError {}

impl PipelineConfig {
    pub(crate) fn effective_clean_frame_duration(&self) -> CleanFrameDuration {
        #[cfg(feature = "_metrics")]
        {
            self.clean_frame_duration
        }
        #[cfg(not(feature = "_metrics"))]
        {
            CleanFrameDuration::default()
        }
    }

    /// Mode-specific defaults. Fast modes use min-duration filtering to remove
    /// single-frame speaker flicker from the larger step size.
    pub fn for_mode(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => Self {
                binarize: BinarizeConfig {
                    min_duration_on: 3,
                    min_duration_off: 3,
                    ..BinarizeConfig::default()
                },
                // fast modes use 3 VBx iterations to avoid posterior overfitting
                // on 2 second step embeddings
                vbx: VbxConfig {
                    max_iters: 3,
                    ..VbxConfig::default()
                },
                ..Self::default()
            },
            _ => Self::default(),
        }
    }

    /// Select a typed clustering backend for metrics experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub fn with_experimental_clustering(mut self, backend: ClusteringBackend) -> Self {
        match backend {
            ClusteringBackend::GaussianVbx(vbx) => {
                self.vbx = vbx;
                self.experimental_clustering = None;
            }
            ClusteringBackend::SphereVbxPf(_) => {
                self.experimental_clustering = Some(backend);
            }
        }
        self
    }

    /// Return the effective clustering backend for metrics diagnostics
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub fn clustering_backend(&self) -> ClusteringBackend {
        self.experimental_clustering
            .unwrap_or(ClusteringBackend::GaussianVbx(self.vbx))
    }
}

/// CoreML chunk or per-window inference layout used by metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlChunkLayout {
    /// Two phased chunk models with an exact 1 second window step
    OneSecondPhased,
    /// One-pass chunk models aligned to 25 ResNet frames at a 2 second step
    FastS25,
    /// Per-window embedding with an exact 1 second window step
    PerWindow,
}

/// Fixed-shape CoreML chunk-model set used by metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CoreMlShapeLadder {
    /// Use every production fixed-shape model
    #[default]
    Full,
    /// Keep the 21, 51, and 111-window one-second models
    Reduced,
}

/// Segmentation worker count used by metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CoreMlSegmentationWorkers {
    /// Use the production host-derived worker count
    #[default]
    Automatic,
    /// Use four workers
    Four,
    /// Use six workers
    Six,
    /// Use eight workers
    Eight,
}

#[cfg(feature = "_metrics")]
impl CoreMlSegmentationWorkers {
    #[cfg(feature = "coreml")]
    pub(crate) fn resolve(self) -> usize {
        match self {
            Self::Automatic => default_coreml_segmentation_worker_count(),
            Self::Four => 4,
            Self::Six => 6,
            Self::Eight => 8,
        }
    }
}

/// Filterbank preparation worker count used by metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CoreMlFbankPreparationWorkers {
    /// Use one worker
    One,
    /// Use the production count of two workers
    #[default]
    Two,
    /// Use four workers
    Four,
}

/// Filterbank normalization scope used by metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CoreMlFbankNormalizationScope {
    /// Normalize all samples in each selected fixed-shape chunk together
    #[default]
    Chunk,
    /// Normalize independent 10-second filterbank segments before stitching them
    TenSecondSegments,
}

#[cfg(feature = "_metrics")]
impl CoreMlFbankPreparationWorkers {
    #[cfg(feature = "coreml")]
    pub(crate) const fn get(self) -> usize {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four => 4,
        }
    }
}

#[cfg(feature = "_metrics")]
impl CoreMlChunkLayout {
    /// Segmentation step in seconds required by this layout
    pub const fn step_seconds(self) -> f64 {
        match self {
            Self::OneSecondPhased | Self::PerWindow => 1.0,
            Self::FastS25 => 2.0,
        }
    }

    /// Whether this layout uses the native full-audio chunk session ladder
    pub const fn uses_native_chunk_sessions(self) -> bool {
        !matches!(self, Self::PerWindow)
    }

    /// Validate that this layout belongs to the requested execution mode
    pub const fn supports_mode(self, mode: ExecutionMode) -> bool {
        matches!(
            (mode, self),
            (
                ExecutionMode::CoreMl,
                Self::OneSecondPhased | Self::PerWindow
            ) | (ExecutionMode::CoreMlFast, Self::FastS25)
        )
    }
}

/// Typed inference configuration for metrics experiments
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExperimentInferenceConfig {
    /// CoreML chunk or per-window layout and its fixed step
    pub coreml_chunk_layout: CoreMlChunkLayout,
    /// Fixed-shape chunk-model ladder
    pub shape_ladder: CoreMlShapeLadder,
    /// Parallel segmentation worker policy
    pub segmentation_workers: CoreMlSegmentationWorkers,
    /// Parallel filterbank preparation worker policy
    pub fbank_preparation_workers: CoreMlFbankPreparationWorkers,
    /// Filterbank normalization scope
    pub fbank_normalization_scope: CoreMlFbankNormalizationScope,
    /// Compute units for native embedding models; filterbank preparation has separate placement
    pub embedding_compute_units: CoreMlComputeUnits,
}

#[cfg(feature = "_metrics")]
impl ExperimentInferenceConfig {
    /// Create an experiment configuration for a fixed CoreML layout
    pub const fn new(coreml_chunk_layout: CoreMlChunkLayout) -> Self {
        Self {
            coreml_chunk_layout,
            shape_ladder: CoreMlShapeLadder::Full,
            segmentation_workers: CoreMlSegmentationWorkers::Automatic,
            fbank_preparation_workers: CoreMlFbankPreparationWorkers::Two,
            fbank_normalization_scope: CoreMlFbankNormalizationScope::Chunk,
            embedding_compute_units: CoreMlComputeUnits::All,
        }
    }

    /// Create an experiment configuration with an explicit execution policy
    pub const fn with_execution_policy(
        coreml_chunk_layout: CoreMlChunkLayout,
        shape_ladder: CoreMlShapeLadder,
        segmentation_workers: CoreMlSegmentationWorkers,
        fbank_preparation_workers: CoreMlFbankPreparationWorkers,
    ) -> Self {
        Self {
            coreml_chunk_layout,
            shape_ladder,
            segmentation_workers,
            fbank_preparation_workers,
            fbank_normalization_scope: CoreMlFbankNormalizationScope::Chunk,
            embedding_compute_units: CoreMlComputeUnits::All,
        }
    }

    /// Select the filterbank normalization scope for a controlled comparison
    pub const fn with_fbank_normalization_scope(
        mut self,
        scope: CoreMlFbankNormalizationScope,
    ) -> Self {
        self.fbank_normalization_scope = scope;
        self
    }

    /// Select compute units for native embedding models
    pub const fn with_embedding_compute_units(mut self, units: CoreMlComputeUnits) -> Self {
        self.embedding_compute_units = units;
        self
    }

    /// Return the fixed segmentation step required by this layout
    pub const fn step_seconds(self) -> f64 {
        self.coreml_chunk_layout.step_seconds()
    }

    /// Return the fixed segmentation step required by this layout
    pub const fn segmentation_step_seconds(self) -> f64 {
        self.step_seconds()
    }

    /// Validate this configuration for an execution mode
    pub const fn validate(self, mode: ExecutionMode) -> Result<(), ExperimentInferenceConfigError> {
        if !self.coreml_chunk_layout.supports_mode(mode) {
            return Err(ExperimentInferenceConfigError::IncompatibleMode {
                mode,
                layout: self.coreml_chunk_layout,
            });
        }

        if matches!(self.shape_ladder, CoreMlShapeLadder::Reduced)
            && !matches!(self.coreml_chunk_layout, CoreMlChunkLayout::OneSecondPhased)
        {
            return Err(ExperimentInferenceConfigError::IncompatibleShapeLadder {
                layout: self.coreml_chunk_layout,
                shape_ladder: self.shape_ladder,
            });
        }

        Ok(())
    }

    /// Validate this configuration for an execution mode and segmentation step
    pub fn validate_for_step(
        self,
        mode: ExecutionMode,
        step_seconds: f64,
    ) -> Result<(), ExperimentInferenceConfigError> {
        self.validate(mode)?;

        let expected = self.step_seconds();
        if !step_seconds.is_finite() || (step_seconds - expected).abs() > 1e-9 {
            return Err(ExperimentInferenceConfigError::IncompatibleStep {
                layout: self.coreml_chunk_layout,
                expected,
                actual: step_seconds,
            });
        }

        Ok(())
    }
}

#[cfg(feature = "_metrics")]
impl Default for ExperimentInferenceConfig {
    fn default() -> Self {
        Self::new(CoreMlChunkLayout::OneSecondPhased)
    }
}

/// Error returned when a metrics inference configuration is incompatible
#[cfg(feature = "_metrics")]
#[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ExperimentInferenceConfigError {
    /// The layout is not supported by the requested execution mode
    IncompatibleMode {
        /// Requested execution mode
        mode: ExecutionMode,
        /// Layout selected for the experiment
        layout: CoreMlChunkLayout,
    },
    /// The step does not match the selected fixed layout
    IncompatibleStep {
        /// Layout selected for the experiment
        layout: CoreMlChunkLayout,
        /// Required step in seconds
        expected: f64,
        /// Requested step in seconds
        actual: f64,
    },
    /// The shape ladder is not defined for the selected layout
    IncompatibleShapeLadder {
        /// Selected inference layout
        layout: CoreMlChunkLayout,
        /// Selected shape ladder
        shape_ladder: CoreMlShapeLadder,
    },
}

#[cfg(feature = "_metrics")]
impl std::fmt::Display for ExperimentInferenceConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompatibleMode { mode, layout } => {
                write!(
                    formatter,
                    "CoreML experiment layout {layout:?} is incompatible with execution mode {mode}"
                )
            }
            Self::IncompatibleStep {
                layout,
                expected,
                actual,
            } => write!(
                formatter,
                "CoreML experiment layout {layout:?} requires a {expected} second step, got {actual}"
            ),
            Self::IncompatibleShapeLadder {
                layout,
                shape_ladder,
            } => write!(
                formatter,
                "CoreML experiment shape ladder {shape_ladder:?} is incompatible with layout {layout:?}"
            ),
        }
    }
}

#[cfg(feature = "_metrics")]
impl std::error::Error for ExperimentInferenceConfigError {}

/// Runtime configuration for the diarization pipeline
///
/// Controls execution parameters that can affect numerical output and performance.
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    /// CoreML compute units for native embedding models (CoreML modes only)
    #[cfg(feature = "coreml")]
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    pub chunk_emb_compute_units: CoreMlComputeUnits,
    /// Optional typed inference layout for metrics experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub experiment: Option<ExperimentInferenceConfig>,
}

impl RuntimeConfig {
    /// Set a typed inference layout for metrics experiments
    #[cfg(feature = "_metrics")]
    #[cfg_attr(docsrs, doc(cfg(feature = "_metrics")))]
    pub fn with_experiment(mut self, experiment: ExperimentInferenceConfig) -> Self {
        self.experiment = Some(experiment);
        self
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn coreml_embedding_compute_units(&self) -> CoreMlComputeUnits {
        #[cfg(feature = "_metrics")]
        if let Some(experiment) = self.experiment {
            return experiment.embedding_compute_units;
        }

        self.chunk_emb_compute_units
    }

    #[cfg(feature = "coreml")]
    pub(crate) fn coreml_chunk_execution_policy(&self) -> CoreMlChunkExecutionPolicy {
        #[cfg(feature = "_metrics")]
        if let Some(experiment) = self.experiment {
            return CoreMlChunkExecutionPolicy {
                segmentation_workers: experiment.segmentation_workers.resolve(),
                fbank_preparation_workers: experiment.fbank_preparation_workers.get(),
                fbank_normalization_scope: match experiment.fbank_normalization_scope {
                    CoreMlFbankNormalizationScope::Chunk => ChunkFbankNormalizationScope::Chunk,
                    CoreMlFbankNormalizationScope::TenSecondSegments => {
                        ChunkFbankNormalizationScope::TenSecondSegments
                    }
                },
            };
        }

        CoreMlChunkExecutionPolicy::default()
    }
}

#[derive(Clone, Copy)]
#[cfg(feature = "coreml")]
pub(crate) struct CoreMlChunkExecutionPolicy {
    pub(crate) segmentation_workers: usize,
    pub(crate) fbank_preparation_workers: usize,
    pub(crate) fbank_normalization_scope: ChunkFbankNormalizationScope,
}

#[derive(Clone, Copy, Eq, PartialEq)]
#[cfg(feature = "coreml")]
pub(crate) enum ChunkFbankNormalizationScope {
    Chunk,
    #[cfg(feature = "_metrics")]
    TenSecondSegments,
}

#[cfg(feature = "coreml")]
impl ChunkFbankNormalizationScope {
    pub(crate) const fn uses_chunk_scope(self) -> bool {
        match self {
            Self::Chunk => true,
            #[cfg(feature = "_metrics")]
            Self::TenSecondSegments => false,
        }
    }
}

#[cfg(feature = "coreml")]
impl Default for CoreMlChunkExecutionPolicy {
    fn default() -> Self {
        Self {
            segmentation_workers: default_coreml_segmentation_worker_count(),
            fbank_preparation_workers: 2,
            fbank_normalization_scope: ChunkFbankNormalizationScope::Chunk,
        }
    }
}

#[cfg(feature = "coreml")]
fn default_coreml_segmentation_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4)
        .min(8)
}

/// Segmentation step size in seconds for the selected execution mode
pub const fn segmentation_step_seconds(mode: ExecutionMode) -> f64 {
    match mode {
        ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => FAST_SEGMENTATION_STEP_SECONDS,
        ExecutionMode::CoreMl => COREML_SEGMENTATION_STEP_SECONDS,
        ExecutionMode::Cuda => CUDA_SEGMENTATION_STEP_SECONDS,
        ExecutionMode::MiGraphX => CUDA_SEGMENTATION_STEP_SECONDS,
        ExecutionMode::Cpu => SEGMENTATION_STEP_SECONDS,
    }
}

/// Sliding window length for segmentation model input, in seconds
pub const SEGMENTATION_WINDOW_SECONDS: f64 = 10.0;
/// Default sliding window step for segmentation, in seconds
pub const SEGMENTATION_STEP_SECONDS: f64 = 1.0;
/// CoreML step in seconds
///
/// The chunk embedding model uses two aligned phases to support this exact step
pub const COREML_SEGMENTATION_STEP_SECONDS: f64 = 1.0;
/// CUDA segmentation step, in seconds
pub const CUDA_SEGMENTATION_STEP_SECONDS: f64 = 1.0;
/// Step size for fast modes, in seconds
pub const FAST_SEGMENTATION_STEP_SECONDS: f64 = 2.0;
/// Duration of each output frame from the segmentation model, in seconds
pub const FRAME_DURATION_SECONDS: f64 = 0.0619375;
/// Hop between consecutive output frames from the segmentation model, in seconds
pub const FRAME_STEP_SECONDS: f64 = 0.016875;

/// Minimum speaker activity (sum of weights) to run embedding inference.
/// Speakers below this threshold are skipped because their NaN embedding is filtered out later
pub(crate) const MIN_SPEAKER_ACTIVITY: f32 = 10.0;

#[cfg(test)]
mod clean_frame_duration_tests {
    use super::*;

    #[test]
    fn pipeline_defaults_keep_gaussian_vbx_and_fixed_mode_steps() {
        let standard_vbx = PipelineConfig::default().vbx;
        let fast_vbx = PipelineConfig::for_mode(ExecutionMode::CoreMlFast).vbx;

        assert_eq!(standard_vbx.max_iters, 20);
        assert_eq!(fast_vbx.max_iters, 3);
        assert_eq!(segmentation_step_seconds(ExecutionMode::CoreMl), 1.0);
        assert_eq!(segmentation_step_seconds(ExecutionMode::CoreMlFast), 2.0);
    }

    #[test]
    fn default_clean_frame_duration_preserves_the_previous_frame_threshold() {
        assert_eq!(CleanFrameDuration::default().minimum_frames(), 118.0);
    }

    #[test]
    fn clean_frame_duration_rejects_invalid_values() {
        for seconds in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(CleanFrameDuration::new(seconds).is_err());
        }
    }
}

#[cfg(all(test, feature = "_metrics"))]
mod tests {
    use super::*;

    #[test]
    fn runtime_defaults_to_mode_owned_inference() {
        assert_eq!(RuntimeConfig::default().experiment, None);
    }

    #[test]
    fn experiment_layouts_have_fixed_steps_and_modes() {
        let cases = [
            (
                CoreMlChunkLayout::OneSecondPhased,
                ExecutionMode::CoreMl,
                1.0,
            ),
            (CoreMlChunkLayout::FastS25, ExecutionMode::CoreMlFast, 2.0),
            (CoreMlChunkLayout::PerWindow, ExecutionMode::CoreMl, 1.0),
        ];

        for (layout, mode, expected_step) in cases {
            let config = ExperimentInferenceConfig::new(layout);
            assert_eq!(config.step_seconds(), expected_step);
            assert!(config.validate(mode).is_ok());
        }
    }

    #[test]
    fn experiment_execution_policy_preserves_production_defaults() {
        let config = ExperimentInferenceConfig::new(CoreMlChunkLayout::OneSecondPhased);

        assert_eq!(config.shape_ladder, CoreMlShapeLadder::Full);
        assert_eq!(
            config.segmentation_workers,
            CoreMlSegmentationWorkers::Automatic
        );
        assert_eq!(
            config.fbank_preparation_workers,
            CoreMlFbankPreparationWorkers::Two
        );
        assert_eq!(
            config.fbank_normalization_scope,
            CoreMlFbankNormalizationScope::Chunk
        );
        assert_eq!(config.embedding_compute_units, CoreMlComputeUnits::All);

        #[cfg(feature = "coreml")]
        {
            let resolved = RuntimeConfig::default()
                .with_experiment(config)
                .coreml_chunk_execution_policy();
            assert_eq!(resolved.fbank_preparation_workers, 2);
            assert!(resolved.segmentation_workers > 0);
            assert!(matches!(
                resolved.fbank_normalization_scope,
                ChunkFbankNormalizationScope::Chunk
            ));
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn experiment_embedding_compute_units_override_runtime_fallback() {
        let experiment = ExperimentInferenceConfig::new(CoreMlChunkLayout::PerWindow)
            .with_embedding_compute_units(CoreMlComputeUnits::CpuOnly);
        let runtime = RuntimeConfig {
            chunk_emb_compute_units: CoreMlComputeUnits::CpuAndNeuralEngine,
            experiment: Some(experiment),
        };

        assert_eq!(
            runtime.coreml_embedding_compute_units(),
            CoreMlComputeUnits::CpuOnly
        );

        let runtime = RuntimeConfig {
            chunk_emb_compute_units: CoreMlComputeUnits::CpuAndNeuralEngine,
            experiment: None,
        };
        assert_eq!(
            runtime.coreml_embedding_compute_units(),
            CoreMlComputeUnits::CpuAndNeuralEngine
        );
    }

    #[test]
    fn reduced_shape_ladder_rejects_unsupported_layouts() {
        let config = ExperimentInferenceConfig::with_execution_policy(
            CoreMlChunkLayout::FastS25,
            CoreMlShapeLadder::Reduced,
            CoreMlSegmentationWorkers::Four,
            CoreMlFbankPreparationWorkers::One,
        );

        assert!(matches!(
            config.validate(ExecutionMode::CoreMlFast),
            Err(ExperimentInferenceConfigError::IncompatibleShapeLadder { .. })
        ));
    }

    #[test]
    fn experiment_validation_rejects_incompatible_mode_and_step() {
        let fast = ExperimentInferenceConfig::new(CoreMlChunkLayout::FastS25);
        assert!(matches!(
            fast.validate(ExecutionMode::CoreMl),
            Err(ExperimentInferenceConfigError::IncompatibleMode { .. })
        ));

        let aligned = ExperimentInferenceConfig::new(CoreMlChunkLayout::OneSecondPhased);
        assert!(matches!(
            aligned.validate_for_step(ExecutionMode::CoreMl, 1.04),
            Err(ExperimentInferenceConfigError::IncompatibleStep { .. })
        ));
    }

    #[test]
    fn per_window_layouts_disable_native_chunk_sessions() {
        assert!(!CoreMlChunkLayout::PerWindow.uses_native_chunk_sessions());
        assert!(CoreMlChunkLayout::OneSecondPhased.uses_native_chunk_sessions());
    }
}
