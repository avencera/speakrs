use crate::binarize::BinarizeConfig;
use crate::clustering::ahc::AhcConfig;
use crate::clustering::vbx::VbxConfig;
#[cfg(feature = "coreml")]
use crate::inference::CoreMlComputeUnits;
use crate::inference::ExecutionMode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconstructMethod {
    /// Standard top-K selection (pyannote-compatible)
    Standard,
    /// Temporal smoothing — when scores are within epsilon, prefer previous frame's speaker
    Smoothed { epsilon: f32 },
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub binarize: BinarizeConfig,
    pub ahc: AhcConfig,
    pub vbx: VbxConfig,
    pub merge_gap: f64,
    pub speaker_keep_threshold: f64,
    pub reconstruct_method: ReconstructMethod,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            binarize: BinarizeConfig::default(),
            ahc: AhcConfig::default(),
            vbx: VbxConfig::default(),
            merge_gap: 0.0,
            speaker_keep_threshold: 1e-7,
            reconstruct_method: ReconstructMethod::Smoothed { epsilon: 0.1 },
        }
    }
}

impl PipelineConfig {
    /// Mode-specific defaults. CoreMLFast uses min-duration filtering to remove
    /// single-frame speaker flickers caused by the larger step size
    pub fn for_mode(mode: ExecutionMode) -> Self {
        match mode {
            ExecutionMode::CoreMlFast | ExecutionMode::CudaFast => Self {
                binarize: BinarizeConfig {
                    min_duration_on: 3,
                    min_duration_off: 3,
                    ..BinarizeConfig::default()
                },
                // fast modes: 3 VBx iters avoids posterior over-fitting,
                // improves DER on 2s step embeddings
                vbx: VbxConfig {
                    max_iters: 3,
                    ..VbxConfig::default()
                },
                ..Self::default()
            },
            _ => Self::default(),
        }
    }
}

/// Runtime configuration for the diarization pipeline
///
/// Controls execution parameters that don't affect correctness
/// but influence performance characteristics
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of parallel chunk embedding workers (default: 1)
    pub chunk_emb_workers: usize,
    /// CoreML compute units for chunk embedding (CoreML modes only)
    #[cfg(feature = "coreml")]
    pub chunk_emb_compute_units: CoreMlComputeUnits,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            chunk_emb_workers: 1,
            #[cfg(feature = "coreml")]
            chunk_emb_compute_units: CoreMlComputeUnits::All,
        }
    }
}

pub const SEGMENTATION_WINDOW_SECONDS: f64 = 10.0;
pub const SEGMENTATION_STEP_SECONDS: f64 = 1.0;
// aligned to 8-frame ResNet stride: 96 fbank frames / 8 = 12 ResNet frames
// closest aligned step below 1.0s, enables chunk embedding
pub const COREML_SEGMENTATION_STEP_SECONDS: f64 = 0.96;
pub const CUDA_SEGMENTATION_STEP_SECONDS: f64 = 1.0;
pub const FAST_SEGMENTATION_STEP_SECONDS: f64 = 2.0;
pub const FRAME_DURATION_SECONDS: f64 = 0.0619375;
pub const FRAME_STEP_SECONDS: f64 = 0.016875;

/// Minimum speaker activity (sum of weights) to run embedding inference.
/// Speakers below this threshold are skipped — their NaN embedding is filtered out later
pub(crate) const MIN_SPEAKER_ACTIVITY: f32 = 10.0;
