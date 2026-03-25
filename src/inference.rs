pub(crate) mod embedding;
pub(crate) mod segmentation;

pub use embedding::EmbeddingModel;
pub use segmentation::{SegmentationError, SegmentationModel};

#[cfg(feature = "coreml")]
pub(crate) mod coreml;

use ort::ep;
use ort::session::builder::SessionBuilder;

/// CoreML compute unit selection for chunk embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMlComputeUnits {
    /// Use all available compute units: CPU + GPU + Neural Engine (default)
    #[default]
    All,
    /// Use CPU + Neural Engine only (skip GPU)
    CpuAndNeuralEngine,
}

#[cfg(feature = "coreml")]
impl CoreMlComputeUnits {
    pub(crate) fn to_ml_compute_units(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::All => crate::inference::coreml::CoreMlModel::default_compute_units(),
            Self::CpuAndNeuralEngine => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
        }
    }
}

/// Which backend and acceleration to use for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// CPU-only via ORT (portable, slowest)
    Cpu,
    /// Native CoreML with FP32 precision and ~1s step
    CoreMl,
    /// Native CoreML with W8A16 segmentation and ~2s step
    CoreMlFast,
    /// NVIDIA GPU with concurrent fused seg+emb via crossbeam
    Cuda,
    /// NVIDIA GPU with concurrent fused seg+emb and ~2s step
    CudaFast,
}

/// Map execution mode to ORT execution providers
///
/// CoreMl modes use ORT CPU for any sessions that still go through ORT (e.g. FBANK),
/// while segmentation/embedding tail sessions use native CoreML directly
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    match mode {
        ExecutionMode::Cpu | ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => Ok(builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?),
        ExecutionMode::Cuda | ExecutionMode::CudaFast => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder.with_execution_providers([ep::CUDA::default()
                    .with_device_id(0)
                    .with_tf32(true)
                    .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                    .with_conv_max_workspace(true)
                    .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                    .with_prefer_nhwc(true)
                    .build()
                    .error_on_failure()])?)
            }

            #[cfg(not(feature = "cuda"))]
            {
                Ok(builder.with_execution_providers([ep::CPU::default()
                    .with_arena_allocator(false)
                    .build()])?)
            }
        }
    }
}
