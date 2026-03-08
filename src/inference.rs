pub mod embedding;
pub mod segmentation;

#[cfg(feature = "native-coreml")]
pub(crate) mod coreml;

use ort::ep;
use ort::session::builder::SessionBuilder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    ExactCpu,
    CoreMl,     // FP32 native CoreML, CPU+GPU, exact parity with pyannote
    MiniCoreMl, // FP16 native CoreML, CPU+GPU+ANE, faster but may drift
    Cuda,
}

/// Map execution mode to ORT execution providers
///
/// CoreMl and MiniCoreMl use ORT CPU for any sessions that still go through ORT (e.g. FBANK),
/// while segmentation/embedding tail sessions use native CoreML directly.
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    match mode {
        ExecutionMode::ExactCpu | ExecutionMode::CoreMl | ExecutionMode::MiniCoreMl => Ok(builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?),
        ExecutionMode::Cuda => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder.with_execution_providers([ep::CUDA::default()
                    .with_device_id(0)
                    .with_tf32(false)
                    .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Default)
                    .with_conv_max_workspace(false)
                    .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                    .build()])?)
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
