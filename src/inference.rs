pub mod embedding;
pub mod segmentation;

use ort::ep;
use ort::session::builder::SessionBuilder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    ExactCpu,
    CoreMl,
    Cuda,
}

pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    match mode {
        ExecutionMode::ExactCpu => Ok(builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?),
        ExecutionMode::CoreMl => {
            #[cfg(feature = "coreml")]
            {
                let cache_dir = std::env::temp_dir().join("speakrs-coreml-cache");
                let _ = std::fs::create_dir_all(&cache_dir);
                Ok(builder.with_execution_providers([ep::CoreML::default()
                    .with_static_input_shapes(true)
                    .with_model_cache_dir(cache_dir.display().to_string())
                    .build()])?)
            }

            #[cfg(not(feature = "coreml"))]
            {
                Ok(builder.with_execution_providers([ep::CPU::default()
                    .with_arena_allocator(false)
                    .build()])?)
            }
        }
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
