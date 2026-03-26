use std::path::Path;

use ort::session::Session;

use crate::inference::with_execution_mode;

use super::{EmbeddingModel, ExecutionMode};

impl EmbeddingModel {
    pub(super) fn build_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session_with_graph(model_path, mode, false)
    }

    pub(super) fn build_session_with_graph(
        model_path: &Path,
        mode: ExecutionMode,
        cuda_graph: bool,
    ) -> Result<Session, ort::Error> {
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder =
            if cuda_graph && matches!(mode, ExecutionMode::Cuda | ExecutionMode::CudaFast) {
                Self::with_cuda_graph_mode(builder)?
            } else {
                with_execution_mode(builder, mode)?
            };
        builder.commit_from_file(model_path)
    }

    #[cfg(feature = "cuda")]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        use ort::ep;

        Ok(builder.with_execution_providers([ep::CUDA::default()
            .with_device_id(0)
            .with_tf32(true)
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
            .with_conv_max_workspace(true)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .with_prefer_nhwc(true)
            .with_cuda_graph(true)
            .build()
            .error_on_failure()])?)
    }

    #[cfg(not(feature = "cuda"))]
    fn with_cuda_graph_mode(
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder, ort::Error> {
        with_execution_mode(builder, ExecutionMode::Cpu)
    }

    pub(super) fn build_fbank_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        let threads = std::thread::available_parallelism()
            .map(|count| count.get().min(4))
            .unwrap_or(1);
        let builder = Session::builder()?
            .with_independent_thread_pool()?
            .with_intra_threads(threads)?
            .with_inter_threads(1)?
            .with_memory_pattern(true)?;
        let mut builder = with_execution_mode(builder, mode)?;
        builder.commit_from_file(model_path)
    }

    pub(super) fn single_execution_mode(mode: ExecutionMode) -> ExecutionMode {
        match mode {
            ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => ExecutionMode::Cpu,
            _ => mode,
        }
    }

    pub(super) fn build_batched_session(
        model_path: &Path,
        mode: ExecutionMode,
    ) -> Result<Session, ort::Error> {
        Self::build_session(model_path, Self::single_execution_mode(mode))
    }
}
