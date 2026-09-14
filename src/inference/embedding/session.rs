use std::path::Path;

use ort::session::Session;
use ort::value::TensorElementType;

use crate::inference::with_execution_mode;

use super::{
    EmbeddingGeometryError, EmbeddingInputGeometry, EmbeddingModel, ExecutionMode,
    PrimaryTensorShape, geometry_from_primary_shapes,
};

impl EmbeddingModel {
    pub(crate) fn validate_primary_session(
        session: &Session,
    ) -> Result<EmbeddingInputGeometry, EmbeddingGeometryError> {
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|input| input.name().to_owned())
            .collect();
        if input_names.len() != 2
            || !input_names.iter().any(|name| name == "waveform")
            || !input_names.iter().any(|name| name == "weights")
        {
            return Err(EmbeddingGeometryError::UnexpectedInputs {
                actual: input_names,
            });
        }

        let waveform = primary_tensor_shape(session, "waveform")?;
        let weights = primary_tensor_shape(session, "weights")?;
        let output_count = session.outputs().len();
        if output_count != 1 {
            return Err(EmbeddingGeometryError::UnexpectedOutputs {
                count: output_count,
            });
        }
        let output = session.outputs()[0].dtype();
        let output_shape =
            output
                .tensor_shape()
                .ok_or_else(|| EmbeddingGeometryError::NotATensor {
                    tensor: session.outputs()[0].name().to_owned(),
                })?;
        if output.tensor_type() != Some(TensorElementType::Float32) {
            return Err(EmbeddingGeometryError::TypeMismatch {
                tensor: session.outputs()[0].name().to_owned(),
                actual: output
                    .tensor_type()
                    .map_or_else(|| output.to_string(), |value| value.to_string()),
            });
        }
        let output =
            PrimaryTensorShape::new(session.outputs()[0].name(), output_shape.iter().copied());

        geometry_from_primary_shapes(&waveform, &weights, &output)
    }

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
            // Embedding inference dominates CPU-mode wall time. With
            // intra_threads(1) the whole pipeline runs at ~1-2x realtime on
            // Apple Silicon; letting ORT use up to 6 cores brings it to
            // ~8-9x realtime (measured on M-series, 5.7 min meeting audio)
            // with identical outputs.
            .with_intra_threads(
                std::thread::available_parallelism()
                    .map(|n| n.get().min(6))
                    .unwrap_or(1),
            )?
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

fn primary_tensor_shape(
    session: &Session,
    name: &'static str,
) -> Result<PrimaryTensorShape, EmbeddingGeometryError> {
    let outlet = session
        .inputs()
        .iter()
        .find(|input| input.name() == name)
        .ok_or(EmbeddingGeometryError::MissingInput { name })?;
    let dtype = outlet.dtype();
    if dtype.tensor_type() != Some(TensorElementType::Float32) {
        return Err(EmbeddingGeometryError::TypeMismatch {
            tensor: name.to_owned(),
            actual: dtype
                .tensor_type()
                .map_or_else(|| dtype.to_string(), |value| value.to_string()),
        });
    }
    let shape = dtype
        .tensor_shape()
        .ok_or_else(|| EmbeddingGeometryError::NotATensor {
            tensor: name.to_owned(),
        })?;
    Ok(PrimaryTensorShape::new(name, shape.iter().copied()))
}
