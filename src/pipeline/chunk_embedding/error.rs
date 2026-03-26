use crate::pipeline::PipelineError;

pub(super) fn backend_error(context: &'static str, message: impl ToString) -> PipelineError {
    PipelineError::Backend {
        context,
        message: message.to_string(),
    }
}

pub(super) fn invariant_error(message: impl Into<String>) -> PipelineError {
    PipelineError::Invariant(message.into())
}

pub(super) fn worker_panic(worker: impl Into<String>) -> PipelineError {
    PipelineError::WorkerPanic {
        worker: worker.into(),
    }
}
