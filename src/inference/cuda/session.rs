use super::{CudaError, CudaRuntime};

/// A CUDA runtime plus the models and buffers built on it, used by one thread at a time
///
/// The pipeline runs segmentation on a scoped worker thread that it starts for each
/// file, while embedding stays on the calling thread, and a queued pipeline moves into
/// its worker thread. So a model must be able to move between threads, but cudarc
/// marks its cuDNN handle, every cuDNN descriptor (each holds the handle) and CUDA
/// graphs as not `Send`. A session keeps all of those together with the runtime they
/// were created on, and [`Self::run`] makes the runtime's context current on the
/// calling thread before any work. The area that owns a session's state type
/// implements `Send` for that one instantiation, next to the state's definition
pub(crate) struct CudaSession<S> {
    // declared before the runtime so the plans, graphs and buffers are released while
    // the runtime's stream and handles still exist
    state: S,
    runtime: CudaRuntime,
}

impl<S> CudaSession<S> {
    /// Builds the session state on `runtime`
    pub(crate) fn new(
        runtime: CudaRuntime,
        build: impl FnOnce(&CudaRuntime) -> Result<S, CudaError>,
    ) -> Result<Self, CudaError> {
        runtime.context().bind_to_thread()?;
        tracing::info!(
            driver_only = super::driver_only(),
            force_library = runtime.force_library(),
            precedence = ?[
                super::implementation::policy::Source::TuneFile.name(),
                super::implementation::policy::Source::Recipe.name(),
                super::implementation::policy::Source::Default.name(),
                super::implementation::policy::Source::Library.name(),
            ],
            "CUDA model-load routing policy"
        );
        let state = build(&runtime)?;
        Ok(Self { state, runtime })
    }

    /// Runs `work` on this thread with the runtime's context current
    ///
    /// cuBLAS and cuDNN calls use whatever context is current on the calling thread,
    /// and cudarc only binds the context itself for driver calls, so every use of the
    /// state goes through here
    pub(crate) fn run<R, E>(
        &mut self,
        work: impl FnOnce(&CudaRuntime, &mut S) -> Result<R, E>,
    ) -> Result<R, E>
    where
        E: From<CudaError>,
    {
        self.runtime
            .context()
            .bind_to_thread()
            .map_err(CudaError::from)?;
        work(&self.runtime, &mut self.state)
    }
}

impl<S> Drop for CudaSession<S> {
    fn drop(&mut self) {
        // the fields drop after this, possibly on another thread than the one that
        // created them; their cuDNN and graph destructors need the context current. A
        // failure means the context is already gone and nothing more can be done
        let _ = self.runtime.context().bind_to_thread();
    }
}
