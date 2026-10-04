use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

use super::super::{CudaError, CudaRuntime};

/// A recorded forward pass of one workspace
///
/// The graph holds raw pointers to the model's weights and the workspace's buffers,
/// which is why workspaces live inside the model that captured them
pub(super) struct CapturedGraph(CudaGraph);

impl std::fmt::Debug for CapturedGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CapturedGraph")
    }
}

impl CapturedGraph {
    /// Records everything `enqueue` queues on the runtime's stream
    pub(super) fn capture(
        runtime: &CudaRuntime,
        enqueue: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<Self, CudaError> {
        let stream = runtime.stream();
        // nothing queued before the capture may be pending when it starts
        stream.synchronize()?;
        let _tracking = EventTrackingPause::new(runtime);
        stream.begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let enqueued = enqueue();
        // always end the capture, so a failed enqueue leaves the stream usable
        let graph = stream.end_capture(CUgraphInstantiate_flags(0));
        enqueued?;

        let graph = graph?.ok_or(CudaError::BufferLength {
            context: "segmentation CUDA graph nodes",
            expected: 1,
            actual: 0,
        })?;
        Ok(Self(graph))
    }

    /// Queues one replay on the stream it was captured from
    pub(super) fn launch(&self) -> Result<(), CudaError> {
        self.0.launch()?;
        Ok(())
    }
}

/// Turns off cudarc's per-buffer event tracking on the runtime's context until dropped
///
/// cudarc makes every buffer access wait on the event of the buffer's last write and
/// record a new one. Inside a stream capture, waiting on an event recorded before the
/// capture fails with `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`. The flag belongs to this
/// runtime's context object and is read at each access, and everything a runtime
/// queues runs in order on its one stream, so pausing it during the capture loses no
/// ordering. Graph replays record no events either, which is fine as long as the
/// workspace buffers are only used on this runtime's stream
struct EventTrackingPause<'a> {
    runtime: &'a CudaRuntime,
    was_tracking: bool,
}

impl<'a> EventTrackingPause<'a> {
    fn new(runtime: &'a CudaRuntime) -> Self {
        let context = runtime.context();
        let was_tracking = context.is_event_tracking();
        // SAFETY: see the type docs; the capture only touches buffers of this runtime,
        // all on its stream, after a synchronize
        unsafe { context.disable_event_tracking() };
        Self {
            runtime,
            was_tracking,
        }
    }
}

impl Drop for EventTrackingPause<'_> {
    fn drop(&mut self) {
        if self.was_tracking {
            // SAFETY: restores the state from before the pause; nothing is allocated
            // during the capture
            unsafe { self.runtime.context().enable_event_tracking() };
        }
    }
}

#[cfg(test)]
mod test_support;
