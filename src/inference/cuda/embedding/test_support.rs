//! Inspection helpers for direct CUDA tests

use super::{EmbeddingBatch, EmbeddingTapFn, SPEAKERS_PER_CHUNK};
use crate::inference::cuda::{CudaError, CudaRuntime};
use cudarc::driver::CudaSlice;

impl EmbeddingBatch {
    /// Runs the forward pass eagerly and calls `tap` with every intermediate
    /// activation listed in [`EmbeddingTap`], in execution order
    ///
    /// The view is only valid during the call; download it there if needed. This
    /// never replays a captured graph
    pub fn forward_with_taps(
        &mut self,
        runtime: &CudaRuntime,
        tap: &mut EmbeddingTapFn<'_>,
    ) -> Result<(), CudaError> {
        self.run(runtime, tap)
    }

    /// Copies host fbank and masks into the batch, runs the forward pass and
    /// downloads the embeddings
    ///
    /// `fbank` is `[chunks, 998, 80]` and `masks` is `[chunks * 3, 589]`; the result
    /// is `[chunks * 3, 256]`, all row-major
    pub fn embed(
        &mut self,
        runtime: &CudaRuntime,
        fbank: &[f32],
        masks: &[f32],
    ) -> Result<Vec<f32>, CudaError> {
        let stream = runtime.stream();
        self.fbank.copy_from_host(stream, fbank)?;
        self.masks.copy_from_host(stream, masks)?;
        self.forward(runtime)?;
        self.download_output(runtime)
    }

    /// Embeddings per forward pass, `chunks * 3`
    pub fn rows(&self) -> usize {
        self.chunks * SPEAKERS_PER_CHUNK
    }

    /// Distinct convolution shapes, one cuDNN plan each
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Whether [`Self::forward`] replays a captured CUDA graph
    pub fn has_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Bytes of the shared cuDNN workspace
    pub fn workspace_bytes(&self) -> usize {
        self.workspace.len()
    }

    /// Bytes held by this batch's activation, input and output buffers, excluding
    /// the cuDNN workspace
    pub fn buffer_bytes(&self) -> usize {
        let floats = self.fbank.len()
            + self.masks.len()
            + self.stem_input.len()
            + self.trunk.iter().map(CudaSlice::len).sum::<usize>()
            + self.hidden.len()
            + self.shortcut.len()
            + self.pooled.len()
            + self.output.len();
        floats * size_of::<f32>()
    }
}
