//! Inspection helpers for direct CUDA tests

use super::{EmbeddingBatch, ResNetEmbedding, SharedEmbeddingActivations};
#[cfg(feature = "_cuda-libraries")]
use super::{EmbeddingTapFn, SPEAKERS_PER_CHUNK};
#[cfg(feature = "_cuda-libraries")]
use super::{dispatch::Plan, trunk::ConvLayer};
#[cfg(feature = "_cuda-libraries")]
use crate::inference::cuda::candidate::{WideconvAlgorithm, WideconvFp16Tiles, WideconvOxide};
use crate::inference::cuda::{CudaError, CudaRuntime};
use cudarc::driver::CudaSlice;
use std::sync::Arc;

#[cfg(feature = "_cuda-libraries")]
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
        let activations = self.activations.clone();
        let mut storage = activations.lock()?;
        self.run_with_activations(
            runtime,
            super::PlanSet::Selected,
            super::HiddenForm::Fp32,
            tap,
            storage.buffers_mut(),
        )
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

    /// Residual blocks whose two convolutions run wide FP16 tiles on 32 or 64
    /// channels, and how many of those [`Self::forward`] passes a half hidden
    /// activation, read from the selected plans
    pub fn half_hidden_blocks(&self) -> (usize, usize) {
        let plan = |layer: &ConvLayer| {
            self.plans.iter().find_map(|(name, plan)| match plan {
                Plan::Wideconv(plan) if name == layer.name() => Some(plan),
                _ => None,
            })
        };
        // the shape test is separate from the half launches so a wide FP16 pair
        // that lost them is counted here but not as half
        let wide = |layer: &ConvLayer| {
            let conv = layer.conv(self.chunks, self.model.math);
            matches!(conv.in_channels, 32 | 64)
                && conv.out_channels == conv.in_channels
                && conv.stride == [1, 1]
                && plan(layer).is_some_and(|plan| {
                    plan.config().algorithm == WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide)
                })
        };
        let half = |layer: &ConvLayer| plan(layer).is_some_and(WideconvOxide::has_half_io);

        let blocks = &self.model.trunk.blocks;
        let wide_pairs = blocks
            .iter()
            .filter(|block| wide(&block.conv1) && wide(&block.conv2))
            .count();
        let half_pairs = blocks
            .iter()
            .filter(|block| half(&block.conv1) && half(&block.conv2))
            .count();
        (wide_pairs, half_pairs)
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
            + self.pooled.len()
            + self.output.len();
        floats * size_of::<f32>() + self.activations.retained_bytes()
    }
}

impl ResNetEmbedding {
    /// Allocates the device buffers and plans the convolutions for `chunks` fbank
    /// chunks per forward pass
    ///
    /// Do this once per batch class and reuse the batch for every forward pass of
    /// that size. The batch keeps a handle to this model and must run on
    /// `runtime`, whose stream owns its buffers
    pub fn batch(&self, runtime: &CudaRuntime, chunks: usize) -> Result<EmbeddingBatch, CudaError> {
        let activations = self.activations(runtime, chunks)?;
        self.batch_with_activations(runtime, chunks, activations)
    }
}

impl SharedEmbeddingActivations {
    pub(crate) fn retained_bytes(&self) -> usize {
        let storage = self.lock().unwrap();
        let buffers = storage.buffers();
        (buffers.trunk.iter().map(CudaSlice::len).sum::<usize>()
            + buffers.hidden.len()
            + buffers.shortcut.len())
            * size_of::<f32>()
    }
}

impl EmbeddingBatch {
    pub(crate) fn shares_activations(&self, other: &SharedEmbeddingActivations) -> bool {
        Arc::ptr_eq(&self.activations.0, &other.0)
    }
}
