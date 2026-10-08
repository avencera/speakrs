//! FP16 range guard inspection for GPU tests in hybrid and driver-only builds

use super::dispatch::Plan;
use super::{EmbeddingBatch, Fp16Policy, ResNetEmbedding};
use crate::inference::cuda::{CudaError, CudaRuntime};

impl ResNetEmbedding {
    /// A batch whose every trunk layer is selected without FP16 tiles: the plans a
    /// batch recomputes with after an activation saturates
    pub(crate) fn batch_without_fp16(
        &self,
        runtime: &CudaRuntime,
        chunks: usize,
    ) -> Result<EmbeddingBatch, CudaError> {
        self.batch_with(runtime, chunks, Fp16Policy::Excluded)
    }
}

impl EmbeddingBatch {
    /// Trunk layers whose selected plan runs FP16 tiles
    pub(crate) fn fp16_convs(&self) -> Vec<&str> {
        self.plans
            .iter()
            .filter(|(_, plan)| matches!(plan, Plan::Wideconv(plan) if plan.is_fp16()))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Whether a download has recomputed this batch without FP16 tiles
    pub(crate) fn recomputed_without_fp16(&self) -> bool {
        self.fallback.is_some()
    }
}
