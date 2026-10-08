//! Native ResNet34, one trunk pass followed by all masks for that chunk

mod pooling;
mod trunk;

use super::gemm::matmul;
use crate::inference::native_model::WeightsFile;
use crate::inference::{CpuError, InferenceError, TensorShapeError};
use ndarray::{Array2, ArrayView2};
use std::sync::Arc;
use trunk::{Trunk, TrunkScratch};

struct ResNetWeights {
    trunk: Trunk,
    head: Array2<f32>,
    bias: Vec<f32>,
}

/// Immutable validated deployed WeSpeaker weights shared between handles
#[derive(Clone)]
pub(crate) struct CpuResNet34(Arc<ResNetWeights>);

/// Private bounded scratch for one chunk, reused for all masks of that chunk
pub(crate) struct ResNetWorkspace {
    fbank: Vec<f32>,
    trunk: TrunkScratch,
    pooled: Array2<f32>,
}

impl CpuResNet34 {
    /// Loads exactly the deployed 74 tensors, with folded convolution biases
    pub(crate) fn load(file: &WeightsFile) -> Result<Self, CpuError> {
        let trunk = Trunk::load(file)?;
        let values = file.read_f32("resnet.seg_1.weight", &[256, 5120])?;
        let head = Array2::from_shape_vec((256, 5120), values).map_err(|_| {
            TensorShapeError::ShapeMismatch {
                context: "CPU embedding head",
                expected: vec![256, 5120],
                actual: vec![],
            }
        })?;
        let bias = file.read_f32("resnet.seg_1.bias", &[256])?;
        Ok(Self(Arc::new(ResNetWeights { trunk, head, bias })))
    }

    /// Creates scratch without reopening files or copying immutable weights
    pub(crate) fn workspace(&self) -> ResNetWorkspace {
        ResNetWorkspace {
            fbank: vec![0.0; 80 * 998],
            trunk: TrunkScratch::new(),
            pooled: Array2::zeros((1, 5120)),
        }
    }

    /// Pads a valid short filterbank, runs the trunk once, and projects each mask without L2 normalization
    pub(crate) fn forward(
        &self,
        fbank: ArrayView2<'_, f32>,
        masks: &[&[f32]],
        scratch: &mut ResNetWorkspace,
    ) -> Result<Array2<f32>, InferenceError> {
        if fbank.nrows() > 998 || fbank.ncols() != 80 {
            return Err(TensorShapeError::ShapeMismatch {
                context: "CPU embedding filterbank",
                expected: vec![998, 80],
                actual: fbank.shape().to_vec(),
            }
            .into());
        }
        let mut output = Array2::zeros((masks.len(), 256));
        if masks.is_empty() {
            return Ok(output);
        }
        scratch.fbank.fill(0.0);
        for time in 0..fbank.nrows() {
            for bin in 0..80 {
                scratch.fbank[bin * 998 + time] = fbank[[time, bin]];
            }
        }
        let features = self.0.trunk.forward(&scratch.fbank, &mut scratch.trunk)?;
        let features = features
            .into_shape_with_order((2560, 125))
            .map_err(|source| InferenceError::OutputArray {
                context: "CPU trunk output",
                source,
            })?;
        for (row, mask) in masks.iter().enumerate() {
            pooling::pool(features, mask, scratch.pooled.row_mut(0))?;
            matmul(
                scratch.pooled.view(),
                self.0.head.t(),
                output.slice_mut(ndarray::s![row..row + 1, ..]),
                0.0,
            )?;
            for (value, bias) in output.row_mut(row).iter_mut().zip(&self.0.bias) {
                *value += bias;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
