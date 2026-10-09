use std::ops::Deref;

use crate::inference::native_model::segmentation::PyanNetWeights;

use super::super::{CudaError, SafetensorsFile};

pub(super) use crate::inference::native_model::segmentation::LstmLayer;

/// ONNX LSTM gates are stored `[i, o, f, c]`; cuDNN gate `g` is ONNX gate
/// `ONNX_GATE[g]`, because cuDNN orders them `[i, f, c, o]`
#[cfg(feature = "_cuda-libraries")]
pub(super) const ONNX_GATE: [usize; 4] = [0, 2, 3, 1];

/// CUDA load adapter for the shared host PyanNet weights
#[derive(Debug, Clone)]
pub(super) struct SegmentationWeights(PyanNetWeights);

impl SegmentationWeights {
    /// Loads the shared recipe without changing CUDA error meanings
    pub(super) fn load(file: &SafetensorsFile) -> Result<Self, CudaError> {
        PyanNetWeights::load(file.host())
            .map(Self)
            .map_err(Into::into)
    }
}

impl Deref for SegmentationWeights {
    type Target = PyanNetWeights;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
