#[cfg(feature = "_cuda-libraries")]
use cudarc::cublas::sys::cublasMath_t;
#[cfg(feature = "_cuda-libraries")]
use cudarc::cudnn::sys::cudnnMathType_t;

/// Arithmetic precision of CUDA kernels, cuBLAS and cuDNN work on FP32 data
///
/// Segmentation and embedding choose it separately through
/// [`RuntimeConfig`](crate::pipeline::RuntimeConfig), which defaults to [`Self::Fp32`]
/// for segmentation and [`Self::Tf32`] for embedding. The filterbank always runs in FP32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CudaMath {
    /// Full FP32 multiply and accumulate; tensor cores only where they keep FP32
    /// precision; never uses FP16 operands
    ///
    /// The closest match to the ONNX references, at some cost in speed. Embedding drift
    /// can change PLDA/VBx clustering (adr/001), so this is the choice to compare against
    Fp32,
    /// FP32 storage and accumulation with reduced-precision tensor-core multiplies
    ///
    /// Ampere and newer may use TF32 (10-bit mantissa) operands. Same-channel,
    /// stride-1 embedding trunk layers may instead use scaled FP16 operands on
    /// Turing and newer, with FP32 accumulation. Startup recipes and defaults
    /// select these on Turing and the RTX 4060 Ti; tuning can select them on other
    /// supported devices. This mode does not require every multiply to use TF32
    ///
    /// FP16 is excluded for a layer with non-finite weights or weight magnitudes
    /// above 65504 / 1024. If an activation exceeds that range, the complete
    /// embedding batch is recomputed with plans selected without FP16 tiles
    /// [`Self::Fp32`] never selects FP16 tiles
    Tf32,
}

#[cfg(feature = "_cuda-libraries")]
impl CudaMath {
    pub(super) fn cublas(self) -> cublasMath_t {
        match self {
            // the default math mode only uses tensor cores at full FP32 precision
            Self::Fp32 => cublasMath_t::CUBLAS_DEFAULT_MATH,
            Self::Tf32 => cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
        }
    }

    pub(super) fn cudnn(self) -> cudnnMathType_t {
        match self {
            // cuDNN's default math type would allow TF32 for FP32 data on Ampere and newer
            Self::Fp32 => cudnnMathType_t::CUDNN_FMA_MATH,
            // tensor-op math runs FP32 data as TF32; `_ALLOW_CONVERSION` is avoided
            // because it may also down-convert FP32 tensors to FP16
            Self::Tf32 => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        }
    }
}
