//! Native CUDA backend for [`ExecutionMode::Cuda`](super::ExecutionMode::Cuda) and
//! [`ExecutionMode::CudaFast`](super::ExecutionMode::CudaFast), without ONNX Runtime
//!
//! - [`CudaRuntime`] owns the device context, one stream, and the cuBLAS and cuDNN
//!   handles lazily bound to that stream; [`CudaSession`] keeps a runtime together with the
//!   models built on it so they can move between threads
//! - [`KernelModule`] embeds the committed cuda-oxide PTX, one module per area, as the
//!   best shipped variant for each enabled GPU target feature
//! - [`DeviceTensor`] is a shaped device buffer and [`SafetensorsFile`] uploads named
//!   weights with shape checks
//! - [`Sgemm`] describes a row-major matrix product; optional library plans use
//!   [`CudaMath`] defaults to FP32 with TF32 disabled
//! - [`CudaFbank`], [`ResNetEmbedding`] and [`CudaSegmentation`] are the three models
//!
//! Public options and errors describe inference policy; [`tune_cuda`] measures
//! approved boundary choices on an explicitly requested device

/// Target-only builds contain no optional numerical libraries
const fn driver_only() -> bool {
    !cfg!(feature = "_cuda-libraries")
}

#[cfg(feature = "_cuda-libraries")]
mod blas;
mod buffer;
// kernel workers' candidates consume this interface; until one lands, parts of it
// and the candidate PTX areas are unused
#[allow(dead_code)]
mod candidate;
mod dense;
mod device;
#[cfg(feature = "_cuda-libraries")]
mod dnn;
mod embedding;
mod error;
mod fbank;
mod gemm;
mod geometry;
pub(crate) mod implementation;
#[cfg(test)]
mod kernel_inventory_tests;
mod kernels;
#[cfg(feature = "_cuda-libraries")]
mod libraries;
#[cfg(all(test, target_os = "linux"))]
mod loader_tests;
mod math;
mod options;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod probe;
mod runtime;
mod segmentation;
mod session;
mod tier;
mod tuning;
mod weights;

#[cfg(test)]
pub(crate) mod batch_class_tests;
#[cfg(test)]
mod fp16_range_tests;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod test_support;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod tests;

pub(crate) use buffer::DeviceTensor;
#[cfg(all(test, feature = "_cuda-libraries"))]
use dnn::ConvPlanner;
pub(crate) use embedding::{
    EMBEDDING_DIM, EmbeddingBatch, EmbeddingBatchClass, ResNetEmbedding, SPEAKERS_PER_CHUNK,
};
pub use error::{CudaError, CudaLibrary, GeometryError, WeightFault};
pub(crate) use fbank::{
    CudaFbank, FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers,
};
use gemm::Sgemm;
#[cfg(all(test, feature = "_cuda-libraries"))]
use geometry::Conv2d;
use kernels::{KernelModule, LoadedKernels};
pub use math::CudaMath;
pub use options::CudaGraphs;
#[cfg(feature = "_cuda-libraries")]
pub use options::CudaLstmAlgorithm;
pub(crate) use runtime::CudaRuntime;
pub(crate) use segmentation::{CudaSegmentation, SegmentationOptions};
pub(crate) use session::CudaSession;
pub use tier::{ComputeCapability, PtxTier};
pub use tuning::{
    CudaTuneError, CudaTuneMeasurement, CudaTuneOptions, CudaTuneReport, CudaTuneRow, tune_cuda,
};
pub(crate) use weights::SafetensorsFile;
