//! Native CUDA backend for [`ExecutionMode::Cuda`](super::ExecutionMode::Cuda) and
//! [`ExecutionMode::CudaFast`](super::ExecutionMode::CudaFast), without ONNX Runtime
//!
//! The generic host stack (context, stream, module loading, buffers, cuBLAS/cuDNN
//! loading, GEMM and graph capture) is the shared `cudakit` crate; this module adds
//! speakrs's kernels, plans and policy
//!
//! - `CudaRuntime` wraps a `cudakit::Runtime` with speakrs's routing, recipe and tuning
//!   state; `CudaSession` keeps a runtime together with the models built on it so they
//!   can move between threads
//! - `KernelModule` embeds the committed cuda-oxide PTX, one module per area, as the
//!   best shipped variant for each enabled GPU target feature
//! - `SafetensorsFile` uploads named weights with shape checks; optional library plans
//!   use [`CudaMath`] defaults to FP32 with TF32 disabled
//! - `CudaFbank`, `ResNetEmbedding` and `CudaSegmentation` are the three models
//!
//! Public options and errors describe inference policy; [`tune_cuda`] measures
//! approved boundary choices on an explicitly requested device

/// Target-only builds contain no optional numerical libraries
const fn driver_only() -> bool {
    !cfg!(feature = "_cuda-libraries")
}

// kernel workers' candidates consume this interface; until one lands, parts of it
// and the candidate PTX areas are unused
#[allow(dead_code)]
mod candidate;
mod dense;
mod embedding;
mod error;
mod fbank;
mod geometry;
pub(crate) mod implementation;
#[cfg(test)]
mod kernel_inventory_tests;
mod kernels;
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
pub(crate) mod fp16_range_tests;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod test_support;
#[cfg(all(test, feature = "_cuda-libraries"))]
mod tests;

pub(crate) use cudakit::DeviceTensor;
use cudakit::Sgemm;
#[cfg(feature = "_cuda-libraries")]
use cudakit::dnn;
#[cfg(all(test, feature = "_cuda-libraries"))]
use cudakit::dnn::ConvPlanner;
pub(crate) use embedding::{
    EMBEDDING_DIM, EmbeddingBatch, EmbeddingBatchClass, ResNetEmbedding, SPEAKERS_PER_CHUNK,
    SharedEmbeddingActivations,
};
pub use error::{CudaError, CudaLibrary, GeometryError, WeightFault};
pub(crate) use fbank::{
    CudaFbank, FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers,
};
#[cfg(all(test, feature = "_cuda-libraries"))]
use geometry::Conv2d;
pub(crate) use kernels::{KernelModule, LoadedKernels};
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
