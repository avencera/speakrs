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
//! Only the option and error types are public; everything else is internal

/// Tier-only builds and the explicit driver-only feature reject optional libraries
const fn driver_only() -> bool {
    !cfg!(feature = "cuda") || cfg!(feature = "cuda-driver-only")
}

mod blas;
mod buffer;
// kernel workers' candidates consume this interface; until one lands, parts of it
// and the candidate PTX areas are unused
#[allow(dead_code)]
mod candidate;
mod dispatch;
mod dnn;
mod embedding;
mod error;
mod fbank;
mod implementation;
mod kernels;
#[cfg(feature = "cuda")]
mod libraries;
#[cfg(all(test, target_os = "linux"))]
mod loader_tests;
mod math;
mod options;
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod probe;
mod runtime;
mod segmentation;
mod session;
mod tier;
mod weights;

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod test_support;
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod tests;

use blas::Sgemm;
pub(crate) use buffer::DeviceTensor;
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
use dnn::{Conv2d, ConvPlanner};
pub(crate) use embedding::{EMBEDDING_DIM, EmbeddingBatch, ResNetEmbedding, SPEAKERS_PER_CHUNK};
pub use error::{CudaError, CudaLibrary};
pub(crate) use fbank::{
    CudaFbank, FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers,
};
use kernels::{KernelModule, LoadedKernels};
pub use math::CudaMath;
pub use options::CudaGraphs;
#[cfg(feature = "cuda")]
pub use options::CudaLstmAlgorithm;
pub(crate) use runtime::CudaRuntime;
pub(crate) use segmentation::{CudaSegmentation, SegmentationOptions};
pub(crate) use session::CudaSession;
pub use tier::{ComputeCapability, PtxTier};
pub(crate) use weights::SafetensorsFile;
