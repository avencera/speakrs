//! Native CUDA backend for [`ExecutionMode::Cuda`](super::ExecutionMode::Cuda) and
//! [`ExecutionMode::CudaFast`](super::ExecutionMode::CudaFast), without ONNX Runtime
//!
//! - [`CudaRuntime`] owns the device context, one stream, and the cuBLAS and cuDNN
//!   handles bound to that stream; [`CudaSession`] keeps a runtime together with the
//!   models built on it so they can move between threads
//! - [`KernelModule`] embeds the committed cuda-oxide PTX, one module per area, as an
//!   `sm75` baseline plus any higher [`PtxTier`] compiled in by the `cuda-sm80`,
//!   `cuda-sm90` and `cuda-sm120` features
//! - [`DeviceTensor`] is a shaped device buffer and [`SafetensorsFile`] uploads named
//!   weights with shape checks
//! - [`Sgemm`] wraps cuBLAS and [`ConvPlanner`] plans every cuDNN convolution; their
//!   [`CudaMath`] defaults to FP32 with TF32 disabled
//! - [`CudaFbank`], [`ResNetEmbedding`] and [`CudaSegmentation`] are the three models
//!
//! Only the option and error types are public; everything else is internal

mod blas;
mod buffer;
// kernel workers' candidates consume this interface; until one lands, parts of it
// and the candidate PTX areas are unused
#[allow(dead_code)]
mod candidate;
mod dnn;
mod embedding;
mod error;
mod fbank;
mod implementation;
mod kernels;
mod math;
mod options;
#[cfg(test)]
mod probe;
mod runtime;
mod segmentation;
mod session;
mod tier;
mod weights;

#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use blas::Sgemm;
pub(crate) use buffer::DeviceTensor;
#[cfg(test)]
use dnn::{Conv2d, ConvPlanner};
pub(crate) use embedding::{EMBEDDING_DIM, EmbeddingBatch, ResNetEmbedding, SPEAKERS_PER_CHUNK};
pub use error::{CudaError, CudaLibrary};
pub(crate) use fbank::{
    CudaFbank, FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers,
};
use kernels::{KernelModule, LoadedKernels};
pub use math::CudaMath;
pub use options::{CudaGraphs, CudaLstmAlgorithm};
pub(crate) use runtime::CudaRuntime;
pub(crate) use segmentation::{CudaSegmentation, SegmentationOptions};
pub(crate) use session::CudaSession;
pub use tier::{ComputeCapability, PtxTier};
pub(crate) use weights::SafetensorsFile;
