#[cfg(feature = "cpu")]
pub(crate) mod cpu;
#[cfg(feature = "cpu")]
pub use cpu::{CpuError, CpuModelFamily};
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
pub(crate) mod embedding;
mod error;
pub(crate) mod geometry;
#[cfg(any(feature = "cpu", feature = "cuda"))]
pub(crate) mod native_model;
#[cfg(feature = "_ort")]
mod ort_runtime;
pub(crate) mod segmentation;

use std::fmt;

pub use embedding::EmbeddingModel;
pub use error::{ExecutionModeError, InferenceError, ModelLoadError};
pub use geometry::TensorShapeError;
#[cfg(any(feature = "cpu", feature = "cuda"))]
pub use native_model::NativeWeightsError;
#[cfg(feature = "_ort")]
pub use ort_runtime::{DynamicRuntimeError, OrtRuntimeError, with_execution_mode};
#[cfg(feature = "migraphx")]
pub(crate) use ort_runtime::{OrtProvider, SharedSession, ensure_ort_ready};
pub use segmentation::{SegmentationError, SegmentationModel};

#[cfg(feature = "coreml")]
pub(crate) mod coreml;
#[cfg(feature = "coreml")]
#[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
pub use coreml::CoreMlError;
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use cuda::{
    ComputeCapability, CudaError, CudaGraphs, CudaLibrary, CudaLstmAlgorithm, CudaMath, PtxTier,
};

/// CoreML compute unit selection for native embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMlComputeUnits {
    /// Use all available compute units: CPU + GPU + Neural Engine (default)
    #[default]
    All,
    /// Use CPU + Neural Engine only (skip GPU)
    CpuAndNeuralEngine,
    /// Use CPU only for native embedding models
    CpuOnly,
}

#[cfg(feature = "coreml")]
impl CoreMlComputeUnits {
    pub(crate) fn to_ml_compute_units(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::All => crate::inference::coreml::CoreMlModel::default_compute_units(),
            Self::CpuAndNeuralEngine => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            Self::CpuOnly => objc2_core_ml::MLComputeUnits::CPUOnly,
        }
    }
}

/// Which backend and acceleration to use for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionMode {
    /// Native CPU inference with Rust operators and safetensors weights
    #[cfg_attr(docsrs, doc(cfg(feature = "cpu")))]
    Cpu,
    /// Native CoreML with FP32 precision and ~1s step
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    CoreMl,
    /// Native CoreML with W8A16 segmentation and ~2s step
    #[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
    CoreMlFast,
    /// Native NVIDIA GPU backend (cuBLAS, cuDNN and custom kernels, no ONNX Runtime)
    /// with concurrent segmentation and embedding and ~1s step
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    Cuda,
    /// Native NVIDIA GPU backend with concurrent segmentation and embedding and ~2s step
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    CudaFast,
    /// AMD GPU via ONNX Runtime's MIGraphX execution provider
    #[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
    MiGraphX,
}

impl ExecutionMode {
    /// Returns true when this mode uses native CoreML execution
    pub const fn is_coreml(self) -> bool {
        matches!(self, Self::CoreMl | Self::CoreMlFast)
    }

    /// Returns true when this mode uses the native CUDA backend
    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda | Self::CudaFast)
    }

    /// Returns true when this mode uses the MIGraphX execution provider
    pub const fn is_migraphx(self) -> bool {
        matches!(self, Self::MiGraphX)
    }

    pub(crate) fn validate(self) -> Result<(), ExecutionModeError> {
        self.backend().map(|_| ())
    }

    /// Resolve the inference backend that owns this mode in the current build
    pub(crate) fn backend(self) -> Result<InferenceBackend, ExecutionModeError> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu => Ok(InferenceBackend::Cpu),
            #[cfg(not(feature = "cpu"))]
            Self::Cpu => Err(ExecutionModeError {
                mode: self,
                feature: "cpu",
            }),
            #[cfg(feature = "coreml")]
            Self::CoreMl | Self::CoreMlFast => Ok(InferenceBackend::CoreMl),
            #[cfg(not(feature = "coreml"))]
            Self::CoreMl | Self::CoreMlFast => Err(ExecutionModeError {
                mode: self,
                feature: "coreml",
            }),
            #[cfg(feature = "cuda")]
            Self::Cuda | Self::CudaFast => Ok(InferenceBackend::Cuda),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda | Self::CudaFast => Err(ExecutionModeError {
                mode: self,
                feature: "cuda",
            }),
            #[cfg(feature = "migraphx")]
            Self::MiGraphX => Ok(InferenceBackend::Ort(OrtProvider::MiGraphX)),
            #[cfg(not(feature = "migraphx"))]
            Self::MiGraphX => Err(ExecutionModeError {
                mode: self,
                feature: "migraphx",
            }),
        }
    }

    /// Lowercase identifier used in logs, docs, and user-facing errors
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::CoreMlFast => "coreml-fast",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
            Self::MiGraphX => "migraphx",
        }
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Inference runtime that owns a model's sessions for one execution mode
///
/// Another accelerator backend adds a variant here and a matching model backend in
/// `segmentation` and `embedding`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferenceBackend {
    /// Native CPU model owners and private scratch
    #[cfg(feature = "cpu")]
    Cpu,
    /// ONNX Runtime with one execution provider
    #[cfg(feature = "migraphx")]
    Ort(OrtProvider),
    /// Native CoreML models
    #[cfg(feature = "coreml")]
    CoreMl,
    /// Native CUDA models (cudarc, cuBLAS, cuDNN and cuda-oxide kernels)
    #[cfg(feature = "cuda")]
    Cuda,
}

#[cfg(test)]
mod tests {
    #[cfg(any(
        not(feature = "cpu"),
        not(feature = "coreml"),
        not(feature = "cuda"),
        not(feature = "migraphx")
    ))]
    use super::ExecutionMode;

    #[cfg(not(feature = "cpu"))]
    #[test]
    fn cpu_mode_requires_feature() {
        let error = ExecutionMode::Cpu.validate().unwrap_err();
        assert_eq!(error.to_string(), "cpu requires the `cpu` Cargo feature");
    }

    #[cfg(not(feature = "coreml"))]
    #[test]
    fn coreml_modes_require_feature() {
        let error = ExecutionMode::CoreMl.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml requires the `coreml` Cargo feature"
        );

        let error = ExecutionMode::CoreMlFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml-fast requires the `coreml` Cargo feature"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_modes_require_feature() {
        let error = ExecutionMode::Cuda.validate().unwrap_err();
        assert_eq!(error.to_string(), "cuda requires the `cuda` Cargo feature");

        let error = ExecutionMode::CudaFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "cuda-fast requires the `cuda` Cargo feature"
        );
    }

    #[cfg(not(feature = "migraphx"))]
    #[test]
    fn migraphx_mode_requires_feature() {
        let error = ExecutionMode::MiGraphX.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "migraphx requires the `migraphx` Cargo feature"
        );
    }
}
