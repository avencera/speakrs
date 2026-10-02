use std::fmt;
use std::path::PathBuf;

use cudarc::cublas::result::CublasError;
use cudarc::cudnn::CudnnError;
use cudarc::driver::DriverError;
use safetensors::SafeTensorError;

use super::{ComputeCapability, PtxTier};

/// A CUDA shared library that the native backend loads at run time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaLibrary {
    /// The driver API, `libcuda`, installed with the NVIDIA driver
    Driver,
    /// cuBLAS
    Cublas,
    /// cuDNN 9
    Cudnn,
}

impl fmt::Display for CudaLibrary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Driver => "CUDA driver (libcuda)",
            Self::Cublas => "cuBLAS",
            Self::Cudnn => "cuDNN",
        })
    }
}

/// Errors from the native CUDA backend
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CudaError {
    /// A CUDA shared library could not be loaded
    #[error("could not load the {library} shared library")]
    LibraryUnavailable {
        /// The library that failed to load
        library: CudaLibrary,
    },
    /// The requested device does not exist
    #[error("CUDA device {ordinal} is not available; {count} device(s) found")]
    NoDevice {
        /// The requested device ordinal
        ordinal: usize,
        /// Devices the driver reports
        count: usize,
    },
    /// The device is older than the PTX baseline every kernel area ships
    #[error(
        "GPU compute capability {capability} is below the native CUDA backend's {baseline} baseline (sm_75, Turing)"
    )]
    UnsupportedDevice {
        /// The device's compute capability
        capability: ComputeCapability,
        /// The lowest PTX tier speakrs ships
        baseline: PtxTier,
    },
    /// `SPEAKRS_CUDA_PTX_TIER` names no known tier
    #[error("unknown PTX tier `{value}`; expected one of sm75, sm80, sm90, sm120")]
    InvalidPtxTier {
        /// The rejected value
        value: String,
    },
    /// The requested PTX tier is not compiled into this build
    #[error(
        "PTX tier {tier} is not compiled in; enable the `cuda-{tier}` feature or request a lower tier"
    )]
    PtxTierNotCompiled {
        /// The requested tier
        tier: PtxTier,
    },
    /// The requested PTX tier needs a newer GPU than the device
    #[error("PTX tier {tier} needs compute capability {}, but the GPU has {capability}", tier.min_capability())]
    PtxTierAboveDevice {
        /// The requested tier
        tier: PtxTier,
        /// The device's compute capability
        capability: ComputeCapability,
    },
    /// The CUDA driver API returned an error
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
    /// cuBLAS returned an error
    #[error("cuBLAS: {0}")]
    Cublas(#[from] CublasError),
    /// cuDNN returned an error
    #[error("cuDNN: {0}")]
    Cudnn(#[from] CudnnError),
    /// The driver could not load an embedded PTX module
    #[error("loading PTX module `{module}`: {source}")]
    ModuleLoad {
        /// The kernel module name
        module: &'static str,
        /// The driver error
        #[source]
        source: DriverError,
    },
    /// A PTX module has no kernel with the requested name
    #[error("PTX module `{module}` has no kernel `{kernel}`: {source}")]
    KernelMissing {
        /// The kernel module name
        module: &'static str,
        /// The requested kernel entry name
        kernel: String,
        /// The driver error
        #[source]
        source: DriverError,
    },
    /// A weights file could not be read
    #[error("reading weights `{path}`: {source}")]
    WeightsIo {
        /// The weights file
        path: PathBuf,
        /// The I/O error
        #[source]
        source: std::io::Error,
    },
    /// A weights file is not valid safetensors
    #[error("parsing weights `{path}`: {source}")]
    WeightsFormat {
        /// The weights file
        path: PathBuf,
        /// The safetensors error
        #[source]
        source: SafeTensorError,
    },
    /// A weights file lacks a required tensor
    #[error("weights `{path}` have no tensor `{name}`")]
    MissingTensor {
        /// The weights file
        path: PathBuf,
        /// The missing tensor name
        name: String,
    },
    /// A weight tensor is not stored as FP32
    #[error("tensor `{name}` is {dtype}, expected F32")]
    TensorDtype {
        /// The tensor name
        name: String,
        /// The stored dtype
        dtype: String,
    },
    /// A weight tensor has a different shape than the model expects
    #[error("tensor `{name}` has shape {actual:?}, expected {expected:?}")]
    TensorShape {
        /// The tensor name
        name: String,
        /// The shape the model expects
        expected: Vec<usize>,
        /// The stored shape
        actual: Vec<usize>,
    },
    /// A buffer length does not match the shape or operation it is used for
    #[error("{context}: buffer holds {actual} elements, needs {expected}")]
    BufferLength {
        /// Which operation checked the buffer
        context: &'static str,
        /// Elements the operation needs
        expected: usize,
        /// Elements the buffer holds
        actual: usize,
    },
    /// A dimension does not fit the 32-bit integers that cuBLAS and cuDNN take
    #[error("{context}: dimension {value} does not fit in a 32-bit CUDA library argument")]
    DimensionOverflow {
        /// Which operation checked the dimension
        context: &'static str,
        /// The dimension that overflowed
        value: usize,
    },
}

impl CudaError {
    /// Returns true when this machine has no usable NVIDIA GPU: the driver library is
    /// missing, the driver reports no device, or the GPU is older than the `sm_75`
    /// baseline
    ///
    /// A missing cuBLAS or cuDNN on a machine with a GPU is a broken install, not a
    /// missing GPU, so it does not count
    pub fn is_device_unavailable(&self) -> bool {
        matches!(
            self,
            Self::LibraryUnavailable {
                library: CudaLibrary::Driver
            } | Self::NoDevice { .. }
                | Self::UnsupportedDevice { .. }
        )
    }
}

/// Converts a dimension for a cuBLAS or cuDNN call
pub(super) fn to_c_int(context: &'static str, value: usize) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::DimensionOverflow { context, value })
}

/// Checks that a buffer holds exactly the elements an operation reads or writes
pub(super) fn check_len(
    context: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), CudaError> {
    if expected != actual {
        return Err(CudaError::BufferLength {
            context,
            expected,
            actual,
        });
    }

    Ok(())
}

/// Number of elements in a shape
pub(super) fn element_count(context: &'static str, shape: &[usize]) -> Result<usize, CudaError> {
    shape.iter().try_fold(1_usize, |count, &dim| {
        count.checked_mul(dim).ok_or(CudaError::DimensionOverflow {
            context,
            value: dim,
        })
    })
}
