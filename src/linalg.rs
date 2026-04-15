#[cfg(all(feature = "intel-mkl", not(target_arch = "x86_64")))]
compile_error!("the `intel-mkl` feature is only supported on x86_64 targets");

#[cfg(feature = "intel-mkl")]
pub(crate) use ndarray_linalg_mkl::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(feature = "openblas-static")]
pub(crate) use ndarray_linalg_static::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(feature = "openblas-system")]
pub(crate) use ndarray_linalg_system::{Eigh, Inverse, UPLO, error::LinalgError};

#[cfg(not(any(
    feature = "intel-mkl",
    feature = "openblas-static",
    feature = "openblas-system"
)))]
pub(crate) use ndarray_linalg_default::{Eigh, Inverse, UPLO, error::LinalgError};
