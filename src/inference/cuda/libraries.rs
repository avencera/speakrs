//! Per-runtime fallible library cells, requested only while a plan is built

use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use cudarc::cublas::{CudaBlas, result::CublasError};
use cudarc::cudnn::{Cudnn, CudnnError};
use cudarc::driver::CudaStream;

use super::{CudaError, CudaLibrary, CudaMath};

#[derive(Debug, Clone, Copy)]
enum Failure {
    Missing(CudaLibrary),
    Blas(CublasError),
    Dnn(CudnnError),
}

impl From<Failure> for CudaError {
    fn from(failure: Failure) -> Self {
        match failure {
            Failure::Missing(library) => Self::LibraryUnavailable { library },
            Failure::Blas(error) => Self::Cublas(error),
            Failure::Dnn(error) => Self::Cudnn(error),
        }
    }
}

#[derive(Debug)]
struct Blas {
    handle: CudaBlas,
    // keep mode selection and enqueue in one critical section
    math: Mutex<CudaMath>,
}

/// Optional libraries drop before the runtime's stream and context
#[derive(Debug, Default)]
pub(super) struct Libraries {
    blas: OnceLock<Result<Blas, Failure>>,
    dnn: OnceLock<Result<Arc<Cudnn>, Failure>>,
    nvrtc: OnceLock<Result<(), Failure>>,
}

impl Libraries {
    /// The policy check is deliberately before the cell lookup and every probe
    pub(super) fn prepare(
        &self,
        library: CudaLibrary,
        stream: &Arc<CudaStream>,
    ) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(CudaError::Unsupported {
                context: "CUDA libraries",
                reason: "driver-only policy prohibits optional libraries".to_owned(),
            });
        }
        match library {
            CudaLibrary::Cublas => cached(self.blas.get_or_init(|| {
                present(library)?;
                let handle = CudaBlas::new(stream.clone()).map_err(Failure::Blas)?;
                set_math(&handle, CudaMath::Fp32).map_err(Failure::Blas)?;
                Ok(Blas {
                    handle,
                    math: Mutex::new(CudaMath::Fp32),
                })
            }))
            .map(|_| ()),
            CudaLibrary::Cudnn => cached(self.dnn.get_or_init(|| {
                present(library)?;
                Cudnn::new(stream.clone()).map_err(Failure::Dnn)
            }))
            .map(|_| ()),
            CudaLibrary::Nvrtc => cached(self.nvrtc.get_or_init(|| present(library))).map(|_| ()),
            CudaLibrary::Driver => Ok(()),
        }
    }

    /// Access an already prepared handle without loading during forward or capture
    pub(super) fn blas(&self) -> Result<&CudaBlas, CudaError> {
        let value = self
            .blas
            .get()
            .ok_or_else(|| unprepared(CudaLibrary::Cublas))?;
        cached(value).map(|blas| &blas.handle)
    }

    pub(super) fn dnn(&self) -> Result<&Arc<Cudnn>, CudaError> {
        cached(
            self.dnn
                .get()
                .ok_or_else(|| unprepared(CudaLibrary::Cudnn))?,
        )
    }

    pub(super) fn lock_blas(&self, math: CudaMath) -> Result<MutexGuard<'_, CudaMath>, CudaError> {
        let blas = cached(
            self.blas
                .get()
                .ok_or_else(|| unprepared(CudaLibrary::Cublas))?,
        )?;
        let mut current = blas.math.lock().unwrap_or_else(PoisonError::into_inner);
        if *current != math {
            set_math(&blas.handle, math)?;
            *current = math;
        }
        Ok(current)
    }
}

fn cached<T>(value: &Result<T, Failure>) -> Result<&T, CudaError> {
    value.as_ref().map_err(|error| (*error).into())
}

fn unprepared(library: CudaLibrary) -> CudaError {
    CudaError::Unsupported {
        context: "CUDA plan",
        reason: format!("{library} was not prepared before forward"),
    }
}

fn set_math(blas: &CudaBlas, math: CudaMath) -> Result<(), CublasError> {
    // SAFETY: the handle is live and the caller holds its mode lock
    unsafe { cudarc::cublas::sys::cublasSetMathMode(*blas.handle(), math.cublas()) }.result()
}

fn present(library: CudaLibrary) -> Result<(), Failure> {
    // SAFETY: NVIDIA library initializers have no caller preconditions
    let present = unsafe {
        match library {
            CudaLibrary::Cublas => cudarc::cublas::sys::is_culib_present(),
            CudaLibrary::Cudnn => cudarc::cudnn::sys::is_culib_present(),
            CudaLibrary::Nvrtc => cudarc::nvrtc::sys::is_culib_present(),
            CudaLibrary::Driver => true,
        }
    };
    if !present {
        return Err(Failure::Missing(library));
    }
    Ok(())
}
