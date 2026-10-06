#[cfg(feature = "cuda")]
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(feature = "cuda")]
use cudarc::cublas::{Gemm, GemmConfig};
use cudarc::driver::{DevicePtr, DevicePtrMut};

#[cfg(feature = "cuda")]
use super::error::{check_len, to_c_int};
use super::{CudaError, CudaMath, CudaRuntime};

/// A row-major GEMM on FP32 buffers: `c = alpha * op(a) · op(b) + beta * c`
///
/// `op(a)` is `m × k` and `op(b)` is `k × n`; `c` is `m × n`. A transposed operand
/// is stored as its transpose in row-major order, so a PyTorch `Linear` weight
/// (`out × in`) is `b` with `b_transposed = true`. cuBLAS is column-major, and
/// [`CudaRuntime::sgemm`] does the operand swap
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgemm {
    /// Rows of `op(a)` and `c`
    pub m: usize,
    /// Columns of `op(b)` and `c`
    pub n: usize,
    /// Columns of `op(a)` and rows of `op(b)`
    pub k: usize,
    /// `a` is stored as `k × m`
    pub a_transposed: bool,
    /// `b` is stored as `n × k`
    pub b_transposed: bool,
    /// Scale for the product
    pub alpha: f32,
    /// Scale for the existing `c`; 0 overwrites it
    pub beta: f32,
    /// Precision of the multiply; FP32 unless the caller opts into TF32
    pub math: CudaMath,
}

impl Sgemm {
    /// `c = a · b` with untransposed operands
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            a_transposed: false,
            b_transposed: false,
            alpha: 1.0,
            beta: 0.0,
            math: CudaMath::Fp32,
        }
    }

    /// The column-major cuBLAS call for this row-major product
    ///
    /// Row-major `c` (`m × n`) is column-major `cᵀ` (`n × m`), so cuBLAS computes
    /// `cᵀ = op(b)ᵀ · op(a)ᵀ` with `b` as its first operand. A row-major matrix read as
    /// column-major is already transposed, so an untransposed row-major operand maps
    /// to `N` and a transposed one to `T`
    #[cfg(feature = "cuda")]
    fn column_major(&self) -> Result<GemmConfig<f32>, CudaError> {
        let context = "sgemm";
        let op = |transposed: bool| {
            if transposed {
                cublasOperation_t::CUBLAS_OP_T
            } else {
                cublasOperation_t::CUBLAS_OP_N
            }
        };

        Ok(GemmConfig {
            transa: op(self.b_transposed),
            transb: op(self.a_transposed),
            m: to_c_int(context, self.n)?,
            n: to_c_int(context, self.m)?,
            k: to_c_int(context, self.k)?,
            alpha: self.alpha,
            // leading dimension = row length of the stored row-major matrix
            lda: to_c_int(context, if self.b_transposed { self.k } else { self.n })?,
            ldb: to_c_int(context, if self.a_transposed { self.m } else { self.k })?,
            beta: self.beta,
            ldc: to_c_int(context, self.n)?,
        })
    }
}

impl CudaRuntime {
    /// Runs a row-major GEMM on this runtime's stream in `spec.math` precision
    ///
    /// Each buffer must hold exactly its matrix; take a view of a larger persistent
    /// buffer to multiply part of it
    pub fn sgemm<A, B, C>(&self, spec: Sgemm, a: &A, b: &B, c: &mut C) -> Result<(), CudaError>
    where
        A: DevicePtr<f32>,
        B: DevicePtr<f32>,
        C: DevicePtrMut<f32>,
    {
        #[cfg(feature = "cuda")]
        {
            if super::driver_only() {
                return Err(Self::library_forbidden(super::CudaLibrary::Cublas));
            }
            check_len("sgemm a", spec.m.saturating_mul(spec.k), a.len())?;
            check_len("sgemm b", spec.k.saturating_mul(spec.n), b.len())?;
            check_len("sgemm c", spec.m.saturating_mul(spec.n), c.len())?;
            let config = spec.column_major()?;
            let _math = self.lock_blas(spec.math)?;
            #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
            let _library =
                super::test_support::call(&format!("cublas.m{}.n{}.k{}", spec.m, spec.n, spec.k));

            // SAFETY: the lengths match the dimensions and leading dimensions checked
            // above, and all buffers are device allocations that cudarc orders on the
            // handle's stream
            unsafe { self.blas()?.gemm(config, b, a, c) }?;
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (spec, a, b, c);
            Err(Self::library_forbidden(super::CudaLibrary::Cublas))
        }
    }
}
