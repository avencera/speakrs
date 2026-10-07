//! Library-independent matrix geometry

use super::CudaMath;

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
}

#[cfg(not(feature = "_cuda-libraries"))]
impl super::CudaRuntime {
    /// Refuse a library multiply in a target-only build
    pub fn sgemm<A, B, C>(
        &self,
        spec: Sgemm,
        a: &A,
        b: &B,
        c: &mut C,
    ) -> Result<(), super::CudaError>
    where
        A: cudarc::driver::DevicePtr<f32>,
        B: cudarc::driver::DevicePtr<f32>,
        C: cudarc::driver::DevicePtrMut<f32>,
    {
        let _ = (spec, a, b, c);
        Err(Self::library_forbidden(super::CudaLibrary::Cublas))
    }
}
