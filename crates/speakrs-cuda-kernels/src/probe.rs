//! Toolchain probe: proves that a cuda-oxide kernel builds to PTX, loads through
//! cudarc and runs on the GPU
//!
//! The probe ships an `sm80` variant next to the `sm75` baseline. Both contain the
//! same kernel; the second variant only exists so the tests can prove that the host
//! picks a tier by compute capability and that forcing the baseline gives the same
//! results

use cuda_device::{DisjointSlice, kernel, thread};

/// Writes `out[i] = alpha * x[i] + y[i]` for every `i` covered by `out`
///
/// Launch one thread per element of `out` with a 1-D grid; `x` and `y` must be at
/// least as long as `out`
#[kernel]
pub fn probe_scale_add(alpha: f32, x: &[f32], y: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = alpha * x[i] + y[i];
    }
}
