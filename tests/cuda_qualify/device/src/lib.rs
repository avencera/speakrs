//! Test-only device helpers and planted faults, never production kernels

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF32};
use cuda_device::{DisjointSlice, kernel, thread};

/// Rounds inputs to BF16 before an otherwise unchanged library computation
#[kernel]
pub fn qualify_round(mut values: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(value) = values.get_mut(idx) {
        let bits = value.to_bits();
        *value = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    }
}

/// Produces the exact abs-before-max, truncated width-three Sinc pool boundary
#[kernel]
pub fn qualify_pool(input: &[f32], in_len: u64, mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if let Some(value) = output.get_mut(idx) {
        let width = in_len as usize / 3;
        let start = (i / width) * in_len as usize + (i % width) * 3;
        *value = input[start].abs().max(input[start + 1].abs()).max(input[start + 2].abs());
    }
}

/// Skips all non-b32 work, or the last partial 256-element tile
#[kernel]
pub fn qualify_fault(mode: u32, batch: u32, mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let len = output.len();
    // a partial batch tile contains real rows, not only zero-padded audio tails
    let skipped = if batch % 32 != 0 {
        (batch % 32) as usize * (len / batch as usize)
    } else if len % 256 == 0 {
        256
    } else {
        len % 256
    };
    let tail = len - skipped;
    if let Some(value) = output.get_mut(idx) {
        if (mode == 1 && batch != 32) || (mode == 2 && i >= tail) {
            *value = 0.0;
        }
    }
}

/// Clears the independent atomic bins before each live nondeterminism test
#[kernel]
pub fn qualify_clear(mut bins: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(value) = bins.get_mut(idx) {
        *value = 0.0;
    }
}

/// Deliberately accumulates non-associative FP32 triples in scheduling order
#[kernel]
pub fn qualify_atomic(bins: &[DeviceAtomicF32]) {
    let i = thread::index_1d().get();
    let value = match (i / bins.len()) % 3 {
        0 => 1.0e8,
        1 => 1.0,
        _ => -1.0e8,
    };
    bins[i % bins.len()].fetch_add(value, AtomicOrdering::Relaxed);
}

/// Adds the atomic result to a real operator output, without a host-side nonce
#[kernel]
pub fn qualify_add(bins: &[f32], mut output: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i < bins.len() {
        if let Some(value) = output.get_mut(idx) {
            *value += bins[i] * 1.0e-5;
        }
    }
}

/// Writes one element beyond the allocation to prove sanitizer filter coverage
#[kernel]
pub fn qualify_oob(output: *mut f32, length: u64) {
    if thread::index_1d().get() == 0 {
        // SAFETY: intentionally invalid only in the isolated sanitizer mutation process
        unsafe { output.add(length as usize).write(1.0) };
    }
}
