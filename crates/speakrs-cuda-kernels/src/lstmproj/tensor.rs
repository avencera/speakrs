//! TF32 projection with two async staging slots and warp matrix fragments

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DynamicSharedArray, ptx_asm, thread};

const ROWS: usize = 128;
const COLS: usize = 256;
const K: usize = 16;
const STRIDE: usize = K + 4;
const A_WORDS: usize = ROWS * STRIDE;
const B_WORDS: usize = COLS * STRIDE;
const SLOT_WORDS: usize = A_WORDS + B_WORDS;

/// Computes a 128-by-256 projection tile in explicit TF32 mode
///
/// # Safety
///
/// The caller supplies 256 threads, grid `[2, ceil(rows / 128), 1]`, non-overlapping
/// FP32 input `[rows, columns]`, weights `[512, padded_columns]`, and output
/// `[rows, 512]`, with 61,440 dynamic shared bytes. Width is 60 or 256, padded to
/// 64 or 256 with a zero K tail in each weight column. Weights are rounded to TF32
#[inline(always)]
pub(super) unsafe fn project(
    input: *const f32,
    weights: *const f32,
    output: *mut f32,
    rows: usize,
    columns: usize,
    padded_columns: usize,
) {
    // shared instructions need a window-relative address, not a truncated generic pointer
    let shared =
        unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
    let tid = thread::threadIdx_x() as usize;
    let lane = tid % 32;
    let warp = tid / 32;
    let warp_row = warp / 4 * 64;
    let warp_col = warp % 4 * 64;
    let first_row = thread::blockIdx_y() as usize * ROWS;
    let first_col = thread::blockIdx_x() as usize * COLS;
    let mut sums = [[[0.0f32; 4]; 8]; 4];
    unsafe {
        stage(
            shared, input, weights, rows, columns, 0, first_row, first_col,
        )
    };
    let mut inner = 0;
    while inner < padded_columns {
        unsafe { wait() };
        // wait completes only this thread's copies; the barrier publishes every loader
        thread::sync_threads();
        let slot = inner / K % 2;
        let current = shared + (slot * SLOT_WORDS * 4) as u32;
        if inner + K < padded_columns {
            let next = shared + ((slot ^ 1) * SLOT_WORDS * 4) as u32;
            unsafe {
                stage(
                    next,
                    input,
                    weights,
                    rows,
                    columns,
                    inner + K,
                    first_row,
                    first_col,
                )
            };
        }

        unroll!(STEP in [0, 1] {
            let mut af = [[0u32; 4]; 4];
            unroll!(M in [0, 1, 2, 3] {
                let a_offset = (warp_row + M * 16 + lane % 8 + (lane / 8 % 2) * 8) * STRIDE
                    + lane / 16 * 4 + STEP * 8;
                let address = current + (a_offset * 4) as u32;
                let raw = unsafe { matrix4(address) };
                af[M] = raw.map(tf32_bits);
            });
            let mut bf = [[0u32; 2]; 8];
            unroll!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
                let offset = A_WORDS + (warp_col + N * 8 + lane % 8) * STRIDE
                    + (lane / 8 % 2) * 4 + STEP * 8;
                bf[N] = unsafe { matrix2(current + (offset * 4) as u32) };
            });
            unroll!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
                unroll!(M in [0, 1, 2, 3] {
                    sums[M][N] = unsafe { mma(af[M], bf[N], sums[M][N]) };
                });
            });
        });
        // the next iteration's barrier finishes every reader before its old slot is reused
        inner += K;
    }

    let col = first_col + warp_col + lane % 4 * 2;
    unroll!(M in [0, 1, 2, 3] {
        let row = first_row + warp_row + M * 16 + lane / 4;
        unroll!(N in [0, 1, 2, 3, 4, 5, 6, 7] {
            if row < rows {
                unsafe {
                    *(output.add(row * 512 + col + N * 8) as *mut Pair) =
                        Pair([sums[M][N][0], sums[M][N][1]]);
                }
            }
            if row + 8 < rows {
                unsafe {
                    *(output.add((row + 8) * 512 + col + N * 8) as *mut Pair) =
                        Pair([sums[M][N][2], sums[M][N][3]]);
                }
            }
        });
    });
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn stage(
    shared: u32,
    input: *const f32,
    weights: *const f32,
    rows: usize,
    columns: usize,
    inner: usize,
    first_row: usize,
    first_col: usize,
) {
    let tid = thread::threadIdx_x() as usize;
    unroll!(J in [0, 1, 2, 3] {
        let vector = tid + J * 256;
        let row = vector / (K / 4);
        let k = vector % (K / 4) * 4;
        let live = first_row + row < rows && inner + k < columns;
        // the zero-fill path passes the valid allocation base, not an out-of-range pointer
        let src = if live {
            unsafe { input.add((first_row + row) * columns + inner + k) }
        } else {
            input
        };
        unsafe {
            if vector < ROWS * K / 4 {
                copy(shared + ((row * STRIDE + k) * 4) as u32, src, if live { 16 } else { 0 });
            }
            if vector < COLS * K / 4 {
                copy(
                shared + ((A_WORDS + row * STRIDE + k) * 4) as u32,
                weights.add((first_col + row) * columns.div_ceil(16) * 16 + inner + k),
                16,
                );
            }
        }
    });

    unsafe { ptx_asm!("cp.async.commit_group;", clobber("memory")) };
}

#[inline(always)]
unsafe fn copy(address: u32, src: *const f32, bytes: u32) {
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.ca.shared.global [%0], [g], 16, %2; }",
            in("r") address, in("l") src, in("r") bytes, clobber("memory"),
        );
    }
}

#[inline(always)]
unsafe fn wait() {
    unsafe { ptx_asm!("cp.async.wait_group 0;", clobber("memory")) };
}

#[inline(always)]
fn tf32(value: f32) -> u32 {
    #[cfg(feature = "tier-sm120")]
    {
        let bits: u32;
        unsafe {
            ptx_asm!("cvt.rn.tf32.f32 %0, %1;", out("=r") bits, in("f") value,
                options(register_only));
        }
        bits
    }
    #[cfg(not(feature = "tier-sm120"))]
    {
        // native ties-to-even TF32 conversion needs sm90; this keeps sm80 precision identical
        let bits = value.to_bits();
        if bits & 0x7f80_0000 == 0x7f80_0000 {
            return bits;
        }
        (bits + 0x0fff + ((bits >> 13) & 1)) & 0xffff_e000
    }
}

#[inline(always)]
unsafe fn matrix4(address: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d, in("r") address);
    }
    [a, b, c, d]
}

#[inline(always)]
unsafe fn matrix2(address: u32) -> [u32; 2] {
    let (a, b): (u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];",
            out("=r") a, out("=r") b, in("r") address);
    }
    [a, b]
}

#[inline(always)]
unsafe fn mma(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = c;
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            inout("+f") c0, inout("+f") c1, inout("+f") c2, inout("+f") c3,
            in("r") a[0], in("r") a[1], in("r") a[2], in("r") a[3],
            in("r") b[0], in("r") b[1], options(register_only),
        );
    }
    [c0, c1, c2, c3]
}

#[inline(always)]
fn tf32_bits(value: u32) -> u32 {
    tf32(f32::from_bits(value))
}

#[repr(C, align(8))]
struct Pair([f32; 2]);
