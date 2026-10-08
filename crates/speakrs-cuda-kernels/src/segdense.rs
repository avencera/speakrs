//! Driver-only replacements for the segmentation convolutions, the segmentation
//! dense heads and the embedding `seg_1` projection
//!
//! Every boundary is a GEMM with a fixed shape, so every kernel is a macro
//! instance with its shape and tile as constants. A kernel serves one call site
//! and one production batch; the host picks it by name
//!
//! - `conv!`: valid five-tap NCW convolution as an implicit GEMM over output
//!   positions and channels. A thread owns 8 adjacent positions of 4 or 8 output
//!   channels and reads one 12-float input window per input channel, which it
//!   reuses for all five taps
//! - `gemm!`: row-major `a [m, k]` times packed `b [k, n]` in register-blocked
//!   `tm x tn` thread tiles, with a fused bias and LeakyReLU, a fused bias, or raw
//!   split-K partials
//! - `classifier!`: the 128-to-7 head with a fused bias and log-softmax; each
//!   thread keeps its share of the weights in registers for all of its rows
//! - `embed_gemv!`: the batch-1 embedding projection, which streams each weight
//!   row once for all three input rows
//! - `mma_gemm!` (sm80 tier only, through `tf32_gemm!`): the TF32-mode GEMMs on
//!   `mma.sync` m16n8k8 tensor cores fed by a `cp.async` pipeline
//!
//! The FP32-mode kernels accumulate in FP32 FMA in a fixed order. The TF32-mode
//! kernels round both operands to TF32 and accumulate in FP32, as the library's
//! TF32 mode does, also in a fixed order. Partial sums of a reduction split meet in a fixed order through shared
//! memory or a separate reduction kernel; there are no float atomics, so output
//! is bitwise deterministic
//!
//! Shared and global accesses in the hot loops use explicit vector instructions,
//! since LLVM does not reliably form them from scalar code. Index arithmetic is
//! 32-bit; every tensor here has far fewer than `u32::MAX` elements

#[cfg(feature = "tier-sm80")]
use cuda_device::DynamicSharedArray;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, ptx_asm, thread, warp};

/// Output channels of both segmentation convolutions
const CONV_OUT: u32 = 60;
/// Output channels padded to whole thread tiles; the packed weights are zero there
const CONV_PADDED: u32 = 64;
/// Convolution taps
const TAPS: u32 = 5;

/// Packs `src [n, k]` into `dst [k, padded]`, with zeros in columns `n..padded`
///
/// Launch one thread per element of `dst`, which has `k * padded` elements
#[kernel]
pub fn spk_segdense_pack(src: &[f32], mut dst: DisjointSlice<f32>, n: u32, k: u32, padded: u32) {
    let index = thread::index_1d();
    let i = index.get() as u32;
    let (row, column) = (i / padded, i % padded);
    if let Some(value) = dst.get_mut(index) {
        *value = if column < n {
            src[(column * k + row) as usize]
        } else {
            0.0
        };
    }
}

/// `value` rounded to the nearest TF32, ties away from zero, as `cvt.rna.tf32.f32`
/// rounds finite values; plain integer arithmetic, so every tier packs alike
#[inline(always)]
fn round_tf32(value: f32) -> f32 {
    f32::from_bits(value.to_bits().wrapping_add(0x1000) & 0xffff_e000)
}

/// Packs conv weights `src [60, cin, 5]` into the m16n8k8 `a` fragments of
/// `mma_conv!`, rounded to TF32
///
/// The reduction index is `ci * 5 + tap` over `cin_pad` input channels, with zeros
/// past `cin` and past output channel 60. `dst` holds `planes` planes of
/// `[cin_pad * 5 / 8][4][32][4]`: k-step, 16-channel tile, lane and fragment
/// register. Plane 0 is the TF32 value, plane 1 (when `planes == 2`) the TF32
/// rounding residual of the 3xTF32 split. Launch one thread per element of `dst`
#[kernel]
pub fn spk_segdense_pack_conv_mma(
    src: &[f32],
    mut dst: DisjointSlice<f32>,
    cin: u32,
    cin_pad: u32,
    planes: u32,
) {
    let index = thread::index_1d();
    let i = index.get() as u32;
    let plane_len = cin_pad * TAPS * CONV_PADDED;
    let (plane, rest) = (i / plane_len, i % plane_len);
    let (step, rest) = (rest / 512, rest % 512);
    let (tile, rest) = (rest / 128, rest % 128);
    let (lane, register) = (rest / 4, rest % 4);
    let channel = tile * 16 + lane / 4 + register % 2 * 8;
    let k = step * 8 + lane % 4 + register / 2 * 4;
    let (ci, tap) = (k / TAPS, k % TAPS);
    let inside = channel < CONV_OUT && ci < cin && plane < planes;
    let value = if inside {
        src[((channel * cin + ci) * TAPS + tap) as usize]
    } else {
        0.0
    };
    let high = round_tf32(value);
    if let Some(out) = dst.get_mut(index) {
        *out = if plane == 0 {
            high
        } else {
            round_tf32(value - high)
        };
    }
}

/// `a + b` rounded and its exact rounding error (Knuth's TwoSum), for any order of
/// magnitudes; no products, so no FMA contraction can change the result
#[inline(always)]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let sum = a + b;
    let b_part = sum - a;
    (sum, (a - (sum - b_part)) + (b - b_part))
}

/// Hides a shared address from LLVM, which otherwise proves the low bits are zero
/// and rewrites later `base + offset` additions as `base | offset`; `ptxas` folds
/// only the additions into the load and store immediates
#[inline(always)]
fn opaque(value: u32) -> u32 {
    let out: u32;
    // safety: a register move with no memory access
    unsafe {
        ptx_asm!("mov.b32 %0, %1;", out("=r") out, in("r") value, options(register_only));
    }
    out
}

/// Four floats from a 16-byte aligned shared address
#[inline(always)]
unsafe fn lds4(address: u32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    // safety: the caller passes an aligned address inside this block's shared tile
    unsafe {
        ptx_asm!(
            "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];",
            out("=f") a,
            out("=f") b,
            out("=f") c,
            out("=f") d,
            in("r") address,
        );
    }
    [a, b, c, d]
}

/// One float from a shared address
#[inline(always)]
unsafe fn lds1(address: u32) -> f32 {
    let a: f32;
    // safety: the caller passes an address inside this block's shared tile
    unsafe {
        ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") a, in("r") address);
    }
    a
}

/// Stores one float at a shared address
#[inline(always)]
unsafe fn sts1(address: u32, value: f32) {
    // safety: the caller passes an address inside this block's shared tile that no
    // other thread accesses until the next barrier
    unsafe {
        ptx_asm!("st.shared.f32 [%0], %1;", in("r") address, in("f") value);
    }
}

/// Stores four floats at a 16-byte aligned shared address
#[inline(always)]
unsafe fn sts4(address: u32, value: [f32; 4]) {
    // safety: as for `sts1`, with an aligned address
    unsafe {
        ptx_asm!(
            "st.shared.v4.f32 [%0], {%1, %2, %3, %4};",
            in("r") address,
            in("f") value[0],
            in("f") value[1],
            in("f") value[2],
            in("f") value[3],
        );
    }
}

/// Stores two floats at an 8-byte aligned shared address
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn sts2(address: u32, value: [f32; 2]) {
    // safety: as for `sts1`, with an aligned address
    unsafe {
        ptx_asm!(
            "st.shared.v2.f32 [%0], {%1, %2};",
            in("r") address,
            in("f") value[0],
            in("f") value[1],
        );
    }
}

/// Four floats from a 16-byte aligned global address, through the read-only path
#[inline(always)]
unsafe fn ldg4(pointer: *const f32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    // safety: the caller passes an aligned pointer to four readable floats of a
    // buffer that no kernel writes while this one runs
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.nc.v4.f32 {%0, %1, %2, %3}, [g]; }",
            out("=f") a,
            out("=f") b,
            out("=f") c,
            out("=f") d,
            in("l") pointer as u64,
        );
    }
    [a, b, c, d]
}

/// One float from a global address, through the read-only path
#[inline(always)]
unsafe fn ldg1(pointer: *const f32) -> f32 {
    let a: f32;
    // safety: the caller passes a pointer to a readable float of a buffer that no
    // kernel writes while this one runs
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.nc.f32 %0, [g]; }",
            out("=f") a,
            in("l") pointer as u64,
        );
    }
    a
}

/// Stores four floats at a 16-byte aligned global address
#[inline(always)]
unsafe fn stg4(pointer: *mut f32, value: [f32; 4]) {
    // safety: the caller passes an aligned pointer to four writable floats that no
    // other thread writes
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.v4.f32 [g], {%1, %2, %3, %4}; }",
            in("l") pointer as u64,
            in("f") value[0],
            in("f") value[1],
            in("f") value[2],
            in("f") value[3],
            clobber("memory"),
        );
    }
}

/// Copies 16 bytes from global to shared memory without a register round trip,
/// filling with zeros past the first `bytes` source bytes
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn cp16(address: u32, pointer: *const f32, bytes: u32) {
    // safety: the caller passes an aligned shared address inside this block's tile
    // that no thread reads before the copy's group completes, and an aligned
    // pointer to readable memory
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.cg.shared.global [%0], [g], 16, %2; }",
            in("r") address,
            in("l") pointer as u64,
            in("r") bytes,
            clobber("memory"),
        );
    }
}

/// Copies 4 bytes from global to shared memory without a register round trip,
/// or stores a zero when `bytes` is 0
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn cp4(address: u32, pointer: *const f32, bytes: u32) {
    // safety: the caller passes a 4-byte aligned shared address inside this
    // block's tile that no thread reads before the copy's group completes, and a
    // pointer to readable memory
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.ca.shared.global [%0], [g], 4, %2; }",
            in("r") address,
            in("l") pointer as u64,
            in("r") bytes,
            clobber("memory"),
        );
    }
}

/// Bytes of dynamic shared memory this launch provides
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn dynamic_smem_bytes() -> u32 {
    let bytes: u32;
    // safety: a special-register read with no memory access
    unsafe {
        ptx_asm!("mov.u32 %0, %%dynamic_smem_size;", out("=r") bytes, options(register_only));
    }
    bytes
}

/// Closes this thread's current group of `cp16` copies
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn cp_commit() {
    // safety: only groups this thread's earlier copies
    unsafe { ptx_asm!("cp.async.commit_group;", clobber("memory")) };
}

/// Waits until at most `pending` of this thread's copy groups are in flight
///
/// The wait count must be an immediate, so each supported count has its own
/// instruction; a constant argument folds the match away
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn cp_wait(pending: u32) {
    // safety: waiting has no effect beyond ordering this thread's copies; a count
    // without its own arm waits for every group, which is never too early
    unsafe {
        match pending {
            1 => ptx_asm!("cp.async.wait_group 1;", clobber("memory")),
            2 => ptx_asm!("cp.async.wait_group 2;", clobber("memory")),
            3 => ptx_asm!("cp.async.wait_group 3;", clobber("memory")),
            4 => ptx_asm!("cp.async.wait_group 4;", clobber("memory")),
            _ => ptx_asm!("cp.async.wait_group 0;", clobber("memory")),
        }
    }
}

/// `value` rounded to the nearest TF32, as raw bits for `mma`
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn tf32(value: f32) -> u32 {
    let bits: u32;
    // safety: a register conversion with no memory access
    unsafe {
        ptx_asm!(
            "cvt.rna.tf32.f32 %0, %1;",
            out("=r") bits,
            in("f") value,
            options(register_only),
        );
    }
    bits
}

/// `acc + a b` for one warp-wide m16n8k8 TF32 tile with FP32 accumulation
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn mma(acc: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = acc;
    // safety: a register-only warp instruction; every lane of the warp executes it
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b[0],
            in("r") b[1],
        );
    }
    [c0, c1, c2, c3]
}

/// `value` as TF32 bits, and with `split` also the TF32 bits of its rounding
/// residual, so `high + low` carries about 22 significant bits
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn split_tf32(value: f32, split: bool) -> (u32, u32) {
    let high = tf32(value);
    if !split {
        return (high, 0);
    }

    (high, tf32(value - f32::from_bits(high)))
}

/// `acc + a b` for one m16n8k8 tile: one TF32 product, or with `split_a` and
/// `split_b` the 3xTF32 product `a_low b + a b_low + a b` that keeps FP32-level
/// accuracy. With `split_b` alone, the 2xTF32 product `a b_low + a b` rounds only
/// `a` to TF32, which halves the error variance of a TF32 product
///
/// The tensor core truncates when it aligns its products to the accumulator, so
/// summing a long reduction inside `mma` loses low bits with a bias that grows with
/// the reduction. The split product therefore starts from zero, with the small
/// cross terms first, and joins `acc` in a rounded FP32 addition. The `a_low b_low`
/// term is below FP32 resolution of the product and is dropped
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn mma_product(
    acc: [f32; 4],
    a: [u32; 4],
    a_low: [u32; 4],
    b: [u32; 2],
    b_low: [u32; 2],
    split_a: bool,
    split_b: bool,
) -> [f32; 4] {
    if !split_a && !split_b {
        return mma(acc, a, b);
    }

    let mut product = [0.0; 4];
    if split_a {
        product = mma(product, a_low, b);
    }
    if split_b {
        product = mma(product, a, b_low);
    }
    let product = mma(product, a, b);
    [
        acc[0] + product[0],
        acc[1] + product[1],
        acc[2] + product[2],
        acc[3] + product[3],
    ]
}

/// `pair` rounded to the nearest FP16 values, packed with `pair[0]` in the low half
/// as the `mma` fragments order them
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn f16x2(pair: [f32; 2]) -> u32 {
    let bits: u32;
    // safety: a register conversion with no memory access
    unsafe {
        ptx_asm!(
            "cvt.rn.f16x2.f32 %0, %1, %2;",
            out("=r") bits,
            in("f") pair[1],
            in("f") pair[0],
            options(register_only),
        );
    }
    bits
}

/// `acc + a b` for one warp-wide m16n8k16 FP16 tile with FP32 accumulation
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn mma16(acc: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = acc;
    // safety: a register-only warp instruction; every lane of the warp executes it
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b[0],
            in("r") b[1],
        );
    }
    [c0, c1, c2, c3]
}

/// Stores two floats at an 8-byte aligned global address
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn stg2(pointer: *mut f32, value: [f32; 2]) {
    // safety: the caller passes an aligned pointer to two writable floats that no
    // other thread writes
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.v2.f32 [g], {%1, %2}; }",
            in("l") pointer as u64,
            in("f") value[0],
            in("f") value[1],
            clobber("memory"),
        );
    }
}

/// `x` for non-negative values and NaN, `x * slope` otherwise, as the library epilogue
#[inline(always)]
fn leaky(value: f32, slope: f32) -> f32 {
    if value >= 0.0 { value } else { value * slope }
}

/// Expands to one valid five-tap NCW convolution without bias
///
/// `x` is `[b, cin, w_in]`, `weight` the packed `[cin][5][64]` weights with zero
/// channels `60..64`, and `y` is `[b, 60, w_in - 4]`. Launch `threads` threads with
/// `grid = (ceil((w_in - 4) / positions), b)`, where `positions` is
/// `8 * threads / (ksplit * 64 / channels)`
///
/// - `channels`: output channels per thread, 8 (`4g..4g + 4` and
///   `32 + 4g..32 + 4g + 4` for channel group `g`) or 4 (`4g..4g + 4`), so the
///   weight reads of an 8-lane phase cover 32 distinct banks
/// - `ksplit`: thread groups that split each stage's input channels; group 0 adds
///   the others' sums in group order
/// - `chunk`: input channels per pipeline stage, a multiple of `ksplit`
macro_rules! conv {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        cin = $cin:expr,
        w_in = $w_in:expr,
        channels = $channels:expr,
        ksplit = $ksplit:expr,
        chunk = $chunk:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
        pub fn $name(x: &[f32], weight: &[f32], mut y: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const CIN: u32 = $cin;
            const W_IN: u32 = $w_in;
            const W_OUT: u32 = W_IN - TAPS + 1;
            const CHANNELS: usize = $channels;
            const LOW: usize = if CHANNELS > 4 { 4 } else { CHANNELS };
            const HIGH: bool = CHANNELS == 8;
            const KSPLIT: u32 = $ksplit;
            const CHUNK: u32 = $chunk;
            const GROUP: u32 = THREADS / KSPLIT;
            const GROUPS: u32 = CONV_PADDED / CHANNELS as u32;
            const POSITIONS: u32 = GROUP / GROUPS * 8;
            const ROW: u32 = POSITIONS + 4;
            const IN_STAGE: u32 = CHUNK * ROW;
            const W_STAGE: u32 = CHUNK * TAPS * CONV_PADDED;
            const STAGE: u32 = IN_STAGE + W_STAGE;
            const STAGES: u32 = CIN / CHUNK;
            const PER_GROUP: u32 = CHUNK / KSPLIT;
            const IN_ITERS: usize = IN_STAGE.div_ceil(THREADS) as usize;
            const W_VECTORS: u32 = W_STAGE / 4;
            const W_ITERS: usize = W_VECTORS.div_ceil(THREADS) as usize;
            // the reduction stores every accumulator of groups 1.. once
            const RED: u32 = (KSPLIT - 1) * GROUP * 8 * CHANNELS as u32;
            // the epilogue stages two of each thread's channels at a time
            const EPI_ROW: u32 = POSITIONS + 4;
            const EPI_CHANNELS: u32 = 2 * GROUPS;
            const EPI: u32 = EPI_CHANNELS * EPI_ROW;
            const EPI_ITERS: usize = (EPI_CHANNELS * POSITIONS).div_ceil(THREADS) as usize;
            const PASSES: usize = CHANNELS / 2;
            const fn max(a: u32, b: u32) -> u32 {
                if a > b { a } else { b }
            }
            const SMEM: usize = max(max(2 * STAGE, RED), EPI) as usize;

            static mut SMEM_TILE: SharedArray<f32, SMEM, 16> = SharedArray::UNINIT;

            // a mismatched launch must not touch other memory
            let item = thread::blockIdx_y();
            let first = thread::blockIdx_x() * POSITIONS;
            let x_base = item * CIN * W_IN;
            let y_base = item * CONV_OUT * W_OUT;
            if (x_base + CIN * W_IN) as usize > x.len()
                || (y_base + CONV_OUT * W_OUT) as usize > y.len()
                || (CIN * TAPS * CONV_PADDED) as usize > weight.len()
                || first >= W_OUT
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let split = tid / GROUP;
            let local = tid % GROUP;
            let group = local % GROUPS;
            let position = local / GROUPS * 8;

            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const SMEM_TILE as *const u8) };

            // per-thread staging assignment, the same in every stage except for the
            // input channel offset
            let mut in_global = [0u32; IN_ITERS];
            let mut in_live = [false; IN_ITERS];
            let mut in_inside = [false; IN_ITERS];
            let mut i = 0;
            #[unroll]
            while i < IN_ITERS {
                let element = tid + THREADS * i as u32;
                let channel = element / ROW;
                let column = element % ROW;
                in_live[i] = element < IN_STAGE;
                in_inside[i] = element < IN_STAGE && first + column < W_IN;
                in_global[i] = x_base + channel * W_IN + first + column;
                i += 1;
            }

            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();
            let mut staged = [0.0f32; IN_ITERS];
            let mut staged_w = [[0.0f32; 4]; W_ITERS];
            let mut i = 0;
            #[unroll]
            while i < IN_ITERS {
                if in_inside[i] {
                    // safety: inside the item's input, which the length check covers
                    staged[i] = unsafe { ldg1(x_ptr.add(in_global[i] as usize)) };
                }
                i += 1;
            }
            let mut i = 0;
            #[unroll]
            while i < W_ITERS {
                let vector = tid + THREADS * i as u32;
                if vector < W_VECTORS {
                    // safety: inside the first stage of the packed weights
                    staged_w[i] = unsafe { ldg4(w_ptr.add((vector * 4) as usize)) };
                }
                i += 1;
            }

            let mut acc = [[0.0f32; CHANNELS]; 8];
            let in_offset = (split * PER_GROUP * ROW + position) * 4;
            let w_offset = (IN_STAGE + split * PER_GROUP * TAPS * CONV_PADDED + group * LOW as u32)
                * 4;

            let mut stage = 0;
            while stage < STAGES {
                // with two buffers one barrier per stage suffices: a thread writes
                // a buffer only after every thread passed the barrier that ended
                // the previous reads of it
                let buffer = smem + (stage % 2) * STAGE * 4;

                let mut i = 0;
                #[unroll]
                while i < IN_ITERS {
                    if in_live[i] {
                        let element = tid + THREADS * i as u32;
                        // safety: inside this buffer's input region, one slot per thread
                        unsafe { sts1(buffer + element * 4, staged[i]) };
                    }
                    i += 1;
                }
                let mut i = 0;
                #[unroll]
                while i < W_ITERS {
                    let vector = tid + THREADS * i as u32;
                    if vector < W_VECTORS {
                        // safety: inside this buffer's weight region, one vector per thread
                        unsafe { sts4(buffer + IN_STAGE * 4 + vector * 16, staged_w[i]) };
                    }
                    i += 1;
                }
                thread::sync_threads();

                // fetch the next stage while this one computes
                if stage + 1 < STAGES {
                    let channel_offset = (stage + 1) * CHUNK * W_IN;
                    let mut i = 0;
                    #[unroll]
                    while i < IN_ITERS {
                        staged[i] = if in_inside[i] {
                            // safety: as for the first stage
                            unsafe { ldg1(x_ptr.add((in_global[i] + channel_offset) as usize)) }
                        } else {
                            0.0
                        };
                        i += 1;
                    }
                    let w_stage = (stage + 1) * W_STAGE;
                    let mut i = 0;
                    #[unroll]
                    while i < W_ITERS {
                        let vector = tid + THREADS * i as u32;
                        if vector < W_VECTORS {
                            // safety: inside the next stage of the packed weights
                            staged_w[i] =
                                unsafe { ldg4(w_ptr.add((w_stage + vector * 4) as usize)) };
                        }
                        i += 1;
                    }
                }

                let in_base = opaque(buffer + in_offset);
                let w_base = opaque(buffer + w_offset);
                let mut channel = 0;
                #[unroll]
                while channel < PER_GROUP {
                    let row = in_base + channel * ROW * 4;
                    // safety: the 12-float window lies inside the staged row
                    let window = unsafe { [lds4(row), lds4(row + 16), lds4(row + 32)] };
                    let mut tap = 0;
                    #[unroll]
                    while tap < TAPS {
                        let address = w_base + (channel * TAPS + tap) * CONV_PADDED * 4;
                        // safety: inside this buffer's weight region, 16-byte aligned
                        let low = unsafe { lds4(address) };
                        let high = if HIGH {
                            // safety: as above
                            unsafe { lds4(address + CONV_PADDED * 2) }
                        } else {
                            [0.0; 4]
                        };
                        let mut p = 0;
                        #[unroll]
                        while p < 8 {
                            let at = p + tap as usize;
                            let value = window[at / 4][at % 4];
                            let mut c = 0;
                            #[unroll]
                            while c < LOW {
                                acc[p][c] = value.mul_add(low[c], acc[p][c]);
                                if HIGH {
                                    acc[p][c + 4] = value.mul_add(high[c], acc[p][c + 4]);
                                }
                                c += 1;
                            }
                            p += 1;
                        }
                        tap += 1;
                    }
                    channel += 1;
                }
                stage += 1;
            }

            // the last stage's readers must finish before the tile is reused
            thread::sync_threads();
            if KSPLIT > 1 {
                if split > 0 {
                    let mut p = 0;
                    #[unroll]
                    while p < 8 {
                        let mut c = 0;
                        #[unroll]
                        while c < CHANNELS {
                            let slot = ((split - 1) * 8 * CHANNELS as u32
                                + (p * CHANNELS + c) as u32)
                                * GROUP
                                + local;
                            // safety: one slot per (group, accumulator, thread)
                            unsafe { sts1(smem + slot * 4, acc[p][c]) };
                            c += 1;
                        }
                        p += 1;
                    }
                }
                thread::sync_threads();
                if split == 0 {
                    let mut other = 1;
                    while other < KSPLIT {
                        let mut p = 0;
                        #[unroll]
                        while p < 8 {
                            let mut c = 0;
                            #[unroll]
                            while c < CHANNELS {
                                let slot = ((other - 1) * 8 * CHANNELS as u32
                                    + (p * CHANNELS + c) as u32)
                                    * GROUP
                                    + local;
                                // safety: written before the barrier
                                acc[p][c] += unsafe { lds1(smem + slot * 4) };
                                c += 1;
                            }
                            p += 1;
                        }
                        other += 1;
                    }
                }
                thread::sync_threads();
            }

            // epilogue: one pass per two of each thread's channels, staged so that
            // every warp stores contiguous runs of one output row
            let epi_write = smem + position * 4;
            let mut pass = 0;
            #[unroll]
            while pass < PASSES {
                if split == 0 {
                    let mut half = 0;
                    #[unroll]
                    while half < 2 {
                        let slot = pass * 2 + half;
                        let address = epi_write + (group * 2 + half as u32) * EPI_ROW * 4;
                        // safety: inside the epilogue region; each (channel,
                        // position) belongs to one thread
                        unsafe {
                            sts4(
                                address,
                                [acc[0][slot], acc[1][slot], acc[2][slot], acc[3][slot]],
                            );
                            sts4(
                                address + 16,
                                [acc[4][slot], acc[5][slot], acc[6][slot], acc[7][slot]],
                            );
                        }
                        half += 1;
                    }
                }
                thread::sync_threads();

                let mut m = 0;
                #[unroll]
                while m < EPI_ITERS {
                    let e = tid + THREADS * m as u32;
                    let staged_channel = e / POSITIONS;
                    let column = e % POSITIONS;
                    let owner = staged_channel / 2;
                    let slot = pass as u32 * 2 + staged_channel % 2;
                    let channel = if slot < 4 {
                        owner * LOW as u32 + slot
                    } else {
                        CONV_PADDED / 2 + owner * 4 + slot - 4
                    };
                    let out = first + column;
                    if e < EPI_CHANNELS * POSITIONS && channel < CONV_OUT && out < W_OUT {
                        // safety: the epilogue slot was written before the barrier
                        let value = unsafe {
                            lds1(smem + (staged_channel * EPI_ROW + column) * 4)
                        };
                        let index = (y_base + channel * W_OUT + out) as usize;
                        // safety: inside the item's output; each element has one writer
                        unsafe { *y.get_unchecked_mut(index) = value };
                    }
                    m += 1;
                }
                thread::sync_threads();
                pass += 1;
            }
        }
    };
}

/// Expands to the given optional macro argument, or to the default without one
macro_rules! segdense_default {
    (, $default:expr) => {
        $default
    };
    ($value:expr, $default:expr) => {
        $value
    };
}

/// Expands to one row-major GEMM `a [m, k] x b [k, n]` with a fused epilogue
///
/// `b` is packed `[k, n]`. Launch `threads` threads with
/// `grid = (ceil(m / bm), n / bn, splits)`; `splits` is the number of reduction
/// slices over blocks, as equal as whole stages allow. Each block `z` of a
/// `Partial` kernel writes the partial plane `z` of `c [splits, m, n]`; the other
/// epilogues take `splits = 1`
///
/// - `tm`, `tn`: thread tile, 4 or 8; an 8 is the two 4-runs `4i..4i + 4` and
///   `b / 2 + 4i..b / 2 + 4i + 4`, so the shared reads of an 8-lane phase cover
///   32 distinct banks
/// - `bk`: reduction terms per group and stage
/// - `ksplit`: thread groups that split each stage's reduction terms; group 0 adds
///   the others' sums in group order
/// - `buffers`: shared stage buffers, 1 or 2; one buffer adds a barrier per stage
///   but allows stages twice as long
/// - `depth`: stages in flight from global memory, 1 or 2; small grids hide the
///   load latency only with more than one stage in flight
/// - `chains` (optional, default 1): accumulator sets per thread, 1 or 2; term
///   `kk` of a stage goes to set `kk % chains` and the sets are added once after
///   the main loop, which halves each rounding chain without more threads
/// - `compensated` (optional, default false): each stage's sums join a running
///   sum through `two_sum` with the rounding errors kept apart, and the groups of
///   `ksplit` join the same way, so only the stage chains of `bk / chains` terms
///   and the final rounding add error. Costs about six adds per accumulator and
///   stage
/// - `epilogue`: `Leaky` for `leaky(acc + bias, 0.01)`, `Bias` for `acc + bias`,
///   `Partial` for raw sums
macro_rules! gemm {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        m = $m:expr,
        n = $n:expr,
        k = $k:expr,
        bm = $bm:expr,
        bn = $bn:expr,
        bk = $bk:expr,
        tm = $tm:expr,
        tn = $tn:expr,
        ksplit = $ksplit:expr,
        buffers = $buffers:expr,
        depth = $depth:expr,
        $(chains = $chains:expr,)?
        $(compensated = $compensated:expr,)?
        epilogue = $epilogue:ident $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
        pub fn $name(a: &[f32], b: &[f32], bias: &[f32], mut c: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
            const BM: u32 = $bm;
            const BN: u32 = $bn;
            const BK: u32 = $bk;
            const TM: usize = $tm;
            const TN: usize = $tn;
            const KSPLIT: u32 = $ksplit;
            const BUFFERS: u32 = $buffers;
            const DEPTH: usize = $depth;
            const CHAINS: usize = segdense_default!($($chains)?, 1);
            const COMPENSATED: bool = segdense_default!($($compensated)?, false);
            const HAS_BIAS: bool = segdense_epilogue!(@bias $epilogue);
            const ROW_GROUPS: u32 = BM / TM as u32;
            const COL_GROUPS: u32 = BN / TN as u32;
            const GROUP: u32 = ROW_GROUPS * COL_GROUPS;
            const KT: u32 = BK * KSPLIT;
            // block `z` reduces stages `z * STEPS / splits..(z + 1) * STEPS / splits`
            const STEPS: u32 = K / KT;
            const A_ROW: u32 = BM + 4;
            const A_STAGE: u32 = KT * A_ROW;
            const B_STAGE: u32 = KT * BN;
            const STAGE: u32 = A_STAGE + B_STAGE;
            const A_VECTORS: u32 = BM * KT / 4;
            const A_ITERS: usize = A_VECTORS.div_ceil(THREADS) as usize;
            const B_VECTORS: u32 = KT * BN / 4;
            const B_ITERS: usize = B_VECTORS.div_ceil(THREADS) as usize;
            const QUADS: usize = TM * TN / 4;
            const RED: u32 = (KSPLIT - 1) * GROUP * (TM * TN) as u32;
            const fn max(a: u32, b: u32) -> u32 {
                if a > b { a } else { b }
            }
            const SMEM: usize = max(BUFFERS * STAGE, RED) as usize;
            const _: () = assert!(
                GROUP * KSPLIT == THREADS
                    && K % KT == 0
                    && N % BN == 0
                    && KT % 8 == 0
                    && BM % 8 == 0
                    && (DEPTH == 1 || DEPTH == 2)
                    && (CHAINS == 1 || CHAINS == 2)
                    && BK as usize % CHAINS == 0
            );

            static mut SMEM_TILE: SharedArray<f32, SMEM, 16> = SharedArray::UNINIT;

            let row0 = thread::blockIdx_x() * BM;
            let col0 = thread::blockIdx_y() * BN;
            let z = thread::blockIdx_z();
            let splits = thread::gridDim_z();
            // a mismatched launch must not touch other memory
            if (M * K) as usize > a.len()
                || (K * N) as usize > b.len()
                || splits as usize * (M * N) as usize > c.len()
                || (HAS_BIAS && (N as usize) > bias.len())
                || row0 >= M
                || col0 >= N
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let split = tid / GROUP;
            let local = tid % GROUP;
            let col_group = local % COL_GROUPS;
            let row_group = local / COL_GROUPS;

            // the epilogue's bias is fetched now, so its latency hides behind the
            // main loop instead of following it
            let mut bias_v = [[0.0f32; 4]; TN / 4];
            if HAS_BIAS {
                let mut h = 0;
                #[unroll]
                while h < TN / 4 {
                    let col = col0 + col_group * 4 + h as u32 * (BN / 2);
                    // safety: `col + 4 <= n`, which the length check covers
                    bias_v[h] = unsafe { ldg4(bias.as_ptr().add(col as usize)) };
                    h += 1;
                }
            }

            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const SMEM_TILE as *const u8) };

            // per-thread staging assignment: lane pairs read one 32-byte sector of
            // an `a` row and 16 consecutive rows share a warp, so the transposed
            // shared stores of a warp hit 32 distinct banks; `b` moves in 16-byte
            // runs of one row
            let first_step = z * STEPS / splits;
            let stages = (z + 1) * STEPS / splits - first_step;
            let k0 = first_step * KT;
            let mut a_global = [0u32; A_ITERS];
            let mut a_shared = [0u32; A_ITERS];
            let mut a_inside = [false; A_ITERS];
            let mut i = 0;
            #[unroll]
            while i < A_ITERS {
                let vector = tid + THREADS * i as u32;
                let pair = vector / 2;
                let row = pair % BM;
                let quad = pair / BM * 2 + vector % 2;
                a_inside[i] = vector < A_VECTORS && row0 + row < M;
                a_global[i] = (row0 + row) * K + k0 + quad * 4;
                a_shared[i] = (quad * 4 * A_ROW + row) * 4;
                i += 1;
            }

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let mut staged_a = [[[0.0f32; 4]; A_ITERS]; DEPTH];
            let mut staged_b = [[[0.0f32; 4]; B_ITERS]; DEPTH];
            let mut d = 0;
            #[unroll]
            while d < DEPTH {
                if (d as u32) < stages {
                    let next = d as u32 * KT;
                    let mut i = 0;
                    #[unroll]
                    while i < A_ITERS {
                        if a_inside[i] {
                            // safety: inside `a`, which the length check covers
                            staged_a[d][i] =
                                unsafe { ldg4(a_ptr.add((a_global[i] + next) as usize)) };
                        }
                        i += 1;
                    }
                    let mut i = 0;
                    #[unroll]
                    while i < B_ITERS {
                        let vector = tid + THREADS * i as u32;
                        if vector < B_VECTORS {
                            let index = (k0 + next + vector / (BN / 4)) * N
                                + col0
                                + vector % (BN / 4) * 4;
                            // safety: inside `b`, which the length check covers
                            staged_b[d][i] = unsafe { ldg4(b_ptr.add(index as usize)) };
                        }
                        i += 1;
                    }
                }
                d += 1;
            }

            let mut chain_acc = [[[0.0f32; TN]; TM]; CHAINS];
            // compensated sums: the running sum and its rounding errors kept apart
            let mut run_sum = [[0.0f32; TN]; TM];
            let mut run_err = [[0.0f32; TN]; TM];
            let a_offset = (split * BK * A_ROW + row_group * 4) * 4;
            let b_offset = (A_STAGE + split * BK * BN + col_group * 4) * 4;

            let mut base = 0;
            while base < stages {
                // stage `base + d` uses register set `d`, a constant after unrolling
                let mut d = 0;
                #[unroll]
                while d < DEPTH {
                    let stage = base + d as u32;
                    if stage < stages {
                        // with two buffers one barrier per stage suffices, as in
                        // `conv!`; one buffer must also wait for the previous
                        // stage's readers
                        if BUFFERS == 1 && stage > 0 {
                            thread::sync_threads();
                        }
                        let buffer = smem + (stage % BUFFERS) * STAGE * 4;
                        let mut i = 0;
                        #[unroll]
                        while i < A_ITERS {
                            let vector = tid + THREADS * i as u32;
                            if vector < A_VECTORS {
                                let mut j = 0;
                                #[unroll]
                                while j < 4 {
                                    // safety: inside this buffer's `a` region, one
                                    // slot per thread
                                    unsafe {
                                        sts1(
                                            buffer + a_shared[i] + j as u32 * A_ROW * 4,
                                            staged_a[d][i][j],
                                        )
                                    };
                                    j += 1;
                                }
                            }
                            i += 1;
                        }
                        let mut i = 0;
                        #[unroll]
                        while i < B_ITERS {
                            let vector = tid + THREADS * i as u32;
                            if vector < B_VECTORS {
                                // safety: inside this buffer's `b` region, one
                                // vector per thread
                                unsafe {
                                    sts4(buffer + A_STAGE * 4 + vector * 16, staged_b[d][i])
                                };
                            }
                            i += 1;
                        }
                        thread::sync_threads();

                        // refill this register set `DEPTH` stages ahead
                        if stage + (DEPTH as u32) < stages {
                            let next = (stage + DEPTH as u32) * KT;
                            let mut i = 0;
                            #[unroll]
                            while i < A_ITERS {
                                if a_inside[i] {
                                    // safety: as for the first stages
                                    staged_a[d][i] =
                                        unsafe { ldg4(a_ptr.add((a_global[i] + next) as usize)) };
                                }
                                i += 1;
                            }
                            let mut i = 0;
                            #[unroll]
                            while i < B_ITERS {
                                let vector = tid + THREADS * i as u32;
                                if vector < B_VECTORS {
                                    let index = (k0 + next + vector / (BN / 4)) * N
                                        + col0
                                        + vector % (BN / 4) * 4;
                                    // safety: as for the first stages
                                    staged_b[d][i] = unsafe { ldg4(b_ptr.add(index as usize)) };
                                }
                                i += 1;
                            }
                        }

                        let a_base = opaque(buffer + a_offset);
                        let b_base = opaque(buffer + b_offset);
                        let mut kk = 0;
                        #[unroll]
                        while kk < BK {
                            let a_row = a_base + kk * A_ROW * 4;
                            let b_row = b_base + kk * BN * 4;
                            // safety: inside this buffer's tiles, 16-byte aligned
                            let a_low = unsafe { lds4(a_row) };
                            let a_high = if TM == 8 {
                                // safety: as above
                                unsafe { lds4(a_row + BM * 2) }
                            } else {
                                [0.0; 4]
                            };
                            // safety: as above
                            let b_low = unsafe { lds4(b_row) };
                            let b_high = if TN == 8 {
                                // safety: as above
                                unsafe { lds4(b_row + BN * 2) }
                            } else {
                                [0.0; 4]
                            };
                            let mut r = 0;
                            #[unroll]
                            while r < TM {
                                let av = if r < 4 { a_low[r] } else { a_high[r - 4] };
                                let mut q = 0;
                                #[unroll]
                                while q < TN {
                                    let bv = if q < 4 { b_low[q] } else { b_high[q - 4] };
                                    let set = kk as usize % CHAINS;
                                    chain_acc[set][r][q] = av.mul_add(bv, chain_acc[set][r][q]);
                                    q += 1;
                                }
                                r += 1;
                            }
                            kk += 1;
                        }

                        if COMPENSATED {
                            let mut r = 0;
                            #[unroll]
                            while r < TM {
                                let mut q = 0;
                                #[unroll]
                                while q < TN {
                                    let mut stage_sum = chain_acc[0][r][q];
                                    if CHAINS == 2 {
                                        stage_sum += chain_acc[CHAINS - 1][r][q];
                                    }
                                    let (sum, err) = two_sum(run_sum[r][q], stage_sum);
                                    run_sum[r][q] = sum;
                                    run_err[r][q] += err;
                                    chain_acc[0][r][q] = 0.0;
                                    chain_acc[CHAINS - 1][r][q] = 0.0;
                                    q += 1;
                                }
                                r += 1;
                            }
                        }
                    }
                    d += 1;
                }
                base += DEPTH as u32;
            }

            let mut acc = chain_acc[0];
            if COMPENSATED {
                // group 0 of a split reduction keeps its error terms apart until
                // the other groups joined; the others hand over one rounded sum
                let mut r = 0;
                #[unroll]
                while r < TM {
                    let mut q = 0;
                    #[unroll]
                    while q < TN {
                        acc[r][q] = if KSPLIT > 1 && split == 0 {
                            run_sum[r][q]
                        } else {
                            run_sum[r][q] + run_err[r][q]
                        };
                        q += 1;
                    }
                    r += 1;
                }
            } else if CHAINS == 2 {
                let mut r = 0;
                #[unroll]
                while r < TM {
                    let mut q = 0;
                    #[unroll]
                    while q < TN {
                        acc[r][q] += chain_acc[CHAINS - 1][r][q];
                        q += 1;
                    }
                    r += 1;
                }
            }

            if KSPLIT > 1 {
                // the last stage's readers must finish before the tile is reused
                thread::sync_threads();
                // one 16-byte slot per (group, accumulator quad, thread), so a
                // warp's accesses are consecutive
                if split > 0 {
                    let mut v = 0;
                    #[unroll]
                    while v < QUADS {
                        let (r, h) = (v / (TN / 4), v % (TN / 4));
                        let slot = ((split - 1) * QUADS as u32 + v as u32) * GROUP + local;
                        let quad = [acc[r][h * 4], acc[r][h * 4 + 1], acc[r][h * 4 + 2], acc[r][h * 4 + 3]];
                        // safety: inside the reduction region, one slot per thread
                        unsafe { sts4(smem + slot * 16, quad) };
                        v += 1;
                    }
                }
                thread::sync_threads();
                if split != 0 {
                    return;
                }
                let mut other = 1;
                #[unroll]
                while other < KSPLIT {
                    let mut v = 0;
                    #[unroll]
                    while v < QUADS {
                        let (r, h) = (v / (TN / 4), v % (TN / 4));
                        let slot = ((other - 1) * QUADS as u32 + v as u32) * GROUP + local;
                        // safety: written before the barrier
                        let quad = unsafe { lds4(smem + slot * 16) };
                        let mut q = 0;
                        #[unroll]
                        while q < 4 {
                            let j = h * 4 + q;
                            if COMPENSATED {
                                let (sum, err) = two_sum(acc[r][j], quad[q]);
                                acc[r][j] = sum;
                                run_err[r][j] += err;
                            } else {
                                acc[r][j] += quad[q];
                            }
                            q += 1;
                        }
                        v += 1;
                    }
                    other += 1;
                }

                if COMPENSATED {
                    let mut r = 0;
                    #[unroll]
                    while r < TM {
                        let mut q = 0;
                        #[unroll]
                        while q < TN {
                            acc[r][q] += run_err[r][q];
                            q += 1;
                        }
                        r += 1;
                    }
                }
            }

            let c_ptr = c.as_mut_ptr();
            let mut r = 0;
            #[unroll]
            while r < TM {
                let row = row0
                    + if r < 4 {
                        row_group * 4 + r as u32
                    } else {
                        BM / 2 + row_group * 4 + r as u32 - 4
                    };
                let mut h = 0;
                #[unroll]
                while h < TN / 4 {
                    let col = col0 + col_group * 4 + h as u32 * (BN / 2);
                    if row < M {
                        let mut value = [0.0f32; 4];
                        let mut q = 0;
                        #[unroll]
                        while q < 4 {
                            value[q] =
                                segdense_epilogue!($epilogue, acc[r][h * 4 + q], bias_v[h][q]);
                            q += 1;
                        }
                        let index = (z * M + row) * N + col;
                        // safety: inside `c`; each element has one writer, and `n`
                        // and `col` are multiples of 4
                        unsafe { stg4(c_ptr.add(index as usize), value) };
                    }
                    h += 1;
                }
                r += 1;
            }
        }
    };
}

/// Expands to the sm80 TF32 tensor-core form of `gemm!` with the same operands,
/// grid and epilogues
///
/// Warps own `wm x wn` sub-tiles of the `bm x bn` block tile and run m16n8k8
/// `mma.sync` with both operands rounded to the nearest TF32 as they leave shared
/// memory, so the weights stay the FP32 values the SIMT form reads. `cp.async`
/// keeps `stages - 1` stages in flight in a ring of `stages` shared buffers
/// without a register round trip
///
/// - `bk`: reduction terms per warp group and stage, a multiple of 8
/// - `ksplit`: warp groups that split each stage's reduction terms; group 0 adds
///   the others' sums in group order
/// - `splits` (optional): as for `gemm!`
/// - `precision`: `Tf32` for TF32 mode, `Split` for 3xTF32 in FP32 mode, which
///   pays three tensor products per tile for FP32-level accuracy, `Half` for
///   2xTF32 in TF32 mode, which splits only `b` for two products per tile, and
///   `Select` for TF32 plus the 3xTF32 residual products of the 16 reduction rows
///   whose indices follow the `k * n` weights in `b` as `u32` bits: TF32 error
///   grows with the weight, so the rows with the largest weights, which the host
///   picks, carry the largest share of it
/// - `epilogue`, grid and split-K slices: as for `gemm!`
#[cfg(feature = "tier-sm80")]
macro_rules! mma_gemm {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        m = $m:expr,
        n = $n:expr,
        k = $k:expr,
        bm = $bm:expr,
        bn = $bn:expr,
        bk = $bk:expr,
        wm = $wm:expr,
        wn = $wn:expr,
        ksplit = $ksplit:expr,
        stages = $stages:expr,
        precision = $precision:ident,
        epilogue = $epilogue:ident $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
        pub fn $name(a: &[f32], b: &[f32], bias: &[f32], mut c: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
            const BM: u32 = $bm;
            const BN: u32 = $bn;
            const BK: u32 = $bk;
            const WM: u32 = $wm;
            const WN: u32 = $wn;
            const KSPLIT: u32 = $ksplit;
            const STAGES: u32 = $stages;
            const SPLIT: bool = segdense_precision!($precision);
            const SPLIT_B: bool = segdense_precision!(@b $precision);
            const SELECT: u32 = segdense_precision!(@select $precision);
            // k8 slots of selected rows per warp group; group `split` takes slots
            // `split * SLOTS..(split + 1) * SLOTS`
            const SLOTS: usize = (SELECT / 8 / KSPLIT) as usize;
            const HAS_BIAS: bool = segdense_epilogue!(@bias $epilogue);
            const MT: usize = (WM / 16) as usize;
            const NT: usize = (WN / 8) as usize;
            const WARPS_N: u32 = BN / WN;
            const GROUP: u32 = BM / WM * WARPS_N * 32;
            const KT: u32 = BK * KSPLIT;
            // block `z` reduces stages `z * STEPS / splits..(z + 1) * STEPS / splits`
            const STEPS: u32 = K / KT;
            // row strides of 4 and 8 floats past a multiple of 32 banks make the
            // fragment reads of a warp hit 32 distinct banks
            const A_ROW: u32 = KT + 4;
            const B_ROW: u32 = BN + 8;
            const A_STAGE: u32 = BM * A_ROW;
            const STAGE: u32 = A_STAGE + KT * B_ROW;
            const A_CHUNKS: u32 = BM * KT / 4;
            const A_ITERS: usize = A_CHUNKS.div_ceil(THREADS) as usize;
            const B_CHUNKS: u32 = KT * BN / 4;
            const B_ITERS: usize = B_CHUNKS.div_ceil(THREADS) as usize;
            const TILES: usize = MT * NT;
            const RED: u32 = (KSPLIT - 1) * GROUP * TILES as u32 * 4;
            const fn max(a: u32, b: u32) -> u32 {
                if a > b { a } else { b }
            }
            const SMEM: usize = max(STAGES * STAGE, RED) as usize;
            const _: () = assert!(
                GROUP * KSPLIT == THREADS
                    && BM % WM == 0
                    && BN % WN == 0
                    && WM % 16 == 0
                    && WN % 8 == 0
                    && BK % 8 == 0
                    && K % KT == 0
                    && N % BN == 0
                    && STAGES >= 2
                    && STAGES <= 5
                    && (A_ROW % 32 == 4 || A_ROW % 32 == 12 || A_ROW % 32 == 20 || A_ROW % 32 == 28)
                    && (B_ROW % 32 == 8 || B_ROW % 32 == 24)
                    && SELECT % (8 * KSPLIT) == 0
            );

            static mut SMEM_TILE: SharedArray<f32, SMEM, 16> = SharedArray::UNINIT;

            let row0 = thread::blockIdx_x() * BM;
            let col0 = thread::blockIdx_y() * BN;
            let z = thread::blockIdx_z();
            let splits = thread::gridDim_z();
            // a mismatched launch must not touch other memory
            if (M * K) as usize > a.len()
                || (K * N + SELECT) as usize > b.len()
                || splits as usize * (M * N) as usize > c.len()
                || (HAS_BIAS && (N as usize) > bias.len())
                || row0 >= M
                || col0 >= N
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp_index = tid / 32;
            let split = tid / GROUP;
            let local = tid % GROUP;
            let warp_row = warp_index % (GROUP / 32) / WARPS_N;
            let warp_col = warp_index % (GROUP / 32) % WARPS_N;
            let g = lane / 4;
            let t = lane % 4;

            // the epilogue's bias is fetched now, so its latency hides behind the
            // main loop instead of following it
            let mut bias_v = [[0.0f32; 2]; NT];
            if HAS_BIAS {
                let mut q = 0;
                #[unroll]
                while q < NT {
                    let col = col0 + warp_col * WN + q as u32 * 8 + t * 2;
                    // safety: `col + 2 <= n`, which the length check covers
                    bias_v[q] = unsafe {
                        [
                            ldg1(bias.as_ptr().add(col as usize)),
                            ldg1(bias.as_ptr().add(col as usize + 1)),
                        ]
                    };
                    q += 1;
                }
            }

            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const SMEM_TILE as *const u8) };

            // per-thread copy assignment, the same in every stage except for the
            // reduction offset; rows past `m` copy zeros from row `row0`, which
            // exists, so no address leaves `a`
            let first_step = z * STEPS / splits;
            let stages = (z + 1) * STEPS / splits - first_step;
            let k0 = first_step * KT;
            let mut a_global = [0u32; A_ITERS];
            let mut a_shared = [0u32; A_ITERS];
            let mut a_bytes = [0u32; A_ITERS];
            let mut i = 0;
            #[unroll]
            while i < A_ITERS {
                let chunk = tid + THREADS * i as u32;
                let row = chunk / (KT / 4);
                let quad = chunk % (KT / 4);
                let inside = row0 + row < M;
                let source = if inside { row0 + row } else { row0 };
                a_global[i] = source * K + k0 + quad * 4;
                a_shared[i] = (row * A_ROW + quad * 4) * 4;
                a_bytes[i] = if inside { 16 } else { 0 };
                i += 1;
            }

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let issue = |stage: u32| {
                let buffer = smem + (stage % STAGES) * STAGE * 4;
                let mut i = 0;
                #[unroll]
                while i < A_ITERS {
                    let chunk = tid + THREADS * i as u32;
                    if chunk < A_CHUNKS {
                        // safety: inside `a` and this buffer's `a` region, one
                        // chunk per thread
                        unsafe {
                            cp16(
                                buffer + a_shared[i],
                                a_ptr.add((a_global[i] + stage * KT) as usize),
                                a_bytes[i],
                            )
                        };
                    }
                    i += 1;
                }
                let mut i = 0;
                #[unroll]
                while i < B_ITERS {
                    let chunk = tid + THREADS * i as u32;
                    if chunk < B_CHUNKS {
                        let row = chunk / (BN / 4);
                        let quad = chunk % (BN / 4);
                        let index = (k0 + stage * KT + row) * N + col0 + quad * 4;
                        // safety: inside `b` and this buffer's `b` region, one
                        // chunk per thread
                        unsafe {
                            cp16(
                                buffer + (A_STAGE + row * B_ROW + quad * 4) * 4,
                                b_ptr.add(index as usize),
                                16,
                            )
                        };
                    }
                    i += 1;
                }
            };

            // every thread commits one group per stage slot, also an empty one, so
            // `cp_wait(STAGES - 2)` always means "the current stage has landed"
            let mut s = 0;
            #[unroll]
            while s < STAGES - 1 {
                if s < stages {
                    issue(s);
                }
                // safety: groups only this thread's copies
                unsafe { cp_commit() };
                s += 1;
            }

            let mut acc = [[[0.0f32; 4]; NT]; MT];
            let a_offset = ((warp_row * WM + g) * A_ROW + split * BK + t) * 4;
            let b_offset = (A_STAGE + (split * BK + t) * B_ROW + warp_col * WN + g) * 4;

            let mut stage = 0;
            while stage < stages {
                // safety: completes this thread's copies of `stage`
                unsafe { cp_wait(STAGES - 2) };
                // publishes every thread's copies, and ends every read of the
                // buffer that the next issue refills
                thread::sync_threads();
                if stage + STAGES - 1 < stages {
                    issue(stage + STAGES - 1);
                }
                // safety: as above
                unsafe { cp_commit() };

                let buffer = smem + (stage % STAGES) * STAGE * 4;
                let a_base = opaque(buffer + a_offset);
                let b_base = opaque(buffer + b_offset);
                let mut kk = 0;
                #[unroll]
                while kk < BK / 8 {
                    let mut af = [[0u32; 4]; MT];
                    let mut af_low = [[0u32; 4]; MT];
                    let mut r = 0;
                    #[unroll]
                    while r < MT {
                        let at = a_base + (r as u32 * 16 * A_ROW + kk * 8) * 4;
                        // safety: inside this buffer's `a` tile; the fragment map
                        // of m16n8k8 is rows `g`, `g + 8` and columns `t`, `t + 4`
                        let values = unsafe {
                            [
                                lds1(at),
                                lds1(at + 8 * A_ROW * 4),
                                lds1(at + 16),
                                lds1(at + (8 * A_ROW + 4) * 4),
                            ]
                        };
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            (af[r][e], af_low[r][e]) = split_tf32(values[e], SPLIT);
                            e += 1;
                        }
                        r += 1;
                    }
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let at = b_base + (kk * 8 * B_ROW + q as u32 * 8) * 4;
                        // safety: inside this buffer's `b` tile; rows `t`, `t + 4`
                        // and column `g` of the 8 x 8 fragment
                        let values = unsafe { [lds1(at), lds1(at + 4 * B_ROW * 4)] };
                        let (b0, b0_low) = split_tf32(values[0], SPLIT_B);
                        let (b1, b1_low) = split_tf32(values[1], SPLIT_B);
                        let mut r = 0;
                        #[unroll]
                        while r < MT {
                            acc[r][q] = mma_product(acc[r][q], af[r], af_low[r], [b0, b1], [b0_low, b1_low], SPLIT, SPLIT_B);
                            r += 1;
                        }
                        q += 1;
                    }
                    kk += 1;
                }
                stage += 1;
            }

            // `Select` operands are fetched once the main loop's registers are free,
            // one slot at a time: held through the loop, or all slots at once, they
            // made the batch-32 tile spill (168 registers, 340 and 256 bytes); slot
            // `j` holds the fragment of rows `select[8j + t]` and `select[8j + t + 4]`,
            // rows past `m` read row `row0` and count as zero. Their residual
            // products, `a_low b + a b_low`, are the part of the 3xTF32 product the
            // main loop's `a b` lacks; unlike `mma_product` they accumulate into
            // `acc` inside the `mma`: a zero-started product per tile kept 4 more
            // registers live per scheduled `mma` and spilled the accumulators
            let mut s = 0;
            #[unroll]
            while s < SLOTS {
                let slot = split as usize * SLOTS + s;
                // safety: `k * n + SELECT` is within `b`; an index past `k` is
                // clamped, so a corrupt list only miscomputes
                let listed = unsafe {
                    [
                        ldg1(b_ptr.add((K * N) as usize + slot * 8 + t as usize)).to_bits(),
                        ldg1(b_ptr.add((K * N) as usize + slot * 8 + t as usize + 4)).to_bits(),
                    ]
                };
                let rows = [
                    if listed[0] < K { listed[0] } else { K - 1 },
                    if listed[1] < K { listed[1] } else { K - 1 },
                ];
                let mut af = [[0u32; 4]; MT];
                let mut af_low = [[0u32; 4]; MT];
                let mut r = 0;
                #[unroll]
                while r < MT {
                    let mut e = 0;
                    #[unroll]
                    while e < 4 {
                        let row = row0 + warp_row * WM + r as u32 * 16 + g + (e as u32 % 2) * 8;
                        let source = if row < M { row } else { row0 };
                        // safety: `source < m` and the clamped index is below `k`
                        let value = unsafe { ldg1(a_ptr.add((source * K + rows[e / 2]) as usize)) };
                        (af[r][e], af_low[r][e]) = split_tf32(if row < M { value } else { 0.0 }, true);
                        e += 1;
                    }
                    r += 1;
                }
                let mut q = 0;
                #[unroll]
                while q < NT {
                    let col = col0 + warp_col * WN + q as u32 * 8 + g;
                    // safety: the clamped index is below `k` and `col < n`
                    let values = unsafe {
                        [
                            ldg1(b_ptr.add((rows[0] * N + col) as usize)),
                            ldg1(b_ptr.add((rows[1] * N + col) as usize)),
                        ]
                    };
                    let (b0, b0_low) = split_tf32(values[0], true);
                    let (b1, b1_low) = split_tf32(values[1], true);
                    let mut r = 0;
                    #[unroll]
                    while r < MT {
                        acc[r][q] = mma(mma(acc[r][q], af_low[r], [b0, b1]), af[r], [b0_low, b1_low]);
                        r += 1;
                    }
                    q += 1;
                }
                s += 1;
            }

            // no copy may still target the tile when it is reused, and the last
            // stage's readers must finish first
            // safety: waits for this thread's remaining, empty groups
            unsafe { cp_wait(0) };
            if KSPLIT > 1 {
                thread::sync_threads();
                // one 16-byte slot per (group, tile, thread), so a warp's accesses
                // are consecutive
                if split > 0 {
                    let mut v = 0;
                    #[unroll]
                    while v < TILES {
                        let slot = ((split - 1) * TILES as u32 + v as u32) * GROUP + local;
                        // safety: inside the reduction region, one slot per thread
                        unsafe { sts4(smem + slot * 16, acc[v / NT][v % NT]) };
                        v += 1;
                    }
                }
                thread::sync_threads();
                if split != 0 {
                    return;
                }
                let mut other = 1;
                #[unroll]
                while other < KSPLIT {
                    let mut v = 0;
                    #[unroll]
                    while v < TILES {
                        let slot = ((other - 1) * TILES as u32 + v as u32) * GROUP + local;
                        // safety: written before the barrier
                        let quad = unsafe { lds4(smem + slot * 16) };
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            acc[v / NT][v % NT][e] += quad[e];
                            e += 1;
                        }
                        v += 1;
                    }
                    other += 1;
                }
            }

            let c_ptr = c.as_mut_ptr();
            let mut r = 0;
            #[unroll]
            while r < MT {
                let mut half = 0;
                #[unroll]
                while half < 2 {
                    let row = row0 + warp_row * WM + r as u32 * 16 + g + half as u32 * 8;
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let col = col0 + warp_col * WN + q as u32 * 8 + t * 2;
                        if row < M {
                            let value = [
                                segdense_epilogue!(
                                    $epilogue,
                                    acc[r][q][half * 2],
                                    bias_v[q][0]
                                ),
                                segdense_epilogue!(
                                    $epilogue,
                                    acc[r][q][half * 2 + 1],
                                    bias_v[q][1]
                                ),
                            ];
                            let index = (z * M + row) * N + col;
                            // safety: inside `c`; each element has one writer, and
                            // `n` and `col` are even
                            unsafe { stg2(c_ptr.add(index as usize), value) };
                        }
                        q += 1;
                    }
                    half += 1;
                }
                r += 1;
            }
        }
    };
}

/// Expands to the sm80 tensor-core form of one valid five-tap NCW convolution
/// without bias: the implicit GEMM `y [64, positions] = w [64, k] x [k, positions]`
/// over the reduction index `k = ci * 5 + tap`, on m16n8k8 `mma.sync`
///
/// `x` is `[b, cin, w_in]`, `weight` the `spk_segdense_pack_conv_mma` fragments
/// with `planes = 1` for `Tf32` and 2 for `Split`, and `y` is `[b, 60, w_in - 4]`.
/// Each warp owns all 64 padded channels of `nt * 8` adjacent positions, so a
/// block covers `positions = threads / 32 * nt * 8`. A stage holds 8 input channels, which
/// is 40 reduction terms or five k-steps: their `positions + 4` input samples and
/// their packed weights. Input rows are not 16-byte aligned, so samples move in
/// 4-byte copies; weights move in 16-byte copies. Input samples are rounded (and
/// with `Split` divided into two TF32 terms) as they leave shared memory; weights
/// arrive rounded
///
/// Launch `threads` threads with `grid = (ceil((w_in - 4) / positions), b)` and
/// `4 * max(stages * (8 * (positions + 16) + 2560 * planes), 64 * (positions + 8))`
/// bytes of dynamic shared memory; a launch with less does nothing
#[cfg(feature = "tier-sm80")]
macro_rules! mma_conv {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        cin = $cin:expr,
        cin_pad = $cin_pad:expr,
        w_in = $w_in:expr,
        nt = $nt:expr,
        stages = $stages:expr,
        precision = $precision:ident $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
        pub fn $name(x: &[f32], weight: &[f32], mut y: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const WARPS: u32 = THREADS / 32;
            const CIN: u32 = $cin;
            const CIN_PAD: u32 = $cin_pad;
            const W_IN: u32 = $w_in;
            const W_OUT: u32 = W_IN - TAPS + 1;
            const NT: usize = $nt;
            const STAGES: u32 = $stages;
            const SPLIT: bool = segdense_precision!($precision);
            const _: () = assert!(
                SPLIT == segdense_precision!(@b $precision)
                    && segdense_precision!(@select $precision) == 0
            );
            const PLANES: u32 = if SPLIT { 2 } else { 1 };
            const NB: u32 = WARPS * NT as u32 * 8;
            // a row stride of 16 floats past a multiple of 32 banks keeps the `b`
            // fragment reads of a warp, which span at most two input channels of
            // 11 samples each, on distinct banks
            const ROW: u32 = NB + 16;
            const X_STAGE: u32 = 8 * ROW;
            const X_ELEMENTS: u32 = 8 * (NB + 4);
            const X_ITERS: u32 = X_ELEMENTS.div_ceil(THREADS);
            // one k-step of one plane: 4 channel tiles x 32 lanes x 4 registers
            const W_STEP: u32 = 512;
            const W_STAGE: u32 = 5 * W_STEP;
            const W_CHUNKS: u32 = PLANES * W_STAGE / 4;
            const W_ITERS: u32 = W_CHUNKS.div_ceil(THREADS);
            const STAGE: u32 = X_STAGE + PLANES * W_STAGE;
            const STEPS: u32 = CIN_PAD / 8;
            const PLANE: u32 = CIN_PAD * TAPS * CONV_PADDED;
            // the epilogue stages the block's outputs with a row stride of 8 floats
            // past a multiple of 32 banks, so the 8-byte stores of a half-warp
            // cover 32 distinct banks
            const EPI_ROW: u32 = NB + 8;
            const EPI: u32 = CONV_PADDED * EPI_ROW;
            const EPI_ITERS: u32 = (CONV_OUT * NB).div_ceil(THREADS);
            const fn max(a: u32, b: u32) -> u32 {
                if a > b { a } else { b }
            }
            const SMEM_BYTES: u32 = 4 * max(STAGES * STAGE, EPI);
            const _: () = assert!(
                THREADS % 32 == 0
                    && CIN_PAD % 8 == 0
                    && CIN_PAD >= CIN
                    && ROW % 32 == 16
                    && EPI_ROW % 32 == 8
                    && STAGES >= 2
                    && STAGES <= 4
                    && STEPS >= STAGES
            );

            let item = thread::blockIdx_y();
            let first = thread::blockIdx_x() * NB;
            let x_base = item * CIN * W_IN;
            let y_base = item * CONV_OUT * W_OUT;
            // a mismatched launch must not touch other memory
            if (x_base + CIN * W_IN) as usize > x.len()
                || (y_base + CONV_OUT * W_OUT) as usize > y.len()
                || (PLANES * PLANE) as usize > weight.len()
                || first >= W_OUT
                || dynamic_smem_bytes() < SMEM_BYTES
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp_index = tid / 32;
            let g = lane / 4;
            let t = lane % 4;
            // safety: the dynamic shared allocation is this kernel's tile; only its
            // address is taken
            let smem =
                unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<u8, 16>::get_raw()) };
            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();

            let issue = |stage: u32| {
                let buffer = smem + (stage % STAGES) * STAGE * 4;
                let mut i = 0;
                #[unroll]
                while i < X_ITERS {
                    let element = tid + THREADS * i;
                    if element < X_ELEMENTS {
                        let channel = element / (NB + 4);
                        let column = element % (NB + 4);
                        let ci = stage * 8 + channel;
                        let inside = ci < CIN && first + column < W_IN;
                        // samples outside the item copy zeros from its first one
                        let source = if inside { x_base + ci * W_IN + first + column } else { x_base };
                        // safety: inside the checked input and this buffer's input
                        // region, one sample per thread
                        unsafe {
                            cp4(
                                buffer + (channel * ROW + column) * 4,
                                x_ptr.add(source as usize),
                                if inside { 4 } else { 0 },
                            )
                        };
                    }
                    i += 1;
                }
                let mut i = 0;
                #[unroll]
                while i < W_ITERS {
                    let chunk = tid + THREADS * i;
                    if chunk < W_CHUNKS {
                        let plane = chunk / (W_STAGE / 4);
                        let within = chunk % (W_STAGE / 4) * 4;
                        let source = plane * PLANE + stage * W_STAGE + within;
                        // safety: inside the checked weight and this buffer's
                        // weight region, one chunk per thread
                        unsafe {
                            cp16(
                                buffer + (X_STAGE + plane * W_STAGE + within) * 4,
                                w_ptr.add(source as usize),
                                16,
                            )
                        };
                    }
                    i += 1;
                }
            };

            // every thread commits one group per stage slot, also an empty one, so
            // `cp_wait(STAGES - 2)` always means "the current stage has landed"
            let mut s = 0;
            #[unroll]
            while s < STAGES - 1 {
                issue(s);
                // safety: groups only this thread's copies
                unsafe { cp_commit() };
                s += 1;
            }

            let mut acc = [[[0.0f32; 4]; NT]; 4];
            let x_offset = (warp_index * NT as u32 * 8 + g) * 4;
            let w_offset = (X_STAGE + lane * 4) * 4;
            let mut stage = 0;
            while stage < STEPS {
                // safety: completes this thread's copies of `stage`
                unsafe { cp_wait(STAGES - 2) };
                // publishes every thread's copies, and ends every read of the
                // buffer that the next issue refills
                thread::sync_threads();
                if stage + STAGES - 1 < STEPS {
                    issue(stage + STAGES - 1);
                }
                // safety: as above
                unsafe { cp_commit() };

                let buffer = smem + (stage % STAGES) * STAGE * 4;
                let x_at = opaque(buffer + x_offset);
                let w_at = opaque(buffer + w_offset);
                let mut step = 0;
                #[unroll]
                while step < 5 {
                    // the input offsets of this lane's `b` rows `t` and `t + 4`; kept
                    // out of an array, which goes to local memory whenever the step
                    // loop is too large to unroll
                    let k = step as u32 * 8 + t;
                    let low = k / TAPS * ROW + k % TAPS;
                    let high = (k + 4) / TAPS * ROW + (k + 4) % TAPS;
                    let mut af = [[0u32; 4]; 4];
                    let mut af_low = [[0u32; 4]; 4];
                    let mut r = 0;
                    #[unroll]
                    while r < 4 {
                        let at = w_at + (step as u32 * W_STEP + r as u32 * 128) * 4;
                        // safety: inside this buffer's weight region, 16-byte aligned
                        let values = unsafe { lds4(at) };
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            af[r][e] = values[e].to_bits();
                            e += 1;
                        }
                        if SPLIT {
                            // safety: as above, in the residual plane
                            let values = unsafe { lds4(at + W_STAGE * 4) };
                            let mut e = 0;
                            #[unroll]
                            while e < 4 {
                                af_low[r][e] = values[e].to_bits();
                                e += 1;
                            }
                        }
                        r += 1;
                    }
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let column = q as u32 * 8 * 4;
                        // safety: inside this buffer's input region; row `t` or
                        // `t + 4` and column `g` of the 8 x 8 fragment
                        let values = unsafe {
                            [
                                lds1(x_at + low * 4 + column),
                                lds1(x_at + high * 4 + column),
                            ]
                        };
                        let (b0, b0_low) = split_tf32(values[0], SPLIT);
                        let (b1, b1_low) = split_tf32(values[1], SPLIT);
                        let mut r = 0;
                        #[unroll]
                        while r < 4 {
                            acc[r][q] = mma_product(
                                acc[r][q],
                                af[r],
                                af_low[r],
                                [b0, b1],
                                [b0_low, b1_low],
                                SPLIT,
                                SPLIT,
                            );
                            r += 1;
                        }
                        q += 1;
                    }
                    step += 1;
                }
                stage += 1;
            }

            // no copy may still target the tile when it is reused, and the last
            // stage's readers must finish first
            // safety: waits for this thread's remaining, empty groups
            unsafe { cp_wait(0) };
            thread::sync_threads();
            let mut r = 0;
            #[unroll]
            while r < 4 {
                let mut half = 0;
                #[unroll]
                while half < 2 {
                    let channel = r as u32 * 16 + g + half as u32 * 8;
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let column = warp_index * NT as u32 * 8 + q as u32 * 8 + t * 2;
                        // safety: inside the epilogue region; each (channel,
                        // position) belongs to one thread
                        unsafe {
                            sts2(
                                smem + (channel * EPI_ROW + column) * 4,
                                [acc[r][q][half * 2], acc[r][q][half * 2 + 1]],
                            )
                        };
                        q += 1;
                    }
                    half += 1;
                }
                r += 1;
            }
            thread::sync_threads();

            let mut m = 0;
            #[unroll]
            while m < EPI_ITERS {
                let e = tid + THREADS * m;
                let channel = e / NB;
                let column = e % NB;
                let out = first + column;
                if e < CONV_OUT * NB && out < W_OUT {
                    // safety: the epilogue slot was written before the barrier
                    let value = unsafe { lds1(smem + (channel * EPI_ROW + column) * 4) };
                    let index = (y_base + channel * W_OUT + out) as usize;
                    // safety: inside the item's output; each element has one writer
                    unsafe { *y.get_unchecked_mut(index) = value };
                }
                m += 1;
            }
        }
    };
}

/// Expands to `mma_conv!` in the sm80 tier, and in the sm75 tier to a kernel with
/// the same name and parameters that traps: every tier exports the same names, and
/// the host selects these kernels only when the sm80 tier or newer is loaded
macro_rules! tensor_conv {
    (
        $(#[$doc:meta])*
        $name:ident,
        $($rest:tt)*
    ) => {
        #[cfg(feature = "tier-sm80")]
        mma_conv! {
            $(#[$doc])*
            $name,
            $($rest)*
        }

        #[cfg(not(feature = "tier-sm80"))]
        $(#[$doc])*
        #[kernel]
        pub fn $name(x: &[f32], weight: &[f32], y: DisjointSlice<f32>) {
            let _ = (x, weight, y);
            // safety: aborts the launch; the sm75 tier has no tensor-core form
            unsafe { ptx_asm!("trap;") };
        }
    };
}

/// Expands to a batch-32 embedding split-K main kernel for TF32 mode on FP16
/// m16n8k16 tensor products with power-of-two scaling (sm80 tier; the sm75 tier
/// exports a trapping stub, which the host never selects)
///
/// FP16 keeps TF32's 11-bit significand in a 5-bit exponent, and consumer GPUs
/// run FP16 products with FP32 accumulation at twice their TF32 rate. Both operands
/// are therefore scaled by powers of two into the FP16 range before rounding: the
/// weights per column on the host, the activations per row and reduction slice
/// here. A power-of-two scale is exact, so the only rounding is to an 11-bit
/// significand, as in TF32 mode, except that values more than 2^28 below the
/// largest magnitude of their column or row slice become FP16 subnormals, whose
/// rounding error stays below 2^-39 of that largest magnitude
///
/// - `a` is `[m, k]` FP32 activations
/// - `b` is `[k / 2, n]` words, each the FP16 pair `(w[2p][col] s, w[2p + 1][col] s)`
///   with the low half first, followed by the `n` inverse column scales `1 / s`
///   as floats
/// - `c` is `[splits, m, n]` partial planes, which the embedding reduction adds
///
/// Block `(0, y, z)` owns the `m x bn` tile at column `y * bn` and reduction
/// stages `z * steps / splits..(z + 1) * steps / splits` of `bk * ksplit` terms
/// each. It first loads its whole `a` slice, finds each row's largest magnitude,
/// and keeps the slice in shared memory as scaled FP16 pairs, so `a` crosses
/// memory once and leaves FP32 once; then a `cp.async` ring of `stages` buffers
/// streams the FP16 weights. A slice may hold at most `slice` terms, so `splits`
/// must be at least `k / slice` rounded up to whole stages; a launch with fewer, or
/// with less dynamic shared memory than below, does nothing
///
/// Launch `threads` threads with `grid = (1, n / bn, splits)` and
/// `4 * (m4 + max(m * (slice / 2 + 4) + stages * bk * ksplit / 2 * (bn + 8),
/// (ksplit - 1) * threads / ksplit * wm * wn / 32))` bytes of dynamic shared
/// memory, `m4` being `m` rounded up to a multiple of 4
#[cfg(feature = "tier-sm80")]
macro_rules! mma_half_split {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        m = $m:expr,
        n = $n:expr,
        k = $k:expr,
        bn = $bn:expr,
        bk = $bk:expr,
        wm = $wm:expr,
        wn = $wn:expr,
        ksplit = $ksplit:expr,
        stages = $stages:expr,
        slice = $slice:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, 1)]
        pub fn $name(a: &[f32], b: &[f32], bias: &[f32], mut c: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const M: u32 = $m;
            const N: u32 = $n;
            const K: u32 = $k;
            const BN: u32 = $bn;
            const BK: u32 = $bk;
            const WM: u32 = $wm;
            const WN: u32 = $wn;
            const KSPLIT: u32 = $ksplit;
            const STAGES: u32 = $stages;
            const SLICE: u32 = $slice;
            const MT: usize = (WM / 16) as usize;
            const NT: usize = (WN / 8) as usize;
            const WARPS_N: u32 = BN / WN;
            const GROUP: u32 = M / WM * WARPS_N * 32;
            const KT: u32 = BK * KSPLIT;
            const STEPS: u32 = K / KT;
            // threads per activation row in the scaling pass, and the float quads
            // each of them loads
            const PARTS: u32 = THREADS / M;
            const QUADS: usize = (SLICE / 4 / PARTS) as usize;
            // word strides of 4 past a multiple of 32 banks make the `a` fragment
            // reads (rows `g`, pairs `t`) and of 8 the `b` ones (pair rows `t`,
            // columns `g`) hit 32 distinct banks
            const A_ROW: u32 = SLICE / 2 + 4;
            const B_ROW: u32 = BN + 8;
            const B_STAGE: u32 = KT / 2 * B_ROW;
            const B_CHUNKS: u32 = KT / 2 * BN / 4;
            const B_ITERS: usize = B_CHUNKS.div_ceil(THREADS) as usize;
            const TILES: usize = MT * NT;
            const RED: u32 = (KSPLIT - 1) * GROUP * TILES as u32 * 4;
            const fn max(a: u32, b: u32) -> u32 {
                if a > b { a } else { b }
            }
            // the row inverse scales, then the `a` slice and the `b` ring, which
            // the cross-group reduction reuses
            const INV: u32 = M.next_multiple_of(4);
            const RING: u32 = M * A_ROW;
            const SMEM: u32 = INV + max(RING + STAGES * B_STAGE, RED);
            const _: () = assert!(
                GROUP * KSPLIT == THREADS
                    && THREADS % M == 0
                    && PARTS.is_power_of_two()
                    && PARTS <= 32
                    && (SLICE / 4) % PARTS == 0
                    && SLICE % KT == 0
                    && M % WM == 0
                    && BN % WN == 0
                    && WM % 16 == 0
                    && WN % 8 == 0
                    && BK % 16 == 0
                    && K % KT == 0
                    && N % BN == 0
                    && STAGES >= 2
                    && STAGES <= 5
                    && A_ROW % 32 == 4
                    && (B_ROW % 32 == 8 || B_ROW % 32 == 24)
            );
            let _ = bias;

            let col0 = thread::blockIdx_y() * BN;
            let z = thread::blockIdx_z();
            let splits = thread::gridDim_z();
            let first_step = z * STEPS / splits;
            let stages = (z + 1) * STEPS / splits - first_step;
            let k0 = first_step * KT;
            // a mismatched launch must not touch other memory
            if (M * K) as usize > a.len()
                || (K / 2 * N + N) as usize > b.len()
                || splits as usize * (M * N) as usize > c.len()
                || thread::blockIdx_x() != 0
                || col0 >= N
                || stages * KT > SLICE
                || dynamic_smem_bytes() < SMEM * 4
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp_index = tid / 32;
            let split = tid / GROUP;
            let local = tid % GROUP;
            let warp_row = warp_index % (GROUP / 32) / WARPS_N;
            let warp_col = warp_index % (GROUP / 32) % WARPS_N;
            let g = lane / 4;
            let t = lane % 4;

            // safety: the dynamic allocation is this kernel's shared tile; only its
            // address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<u8, 16>::get_raw()) };
            let slice_base = smem + INV * 4;
            let ring = slice_base + RING * 4;

            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let issue = |stage: u32| {
                let buffer = ring + (stage % STAGES) * B_STAGE * 4;
                let mut i = 0;
                #[unroll]
                while i < B_ITERS {
                    let chunk = tid + THREADS * i as u32;
                    if chunk < B_CHUNKS {
                        let row = chunk / (BN / 4);
                        let quad = chunk % (BN / 4);
                        let index = ((k0 + stage * KT) / 2 + row) * N + col0 + quad * 4;
                        // safety: inside the weight words and this buffer, one
                        // chunk per thread
                        unsafe {
                            cp16(buffer + (row * B_ROW + quad * 4) * 4, b_ptr.add(index as usize), 16)
                        };
                    }
                    i += 1;
                }
            };

            // every thread commits one group per stage slot, also an empty one, so
            // `cp_wait(STAGES - 2)` always means "the current stage has landed"
            let mut s = 0;
            #[unroll]
            while s < STAGES - 1 {
                if s < stages {
                    issue(s);
                }
                // safety: groups only this thread's copies
                unsafe { cp_commit() };
                s += 1;
            }

            // the epilogue's column scales are fetched now, so their latency hides
            // behind the main loop
            let mut col_inv = [[0.0f32; 2]; NT];
            let mut q = 0;
            #[unroll]
            while q < NT {
                let col = col0 + warp_col * WN + q as u32 * 8 + t * 2;
                // safety: `col + 2 <= n`, inside the scales after the words
                col_inv[q] = unsafe {
                    [
                        ldg1(b_ptr.add((K / 2 * N + col) as usize)),
                        ldg1(b_ptr.add((K / 2 * N + col + 1) as usize)),
                    ]
                };
                q += 1;
            }

            // scaling pass: thread `(row, part)` loads quads `part, part + PARTS, ...`
            // of its row's slice, all before it waits on any
            let row = tid / PARTS;
            let part = tid % PARTS;
            let len = stages * KT;
            let mut quads = [[0.0f32; 4]; QUADS];
            let mut i = 0;
            #[unroll]
            while i < QUADS {
                let quad = part + PARTS * i as u32;
                if quad * 4 < len {
                    // safety: `row < m` and the slice lies inside the row
                    quads[i] = unsafe { ldg4(a_ptr.add((row * K + k0 + quad * 4) as usize)) };
                }
                i += 1;
            }
            let mut largest = 0.0f32;
            let mut i = 0;
            #[unroll]
            while i < QUADS {
                let mut e = 0;
                #[unroll]
                while e < 4 {
                    largest = largest.max(quads[i][e].abs());
                    e += 1;
                }
                i += 1;
            }
            let mut offset = 1;
            while offset < PARTS {
                largest = largest.max(warp::shuffle_xor_f32(largest, offset));
                offset *= 2;
            }
            let (scale, inverse) = half_scale(largest);
            if part == 0 {
                // safety: inside the inverse-scale region, one slot per row
                unsafe { sts1(smem + row * 4, inverse) };
            }
            let mut i = 0;
            #[unroll]
            while i < QUADS {
                let quad = part + PARTS * i as u32;
                if quad * 4 < len {
                    let v = quads[i];
                    let words = [f16x2([v[0] * scale, v[1] * scale]), f16x2([v[2] * scale, v[3] * scale])];
                    // safety: inside the slice region, one word pair per quad
                    unsafe {
                        sts2(
                            slice_base + (row * A_ROW + quad * 2) * 4,
                            [f32::from_bits(words[0]), f32::from_bits(words[1])],
                        )
                    };
                }
                i += 1;
            }

            let mut acc = [[[0.0f32; 4]; NT]; MT];
            let a_offset = ((warp_row * WM + g) * A_ROW + split * BK / 2 + t) * 4;
            let b_offset = ((split * BK / 2 + t) * B_ROW + warp_col * WN + g) * 4;

            let mut stage = 0;
            while stage < stages {
                // safety: completes this thread's copies of `stage`
                unsafe { cp_wait(STAGES - 2) };
                // publishes every thread's copies and, the first time, the scaled
                // slice, and ends every read of the buffer that the next issue refills
                thread::sync_threads();
                if stage + STAGES - 1 < stages {
                    issue(stage + STAGES - 1);
                }
                // safety: as above
                unsafe { cp_commit() };

                // opaque bases keep the fragment addresses as base plus constant
                // offsets
                let a_base = opaque(slice_base + a_offset + stage * KT / 2 * 4);
                let b_base = opaque(ring + (stage % STAGES) * B_STAGE * 4 + b_offset);
                let mut kk = 0;
                #[unroll]
                while kk < BK / 16 {
                    let mut af = [[0u32; 4]; MT];
                    let mut r = 0;
                    #[unroll]
                    while r < MT {
                        let at = a_base + (r as u32 * 16 * A_ROW + kk * 8) * 4;
                        // safety: inside the slice; the m16n8k16 fragment is rows
                        // `g`, `g + 8` and pairs `t`, `t + 4`
                        let words = unsafe {
                            [
                                lds1(at),
                                lds1(at + 8 * A_ROW * 4),
                                lds1(at + 16),
                                lds1(at + (8 * A_ROW + 4) * 4),
                            ]
                        };
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            af[r][e] = words[e].to_bits();
                            e += 1;
                        }
                        r += 1;
                    }
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let at = b_base + (kk * 8 * B_ROW + q as u32 * 8) * 4;
                        // safety: inside this buffer; pair rows `t`, `t + 4` and
                        // column `g` of the fragment
                        let words = unsafe { [lds1(at), lds1(at + 4 * B_ROW * 4)] };
                        let bf = [words[0].to_bits(), words[1].to_bits()];
                        let mut r = 0;
                        #[unroll]
                        while r < MT {
                            acc[r][q] = mma16(acc[r][q], af[r], bf);
                            r += 1;
                        }
                        q += 1;
                    }
                    kk += 1;
                }
                stage += 1;
            }

            // safety: waits for this thread's remaining, empty groups
            unsafe { cp_wait(0) };
            if KSPLIT > 1 {
                thread::sync_threads();
                // one 16-byte slot per (group, tile, thread) past the inverse
                // scales, so a warp's accesses are consecutive
                if split > 0 {
                    let mut v = 0;
                    #[unroll]
                    while v < TILES {
                        let slot = ((split - 1) * TILES as u32 + v as u32) * GROUP + local;
                        // safety: inside the reduction region, one slot per thread
                        unsafe { sts4(slice_base + slot * 16, acc[v / NT][v % NT]) };
                        v += 1;
                    }
                }
                thread::sync_threads();
                if split != 0 {
                    return;
                }
                let mut other = 1;
                #[unroll]
                while other < KSPLIT {
                    let mut v = 0;
                    #[unroll]
                    while v < TILES {
                        let slot = ((other - 1) * TILES as u32 + v as u32) * GROUP + local;
                        // safety: written before the barrier
                        let quad = unsafe { lds4(slice_base + slot * 16) };
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            acc[v / NT][v % NT][e] += quad[e];
                            e += 1;
                        }
                        v += 1;
                    }
                    other += 1;
                }
            }

            // undo both scales; a product of two powers of two is exact unless it
            // leaves the FP32 range, which the clamps in `half_scale` prevent
            let c_ptr = c.as_mut_ptr();
            let mut r = 0;
            #[unroll]
            while r < MT {
                let mut half = 0;
                #[unroll]
                while half < 2 {
                    let row = warp_row * WM + r as u32 * 16 + g + half as u32 * 8;
                    // safety: written by the scaling pass before the main loop's
                    // first barrier
                    let row_inv = unsafe { lds1(smem + row * 4) };
                    let mut q = 0;
                    #[unroll]
                    while q < NT {
                        let col = col0 + warp_col * WN + q as u32 * 8 + t * 2;
                        let value = [
                            acc[r][q][half * 2] * row_inv * col_inv[q][0],
                            acc[r][q][half * 2 + 1] * row_inv * col_inv[q][1],
                        ];
                        let index = (z * M + row) * N + col;
                        // safety: inside `c`; each element has one writer, and
                        // `n` and `col` are even
                        unsafe { stg2(c_ptr.add(index as usize), value) };
                        q += 1;
                    }
                    half += 1;
                }
                r += 1;
            }
        }
    };
}

/// Expands to `mma_half_split!` in the sm80 tier and to a trapping stub with the
/// same parameters in the sm75 tier
macro_rules! half_split {
    (
        $(#[$doc:meta])*
        $name:ident,
        $($rest:tt)*
    ) => {
        #[cfg(feature = "tier-sm80")]
        mma_half_split! {
            $(#[$doc])*
            $name,
            $($rest)*
        }

        #[cfg(not(feature = "tier-sm80"))]
        $(#[$doc])*
        #[kernel]
        pub fn $name(a: &[f32], b: &[f32], bias: &[f32], c: DisjointSlice<f32>) {
            let _ = (a, b, bias, c);
            // safety: aborts the launch; the sm75 tier has no FP16 tensor form
            unsafe { ptx_asm!("trap;") };
        }
    };
}

/// The power of two `s` that puts the largest magnitude `largest` of a row slice
/// in `[2^14, 2^15)`, below the FP16 maximum of 65504 even after rounding, and
/// `1 / s`
///
/// The exponent is clamped so both are normal FP32 numbers: a row whose largest
/// magnitude is below 2^-112 is scaled by 2^126 and keeps fewer FP16 bits, and an
/// infinite or NaN row keeps its infinities and NaNs
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn half_scale(largest: f32) -> (f32, f32) {
    let exponent = ((largest.to_bits() >> 23) & 0xff) as i32 - 127;
    let mut shift = 14 - exponent;
    if shift < -113 {
        shift = -113;
    }
    if shift > 126 {
        shift = 126;
    }
    (
        f32::from_bits(((shift + 127) as u32) << 23),
        f32::from_bits(((127 - shift) as u32) << 23),
    )
}

/// Expands to a GEMM that runs `mma_gemm!` in the sm80 tier and the SIMT `gemm!`
/// with the same grid, block and operands in the sm75 tier, which computes in full
/// FP32 for either precision
macro_rules! tf32_gemm {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        m = $m:expr,
        n = $n:expr,
        k = $k:expr,
        bm = $bm:expr,
        bn = $bn:expr,
        epilogue = $epilogue:ident,
        simt = {
            bk = $sbk:expr,
            tm = $tm:expr,
            tn = $tn:expr,
            ksplit = $sksplit:expr,
            buffers = $buffers:expr,
            depth = $depth:expr $(,)?
        },
        mma = {
            bk = $mbk:expr,
            wm = $wm:expr,
            wn = $wn:expr,
            ksplit = $mksplit:expr,
            stages = $stages:expr,
            precision = $precision:ident $(,)?
        } $(,)?
    ) => {
        #[cfg(not(feature = "tier-sm80"))]
        gemm! {
            $(#[$doc])*
            $name, threads = $threads, min_blocks = $min_blocks, m = $m, n = $n, k = $k,
            bm = $bm, bn = $bn, bk = $sbk, tm = $tm, tn = $tn, ksplit = $sksplit,
            buffers = $buffers, depth = $depth, epilogue = $epilogue,
        }
        #[cfg(feature = "tier-sm80")]
        mma_gemm! {
            $(#[$doc])*
            $name, threads = $threads, min_blocks = $min_blocks, m = $m, n = $n, k = $k,
            bm = $bm, bn = $bn, bk = $mbk, wm = $wm, wn = $wn, ksplit = $mksplit,
            stages = $stages, precision = $precision, epilogue = $epilogue,
        }
    };
}

/// Whether a tensor-core precision splits the `a` operand (or with `@b` the `b`
/// operand) into two TF32 terms, and with `@select` how many reduction rows it
/// recomputes in 3xTF32: `Tf32` rounds both operands once, `Split` is the 3xTF32
/// FP32 emulation, `Half` splits only `b`, `Select` is `Tf32` plus the 3xTF32
/// residual products of the 16 reduction rows listed after the weights
#[cfg(feature = "tier-sm80")]
macro_rules! segdense_precision {
    (Tf32) => {
        false
    };
    (Split) => {
        true
    };
    (Half) => {
        false
    };
    (Select) => {
        false
    };
    (@b Tf32) => {
        false
    };
    (@b Split) => {
        true
    };
    (@b Half) => {
        true
    };
    (@b Select) => {
        false
    };
    (@select Select) => {
        16
    };
    (@select $other:ident) => {
        0
    };
}

/// One epilogue element of `gemm!`, and whether the epilogue reads the bias
macro_rules! segdense_epilogue {
    (@bias Partial) => {
        false
    };
    (@bias $other:ident) => {
        true
    };
    (Leaky, $value:expr, $bias:expr) => {
        leaky($value + $bias, 0.01)
    };
    (Bias, $value:expr, $bias:expr) => {
        $value + $bias
    };
    (Partial, $value:expr, $bias:expr) => {
        $value
    };
}

/// Embedding `seg_1` output elements for batch 32: 96 rows of 256
const EMBED_ELEMENTS: u32 = 96 * 256;
/// Embedding `seg_1` output columns
const EMBED_COLS: u32 = 256;
/// Warps of a reduction block; warp `w` adds partial planes `w, w + 8, ...` of
/// the block's 128 outputs
const REDUCE_WARPS: u32 = 8;
/// Partial planes one reduction thread loads at most, all before it adds any, so
/// up to [`EMBED_MAX_SPLITS`] planes cost one round trip to memory rather than
/// one per batch of planes
const REDUCE_PLANES: u32 = 16;
/// Most split-K partial planes `spk_segdense_reduce_embed` adds
const EMBED_MAX_SPLITS: u32 = REDUCE_WARPS * REDUCE_PLANES;

/// Adds the `splits` split-K partial planes of the batch-32 embedding projection,
/// then the column bias
///
/// `partials` is `[splits, 96 * 256]`, `bias` `[256]` and `output` `[96 * 256]`,
/// with `1 <= splits <= EMBED_MAX_SPLITS`. Lane `l` of warp `w` adds planes
/// `w, w + 8, ...` of outputs `4l..4l + 4` of the block's 128 in plane order, and
/// warp 0 adds the eight warp sums in warp order, so the order is fixed for a
/// given `splits`. Launch 192 blocks of 256 threads
#[kernel]
#[launch_bounds(256, 4)]
pub fn spk_segdense_reduce_embed(
    partials: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    splits: u32,
) {
    const BLOCK: u32 = REDUCE_WARPS * 32;
    static mut SUMS: SharedArray<f32, { (REDUCE_WARPS * 32 * 4) as usize }, 16> =
        SharedArray::UNINIT;
    const _: () = assert!(EMBED_ELEMENTS % 128 == 0);

    let tid = thread::threadIdx_x();
    let lane = tid % 32;
    let warp_index = tid / 32;
    let first = (thread::blockIdx_x() * 32 + lane) * 4;
    // every condition is uniform over the block, since 128 divides the outputs,
    // so no thread skips the barrier below alone
    if thread::blockDim_x() != BLOCK
        || first >= EMBED_ELEMENTS
        || splits == 0
        || splits > EMBED_MAX_SPLITS
        || splits as usize * EMBED_ELEMENTS as usize > partials.len()
        || (EMBED_COLS as usize) > bias.len()
        || EMBED_ELEMENTS as usize > output.len()
    {
        return;
    }

    let p_ptr = partials.as_ptr();
    let mut planes = [[0.0f32; 4]; REDUCE_PLANES as usize];
    let mut j = 0;
    #[unroll]
    while j < REDUCE_PLANES {
        let plane = warp_index + j * REDUCE_WARPS;
        if plane < splits {
            let offset = plane as usize * EMBED_ELEMENTS as usize + first as usize;
            // safety: inside the checked partials
            planes[j as usize] = unsafe { ldg4(p_ptr.add(offset)) };
        }
        j += 1;
    }

    let mut total = [0.0f32; 4];
    let mut j = 0;
    #[unroll]
    while j < REDUCE_PLANES {
        if warp_index + j * REDUCE_WARPS < splits {
            let mut q = 0;
            #[unroll]
            while q < 4 {
                total[q] += planes[j as usize][q];
                q += 1;
            }
        }
        j += 1;
    }

    // safety: the static is this kernel's shared tile; only its address is taken
    let smem = unsafe { cvta_generic_to_shared_u32(&raw const SUMS as *const u8) };
    // safety: one 16-byte slot per thread
    unsafe { sts4(smem + tid * 16, total) };
    thread::sync_threads();
    if warp_index != 0 {
        return;
    }

    // fetched before the warp sums, so its latency overlaps them
    // safety: `first % 256 + 4 <= 256`, inside the checked bias
    let b = unsafe { ldg4(bias.as_ptr().add((first % EMBED_COLS) as usize)) };
    // safety: written before the barrier
    let mut total = unsafe { lds4(smem + lane * 16) };
    let mut w = 1;
    #[unroll]
    while w < REDUCE_WARPS {
        // safety: as above
        let sums = unsafe { lds4(smem + (w * 32 + lane) * 16) };
        let mut q = 0;
        #[unroll]
        while q < 4 {
            total[q] += sums[q];
            q += 1;
        }
        w += 1;
    }

    // the library adds the bias after the full product, as `beta = 1` does
    let mut q = 0;
    #[unroll]
    while q < 4 {
        total[q] += b[q];
        q += 1;
    }
    // safety: inside the checked output; one thread per four elements
    unsafe { stg4(output.as_mut_ptr().add(first as usize), total) };
}

/// Most split-K partial planes `spk_segdense_reduce_embed_flat` adds; its thread
/// holds all of them in registers
const FLAT_PLANES: u32 = 40;

/// [`spk_segdense_reduce_embed`] for at most [`FLAT_PLANES`] planes, with no
/// barrier: thread `i` loads every plane of outputs `4i..4i + 4` before it adds
/// them in plane order, then adds the bias. Launch `96 * 256 / 4` threads
#[kernel]
pub fn spk_segdense_reduce_embed_flat(
    partials: &[f32],
    bias: &[f32],
    mut output: DisjointSlice<f32>,
    splits: u32,
) {
    let first = thread::index_1d().get() as u32 * 4;
    if first >= EMBED_ELEMENTS
        || splits == 0
        || splits > FLAT_PLANES
        || splits as usize * EMBED_ELEMENTS as usize > partials.len()
        || (EMBED_COLS as usize) > bias.len()
        || EMBED_ELEMENTS as usize > output.len()
    {
        return;
    }

    let p_ptr = partials.as_ptr();
    let mut planes = [[0.0f32; 4]; FLAT_PLANES as usize];
    let mut s = 0;
    #[unroll]
    while s < FLAT_PLANES {
        if s < splits {
            let offset = s as usize * EMBED_ELEMENTS as usize + first as usize;
            // safety: inside the checked partials
            planes[s as usize] = unsafe { ldg4(p_ptr.add(offset)) };
        }
        s += 1;
    }
    // safety: `first % 256 + 4 <= 256`, inside the checked bias
    let b = unsafe { ldg4(bias.as_ptr().add((first % EMBED_COLS) as usize)) };

    let mut total = planes[0];
    let mut s = 1;
    #[unroll]
    while s < FLAT_PLANES {
        if s < splits {
            let mut q = 0;
            #[unroll]
            while q < 4 {
                total[q] += planes[s as usize][q];
                q += 1;
            }
        }
        s += 1;
    }
    // the library adds the bias after the full product, as `beta = 1` does
    let mut q = 0;
    #[unroll]
    while q < 4 {
        total[q] += b[q];
        q += 1;
    }
    // safety: inside the checked output; one thread per four elements
    unsafe { stg4(output.as_mut_ptr().add(first as usize), total) };
}

/// Expands to [`spk_segdense_reduce_embed_flat`] for exactly `splits` planes, which
/// the compiler fully unrolls with no per-plane test; a launch with another
/// `splits` does nothing. Launch `96 * 256 / 4` threads
macro_rules! reduce_fixed {
    (
        $(#[$doc:meta])*
        $name:ident,
        splits = $splits:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        pub fn $name(
            partials: &[f32],
            bias: &[f32],
            mut output: DisjointSlice<f32>,
            splits: u32,
        ) {
            const SPLITS: u32 = $splits;
            let first = thread::index_1d().get() as u32 * 4;
            if first >= EMBED_ELEMENTS
                || splits != SPLITS
                || SPLITS as usize * EMBED_ELEMENTS as usize > partials.len()
                || (EMBED_COLS as usize) > bias.len()
                || EMBED_ELEMENTS as usize > output.len()
            {
                return;
            }

            let p_ptr = partials.as_ptr();
            let mut planes = [[0.0f32; 4]; SPLITS as usize];
            let mut s = 0;
            #[unroll]
            while s < SPLITS {
                let offset = s as usize * EMBED_ELEMENTS as usize + first as usize;
                // safety: inside the checked partials
                planes[s as usize] = unsafe { ldg4(p_ptr.add(offset)) };
                s += 1;
            }
            // safety: `first % 256 + 4 <= 256`, inside the checked bias
            let b = unsafe { ldg4(bias.as_ptr().add((first % EMBED_COLS) as usize)) };

            let mut total = planes[0];
            let mut s = 1;
            #[unroll]
            while s < SPLITS {
                let mut q = 0;
                #[unroll]
                while q < 4 {
                    total[q] += planes[s as usize][q];
                    q += 1;
                }
                s += 1;
            }
            // the library adds the bias after the full product, as `beta = 1` does
            let mut q = 0;
            #[unroll]
            while q < 4 {
                total[q] += b[q];
                q += 1;
            }
            // safety: inside the checked output; one thread per four elements
            unsafe { stg4(output.as_mut_ptr().add(first as usize), total) };
        }
    };
}

reduce_fixed! {
    /// 18 partial planes plus the `seg_1` bias
    spk_segdense_reduce_e18,
    splits = 18,
}

reduce_fixed! {
    /// 34 partial planes plus the `seg_1` bias
    spk_segdense_reduce_e34,
    splits = 34,
}

reduce_fixed! {
    /// 36 partial planes plus the `seg_1` bias
    spk_segdense_reduce_e36,
    splits = 36,
}

/// Expands to the 128-to-7 classifier with its bias and log-softmax
///
/// `x` is `[m, 128]`, `weight` `[128, 7]`, `bias` `[7]` and `y` `[m, 7]`.
/// `lanes` threads share a row, so a warp covers `32 / lanes` rows per step. Lane
/// `s` of a row owns the reduction terms `4(q * lanes + s)..+4` for every `q`, and
/// keeps those weights in registers for all of its rows. Launch `threads` threads
/// with `grid = ceil(m / (rows * threads / lanes))`
///
/// - `compensated` (optional, default false): each 4-term vector sum starts from
///   zero and joins the lane's running sum through `two_sum`, the butterfly and the
///   bias join the same way, and the rounding errors are added once at the end, so
///   only the 4-term chains and the final rounding add error
macro_rules! classifier {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        m = $m:expr,
        lanes = $lanes:expr,
        rows = $rows:expr
        $(, compensated = $compensated:expr)? $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
        pub fn $name(x: &[f32], weight: &[f32], bias: &[f32], mut y: DisjointSlice<f32>) {
            const THREADS: u32 = $threads;
            const M: u32 = $m;
            const LANES: u32 = $lanes;
            const ROWS: u32 = $rows;
            const K: u32 = 128;
            const CLASSES: usize = 7;
            const VECTORS: usize = (K / 4 / LANES) as usize;
            const ROWS_PER_STEP: u32 = THREADS / LANES;
            const LEVELS: u32 = LANES.trailing_zeros();
            const COMPENSATED: bool = segdense_default!($($compensated)?, false);

            if (M * K) as usize > x.len()
                || K as usize * CLASSES > weight.len()
                || CLASSES > bias.len()
                || (M as usize) * CLASSES > y.len()
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let sub = tid % LANES;
            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();
            let mut w = [[[0.0f32; CLASSES]; 4]; VECTORS];
            let mut q = 0;
            #[unroll]
            while q < VECTORS {
                let mut j = 0;
                #[unroll]
                while j < 4 {
                    let k = (q as u32 * LANES + sub) * 4 + j as u32;
                    let mut class = 0;
                    #[unroll]
                    while class < CLASSES {
                        // safety: `k < 128`, inside the checked weight
                        w[q][j][class] =
                            unsafe { ldg1(w_ptr.add((k as usize) * CLASSES + class)) };
                        class += 1;
                    }
                    j += 1;
                }
                q += 1;
            }
            let mut b = [0.0f32; CLASSES];
            let mut class = 0;
            #[unroll]
            while class < CLASSES {
                // safety: inside the checked bias
                b[class] = unsafe { ldg1(bias.as_ptr().add(class)) };
                class += 1;
            }

            let first = thread::blockIdx_x() * ROWS * ROWS_PER_STEP + tid / LANES;
            let mut values = [[0.0f32; 4]; VECTORS];
            let mut q = 0;
            #[unroll]
            while q < VECTORS {
                if first < M {
                    let k = (q as u32 * LANES + sub) * 4;
                    // safety: inside the checked input row
                    values[q] = unsafe { ldg4(x_ptr.add((first * K + k) as usize)) };
                }
                q += 1;
            }

            let mut step = 0;
            #[unroll]
            while step < ROWS {
                let row = first + step * ROWS_PER_STEP;
                let current = values;
                // fetch the next row while this one reduces
                let next = row + ROWS_PER_STEP;
                if step + 1 < ROWS {
                    let mut q = 0;
                    #[unroll]
                    while q < VECTORS {
                        if next < M {
                            let k = (q as u32 * LANES + sub) * 4;
                            // safety: inside the checked input row
                            values[q] = unsafe { ldg4(x_ptr.add((next * K + k) as usize)) };
                        }
                        q += 1;
                    }
                }

                let mut sums = [0.0f32; CLASSES];
                let mut errors = [0.0f32; CLASSES];
                let mut q = 0;
                #[unroll]
                while q < VECTORS {
                    let mut part = if COMPENSATED { [0.0f32; CLASSES] } else { sums };
                    let mut j = 0;
                    #[unroll]
                    while j < 4 {
                        let mut class = 0;
                        #[unroll]
                        while class < CLASSES {
                            part[class] = current[q][j].mul_add(w[q][j][class], part[class]);
                            class += 1;
                        }
                        j += 1;
                    }
                    let mut class = 0;
                    #[unroll]
                    while class < CLASSES {
                        if COMPENSATED {
                            let (sum, error) = two_sum(sums[class], part[class]);
                            sums[class] = sum;
                            errors[class] += error;
                        } else {
                            sums[class] = part[class];
                        }
                        class += 1;
                    }
                    q += 1;
                }

                // every lane of the warp joins the butterfly, also for rows past `m`
                let mut class = 0;
                #[unroll]
                while class < CLASSES {
                    let mut level = 0;
                    #[unroll]
                    while level < LEVELS {
                        let other = warp::shuffle_xor_f32(sums[class], 1 << level);
                        if COMPENSATED {
                            // both lanes of a pair add the errors in the same order, so
                            // every lane of a row holds the same logits
                            let other_error = warp::shuffle_xor_f32(errors[class], 1 << level);
                            let (sum, error) = two_sum(sums[class], other);
                            sums[class] = sum;
                            errors[class] = (errors[class] + other_error) + error;
                        } else {
                            sums[class] += other;
                        }
                        level += 1;
                    }
                    if COMPENSATED {
                        let (sum, error) = two_sum(sums[class], b[class]);
                        sums[class] = sum + (error + errors[class]);
                    } else {
                        sums[class] += b[class];
                    }
                    class += 1;
                }

                let mut largest = f32::NEG_INFINITY;
                let mut class = 0;
                #[unroll]
                while class < CLASSES {
                    largest = largest.max(sums[class]);
                    class += 1;
                }
                let mut total = 0.0f32;
                let mut class = 0;
                #[unroll]
                while class < CLASSES {
                    total += (sums[class] - largest).exp();
                    class += 1;
                }
                let log = total.ln();
                let mut mine = 0.0f32;
                let mut class = 0;
                #[unroll]
                while class < CLASSES {
                    if sub == class as u32 {
                        mine = sums[class] - largest - log;
                    }
                    class += 1;
                }
                if row < M && sub < CLASSES as u32 {
                    // safety: inside the checked output; lane `sub` owns class `sub`
                    unsafe { *y.get_unchecked_mut((row * CLASSES as u32 + sub) as usize) = mine };
                }
                step += 1;
            }
        }
    };
}

/// Expands to the batch-1 embedding projection `y = x weight^T + bias`
///
/// `x` is `[3, 5120]`, `weight` the unpacked `[256, 5120]`, `bias` `[256]` and `y`
/// `[3, 256]`. A block owns `cols` output columns; its warp `w` reduces the terms
/// `w * 5120 / warps..` with 16-byte loads, and lane sums meet in warp order
/// through shared memory. Launch `32 * warps` threads with `grid = 256 / cols`
macro_rules! embed_gemv {
    (
        $(#[$doc:meta])*
        $name:ident,
        warps = $warps:expr,
        cols = $cols:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        pub fn $name(x: &[f32], weight: &[f32], bias: &[f32], mut y: DisjointSlice<f32>) {
            const WARPS: u32 = $warps;
            const COLS: usize = $cols;
            const ROWS: usize = 3;
            const K: u32 = 5120;
            const N: u32 = 256;
            const SLICE: u32 = K / WARPS;
            const STEPS: u32 = SLICE / 128;
            const VALUES: usize = ROWS * COLS;
            const _: () = assert!(SLICE % 128 == 0 && N as usize % COLS == 0);

            static mut PARTIAL: SharedArray<f32, { 32 * 3 * 8 }, 16> = SharedArray::UNINIT;
            const _: () = assert!(WARPS as usize * VALUES <= 32 * 3 * 8);

            let col0 = thread::blockIdx_x() * COLS as u32;
            if (ROWS as u32 * K) as usize > x.len()
                || (N * K) as usize > weight.len()
                || N as usize > bias.len()
                || ROWS * N as usize > y.len()
                || col0 >= N
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp_index = tid / 32;
            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();
            let mut sums = [[0.0f32; COLS]; ROWS];
            let start = warp_index * SLICE + lane * 4;
            let mut step = 0;
            #[unroll]
            while step < STEPS {
                let k = start + step * 128;
                let mut xv = [[0.0f32; 4]; ROWS];
                let mut r = 0;
                #[unroll]
                while r < ROWS {
                    // safety: inside the checked input
                    xv[r] = unsafe { ldg4(x_ptr.add((r as u32 * K + k) as usize)) };
                    r += 1;
                }
                let mut col = 0;
                #[unroll]
                while col < COLS {
                    // safety: inside the checked weight
                    let wv = unsafe { ldg4(w_ptr.add(((col0 + col as u32) * K + k) as usize)) };
                    let mut r = 0;
                    #[unroll]
                    while r < ROWS {
                        let mut j = 0;
                        #[unroll]
                        while j < 4 {
                            sums[r][col] = xv[r][j].mul_add(wv[j], sums[r][col]);
                            j += 1;
                        }
                        r += 1;
                    }
                    col += 1;
                }
                step += 1;
            }

            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const PARTIAL as *const u8) };
            let mut r = 0;
            #[unroll]
            while r < ROWS {
                let mut col = 0;
                #[unroll]
                while col < COLS {
                    let mut value = sums[r][col];
                    let mut level = 0;
                    #[unroll]
                    while level < 5 {
                        value += warp::shuffle_xor_f32(value, 16 >> level);
                        level += 1;
                    }
                    if lane == 0 {
                        let slot = warp_index * VALUES as u32 + (r * COLS + col) as u32;
                        // safety: one slot per (warp, value)
                        unsafe { sts1(smem + slot * 4, value) };
                    }
                    col += 1;
                }
                r += 1;
            }
            thread::sync_threads();

            if tid < VALUES as u32 {
                let mut total = 0.0f32;
                let mut w = 0;
                while w < WARPS {
                    // safety: written before the barrier
                    total += unsafe { lds1(smem + (w * VALUES as u32 + tid) * 4) };
                    w += 1;
                }
                let r = tid / COLS as u32;
                let col = col0 + tid % COLS as u32;
                // safety: inside the checked bias
                let value = total + unsafe { *bias.as_ptr().add(col as usize) };
                // safety: inside the checked output; one thread per element
                unsafe { *y.get_unchecked_mut((r * N + col) as usize) = value };
            }
        }
    };
}

conv! {
    /// segmentation `conv1d.1` for batch 32: 80 to 60 channels over 5325 samples
    spk_segdense_conv1_b32,
    threads = 128,
    min_blocks = 4,
    cin = 80,
    w_in = 5325,
    channels = 8,
    ksplit = 1,
    chunk = 4,
}

conv! {
    /// segmentation `conv1d.1` for batch 1
    spk_segdense_conv1_b1,
    threads = 256,
    min_blocks = 2,
    cin = 80,
    w_in = 5325,
    channels = 4,
    ksplit = 2,
    chunk = 4,
}

conv! {
    /// segmentation `conv1d.2` for batch 32: 60 to 60 channels over 1773 samples
    spk_segdense_conv2_b32,
    threads = 128,
    min_blocks = 4,
    cin = 60,
    w_in = 1773,
    channels = 8,
    ksplit = 1,
    chunk = 4,
}

conv! {
    /// segmentation `conv1d.2` for batch 1
    spk_segdense_conv2_b1,
    threads = 256,
    min_blocks = 2,
    cin = 60,
    w_in = 1773,
    channels = 4,
    ksplit = 4,
    chunk = 4,
}

gemm! {
    /// segmentation `linear.0` with LeakyReLU for batch 32: 18848 x 256 to 128
    spk_segdense_linear0_b32,
    threads = 128,
    min_blocks = 3,
    m = 18848,
    n = 128,
    k = 256,
    bm = 64,
    bn = 128,
    bk = 16,
    tm = 8,
    tn = 8,
    ksplit = 1,
    buffers = 2,
    depth = 1,
    epilogue = Leaky,
}

gemm! {
    /// segmentation `linear.0` with LeakyReLU for batch 1: 589 x 256 to 128
    ///
    /// Two accumulator sets keep each rounding chain at 32 terms: with one set
    /// the 64-term chains measured relative L2 1.142e-7 against an f64 truth,
    /// above 1.10 x the A100 cuBLAS FP32 error (9.554e-8); two sets give
    /// 9.252e-8 for 4% more time on Ada
    spk_segdense_linear0_b1,
    threads = 128,
    min_blocks = 4,
    m = 589,
    n = 128,
    k = 256,
    bm = 16,
    bn = 32,
    bk = 8,
    tm = 4,
    tn = 4,
    ksplit = 4,
    buffers = 2,
    depth = 2,
    chains = 2,
    epilogue = Leaky,
}

gemm! {
    /// segmentation `linear.1` with LeakyReLU for batch 32: 18848 x 128 to 128
    spk_segdense_linear1_b32,
    threads = 128,
    min_blocks = 3,
    m = 18848,
    n = 128,
    k = 128,
    bm = 64,
    bn = 128,
    bk = 16,
    tm = 8,
    tn = 8,
    ksplit = 1,
    buffers = 2,
    depth = 1,
    epilogue = Leaky,
}

gemm! {
    /// segmentation `linear.1` with LeakyReLU for batch 1: 589 x 128 to 128
    ///
    /// Compensated stage sums: the plain kernel measured relative L2 1.335e-7
    /// against an f64 truth, 1.01 x the Ada and A100 cuBLAS FP32 error, which puts
    /// the layer's geometric mean over 1.00, and two accumulator sets made it worse
    /// (1.543e-7); compensation gives 7.359e-8 for 25% more time on Ada. TF32 mode
    /// runs `spk_segdense_linear1_b1_tf32` instead, whose sm75 expansion is the
    /// plain kernel
    spk_segdense_linear1_b1,
    threads = 128,
    min_blocks = 4,
    m = 589,
    n = 128,
    k = 128,
    bm = 16,
    bn = 32,
    bk = 8,
    tm = 4,
    tn = 4,
    ksplit = 4,
    buffers = 2,
    depth = 2,
    compensated = true,
    epilogue = Leaky,
}

gemm! {
    /// embedding `seg_1` split-K partials for batch 32: 96 x 5120 to 256 in 96 x 128
    /// tiles; `b` is the weight packed to `[5120, 256]`
    spk_segdense_embed_b32,
    threads = 192,
    min_blocks = 2,
    m = 96,
    n = 256,
    k = 5120,
    bm = 96,
    bn = 128,
    bk = 16,
    tm = 8,
    tn = 8,
    ksplit = 1,
    buffers = 2,
    depth = 1,
    epilogue = Partial,
}

classifier! {
    /// segmentation `classifier` with log-softmax for batch 32: 18848 rows; one
    /// block per SM of register budget, since two blocks spill the weights.
    /// Compensated, because the A100's cuBLAS FP32 GEMM rounds less than the plain
    /// chains at large-magnitude log-probabilities (max-abs 3.2e-6 against 4.0e-6)
    spk_segdense_classifier_b32,
    threads = 256,
    min_blocks = 1,
    m = 18848,
    lanes = 8,
    rows = 4,
    compensated = true,
}

classifier! {
    /// segmentation `classifier` with log-softmax for batch 1: 589 rows
    spk_segdense_classifier_b1,
    threads = 64,
    min_blocks = 4,
    m = 589,
    lanes = 8,
    rows = 1,
}

embed_gemv! {
    /// embedding `seg_1` for batch 1: three mask rows of 5120 to 256
    spk_segdense_embed_b1,
    warps = 8,
    cols = 2,
}

// tensor-core convolutions for batch 32, selected on GPUs whose TF32 tensor rate is
// several times their FP32 rate: TF32 for TF32 mode and 3xTF32 for FP32 mode

tensor_conv! {
    /// segmentation `conv1d.1` for batch 32 in TF32 mode on tensor cores
    spk_segdense_conv1_b32_tc, threads = 128, min_blocks = 2, cin = 80, cin_pad = 80, w_in = 5325,
    nt = 4, stages = 3, precision = Tf32,
}

tensor_conv! {
    /// segmentation `conv1d.1` for batch 32 in FP32 mode on tensor cores, as 3xTF32;
    /// eight warps of two position tiles, because the unrolled body of a warp's four
    /// split products per channel tile stays under cuda-oxide's clone limit only
    /// for two tiles, and a rolled tile loop moves the accumulators to local memory.
    /// One resident block per SM: the 128-register cap of two spills the split
    /// fragments
    spk_segdense_conv1_b32_x3, threads = 256, min_blocks = 1, cin = 80, cin_pad = 80, w_in = 5325,
    nt = 2, stages = 3, precision = Split,
}

tensor_conv! {
    /// segmentation `conv1d.2` for batch 32 in TF32 mode on tensor cores; channels
    /// 60..64 are zero padding
    spk_segdense_conv2_b32_tc, threads = 128, min_blocks = 2, cin = 60, cin_pad = 64, w_in = 1773,
    nt = 4, stages = 3, precision = Tf32,
}

tensor_conv! {
    /// segmentation `conv1d.2` for batch 32 in FP32 mode on tensor cores, as 3xTF32,
    /// with the warp shape and residency of `spk_segdense_conv1_b32_x3`
    spk_segdense_conv2_b32_x3, threads = 256, min_blocks = 1, cin = 60, cin_pad = 64, w_in = 1773,
    nt = 2, stages = 3, precision = Split,
}

// TF32-mode kernels: tensor-core `mma.sync` in the sm80 tier. The host selects them
// only for TF32 math on the sm80 tier or newer; their sm75 expansions are the SIMT
// kernels with the same grid, which exist because every tier exports the same names

tf32_gemm! {
    /// segmentation `linear.0` with LeakyReLU for batch 1 in TF32 mode
    spk_segdense_linear0_b1_tf32, threads = 128, min_blocks = 4, m = 589, n = 128, k = 256,
    bm = 32, bn = 32, epilogue = Leaky,
    simt = { bk = 8, tm = 4, tn = 4, ksplit = 2, buffers = 2, depth = 2 },
    mma = { bk = 8, wm = 16, wn = 32, ksplit = 2, stages = 4, precision = Tf32 },
}

tf32_gemm! {
    /// segmentation `linear.1` with LeakyReLU for batch 1 in TF32 mode; four warps
    /// split the reduction. 2xTF32 keeps the weights exact: plain TF32 differs from
    /// cuBLAS TF32 only in summation order and tied its error to 0.01%, while the
    /// split measured 0.78 x the library's relative L2 for 8% more time on Ada
    spk_segdense_linear1_b1_tf32, threads = 128, min_blocks = 4, m = 589, n = 128, k = 128,
    bm = 16, bn = 32, epilogue = Leaky,
    simt = { bk = 8, tm = 4, tn = 4, ksplit = 4, buffers = 2, depth = 2 },
    mma = { bk = 8, wm = 16, wn = 32, ksplit = 4, stages = 3, precision = Half },
}

tf32_gemm! {
    /// segmentation `linear.0` with LeakyReLU for batch 32 in TF32 mode, with the 16
    /// reduction rows of largest weight energy recomputed in 3xTF32: the plain TF32
    /// product only ties the library's error in a different summation order
    spk_segdense_linear0_b32_tf32, threads = 128, min_blocks = 3, m = 18848, n = 128, k = 256,
    bm = 64, bn = 128, epilogue = Leaky,
    simt = { bk = 16, tm = 8, tn = 8, ksplit = 1, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 64, ksplit = 1, stages = 4, precision = Select },
}

tf32_gemm! {
    /// segmentation `linear.1` with LeakyReLU for batch 32 in TF32 mode
    spk_segdense_linear1_b32_tf32, threads = 128, min_blocks = 3, m = 18848, n = 128, k = 128,
    bm = 64, bn = 128, epilogue = Leaky,
    simt = { bk = 16, tm = 8, tn = 8, ksplit = 1, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 64, ksplit = 1, stages = 4, precision = Tf32 },
}

tf32_gemm! {
    /// embedding `seg_1` split-K partials for batch 32 in TF32 mode: 96 x 128 tiles
    /// of six warps, two resident per SM, so the host's one-wave split count keeps
    /// every SM busy without a second, partial wave
    spk_segdense_embed_b32_tf32, threads = 192, min_blocks = 2, m = 96, n = 256, k = 5120,
    bm = 96, bn = 128, epilogue = Partial,
    simt = { bk = 8, tm = 8, tn = 8, ksplit = 1, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 64, ksplit = 1, stages = 4, precision = Tf32 },
}

tf32_gemm! {
    /// embedding `seg_1` split-K partials for batch 32 in TF32 mode on consumer
    /// Blackwell: 96 x 128 tiles of twelve warps in two reduction groups, one
    /// resident per SM
    spk_segdense_embed_b32_tf32_k2, threads = 384, min_blocks = 1, m = 96, n = 256, k = 5120,
    bm = 96, bn = 128, epilogue = Partial,
    simt = { bk = 8, tm = 8, tn = 8, ksplit = 2, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 64, ksplit = 2, stages = 3, precision = Tf32 },
}

tf32_gemm! {
    /// embedding `seg_1` split-K partials for batch 32 in FP32 mode on GPUs whose
    /// TF32 tensor rate is several times their FP32 rate: 3xTF32 in the sm80 tier on
    /// 96 x 64 tiles of six 32 x 32 warps, two resident per SM. A 32 x 64 warp tile
    /// has 16 split products per k-step, whose unrolled body exceeds cuda-oxide's
    /// clone limit and moves the accumulators to local memory
    spk_segdense_embed_b32_x3, threads = 192, min_blocks = 2, m = 96, n = 256, k = 5120,
    bm = 96, bn = 64, epilogue = Partial,
    simt = { bk = 8, tm = 8, tn = 8, ksplit = 2, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 32, ksplit = 1, stages = 4, precision = Split },
}

// embedding projection on tensor-heavy GPUs with many SMs, where a one-wave split
// of 96 x 128 tiles leaves short slices

tf32_gemm! {
    /// embedding `seg_1` split-K partials for batch 32 in TF32 mode on tensor-heavy
    /// parts: 96 x 64 tiles of six warps in two reduction groups, two resident per
    /// SM, so a one-wave grid needs half the slices of the 96 x 128 tiles
    spk_segdense_embed_b32_tf32_e64, threads = 192, min_blocks = 2, m = 96, n = 256,
    k = 5120, bm = 96, bn = 64, epilogue = Partial,
    simt = { bk = 8, tm = 8, tn = 8, ksplit = 2, buffers = 2, depth = 1 },
    mma = { bk = 8, wm = 32, wn = 64, ksplit = 2, stages = 3, precision = Tf32 },
}

half_split! {
    /// embedding `seg_1` for batch 32 in TF32 mode as scaled FP16 products: 96 x 128
    /// tiles, twelve warps in two groups that split each 32-term stage, reduction
    /// slices of at most 320 terms (16 or more slices)
    spk_segdense_embed_b32_f16, threads = 384, m = 96, n = 256, k = 5120, bn = 128, bk = 16,
    wm = 32, wn = 64, ksplit = 2, stages = 3, slice = 320,
}
