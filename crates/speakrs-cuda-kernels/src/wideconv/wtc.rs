//! Fused Winograd F(2x2, 3x3) on TF32 tensor cores, for parts whose TF32 rate is a
//! large multiple of their FP32 rate
//!
//! The CTA, tiling, input pipeline and epilogue follow the FP32 kernel in `winograd`:
//! 256 threads own 64 output channels by one row of 32 2x2 tiles, and warp `w`
//! accumulates Winograd elements `2w` and `2w + 1`. Here the element-wise products run
//! as `mma.sync` m16n8k8: per element and chunk of eight input channels a warp
//! multiplies `U` (64 channels by 8, the A operand) with `V` (8 by 32 tiles, the B
//! operand) in 4 x 4 tiles. Raw input windows arrive with `cp.async`, which zero-fills
//! out-of-image words, and `V` is stored `[element][channel][tile]` with a 40-word
//! channel stride, so the 32 lanes of a B fragment load hit 32 banks
//!
//! Precision splits each operand into a TF32 high part and the remainder:
//!
//! - three products (`lo x hi + hi x lo + hi x hi`, the 3xTF32 scheme) reach FP32-level
//!   error and serve FP32 mode
//! - two products (`U` split, `V` rounded) round only the transformed input, which on
//!   the 128-channel layers keeps the error at that of a direct TF32 convolution, and
//!   serve TF32 mode there
//! - one product (`U` and `V` rounded) is ordinary TF32 arithmetic in the Winograd
//!   domain. It loads only the high weight fragments and accumulates straight into the
//!   running sums, for TF32 mode where end-to-end DER, not per-layer parity with a
//!   direct TF32 convolution, is the bar
//!
//! Tensor cores truncate while accumulating, which over 16 or more chained products
//! left the 3xTF32 error 4x above cuDNN's FP32 error. Each chunk's products therefore
//! accumulate from zero and join the running sum with one round-to-nearest FADD per
//! element, which brings the error below cuDNN's FP32 error
//!
//! `U` fragments arrive pre-split and pre-rounded in fragment order from
//! `spk_wideconv_pack_wtc` and go straight from global memory into registers, one
//! element of the next chunk while the other element computes
//!
//! The staged one-product entries (`wtp1`) keep three raw stages and split the warps
//! into two phases: warps 0..4 transform the next chunk before their products and
//! warps 4..8 after them, so on each scheduler one warp feeds the tensor pipe while
//! the other transforms, and a chunk ends with one barrier instead of two. Their `V`
//! holds TF32 values rounded at the transform, `[element][channel][32]` with the tile
//! index swizzled by `(channel % 4) * 8`, which keeps B fragment loads and transform
//! stores on 32 banks without padding
//!
//! These entries exist in the sm75 variant only to keep the area's ABI equal; there
//! they trap, and the host plans them only for the sm80 tier

use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

#[cfg(feature = "tier-sm80")]
use cuda_device::{DynamicSharedArray, shared::cvta_generic_to_shared_u32};

use super::tensor::round_tf32;

/// Output channels per tensor-core Winograd CTA
pub const WTC_CHANNELS: u32 = 64;
/// 2x2 output tiles per CTA, consecutive in one tile row
pub const WTC_TILES: u32 = 32;
/// Input channels per pipeline stage: the k extent of one `mma.sync`
const WTC_CHUNK: u32 = 8;
/// Threads per CTA
const WTC_THREADS: u32 = 256;
/// Raw input columns per channel and row: two per tile plus the halo
#[cfg(feature = "tier-sm80")]
const RAW_COLUMNS: u32 = 2 * WTC_TILES + 2;
/// Words per column-parity half of a raw row; 48 puts the halves 16 banks apart
const RAW_PARITY: u32 = 48;
/// Words per raw row: both parity halves plus 4, which puts the four rows a warp copies
/// on different banks
const RAW_ROW: u32 = 2 * RAW_PARITY + 4;
/// Raw words per stage
const RAW_WORDS: u32 = WTC_CHUNK * 4 * RAW_ROW;
/// Words per `V` channel row; 8 mod 32 puts the four channel lanes of a B fragment
/// on disjoint bank octets
const V_STRIDE: u32 = 40;
/// `V` words per stage, `[element][channel][V_STRIDE]`
const V_WORDS: u32 = 16 * WTC_CHUNK * V_STRIDE;
/// Threads per raw row: each copies every 8th column of one (channel, row)
#[cfg(feature = "tier-sm80")]
const RAW_LANES: u32 = WTC_THREADS / (WTC_CHUNK * 4);
/// Columns one thread copies per stage; columns past 65 land in words 33..35 of their
/// parity half, which no tile reads
#[cfg(feature = "tier-sm80")]
const RAW_PASSES: u32 = RAW_COLUMNS.div_ceil(RAW_LANES);
/// Half patches one thread transforms per stage
const HALVES: u32 = 2 * WTC_CHUNK * WTC_TILES / WTC_THREADS;
/// Words per epilogue row
const EPILOGUE_ROW: u32 = 40;
/// Dynamic shared bytes of a launch: two stages of raw windows and `V`, which also
/// hold the epilogue tile
pub const WTC_SHARED_BYTES: u32 = 2 * (RAW_WORDS + V_WORDS) * 4;
/// Floats of one packed fragment block: 32 lanes of four registers
const FRAGMENT: u32 = 128;
/// Words per channel row of a staged `V`: one per tile, swizzled instead of padded
const VP_STRIDE: u32 = WTC_TILES;
/// Staged `V` words per stage, `[element][channel][VP_STRIDE]`
const VP_WORDS: u32 = 16 * WTC_CHUNK * VP_STRIDE;
/// Raw stages of a staged launch
const RAW_STAGES: u32 = 3;
/// Dynamic shared bytes of a staged launch: three raw stages and two of `V`, which
/// also hold the epilogue tile
pub const WTP_SHARED_BYTES: u32 = (RAW_STAGES * RAW_WORDS + 2 * VP_WORDS) * 4;

const _: () = assert!(16 * 16 * EPILOGUE_ROW * 4 <= WTC_SHARED_BYTES && HALVES == 2);
const _: () = assert!(16 * 16 * EPILOGUE_ROW * 4 <= WTP_SHARED_BYTES && VP_STRIDE == 32);
const _: () = assert!(WTC_CHUNK * WTC_TILES == WTC_THREADS);
#[cfg(feature = "tier-sm80")]
const _: () =
    assert!(RAW_LANES * WTC_CHUNK * 4 == WTC_THREADS && RAW_PASSES * RAW_LANES / 2 <= RAW_PARITY);

/// Packs folded weights `[cout][cin][3][3]` into TF32 `mma.sync` A fragments of
/// `U = G g G^T`, split into a high part and the rounded remainder
///
/// Layout `[cout / 64][cin / 8][16 elements][4 tiles][2 parts][32 lanes][4]`; fragment
/// registers hold rows `g`, `g + 8`, `g`, `g + 8` and columns `t`, `t`, `t + 4`,
/// `t + 4` of a 16x8 tile, with `g = lane / 4` and `t = lane % 4`. The transform sums
/// in f64, rounds to FP32 once and then splits like `cvt.rna.tf32.f32`
///
/// Launch one thread per element of `packed`, which has `2 * cout * cin * 16` elements
#[kernel]
pub fn spk_wideconv_pack_wtc(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let slot = index % 4;
    let lane = index / 4 % 32;
    let part = index / FRAGMENT % 2;
    let tile = index / (2 * FRAGMENT) % 4;
    let element = index / (8 * FRAGMENT) % 16;
    let step = index / (128 * FRAGMENT) % (cin / WTC_CHUNK);
    let block = index / (128 * FRAGMENT * (cin / WTC_CHUNK));
    let row = tile * 16 + lane / 4 + slot % 2 * 8;
    let channel_out = block * WTC_CHANNELS + row;
    let channel_in = step * WTC_CHUNK + lane % 4 + slot / 2 * 4;
    let source = ((channel_out * cin + channel_in) * 9) as usize;
    if channel_out >= cout || source + 9 > weight.len() {
        return;
    }
    let (a, b) = (element / 4, element % 4);
    let mut sum = 0.0f64;
    let mut p = 0;
    while p < 3 {
        let mut q = 0;
        while q < 3 {
            let g = weight[source + (p * 3 + q) as usize] as f64;
            sum += filter_transform(a, p) * filter_transform(b, q) * g;
            q += 1;
        }
        p += 1;
    }
    let u = sum as f32;
    let high = round_tf32(u);
    if let Some(out) = packed.get_mut(i) {
        *out = if part == 0 {
            high
        } else {
            round_tf32(u - high)
        };
    }
}

/// `G` of F(2x2, 3x3), row `a`, column `p`
#[inline(always)]
fn filter_transform(a: u32, p: u32) -> f64 {
    match (a, p) {
        (0, 0) | (3, 2) => 1.0,
        (1, _) | (2, 0) | (2, 2) => 0.5,
        (2, 1) => -0.5,
        _ => 0.0,
    }
}

/// TF32 high part of a finite float, rounded like `cvt.rna.tf32.f32`, as raw bits
///
/// Integer arithmetic instead of the conversion, which sm_80 and sm_89 expand to a
/// NaN-safe sequence of four instructions; transformed activations are finite
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn high_bits(value: f32) -> u32 {
    value.to_bits().wrapping_add(0x1000) & 0xffff_e000
}

/// Transforms half a 4x4 patch from a raw stage into its eight `V` words
///
/// Half 0 holds patch rows 0..2 and yields `B^T` rows 0 and 1; half 1 holds rows 1..3
/// and yields rows 2 and 3. `V` is `[element][channel][V_STRIDE]`
///
/// # Safety
///
/// `raw` and `v` must be word offsets of a published raw stage and of a V stage whose
/// slots of (`half`, `channel`, `tile`) only this thread writes before the next barrier
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn transform(smem: u32, raw: u32, v: u32, half: u32, channel: u32, tile: u32) {
    use super::tensor::ops::opaque;

    let src = opaque(smem + (raw + (channel * 4 + half) * RAW_ROW + tile) * 4);
    // written out: helper functions get no `#[unroll]`
    // safety: inside the published raw stage per this function's contract
    let d = unsafe {
        [
            raw_row(src),
            raw_row(src + RAW_ROW * 4),
            raw_row(src + 2 * RAW_ROW * 4),
        ]
    };
    let rows = if half == 0 {
        [sub4(d[0], d[2]), add4(d[1], d[2])]
    } else {
        [sub4(d[1], d[0]), sub4(d[0], d[2])]
    };
    let dst = opaque(smem + (v + (half * 8 * WTC_CHUNK + channel) * V_STRIDE + tile) * 4);
    // safety: this thread's slots per this function's contract
    unsafe {
        store_column_transform(dst, rows[0]);
        store_column_transform(dst + 4 * WTC_CHUNK * V_STRIDE * 4, rows[1]);
    }
}

/// The four parity-split words of one raw row a tile reads: columns 0, 1, 2 and 3
///
/// # Safety
///
/// `row` must address a tile's first word of a published raw row
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn raw_row(row: u32) -> [f32; 4] {
    use super::tensor::ops::lds;

    // safety: forwarded from this function's contract
    unsafe {
        [
            lds(row),
            lds(row + RAW_PARITY * 4),
            lds(row + 4),
            lds(row + (RAW_PARITY + 1) * 4),
        ]
    }
}

#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Stores `row B` of one `B^T d` row: four `V` words, one element apart
///
/// # Safety
///
/// `dst` must address this thread's slot of the first of four consecutive elements
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn store_column_transform(dst: u32, [r0, r1, r2, r3]: [f32; 4]) {
    use super::sts1;

    const STEP: u32 = WTC_CHUNK * V_STRIDE * 4;
    // safety: forwarded from this function's contract
    unsafe {
        sts1(dst, r0 - r2);
        sts1(dst + STEP, r1 + r2);
        sts1(dst + 2 * STEP, r2 - r1);
        sts1(dst + 3 * STEP, r1 - r3);
    }
}

/// As [`transform`], into a staged `V`: values rounded to TF32 and the tile index
/// swizzled by `(channel % 4) * 8`
///
/// # Safety
///
/// As [`transform`], with `v` the word offset of a staged `V` stage
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn transform_staged(smem: u32, raw: u32, v: u32, half: u32, channel: u32, tile: u32) {
    use super::tensor::ops::opaque;

    let src = opaque(smem + (raw + (channel * 4 + half) * RAW_ROW + tile) * 4);
    // safety: inside the published raw stage per this function's contract
    let d = unsafe {
        [
            raw_row(src),
            raw_row(src + RAW_ROW * 4),
            raw_row(src + 2 * RAW_ROW * 4),
        ]
    };
    let rows = if half == 0 {
        [sub4(d[0], d[2]), add4(d[1], d[2])]
    } else {
        [sub4(d[1], d[0]), sub4(d[0], d[2])]
    };
    let column = tile ^ ((channel % 4) << 3);
    let dst = opaque(smem + (v + (half * 8 * WTC_CHUNK + channel) * VP_STRIDE + column) * 4);
    // safety: this thread's slots per this function's contract
    unsafe {
        store_column_rounded(dst, rows[0]);
        store_column_rounded(dst + 4 * WTC_CHUNK * VP_STRIDE * 4, rows[1]);
    }
}

/// As [`store_column_transform`], rounded to TF32, in a staged `V`
///
/// # Safety
///
/// `dst` must address this thread's slot of the first of four consecutive elements
#[cfg(feature = "tier-sm80")]
#[inline(always)]
unsafe fn store_column_rounded(dst: u32, [r0, r1, r2, r3]: [f32; 4]) {
    use super::sts1;

    const STEP: u32 = WTC_CHUNK * VP_STRIDE * 4;
    let round = |value: f32| f32::from_bits(high_bits(value));
    // safety: forwarded from this function's contract
    unsafe {
        sts1(dst, round(r0 - r2));
        sts1(dst + STEP, round(r1 + r2));
        sts1(dst + 2 * STEP, round(r2 - r1));
        sts1(dst + 3 * STEP, round(r1 - r3));
    }
}

/// Expands to one tensor-core Winograd convolution for a fixed same-channel shape
macro_rules! wtc3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        channels = $channels:expr,
        h = $h:expr,
        w = $w:expr,
        products = $products:expr,
        stages = $stages:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds(256, 1)]
        pub fn $name(
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            residual: &[f32],
            add_residual: u32,
            batch: u32,
            splits: u32,
            split_from: u32,
            mut y: DisjointSlice<f32>,
            mut partial: DisjointSlice<f32>,
        ) {
            #[cfg(not(feature = "tier-sm80"))]
            {
                let _ = (x, weight, bias, residual, add_residual, batch, splits, split_from, y.as_mut_ptr(), partial.as_mut_ptr());
                cuda_device::debug::trap();
            }

            #[cfg(feature = "tier-sm80")]
            {
                use super::tensor::ops::{commit, copy4, fragment, lds, mma, opaque, wait_all};
                use super::winograd::{ldg1, relu};
                use super::{stg2, sts2};

                const THREADS: u32 = WTC_THREADS;
                const C: u32 = $channels;
                const H: u32 = $h;
                const W: u32 = $w;
                const HW: u32 = H * W;
                const TH: u32 = H.div_ceil(2);
                const TW: u32 = W.div_ceil(2);
                const XB: u32 = TW.div_ceil(WTC_TILES);
                const KB: u32 = WTC_CHANNELS;
                const T: u32 = WTC_TILES;
                const CC: u32 = WTC_CHUNK;
                const PRODUCTS: u32 = $products;
                // 2: the plain pipeline; 3: the staged one-product pipeline
                const STAGES: u32 = $stages;
                const RPT: usize = RAW_PASSES as usize;
                const STAGE_V: u32 = 2 * RAW_WORDS;
                const STAGED_V: u32 = RAW_STAGES * RAW_WORDS;
                const _: () = assert!(C % KB == 0 && C % CC == 0 && PRODUCTS >= 1 && PRODUCTS <= 3);
                const _: () = assert!(STAGES == 2 || (STAGES == RAW_STAGES && PRODUCTS == 1));

                let len = batch * C * HW;
                if len as usize > x.len()
                    || splits == 0
                    || C % (splits * CC) != 0
                    || len as usize > y.len()
                    || (split_from < batch * TH * XB && (splits * len) as usize > partial.len())
                    || (add_residual != 0 && len as usize > residual.len())
                    || (2 * C * C * 16) as usize > weight.len()
                    || C as usize > bias.len()
                {
                    return;
                }
                // cells below `split_from` reduce every input channel; each later cell takes
                // `splits` CTAs, one per input-channel partition
                let cells = batch * TH * XB;
                let cotile = thread::blockIdx_x();
                let slot = thread::blockIdx_y();
                if cotile >= C / KB || split_from > cells || slot >= split_from + (cells - split_from) * splits {
                    return;
                }
                let (cell, split, parts) = if slot < split_from {
                    (slot, 0, 1)
                } else {
                    let k = slot - split_from;
                    (split_from + k / splits, k % splits, splits)
                };
                let item = cell / (TH * XB);
                let ty = cell / XB % TH;
                let tx0 = cell % XB * T;
                let channels = C / parts;
                let chunks = channels / CC;
                let c0 = split * channels;

                let tid = thread::threadIdx_x();
                let lane = tid % 32;
                let warp = tid / 32;
                let g = lane / 4;
                let t4 = lane % 4;
                // safety: the dynamic shared base of this CTA; only its address is taken
                let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
                let x_ptr = x.as_ptr();

                // raw window of this thread: (channel, row) `tid / 8` of each chunk and its
                // columns `tid % 8 + 8 j`; bit `j` of `raw_valid` marks in-image words, the
                // others zero-fill without a read; shared slots step 4 words per pass
                let raw_row = tid / RAW_LANES;
                let raw_lane = tid % RAW_LANES;
                let iy = (2 * ty + raw_row % 4) as i32 - 1;
                let row_inside = iy >= 0 && iy < H as i32;
                let row_offset = ((item * C + c0 + raw_row / 4) * H + iy.clamp(0, H as i32 - 1) as u32) * W;
                let col0 = (2 * tx0 + raw_lane) as i32 - 1;
                let mut raw_valid = 0u32;
                let mut j = 0;
                #[unroll]
                while j < RPT {
                    let ix = col0 + 8 * j as i32;
                    let inside = row_inside && raw_lane + 8 * (j as u32) < RAW_COLUMNS && ix >= 0 && ix < W as i32;
                    raw_valid |= u32::from(inside) << j;
                    j += 1;
                }
                let raw_slot = raw_row * RAW_ROW + raw_lane % 2 * RAW_PARITY + raw_lane / 2;
                // this warp's fragments: element 2 warp, tile 0, high part, this lane
                // safety: offsets below stay inside the checked packed weights
                let fragments = unsafe {
                    weight.as_ptr().add(
                        ((((cotile * (C / CC) + c0 / CC) * 16 + 2 * warp) * 8) * FRAGMENT + lane * 4) as usize,
                    )
                };

                let mut acc = [[[[0.0f32; 4]; 4]; 4]; 2];
                // [element][tile][high, low]
                let mut frag = [[[[0u32; 4]; 2]; 4]; 2];

                // transform slots: both halves of the patch of channel `vc`, tile `vt`
                let vc = tid / T;
                let vt = tid % T;
                if STAGES == RAW_STAGES {
                    // prologue: raw windows of chunks 0 and 1, fragments of chunk 0, V of chunk 0
                    let mut stage = 0;
                    #[unroll]
                    while stage < 2 {
                        let chunk = if (stage as u32) < chunks { stage as u32 } else { chunks - 1 };
                        let dst = smem + stage as u32 * RAW_WORDS * 4;
                        let mut i = 0;
                        #[unroll]
                        while i < RPT {
                            let valid = (raw_valid >> i) & 1 != 0;
                            let column = (col0 + 8 * i as i32).clamp(0, W as i32 - 1) as u32;
                            // safety: valid copies stay inside the checked input; others read nothing
                            unsafe {
                                copy4(dst + (raw_slot + 4 * i as u32) * 4, x_ptr.add((chunk * CC * HW + row_offset + column) as usize), valid)
                            };
                            i += 1;
                        }
                        commit();
                        stage += 1;
                    }
                    let mut e = 0;
                    #[unroll]
                    while e < 2 {
                        let mut m = 0;
                        #[unroll]
                        while m < 4 {
                            let block = ((e * 4 + m) * 2) as u32 * FRAGMENT;
                            // safety: fragments of chunk 0 of this warp's elements
                            unsafe { frag[e][m][0] = fragment(fragments.add(block as usize)) };
                            m += 1;
                        }
                        e += 1;
                    }
                    wait_all();
                    thread::sync_threads();
                    // safety: reads published raw stage 0 and writes this thread's V slots
                    unsafe {
                        transform_staged(smem, 0, STAGED_V, 0, vc, vt);
                        transform_staged(smem, 0, STAGED_V, 1, vc, vt);
                    }
                    thread::sync_threads();

                    // warps 0..4 transform raw(c + 1) before their products, warps 4..8
                    // after them: each scheduler runs one warp of each phase
                    let early = warp < 4;
                    let mut c = 0;
                    while c < chunks {
                        let next = if c + 1 < chunks { c + 1 } else { chunks - 1 };
                        let after = if c + 2 < chunks { c + 2 } else { chunks - 1 };
                        // raw(c + 2) goes into the stage of raw(c - 1), transformed in
                        // iteration c - 2
                        let dst = opaque(smem + (c + 2) % RAW_STAGES * RAW_WORDS * 4);
                        let mut i = 0;
                        #[unroll]
                        while i < RPT {
                            let valid = (raw_valid >> i) & 1 != 0;
                            let column = (col0 + 8 * i as i32).clamp(0, W as i32 - 1) as u32;
                            // safety: valid copies stay inside the checked input; others read nothing
                            unsafe {
                                copy4(dst + (raw_slot + 4 * i as u32) * 4, x_ptr.add((after * CC * HW + row_offset + column) as usize), valid)
                            };
                            i += 1;
                        }
                        commit();

                        let raw = (c + 1) % RAW_STAGES * RAW_WORDS;
                        let v_next = STAGED_V + (c + 1) % 2 * VP_WORDS;
                        if early {
                            // safety: reads raw(c + 1), published by the last barrier, and
                            // writes V stage (c + 1) % 2, last read in iteration c - 1
                            unsafe {
                                transform_staged(smem, raw, v_next, 0, vc, vt);
                                transform_staged(smem, raw, v_next, 1, vc, vt);
                            }
                        }
                        let sv = opaque(smem + (STAGED_V + c % 2 * VP_WORDS) * 4);
                        let mut e = 0;
                        #[unroll]
                        while e < 2 {
                            let element = 2 * warp + e as u32;
                            let mut n = 0;
                            #[unroll]
                            while n < 4 {
                                let tile = (8 * n as u32 + g) ^ (t4 << 3);
                                let at = sv + ((element * CC + t4) * VP_STRIDE + tile) * 4;
                                // safety: words of the published V stage, already rounded
                                let (h0, h1) = unsafe { (lds(at).to_bits(), lds(at + 4 * VP_STRIDE * 4).to_bits()) };
                                let mut m = 0;
                                #[unroll]
                                while m < 4 {
                                    acc[e][m][n] = mma(acc[e][m][n], frag[e][m][0], h0, h1);
                                    m += 1;
                                }
                                n += 1;
                            }
                            let mut m = 0;
                            #[unroll]
                            while m < 4 {
                                let block = ((next * 16 * 4 * 2) + ((e * 4 + m) * 2) as u32) * FRAGMENT;
                                // safety: fragments of a chunk of this warp's elements
                                unsafe { frag[e][m][0] = fragment(fragments.add(block as usize)) };
                                m += 1;
                            }
                            e += 1;
                        }
                        if !early {
                            // safety: as the early transform above
                            unsafe {
                                transform_staged(smem, raw, v_next, 0, vc, vt);
                                transform_staged(smem, raw, v_next, 1, vc, vt);
                            }
                        }
                        wait_all();
                        // publishes raw(c + 2) and V(c + 1) and closes all reads of V(c)
                        thread::sync_threads();
                        c += 1;
                    }
                } else {
                    // prologue: raw windows of chunks 0 and 1, fragments of chunk 0, V of chunk 0
                    let mut stage = 0;
                    #[unroll]
                    while stage < 2 {
                        let chunk = if (stage as u32) < chunks { stage as u32 } else { chunks - 1 };
                        let dst = smem + stage as u32 * RAW_WORDS * 4;
                        let mut i = 0;
                        #[unroll]
                        while i < RPT {
                            let valid = (raw_valid >> i) & 1 != 0;
                            let column = (col0 + 8 * i as i32).clamp(0, W as i32 - 1) as u32;
                            // safety: valid copies stay inside the checked input; others read nothing
                            unsafe {
                                copy4(dst + (raw_slot + 4 * i as u32) * 4, x_ptr.add((chunk * CC * HW + row_offset + column) as usize), valid)
                            };
                            i += 1;
                        }
                        commit();
                        stage += 1;
                    }
                    let mut e = 0;
                    #[unroll]
                    while e < 2 {
                        let mut m = 0;
                        #[unroll]
                        while m < 4 {
                            let block = ((e * 4 + m) * 2) as u32 * FRAGMENT;
                            // safety: fragments of chunk 0 of this warp's elements
                            unsafe {
                                frag[e][m][0] = fragment(fragments.add(block as usize));
                                if PRODUCTS > 1 {
                                    frag[e][m][1] = fragment(fragments.add((block + FRAGMENT) as usize));
                                }
                            }
                            m += 1;
                        }
                        e += 1;
                    }
                    wait_all();
                    thread::sync_threads();
                    // safety: reads published raw stage 0 and writes this thread's V slots
                    unsafe {
                        transform(smem, 0, STAGE_V, 0, vc, vt);
                        transform(smem, 0, STAGE_V, 1, vc, vt);
                    }
                    thread::sync_threads();

                    let mut c = 0;
                    while c < chunks {
                        let next = if c + 1 < chunks { c + 1 } else { chunks - 1 };
                        let after = if c + 2 < chunks { c + 2 } else { chunks - 1 };
                        // raw(c + 2) goes into the buffer raw(c) left in iteration c - 1's transform
                        let dst = opaque(smem + c % 2 * RAW_WORDS * 4);
                        let mut i = 0;
                        #[unroll]
                        while i < RPT {
                            let valid = (raw_valid >> i) & 1 != 0;
                            let column = (col0 + 8 * i as i32).clamp(0, W as i32 - 1) as u32;
                            // safety: valid copies stay inside the checked input; others read nothing
                            unsafe {
                                copy4(dst + (raw_slot + 4 * i as u32) * 4, x_ptr.add((after * CC * HW + row_offset + column) as usize), valid)
                            };
                            i += 1;
                        }
                        commit();

                        let sv = opaque(smem + (STAGE_V + c % 2 * V_WORDS) * 4);
                        let mut e = 0;
                        #[unroll]
                        while e < 2 {
                            let element = 2 * warp + e as u32;
                            let mut n = 0;
                            #[unroll]
                            while n < 4 {
                                let at = sv + ((element * CC + t4) * V_STRIDE + 8 * n as u32 + g) * 4;
                                // safety: words of the published V stage
                                let (v0, v1) = unsafe { (lds(at), lds(at + 4 * V_STRIDE * 4)) };
                                let (h0, h1) = (high_bits(v0), high_bits(v1));
                                let (l0, l1) = if PRODUCTS == 3 {
                                    // the exact remainder; the tensor core ignores its low 13 bits
                                    ((v0 - f32::from_bits(h0)).to_bits(), (v1 - f32::from_bits(h1)).to_bits())
                                } else {
                                    (0, 0)
                                };
                                let mut m = 0;
                                #[unroll]
                                while m < 4 && PRODUCTS == 1 {
                                    acc[e][m][n] = mma(acc[e][m][n], frag[e][m][0], h0, h1);
                                    m += 1;
                                }
                                #[unroll]
                                while m < 4 {
                                    // tensor cores truncate while accumulating: the chunk's products
                                    // start from zero and join the running sum with rounded FADDs
                                    let mut part = [0.0f32; 4];
                                    if PRODUCTS == 3 {
                                        part = mma(part, frag[e][m][0], l0, l1);
                                    }
                                    part = mma(part, frag[e][m][1], h0, h1);
                                    part = mma(part, frag[e][m][0], h0, h1);
                                    let mut r = 0;
                                    #[unroll]
                                    while r < 4 {
                                        acc[e][m][n][r] += part[r];
                                        r += 1;
                                    }
                                    m += 1;
                                }
                                n += 1;
                            }
                            // this element's fragments of the next chunk land while the other
                            // element computes
                            let mut m = 0;
                            #[unroll]
                            while m < 4 {
                                let block = ((next * 16 * 4 * 2) + ((e * 4 + m) * 2) as u32) * FRAGMENT;
                                // safety: fragments of a chunk of this warp's elements
                                unsafe {
                                    frag[e][m][0] = fragment(fragments.add(block as usize));
                                    if PRODUCTS > 1 {
                                        frag[e][m][1] = fragment(fragments.add((block + FRAGMENT) as usize));
                                    }
                                }
                                m += 1;
                            }
                            e += 1;
                        }

                        // safety: reads raw stage (c + 1) % 2, published by the last barrier,
                        // and writes V stage (c + 1) % 2, last read in iteration c - 1
                        unsafe {
                            let raw = (c + 1) % 2 * RAW_WORDS;
                            let v = STAGE_V + (c + 1) % 2 * V_WORDS;
                            transform(smem, raw, v, 0, vc, vt);
                            transform(smem, raw, v, 1, vc, vt);
                        }
                        wait_all();
                        // publishes this iteration's stages and closes all reads of the others
                        thread::sync_threads();
                        c += 1;
                    }
                }

                // round `r` moves 16-channel tile `r` of both elements to
                // `M[element][16][EPILOGUE_ROW]`; the round's residual loads first, so its
                // latency overlaps the shared-memory round trip
                let base = opaque(smem);
                // whole cells write the output, split cells their partition's plane of the workspace
                let out = if parts == 1 { y.as_mut_ptr() } else { partial.as_mut_ptr() };
                let combine = parts == 1;
                let shortcut_on = combine && add_residual != 0;
                let residual_ptr = residual.as_ptr();
                let plane = if parts == 1 { 0 } else { split * len };
                let mut r = 0;
                #[unroll]
                while r < 4 {
                    let mut shortcut = [[0.0f32; 4]; 2];
                    let mut pass = 0;
                    #[unroll]
                    while pass < 2 {
                        let q = tid + pass as u32 * THREADS;
                        let tx = tx0 + q % T;
                        let channel = cotile * KB + 16 * r as u32 + q / T;
                        let mut k = 0;
                        #[unroll]
                        while k < 4 {
                            let (oy, ox) = (2 * ty + k as u32 / 2, 2 * tx + k as u32 % 2);
                            let valid = shortcut_on && oy < H && ox < W;
                            let word = if valid { (((item * C + channel) * H + oy) * W + ox) as usize } else { 0 };
                            // safety: valid words lie inside the checked residual
                            shortcut[pass][k] = unsafe { ldg1(residual_ptr.add(word), valid) };
                            k += 1;
                        }
                        pass += 1;
                    }
                    let mut e = 0;
                    #[unroll]
                    while e < 2 {
                        let element = 2 * warp + e as u32;
                        let mut n = 0;
                        #[unroll]
                        while n < 4 {
                            let column = 8 * n as u32 + 2 * t4;
                            let low = base + ((element * 16 + g) * EPILOGUE_ROW + column) * 4;
                            let high = base + ((element * 16 + g + 8) * EPILOGUE_ROW + column) * 4;
                            // safety: aligned pairs of the epilogue tile, one writer each
                            unsafe {
                                sts2(low, [acc[e][r][n][0], acc[e][r][n][1]]);
                                sts2(high, [acc[e][r][n][2], acc[e][r][n][3]]);
                            }
                            n += 1;
                        }
                        e += 1;
                    }
                    thread::sync_threads();

                    let mut pass = 0;
                    #[unroll]
                    while pass < 2 {
                        let q = tid + pass as u32 * THREADS;
                        let t = q % T;
                        let kr = q / T;
                        let tx = tx0 + t;
                        let channel = cotile * KB + 16 * r as u32 + kr;
                        if tx < TW {
                            let mut m = [0.0f32; 16];
                            let mut e = 0;
                            #[unroll]
                            while e < 16 {
                                // safety: written before the barrier above
                                m[e] = unsafe { lds(base + ((e as u32 * 16 + kr) * EPILOGUE_ROW + t) * 4) };
                                e += 1;
                            }
                            let mut columns = [0.0f32; 8];
                            let mut b = 0;
                            #[unroll]
                            while b < 4 {
                                columns[b] = m[b] + m[4 + b] + m[8 + b];
                                columns[4 + b] = m[4 + b] - m[8 + b] - m[12 + b];
                                b += 1;
                            }
                            let bias_value = if combine { bias[channel as usize] } else { 0.0 };
                            let mut a = 0;
                            #[unroll]
                            while a < 2 {
                                let oy = 2 * ty + a as u32;
                                let (c0, c1, c2, c3) =
                                    (columns[4 * a], columns[4 * a + 1], columns[4 * a + 2], columns[4 * a + 3]);
                                let pair = [c0 + c1 + c2, c1 - c2 - c3];
                                // the residual goes in before the bias, as in cuDNN's fused call
                                let value = if combine {
                                    [
                                        relu(pair[0] + shortcut[pass][2 * a] + bias_value),
                                        relu(pair[1] + shortcut[pass][2 * a + 1] + bias_value),
                                    ]
                                } else {
                                    pair
                                };
                                let index = plane + ((item * C + channel) * H + oy) * W + 2 * tx;
                                if oy < H && W % 2 == 0 {
                                    // safety: an even pair of the checked output with one writer;
                                    // the host checks 8-byte alignment of output and workspace
                                    unsafe { stg2(out.add(index as usize), value) };
                                } else if oy < H {
                                    let mut k = 0;
                                    #[unroll]
                                    while k < 2 {
                                        if 2 * tx + (k as u32) < W {
                                            // safety: inside the checked output; one writer
                                            unsafe { *out.add((index + k as u32) as usize) = value[k] };
                                        }
                                        k += 1;
                                    }
                                }
                                a += 1;
                            }
                        }
                        pass += 1;
                    }
                    // closes the reads of this round before the next round's writes
                    thread::sync_threads();
                    r += 1;
                }
            }
        }
    };
}

wtc3x3! {
    /// 3xTF32 Winograd 128 -> 128 3x3 convolution, stride 1, on 20x250 inputs, with
    /// optional residual, folded bias and ReLU, or raw input-channel partial sums;
    /// FP32-level error
    ///
    /// Launch 256 threads with `WTC_SHARED_BYTES` of dynamic shared memory, grid
    /// `(2, split_from + (cells - split_from) * splits)` with `cells = batch * 10 * 4`;
    /// weights from `spk_wideconv_pack_wtc`; `splits` divides 16; cells from
    /// `split_from` on write `splits` partial planes of the output size for
    /// `spk_wideconv_wino_fixup`; output and residual 8-byte aligned
    spk_wideconv_wtc3_c128,
    channels = 128,
    h = 20,
    w = 250,
    products = 3,
    stages = 2,
}

wtc3x3! {
    /// 3xTF32 Winograd 256 -> 256 3x3 convolution, stride 1, on 10x125 inputs; as
    /// `spk_wideconv_wtc3_c128` with `cells = batch * 5 * 2` and grid x 4, `splits`
    /// dividing 32
    spk_wideconv_wtc3_c256,
    channels = 256,
    h = 10,
    w = 125,
    products = 3,
    stages = 2,
}

wtc3x3! {
    /// Two-product TF32 Winograd 128 -> 128 3x3 convolution for TF32 mode: weights
    /// split, transformed inputs rounded; launch as `spk_wideconv_wtc3_c128`
    spk_wideconv_wtc2_c128,
    channels = 128,
    h = 20,
    w = 250,
    products = 2,
    stages = 2,
}

wtc3x3! {
    /// Two-product TF32 Winograd 256 -> 256 3x3 convolution for TF32 mode; launch as
    /// `spk_wideconv_wtc3_c256`
    spk_wideconv_wtc2_c256,
    channels = 256,
    h = 10,
    w = 125,
    products = 2,
    stages = 2,
}

wtc3x3! {
    /// One-product TF32 Winograd 128 -> 128 3x3 convolution for TF32 mode: weights and
    /// transformed inputs rounded; launch as `spk_wideconv_wtc3_c128`
    spk_wideconv_wtc1_c128,
    channels = 128,
    h = 20,
    w = 250,
    products = 1,
    stages = 2,
}

wtc3x3! {
    /// One-product TF32 Winograd 256 -> 256 3x3 convolution for TF32 mode; launch as
    /// `spk_wideconv_wtc3_c256`
    spk_wideconv_wtc1_c256,
    channels = 256,
    h = 10,
    w = 125,
    products = 1,
    stages = 2,
}

wtc3x3! {
    /// Staged one-product TF32 Winograd 128 -> 128 3x3 convolution for TF32 mode; launch
    /// as `spk_wideconv_wtc3_c128` with `WTP_SHARED_BYTES` of dynamic shared memory
    spk_wideconv_wtp1_c128,
    channels = 128,
    h = 20,
    w = 250,
    products = 1,
    stages = 3,
}

wtc3x3! {
    /// Staged one-product TF32 Winograd 256 -> 256 3x3 convolution for TF32 mode; launch
    /// as `spk_wideconv_wtc3_c256` with `WTP_SHARED_BYTES` of dynamic shared memory
    spk_wideconv_wtp1_c256,
    channels = 256,
    h = 10,
    w = 125,
    products = 1,
    stages = 3,
}
