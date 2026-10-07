//! Fused Winograd F(2x2, 3x3) on BF16 tensor cores with three products, for TF32 mode
//! on parts whose BF16 tensor rate is a large multiple of their FP32 rate
//!
//! The CTA, tiling and epilogue follow `wtc`: 256 threads own 64 output channels by one
//! row of 32 2x2 tiles, and warp `w` accumulates Winograd elements `2w` and `2w + 1`.
//! Here a pipeline stage holds 16 input channels, the k extent of one `mma.sync`
//! m16n8k16, and both operands split into a BF16 high part and the rounded remainder;
//! each stage multiplies `lo x hi + hi x lo + hi x hi`. The three products keep about 16
//! significant bits of each operand, so the error stays far below that of a TF32
//! convolution (2-5% of cuDNN's TF32 error on the reference activations, host
//! simulation), while BF16 runs at twice the TF32 tensor rate and the split weights take
//! half the bytes of the two TF32 parts
//!
//! The products of a stage start from zero and join the running sum with one
//! round-to-nearest FADD per element, as in `wtc`
//!
//! `U` fragments arrive split and packed (`spk_wideconv_pack_wbf`), two BF16 values per
//! word in fragment order. `V` stays FP32 in shared memory and splits at fragment load,
//! where two channels pack into one register. One `V` stage and two raw-window stages fit
//! 86 KiB, under the 99 KiB a block may opt into on sm_86, sm_89 and sm_120: a raw window
//! lands while the CTA multiplies, and the CTA transforms between two barriers
//!
//! These entries exist in the sm75 variant only to keep the area's ABI equal; there
//! they trap, and the host plans them only for the sm80 tier

use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

#[cfg(feature = "tier-sm80")]
use cuda_device::{DynamicSharedArray, shared::cvta_generic_to_shared_u32};

/// Output channels per BF16 Winograd CTA
pub const WBF_CHANNELS: u32 = 64;
/// 2x2 output tiles per CTA, consecutive in one tile row
pub const WBF_TILES: u32 = 32;
/// Input channels per pipeline stage: the k extent of one `mma.sync` m16n8k16
const WBF_CHUNK: u32 = 16;
/// Threads per CTA
const WBF_THREADS: u32 = 256;
/// Raw input columns per channel and row: two per tile plus the halo
#[cfg(feature = "tier-sm80")]
const RAW_COLUMNS: u32 = 2 * WBF_TILES + 2;
/// Words per column-parity half of a raw row; 48 puts the halves 16 banks apart
const RAW_PARITY: u32 = 48;
/// Words per raw row: both parity halves plus 4
const RAW_ROW: u32 = 2 * RAW_PARITY + 4;
/// Raw words per stage
const RAW_WORDS: u32 = WBF_CHUNK * 4 * RAW_ROW;
/// Words per `V` channel row; 36 puts the even channels of a B fragment's four lane
/// groups on disjoint bank octets
const V_STRIDE: u32 = 36;
/// `V` words of the one stage, `[element][channel][V_STRIDE]`
const V_WORDS: u32 = 16 * WBF_CHUNK * V_STRIDE;
/// Threads per raw row: each copies every 4th column of one (channel, row)
#[cfg(feature = "tier-sm80")]
const RAW_LANES: u32 = WBF_THREADS / (WBF_CHUNK * 4);
/// Columns one thread copies per stage; columns past 65 land in words 33 and 34 of
/// their parity half, which no tile reads
#[cfg(feature = "tier-sm80")]
const RAW_PASSES: u32 = RAW_COLUMNS.div_ceil(RAW_LANES);
/// Words per epilogue row
const EPILOGUE_ROW: u32 = 40;
/// Dynamic shared bytes of a launch: two raw stages and one `V` stage, which also hold
/// the epilogue tile
pub const WBF_SHARED_BYTES: u32 = (2 * RAW_WORDS + V_WORDS) * 4;
/// Words of one packed fragment block: 32 lanes of four registers
const FRAGMENT: u32 = 128;

const _: () =
    assert!(16 * 16 * EPILOGUE_ROW * 4 <= WBF_SHARED_BYTES && WBF_SHARED_BYTES <= 99 * 1024);
const _: () = assert!(WBF_CHUNK * WBF_TILES == 2 * WBF_THREADS && V_STRIDE >= WBF_TILES);
#[cfg(feature = "tier-sm80")]
const _: () =
    assert!(RAW_LANES * WBF_CHUNK * 4 == WBF_THREADS && RAW_PASSES * RAW_LANES / 2 <= RAW_PARITY);

/// BF16 bits of a finite float, rounded to nearest even like `cvt.rn.bf16.f32`
#[inline(always)]
fn bf16_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
}

/// Packs folded weights `[cout][cin][3][3]` into BF16 `mma.sync` m16n8k16 A fragments of
/// `U = G g G^T`, split into a high part and the rounded remainder
///
/// Layout `[cout / 64][cin / 16][16 elements][4 tiles][2 parts][32 lanes][4]` of words
/// holding two BF16 values, the lower column in the low half. Fragment registers hold
/// rows `g`, `g + 8`, `g`, `g + 8` and column pairs `2t`, `2t`, `2t + 8`, `2t + 8` of a
/// 16x16 tile, with `g = lane / 4` and `t = lane % 4`. The transform sums in f64 and
/// rounds to FP32 once
///
/// Launch one thread per word of `packed`, which has `cout * cin * 16` words
#[kernel]
pub fn spk_wideconv_pack_wbf(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let slot = index % 4;
    let lane = index / 4 % 32;
    let part = index / FRAGMENT % 2;
    let tile = index / (2 * FRAGMENT) % 4;
    let element = index / (8 * FRAGMENT) % 16;
    let step = index / (128 * FRAGMENT) % (cin / WBF_CHUNK);
    let block = index / (128 * FRAGMENT * (cin / WBF_CHUNK));
    let row = tile * 16 + lane / 4 + slot % 2 * 8;
    let channel_out = block * WBF_CHANNELS + row;
    let channel_in = step * WBF_CHUNK + 2 * (lane % 4) + slot / 2 * 8;
    let source = ((channel_out * cin + channel_in + 1) * 9) as usize;
    if channel_out >= cout || source + 9 > weight.len() {
        return;
    }
    // two calls, not a loop over a two-element array, which landed in local memory
    let low = packed_part(
        weight,
        ((channel_out * cin + channel_in) * 9) as usize,
        element,
        part,
    );
    let high = packed_part(
        weight,
        ((channel_out * cin + channel_in + 1) * 9) as usize,
        element,
        part,
    );
    if let Some(out) = packed.get_mut(i) {
        *out = f32::from_bits(high << 16 | low);
    }
}

/// BF16 part `part` (0 high, 1 the rounded remainder) of element `element` of `U` for the
/// 3x3 filter at `source`
#[inline(always)]
fn packed_part(weight: &[f32], source: usize, element: u32, part: u32) -> u32 {
    let u = transformed(weight, source, element);
    let high = bf16_bits(u);
    if part == 0 {
        high
    } else {
        bf16_bits(u - f32::from_bits(high << 16))
    }
}

/// Element `element` of `U = G g G^T` for the 3x3 filter at `source`, summed in f64
#[inline(always)]
fn transformed(weight: &[f32], source: usize, element: u32) -> f32 {
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
    sum as f32
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

/// High and low BF16 B-fragment registers of two channels of one tile: the lower
/// channel in each low half
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn split_pair(low: f32, high: f32) -> (u32, u32) {
    let hi = (bf16_bits(high) << 16) | bf16_bits(low);
    let (h0, h1) = (f32::from_bits(hi << 16), f32::from_bits(hi & 0xffff_0000));
    let lo = (bf16_bits(high - h1) << 16) | bf16_bits(low - h0);
    (hi, lo)
}

/// Accumulates one m16n8k16 BF16 product into `c`
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn mma_bf16(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
    use cuda_device::ptx_asm;

    let [mut c0, mut c1, mut c2, mut c3] = c;
    // safety: a convergent warp-wide register operation; every caller runs it with all
    // 32 lanes active
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b0,
            in("r") b1,
        );
    }
    [c0, c1, c2, c3]
}

/// Transforms half a 4x4 patch from a raw stage into its eight `V` words
///
/// As `wtc::transform`, for 16-channel stages and the `V` stride of this kernel
///
/// # Safety
///
/// `raw` and `v` must be word offsets of a published raw stage and of the V stage whose
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
    let dst = opaque(smem + (v + (half * 8 * WBF_CHUNK + channel) * V_STRIDE + tile) * 4);
    // safety: this thread's slots per this function's contract
    unsafe {
        store_column_transform(dst, rows[0]);
        store_column_transform(dst + 4 * WBF_CHUNK * V_STRIDE * 4, rows[1]);
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

    const STEP: u32 = WBF_CHUNK * V_STRIDE * 4;
    // safety: forwarded from this function's contract
    unsafe {
        sts1(dst, r0 - r2);
        sts1(dst + STEP, r1 + r2);
        sts1(dst + 2 * STEP, r2 - r1);
        sts1(dst + 3 * STEP, r1 - r3);
    }
}

/// Expands to one BF16 tensor-core Winograd convolution for a fixed same-channel shape
macro_rules! wbf3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        channels = $channels:expr,
        h = $h:expr,
        w = $w:expr $(,)?
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
                use super::tensor::ops::{commit, copy4, fragment, lds, opaque, wait_all, wait_one};
                use super::winograd::{ldg1, relu};
                use super::{stg2, sts2};

                const THREADS: u32 = WBF_THREADS;
                const C: u32 = $channels;
                const H: u32 = $h;
                const W: u32 = $w;
                const HW: u32 = H * W;
                const TH: u32 = H.div_ceil(2);
                const TW: u32 = W.div_ceil(2);
                const XB: u32 = TW.div_ceil(WBF_TILES);
                const KB: u32 = WBF_CHANNELS;
                const T: u32 = WBF_TILES;
                const CC: u32 = WBF_CHUNK;
                const RPT: usize = RAW_PASSES as usize;
                const STAGE_V: u32 = 2 * RAW_WORDS;
                const _: () = assert!(C % KB == 0 && C % CC == 0);

                let len = batch * C * HW;
                if len as usize > x.len()
                    || splits == 0
                    || C % (splits * CC) != 0
                    || len as usize > y.len()
                    || (split_from < batch * TH * XB && (splits * len) as usize > partial.len())
                    || (add_residual != 0 && len as usize > residual.len())
                    || (C * C * 16) as usize > weight.len()
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

                // raw window of this thread: (channel, row) `tid / 4` of each chunk and its
                // columns `tid % 4 + 4 j`; bit `j` of `raw_valid` marks in-image words, the
                // others zero-fill without a read; shared slots step 2 words per pass
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
                    let ix = col0 + (RAW_LANES * j as u32) as i32;
                    let inside = row_inside && raw_lane + RAW_LANES * (j as u32) < RAW_COLUMNS && ix >= 0 && ix < W as i32;
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
                        let column = (col0 + (RAW_LANES * i as u32) as i32).clamp(0, W as i32 - 1) as u32;
                        // safety: valid copies stay inside the checked input; others read nothing
                        unsafe {
                            copy4(
                                dst + (raw_slot + RAW_LANES / 2 * i as u32) * 4,
                                x_ptr.add((chunk * CC * HW + row_offset + column) as usize),
                                valid,
                            )
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
                            frag[e][m][1] = fragment(fragments.add((block + FRAGMENT) as usize));
                        }
                        m += 1;
                    }
                    e += 1;
                }
                // transform slots: both halves of the patches of channels `vc` and
                // `vc + 8`, tile `vt`
                let vc = tid / T;
                let vt = tid % T;
                // raw(0) has landed once at most raw(1) is pending
                wait_one();
                thread::sync_threads();
                // safety: reads published raw stage 0 and writes this thread's V slots
                unsafe {
                    transform(smem, 0, STAGE_V, 0, vc, vt);
                    transform(smem, 0, STAGE_V, 1, vc, vt);
                    transform(smem, 0, STAGE_V, 0, vc + 8, vt);
                    transform(smem, 0, STAGE_V, 1, vc + 8, vt);
                }
                thread::sync_threads();

                // iteration `c` copies raw(c + 2) into the stage raw(c) left, multiplies
                // V(c), then between two barriers transforms raw(c + 1) into V
                let sv = opaque(smem + STAGE_V * 4);
                let mut c = 0;
                while c < chunks {
                    let next = if c + 1 < chunks { c + 1 } else { chunks - 1 };
                    let after = if c + 2 < chunks { c + 2 } else { chunks - 1 };
                    let dst = opaque(smem + c % 2 * RAW_WORDS * 4);
                    let mut i = 0;
                    #[unroll]
                    while i < RPT {
                        let valid = (raw_valid >> i) & 1 != 0;
                        let column = (col0 + (RAW_LANES * i as u32) as i32).clamp(0, W as i32 - 1) as u32;
                        // safety: valid copies stay inside the checked input; others read nothing
                        unsafe {
                            copy4(
                                dst + (raw_slot + RAW_LANES / 2 * i as u32) * 4,
                                x_ptr.add((after * CC * HW + row_offset + column) as usize),
                                valid,
                            )
                        };
                        i += 1;
                    }
                    commit();

                    let mut e = 0;
                    #[unroll]
                    while e < 2 {
                        let element = 2 * warp + e as u32;
                        let mut n = 0;
                        #[unroll]
                        while n < 4 {
                            // channels 2 t4, 2 t4 + 1, 2 t4 + 8 and 2 t4 + 9 of tile 8 n + g
                            let at = sv + ((element * CC + 2 * t4) * V_STRIDE + 8 * n as u32 + g) * 4;
                            // safety: words of the published V stage
                            let (v0, v1, v8, v9) = unsafe {
                                (
                                    lds(at),
                                    lds(at + V_STRIDE * 4),
                                    lds(at + 8 * V_STRIDE * 4),
                                    lds(at + 9 * V_STRIDE * 4),
                                )
                            };
                            let (h0, l0) = split_pair(v0, v1);
                            let (h1, l1) = split_pair(v8, v9);
                            let mut m = 0;
                            #[unroll]
                            while m < 4 {
                                // the chunk's products start from zero and join the running
                                // sum with rounded FADDs
                                let mut part = mma_bf16([0.0f32; 4], frag[e][m][0], l0, l1);
                                part = mma_bf16(part, frag[e][m][1], h0, h1);
                                part = mma_bf16(part, frag[e][m][0], h0, h1);
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
                                frag[e][m][1] = fragment(fragments.add((block + FRAGMENT) as usize));
                            }
                            m += 1;
                        }
                        e += 1;
                    }

                    // raw(c + 1) has landed once at most raw(c + 2) is pending; the barrier
                    // publishes it and closes this iteration's reads of V
                    wait_one();
                    thread::sync_threads();
                    // safety: reads raw stage (c + 1) % 2 and writes V, read by no thread
                    // until the next barrier
                    unsafe {
                        let raw = (c + 1) % 2 * RAW_WORDS;
                        transform(smem, raw, STAGE_V, 0, vc, vt);
                        transform(smem, raw, STAGE_V, 1, vc, vt);
                        transform(smem, raw, STAGE_V, 0, vc + 8, vt);
                        transform(smem, raw, STAGE_V, 1, vc + 8, vt);
                    }
                    // publishes V(c + 1) and closes the reads of raw stage (c + 1) % 2
                    thread::sync_threads();
                    c += 1;
                }
                // the last iterations' copies of clamped chunks may still be in flight into
                // the shared memory the epilogue reuses
                wait_all();
                thread::sync_threads();

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

wbf3x3! {
    /// Three-product BF16 Winograd 128 -> 128 3x3 convolution for TF32 mode, stride 1,
    /// on 20x250 inputs, with optional residual, folded bias and ReLU, or raw
    /// input-channel partial sums
    ///
    /// Launch 256 threads with `WBF_SHARED_BYTES` of dynamic shared memory, grid
    /// `(2, split_from + (cells - split_from) * splits)` with `cells = batch * 10 * 4`;
    /// weights from `spk_wideconv_pack_wbf`; `splits` divides 8; cells from `split_from`
    /// on write `splits` partial planes of the output size for `spk_wideconv_wino_fixup`;
    /// output and residual 8-byte aligned
    spk_wideconv_wbf_c128,
    channels = 128,
    h = 20,
    w = 250,
}

wbf3x3! {
    /// Three-product BF16 Winograd 256 -> 256 3x3 convolution for TF32 mode, stride 1,
    /// on 10x125 inputs; launch as `spk_wideconv_wbf_c128` with `cells = batch * 5 * 2`
    /// and grid x 4, `splits` dividing 16
    spk_wideconv_wbf_c256,
    channels = 256,
    h = 10,
    w = 125,
}
