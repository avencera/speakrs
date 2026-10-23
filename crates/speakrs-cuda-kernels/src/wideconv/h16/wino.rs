//! Fused Winograd F(2x2, 3x3) on FP16 tensor cores for the same-channel stride-1 trunk
//! convolutions, in the operands and launch contract of the direct FP16 kernels
//!
//! The CTA and pipeline follow the staged one-product TF32 kernels in `wtc`: 256 threads
//! own 64 output channels by one row of 32 2x2 output tiles, warp `w` accumulates
//! Winograd elements `2w` and `2w + 1`, three raw stages feed two stages of `V`, and warps
//! 0..4 transform the next chunk before their products while warps 4..8 transform it
//! after them. The products are `mma.sync` m16n8k16 FP16 products accumulated in FP32,
//! so a chunk covers 16 input channels and a warp issues half the tensor instructions of
//! the TF32 kernel for the same work
//!
//! Activations are staged as the half tensors of the direct kernels: FP16 scaled by 2^10,
//! saturated and range-checked, eight channels and 16 bytes per pixel. A half input
//! already is that form and arrives with 16-byte `cp.async` copies, which zero-fill the
//! padding; an FP32 input becomes it with the direct kernels' checked conversion. A half
//! input therefore gives exactly the result of the FP32 input it was made from
//!
//! `V = B^T d B` is computed in FP32 from the staged halves, scaled by 1/4 and rounded to
//! FP16 once. Each element is a signed sum of four staged values, so the quarter scale
//! keeps it finite. `U = G g G^T` comes from [`spk_wideconv_pack_wh16`] scaled by 2^10,
//! and the epilogue removes the 2^18 product scale after `A^T M A`. FP16 keeps the 11
//! significant bits of TF32, so the rounding matches the one-product TF32 kernels'
//!
//! `V` holds words of two adjacent input channels, `[element][pair][tile]`, which is the
//! m16n8k16 B fragment layout. The tile index is swizzled by the pair, so fragment loads,
//! transform stores and the transform's raw reads each hit 32 banks
//!
//! The epilogue moves the products through shared memory in four rounds of 16 channels,
//! as in `wtc`, and gives each thread two adjacent channels of one tile. A half residual
//! is read as those channels' words, and a half output passes through a pixel-major tile
//! so it leaves as whole 16-byte pixels

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

use super::super::tensor::ops::{commit, fragment, opaque, wait_all};
use super::{flag_out_of_range, half2, half2_checked, ldg, lds4, relu, stg4};

/// Output channels per CTA
const CHANNELS: u32 = 64;
/// 2x2 output tiles per CTA, consecutive in one tile row
const TILES: u32 = 32;
/// Threads per CTA
const THREADS: u32 = 256;
/// Input channels per pipeline stage: the k extent of one m16n8k16 product
const CHUNK: u32 = 16;
/// Raw input columns per row: two per tile plus the halo
const RAW_COLUMNS: u32 = 2 * TILES + 2;
/// Words per channel group of a raw stage: four rows of 16-byte pixels, plus 4 words that
/// put the two groups 4 banks apart
const GROUP_WORDS: u32 = 4 * RAW_COLUMNS * 4 + 4;
/// Words per raw stage: two channel groups
const RAW_WORDS: u32 = 2 * GROUP_WORDS;
/// Raw stages
const RAW_STAGES: u32 = 3;
/// Words per `V` stage, `[element][pair][tile]`
const V_WORDS: u32 = 16 * (CHUNK / 2) * TILES;
/// First word of the `V` stages
const V_BASE: u32 = RAW_STAGES * RAW_WORDS;
/// Words per epilogue row of `M[element][channel][row]`; 40 keeps a quarter warp's two
/// channel rows 16 banks apart
const EPILOGUE_ROW: u32 = 40;
/// Words of the epilogue's product tile
const M_WORDS: u32 = 16 * 16 * EPILOGUE_ROW;
/// Words of a pixel-major epilogue tile: two channel groups of two rows of 64 pixels
const PIXEL_WORDS: u32 = 2 * 2 * 2 * TILES * 4;
/// Dynamic shared bytes of a launch: three raw stages and two of `V`, which also hold
/// the epilogue's product tile and its residual and output pixel tiles
pub const WH16_SHARED_BYTES: u32 = (V_BASE + 2 * V_WORDS) * 4;
/// Floats of one packed fragment block: 32 lanes of four words
const FRAGMENT: u32 = 128;
/// Removes the 2^18 scale of a product of a 2^10-scaled weight and a 2^8-scaled input;
/// exact for finite sums
const PRODUCT_UNSCALE: f32 = 1.0 / (1024.0 * 256.0);
/// Removes the 2^10 operand scale of a half residual; exact, as a power of two
const RESIDUAL_UNSCALE: f32 = 1.0 / 1024.0;

const _: () = assert!(M_WORDS + 2 * PIXEL_WORDS <= V_BASE + 2 * V_WORDS);
const _: () = assert!(GROUP_WORDS % 32 == 4 && (CHUNK / 2) * TILES == THREADS);

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

/// `U` of one weight filter and element, summed in f64 and rounded to FP32 once
#[inline(always)]
fn transformed_weight(weight: &[f32], source: usize, element: u32) -> f32 {
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

/// Packs folded weights `[cout][cin][3][3]` into FP16 m16n8k16 A fragments of
/// `U = G g G^T`, scaled by 2^10
///
/// Layout `[cout / 64][cin / 16][16 elements][4 tiles][32 lanes][4]` in 32-bit words of
/// two FP16 values; fragment words hold rows `g`, `g + 8`, `g`, `g + 8` and channel pairs
/// `2t`, `2t`, `2t + 8`, `2t + 8` of a 16x16 tile, with `g = lane / 4` and
/// `t = lane % 4`. The packed buffer holds `cout * cin * 8` words in an `f32` slice
///
/// Launch one thread per word of `packed`
#[kernel]
pub fn spk_wideconv_pack_wh16(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let slot = index % 4;
    let lane = index / 4 % 32;
    let tile = index / FRAGMENT % 4;
    let element = index / (4 * FRAGMENT) % 16;
    let step = index / (64 * FRAGMENT) % (cin / CHUNK);
    let block = index / (64 * FRAGMENT * (cin / CHUNK));
    let row = tile * 16 + lane / 4 + slot % 2 * 8;
    let channel_out = block * CHANNELS + row;
    let channel_in = step * CHUNK + lane % 4 * 2 + slot / 2 * 8;
    let source = ((channel_out * cin + channel_in) * 9) as usize;
    // the high half is the next input channel's filter
    if channel_out >= cout || source + 18 > weight.len() {
        return;
    }
    let lo = transformed_weight(weight, source, element);
    let hi = transformed_weight(weight, source + 9, element);
    if let Some(out) = packed.get_mut(i) {
        *out = f32::from_bits(half2(lo, hi));
    }
}

/// Starts a 16-byte asynchronous copy through L2; an invalid source writes zeros
///
/// # Safety
///
/// `dst` must be a 16-byte aligned shared address of this CTA and `src` a 16-byte
/// aligned readable global address; with `valid` false the source is not read
#[inline(always)]
unsafe fn copy16(dst: u32, src: *const f32, valid: bool) {
    let size: u32 = if valid { 16 } else { 0 };
    // safety: forwarded from this function's contract
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.cg.shared.global [%0], [g], 16, %2; }",
            in("r") dst,
            in("l") src as u64,
            in("r") size,
            clobber("memory"),
        );
    }
}

/// One shared word
///
/// # Safety
///
/// `address` must lie inside this CTA's published shared memory
#[inline(always)]
unsafe fn lds_word(address: u32) -> u32 {
    let value: u32;
    // safety: forwarded from this function's contract
    unsafe {
        ptx_asm!(
            "ld.shared.b32 %0, [%1];",
            out("=r") value,
            in("r") address,
            clobber("memory"),
        );
    }
    value
}

/// Stores one shared word
///
/// # Safety
///
/// `address` must lie inside this CTA's shared memory, in a slot no other thread
/// accesses before the next barrier
#[inline(always)]
unsafe fn sts_word(address: u32, value: u32) {
    // safety: forwarded from this function's contract
    unsafe {
        ptx_asm!(
            "st.shared.b32 [%0], %1;",
            in("r") address,
            in("r") value,
            clobber("memory"),
        );
    }
}

/// One global word of a half tensor, through the coherent path, since a half residual
/// may alias the output
///
/// # Safety
///
/// `pointer` must address a readable word
#[inline(always)]
unsafe fn ld_word(pointer: *const f32) -> u32 {
    let value: u32;
    // safety: forwarded from this function's contract
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.b32 %0, [g]; }",
            out("=r") value,
            in("l") pointer as u64,
            clobber("memory"),
        );
    }
    value
}

/// The two FP16 values of a word as FP32, low half first
#[inline(always)]
fn unpack(word: u32) -> [f32; 2] {
    let (lo, hi): (f32, f32);
    // safety: register conversions with no memory access
    unsafe {
        ptx_asm!(
            "{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }",
            out("=f") lo,
            out("=f") hi,
            in("r") word,
            options(register_only),
        );
    }
    [lo, hi]
}

/// `lo` and `hi` rounded to FP16 with round-to-nearest-even, `lo` in the low half
#[inline(always)]
fn pack(lo: f32, hi: f32) -> u32 {
    let word: u32;
    // safety: a register conversion with no memory access
    unsafe {
        ptx_asm!(
            "cvt.rn.f16x2.f32 %0, %1, %2;",
            out("=r") word,
            in("f") hi,
            in("f") lo,
            options(register_only),
        );
    }
    word
}

/// Accumulates one m16n8k16 FP16 product into `c` in FP32
#[inline(always)]
fn mma16(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = c;
    // safety: a convergent warp-wide register operation; every caller runs it with all
    // 32 lanes active
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
            in("r") b0,
            in("r") b1,
        );
    }
    [c0, c1, c2, c3]
}

/// Swizzled `V` column of `tile` for channel pair `pair`: bits 3 and 4 take the pair's
/// low bits and bit 2 its high bit, so the four pairs of a B fragment load and the eight
/// pairs of a transform store land on different banks
#[inline(always)]
const fn v_column(tile: u32, pair: u32) -> u32 {
    tile ^ ((pair & 3) << 3) ^ ((pair >> 2) << 2)
}

/// Word offset of one tile's `V` word for an element and channel pair
#[inline(always)]
const fn v_word(element: u32, pair: u32, tile: u32) -> u32 {
    (element * (CHUNK / 2) + pair) * TILES + v_column(tile, pair)
}

/// Four words of one raw row: columns `column..column + 4` of one channel pair
///
/// # Safety
///
/// `row` must address the pair's word of a tile's first column in a published raw stage
#[inline(always)]
unsafe fn raw_row(row: u32) -> [[f32; 2]; 4] {
    // safety: forwarded from this function's contract; pixels are 16 bytes apart
    unsafe {
        [
            unpack(lds_word(row)),
            unpack(lds_word(row + 16)),
            unpack(lds_word(row + 32)),
            unpack(lds_word(row + 48)),
        ]
    }
}

/// `B^T` applied to four rows of one channel, the rows given as `[row][column]`
#[inline(always)]
fn rows_transform(d: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    let mut column = 0;
    while column < 4 {
        out[0][column] = d[0][column] - d[2][column];
        out[1][column] = d[1][column] + d[2][column];
        out[2][column] = d[2][column] - d[1][column];
        out[3][column] = d[1][column] - d[3][column];
        column += 1;
    }
    out
}

/// `B` applied to one row, scaled by 1/4; exact apart from the sums' rounding
#[inline(always)]
fn column_transform(r: [f32; 4]) -> [f32; 4] {
    [
        (r[0] - r[2]) * 0.25,
        (r[1] + r[2]) * 0.25,
        (r[2] - r[1]) * 0.25,
        (r[1] - r[3]) * 0.25,
    ]
}

/// Transforms the 4x4 patch of one tile and channel pair from a raw stage into its 16
/// `V` words
///
/// # Safety
///
/// `raw` must be the word offset of a published raw stage and `v` that of a `V` stage
/// whose slots of (`pair`, `tile`) only this thread writes before the next barrier
#[inline(always)]
unsafe fn transform(smem: u32, raw: u32, v: u32, pair: u32, tile: u32) {
    let src = opaque(smem + (raw + pair / 4 * GROUP_WORDS + 2 * tile * 4 + pair % 4) * 4);
    const ROW: u32 = RAW_COLUMNS * 4 * 4;
    // safety: inside the published raw stage per this function's contract
    let d = unsafe {
        [
            raw_row(src),
            raw_row(src + ROW),
            raw_row(src + 2 * ROW),
            raw_row(src + 3 * ROW),
        ]
    };
    // written out: helper functions get no `#[unroll]`, and constant indices keep the
    // values in registers
    let lo = rows_transform([
        [d[0][0][0], d[0][1][0], d[0][2][0], d[0][3][0]],
        [d[1][0][0], d[1][1][0], d[1][2][0], d[1][3][0]],
        [d[2][0][0], d[2][1][0], d[2][2][0], d[2][3][0]],
        [d[3][0][0], d[3][1][0], d[3][2][0], d[3][3][0]],
    ]);
    let hi = rows_transform([
        [d[0][0][1], d[0][1][1], d[0][2][1], d[0][3][1]],
        [d[1][0][1], d[1][1][1], d[1][2][1], d[1][3][1]],
        [d[2][0][1], d[2][1][1], d[2][2][1], d[2][3][1]],
        [d[3][0][1], d[3][1][1], d[3][2][1], d[3][3][1]],
    ]);
    let dst = opaque(smem + (v + v_word(0, pair, tile)) * 4);
    // safety: this thread's slots per this function's contract
    unsafe {
        store_row(dst, 0, column_transform(lo[0]), column_transform(hi[0]));
        store_row(dst, 1, column_transform(lo[1]), column_transform(hi[1]));
        store_row(dst, 2, column_transform(lo[2]), column_transform(hi[2]));
        store_row(dst, 3, column_transform(lo[3]), column_transform(hi[3]));
    }
}

/// Stores elements `4a..4a + 4` of one tile and pair as packed FP16 words
///
/// # Safety
///
/// `dst` must address this thread's element-0 word of a `V` stage it alone writes
#[inline(always)]
unsafe fn store_row(dst: u32, a: u32, lo: [f32; 4], hi: [f32; 4]) {
    const STEP: u32 = (CHUNK / 2) * TILES * 4;
    let row = dst + 4 * a * STEP;
    // safety: forwarded from this function's contract
    unsafe {
        sts_word(row, pack(lo[0], hi[0]));
        sts_word(row + STEP, pack(lo[1], hi[1]));
        sts_word(row + 2 * STEP, pack(lo[2], hi[2]));
        sts_word(row + 3 * STEP, pack(lo[3], hi[3]));
    }
}

/// Expands to one FP16 Winograd convolution for a fixed same-channel shape and half
/// operand set
macro_rules! wh16_3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        channels = $channels:expr,
        h = $h:expr,
        w = $w:expr,
        half_input = $half_input:literal,
        half_residual = $half_residual:literal,
        half_output = $half_output:literal $(,)?
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
            mut y: DisjointSlice<f32>,
            mut range: DisjointSlice<f32>,
        ) {
            const HALF_IN: bool = $half_input;
            const HALF_RES: bool = $half_residual;
            const HALF_OUT: bool = $half_output;
            const C: u32 = $channels;
            const H: u32 = $h;
            const W: u32 = $w;
            const HW: u32 = H * W;
            const TH: u32 = H.div_ceil(2);
            const TW: u32 = W.div_ceil(2);
            const XB: u32 = TW.div_ceil(TILES);
            const T: u32 = TILES;
            const CHUNKS: u32 = C / CHUNK;
            const GROUPS: u32 = C / 8;
            // an FP32 input stage: one channel pair and raw row per thread group of 8,
            // every 8th column
            const FP32_PASSES: usize = RAW_COLUMNS.div_ceil(8) as usize;
            const _: () = assert!(C % CHANNELS == 0 && C % CHUNK == 0 && CHUNKS >= 2);

            let x_words = if HALF_IN { batch * C * HW / 2 } else { batch * C * HW };
            let y_words = if HALF_OUT { batch * C * HW / 2 } else { batch * C * HW };
            let residual_words = if HALF_RES { batch * C * HW / 2 } else { batch * C * HW };
            let cells = batch * TH * XB;
            let cotile = thread::blockIdx_x();
            let cell = thread::blockIdx_y();
            // the host sizes every buffer; a mismatch must not touch other memory
            if x_words as usize > x.len()
                || y_words as usize > y.len()
                || (add_residual != 0 && residual_words as usize > residual.len())
                || (C * C * 8) as usize > weight.len()
                || C as usize > bias.len()
                || range.len() == 0
                || cotile >= C / CHANNELS
                || cell >= cells
            {
                return;
            }
            let item = cell / (TH * XB);
            let ty = cell / XB % TH;
            let tx0 = cell % XB * T;

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let g = lane / 4;
            let t4 = lane % 4;
            // safety: the dynamic shared base of this CTA; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
            let x_ptr = x.as_ptr();
            // nonzero once this thread converts an activation that saturates
            let mut out_of_range = 0u32;

            // half staging: warp `w` copies channel group `w / 4`, raw row `w % 4`, at
            // columns `lane`, `lane + 32` and `lane + 64`
            let copy_group = warp / 4;
            let copy_row = warp % 4;
            let copy_iy = (2 * ty + copy_row) as i32 - 1;
            let copy_row_inside = copy_iy >= 0 && copy_iy < H as i32;
            let copy_iy = copy_iy.clamp(0, H as i32 - 1) as u32;
            // FP32 staging: lanes `8r..8r + 8` of warp `p` convert channel pair `p` of
            // raw row `r`, every 8th column
            let fp32_pair = warp;
            let fp32_row = lane / 8;
            let fp32_iy = (2 * ty + fp32_row) as i32 - 1;
            let fp32_row_inside = fp32_iy >= 0 && fp32_iy < H as i32;
            let fp32_iy = fp32_iy.clamp(0, H as i32 - 1) as u32;
            let mut staged = [[0.0f32; 2]; FP32_PASSES];

            // transform slots: tile `tid / 8`, channel pair `tid % 8`
            let vt = tid / 8;
            let vp = tid % 8;
            let early = warp < 4;

            // this warp's U fragments: element 2 warp, tile 0, this lane
            // safety: offsets below stay inside the checked packed weights
            let fragments = unsafe {
                weight.as_ptr().add(((cotile * CHUNKS * 16 + 2 * warp) * 4 * FRAGMENT + lane * 4) as usize)
            };
            let mut acc = [[[[0.0f32; 4]; 4]; 4]; 2];
            let mut frag = [[[0u32; 4]; 4]; 2];

            // prologue: raw stages of chunks 0 and 1, fragments of chunk 0, V of chunk 0
            let mut stage = 0;
            #[unroll]
            while stage < 2 {
                let chunk = stage as u32;
                let raw = smem + chunk * RAW_WORDS * 4;
                if HALF_IN {
                    let group = (item * GROUPS + chunk * 2 + copy_group) * HW + copy_iy * W;
                    let mut j = 0;
                    #[unroll]
                    while j < 3 {
                        let column = lane + 32 * j as u32;
                        let ix = column as i32 - 1 + 2 * tx0 as i32;
                        let valid = copy_row_inside && ix >= 0 && ix < W as i32;
                        let ix = ix.clamp(0, W as i32 - 1) as u32;
                        if column < RAW_COLUMNS {
                            let dst = raw + (copy_group * GROUP_WORDS + (copy_row * RAW_COLUMNS + column) * 4) * 4;
                            // safety: valid copies stay inside the checked half input; others
                            // read nothing
                            unsafe { copy16(dst, x_ptr.add(((group + ix) * 4) as usize), valid) };
                        }
                        j += 1;
                    }
                    commit();
                } else {
                    let plane = (item * C + chunk * CHUNK + 2 * fp32_pair) * HW + fp32_iy * W;
                    let mut j = 0;
                    #[unroll]
                    while j < FP32_PASSES {
                        let column = lane % 8 + 8 * j as u32;
                        let ix = column as i32 - 1 + 2 * tx0 as i32;
                        let valid = fp32_row_inside && column < RAW_COLUMNS && ix >= 0 && ix < W as i32;
                        let ix = ix.clamp(0, W as i32 - 1) as u32;
                        // safety: inside the checked input; padding reads a clamped word and
                        // discards it
                        let pair = unsafe { [ldg(x_ptr.add((plane + ix) as usize)), ldg(x_ptr.add((plane + HW + ix) as usize))] };
                        let (word, flag) = half2_checked(pair[0], pair[1]);
                        out_of_range |= if valid { flag } else { 0 };
                        if column < RAW_COLUMNS {
                            let dst = raw + (fp32_pair / 4 * GROUP_WORDS + (fp32_row * RAW_COLUMNS + column) * 4 + fp32_pair % 4) * 4;
                            // safety: this thread's slot of the stage, published by the barrier below
                            unsafe { sts_word(dst, if valid { word } else { 0 }) };
                        }
                        j += 1;
                    }
                }
                stage += 1;
            }
            let mut e = 0;
            #[unroll]
            while e < 2 {
                let mut m = 0;
                #[unroll]
                while m < 4 {
                    // safety: fragments of chunk 0 of this warp's elements
                    unsafe { frag[e][m] = fragment(fragments.add(((e * 4 + m) as u32 * FRAGMENT) as usize)) };
                    m += 1;
                }
                e += 1;
            }
            if HALF_IN {
                wait_all();
            }
            thread::sync_threads();
            // safety: reads published raw stage 0 and writes this thread's V slots
            unsafe { transform(smem, 0, V_BASE, vp, vt) };
            thread::sync_threads();

            let mut c = 0;
            while c < CHUNKS {
                let refill = c + 2 < CHUNKS;
                // raw(c + 2) goes into the stage of raw(c - 1), transformed in iteration c - 2
                let raw_next = opaque(smem + (c + 2) % RAW_STAGES * RAW_WORDS * 4);
                if refill {
                    if HALF_IN {
                        let group = (item * GROUPS + (c + 2) * 2 + copy_group) * HW + copy_iy * W;
                        let mut j = 0;
                        #[unroll]
                        while j < 3 {
                            let column = lane + 32 * j as u32;
                            let ix = column as i32 - 1 + 2 * tx0 as i32;
                            let valid = copy_row_inside && ix >= 0 && ix < W as i32;
                            let ix = ix.clamp(0, W as i32 - 1) as u32;
                            if column < RAW_COLUMNS {
                                let dst = raw_next + (copy_group * GROUP_WORDS + (copy_row * RAW_COLUMNS + column) * 4) * 4;
                                // safety: valid copies stay inside the checked half input
                                unsafe { copy16(dst, x_ptr.add(((group + ix) * 4) as usize), valid) };
                            }
                            j += 1;
                        }
                        commit();
                    } else {
                        let plane = (item * C + (c + 2) * CHUNK + 2 * fp32_pair) * HW + fp32_iy * W;
                        let mut j = 0;
                        #[unroll]
                        while j < FP32_PASSES {
                            let column = lane % 8 + 8 * j as u32;
                            let ix = (column as i32 - 1 + 2 * tx0 as i32).clamp(0, W as i32 - 1) as u32;
                            // safety: inside the checked input; padding reads a clamped word
                            staged[j] = unsafe { [ldg(x_ptr.add((plane + ix) as usize)), ldg(x_ptr.add((plane + HW + ix) as usize))] };
                            j += 1;
                        }
                    }
                }

                let raw = (c + 1) % RAW_STAGES * RAW_WORDS;
                let v_next = V_BASE + (c + 1) % 2 * V_WORDS;
                let transform_next = c + 1 < CHUNKS;
                if early && transform_next {
                    // safety: reads raw(c + 1), published by the last barrier, and writes V
                    // stage (c + 1) % 2, last read in iteration c - 1
                    unsafe { transform(smem, raw, v_next, vp, vt) };
                }
                let sv = opaque(smem + (V_BASE + c % 2 * V_WORDS) * 4);
                let mut e = 0;
                #[unroll]
                while e < 2 {
                    let element = 2 * warp + e as u32;
                    let mut n = 0;
                    #[unroll]
                    while n < 4 {
                        let tile = 8 * n as u32 + g;
                        // safety: words of the published V stage
                        let (b0, b1) = unsafe {
                            (
                                lds_word(sv + v_word(element, t4, tile) * 4),
                                lds_word(sv + v_word(element, t4 + 4, tile) * 4),
                            )
                        };
                        let mut m = 0;
                        #[unroll]
                        while m < 4 {
                            acc[e][m][n] = mma16(acc[e][m][n], frag[e][m], b0, b1);
                            m += 1;
                        }
                        n += 1;
                    }
                    if transform_next {
                        // this element's fragments of the next chunk land while the other
                        // element computes
                        let mut m = 0;
                        #[unroll]
                        while m < 4 {
                            let block = ((c + 1) * 16 * 4 + (e * 4 + m) as u32) * FRAGMENT;
                            // safety: fragments of a chunk of this warp's elements
                            unsafe { frag[e][m] = fragment(fragments.add(block as usize)) };
                            m += 1;
                        }
                    }
                    e += 1;
                }
                if !early && transform_next {
                    // safety: as the early transform above
                    unsafe { transform(smem, raw, v_next, vp, vt) };
                }
                if refill && !HALF_IN {
                    let mut j = 0;
                    #[unroll]
                    while j < FP32_PASSES {
                        let column = lane % 8 + 8 * j as u32;
                        let ix = column as i32 - 1 + 2 * tx0 as i32;
                        let valid = fp32_row_inside && column < RAW_COLUMNS && ix >= 0 && ix < W as i32;
                        let (word, flag) = half2_checked(staged[j][0], staged[j][1]);
                        out_of_range |= if valid { flag } else { 0 };
                        if column < RAW_COLUMNS {
                            let dst = raw_next + (fp32_pair / 4 * GROUP_WORDS + (fp32_row * RAW_COLUMNS + column) * 4 + fp32_pair % 4) * 4;
                            // safety: this thread's slot of the stage of raw(c - 1), which the
                            // transform of iteration c - 2 last read
                            unsafe { sts_word(dst, if valid { word } else { 0 }) };
                        }
                        j += 1;
                    }
                }
                if HALF_IN {
                    wait_all();
                }
                // publishes raw(c + 2) and V(c + 1) and closes all reads of V(c)
                thread::sync_threads();
                c += 1;
            }
            if out_of_range != 0 {
                // safety: `range` holds at least one word, checked above
                unsafe { flag_out_of_range(range.as_mut_ptr()) };
            }

            // round `r` moves 16-channel tile `r` of both elements to
            // `M[element][16][EPILOGUE_ROW]`; thread `(pair, t)` then takes channels
            // `2 pair` and `2 pair + 1` of tile `t`
            let base = opaque(smem);
            let pixels = opaque(smem + M_WORDS * 4);
            let pair = tid / 32;
            let t = lane;
            let tx = tx0 + t;
            let tile_inside = tx < TW;
            let residual_ptr = residual.as_ptr();
            let y_ptr = y.as_mut_ptr();
            let mut output_out_of_range = 0u32;
            let mut r = 0;
            #[unroll]
            while r < 4 {
                let channel0 = cotile * CHANNELS + 16 * r as u32 + 2 * pair;
                // the residual may alias the output, so this round's residual is read
                // before any of its outputs is stored
                let mut shortcut = [[0.0f32; 2]; 4];
                if add_residual != 0 && tile_inside {
                    let mut k = 0;
                    #[unroll]
                    while k < 4 {
                        let (oy, ox) = (2 * ty + k as u32 / 2, 2 * tx + k as u32 % 2);
                        if oy < H && ox < W {
                            if HALF_RES {
                                let group = item * GROUPS + channel0 / 8;
                                let word_index = (group * HW + oy * W + ox) * 4 + pair % 4;
                                // safety: inside the checked half residual
                                let word = unsafe { ld_word(residual_ptr.add(word_index as usize)) };
                                let [lo, hi] = unpack(word);
                                shortcut[k] = [lo * RESIDUAL_UNSCALE, hi * RESIDUAL_UNSCALE];
                            } else {
                                let at = (item * C + channel0) * HW + oy * W + ox;
                                // safety: inside the checked residual
                                shortcut[k] = unsafe { [*residual_ptr.add(at as usize), *residual_ptr.add((at + HW) as usize)] };
                            }
                        }
                        k += 1;
                    }
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
                            sts_word(low, acc[e][r][n][0].to_bits());
                            sts_word(low + 4, acc[e][r][n][1].to_bits());
                            sts_word(high, acc[e][r][n][2].to_bits());
                            sts_word(high + 4, acc[e][r][n][3].to_bits());
                        }
                        n += 1;
                    }
                    e += 1;
                }
                thread::sync_threads();

                if tile_inside {
                    // `values[channel][k]` with output pixel `k` of the tile, row-major
                    let mut values = [[0.0f32; 4]; 2];
                    let mut ch = 0;
                    #[unroll]
                    while ch < 2 {
                        let row = 2 * pair + ch as u32;
                        let mut m = [0.0f32; 16];
                        let mut e = 0;
                        #[unroll]
                        while e < 16 {
                            // safety: written before the barrier above
                            m[e] = f32::from_bits(unsafe { lds_word(base + ((e as u32 * 16 + row) * EPILOGUE_ROW + t) * 4) });
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
                        let channel = channel0 + ch as u32;
                        // safety: `channel < C`, inside the checked bias
                        let bias_value = unsafe { *bias.as_ptr().add(channel as usize) };
                        let mut a = 0;
                        #[unroll]
                        while a < 2 {
                            let (c0, c1, c2, c3) = (columns[4 * a], columns[4 * a + 1], columns[4 * a + 2], columns[4 * a + 3]);
                            let pair_values = [(c0 + c1 + c2) * PRODUCT_UNSCALE, (c1 - c2 - c3) * PRODUCT_UNSCALE];
                            let mut b = 0;
                            #[unroll]
                            while b < 2 {
                                // the residual goes in before the bias, as in cuDNN's fused call
                                values[ch][2 * a + b] = relu(pair_values[b] + shortcut[2 * a + b][ch] + bias_value);
                                b += 1;
                            }
                            a += 1;
                        }
                        ch += 1;
                    }

                    let mut k = 0;
                    #[unroll]
                    while k < 4 {
                        let (oy, ox) = (2 * ty + k as u32 / 2, 2 * tx + k as u32 % 2);
                        if HALF_OUT {
                            let (word, flag) = half2_checked(values[0][k], values[1][k]);
                            let inside = oy < H && ox < W;
                            output_out_of_range |= if inside { flag } else { 0 };
                            let slot = pixels + ((pair / 4 * 2 + k as u32 / 2) * 2 * T + 2 * t + k as u32 % 2) * 16 + pair % 4 * 4;
                            // safety: this thread's word of the pixel tile, published below
                            unsafe { sts_word(slot, word) };
                        } else if oy < H && ox < W {
                            let at = (item * C + channel0) * HW + oy * W + ox;
                            // safety: inside the checked output; this thread is its only writer
                            unsafe {
                                *y.get_unchecked_mut(at as usize) = values[0][k];
                                *y.get_unchecked_mut((at + HW) as usize) = values[1][k];
                            }
                        }
                        k += 1;
                    }
                }
                // publishes the pixel tile and closes this round's reads of the product tile
                thread::sync_threads();

                if HALF_OUT {
                    // thread `(group, row, column)` stores one 16-byte pixel
                    let group = tid / (4 * T);
                    let row = tid / (2 * T) % 2;
                    let column = tid % (2 * T);
                    let (oy, ox) = (2 * ty + row, 2 * tx0 + column);
                    if oy < H && ox < W {
                        // safety: written before the barrier above
                        let words = unsafe { lds4(pixels + ((group * 2 + row) * 2 * T + column) * 16) };
                        let at = ((item * GROUPS + (cotile * CHANNELS + 16 * r as u32) / 8 + group) * HW + oy * W + ox) * 4;
                        // safety: inside the checked half output, 16-byte aligned; this thread
                        // is the pixel's only writer
                        unsafe { stg4(y_ptr.add(at as usize), words) };
                    }
                    // closes this round's reads of the pixel tile before the next round's writes
                    thread::sync_threads();
                }
                r += 1;
            }
            if HALF_OUT && output_out_of_range != 0 {
                // safety: `range` holds at least one word, checked above
                unsafe { flag_out_of_range(range.as_mut_ptr()) };
            }
        }
    };
}

wh16_3x3! {
    /// FP16 Winograd `y = relu(conv3x3(x, weight) + bias [+ residual])` for 128 channels,
    /// stride 1, on 20x250 inputs, with the operands of `spk_wideconv_h16_c128`
    ///
    /// `weight` comes from [`spk_wideconv_pack_wh16`]; `range` is one word the launch sets
    /// nonzero when an activation saturates. Launch 256 threads with `grid = (2, batch *
    /// 40)` and [`WH16_SHARED_BYTES`] of dynamic shared memory
    spk_wideconv_wh16_c128,
    channels = 128,
    h = 20,
    w = 250,
    half_input = false,
    half_residual = false,
    half_output = false,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c128`, writing `y` as the half tensor a block's second convolution reads, like `spk_wideconv_h16_c128_half_out`
    spk_wideconv_wh16_c128_half_out,
    channels = 128,
    h = 20,
    w = 250,
    half_input = false,
    half_residual = false,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c128`, reading `x` as a half tensor, like `spk_wideconv_h16_c128_half_in`
    spk_wideconv_wh16_c128_half_in,
    channels = 128,
    h = 20,
    w = 250,
    half_input = true,
    half_residual = false,
    half_output = false,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c128`, reading `x` and writing `y` as half tensors, like `spk_wideconv_h16_c128_half_in_out`
    spk_wideconv_wh16_c128_half_in_out,
    channels = 128,
    h = 20,
    w = 250,
    half_input = true,
    half_residual = false,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c128`, with `x`, `residual` and `y` all half tensors, like `spk_wideconv_h16_c128_half_all`
    spk_wideconv_wh16_c128_half_all,
    channels = 128,
    h = 20,
    w = 250,
    half_input = true,
    half_residual = true,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c128`, reading `x` and `residual` as half tensors and writing `y` as FP32, like `spk_wideconv_h16_c128_half_in_res`
    spk_wideconv_wh16_c128_half_in_res,
    channels = 128,
    h = 20,
    w = 250,
    half_input = true,
    half_residual = true,
    half_output = false,
}

wh16_3x3! {
    /// FP16 Winograd `y = relu(conv3x3(x, weight) + bias [+ residual])` for 256 channels,
    /// stride 1, on 10x125 inputs, with the operands of `spk_wideconv_h16_c256`
    ///
    /// `weight` comes from [`spk_wideconv_pack_wh16`]; `range` is one word the launch sets
    /// nonzero when an activation saturates. Launch 256 threads with `grid = (4, batch *
    /// 10)` and [`WH16_SHARED_BYTES`] of dynamic shared memory
    spk_wideconv_wh16_c256,
    channels = 256,
    h = 10,
    w = 125,
    half_input = false,
    half_residual = false,
    half_output = false,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c256`, writing `y` as the half tensor a block's second convolution reads, like `spk_wideconv_h16_c256_half_out`
    spk_wideconv_wh16_c256_half_out,
    channels = 256,
    h = 10,
    w = 125,
    half_input = false,
    half_residual = false,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c256`, reading `x` as a half tensor, like `spk_wideconv_h16_c256_half_in`
    spk_wideconv_wh16_c256_half_in,
    channels = 256,
    h = 10,
    w = 125,
    half_input = true,
    half_residual = false,
    half_output = false,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c256`, reading `x` and writing `y` as half tensors, like `spk_wideconv_h16_c256_half_in_out`
    spk_wideconv_wh16_c256_half_in_out,
    channels = 256,
    h = 10,
    w = 125,
    half_input = true,
    half_residual = false,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c256`, with `x`, `residual` and `y` all half tensors, like `spk_wideconv_h16_c256_half_all`
    spk_wideconv_wh16_c256_half_all,
    channels = 256,
    h = 10,
    w = 125,
    half_input = true,
    half_residual = true,
    half_output = true,
}

wh16_3x3! {
    /// As `spk_wideconv_wh16_c256`, reading `x` and `residual` as half tensors and writing `y` as FP32, like `spk_wideconv_h16_c256_half_in_res`
    spk_wideconv_wh16_c256_half_in_res,
    channels = 256,
    h = 10,
    w = 125,
    half_input = true,
    half_residual = true,
    half_output = false,
}
