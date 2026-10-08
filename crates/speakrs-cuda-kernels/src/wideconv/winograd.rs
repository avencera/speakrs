//! Fused FP32 Winograd F(2x2, 3x3) for the same-channel stride-1 3x3 convolutions
//!
//! A CTA has 256 threads and owns 64 output channels by one row of 32 2x2 output
//! tiles. Warp `w` accumulates Winograd elements `2w` and `2w + 1`; per element a lane
//! holds 8 channels by 8 tiles. Input channels stream through two shared stages of
//! four channels, in a pipeline whose iteration `c`
//!
//! - loads the raw 4 x 66 input window of chunk `c + 2` and the weights of chunk
//!   `c + 1`, which the pack kernel already transformed to `U = G g G^T`
//! - accumulates chunk `c` from shared memory
//! - transforms the raw window of chunk `c + 1`, stored one iteration earlier, into
//!   `V = B^T d B`, half a patch per thread
//! - stores the loaded window and weights
//!
//! Raw windows load with one coalesced word per thread and row segment instead of a
//! 4x4 patch per tile. Out-of-image words load from clamped addresses and are zeroed
//! with precomputed bit masks, so the loads carry no predicates, and the window lands
//! in shared memory split by column parity, which puts the patch reads of 32
//! consecutive tiles on 32 banks. Weights stay in global memory between chunks; they
//! are the operand most sensitive to L2 bandwidth, which bounds this kernel on parts
//! with 128 FP32 lanes per SM
//!
//! The epilogue moves the products through shared memory in four rounds of 16
//! channels and applies `A^T M A`, then either the residual, the bias and a
//! NaN-propagating ReLU, or, with `splits > 1`, stores the raw partial sum of this
//! CTA's input-channel split for the fixed-order `spk_wideconv_reduce`
//!
//! `spk_wideconv_wino_c128_sweep2` halves the FP32 accumulation chains: a CTA runs its
//! input channels in two sweeps, each from its own prologue to its own epilogue, and the
//! second sweep adds its output-domain partial sum to the first's, read back from the
//! words this thread stored. On a 4060 Ti, where cuDNN's FP32 mode runs its own fused
//! Winograd, the single 128-term chain sat at cuDNN's error on the 128-channel layers;
//! two sweeps bring every layer below it (host emulation of the reference samples). Each
//! extra sweep cost about 8% at batch 32 there, so four sweeps were too slow
//!
//! Everything runs in full FP32 in both math modes and both tiers. F(2x2, 3x3) only
//! adds and subtracts in its data transforms, and on the reference activations its
//! error against an f64 truth is at or below that of cuDNN's fused FP32 Winograd

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

use super::{ldg4, lds1, lds4, opaque, stg2, sts1, sts4};

/// Output channels per Winograd CTA
pub const WINO_CHANNELS: u32 = 64;
/// 2x2 output tiles per Winograd CTA, consecutive in one tile row
pub const WINO_TILES: u32 = 32;
/// Threads per Winograd CTA
const WINO_THREADS: u32 = 256;
/// Input channels per pipeline stage
const WINO_CHUNK: u32 = 4;
/// Raw input columns per channel and row: two per tile plus the left and right halo
const RAW_COLUMNS: u32 = 2 * WINO_TILES + 2;
/// Words per column-parity half of a raw row; 48 puts the halves 16 banks apart
const RAW_PARITY: u32 = 48;
/// Words per raw row: both parity halves plus 8, which puts the two rows a warp stores
/// on different banks
const RAW_ROW: u32 = 2 * RAW_PARITY + 8;
/// Raw words per stage: four rows of each chunk channel
const RAW_WORDS: u32 = WINO_CHUNK * 4 * RAW_ROW;
/// Threads per raw row: each copies every 16th column of one (channel, row)
const RAW_LANES: u32 = WINO_THREADS / (WINO_CHUNK * 4);
/// Columns one thread copies per stage; columns past 65 land in words 33..39 of their
/// parity half, which no tile reads
const RAW_PASSES: u32 = RAW_COLUMNS.div_ceil(RAW_LANES);
/// Transformed input words per stage
const V_WORDS: u32 = WINO_CHUNK * 16 * WINO_TILES;
/// Transformed weight words per stage
const U_WORDS: u32 = WINO_CHUNK * 16 * WINO_CHANNELS;
/// Weight vectors one thread moves per stage
const U_QUADS: u32 = U_WORDS / 4 / WINO_THREADS;
/// Epilogue rounds; each moves 16 output channels of all 16 elements
const ROUNDS: u32 = 4;
/// Output channels per epilogue round
const ROUND_CHANNELS: u32 = WINO_CHANNELS / ROUNDS;
/// Words per epilogue row; 40 keeps the two channel rows of a quarter warp 16 banks apart
const EPILOGUE_ROW: u32 = 40;
/// Epilogue passes per round: one (channel, tile) pair per thread and pass
const PASSES: u32 = ROUND_CHANNELS * WINO_TILES / WINO_THREADS;
/// Dynamic shared bytes of a Winograd launch: two pipeline stages, which also hold the
/// epilogue tile
pub const WINO_SHARED_BYTES: u32 = 2 * (RAW_WORDS + V_WORDS + U_WORDS) * 4;

const _: () = assert!(
    2 * WINO_CHUNK * WINO_TILES == WINO_THREADS && U_WORDS.is_multiple_of(4 * WINO_THREADS)
);
// one (channel, row) per 16 threads; the last pass stays inside its parity halves
const _: () =
    assert!(RAW_LANES * WINO_CHUNK * 4 == WINO_THREADS && RAW_PASSES * RAW_LANES / 2 <= RAW_PARITY);
const _: () = assert!(16 * ROUND_CHANNELS * EPILOGUE_ROW * 4 <= WINO_SHARED_BYTES);
const _: () = assert!((ROUND_CHANNELS * WINO_TILES).is_multiple_of(WINO_THREADS));

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

/// Transforms folded weights `[cout][cin][3][3]` to `U = G g G^T`
///
/// Layout `[cout / 64][cin][16][64]`: one chunk of input channels of one CTA's 64
/// output channels is contiguous. The transform sums in f64 and rounds once
///
/// Launch one thread per element of `packed`, which has `cout * cin * 16` elements
#[kernel]
pub fn spk_wideconv_pack_winograd(
    weight: &[f32],
    cin: u32,
    cout: u32,
    mut packed: DisjointSlice<f32>,
) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let lane = index % WINO_CHANNELS;
    let element = index / WINO_CHANNELS % 16;
    let channel_in = index / (WINO_CHANNELS * 16) % cin;
    let channel_out = index / (WINO_CHANNELS * 16 * cin) * WINO_CHANNELS + lane;
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
    if let Some(out) = packed.get_mut(i) {
        *out = sum as f32;
    }
}

/// One word through the read-only path; the caller masks it
#[inline(always)]
unsafe fn ldg_bits(pointer: *const f32) -> u32 {
    let value: u32;
    // safety: the caller passes a readable word of a buffer no kernel writes during
    // the launch
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.nc.u32 %0, [g]; }",
            out("=r") value,
            in("l") pointer as u64,
        );
    }
    value
}

/// One float through the read-only path, or zero without a load when `valid` is false
#[inline(always)]
pub(super) unsafe fn ldg1(pointer: *const f32, valid: bool) -> f32 {
    let value: f32;
    let valid = u32::from(valid);
    // safety: the caller passes a readable word of a buffer no kernel writes during
    // the launch whenever `valid` is set; otherwise the predicated load does not run
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; .reg .pred p; cvta.to.global.u64 g, %1; setp.ne.u32 p, %2, 0; mov.f32 %0, 0f00000000; @p ld.global.nc.f32 %0, [g]; }",
            out("=f") value,
            in("l") pointer as u64,
            in("r") valid,
        );
    }
    value
}

/// Passes `value` through a side-effecting register move
///
/// Thread geometry derived from the result is recomputed in every pipeline iteration;
/// otherwise LLVM keeps it live across the main loop, where it competes with the
/// accumulators for registers and ptxas spills
#[inline(always)]
fn fresh(value: u32) -> u32 {
    let out: u32;
    // safety: a register move with no memory access
    unsafe {
        ptx_asm!("mov.u32 %0, %1;", out("=r") out, in("r") value);
    }
    out
}

/// Output channel of lane-local channel `i` of lane group `kq`
#[inline(always)]
fn lane_channel(kq: u32, i: u32) -> u32 {
    if i < 4 {
        4 * kq + i
    } else {
        WINO_CHANNELS / 2 + 4 * kq + i - 4
    }
}

/// Geometry of one Winograd CTA's output tile
struct Site {
    c: u32,
    h: u32,
    w: u32,
    item: u32,
    ty: u32,
    tx0: u32,
    cotile: u32,
}

impl Site {
    /// Tile column and output channel of pass `pass` of round `r` of thread `tid`
    #[inline(always)]
    fn pass(&self, tid: u32, pass: u32, r: u32) -> (u32, u32) {
        let q = tid + pass * WINO_THREADS;
        let kr = q / WINO_TILES;
        let lane_channel = lane_channel(kr / 2, r * 2 + kr % 2);
        (
            self.tx0 + q % WINO_TILES,
            self.cotile * WINO_CHANNELS + lane_channel,
        )
    }

    /// Word of output pixel (`oy`, `ox`) of channel `channel`
    #[inline(always)]
    fn word(&self, channel: u32, oy: u32, ox: u32) -> u32 {
        ((self.item * self.c + channel) * self.h + oy) * self.w + ox
    }
}

/// Optional output-sized operand of a Winograd epilogue: the residual, or the partial
/// sum an earlier pass of this CTA stored in its own output words
struct Residual {
    words: *const f32,
    enabled: bool,
    /// Word offset of the operand's plane: a split CTA's partial plane, else zero
    plane: u32,
    /// Loads through the coherent path, for words this launch wrote
    coherent: bool,
}

impl Residual {
    /// Residual words of epilogue round `r` of thread `tid`, `[pass][2 row + column]`
    /// of the pass's 2x2 output tile, zero where the residual is off or the pixel is
    /// outside the output
    ///
    /// Every word is a predicated load, so none sits in a conditional block the
    /// compiler could sink below the round's shared-memory work
    ///
    /// # Safety
    ///
    /// When enabled, `words` must hold every output word of the launch, readable and
    /// unwritten by any kernel during the launch
    #[inline(always)]
    unsafe fn round(&self, site: &Site, tid: u32, r: u32) -> [[f32; 4]; 2] {
        // written out so no loop over the arrays can leave them in local memory
        // safety: forwarded from this function's contract
        unsafe {
            [
                [
                    self.word(site, tid, 0, r, 0),
                    self.word(site, tid, 0, r, 1),
                    self.word(site, tid, 0, r, 2),
                    self.word(site, tid, 0, r, 3),
                ],
                [
                    self.word(site, tid, 1, r, 0),
                    self.word(site, tid, 1, r, 1),
                    self.word(site, tid, 1, r, 2),
                    self.word(site, tid, 1, r, 3),
                ],
            ]
        }
    }

    /// Residual word `k` (`2 row + column`) of pass `pass` of round `r` of thread `tid`
    ///
    /// # Safety
    ///
    /// As for [`Residual::round`]
    #[inline(always)]
    unsafe fn word(&self, site: &Site, tid: u32, pass: u32, r: u32, k: u32) -> f32 {
        let (tx, channel) = site.pass(tid, pass, r);
        let (oy, ox) = (2 * site.ty + k / 2, 2 * tx + k % 2);
        let valid = self.enabled && oy < site.h && ox < site.w;
        let word = if valid {
            (self.plane + site.word(channel, oy, ox)) as usize
        } else {
            0
        };
        // safety: valid words lie inside the operand per this function's contract
        unsafe {
            if self.coherent {
                ld1(self.words.add(word), valid)
            } else {
                ldg1(self.words.add(word), valid)
            }
        }
    }
}

/// One float through the coherent path, or zero without a load when `valid` is false
///
/// For words this launch wrote earlier: the non-coherent path may return stale data
///
/// # Safety
///
/// Whenever `valid` is set, `pointer` must address a readable word that no other thread
/// writes during the launch
#[inline(always)]
unsafe fn ld1(pointer: *const f32, valid: bool) -> f32 {
    let value: f32;
    let valid = u32::from(valid);
    // safety: forwarded from this function's contract; the predicated load does not run
    // when `valid` is clear
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; .reg .pred p; cvta.to.global.u64 g, %1; setp.ne.u32 p, %2, 0; mov.f32 %0, 0f00000000; @p ld.global.f32 %0, [g]; }",
            out("=f") value,
            in("l") pointer as u64,
            in("r") valid,
            clobber("memory"),
        );
    }
    value
}

/// ReLU that passes NaN through, like cuDNN's propagating activation
#[inline(always)]
pub(super) fn relu(value: f32) -> f32 {
    if value < 0.0 { 0.0 } else { value }
}

/// Expands to one fused Winograd convolution for a fixed same-channel shape
macro_rules! winograd3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        channels = $channels:expr,
        h = $h:expr,
        w = $w:expr,
        sweeps = $sweeps:expr $(,)?
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
            const THREADS: u32 = WINO_THREADS;
            const C: u32 = $channels;
            const H: u32 = $h;
            const W: u32 = $w;
            const HW: u32 = H * W;
            const TH: u32 = H.div_ceil(2);
            const TW: u32 = W.div_ceil(2);
            const XB: u32 = TW.div_ceil(WINO_TILES);
            const KB: u32 = WINO_CHANNELS;
            const T: u32 = WINO_TILES;
            const CC: u32 = WINO_CHUNK;
            const RPT: usize = RAW_PASSES as usize;
            const QUADS: usize = U_QUADS as usize;
            const STAGE_RAW: u32 = 0;
            const STAGE_V: u32 = 2 * RAW_WORDS;
            const STAGE_U: u32 = STAGE_V + 2 * V_WORDS;
            const SWEEPS: u32 = $sweeps;
            const _: () = assert!(C % KB == 0 && C % CC == 0 && SWEEPS > 0);

            let len = batch * C * HW;
            if len as usize > x.len()
                || splits == 0
                || C % (splits * CC * SWEEPS) != 0
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
            // each sweep reduces its share of the partition's input channels and stores or
            // accumulates its partial sum in the CTA's own output words
            let channels = C / parts / SWEEPS;
            let chunks = channels / CC;

            let tid = thread::threadIdx_x();
            // safety: the dynamic shared base of this CTA; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
            let x_ptr = x.as_ptr();
            let mut sweep = 0;
            while sweep < SWEEPS {
                let c0 = (split * SWEEPS + sweep) * channels;

                // raw window of this thread: (channel, row) `tid / 16` of each chunk and its
                // columns `tid % 16 + 16 j`; out-of-image words load from clamped addresses
                // and are zeroed by bit `j` of `raw_valid`; shared slots step 8 words per pass
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
                    let ix = col0 + 16 * j as i32;
                    let inside = row_inside && raw_lane + 16 * (j as u32) < RAW_COLUMNS && ix >= 0 && ix < W as i32;
                    raw_valid |= u32::from(inside) << j;
                    j += 1;
                }
                let raw_slot = raw_row * RAW_ROW + raw_lane % 2 * RAW_PARITY + raw_lane / 2;
                // transform slot: rows 0..2 (half 0) or 1..3 (half 1) of channel `vc`, tile `vt`
                let half = tid / (CC * T);
                let vc = tid % (CC * T) / T;
                let vt = tid % T;
                // safety: the CTA's weight block lies inside the checked packed weights
                let weight_block = unsafe { weight.as_ptr().add(((cotile * C + c0) * 16 * KB) as usize) };

                let mut acc = [[[0.0f32; 8]; 8]; 2];
                let mut quads = [[0.0f32; 4]; QUADS];

                // prologue: the raw windows of chunks 0 and 1 and the weights of chunk 0 load
                // together so their latencies overlap, then V of chunk 0
                let mut first = [[0u32; RPT]; 2];
                let mut stage = 0;
                #[unroll]
                while stage < 2 {
                    let chunk = if (stage as u32) < chunks { stage as u32 } else { chunks - 1 };
                    let mut i = 0;
                    #[unroll]
                    while i < RPT {
                        let column = (col0 + 16 * i as i32).clamp(0, W as i32 - 1) as u32;
                        let word = (chunk * CC * HW + row_offset + column) as usize;
                        let mask = ((raw_valid >> i) & 1).wrapping_neg();
                        // safety: clamped coordinates address words inside the checked input
                        first[stage][i] = unsafe { ldg_bits(x_ptr.add(word)) } & mask;
                        i += 1;
                    }
                    stage += 1;
                }
                let mut m = 0;
                #[unroll]
                while m < QUADS {
                    let q = tid + m as u32 * THREADS;
                    // safety: whole aligned vectors of this CTA's first chunk of the weights
                    quads[m] = unsafe { ldg4(weight_block.add((4 * q) as usize)) };
                    m += 1;
                }
                let mut stage = 0;
                #[unroll]
                while stage < 2 {
                    let base = smem + (STAGE_RAW + stage as u32 * RAW_WORDS) * 4;
                    let mut i = 0;
                    #[unroll]
                    while i < RPT {
                        // safety: one writer per slot of this stage
                        unsafe { sts1(base + (raw_slot + 8 * i as u32) * 4, f32::from_bits(first[stage][i])) };
                        i += 1;
                    }
                    stage += 1;
                }
                let mut m = 0;
                #[unroll]
                while m < QUADS {
                    // safety: aligned vectors of the first weight stage, one writer each
                    unsafe { sts4(smem + (STAGE_U + 4 * (tid + m as u32 * THREADS)) * 4, quads[m]) };
                    m += 1;
                }
                thread::sync_threads();
                // safety: reads the published raw stage 0 and writes this thread's V slots
                unsafe { transform(smem, STAGE_RAW, STAGE_V, half, vc, vt) };
                thread::sync_threads();

                // iteration `c` loads chunk `c + 2`'s raw window and chunk `c + 1`'s weights,
                // computes chunk `c`, transforms chunk `c + 1` and stores the loads; loads and
                // stores run unconditionally (clamped chunks reload the last one into idle
                // buffers), so no conditional block lets the compiler sink the loads below
                // the products; thread geometry is rederived from `fresh(tid)` each iteration
                // so it does not stay live next to the accumulators
                let mut raw = [0u32; RPT];
                let mut c = 0;
                while c < chunks {
                    let t = fresh(tid);
                    let next = if c + 1 < chunks { c + 1 } else { chunks - 1 };
                    let after = if c + 2 < chunks { c + 2 } else { chunks - 1 };
                    let raw_row = t / RAW_LANES;
                    let raw_lane = t % RAW_LANES;
                    let iy = ((2 * ty + raw_row % 4) as i32 - 1).clamp(0, H as i32 - 1) as u32;
                    let row_word = ((item * C + c0 + raw_row / 4) * H + iy) * W;
                    let col = (2 * tx0 + raw_lane) as i32 - 1;
                    let mut i = 0;
                    #[unroll]
                    while i < RPT {
                        let column = (col + 16 * i as i32).clamp(0, W as i32 - 1) as u32;
                        let word = (after * CC * HW + row_word + column) as usize;
                        let mask = ((raw_valid >> i) & 1).wrapping_neg();
                        // safety: clamped coordinates address words inside the checked input
                        raw[i] = unsafe { ldg_bits(x_ptr.add(word)) } & mask;
                        i += 1;
                    }
                    let mut m = 0;
                    #[unroll]
                    while m < QUADS {
                        let q = t + m as u32 * THREADS;
                        // safety: whole aligned vectors of this CTA's chunk of the weights
                        quads[m] = unsafe { ldg4(weight_block.add((next * U_WORDS + 4 * q) as usize)) };
                        m += 1;
                    }

                    let warp = t / 32;
                    let tq = t % 4;
                    let kq = t % 32 / 4;
                    let su = opaque(smem + (STAGE_U + c % 2 * U_WORDS) * 4);
                    let sv = opaque(smem + (STAGE_V + c % 2 * V_WORDS) * 4);
                    let mut ci = 0;
                    #[unroll]
                    while ci < CC {
                        let mut e = 0;
                        #[unroll]
                        while e < 2 {
                            let element = warp * 2 + e as u32;
                            let ua = su + ((ci * 16 + element) * KB + 4 * kq) * 4;
                            let va = sv + ((ci * 16 + element) * T + 4 * tq) * 4;
                            // safety: aligned vectors inside the published stage
                            let (u0, u1, v0, v1) = unsafe {
                                (lds4(ua), lds4(ua + KB / 2 * 4), lds4(va), lds4(va + T / 2 * 4))
                            };
                            let u = [u0[0], u0[1], u0[2], u0[3], u1[0], u1[1], u1[2], u1[3]];
                            let v = [v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]];
                            let mut i = 0;
                            #[unroll]
                            while i < 8 {
                                let mut j = 0;
                                #[unroll]
                                while j < 8 {
                                    acc[e][i][j] += u[i] * v[j];
                                    j += 1;
                                }
                                i += 1;
                            }
                            e += 1;
                        }
                        ci += 1;
                    }

                    // the V and weight buffers written here were last read in iteration c - 1,
                    // and the raw buffer of chunk c in iteration c - 1's transform
                    let t = fresh(tid);
                    // safety: reads raw stage (c + 1) % 2, published by the last barrier
                    unsafe {
                        transform(
                            smem,
                            STAGE_RAW + (c + 1) % 2 * RAW_WORDS,
                            STAGE_V + (c + 1) % 2 * V_WORDS,
                            t / (CC * T),
                            t % (CC * T) / T,
                            t % T,
                        )
                    };
                    let slot = t / RAW_LANES * RAW_ROW + t % 2 * RAW_PARITY + t % RAW_LANES / 2;
                    let base = opaque(smem + (STAGE_RAW + c % 2 * RAW_WORDS) * 4);
                    let mut i = 0;
                    #[unroll]
                    while i < RPT {
                        // safety: one writer per slot of this stage
                        unsafe { sts1(base + (slot + 8 * i as u32) * 4, f32::from_bits(raw[i])) };
                        i += 1;
                    }
                    let su = opaque(smem + (STAGE_U + (c + 1) % 2 * U_WORDS) * 4);
                    let mut m = 0;
                    #[unroll]
                    while m < QUADS {
                        // safety: aligned vectors of this stage, one writer each
                        unsafe { sts4(su + (t + m as u32 * THREADS) * 16, quads[m]) };
                        m += 1;
                    }
                    // publishes this iteration's stages and closes all reads of the others
                    thread::sync_threads();
                    c += 1;
                }

                // thread geometry for the epilogue, rederived instead of kept live by the loop
                let tid = fresh(tid);
                let warp = tid / 32;
                let tq = tid % 4;
                let kq = tid % 32 / 4;
                // round `r` moves lane-local channels `2r` and `2r + 1` of both elements to
                // `M[element][kr][tile]` with `kr = 2 kq + ii`; each round's residual is
                // loaded one round ahead so its latency hides behind the previous round
                let base = opaque(smem);
                // whole cells write the output, split cells their partition's plane of the workspace
                let out = if parts == 1 { y.as_mut_ptr() } else { partial.as_mut_ptr() };
                let site = Site { c: C, h: H, w: W, item, ty, tx0, cotile };
                let last = sweep == SWEEPS - 1;
                let combine = parts == 1 && last;
                let shortcut_words = Residual {
                    words: residual.as_ptr(),
                    enabled: combine && add_residual != 0,
                    plane: 0,
                    coherent: false,
                };
                // a split CTA writes its own plane of the workspace
                let plane = if parts == 1 { 0 } else { split * len };
                // the partial sum of the earlier sweeps, in the words this thread stored
                let prior_words = Residual { words: out as *const f32, enabled: SWEEPS > 1 && sweep > 0, plane, coherent: true };
                // safety: the residual length was checked above whenever it is enabled, and
                // the earlier sweeps stored every valid word of the prior sum
                let mut shortcut = unsafe { shortcut_words.round(&site, tid, 0) };
                // one sweep loads no prior sum, so the single-sweep entries keep their code
                // safety: as above
                let mut prior = if SWEEPS > 1 { unsafe { prior_words.round(&site, tid, 0) } } else { [[0.0; 4]; 2] };
                let mut r = 0;
                #[unroll]
                while r < ROUNDS {
                    let mut e = 0;
                    #[unroll]
                    while e < 2 {
                        let element = warp * 2 + e as u32;
                        let mut ii = 0;
                        #[unroll]
                        while ii < 2 {
                            let i = 2 * r as usize + ii;
                            let row = base + ((element * ROUND_CHANNELS + 2 * kq + ii as u32) * EPILOGUE_ROW + 4 * tq) * 4;
                            // safety: aligned vectors of the epilogue tile, one writer each
                            unsafe {
                                sts4(row, [acc[e][i][0], acc[e][i][1], acc[e][i][2], acc[e][i][3]]);
                                sts4(row + T / 2 * 4, [acc[e][i][4], acc[e][i][5], acc[e][i][6], acc[e][i][7]]);
                            }
                            ii += 1;
                        }
                        e += 1;
                    }
                    thread::sync_threads();
                    // the last round reloads its own words, which keeps every load unconditional
                    let ahead = if r < ROUNDS - 1 { r as u32 + 1 } else { r as u32 };
                    // safety: as for the first round's loads
                    let next = unsafe { shortcut_words.round(&site, tid, ahead) };
                    // safety: as above
                    let next_prior = if SWEEPS > 1 { unsafe { prior_words.round(&site, tid, ahead) } } else { [[0.0; 4]; 2] };

                    let mut pass = 0;
                    #[unroll]
                    while pass < PASSES {
                        let (tx, channel) = site.pass(tid, pass, r as u32);
                        if tx < TW {
                            let t = tx - tx0;
                            let kr = (tid + pass * THREADS) / T;
                            let mut m = [0.0f32; 16];
                            let mut e = 0;
                            #[unroll]
                            while e < 16 {
                                // safety: written before the barrier above
                                m[e] = unsafe { lds1(base + ((e as u32 * ROUND_CHANNELS + kr) * EPILOGUE_ROW + t) * 4) };
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
                                // with one sweep there is no prior sum, and adding zero would turn
                                // a negative zero positive
                                let pair = if SWEEPS > 1 {
                                    [prior[pass as usize][2 * a] + pair[0], prior[pass as usize][2 * a + 1] + pair[1]]
                                } else {
                                    pair
                                };
                                // the residual goes in before the bias, as in cuDNN's fused call
                                let value = if combine {
                                    [
                                        relu(pair[0] + shortcut[pass as usize][2 * a] + bias_value),
                                        relu(pair[1] + shortcut[pass as usize][2 * a + 1] + bias_value),
                                    ]
                                } else {
                                    pair
                                };
                                let index = plane + site.word(channel, oy, 2 * tx);
                                if oy < H && W % 2 == 0 {
                                    // safety: an even pair of the checked output with one writer;
                                    // the host checks 8-byte alignment of output and workspace
                                    unsafe { stg2(out.add(index as usize), value) };
                                } else if oy < H {
                                    let mut k = 0;
                                    #[unroll]
                                    while k < 2 {
                                        if 2 * tx + (k as u32) < W {
                                            let at = (index + k as u32) as usize;
                                            // safety: inside the checked output or plane; one writer
                                            unsafe { *out.add(at) = value[k] };
                                        }
                                        k += 1;
                                    }
                                }
                                a += 1;
                            }
                        }
                        pass += 1;
                    }
                    shortcut = next;
                    prior = next_prior;
                    // closes the reads of this round before the next round's writes
                    thread::sync_threads();
                    r += 1;
                }
                sweep += 1;
            }
        }
    };
}

/// Sums the partial planes of the split cells of a Winograd launch in split order, then
/// adds the residual and the bias and applies a NaN-propagating ReLU
///
/// Cells (64 output channels by one row of 32 2x2 tiles) from `split_from` on hold
/// `splits` partial planes of the output size in `partial`; cells before it were
/// written whole by the convolution. One thread per output word of a split cell, so
/// the few split cells of a batch-1 launch cost one round of load latency, not 32
///
/// Launch 256 threads, grid `(32, cells - split_from, channels / 64)`
#[kernel]
pub fn spk_wideconv_wino_fixup(
    partial: &[f32],
    bias: &[f32],
    residual: &[f32],
    add_residual: u32,
    batch: u32,
    channels: u32,
    h: u32,
    w: u32,
    splits: u32,
    split_from: u32,
    mut y: DisjointSlice<f32>,
) {
    let th = h.div_ceil(2);
    let segments = w.div_ceil(2).div_ceil(WINO_TILES);
    let len = batch * channels * h * w;
    if len as usize > y.len()
        || (splits * len) as usize > partial.len()
        || (add_residual != 0 && len as usize > residual.len())
        || channels as usize > bias.len()
    {
        return;
    }
    let cell = split_from + thread::blockIdx_y();
    let cotile = thread::blockIdx_z();
    // 64 channels by 2 rows by 64 columns; consecutive threads take consecutive columns
    let q = thread::blockIdx_x() * WINO_THREADS + thread::threadIdx_x();
    if cotile >= channels / WINO_CHANNELS
        || cell >= batch * th * segments
        || q >= WINO_CHANNELS * 4 * WINO_TILES
    {
        return;
    }
    let item = cell / (th * segments);
    let ty = cell / segments % th;
    let ox = cell % segments * 2 * WINO_TILES + q % (2 * WINO_TILES);
    let oy = 2 * ty + q / (2 * WINO_TILES) % 2;
    let channel = cotile * WINO_CHANNELS + q / (4 * WINO_TILES);
    if ox >= w || oy >= h {
        return;
    }
    let index = ((item * channels + channel) * h + oy) * w + ox;
    let mut value = 0.0f32;
    let mut s = 0;
    while s < splits {
        value += partial[(s * len + index) as usize];
        s += 1;
    }
    if add_residual != 0 {
        value += residual[index as usize];
    }
    value += bias[channel as usize];
    // safety: one thread per output word of a split cell, inside the checked output
    unsafe { *y.get_unchecked_mut(index as usize) = relu(value) };
}

/// Transforms half a 4x4 patch from a raw stage into its eight `V` words
///
/// Half 0 holds patch rows 0..2 and yields `B^T` rows 0 and 1; half 1 holds rows 1..3
/// and yields rows 2 and 3. Tile `t` reads parity-split words `t` and `t + 1` of each
/// row, so 32 consecutive tiles hit 32 banks
///
/// # Safety
///
/// `raw` and `v` must be word offsets of a published raw stage and of a V stage whose
/// slots of (`half`, `channel`, `tile`) only this thread writes before the next barrier
#[inline(always)]
unsafe fn transform(smem: u32, raw: u32, v: u32, half: u32, channel: u32, tile: u32) {
    let src = opaque(smem + (raw + (channel * 4 + half) * RAW_ROW + tile) * 4);
    // written out: helper functions get no `#[unroll]`, and loops over these arrays
    // could leave them in local memory
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
    let dst = opaque(smem + (v + (channel * 16 + half * 8) * WINO_TILES + tile) * 4);
    // safety: this thread's slots of each element per this function's contract
    unsafe {
        store_column_transform(dst, rows[0]);
        store_column_transform(dst + 4 * WINO_TILES * 4, rows[1]);
    }
}

/// The four parity-split words of one raw row a tile reads: columns 0, 1, 2 and 3
///
/// # Safety
///
/// `row` must address a tile's first word of a published raw row
#[inline(always)]
unsafe fn raw_row(row: u32) -> [f32; 4] {
    // safety: forwarded from this function's contract
    unsafe {
        [
            lds1(row),
            lds1(row + RAW_PARITY * 4),
            lds1(row + 4),
            lds1(row + (RAW_PARITY + 1) * 4),
        ]
    }
}

#[inline(always)]
fn sub4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[inline(always)]
fn add4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Stores `row B` of one `B^T d` row: four `V` words, one element row apart
///
/// # Safety
///
/// `dst` must address this thread's slot of the first of four consecutive elements
#[inline(always)]
unsafe fn store_column_transform(dst: u32, [r0, r1, r2, r3]: [f32; 4]) {
    const STEP: u32 = WINO_TILES * 4;
    // safety: forwarded from this function's contract
    unsafe {
        sts1(dst, r0 - r2);
        sts1(dst + STEP, r1 + r2);
        sts1(dst + 2 * STEP, r2 - r1);
        sts1(dst + 3 * STEP, r1 - r3);
    }
}

winograd3x3! {
    /// Fused Winograd 128 -> 128 3x3 convolution, stride 1, on 20x250 inputs, with
    /// optional residual, folded bias and ReLU, or raw input-channel partial sums
    ///
    /// Launch 256 threads with `WINO_SHARED_BYTES` of dynamic shared memory, grid
    /// `(2, split_from + (cells - split_from) * splits)` with `cells = batch * 10 * 4`;
    /// weights from `spk_wideconv_pack_winograd`; `splits` divides 32; cells from
    /// `split_from` on write `splits` partial planes of the output size for
    /// `spk_wideconv_wino_fixup`; output and residual 8-byte aligned
    spk_wideconv_wino_c128,
    channels = 128,
    h = 20,
    w = 250,
    sweeps = 1,
}

winograd3x3! {
    /// Fused Winograd 256 -> 256 3x3 convolution, stride 1, on 10x125 inputs, with
    /// optional residual, folded bias and ReLU, or raw input-channel partial sums
    ///
    /// Launch as `spk_wideconv_wino_c128` with `cells = batch * 5 * 2`, grid x 4 and
    /// `splits` dividing 64
    spk_wideconv_wino_c256,
    channels = 256,
    h = 10,
    w = 125,
    sweeps = 1,
}

winograd3x3! {
    /// As `spk_wideconv_wino_c128`, with shorter FP32 accumulation chains: the input
    /// channels run in two sweeps of 64, and the second sweep adds its output-domain
    /// partial sum to the first's, in the CTA's own output words
    ///
    /// Launch as `spk_wideconv_wino_c128`; `splits` divides 16
    spk_wideconv_wino_c128_sweep2,
    channels = 128,
    h = 20,
    w = 250,
    sweeps = 2,
}

winograd3x3! {
    /// Fused Winograd 64 -> 64 3x3 convolution, stride 1, on 40x499 inputs, with
    /// optional residual, folded bias and ReLU, or raw input-channel partial sums
    ///
    /// Launch as `spk_wideconv_wino_c128` with `cells = batch * 20 * 8`, grid x 1 and
    /// `splits` dividing 16. The odd output width takes the scalar store path
    spk_wideconv_wino_c64,
    channels = 64,
    h = 40,
    w = 499,
    sweeps = 1,
}
