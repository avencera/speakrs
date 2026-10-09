//! Direct FP16 tensor-core implicit GEMM for the same-channel stride-1 trunk
//! convolutions in TF32 mode, for parts whose FP16 tensor rate is a large multiple of
//! their FP32 rate and that have no TF32 tensor cores
//!
//! Both operands are scaled by 2^10 and rounded to FP16 with `cvt.rn`, which keeps FP16
//! subnormals, and `mma.sync` m16n8k8 accumulates the products in FP32; the epilogue
//! removes the 2^20 product scale before the residual and bias. The scale is a power of
//! two, so it changes only which values fall below the FP16 subnormal range. Scaled
//! operands saturate at the largest finite FP16 value instead of overflowing to
//! infinity. Over the measured trunk activations the largest scaled operand is about
//! 38000, below that bound
//!
//! Nothing bounds new audio, so every activation conversion also checks its scaled
//! value: a magnitude above the largest finite FP16 value, or NaN, sets the `range`
//! word with one `red.global.or` per offending thread, and the host recomputes the
//! batch without FP16 tiles. In-range launches never write it. The host checks the
//! weights before it packs them
//!
//! A CTA has `channel_warps * rows` warps and covers `rows` output rows by
//! `8 * n_tiles` columns of one item, for `16 * m_tiles * channel_warps` output
//! channels. Warp `(m, r)` owns output row `r` of the tile and its `m_tiles` channel
//! tiles, so each B fragment it loads feeds `m_tiles` products and each A fragment
//! `n_tiles` products
//!
//! Activations stay NCHW FP32 in global memory. Each pipeline stage converts eight input
//! channels of the input rows behind the tile, with their one-pixel halo, to FP16 and
//! stores them pixel-major, 16 bytes per pixel, so `ldmatrix` loads four B fragments in
//! one instruction and eight consecutive pixels of a fragment cover all 32 banks. There
//! is no `cp.async` on sm_75, so a stage's global loads go to registers before the
//! previous stage's products and are converted and stored after them, behind one CTA
//! barrier per stage. Weights come from [`spk_wideconv_pack_h16`] in fragment order and
//! go straight from global memory into registers two MMA steps ahead
//!
//! The epilogue adds the residual before the bias and applies a ReLU that keeps NaN
//! and negative zero, as the other wideconv kernels do

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

/// Rounds `lo` and `hi` scaled by 2^10 to FP16 with round-to-nearest-even, saturating at
/// the largest finite FP16 value, and packs them with `lo` in the low half
#[inline(always)]
fn half2(lo: f32, hi: f32) -> u32 {
    let packed: u32;
    // safety: register arithmetic and conversions with no memory access
    unsafe {
        ptx_asm!(
            "{ .reg .b16 l, h; .reg .f32 a, b; mul.rn.f32 a, %1, 0f44800000; mul.rn.f32 b, %2, 0f44800000; max.f32 a, a, 0fC77FE000; max.f32 b, b, 0fC77FE000; min.f32 a, a, 0f477FE000; min.f32 b, b, 0f477FE000; cvt.rn.f16.f32 l, a; cvt.rn.f16.f32 h, b; mov.b32 %0, {l, h}; }",
            out("=r") packed,
            in("f") lo,
            in("f") hi,
            options(register_only),
        );
    }
    packed
}

/// As [`half2`] for activations, and also returns 1 when either scaled value is above
/// the largest finite FP16 value in magnitude or is NaN, so it would saturate
#[inline(always)]
fn half2_checked(lo: f32, hi: f32) -> (u32, u32) {
    let (packed, out_of_range): (u32, u32);
    // safety: register arithmetic and conversions with no memory access
    unsafe {
        ptx_asm!(
            "{ .reg .b16 l, h; .reg .f32 a, b, m; .reg .pred p; mul.rn.f32 a, %2, 0f44800000; mul.rn.f32 b, %3, 0f44800000; abs.f32 m, a; setp.gtu.f32 p, m, 0f477FE000; abs.f32 m, b; setp.gtu.or.f32 p, m, 0f477FE000, p; selp.u32 %1, 1, 0, p; max.f32 a, a, 0fC77FE000; max.f32 b, b, 0fC77FE000; min.f32 a, a, 0f477FE000; min.f32 b, b, 0f477FE000; cvt.rn.f16.f32 l, a; cvt.rn.f16.f32 h, b; mov.b32 %0, {l, h}; }",
            out("=r") packed,
            out("=r") out_of_range,
            in("f") lo,
            in("f") hi,
            options(register_only),
        );
    }
    (packed, out_of_range)
}

/// Sets the host's out-of-range word; only threads that saw a saturating activation
/// call this, so in-range launches issue no store
#[inline(always)]
unsafe fn flag_out_of_range(pointer: *mut f32) {
    // safety: the caller passes the checked one-word range buffer; the atomic OR keeps
    // concurrent writers from different CTAs well defined
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; red.global.or.b32 [g], 1; }",
            in("l") pointer as u64,
            clobber("memory"),
        );
    }
}

/// Removes the 2^20 scale of a product of two scaled operands; exact for finite sums
const PRODUCT_UNSCALE: f32 = 1.0 / (1024.0 * 1024.0);

/// Packs folded weights `[cout][cin][3][3]` into FP16 `mma.sync` m16n8k8 A fragments,
/// scaled by 2^10
///
/// Layout `[cin / 8][tap][cout / 16][lane][2]` in 32-bit words of two FP16 values: one
/// 256-byte block per 16x8 fragment, so a warp loads a fragment with one 8-byte load per
/// lane. The packed buffer holds `cout * cin * 9 / 2` words in an `f32` slice
///
/// Launch one thread per word of `packed`
#[kernel]
pub fn spk_wideconv_pack_h16(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let tiles = cout / 16;
    let slot = index % 2;
    let lane = index / 2 % 32;
    let tile = index / 64 % tiles;
    let tap = index / (64 * tiles) % 9;
    let chunk = index / (64 * tiles * 9);
    // fragment registers: row g, then row g + 8, each with columns 2t and 2t + 1
    let row = lane / 4 + slot * 8;
    let col = lane % 4 * 2;
    let channel_out = tile * 16 + row;
    let channel_in = chunk * 8 + col;
    let source = ((channel_out * cin + channel_in) * 9 + tap) as usize;
    // the high half is the next input channel's weight of the same tap
    if source + 9 >= weight.len() {
        return;
    }
    if let Some(out) = packed.get_mut(i) {
        *out = f32::from_bits(half2(weight[source], weight[source + 9]));
    }
}

/// One FP32 word from a global buffer no kernel writes during this launch
#[inline(always)]
unsafe fn ldg(pointer: *const f32) -> f32 {
    let value: f32;
    // safety: the caller passes a readable word of a read-only input
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; ld.global.nc.f32 %0, [g]; }",
            out("=f") value,
            in("l") pointer as u64,
        );
    }
    value
}

/// One FP16 A fragment, 8 bytes per lane, through the read-only path
#[inline(always)]
unsafe fn fragment(pointer: *const f32) -> [u32; 2] {
    let (a, b): (u32, u32);
    // safety: the caller passes an 8-byte aligned pointer into the packed weights, which
    // no kernel writes during a launch
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.nc.v2.u32 {%0, %1}, [g]; }",
            out("=r") a,
            out("=r") b,
            in("l") pointer as u64,
        );
    }
    [a, b]
}

/// Stores four words at a 16-byte aligned shared address
#[inline(always)]
unsafe fn sts4(address: u32, value: [u32; 4]) {
    // safety: the caller passes an address inside this CTA's stage that no other thread
    // accesses before the next barrier
    unsafe {
        ptx_asm!(
            "st.shared.v4.b32 [%0], {%1, %2, %3, %4};",
            in("r") address,
            in("r") value[0],
            in("r") value[1],
            in("r") value[2],
            in("r") value[3],
            clobber("memory"),
        );
    }
}

/// Four B fragments: lanes `8q..8q + 8` name the eight 16-byte pixel rows of fragment `q`
#[inline(always)]
unsafe fn ldsm4(address: u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // safety: a convergent warp-wide load; every caller runs it with all 32 lanes active
    // and passes a 16-byte aligned row of this CTA's published stage
    unsafe {
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("r") address,
            clobber("memory"),
        );
    }
    [a, b, c, d]
}

/// Accumulates one m16n8k8 FP16 product into `c` in FP32
#[inline(always)]
fn mma(c: [f32; 4], a: [u32; 2], b: u32) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = c;
    // safety: a convergent warp-wide register operation; every caller runs it with all
    // 32 lanes active
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") b,
        );
    }
    [c0, c1, c2, c3]
}

/// Passes `value` through a side-effecting register move, so staging indices derived
/// from it are recomputed in every pipeline iteration instead of held in registers
#[inline(always)]
fn opaque(value: u32) -> u32 {
    let out: u32;
    // safety: a register move with no memory access
    unsafe {
        ptx_asm!("mov.u32 %0, %1;", out("=r") out, in("r") value);
    }
    out
}

/// ReLU that passes NaN and negative zero through, like the FP32 kernels
#[inline(always)]
fn relu(value: f32) -> f32 {
    if value < 0.0 { 0.0 } else { value }
}

/// One A fragment of this warp: MMA step `$step`, channel tile `$tile` of the warp
macro_rules! fragment_at {
    ($wp:expr, $step:expr, $tile:expr) => {
        // safety: `step < STEPS` and the warp's tiles index a fragment of the packed weights
        unsafe { fragment($wp.add((($step * TILES + $tile) * 64) as usize)) }
    };
}

/// Expands to one 3x3, stride-1, padding-1 FP16 convolution over `channels` channels on
/// `h x w` inputs
macro_rules! h16_conv3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        channels = $channels:expr,
        h = $h:expr,
        w = $w:expr,
        m_tiles = $mt:expr,
        channel_warps = $wm:expr,
        rows = $rows:expr,
        n_tiles = $nt:expr,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
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
            const C: u32 = $channels;
            const H: u32 = $h;
            const W: u32 = $w;
            const HW: u32 = H * W;
            const MT: usize = $mt;
            const WM: u32 = $wm;
            const ROWS: u32 = $rows;
            const NT: usize = $nt;
            const THREADS: u32 = $threads;
            const COLS: u32 = NT as u32 * 8;
            const CTA_CHANNELS: u32 = WM * MT as u32 * 16;
            const BLOCKS: u32 = C / CTA_CHANNELS;
            const TILES: u32 = C / 16;
            // staged input rows and columns behind the tile, with the one-pixel halo
            const IN_ROWS: u32 = ROWS + 2;
            const IN_COLS: u32 = COLS + 2;
            const PIXELS: u32 = IN_ROWS * IN_COLS;
            const SLOTS: usize = PIXELS.div_ceil(THREADS) as usize;
            const STAGE_BYTES: u32 = PIXELS * 16;
            const CHUNKS: u32 = C / 8;
            const STEPS: u32 = CHUNKS * 9;
            const _: () = assert!(
                THREADS == 32 * WM * ROWS && C % CTA_CHANNELS == 0 && NT % 4 == 0 && C % 16 == 0
            );

            let item = thread::blockIdx_z() / BLOCKS;
            let block = thread::blockIdx_z() % BLOCKS;
            let oy0 = thread::blockIdx_y() * ROWS;
            let ox0 = thread::blockIdx_x() * COLS;
            // the host sizes every buffer; a mismatch must not touch other memory
            if (batch * C * HW) as usize > x.len()
                || (batch * C * HW) as usize > y.len()
                || (add_residual != 0 && (batch * C * HW) as usize > residual.len())
                || (C * C * 9 / 2) as usize > weight.len()
                || C as usize > bias.len()
                || range.len() == 0
                || item >= batch
                || oy0 >= H
                || ox0 >= W
            {
                return;
            }
            let base = item * C * HW;

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let warp_m = warp % WM;
            let warp_r = warp / WM;
            let g = lane / 4;
            let t = lane % 4;
            // safety: the dynamic shared base of this CTA; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
            let x_ptr = x.as_ptr();
            // this lane's `ldmatrix` row for tap (0, 0) and fragments 0..4: staged row
            // `warp_r`, column `lane`, which is pixel `lane % 8` of fragment `lane / 8`
            let lane_row = (warp_r * IN_COLS + lane) * 16;
            let tile0 = block * (CTA_CHANNELS / 16) + warp_m * MT as u32;
            // safety: offsets below stay inside the checked packed weights
            let wp = unsafe { weight.as_ptr().add(((tile0 * 32 + lane) * 2) as usize) };

            let mut ring = [[[0u32; 2]; MT]; 3];
            let mut i = 0;
            #[unroll]
            while i < MT {
                ring[0][i] = fragment_at!(wp, 0, i as u32);
                ring[1][i] = fragment_at!(wp, 1, i as u32);
                i += 1;
            }
            let mut acc = [[[0.0f32; 4]; NT]; MT];
            let mut staged = [[0.0f32; 8]; SLOTS];
            // nonzero once this thread converts an activation that saturates
            let mut out_of_range = 0u32;

            // iteration `i_stage` loads chunk `i_stage`, computes the chunk before it,
            // then stores the loaded chunk, so one copy of the staging code serves the
            // prologue and the loop
            let mut i_stage = 0;
            while i_stage <= CHUNKS {
                if i_stage < CHUNKS {
                    let channels = base + i_stage * 8 * HW;
                    let tid = opaque(tid);
                    let mut k = 0;
                    #[unroll]
                    while k < SLOTS {
                        let e = tid + k as u32 * THREADS;
                        let r = e / IN_COLS;
                        let column = e - r * IN_COLS;
                        // padded coordinates, one above and left of the input's
                        let iy = oy0 + r;
                        let ix = ox0 + column;
                        let inside = e < PIXELS && iy >= 1 && iy <= H && ix >= 1 && ix <= W;
                        // padding loads the item's first word and discards it, so every
                        // lane issues the same loads
                        let offset = if inside { channels + (iy - 1) * W + ix - 1 } else { base };
                        let mut ci = 0;
                        #[unroll]
                        while ci < 8 {
                            // safety: inside the checked input
                            let value = unsafe { ldg(x_ptr.add((offset + ci as u32 * HW) as usize)) };
                            staged[k][ci] = if inside { value } else { 0.0 };
                            ci += 1;
                        }
                        k += 1;
                    }
                }
                if i_stage > 0 {
                    let chunk = i_stage - 1;
                    let stage = smem + chunk % 2 * STAGE_BYTES + lane_row;
                    let mut tap = 0;
                    #[unroll]
                    while tap < 9 {
                        let step = chunk * 9 + tap as u32 + 2;
                        let step = if step >= STEPS { step - STEPS } else { step };
                        let mut i = 0;
                        #[unroll]
                        while i < MT {
                            ring[(tap + 2) % 3][i] = fragment_at!(wp, step, i as u32);
                            i += 1;
                        }
                        let (ky, kx) = (tap as u32 / 3, tap as u32 % 3);
                        let tap_offset = (ky * IN_COLS + kx) * 16;
                        let mut b = [0u32; NT];
                        let mut q = 0;
                        #[unroll]
                        while q < NT / 4 {
                            // safety: the row lies inside this buffer's published stage
                            let quad = unsafe { ldsm4(stage + tap_offset + q as u32 * 32 * 16) };
                            b[4 * q] = quad[0];
                            b[4 * q + 1] = quad[1];
                            b[4 * q + 2] = quad[2];
                            b[4 * q + 3] = quad[3];
                            q += 1;
                        }
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let mut i = 0;
                            #[unroll]
                            while i < MT {
                                acc[i][j] = mma(acc[i][j], ring[tap % 3][i], b[j]);
                                i += 1;
                            }
                            j += 1;
                        }
                        tap += 1;
                    }
                }
                if i_stage < CHUNKS {
                    let stage = smem + i_stage % 2 * STAGE_BYTES;
                    let tid = opaque(tid);
                    let mut k = 0;
                    #[unroll]
                    while k < SLOTS {
                        let e = tid + k as u32 * THREADS;
                        if e < PIXELS {
                            let s = staged[k];
                            let (w0, r0) = half2_checked(s[0], s[1]);
                            let (w1, r1) = half2_checked(s[2], s[3]);
                            let (w2, r2) = half2_checked(s[4], s[5]);
                            let (w3, r3) = half2_checked(s[6], s[7]);
                            out_of_range |= r0 | r1 | r2 | r3;
                            let words = [w0, w1, w2, w3];
                            // safety: pixel `e` of the stage that no warp reads until
                            // the barrier below
                            unsafe { sts4(stage + e * 16, words) };
                        }
                        k += 1;
                    }
                }
                // publishes the stored stage and closes every read of the stage the next
                // iteration overwrites
                thread::sync_threads();
                i_stage += 1;
            }
            if out_of_range != 0 {
                // safety: `range` holds at least one word, checked above
                unsafe { flag_out_of_range(range.as_mut_ptr()) };
            }

            let oy = oy0 + warp_r;
            if oy >= H {
                return;
            }
            let row = base + oy * W;
            let mut i = 0;
            #[unroll]
            while i < MT {
                let mut slot = 0;
                #[unroll]
                while slot < 4 {
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        acc[i][j][slot] *= PRODUCT_UNSCALE;
                        j += 1;
                    }
                    slot += 1;
                }
                i += 1;
            }
            // the residual may alias the output, so reading it all before any store lets
            // the loads overlap; the sum goes in before the bias, as cuDNN's fused call
            // adds them
            if add_residual != 0 {
                let residual_ptr = residual.as_ptr();
                let mut i = 0;
                #[unroll]
                while i < MT {
                    let mut slot = 0;
                    #[unroll]
                    while slot < 4 {
                        let channel = (tile0 + i as u32) * 16 + g + slot as u32 / 2 * 8;
                        let channel_row = row + channel * HW;
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                            if ox < W {
                                // safety: inside the checked residual
                                acc[i][j][slot] += unsafe { *residual_ptr.add((channel_row + ox) as usize) };
                            }
                            j += 1;
                        }
                        slot += 1;
                    }
                    i += 1;
                }
            }

            let bias_ptr = bias.as_ptr();
            let mut i = 0;
            #[unroll]
            while i < MT {
                let mut slot = 0;
                #[unroll]
                while slot < 4 {
                    let channel = (tile0 + i as u32) * 16 + g + slot as u32 / 2 * 8;
                    // safety: `channel < C`, inside the checked bias
                    let b = unsafe { *bias_ptr.add(channel as usize) };
                    let channel_row = row + channel * HW;
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                        if ox < W {
                            // safety: inside the checked output; this lane is its only writer
                            unsafe { *y.get_unchecked_mut((channel_row + ox) as usize) = relu(acc[i][j][slot] + b) };
                        }
                        j += 1;
                    }
                    slot += 1;
                }
                i += 1;
            }
        }
    };
}

h16_conv3x3! {
    /// FP16 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 32 channels, stride 1,
    /// on 80x998 inputs
    ///
    /// `x`, `y` and `residual` are `[batch, 32, 80, 998]`; `weight` comes from
    /// [`spk_wideconv_pack_h16`]; `range` is one word the launch sets nonzero when an
    /// activation saturates. Launch 128 threads with `grid = (16, 20, batch)` and
    /// 12672 dynamic shared bytes
    spk_wideconv_h16_c32,
    channels = 32,
    h = 80,
    w = 998,
    m_tiles = 2,
    channel_warps = 1,
    rows = 4,
    n_tiles = 8,
    threads = 128,
    min_blocks = 3,
}

h16_conv3x3! {
    /// FP16 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 64 channels, stride 1,
    /// on 40x499 inputs
    ///
    /// `x`, `y` and `residual` are `[batch, 64, 40, 499]`; `weight` comes from
    /// [`spk_wideconv_pack_h16`]; `range` is one word the launch sets nonzero when an
    /// activation saturates. Launch 128 threads with `grid = (8, 10, batch)` and
    /// 12672 dynamic shared bytes
    spk_wideconv_h16_c64,
    channels = 64,
    h = 40,
    w = 499,
    m_tiles = 4,
    channel_warps = 1,
    rows = 4,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3! {
    /// FP16 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 128 channels, stride 1,
    /// on 20x250 inputs
    ///
    /// `x`, `y` and `residual` are `[batch, 128, 20, 250]`; `weight` comes from
    /// [`spk_wideconv_pack_h16`]; `range` is one word the launch sets nonzero when an
    /// activation saturates. Launch 128 threads with `grid = (4, 10, batch)` and
    /// 8448 dynamic shared bytes
    spk_wideconv_h16_c128,
    channels = 128,
    h = 20,
    w = 250,
    m_tiles = 4,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3! {
    /// As `spk_wideconv_h16_c128` with 64 output channels per CTA, which doubles the
    /// CTAs of small batches: `grid = (4, 10, batch * 2)`, 8448 dynamic shared bytes
    spk_wideconv_h16_c128_narrow,
    channels = 128,
    h = 20,
    w = 250,
    m_tiles = 2,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 3,
}

h16_conv3x3! {
    /// FP16 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 256 channels, stride 1,
    /// on 10x125 inputs
    ///
    /// `x`, `y` and `residual` are `[batch, 256, 10, 125]`; `weight` comes from
    /// [`spk_wideconv_pack_h16`]; `range` is one word the launch sets nonzero when an
    /// activation saturates. Launch 128 threads with `grid = (2, 5, batch * 2)` and
    /// 8448 dynamic shared bytes
    spk_wideconv_h16_c256,
    channels = 256,
    h = 10,
    w = 125,
    m_tiles = 4,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3! {
    /// As `spk_wideconv_h16_c256` with 64 output channels per CTA, which doubles the
    /// CTAs of small batches: `grid = (2, 5, batch * 4)`, 8448 dynamic shared bytes
    spk_wideconv_h16_c256_narrow,
    channels = 256,
    h = 10,
    w = 125,
    m_tiles = 2,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 3,
}

/// Expands to one 3x3, stride-2, padding-1 FP16 convolution from `in_channels` to
/// `out_channels` channels on `h x w` inputs
///
/// The tile is the stride-1 one with every output pixel reading the input at twice its
/// coordinates, so a CTA stages `2 * rows + 1` input rows of `2 * 8 * n_tiles + 1`
/// columns. A stage row keeps its even padded columns in one plane and its odd ones in
/// another, `S2_ODD` pixels later: eight consecutive output columns of one tap then read
/// eight consecutive 16-byte pixels of one plane, which `ldmatrix` serves without bank
/// conflicts, and the odd plane's offset of 64 mod 128 bytes keeps the stores of
/// neighbouring lanes, which alternate planes, on different banks
macro_rules! h16_conv3x3_s2 {
    (
        $(#[$doc:meta])*
        $name:ident,
        in_channels = $cin:expr,
        out_channels = $cout:expr,
        h = $h:expr,
        w = $w:expr,
        m_tiles = $mt:expr,
        channel_warps = $wm:expr,
        rows = $rows:expr,
        n_tiles = $nt:expr,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads, $min_blocks)]
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
            const CIN: u32 = $cin;
            const C: u32 = $cout;
            const H: u32 = $h;
            const W: u32 = $w;
            const HW: u32 = H * W;
            const OH: u32 = (H - 1) / 2 + 1;
            const OW: u32 = (W - 1) / 2 + 1;
            const OHW: u32 = OH * OW;
            const MT: usize = $mt;
            const WM: u32 = $wm;
            const ROWS: u32 = $rows;
            const NT: usize = $nt;
            const THREADS: u32 = $threads;
            const COLS: u32 = NT as u32 * 8;
            const CTA_CHANNELS: u32 = WM * MT as u32 * 16;
            const BLOCKS: u32 = C / CTA_CHANNELS;
            const TILES: u32 = C / 16;
            // staged input rows and padded columns behind the tile
            const IN_ROWS: u32 = 2 * ROWS + 1;
            const IN_COLS: u32 = 2 * COLS + 1;
            const PIXELS: u32 = IN_ROWS * IN_COLS;
            // stage pixels per input row: `COLS + 1` even columns, then the odd plane
            const S2_ODD: u32 = (COLS + 1).next_multiple_of(8) - 4;
            const ROW_PIXELS: u32 = (S2_ODD + COLS).next_multiple_of(8);
            const SLOTS: usize = PIXELS.div_ceil(THREADS) as usize;
            const STAGE_BYTES: u32 = IN_ROWS * ROW_PIXELS * 16;
            const CHUNKS: u32 = CIN / 8;
            const STEPS: u32 = CHUNKS * 9;
            const _: () = assert!(
                THREADS == 32 * WM * ROWS
                    && C % CTA_CHANNELS == 0
                    && NT % 4 == 0
                    && C % 16 == 0
                    && CIN % 8 == 0
                    && S2_ODD % 8 == 4
                    && S2_ODD >= COLS + 1
            );

            let item = thread::blockIdx_z() / BLOCKS;
            let block = thread::blockIdx_z() % BLOCKS;
            let oy0 = thread::blockIdx_y() * ROWS;
            let ox0 = thread::blockIdx_x() * COLS;
            // the host sizes every buffer; a mismatch must not touch other memory
            if (batch * CIN * HW) as usize > x.len()
                || (batch * C * OHW) as usize > y.len()
                || (add_residual != 0 && (batch * C * OHW) as usize > residual.len())
                || (C * CIN * 9 / 2) as usize > weight.len()
                || C as usize > bias.len()
                || range.len() == 0
                || item >= batch
                || oy0 >= OH
                || ox0 >= OW
            {
                return;
            }
            let base = item * CIN * HW;
            let out_base = item * C * OHW;

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let warp_m = warp % WM;
            let warp_r = warp / WM;
            let g = lane / 4;
            let t = lane % 4;
            // safety: the dynamic shared base of this CTA; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
            let x_ptr = x.as_ptr();
            // this lane's `ldmatrix` row for tap (0, 0) and fragments 0..4: staged input
            // row `2 * warp_r`, even-plane pixel `lane`, which is output column `lane % 8`
            // of fragment `lane / 8`
            let lane_row = (2 * warp_r * ROW_PIXELS + lane) * 16;
            let tile0 = block * (CTA_CHANNELS / 16) + warp_m * MT as u32;
            // safety: offsets below stay inside the checked packed weights
            let wp = unsafe { weight.as_ptr().add(((tile0 * 32 + lane) * 2) as usize) };

            let mut ring = [[[0u32; 2]; MT]; 3];
            let mut i = 0;
            #[unroll]
            while i < MT {
                ring[0][i] = fragment_at!(wp, 0, i as u32);
                ring[1][i] = fragment_at!(wp, 1, i as u32);
                i += 1;
            }
            let mut acc = [[[0.0f32; 4]; NT]; MT];
            let mut staged = [[0.0f32; 8]; SLOTS];
            // nonzero once this thread converts an activation that saturates
            let mut out_of_range = 0u32;

            // iteration `i_stage` loads chunk `i_stage`, computes the chunk before it,
            // then stores the loaded chunk, as in the stride-1 kernels
            let mut i_stage = 0;
            while i_stage <= CHUNKS {
                if i_stage < CHUNKS {
                    let channels = base + i_stage * 8 * HW;
                    let tid = opaque(tid);
                    let mut k = 0;
                    #[unroll]
                    while k < SLOTS {
                        let e = tid + k as u32 * THREADS;
                        let r = e / IN_COLS;
                        let column = e - r * IN_COLS;
                        // padded coordinates, one above and left of the input's
                        let iy = 2 * oy0 + r;
                        let ix = 2 * ox0 + column;
                        let inside = e < PIXELS && iy >= 1 && iy <= H && ix >= 1 && ix <= W;
                        // padding loads the item's first word and discards it, so every
                        // lane issues the same loads
                        let offset = if inside { channels + (iy - 1) * W + ix - 1 } else { base };
                        let mut ci = 0;
                        #[unroll]
                        while ci < 8 {
                            // safety: inside the checked input
                            let value = unsafe { ldg(x_ptr.add((offset + ci as u32 * HW) as usize)) };
                            staged[k][ci] = if inside { value } else { 0.0 };
                            ci += 1;
                        }
                        k += 1;
                    }
                }
                if i_stage > 0 {
                    let chunk = i_stage - 1;
                    let stage = smem + chunk % 2 * STAGE_BYTES + lane_row;
                    let mut tap = 0;
                    #[unroll]
                    while tap < 9 {
                        let step = chunk * 9 + tap as u32 + 2;
                        let step = if step >= STEPS { step - STEPS } else { step };
                        let mut i = 0;
                        #[unroll]
                        while i < MT {
                            ring[(tap + 2) % 3][i] = fragment_at!(wp, step, i as u32);
                            i += 1;
                        }
                        let (ky, kx) = (tap as u32 / 3, tap as u32 % 3);
                        // padded column `2 * ox + kx`: even plane `ox` for kx 0, odd
                        // plane `ox` for kx 1, even plane `ox + 1` for kx 2
                        let plane = if kx == 1 { S2_ODD } else { kx / 2 };
                        let tap_offset = (ky * ROW_PIXELS + plane) * 16;
                        let mut b = [0u32; NT];
                        let mut q = 0;
                        #[unroll]
                        while q < NT / 4 {
                            // safety: the row lies inside this buffer's published stage
                            let quad = unsafe { ldsm4(stage + tap_offset + q as u32 * 32 * 16) };
                            b[4 * q] = quad[0];
                            b[4 * q + 1] = quad[1];
                            b[4 * q + 2] = quad[2];
                            b[4 * q + 3] = quad[3];
                            q += 1;
                        }
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let mut i = 0;
                            #[unroll]
                            while i < MT {
                                acc[i][j] = mma(acc[i][j], ring[tap % 3][i], b[j]);
                                i += 1;
                            }
                            j += 1;
                        }
                        tap += 1;
                    }
                }
                if i_stage < CHUNKS {
                    let stage = smem + i_stage % 2 * STAGE_BYTES;
                    let tid = opaque(tid);
                    let mut k = 0;
                    #[unroll]
                    while k < SLOTS {
                        let e = tid + k as u32 * THREADS;
                        if e < PIXELS {
                            let r = e / IN_COLS;
                            let column = e - r * IN_COLS;
                            let pixel = r * ROW_PIXELS + column % 2 * S2_ODD + column / 2;
                            let s = staged[k];
                            let (w0, r0) = half2_checked(s[0], s[1]);
                            let (w1, r1) = half2_checked(s[2], s[3]);
                            let (w2, r2) = half2_checked(s[4], s[5]);
                            let (w3, r3) = half2_checked(s[6], s[7]);
                            out_of_range |= r0 | r1 | r2 | r3;
                            let words = [w0, w1, w2, w3];
                            // safety: pixel `e` of the stage that no warp reads until
                            // the barrier below
                            unsafe { sts4(stage + pixel * 16, words) };
                        }
                        k += 1;
                    }
                }
                // publishes the stored stage and closes every read of the stage the next
                // iteration overwrites
                thread::sync_threads();
                i_stage += 1;
            }
            if out_of_range != 0 {
                // safety: `range` holds at least one word, checked above
                unsafe { flag_out_of_range(range.as_mut_ptr()) };
            }

            let oy = oy0 + warp_r;
            if oy >= OH {
                return;
            }
            let row = out_base + oy * OW;
            let mut i = 0;
            #[unroll]
            while i < MT {
                let mut slot = 0;
                #[unroll]
                while slot < 4 {
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        acc[i][j][slot] *= PRODUCT_UNSCALE;
                        j += 1;
                    }
                    slot += 1;
                }
                i += 1;
            }
            // the residual may alias the output, so reading it all before any store lets
            // the loads overlap; the sum goes in before the bias, as cuDNN's fused call
            // adds them
            if add_residual != 0 {
                let residual_ptr = residual.as_ptr();
                let mut i = 0;
                #[unroll]
                while i < MT {
                    let mut slot = 0;
                    #[unroll]
                    while slot < 4 {
                        let channel = (tile0 + i as u32) * 16 + g + slot as u32 / 2 * 8;
                        let channel_row = row + channel * OHW;
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                            if ox < OW {
                                // safety: inside the checked residual
                                acc[i][j][slot] += unsafe { *residual_ptr.add((channel_row + ox) as usize) };
                            }
                            j += 1;
                        }
                        slot += 1;
                    }
                    i += 1;
                }
            }

            let bias_ptr = bias.as_ptr();
            let mut i = 0;
            #[unroll]
            while i < MT {
                let mut slot = 0;
                #[unroll]
                while slot < 4 {
                    let channel = (tile0 + i as u32) * 16 + g + slot as u32 / 2 * 8;
                    // safety: `channel < C`, inside the checked bias
                    let b = unsafe { *bias_ptr.add(channel as usize) };
                    let channel_row = row + channel * OHW;
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                        if ox < OW {
                            // safety: inside the checked output; this lane is its only writer
                            unsafe { *y.get_unchecked_mut((channel_row + ox) as usize) = relu(acc[i][j][slot] + b) };
                        }
                        j += 1;
                    }
                    slot += 1;
                }
                i += 1;
            }
        }
    };
}

h16_conv3x3_s2! {
    /// FP16 `y = relu(conv3x3(x, weight, stride 2) + bias [+ residual])` from 32 to 64
    /// channels on 80x998 inputs
    ///
    /// `x` is `[batch, 32, 80, 998]`, `y` and `residual` are `[batch, 64, 40, 499]`;
    /// `weight` comes from [`spk_wideconv_pack_h16`]; `range` is one word the launch sets
    /// nonzero when an activation saturates. Launch 128 threads with
    /// `grid = (8, 20, batch)` and 21760 dynamic shared bytes
    spk_wideconv_h16_c32s2,
    in_channels = 32,
    out_channels = 64,
    h = 80,
    w = 998,
    m_tiles = 2,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3_s2! {
    /// FP16 `y = relu(conv3x3(x, weight, stride 2) + bias [+ residual])` from 64 to 128
    /// channels on 40x499 inputs
    ///
    /// `x` is `[batch, 64, 40, 499]`, `y` and `residual` are `[batch, 128, 20, 250]`;
    /// `weight` comes from [`spk_wideconv_pack_h16`]; `range` is one word the launch sets
    /// nonzero when an activation saturates. Launch 128 threads with
    /// `grid = (4, 10, batch)` and 21760 dynamic shared bytes
    spk_wideconv_h16_c64s2,
    in_channels = 64,
    out_channels = 128,
    h = 40,
    w = 499,
    m_tiles = 4,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3_s2! {
    /// As `spk_wideconv_h16_c64s2` with 64 output channels per CTA, which doubles the
    /// CTAs of small batches: `grid = (4, 10, batch * 2)`, 21760 dynamic shared bytes
    spk_wideconv_h16_c64s2_narrow,
    in_channels = 64,
    out_channels = 128,
    h = 40,
    w = 499,
    m_tiles = 2,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3_s2! {
    /// FP16 `y = relu(conv3x3(x, weight, stride 2) + bias [+ residual])` from 128 to
    /// 256 channels on 20x250 inputs
    ///
    /// `x` is `[batch, 128, 20, 250]`, `y` and `residual` are `[batch, 256, 10, 125]`;
    /// `weight` comes from [`spk_wideconv_pack_h16`]; `range` is one word the launch sets
    /// nonzero when an activation saturates. Launch 128 threads with
    /// `grid = (2, 5, batch * 2)` and 21760 dynamic shared bytes
    spk_wideconv_h16_c128s2,
    in_channels = 128,
    out_channels = 256,
    h = 20,
    w = 250,
    m_tiles = 4,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}

h16_conv3x3_s2! {
    /// As `spk_wideconv_h16_c128s2` with 64 output channels per CTA, which doubles the
    /// CTAs of small batches: `grid = (2, 5, batch * 4)`, 21760 dynamic shared bytes
    spk_wideconv_h16_c128s2_narrow,
    in_channels = 128,
    out_channels = 256,
    h = 20,
    w = 250,
    m_tiles = 2,
    channel_warps = 2,
    rows = 2,
    n_tiles = 8,
    threads = 128,
    min_blocks = 2,
}
