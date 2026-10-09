//! TF32 tensor-core implicit GEMM for the 32- and 64-channel trunk layers, for parts
//! whose TF32 rate is a large multiple of their FP32 rate
//!
//! One product per term: weights arrive rounded to TF32, activations are rounded with
//! `cvt.rna` as they leave shared memory, and `mma.sync` m16n8k8 accumulates in FP32
//! across the whole reduction. This is ordinary TF32 arithmetic, as cuDNN's TF32
//! convolutions use, with no correction products
//!
//! A CTA has four warps and covers four output rows of one item by `8 * tiles`
//! columns. Warp `w` owns output row `w` of the tile and every output channel, so each
//! B fragment it loads feeds `cout / 16` products. Activations stay NCHW: each stage
//! stages eight input channels of the input rows behind the tile, with their one-pixel
//! halo, through `cp.async`, which zero-fills padding, into a two-stage ring. A
//! channel stride of 8 mod 32 words puts the four `t` lanes of a B fragment on
//! disjoint bank octets. Weights come from [`spk_resnet_pack_tc`] in fragment order and
//! go straight from global memory into registers two MMA steps ahead
//!
//! The epilogue adds the residual before the bias and applies a ReLU that keeps NaN
//! and negative zero, as the FP32 kernels do
//!
//! These entries exist only in the sm80 variant; the host plans them only for that
//! tier in TF32 mode

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, DynamicSharedArray, kernel, launch_bounds, ptx_asm, thread};

/// Output rows per CTA, one per warp
pub const TC_ROWS: u32 = 4;
/// Threads per CTA
const THREADS: u32 = 128;

/// Rounds to TF32 like `cvt.rna.tf32.f32`: nearest, ties away from zero
///
/// Infinities and NaNs keep their bits; folded weights are finite
#[inline(always)]
fn round_tf32(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    f32::from_bits(bits.wrapping_add(0x1000) & 0xffff_e000)
}

/// Packs folded weights `[cout][cin][3][3]` into TF32 `mma.sync` A fragments
///
/// Layout `[cin / 8][tap][cout / 16][lane][4]`: one 512-byte block per 16x8 fragment,
/// so a warp loads a fragment with one 16-byte load per lane
///
/// Launch one thread per element of `packed`, which has as many elements as `weight`
#[kernel]
pub fn spk_resnet_pack_tc(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = thread::index_1d();
    let index = i.get() as u32;
    let slot = index % 4;
    let lane = index / 4 % 32;
    let tile = index / 128 % (cout / 16);
    let tap = index / (128 * (cout / 16)) % 9;
    let chunk = index / (128 * (cout / 16) * 9);
    // fragment registers: (g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4)
    let row = lane / 4 + slot % 2 * 8;
    let col = lane % 4 + slot / 2 * 4;
    let channel_out = tile * 16 + row;
    let channel_in = chunk * 8 + col;
    let source = ((channel_out * cin + channel_in) * 9 + tap) as usize;
    if source >= weight.len() {
        return;
    }
    if let Some(out) = packed.get_mut(i) {
        *out = round_tf32(weight[source]);
    }
}

/// Starts a 4-byte asynchronous copy; an invalid source writes a zero instead
#[inline(always)]
unsafe fn copy4(dst: u32, src: *const f32, valid: bool) {
    let size: u32 = if valid { 4 } else { 0 };
    // safety: the caller passes a shared word of this CTA and a readable global word;
    // with size zero the source is not read
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.ca.shared.global [%0], [g], 4, %2; }",
            in("r") dst,
            in("l") src as u64,
            in("r") size,
            clobber("memory"),
        );
    }
}

#[inline(always)]
fn commit() {
    // safety: groups only this lane's issued copies
    unsafe {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
fn wait_all() {
    // safety: waits for this lane's copies; a CTA barrier then publishes them
    unsafe {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
}

/// Passes `value` through a side-effecting register move
///
/// Staging indices derived from the result are recomputed in every pipeline
/// iteration; without it LLVM hoists them out of the loop, where they compete with the
/// accumulators for registers and ptxas spills them
#[inline(always)]
fn opaque(value: u32) -> u32 {
    let out: u32;
    // safety: a register move with no memory access
    unsafe {
        ptx_asm!("mov.u32 %0, %1;", out("=r") out, in("r") value);
    }
    out
}

/// One TF32 A fragment, 16 bytes per lane, through the read-only path
#[inline(always)]
unsafe fn fragment(pointer: *const f32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    // safety: the caller passes a 16-byte aligned pointer into the packed weights, which
    // no kernel writes during a launch
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %4; ld.global.nc.v4.u32 {%0, %1, %2, %3}, [g]; }",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") pointer as u64,
        );
    }
    [a, b, c, d]
}

#[inline(always)]
unsafe fn lds(address: u32) -> f32 {
    let value: f32;
    // safety: the caller passes a staged word of this CTA's shared tile
    unsafe {
        ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") value, in("r") address);
    }
    value
}

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

/// Accumulates one m16n8k8 TF32 product into `c`
#[inline(always)]
fn mma(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
    let [mut c0, mut c1, mut c2, mut c3] = c;
    // safety: a convergent warp-wide register operation; every caller runs it with all
    // 32 lanes active
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
            in("r") b0,
            in("r") b1,
        );
    }
    [c0, c1, c2, c3]
}

/// ReLU that passes NaN and negative zero through, like the FP32 kernels
#[inline(always)]
fn relu(value: f32) -> f32 {
    if value < 0.0 { 0.0 } else { value }
}

/// One A fragment: MMA step `$step`, 16-channel tile `$tile`
macro_rules! fragment_at {
    ($wp:expr, $step:expr, $tile:expr) => {
        // safety: `step < STEPS` and `tile < MT` index a fragment of the packed weights
        unsafe { fragment($wp.add((($step * MT as u32 + $tile) * 128) as usize)) }
    };
}

/// Expands to one 3x3, padding-1 TF32 convolution from `cin` to `cout` channels at
/// `stride`, with `tiles` 8-column MMA tiles per warp
///
/// At stride 2 each staged row keeps its even input columns ahead of its odd ones, so
/// the eight output columns of a B fragment read eight consecutive words, as at stride
/// 1, and the same channel stride keeps the fragment conflict-free
macro_rules! tc_conv3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        cin = $cin:expr,
        cout = $cout:expr,
        stride = $stride:expr,
        tiles = $tiles:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds(128, 2)]
        pub fn $name(
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            residual: &[f32],
            add_residual: u32,
            h_in: u32,
            w_in: u32,
            mut y: DisjointSlice<f32>,
        ) {
            const CIN: u32 = $cin;
            const COUT: u32 = $cout;
            const S: u32 = $stride;
            const MT: usize = (COUT / 16) as usize;
            const NT: usize = $tiles;
            const COLS: u32 = NT as u32 * 8;
            // input rows and columns behind the tile, with the one-pixel halo
            const IN_ROWS: u32 = (TC_ROWS - 1) * S + 3;
            const IN_COLS: u32 = (COLS - 1) * S + 3;
            // staged words per column phase and per input row
            const PH: u32 = IN_COLS.div_ceil(S);
            const RS: u32 = S * PH;
            const ELEMS: u32 = IN_ROWS * IN_COLS;
            const CS: u32 = (IN_ROWS * RS + 23) / 32 * 32 + 8;
            const SLOTS: u32 = ELEMS.div_ceil(THREADS);
            const CHUNKS: u32 = CIN / 8;
            const STEPS: u32 = CHUNKS * 9;
            const STAGE_BYTES: u32 = 8 * CS * 4;
            const _: () = assert!(
                CS % 32 == 8 && CS >= IN_ROWS * RS && CIN % 8 == 0 && COUT % 16 == 0
            );

            let h = (h_in - 1) / S + 1;
            let w = (w_in - 1) / S + 1;
            let plane_in = h_in * w_in;
            let plane = h * w;
            let item = thread::blockIdx_z();
            let oy0 = thread::blockIdx_y() * TC_ROWS;
            let ox0 = thread::blockIdx_x() * COLS;
            let x_base = item * CIN * plane_in;
            let y_base = item * COUT * plane;
            // the host sizes every buffer; a mismatch must not touch other memory
            if (x_base + CIN * plane_in) as usize > x.len()
                || (y_base + COUT * plane) as usize > y.len()
                || (add_residual != 0 && (y_base + COUT * plane) as usize > residual.len())
                || (CIN * COUT * 9) as usize > weight.len()
                || COUT as usize > bias.len()
                || oy0 >= h
                || ox0 >= w
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let g = lane / 4;
            let t = lane % 4;
            // safety: the dynamic shared base of this CTA; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
            let x_ptr = x.as_ptr();
            // this lane's B word of tap (0, 0) and tile 0: channel `t`, the first input row
            // of output row `warp`, output column `g`
            let lane_offset = (t * CS + warp * S * RS + g) * 4;
            // safety: offsets below stay inside the checked packed weights
            let wp = unsafe { weight.as_ptr().add((lane * 4) as usize) };

            let mut ring = [[[0u32; 4]; MT]; 3];
            let mut i = 0;
            #[unroll]
            while i < MT {
                ring[0][i] = fragment_at!(wp, 0, i as u32);
                ring[1][i] = fragment_at!(wp, 1, i as u32);
                i += 1;
            }
            let mut acc = [[[0.0f32; 4]; NT]; MT];

            // iteration `i_stage` stages chunk `i_stage` and computes the chunk before it,
            // so one copy of the staging code serves the prologue and the loop
            let mut i_stage = 0;
            while i_stage <= CHUNKS {
                if i_stage > 0 {
                    wait_all();
                    // publishes the previous chunk's stage and closes all reads of the
                    // buffer the next copies overwrite
                    thread::sync_threads();
                }
                if i_stage < CHUNKS {
                    let dst0 = smem + (i_stage % 2) * STAGE_BYTES;
                    let channels = x_base + i_stage * 8 * plane_in;
                    let tid = opaque(tid);
                    let mut k = 0;
                    #[unroll]
                    while k < SLOTS {
                        let e = tid + k * THREADS;
                        if e < ELEMS {
                            let r = e / IN_COLS;
                            let column = e - r * IN_COLS;
                            let word = r * RS + column % S * PH + column / S;
                            // padded coordinates, one above and left of the input's
                            let iy = oy0 * S + r;
                            let ix = ox0 * S + column;
                            let inside = iy >= 1 && iy <= h_in && ix >= 1 && ix <= w_in;
                            let offset = if inside { channels + (iy - 1) * w_in + ix - 1 } else { 0 };
                            let mut ci = 0;
                            #[unroll]
                            while ci < 8 {
                                // safety: valid copies stay inside the checked input; zero
                                // fills read nothing
                                unsafe {
                                    let src = if inside { x_ptr.add((offset + ci * plane_in) as usize) } else { x_ptr };
                                    copy4(dst0 + (ci * CS + word) * 4, src, inside);
                                }
                                ci += 1;
                            }
                        }
                        k += 1;
                    }
                    commit();
                }
                if i_stage == 0 {
                    i_stage += 1;
                    continue;
                }
                let chunk = i_stage - 1;
                let base = smem + chunk % 2 * STAGE_BYTES + lane_offset;
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
                    let tap_offset = (ky * RS + kx % S * PH + kx / S) * 4;
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        let address = base + tap_offset + j as u32 * 32;
                        // safety: the address lies inside this buffer's published words
                        let (b0, b1) = unsafe { (tf32(lds(address)), tf32(lds(address + CS * 16))) };
                        let mut i = 0;
                        #[unroll]
                        while i < MT {
                            acc[i][j] = mma(acc[i][j], ring[tap % 3][i], b0, b1);
                            i += 1;
                        }
                        j += 1;
                    }
                    tap += 1;
                }
                i_stage += 1;
            }

            let oy = oy0 + warp;
            if oy >= h {
                return;
            }
            let row = y_base + oy * w;
            // the residual may alias the output, so reading it all before any store lets
            // the loads overlap instead of each waiting behind the previous store; the
            // sum still goes in before the bias, as cuDNN's fused call adds them
            if add_residual != 0 {
                let residual_ptr = residual.as_ptr();
                let mut i = 0;
                #[unroll]
                while i < MT {
                    let mut slot = 0;
                    #[unroll]
                    while slot < 4 {
                        let channel = i as u32 * 16 + g + slot as u32 / 2 * 8;
                        let channel_row = row + channel * plane;
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                            if ox < w {
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
                    let channel = i as u32 * 16 + g + slot as u32 / 2 * 8;
                    // safety: `channel < COUT`, inside the checked bias
                    let b = unsafe { *bias_ptr.add(channel as usize) };
                    let channel_row = row + channel * plane;
                    let mut j = 0;
                    #[unroll]
                    while j < NT {
                        let ox = ox0 + j as u32 * 8 + 2 * t + slot as u32 % 2;
                        if ox < w {
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

tc_conv3x3! {
    /// TF32 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 32 -> 32 channels,
    /// stride 1, padding 1
    ///
    /// `x`, `y` and `residual` are `[b, 32, h, w]`; `weight` comes from
    /// [`spk_resnet_pack_tc`]. Launch 128 threads with
    /// `grid = (ceil(w / 112), ceil(h / 4), b)` and 45568 dynamic shared bytes
    spk_resnet_tc_c32,
    cin = 32,
    cout = 32,
    stride = 1,
    tiles = 14,
}

tc_conv3x3! {
    /// TF32 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 64 -> 64 channels,
    /// stride 1, padding 1
    ///
    /// `x`, `y` and `residual` are `[b, 64, h, w]`; `weight` comes from
    /// [`spk_resnet_pack_tc`]. Launch 128 threads with
    /// `grid = (ceil(w / 56), ceil(h / 4), b)` and 23040 dynamic shared bytes
    spk_resnet_tc_c64,
    cin = 64,
    cout = 64,
    stride = 1,
    tiles = 7,
}

tc_conv3x3! {
    /// TF32 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 64 -> 64 channels,
    /// stride 1, padding 1, in 32-column tiles: where the 56-column grid gives fewer
    /// CTAs than SMs, these give 1.8x as many with the same products per output
    ///
    /// `x`, `y` and `residual` are `[b, 64, h, w]`; `weight` comes from
    /// [`spk_resnet_pack_tc`]. Launch 128 threads with
    /// `grid = (ceil(w / 32), ceil(h / 4), b)` and 14848 dynamic shared bytes
    spk_resnet_tc_c64_slim,
    cin = 64,
    cout = 64,
    stride = 1,
    tiles = 4,
}

tc_conv3x3! {
    /// TF32 `y = relu(conv3x3(x, weight) + bias [+ residual])` for 32 -> 64 channels,
    /// stride 2, padding 1
    ///
    /// `x` is `[b, 32, h_in, w_in]`; `y` and `residual` are `[b, 64, h, w]` with
    /// `h = (h_in + 1) / 2` and `w = (w_in + 1) / 2`; `weight` comes from
    /// [`spk_resnet_pack_tc`]. Launch 128 threads with
    /// `grid = (ceil(w / 32), ceil(h / 4), b)` and 39424 dynamic shared bytes
    spk_resnet_tc_c32s2,
    cin = 32,
    cout = 64,
    stride = 2,
    tiles = 4,
}
