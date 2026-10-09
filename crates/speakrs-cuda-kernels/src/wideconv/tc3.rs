//! Stride-2 tensor-core implicit GEMM with three TF32 products (3xTF32), for FP32 mode
//! on parts whose TF32 rate is a large multiple of their FP32 rate
//!
//! The CTA, staging and epilogue follow the stride-2 kernels in `tensor`: four warps own
//! 128 output channels by one output row segment, and each staged input row stores its
//! even padded columns, then its odd ones. Here every operand splits into a TF32 high
//! part and the remainder, and each 8-channel tap step multiplies
//! `lo x hi + hi x lo + hi x hi`, which reaches FP32-level error
//!
//! Tensor cores truncate while accumulating, so a step's three products start from zero
//! and join the running sum with one round-to-nearest FADD per element, as in the
//! tensor-core Winograd kernels
//!
//! Segments are 32 or 48 columns wide and the tap loop stays rolled: the split weight
//! fragments double the weight registers of the TF32 kernels, and with 64 columns and
//! unrolled taps ptxas hoisted the weight loads of later taps next to the 64
//! accumulators and spilled inside the channel loop on sm_80 in every register layout
//! tried
//!
//! These entries exist in the sm75 variant only to keep the area's ABI equal; there
//! they trap, and the host plans them only for the sm80 tier

use cuda_device::{DisjointSlice, kernel, launch_bounds};

#[cfg(feature = "tier-sm80")]
use cuda_device::{DynamicSharedArray, shared::cvta_generic_to_shared_u32, thread};

#[cfg(feature = "tier-sm80")]
use super::tensor::TC_CHANNELS;
use super::tensor::round_tf32;

/// Output columns per 3xTF32 stride-2 CTA
pub const TC3_COLUMNS: u32 = 32;
/// Output columns per wide 3xTF32 stride-2 CTA
pub const TC3_WIDE_COLUMNS: u32 = 48;

/// Packs folded weights `[cout][cin][3][3]` into split TF32 `mma.sync` A fragments
///
/// Layout `[cin / 8][tap][cout / 16][part][lane][4]`: part 0 is the weight rounded like
/// `cvt.rna.tf32.f32`, part 1 the rounded remainder, each a 512-byte block in the
/// fragment order of `spk_wideconv_pack_tc`
///
/// Launch one thread per element of `packed`, which has twice as many elements as
/// `weight`
#[kernel]
pub fn spk_wideconv_pack_tc3(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = cuda_device::thread::index_1d();
    let index = i.get() as u32;
    let slot = index % 4;
    let lane = index / 4 % 32;
    let part = index / 128 % 2;
    let tile = index / 256 % (cout / 16);
    let tap = index / (256 * (cout / 16)) % 9;
    let chunk = index / (256 * (cout / 16) * 9);
    // fragment registers: (g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4)
    let row = lane / 4 + slot % 2 * 8;
    let col = lane % 4 + slot / 2 * 4;
    let channel_out = tile * 16 + row;
    let channel_in = chunk * 8 + col;
    let source = ((channel_out * cin + channel_in) * 9 + tap) as usize;
    if source >= weight.len() {
        return;
    }
    let value = weight[source];
    let high = round_tf32(value);
    if let Some(out) = packed.get_mut(i) {
        *out = if part == 0 {
            high
        } else {
            round_tf32(value - high)
        };
    }
}

/// TF32 high part of a finite activation, rounded like `cvt.rna.tf32.f32`
///
/// Integer arithmetic instead of the conversion, which sm_80 and sm_89 expand to a
/// NaN-safe sequence of four instructions; a NaN stays a NaN and an infinity stays
/// infinite, so their products still propagate
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn high_bits(value: f32) -> u32 {
    value.to_bits().wrapping_add(0x1000) & 0xffff_e000
}

/// The high and low A fragments of this warp: MMA step `$step`, 16-row tile `$tile`
#[cfg(feature = "tier-sm80")]
macro_rules! split_fragment_at {
    ($wp:expr, $step:expr, $tile:expr) => {
        // safety: `step < STEPS` and `tile < 4` index a fragment pair of this warp
        unsafe {
            let at = $wp.add((($step * (COUT / 16) + $tile) * 256) as usize);
            [fragment(at), fragment(at.add(128))]
        }
    };
}

/// Expands to one stride-2 3x3 3xTF32 tensor-core convolution
macro_rules! tc3_conv3x3_s2 {
    (
        $(#[$doc:meta])*
        $name:ident,
        cin = $cin:expr,
        cout = $cout:expr,
        h_in = $h:expr,
        w_in = $w:expr,
        columns = $columns:expr $(,)?
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
            batch: u32,
            mut y: DisjointSlice<f32>,
        ) {
            #[cfg(not(feature = "tier-sm80"))]
            {
                let _ = (x, weight, bias, residual, add_residual, batch, y.as_mut_ptr());
                cuda_device::debug::trap();
            }

            #[cfg(feature = "tier-sm80")]
            {
                use super::tensor::ops::{commit, copy4, fragment, lds, mma, nounroll, opaque, relu, wait_all};

                const THREADS: u32 = 128;
                const NPW: u32 = $columns;
                const NT: usize = (NPW / 16) as usize;
                const CIN: u32 = $cin;
                const COUT: u32 = $cout;
                const HI: u32 = $h;
                const WI: u32 = $w;
                const HO: u32 = HI / 2;
                const WO: u32 = (WI + 1) / 2;
                const COLS: u32 = WO.div_ceil(NPW);
                // even padded columns 0, 2, .., 2 * NPW, then odd ones 1, 3, .., 2 * NPW - 1
                const RS: u32 = 2 * NPW + 1;
                const ELEMS: u32 = 3 * RS;
                const CS: u32 = (ELEMS + 23) / 32 * 32 + 8;
                const SLOTS: u32 = ELEMS.div_ceil(THREADS);
                const CHUNKS: u32 = CIN / 8;
                const STEPS: u32 = CHUNKS * 9;
                const STAGE_BYTES: u32 = 8 * CS * 4;
                // an even input height means no output row reads below the image
                const _: () = assert!(CS % 32 == 8 && CS >= ELEMS && HI % 2 == 0);
                const _: () = assert!(NPW == 2 * NT as u32 * 8 && COUT % TC_CHANNELS == 0);

                if (batch * CIN * HI * WI) as usize > x.len()
                    || (batch * COUT * HO * WO) as usize > y.len()
                    || (add_residual != 0 && (batch * COUT * HO * WO) as usize > residual.len())
                    || (2 * CIN * COUT * 9) as usize > weight.len()
                    || COUT as usize > bias.len()
                {
                    return;
                }
                let cotile = thread::blockIdx_x();
                let tile = thread::blockIdx_y();
                if cotile >= COUT / TC_CHANNELS || tile >= batch * HO * COLS {
                    return;
                }
                let item = tile / (HO * COLS);
                let ho = tile / COLS % HO;
                let c0 = tile % COLS * NPW;

                let tid = thread::threadIdx_x();
                let lane = tid % 32;
                let warp = tid / 32;
                let warp_m = warp % 2;
                let warp_n = warp / 2;
                let g = lane / 4;
                let t = lane % 4;
                // safety: the dynamic shared base of this CTA; only its address is taken
                let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
                let x_ptr = x.as_ptr();

                let co0 = cotile * TC_CHANNELS + warp_m * 64;
                // safety: offsets below stay inside the checked packed weights
                let wp = unsafe { weight.as_ptr().add(((co0 / 16 * 64 + lane) * 4) as usize) };
                let lane_offset = (warp_n * NT as u32 * 8 + g + t * CS) * 4;

                // the next step's split weights, refilled tile by tile: a three-step ring
                // of split fragments needs 96 registers next to the 64 accumulators and spilled
                let mut next = [[[0u32; 4]; 2]; 4];
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    next[i] = split_fragment_at!(wp, 0, i as u32);
                    i += 1;
                }
                let mut acc = [[[0.0f32; 4]; NT]; 4];

                let mut i_stage = 0;
                while i_stage <= CHUNKS {
                    if i_stage > 0 {
                        wait_all();
                        thread::sync_threads();
                    }
                    if i_stage < CHUNKS {
                        let dst0 = smem + (i_stage % 2) * STAGE_BYTES;
                        let plane = (item * CIN + i_stage * 8) * HI * WI;
                        let tid = opaque(tid);
                        let mut k = 0;
                        #[unroll]
                        while k < SLOTS {
                            let e = tid + k * THREADS;
                            if e < ELEMS {
                                let r = e / RS;
                                let j = e - r * RS;
                                // padded coordinates, one above and left of the input's
                                let hi = 2 * ho + r;
                                let wi = 2 * c0 + j;
                                let inside = hi >= 1 && hi <= HI && wi >= 1 && wi <= WI;
                                let offset = if inside { plane + (hi - 1) * WI + wi - 1 } else { 0 };
                                let column = if j % 2 == 1 { NPW + 1 + j / 2 } else { j / 2 };
                                let mut ci = 0;
                                #[unroll]
                                while ci < 8 {
                                    // safety: valid copies stay inside the checked input; zero fills
                                    // read nothing
                                    unsafe {
                                        let src = if inside { x_ptr.add((offset + ci * HI * WI) as usize) } else { x_ptr };
                                        copy4(dst0 + (ci * CS + r * RS + column) * 4, src, inside);
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
                    // rolled, in PTX and for ptxas: unrolled taps let ptxas hoist later taps'
                    // weight loads next to the accumulators, and it spilled
                    let mut tap = 0u32;
                    while tap < 9 {
                        nounroll();
                        let kh = tap / 3;
                        let kw = tap % 3;
                        let step = chunk * 9 + tap + 1;
                        let step = if step >= STEPS { step - STEPS } else { step };
                        // kw 0 and 2 read even columns 2 * wo and 2 * wo + 2, kw 1 the odd 2 * wo + 1
                        let column = if kw == 1 { NPW + 1 } else { kw / 2 };
                        let tap_offset = (kh * RS + column) * 4;
                        // split activations of every column group first, so each weight
                        // tile below frees its fragment pair for the next step's load
                        let mut split = [[0u32; 4]; NT];
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let address = base + j as u32 * 32 + tap_offset;
                            // safety: the address lies inside this buffer's published rows
                            let (v0, v1) = unsafe { (lds(address), lds(address + CS * 16)) };
                            let (h0, h1) = (high_bits(v0), high_bits(v1));
                            // the exact remainder; the tensor core ignores its low 13 bits
                            let (l0, l1) = ((v0 - f32::from_bits(h0)).to_bits(), (v1 - f32::from_bits(h1)).to_bits());
                            split[j] = [h0, h1, l0, l1];
                            j += 1;
                        }

                        let mut i = 0;
                        #[unroll]
                        while i < 4 {
                            let [high, low] = next[i];
                            // one tile's pair in flight instead of the whole step's: a full
                            // step of current and next fragments spilled on sm_80
                            next[i] = split_fragment_at!(wp, step, i as u32);
                            let mut j = 0;
                            #[unroll]
                            while j < NT {
                                let [h0, h1, l0, l1] = split[j];
                                // tensor cores truncate while accumulating: the step's products
                                // start from zero and join the running sum with rounded FADDs
                                let mut part = mma([0.0; 4], high, l0, l1);
                                part = mma(part, low, h0, h1);
                                part = mma(part, high, h0, h1);
                                let mut r = 0;
                                #[unroll]
                                while r < 4 {
                                    acc[i][j][r] += part[r];
                                    r += 1;
                                }
                                j += 1;
                            }
                            i += 1;
                        }
                        tap += 1;
                    }
                    i_stage += 1;
                }

                let bias_ptr = bias.as_ptr();
                let residual_ptr = residual.as_ptr();
                let mut j = 0;
                #[unroll]
                while j < NT {
                    let wo = c0 + warp_n * NT as u32 * 8 + j as u32 * 8 + 2 * t;
                    let mut i = 0;
                    #[unroll]
                    while i < 4 {
                        let mut u = 0;
                        #[unroll]
                        while u < 2 {
                            let channel = co0 + i as u32 * 16 + g + u as u32 * 8;
                            let row = ((item * COUT + channel) * HO + ho) * WO;
                            // safety: `channel < COUT`, inside the checked bias
                            let b = unsafe { *bias_ptr.add(channel as usize) };
                            let mut h = 0;
                            #[unroll]
                            while h < 2 {
                                if wo + (h as u32) < WO {
                                    let index = (row + wo + h as u32) as usize;
                                    let mut value = acc[i][j][2 * u + h];
                                    if add_residual != 0 {
                                        // safety: inside the checked residual
                                        value += unsafe { *residual_ptr.add(index) };
                                    }
                                    // safety: inside the checked output; this lane is its only writer
                                    unsafe { *y.get_unchecked_mut(index) = relu(value + b) };
                                }
                                h += 1;
                            }
                            u += 1;
                        }
                        i += 1;
                    }
                    j += 1;
                }
            }
        }
    };
}

tc3_conv3x3_s2! {
    /// 3xTF32 64 -> 128 3x3 convolution, stride 2, from 40x499 to 20x250
    ///
    /// Launch 128 threads, grid `(1, batch * 20 * 8)`, 12800 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc3`
    spk_wideconv_tc3_c64s2,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    columns = TC3_COLUMNS,
}

tc3_conv3x3_s2! {
    /// 3xTF32 64 -> 128 3x3 convolution, stride 2, from 40x499 to 20x250, in 48-column
    /// segments
    ///
    /// Launch 128 threads, grid `(1, batch * 20 * 6)`, 18944 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc3`
    spk_wideconv_tc3_c64s2_wide,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    columns = TC3_WIDE_COLUMNS,
}

tc3_conv3x3_s2! {
    /// 3xTF32 128 -> 256 3x3 convolution, stride 2, from 20x250 to 10x125
    ///
    /// Launch 128 threads, grid `(2, batch * 10 * 4)`, 12800 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc3`
    spk_wideconv_tc3_c128s2,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    columns = TC3_COLUMNS,
}

tc3_conv3x3_s2! {
    /// 3xTF32 128 -> 256 3x3 convolution, stride 2, from 20x250 to 10x125, in
    /// 48-column segments
    ///
    /// Launch 128 threads, grid `(2, batch * 10 * 3)`, 18944 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc3`
    spk_wideconv_tc3_c128s2_wide,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    columns = TC3_WIDE_COLUMNS,
}
