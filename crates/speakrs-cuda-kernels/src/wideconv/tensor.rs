//! Tensor-core implicit GEMM for batched wide convolutions in TF32 mode
//!
//! A CTA has four warps and owns 128 output channels. Stride-1 tiles cover 112
//! consecutive flattened output pixels, which may cross row and image boundaries, so
//! no tile pads a 125- or 250-wide row up to a multiple of eight. Stride-2 tiles
//! cover 128 columns of one output row, or 96 in the narrow variant for small batches
//!
//! Activations stream through two `cp.async` stages of eight input channels each.
//! Weights arrive pre-rounded to TF32 in `mma.sync` fragment order and go straight
//! from global memory into registers two MMA steps ahead, which hides L2 latency
//! without spending shared memory. Activations are rounded to TF32 with
//! `cvt.rna` when loaded, matching cuDNN's tensor-op rounding of both operands
//!
//! These entries exist in the sm75 variant only to keep the area's ABI equal; there
//! they trap, and the host plans them only for the sm80 tier in TF32 mode

use cuda_device::{DisjointSlice, kernel, launch_bounds};

#[cfg(feature = "tier-sm80")]
use cuda_device::{DynamicSharedArray, shared::cvta_generic_to_shared_u32, thread};

/// Output channels per CTA
pub const TC_CHANNELS: u32 = 128;
/// Flattened output pixels per stride-1 CTA
pub const TC_PIXELS: u32 = 112;
/// Output columns per stride-2 CTA
pub const TC_COLUMNS: u32 = 128;
/// Output columns per narrow stride-2 CTA
///
/// At batch 1 the 64 -> 128 layer has 40 wide tiles for 36 SMs, so a few SMs run two
/// tiles while the rest idle; three 96-column tiles per row give 60 CTAs that all fit
/// at two per SM, and the busiest SM computes 192 columns instead of 256
pub const TC_NARROW_COLUMNS: u32 = 96;

/// Output columns per narrow 128 -> 256 stride-2 CTA
///
/// At batch 1 the layer's 128-column tiles give 20 CTAs; 64-column tiles give 40, one
/// wave at two per SM on 34 or 36 SMs
pub const TC_C128S2_NARROW_COLUMNS: u32 = 64;

/// Output columns per slim 64 -> 128 stride-2 CTA
///
/// At batch 1 the narrow tiles give 60 CTAs, fewer than the 108 SMs of an A100; 64-column
/// tiles give 80, each with two thirds of the work
pub const TC_C64S2_SLIM_COLUMNS: u32 = 64;

/// Output columns per slim 128 -> 256 stride-2 CTA: 80 CTAs at batch 1 instead of the
/// narrow tiles' 40
pub const TC_C128S2_SLIM_COLUMNS: u32 = 32;

/// Rounds to TF32 like `cvt.rna.tf32.f32`: nearest, ties away from zero
///
/// Infinities and NaNs keep their bits; folded weights are finite
#[inline(always)]
pub(super) fn round_tf32(value: f32) -> f32 {
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
pub fn spk_wideconv_pack_tc(weight: &[f32], cin: u32, cout: u32, mut packed: DisjointSlice<f32>) {
    let i = cuda_device::thread::index_1d();
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

#[cfg(feature = "tier-sm80")]
pub(super) mod ops {
    use cuda_device::ptx_asm;

    /// Starts a 4-byte asynchronous copy; an invalid source writes a zero instead
    #[inline(always)]
    pub(crate) unsafe fn copy4(dst: u32, src: *const f32, valid: bool) {
        let size: u32 = if valid { 4 } else { 0 };
        // safety: the caller passes a shared word of this CTA and a readable global
        // word; with size zero the source is not read
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

    /// Passes `value` through a side-effecting register move
    ///
    /// Staging indices derived from the result are recomputed in every pipeline
    /// iteration; without it LLVM hoists them out of the loop, where they compete with
    /// the accumulators for registers and ptxas spills them
    #[inline(always)]
    pub(crate) fn opaque(value: u32) -> u32 {
        let out: u32;
        // safety: a register move with no memory access
        unsafe {
            ptx_asm!("mov.u32 %0, %1;", out("=r") out, in("r") value);
        }
        out
    }

    #[inline(always)]
    pub(crate) fn commit() {
        // safety: groups only this lane's issued copies
        unsafe {
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
    }

    /// Keeps ptxas from unrolling the loop whose header block this starts
    ///
    /// ptxas fully unrolls small constant-trip loops on its own, even where the PTX keeps
    /// them rolled, and then schedules every iteration's loads at once
    #[inline(always)]
    pub(crate) fn nounroll() {
        // safety: a compiler directive with no effect on registers or memory
        unsafe {
            ptx_asm!(".pragma \"nounroll\";");
        }
    }

    /// Waits for all but this lane's most recent copy group
    #[inline(always)]
    pub(crate) fn wait_one() {
        // safety: waits for this lane's copies; a CTA barrier then publishes them
        unsafe {
            ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
        }
    }

    #[inline(always)]
    pub(crate) fn wait_all() {
        // safety: waits for this lane's copies; a CTA barrier then publishes them
        unsafe {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
        }
    }

    /// One TF32 A fragment, 16 bytes per lane, through the read-only path
    #[inline(always)]
    pub(crate) unsafe fn fragment(pointer: *const f32) -> [u32; 4] {
        let (a, b, c, d): (u32, u32, u32, u32);
        // safety: the caller passes a 16-byte aligned pointer into the packed weights,
        // which no kernel writes during a launch
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
    pub(crate) unsafe fn lds(address: u32) -> f32 {
        let value: f32;
        // safety: the caller passes a staged word of this CTA's shared tile
        unsafe {
            ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") value, in("r") address);
        }
        value
    }

    #[inline(always)]
    pub(crate) fn tf32(value: f32) -> u32 {
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
    pub(crate) fn mma(c: [f32; 4], a: [u32; 4], b0: u32, b1: u32) -> [f32; 4] {
        let [mut c0, mut c1, mut c2, mut c3] = c;
        // safety: a convergent warp-wide register operation; every caller runs it
        // with all 32 lanes active
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

    /// Asks L2 for the line holding `pointer`; it never faults and never blocks
    #[inline(always)]
    pub(crate) unsafe fn prefetch(pointer: *const f32) {
        // safety: the caller passes an address inside a live allocation
        unsafe {
            ptx_asm!(
                "{ .reg .u64 g; cvta.to.global.u64 g, %0; prefetch.global.L2 [g]; }",
                in("l") pointer as u64,
            );
        }
    }

    /// Two floats from an 8-byte aligned read-only global address
    #[inline(always)]
    pub(crate) unsafe fn load2(pointer: *const f32) -> [f32; 2] {
        let (a, b): (f32, f32);
        // safety: the caller passes an aligned pair of a buffer no kernel writes now
        unsafe {
            ptx_asm!(
                "{ .reg .u64 g; cvta.to.global.u64 g, %2; ld.global.nc.v2.f32 {%0, %1}, [g]; }",
                out("=f") a,
                out("=f") b,
                in("l") pointer as u64,
            );
        }
        [a, b]
    }

    /// Stores two floats at an 8-byte aligned global address
    #[inline(always)]
    pub(crate) unsafe fn store2(pointer: *mut f32, value: [f32; 2]) {
        // safety: the caller passes an aligned pair that only this lane writes
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

    /// ReLU that passes NaN through, like cuDNN's propagating activation
    #[inline(always)]
    pub(crate) fn relu(value: f32) -> f32 {
        if value < 0.0 { 0.0 } else { value }
    }
}

/// Virtual word of an output pixel's centre tap in the stride-1 staging layout
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn centre(pixel: u32, width: u32, row_stride: u32) -> u32 {
    (pixel / width + 1) * row_stride + pixel % width + 1
}

/// One A fragment of this warp: MMA step `$step`, 16-row tile `$tile`
#[cfg(feature = "tier-sm80")]
macro_rules! fragment_at {
    ($wp:expr, $step:expr, $tile:expr) => {
        // safety: `step < STEPS` and `tile < 4` index a fragment of this warp
        unsafe { fragment($wp.add((($step * (COUT / 16) + $tile) * 128) as usize)) }
    };
}

/// Expands to one stride-1 3x3 tensor-core convolution
///
/// Stride-1 staging: a CTA stages the contiguous range of a virtual input whose rows
/// carry one zero column on each side and run across images, so every tap is a
/// fixed offset from the pixel's own word. Taps above the first or below the last
/// row of an image would read the neighbouring image and are masked per lane
macro_rules! tc_conv3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        cin = $cin:expr,
        cout = $cout:expr,
        h = $h:expr,
        w = $w:expr $(,)?
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
                use ops::{commit, copy4, fragment, lds, load2, mma, opaque, prefetch, relu, store2, tf32, wait_all};

                const THREADS: u32 = 128;
                const NT: usize = 7;
                const CIN: u32 = $cin;
                const COUT: u32 = $cout;
                const H: u32 = $h;
                const W: u32 = $w;
                const HW: u32 = H * W;
                const NP: u32 = TC_PIXELS;
                const RS: u32 = W + 2;
                const CHUNKS: u32 = CIN / 8;
                const STEPS: u32 = CHUNKS * 9;
                // a tile spans at most two rows, so its staged range is at most NP + 2 * RS + 3
                const LMAX: u32 = NP + 2 * RS + 8;
                // a channel stride of 8 mod 32 words puts the four `t` lanes of a
                // B fragment on disjoint bank octets
                const CS: u32 = (LMAX + 23) / 32 * 32 + 8;
                const SLOTS: u32 = LMAX.div_ceil(THREADS);
                const STAGE_BYTES: u32 = 8 * CS * 4;
                const _: () = assert!(CS % 32 == 8 && CS >= LMAX && NP < W && HW % 2 == 0);
                const _: () = assert!(NP == 2 * NT as u32 * 8 && COUT % TC_CHANNELS == 0);

                let total = batch * HW;
                if (batch * CIN * HW) as usize > x.len()
                    || (batch * COUT * HW) as usize > y.len()
                    || (add_residual != 0 && (batch * COUT * HW) as usize > residual.len())
                    || (CIN * COUT * 9) as usize > weight.len()
                    || COUT as usize > bias.len()
                {
                    return;
                }
                let cotile = thread::blockIdx_x();
                let p0 = thread::blockIdx_y() * NP;
                if cotile >= COUT / TC_CHANNELS || p0 >= total {
                    return;
                }

                let tid = thread::threadIdx_x();
                let lane = tid % 32;
                let warp = tid / 32;
                let warp_m = warp % 2;
                let warp_n = warp / 2;
                let g = lane / 4;
                let t = lane % 4;
                let plast = if p0 + NP < total { p0 + NP - 1 } else { total - 1 };
                // virtual word of an output pixel's centre tap
                let qlo = centre(p0, W, RS) - RS - 1;
                let len = centre(plast, W, RS) + RS + 2 - qlo;
                // safety: the dynamic shared base of this CTA; only its address is taken
                let smem = unsafe { cvta_generic_to_shared_u32(DynamicSharedArray::<f32, 16>::get() as *const u8) };
                let x_ptr = x.as_ptr();
                let rows = batch * H;

                let mut boff = [0u32; NT];
                let mut top = 0u32;
                let mut bottom = 0u32;
                let mut j = 0;
                #[unroll]
                while j < NT {
                    let p = p0 + warp_n * NT as u32 * 8 + j as u32 * 8 + g;
                    let p = if p > plast { plast } else { p };
                    boff[j] = (centre(p, W, RS) - qlo - RS - 1 + t * CS) * 4;
                    let row = p / W % H;
                    top |= ((row == 0) as u32) << j;
                    bottom |= ((row == H - 1) as u32) << j;
                    j += 1;
                }

                let co0 = cotile * TC_CHANNELS + warp_m * 64;
                // safety: offsets below stay inside the checked packed weights
                let wp = unsafe { weight.as_ptr().add(((co0 / 16 * 32 + lane) * 4) as usize) };

                let mut ring = [[[0u32; 4]; 4]; 3];
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    ring[0][i] = fragment_at!(wp, 0, i as u32);
                    ring[1][i] = fragment_at!(wp, 1, i as u32);
                    i += 1;
                }
                let mut acc = [[[0.0f32; 4]; NT]; 4];
                let residual_ptr = residual.as_ptr();

                // iteration `i_stage` stages chunk `i_stage` and computes the chunk before it,
                // so one copy of the staging code serves the prologue and the loop
                let mut i_stage = 0;
                while i_stage <= CHUNKS {
                    if i_stage > 0 {
                        wait_all();
                        // publishes the previous chunk's stage and closes all reads of
                        // the buffer the next copies overwrite
                        thread::sync_threads();
                    }
                    if i_stage < CHUNKS {
                        let dst0 = smem + (i_stage % 2) * STAGE_BYTES;
                        let tid = opaque(tid);
                        let mut k = 0;
                        #[unroll]
                        while k < SLOTS {
                            let j = tid + k * THREADS;
                            if j < len && j < LMAX {
                                let q = qlo + j;
                                let vq = q / RS;
                                let pc = q - vq * RS;
                                let inside = vq >= 1 && vq <= rows && pc >= 1 && pc <= W;
                                let v = if inside { vq - 1 } else { 0 };
                                let column = if inside { pc - 1 } else { 0 };
                                let offset = (v / H * CIN + i_stage * 8) * HW + v % H * W + column;
                                let mut ci = 0;
                                #[unroll]
                                while ci < 8 {
                                    // safety: valid copies stay inside the checked input; zero fills
                                    // read nothing
                                    unsafe {
                                        let src = if inside { x_ptr.add((offset + ci * HW) as usize) } else { x_ptr };
                                        copy4(smem_word(dst0, ci * CS + j), src, inside);
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
                    if add_residual != 0 && chunk + 2 == CHUNKS {
                        // pulls this warp's residual rows into L2 while the last
                        // chunks compute, so the epilogue does not wait on DRAM
                        let pw = p0 + warp_n * NT as u32 * 8;
                        let mut k = 0;
                        #[unroll]
                        while k < 3 {
                            let p = pw + k * (NT as u32 * 8 - 1) / 2;
                            let p = if p >= total { total - 1 } else { p };
                            let index = ((p / HW * COUT + co0 + lane) * HW + p % HW) as usize;
                            // safety: both rows lie inside the checked residual
                            unsafe {
                                prefetch(residual_ptr.add(index));
                                prefetch(residual_ptr.add(index + (32 * HW) as usize));
                            }
                            k += 1;
                        }
                    }
                    let base = smem + chunk % 2 * STAGE_BYTES;
                    let mut tap = 0;
                    #[unroll]
                    while tap < 9 {
                        let kh = tap / 3;
                        let step = chunk * 9 + tap as u32 + 2;
                        let step = if step >= STEPS { step - STEPS } else { step };
                        let mut i = 0;
                        #[unroll]
                        while i < 4 {
                            ring[(tap + 2) % 3][i] = fragment_at!(wp, step, i as u32);
                            i += 1;
                        }
                        let tap_offset = (kh as u32 * RS + tap as u32 % 3) * 4;
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let address = base + boff[j] + tap_offset;
                            let keep = if kh == 0 {
                                ((top >> j) & 1).wrapping_sub(1)
                            } else if kh == 2 {
                                ((bottom >> j) & 1).wrapping_sub(1)
                            } else {
                                u32::MAX
                            };
                            // safety: the address lies inside this buffer's published range
                            let (b0, b1) = unsafe { (tf32(lds(address)) & keep, tf32(lds(address + CS * 16)) & keep) };
                            let mut i = 0;
                            #[unroll]
                            while i < 4 {
                                acc[i][j] = mma(acc[i][j], ring[tap % 3][i], b0, b1);
                                i += 1;
                            }
                            j += 1;
                        }
                        tap += 1;
                    }
                    i_stage += 1;
                }

                let bias_ptr = bias.as_ptr();
                let y_ptr = y.as_mut_ptr();
                let mut j = 0;
                #[unroll]
                while j < NT {
                    // even pixel pairs never straddle an image because the plane is even
                    let p = p0 + warp_n * NT as u32 * 8 + j as u32 * 8 + 2 * t;
                    if p < total {
                        let base = p / HW * COUT * HW + p % HW;
                        let mut i = 0;
                        #[unroll]
                        while i < 4 {
                            let mut u = 0;
                            #[unroll]
                            while u < 2 {
                                let channel = co0 + i as u32 * 16 + g + u as u32 * 8;
                                let index = (base + channel * HW) as usize;
                                let mut value = [acc[i][j][2 * u], acc[i][j][2 * u + 1]];
                                // safety: `index` is an even element of the checked output
                                // and residual; this lane is its only writer
                                unsafe {
                                    if add_residual != 0 {
                                        let shortcut = load2(residual_ptr.add(index));
                                        value[0] += shortcut[0];
                                        value[1] += shortcut[1];
                                    }
                                    // the residual goes in before the bias, as cuDNN's fused call adds them
                                    let b = *bias_ptr.add(channel as usize);
                                    store2(y_ptr.add(index), [relu(value[0] + b), relu(value[1] + b)]);
                                }
                                u += 1;
                            }
                            i += 1;
                        }
                    }
                    j += 1;
                }
            }
        }
    };
}

/// Expands to one stride-2 3x3 tensor-core convolution
///
/// Each staged input row stores its even padded columns, then its odd ones, so the
/// stride-2 reads of consecutive lanes hit consecutive words
macro_rules! tc_conv3x3_s2 {
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
                use ops::{commit, copy4, fragment, lds, mma, opaque, relu, tf32, wait_all};

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
                    || (CIN * COUT * 9) as usize > weight.len()
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
                let wp = unsafe { weight.as_ptr().add(((co0 / 16 * 32 + lane) * 4) as usize) };
                let lane_offset = (warp_n * NT as u32 * 8 + g + t * CS) * 4;

                let mut ring = [[[0u32; 4]; 4]; 3];
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    ring[0][i] = fragment_at!(wp, 0, i as u32);
                    ring[1][i] = fragment_at!(wp, 1, i as u32);
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
                                        copy4(smem_word(dst0, ci * CS + r * RS + column), src, inside);
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
                        let kh = tap as u32 / 3;
                        let kw = tap as u32 % 3;
                        let step = chunk * 9 + tap as u32 + 2;
                        let step = if step >= STEPS { step - STEPS } else { step };
                        let mut i = 0;
                        #[unroll]
                        while i < 4 {
                            ring[(tap + 2) % 3][i] = fragment_at!(wp, step, i as u32);
                            i += 1;
                        }
                        // kw 0 and 2 read even columns 2 * wo and 2 * wo + 2, kw 1 the odd 2 * wo + 1
                        let column = if kw == 1 { NPW + 1 } else { kw / 2 };
                        let tap_offset = (kh * RS + column) * 4;
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            let address = base + j as u32 * 32 + tap_offset;
                            // safety: the address lies inside this buffer's published rows
                            let (b0, b1) = unsafe { (tf32(lds(address)), tf32(lds(address + CS * 16))) };
                            let mut i = 0;
                            #[unroll]
                            while i < 4 {
                                acc[i][j] = mma(acc[i][j], ring[tap % 3][i], b0, b1);
                                i += 1;
                            }
                            j += 1;
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

/// Byte address of word `index` in a shared stage
#[cfg(feature = "tier-sm80")]
#[inline(always)]
fn smem_word(base: u32, index: u32) -> u32 {
    base + index * 4
}

tc_conv3x3! {
    /// TF32 128 -> 128 3x3 convolution, stride 1, on 20x250 planes
    ///
    /// Launch 128 threads, grid `(1, ceil(batch * 5000 / 112))`, 41472 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc`
    spk_wideconv_tc_c128,
    cin = 128,
    cout = 128,
    h = 20,
    w = 250,
}

tc_conv3x3! {
    /// TF32 256 -> 256 3x3 convolution, stride 1, on 10x125 planes
    ///
    /// Launch 128 threads, grid `(2, ceil(batch * 1250 / 112))`, 25088 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc`
    spk_wideconv_tc_c256,
    cin = 256,
    cout = 256,
    h = 10,
    w = 125,
}

tc_conv3x3_s2! {
    /// TF32 64 -> 128 3x3 convolution, stride 2, from 40x499 to 20x250
    ///
    /// Launch 128 threads, grid `(1, batch * 20 * 2)`, 49664 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc`
    spk_wideconv_tc_c64s2,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    columns = TC_COLUMNS,
}

tc_conv3x3_s2! {
    /// TF32 64 -> 128 3x3 convolution, stride 2, from 40x499 to 20x250, for small batches
    ///
    /// Launch 128 threads, grid `(1, batch * 20 * 3)`, 37376 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc`
    spk_wideconv_tc_c64s2_narrow,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    columns = TC_NARROW_COLUMNS,
}

tc_conv3x3_s2! {
    /// TF32 128 -> 256 3x3 convolution, stride 2, from 20x250 to 10x125
    ///
    /// Launch 128 threads, grid `(2, batch * 10)`, 49664 dynamic shared bytes
    /// Weights come from `spk_wideconv_pack_tc`
    spk_wideconv_tc_c128s2,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    columns = TC_COLUMNS,
}

tc_conv3x3_s2! {
    /// As `spk_wideconv_tc_c128s2` with 64-column tiles for small batches: grid
    /// `(2, batch * 10 * 2)`
    spk_wideconv_tc_c128s2_narrow,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    columns = TC_C128S2_NARROW_COLUMNS,
}

tc_conv3x3_s2! {
    /// As `spk_wideconv_tc_c64s2` with 64-column tiles for parts with more SMs than
    /// narrow tiles: grid `(1, batch * 20 * 4)`, 25088 dynamic shared bytes
    spk_wideconv_tc_c64s2_slim,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    columns = TC_C64S2_SLIM_COLUMNS,
}

tc_conv3x3_s2! {
    /// As `spk_wideconv_tc_c128s2` with 32-column tiles for parts with more SMs than
    /// narrow tiles: grid `(2, batch * 10 * 4)`, 12800 dynamic shared bytes
    spk_wideconv_tc_c128s2_slim,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    columns = TC_C128S2_SLIM_COLUMNS,
}
