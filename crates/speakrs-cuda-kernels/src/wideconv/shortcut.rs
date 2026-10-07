//! Register-tiled 1x1 stride-2 shortcut convolutions with folded bias and no ReLU
//!
//! A CTA has 128 threads and owns `channels` output channels by 64 consecutive
//! output pixels of one image. A 1x1 convolution has no halo, so a tile may cross
//! output rows. Input channels stream through two stages of `chunk` channels:
//! `cp.async` on sm80, synchronous loads on sm75. Each thread accumulates 8 channels
//! by 4 or 8 consecutive pixels in full FP32, in both math modes
//!
//! Each shape has a 64-channel tile, which gives small batches enough CTAs, and the
//! larger shapes add a 128-channel tile, which halves the weight traffic per pixel
//! when the batch fills the GPU

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread};

use super::{commit, lds4, opaque, stage_quad, stage_scalar, stg2, sts1, wait};

/// Output pixels per shortcut CTA
pub const SHORTCUT_PIXELS: u32 = 64;

/// Expands to one shortcut kernel for a fixed shape and channel tile
///
/// Weights are packed `[cin][cout]` by `spk_wideconv_pack_weights` with one tap
macro_rules! shortcut_tile {
    (
        $(#[$doc:meta])*
        $name:ident,
        cin = $cin:expr,
        cout = $cout:expr,
        h_in = $h:expr,
        w_in = $w:expr,
        channels = $channels:expr,
        chunk = $chunk:expr $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds(128)]
        pub fn $name(
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            batch: u32,
            mut y: DisjointSlice<f32>,
        ) {
            const CI: u32 = $cin;
            const CO: u32 = $cout;
            const HI: u32 = $h;
            const WI: u32 = $w;
            const HO: u32 = (HI - 1) / 2 + 1;
            const WO: u32 = (WI - 1) / 2 + 1;
            const PLANE: u32 = HO * WO;
            const COT: u32 = $channels;
            const KC: u32 = $chunk;
            const NPX: u32 = SHORTCUT_PIXELS;
            // channel groups of 8 per CTA, pixel groups, and pixels per thread
            const CG: u32 = COT / 8;
            const PG: u32 = 128 / CG;
            const TN: usize = (NPX / PG) as usize;
            const ROWS: u32 = 128 / NPX;
            const AW: u32 = KC * COT;
            const STAGE: u32 = AW + KC * NPX;
            const CHUNKS: u32 = CI / KC;
            const QUADS: u32 = AW / 4;
            const SMEM: usize = 2 * STAGE as usize;
            // even planes keep every pixel pair inside one plane and 8-byte aligned
            const _: () = assert!(CI % KC == 0 && CO % COT == 0 && PLANE % 2 == 0);
            const _: () = assert!(TN % 4 == 0 && KC % ROWS == 0 && QUADS % 128 == 0 && CG % 8 == 0);

            static mut TILE: SharedArray<f32, SMEM, 16> = SharedArray::UNINIT;

            if (batch * CI * HI * WI) as usize > x.len()
                || (batch * CO * PLANE) as usize > y.len()
                || (CI * CO) as usize > weight.len()
                || CO as usize > bias.len()
            {
                return;
            }
            let p0 = thread::blockIdx_x() * NPX;
            let item = thread::blockIdx_y();
            let cotile = thread::blockIdx_z();
            if p0 >= PLANE || item >= batch || cotile >= CO / COT {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let cg = warp % (CG / 8) * 8 + lane % 8;
            let pg = warp / (CG / 8) * 4 + lane / 8;
            // staging: this thread copies pixel column `tid % NPX` of rows `tid / NPX`,
            // `tid / NPX + ROWS`, .. of every chunk
            let f = p0 + tid % NPX;
            let valid = f < PLANE;
            let spatial = if valid { 2 * (f / WO) * WI + 2 * (f % WO) } else { 0 };
            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();
            let x_base = item * CI * HI * WI + spatial;
            let channel0 = cotile * COT;
            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const TILE as *const u8) };

            let mut acc = [[0.0f32; TN]; 8];
            // iteration `i_stage` stages chunk `i_stage` and computes the chunk before it
            let mut i_stage = 0;
            while i_stage <= CHUNKS {
                if i_stage < CHUNKS {
                    let buffer = smem + i_stage % 2 * STAGE * 4;
                    let mut m = 0;
                    #[unroll]
                    while m < QUADS / 128 {
                        let q = tid + m * 128;
                        let k = q / (COT / 4);
                        let c4 = q % (COT / 4) * 4;
                        let source = (i_stage * KC + k) * CO + channel0 + c4;
                        // safety: complete aligned channel vectors inside the checked weights
                        unsafe { stage_quad(buffer + (k * COT + c4) * 4, w_ptr.add(source as usize)) };
                        m += 1;
                    }
                    let mut m = 0;
                    #[unroll]
                    while m < KC / ROWS {
                        let k = tid / NPX + m * ROWS;
                        let dst = buffer + (AW + k * NPX + tid % NPX) * 4;
                        if valid {
                            let offset = x_base + (i_stage * KC + k) * HI * WI;
                            // safety: a valid pixel of a channel inside the checked input
                            unsafe { stage_scalar(dst, x_ptr.add(offset as usize)) };
                        } else {
                            // safety: masked pixels have one writer before publication
                            unsafe { sts1(dst, 0.0) };
                        }
                        m += 1;
                    }
                    commit();
                }
                if i_stage == 0 {
                    i_stage += 1;
                    continue;
                }
                // one newer group is outstanding except after the last chunk
                wait(i_stage == CHUNKS);
                thread::sync_threads();
                let chunk = i_stage - 1;
                let base = opaque(smem + chunk % 2 * STAGE * 4);
                let mut k = 0;
                #[unroll]
                while k < KC {
                    // safety: aligned addresses inside the published stage
                    let (a0, a1) = unsafe {
                        (
                            lds4(base + (k * COT + 4 * cg) * 4),
                            lds4(base + (k * COT + COT / 2 + 4 * cg) * 4),
                        )
                    };
                    let a = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]];
                    let mut b = [0.0f32; TN];
                    let mut q = 0;
                    #[unroll]
                    while q < TN / 4 {
                        // safety: as above
                        let v = unsafe { lds4(base + (AW + k * NPX + TN as u32 * pg + 4 * q as u32) * 4) };
                        b[4 * q] = v[0];
                        b[4 * q + 1] = v[1];
                        b[4 * q + 2] = v[2];
                        b[4 * q + 3] = v[3];
                        q += 1;
                    }
                    let mut i = 0;
                    #[unroll]
                    while i < 8 {
                        let mut j = 0;
                        #[unroll]
                        while j < TN {
                            acc[i][j] += a[i] * b[j];
                            j += 1;
                        }
                        i += 1;
                    }
                    k += 1;
                }
                // closes reads of this buffer before the next iteration restages it
                thread::sync_threads();
                i_stage += 1;
            }

            let out = y.as_mut_ptr();
            let mut i = 0;
            #[unroll]
            while i < 8 {
                let channel = channel0 + if i < 4 { 4 * cg + i as u32 } else { COT / 2 + 4 * cg + i as u32 - 4 };
                let b = bias[channel as usize];
                let row = (item * CO + channel) * PLANE;
                let mut j = 0;
                #[unroll]
                while j < TN {
                    let fo = p0 + TN as u32 * pg + j as u32;
                    if fo < PLANE {
                        // safety: an even pixel of the checked output, so the pair stays in
                        // its plane; this thread is its only writer
                        unsafe { stg2(out.add((row + fo) as usize), [acc[i][j] + b, acc[i][j + 1] + b]) };
                    }
                    j += 2;
                }
                i += 1;
            }
        }
    };
}

shortcut_tile! {
    /// 32 -> 64 shortcut from 80x998 to 40x499, 64-channel tiles
    ///
    /// Launch 128 threads, grid `(ceil(40 * 499 / 64), batch, 1)`
    spk_wideconv_shortcut_c32,
    cin = 32,
    cout = 64,
    h_in = 80,
    w_in = 998,
    channels = 64,
    chunk = 8,
}

shortcut_tile! {
    /// 64 -> 128 shortcut from 40x499 to 20x250, 64-channel tiles for small batches
    ///
    /// Launch 128 threads, grid `(ceil(20 * 250 / 64), batch, 2)`
    spk_wideconv_shortcut_c64,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    channels = 64,
    chunk = 16,
}

shortcut_tile! {
    /// 64 -> 128 shortcut from 40x499 to 20x250, 128-channel tiles for large batches
    ///
    /// Launch 128 threads, grid `(ceil(20 * 250 / 64), batch, 1)`
    spk_wideconv_shortcut_c64_wide,
    cin = 64,
    cout = 128,
    h_in = 40,
    w_in = 499,
    channels = 128,
    chunk = 16,
}

shortcut_tile! {
    /// 128 -> 256 shortcut from 20x250 to 10x125, 64-channel tiles for small batches
    ///
    /// Launch 128 threads, grid `(ceil(10 * 125 / 64), batch, 4)`
    spk_wideconv_shortcut_c128,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    channels = 64,
    chunk = 16,
}

shortcut_tile! {
    /// 128 -> 256 shortcut from 20x250 to 10x125, 128-channel tiles for large batches
    ///
    /// Launch 128 threads, grid `(ceil(10 * 125 / 64), batch, 2)`
    spk_wideconv_shortcut_c128_wide,
    cin = 128,
    cout = 256,
    h_in = 20,
    w_in = 250,
    channels = 128,
    chunk = 16,
}
