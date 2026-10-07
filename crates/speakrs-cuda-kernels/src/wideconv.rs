//! Wide ResNet convolutions with independent output-channel tiles
//!
//! A spatial CTA owns 64 channels by one row, or 32 channels by two rows
//! It covers 64 columns in either layout
//! Input-channel partitions write separate planes, then a fixed-order reduction
//! applies the folded bias, optional residual and ReLU without float atomics
//! The SIMT path always uses full FP32, including in TF32 mode

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, ptx_asm, thread};

pub mod shortcut;
pub mod tc3;
pub mod tensor;
pub mod wbf;
pub mod winograd;
pub mod wtc;

/// Output columns per block
pub const CONV_TILE_COLS: u32 = 64;

/// Packs folded weights `[cout][cin][taps]` into `[cin][taps][cout]`
///
/// The host supplies 9 taps for wide convolutions and 1 for shortcuts
///
/// Launch one thread per element of `packed`, which has as many elements as `weight`
#[kernel]
pub fn spk_wideconv_pack_weights(
    weight: &[f32],
    cin: u32,
    cout: u32,
    taps: u32,
    mut packed: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    let channel_out = i % cout;
    let rest = i / cout;
    let tap = rest % taps;
    let channel_in = rest / taps;
    if let Some(out) = packed.get_mut(idx) {
        *out = weight[((channel_out * cin + channel_in) * taps + tap) as usize];
    }
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

/// Two floats from an 8-byte aligned shared address
#[inline(always)]
unsafe fn lds2(address: u32) -> [f32; 2] {
    let (a, b): (f32, f32);
    // safety: the caller passes an aligned address inside this block's shared tile
    unsafe {
        ptx_asm!(
            "ld.shared.v2.f32 {%0, %1}, [%2];",
            out("=f") a,
            out("=f") b,
            in("r") address,
        );
    }
    [a, b]
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

/// Expands to one convolution kernel for a fixed shape and block size
///
/// - `threads`, `min_blocks`: threads per block, a multiple of 32, and the blocks per
///   SM the register budget must allow; a block covers `threads / (8 * groups)` rows
/// - `cin`, `cout`, `stride`: the convolution
/// - `channels`: output channels per thread, 8 (`4g..4g + 4` and
///   `cout / 2 + 4g..cout / 2 + 4g + 4`), 4 (`4g..4g + 4`) or 2 (`2g..2g + 2`)
/// - `groups`: channel groups per block, `cout / channels`
/// - `warp_cols`, `warp_rows`: column groups and rows of one warp, with
///   `groups * warp_cols * warp_rows == 32`
/// - `chunk`: input channels per pipeline stage
/// - `buffers`: shared stage buffers, 1 or 2
/// - `row_stride`: floats per staged input row, a multiple of 4 that keeps the
///   rows of one warp on distinct banks
/// - `window`: input columns one thread reads per kernel row, `7 * stride + 3`
macro_rules! conv3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        min_blocks = $min_blocks:literal,
        cin = $cin:expr,
        cout = $cout:expr,
        total_out = $total_out:expr,
        stride = $stride:expr,
        channels = $channels:expr,
        groups = $groups:expr,
        warp_cols = $warp_cols:expr,
        warp_rows = $warp_rows:expr,
        chunk = $chunk:expr,
        buffers = $buffers:expr,
        row_stride = $row_stride:expr,
        window = $window:expr $(,)?
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
            h_in: u32,
            w_in: u32,
            batch: u32,
            splits: u32,
            mut y: DisjointSlice<f32>,
        ) {
            const THREADS: u32 = $threads;
            const WARPS: u32 = THREADS / 32;
            const CIN: u32 = $cin;
            const COUT: u32 = $cout;
            const TOTAL_OUT: u32 = $total_out;
            const STRIDE: u32 = $stride;
            const CHANNELS: usize = $channels;
            // channels of a thread's group in the lower half, and whether it also owns
            // the matching channels of the upper half
            const LOW: usize = if CHANNELS > 4 { 4 } else { CHANNELS };
            const HIGH: bool = CHANNELS == 8;
            const PASSES: usize = CHANNELS / 2;
            const GROUPS: u32 = $groups;
            const WARP_COLS: u32 = $warp_cols;
            const WARP_ROWS: u32 = $warp_rows;
            const CHUNK: u32 = $chunk;
            const BUFFERS: u32 = $buffers;
            const ROW_STRIDE: u32 = $row_stride;
            const WINDOW: usize = $window;

            const TILE_ROWS: u32 = THREADS / (8 * GROUPS);
            const TILE_COLS: u32 = CONV_TILE_COLS;
            const POSITIONS: u32 = TILE_ROWS * TILE_COLS;
            const IN_ROWS: u32 = (TILE_ROWS - 1) * STRIDE + 3;
            const IN_COLS: u32 = (TILE_COLS - 1) * STRIDE + 3;
            const CHANNEL_STRIDE: u32 = IN_ROWS * ROW_STRIDE;
            const IN_STAGE: u32 = CHUNK * CHANNEL_STRIDE;
            const W_STAGE: u32 = CHUNK * 9 * COUT;
            const STAGE: u32 = IN_STAGE + W_STAGE;
            let stages = CIN / (CHUNK * splits);
            // the epilogue stages two of each thread's eight channels at a time
            const EPI_STRIDE: u32 = POSITIONS + 4;
            const EPI: u32 = 2 * GROUPS * EPI_STRIDE;
            const SMEM: usize = if BUFFERS * STAGE > EPI {
                (BUFFERS * STAGE) as usize
            } else {
                EPI as usize
            };
            // input row segments per stage, spread over the warps
            const SEGMENTS: u32 = CHUNK * IN_ROWS;
            const SEG_ITERS: usize = SEGMENTS.div_ceil(WARPS) as usize;
            const COL_ITERS: usize = IN_COLS.div_ceil(32) as usize;
            const W_VECTORS: u32 = W_STAGE / 4;
            const W_ITERS: usize = W_VECTORS.div_ceil(THREADS) as usize;
            const EPI_ITERS: usize = (2 * GROUPS * POSITIONS / THREADS) as usize;

            static mut SMEM_TILE: SharedArray<f32, SMEM, 16> = SharedArray::UNINIT;

            // model heights are positive; h_in == 0 wraps and stores no output
            let h_out = (h_in - 1) / STRIDE + 1;
            let w_out = (w_in - 1) / STRIDE + 1;
            let in_plane = h_in * w_in;
            let out_plane = h_out * w_out;
            let grid_z = thread::blockIdx_z();
            let channel_tile = grid_z % (TOTAL_OUT / COUT);
            let item = (grid_z / (TOTAL_OUT / COUT)) % batch;
            let split = grid_z / ((TOTAL_OUT / COUT) * batch);
            let channel_base_out = channel_tile * COUT;
            let input_start = split * (CIN / splits);
            let tile_y0 = thread::blockIdx_y() * TILE_ROWS;
            let tile_x0 = thread::blockIdx_x() * TILE_COLS;

            // the host sizes every buffer; a mismatch must not touch other memory
            let x_base = item * CIN * in_plane;
            let y_base = (split * batch + item) * TOTAL_OUT * out_plane;
            if (x_base + CIN * in_plane) as usize > x.len()
                || (y_base + TOTAL_OUT * out_plane) as usize > y.len()
                || (splits == 1 && add_residual != 0 && (y_base + TOTAL_OUT * out_plane) as usize > residual.len())
                || (CIN * 9 * TOTAL_OUT) as usize > weight.len()
                || TOTAL_OUT as usize > bias.len()
                || tile_y0 >= h_out
                || tile_x0 >= w_out
            {
                return;
            }

            let tid = thread::threadIdx_x();
            let lane = tid % 32;
            let warp = tid / 32;
            let group = lane % GROUPS;
            let within = lane / GROUPS;
            let col_group = (warp % (8 / WARP_COLS)) * WARP_COLS + within % WARP_COLS;
            let row = (warp / (8 / WARP_COLS)) * WARP_ROWS + within / WARP_COLS;

            // safety: the static is this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const SMEM_TILE as *const u8) };

            // per-thread staging assignment, the same in every stage except for the
            // input channel offset
            let mut seg_global = [0u32; SEG_ITERS];
            let mut seg_shared = [0u32; SEG_ITERS];
            let mut seg_live = [false; SEG_ITERS];
            let mut seg_inside = [false; SEG_ITERS];
            let mut k = 0;
            #[unroll]
            while k < SEG_ITERS {
                let segment = warp + WARPS * k as u32;
                let channel = segment / IN_ROWS;
                let in_row = segment % IN_ROWS;
                let y_in = (tile_y0 * STRIDE + in_row) as i32 - 1;
                seg_live[k] = segment < SEGMENTS;
                seg_inside[k] = segment < SEGMENTS && y_in >= 0 && (y_in as u32) < h_in;
                seg_global[k] = channel * in_plane + (y_in.max(0) as u32) * w_in;
                seg_shared[k] = (channel * CHANNEL_STRIDE + in_row * ROW_STRIDE) * 4;
                k += 1;
            }
            let mut col_x = [0u32; COL_ITERS];
            let mut col_live = [false; COL_ITERS];
            let mut col_inside = [false; COL_ITERS];
            let mut j = 0;
            #[unroll]
            while j < COL_ITERS {
                let column = lane + 32 * j as u32;
                let x_in = (tile_x0 * STRIDE + column) as i32 - 1;
                col_live[j] = column < IN_COLS;
                col_inside[j] = column < IN_COLS && x_in >= 0 && (x_in as u32) < w_in;
                col_x[j] = x_in.max(0) as u32;
                j += 1;
            }

            let x_ptr = x.as_ptr();
            let w_ptr = weight.as_ptr();
            let mut staged = [[0.0f32; COL_ITERS]; SEG_ITERS];
            let mut staged_w = [[0.0f32; 4]; W_ITERS];

            // the first stage
            let mut k = 0;
            #[unroll]
            while k < SEG_ITERS {
                let mut j = 0;
                #[unroll]
                while j < COL_ITERS {
                    staged[k][j] = if seg_inside[k] && col_inside[j] {
                        // safety: inside the item's input, which the length check covers
                        unsafe { *x_ptr.add((x_base + input_start * in_plane + seg_global[k] + col_x[j]) as usize) }
                    } else {
                        0.0
                    };
                    j += 1;
                }
                k += 1;
            }
            let mut i = 0;
            #[unroll]
            while i < W_ITERS {
                let vector = tid + THREADS * i as u32;
                if vector < W_VECTORS {
                    // safety: inside the first stage of the packed weights
                    staged_w[i] = unsafe { ldg4(w_ptr.add((input_start * 9 * TOTAL_OUT + (vector * 4 / COUT) * TOTAL_OUT + channel_base_out + vector * 4 % COUT) as usize)) };
                }
                i += 1;
            }

            let mut acc = [[0.0f32; CHANNELS]; 8];
            let in_offset = ((row * STRIDE) * ROW_STRIDE + col_group * 8 * STRIDE) * 4;
            let w_offset = IN_STAGE * 4 + group * LOW as u32 * 4;

            let mut stage = 0;
            while stage < stages {
                // the barrier closing each stage frees its buffer for any later stage
                let buffer = smem + (stage % BUFFERS) * STAGE * 4;

                // publish the staged registers into this stage's buffer
                let mut k = 0;
                #[unroll]
                while k < SEG_ITERS {
                    let mut j = 0;
                    #[unroll]
                    while j < COL_ITERS {
                        if seg_live[k] && col_live[j] {
                            let column = lane + 32 * j as u32;
                            // safety: inside this buffer's input region; each
                            // (segment, column) belongs to one thread
                            unsafe { sts1(buffer + seg_shared[k] + column * 4, staged[k][j]) };
                        }
                        j += 1;
                    }
                    k += 1;
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
                if stage + 1 < stages {
                    let channel_offset = x_base + (input_start + (stage + 1) * CHUNK) * in_plane;
                    let mut k = 0;
                    #[unroll]
                    while k < SEG_ITERS {
                        let mut j = 0;
                        #[unroll]
                        while j < COL_ITERS {
                            staged[k][j] = if seg_inside[k] && col_inside[j] {
                                // safety: as for the first stage
                                unsafe {
                                    *x_ptr.add((channel_offset + seg_global[k] + col_x[j]) as usize)
                                }
                            } else {
                                0.0
                            };
                            j += 1;
                        }
                        k += 1;
                    }
                    let w_stage = (stage + 1) * W_STAGE;
                    let mut i = 0;
                    #[unroll]
                    while i < W_ITERS {
                        let vector = tid + THREADS * i as u32;
                        if vector < W_VECTORS {
                            // safety: inside the next stage of the packed weights
                            staged_w[i] =
                                unsafe { ldg4(w_ptr.add((input_start * 9 * TOTAL_OUT + ((w_stage + vector * 4) / COUT) * TOTAL_OUT + channel_base_out + vector * 4 % COUT) as usize)) };
                        }
                        i += 1;
                    }
                }

                let in_base = opaque(buffer + in_offset);
                let w_base = opaque(buffer + w_offset);
                let mut channel = 0;
                #[unroll]
                while channel < CHUNK {
                    let mut ky = 0;
                    #[unroll]
                    while ky < 3 {
                        let row_address = in_base + (channel * CHANNEL_STRIDE + ky * ROW_STRIDE) * 4;
                        let mut window = [0.0f32; WINDOW];
                        let mut v = 0;
                        #[unroll]
                        while v + 4 <= WINDOW {
                            // safety: the window lies inside the staged row
                            let quad = unsafe { lds4(row_address + v as u32 * 4) };
                            window[v] = quad[0];
                            window[v + 1] = quad[1];
                            window[v + 2] = quad[2];
                            window[v + 3] = quad[3];
                            v += 4;
                        }
                        if WINDOW - v >= 2 {
                            // safety: as above, 8-byte aligned
                            let pair = unsafe { lds2(row_address + v as u32 * 4) };
                            window[v] = pair[0];
                            window[v + 1] = pair[1];
                            v += 2;
                        }
                        if WINDOW - v == 1 {
                            // safety: as above
                            window[v] = unsafe { lds1(row_address + v as u32 * 4) };
                        }

                        let mut kx = 0;
                        #[unroll]
                        while kx < 3 {
                            let tap = (channel * 9 + ky * 3 + kx) * COUT * 4;
                            let low = if LOW == 4 {
                                // safety: inside this buffer's weight region, 16-byte
                                // aligned
                                unsafe { lds4(w_base + tap) }
                            } else {
                                // safety: as above, 8-byte aligned
                                let pair = unsafe { lds2(w_base + tap) };
                                [pair[0], pair[1], 0.0, 0.0]
                            };
                            let high = if HIGH {
                                // safety: as above
                                unsafe { lds4(w_base + tap + COUT * 2) }
                            } else {
                                [0.0; 4]
                            };
                            let mut p = 0;
                            #[unroll]
                            while p < 8 {
                                let value = window[p * STRIDE as usize + kx as usize];
                                let mut c = 0;
                                #[unroll]
                                while c < LOW {
                                    acc[p][c] += value * low[c];
                                    if HIGH {
                                        acc[p][c + 4] += value * high[c];
                                    }
                                    c += 1;
                                }
                                p += 1;
                            }
                            kx += 1;
                        }
                        ky += 1;
                    }
                    channel += 1;
                }
                thread::sync_threads();
                stage += 1;
            }

            // epilogue: one pass per two of this thread's channels. A
            // pass first issues all of its residual loads, so their latency
            // overlaps the shared-memory exchange instead of serializing per element
            let bias_ptr = bias.as_ptr();
            let residual_ptr = residual.as_ptr();
            let epi_write = smem + (row * TILE_COLS + col_group * 8) * 4;
            let mut pass = 0;
            #[unroll]
            while pass < PASSES {
                let channel_base = if pass < 2 {
                    2 * pass as u32
                } else {
                    COUT / 2 + 2 * pass as u32 - 4
                };
                let mut shortcut = [0.0f32; EPI_ITERS];
                let mut m = 0;
                #[unroll]
                while m < EPI_ITERS {
                    let e = tid + THREADS * m as u32;
                    let local = e / POSITIONS;
                    let position = e % POSITIONS;
                    let oy = tile_y0 + position / TILE_COLS;
                    let ox = tile_x0 + position % TILE_COLS;
                    if splits == 1 && add_residual != 0 && oy < h_out && ox < w_out {
                        let channel = channel_base_out + channel_base + (local / 2) * LOW as u32 + local % 2;
                        let index = y_base + channel * out_plane + oy * w_out + ox;
                        // safety: inside the item's residual, which the length check
                        // covers
                        shortcut[m] = unsafe { *residual_ptr.add(index as usize) };
                    }
                    m += 1;
                }

                let mut half = 0;
                #[unroll]
                while half < 2 {
                    let slot = pass * 2 + half;
                    let local = group * 2 + half as u32;
                    let address = epi_write + local * EPI_STRIDE * 4;
                    // safety: inside the epilogue region; each (channel, position)
                    // belongs to one thread
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
                thread::sync_threads();

                let mut m = 0;
                #[unroll]
                while m < EPI_ITERS {
                    let e = tid + THREADS * m as u32;
                    let local = e / POSITIONS;
                    let position = e % POSITIONS;
                    let oy = tile_y0 + position / TILE_COLS;
                    let ox = tile_x0 + position % TILE_COLS;
                    if oy < h_out && ox < w_out {
                        let channel = channel_base_out + channel_base + (local / 2) * LOW as u32 + local % 2;
                        let index = (y_base + channel * out_plane + oy * w_out + ox) as usize;
                        // safety: the epilogue slot was written before the barrier
                        let sum = unsafe { lds1(smem + (local * EPI_STRIDE + position) * 4) };
                        // the residual goes in before the bias, as cuDNN's fused call
                        // adds them, so the shared FMA order gives the same result
                        let value = if splits == 1 && add_residual != 0 {
                            sum + shortcut[m]
                        } else {
                            sum
                        };
                        // safety: `channel < COUT`, which the length check covers;
                        // the bias is immutable and reused across spatial tiles
                        let value = if splits == 1 { value + unsafe { *bias_ptr.add(channel as usize) } } else { value };
                        // a NaN and negative zero fail the comparison and pass through
                        let value = if splits == 1 && value < 0.0 { 0.0 } else { value };
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

conv3x3! {
    /// Computes a 64-channel tile of 128 -> 128 3x3 convolution, stride 1
    ///
    /// Launch 128 threads, grid `(ceil(w_out / 64), h_out, batch * 2 * splits)`
    /// Packed weights have layout `[cin][ky][kx][cout]`
    /// With splits > 1, output is `[splits][batch][cout][h_out][w_out]`
    /// The host requires splits to divide `cin / 4` and launches a reduction next
    spk_wideconv_c128,
    threads = 128,
    min_blocks = 4,
    cin = 128,
    cout = 64,
    total_out = 128,
    stride = 1,
    channels = 4,
    groups = 16,
    warp_cols = 2,
    warp_rows = 1,
    chunk = 4,
    buffers = 1,
    row_stride = 68,
    window = 10,
}

conv3x3! {
    /// Computes a 64-channel tile of 256 -> 256 3x3 convolution, stride 1
    ///
    /// Launch 128 threads, grid `(ceil(w_out / 64), h_out, batch * 4 * splits)`
    /// Packed weights have layout `[cin][ky][kx][cout]`
    /// With splits > 1, output is `[splits][batch][cout][h_out][w_out]`
    /// The host requires splits to divide `cin / 4` and launches a reduction next
    spk_wideconv_c256,
    threads = 128,
    min_blocks = 4,
    cin = 256,
    cout = 64,
    total_out = 256,
    stride = 1,
    channels = 4,
    groups = 16,
    warp_cols = 2,
    warp_rows = 1,
    chunk = 4,
    buffers = 1,
    row_stride = 68,
    window = 10,
}

conv3x3! {
    /// Computes a 32-channel tile of 64 -> 128 3x3 convolution, stride 2
    ///
    /// Launch 128 threads, grid `(ceil(w_out / 64), ceil(h_out / 2), batch * 4 * splits)`
    /// Packed weights have layout `[cin][ky][kx][cout]`
    /// With splits > 1, output is `[splits][batch][cout][h_out][w_out]`
    /// The host requires splits to divide `cin / 4` and launches a reduction next
    spk_wideconv_c64s2,
    threads = 128,
    min_blocks = 3,
    cin = 64,
    cout = 32,
    total_out = 128,
    stride = 2,
    channels = 4,
    groups = 8,
    warp_cols = 2,
    warp_rows = 2,
    chunk = 4,
    buffers = 1,
    row_stride = 132,
    window = 17,
}

conv3x3! {
    /// Computes a 64-channel tile of 128 -> 256 3x3 convolution, stride 2
    ///
    /// Launch 128 threads, grid `(ceil(w_out / 64), h_out, batch * 4 * splits)`
    /// Packed weights have layout `[cin][ky][kx][cout]`
    /// With splits > 1, output is `[splits][batch][cout][h_out][w_out]`
    /// The host requires splits to divide `cin / 4` and launches a reduction next
    spk_wideconv_c128s2,
    threads = 128,
    min_blocks = 3,
    cin = 128,
    cout = 64,
    total_out = 256,
    stride = 2,
    channels = 4,
    groups = 16,
    warp_cols = 2,
    warp_rows = 1,
    chunk = 4,
    buffers = 1,
    row_stride = 132,
    window = 17,
}

/// Reduces input-channel partitions in a fixed order, then adds residual and bias
///
/// Launch one thread per output element; `partial` is `[splits][output.len()]`
#[kernel]
#[launch_bounds(256)]
pub fn spk_wideconv_reduce(
    partial: &[f32],
    bias: &[f32],
    residual: &[f32],
    add_residual: u32,
    splits: u32,
    plane: u32,
    channels: u32,
    mut output: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get();
    if i >= output.len() {
        return;
    }
    let mut value = 0.0f32;
    let mut split = 0;
    while split < splits {
        value += partial[split as usize * output.len() + i];
        split += 1;
    }
    if add_residual != 0 {
        value += residual[i];
    }
    value += bias[(i as u32 / plane % channels) as usize];
    if value < 0.0 {
        value = 0.0;
    }
    if let Some(out) = output.get_mut(idx) {
        *out = value;
    }
}

/// Expands to one 1 -> 32 stem kernel; each thread computes `pixels` consecutive pixels
/// of one plane, in all channels, and stores each channel's run with `store`
///
/// The layer only writes: 32 output floats per input float. Each output plane
/// `(item, channel)` is contiguous, so a thread's pixels take 16-byte stores per channel
/// even where they straddle a row. `vector` is nonzero only when the plane length is a
/// multiple of `pixels` and the output is 16-byte aligned; otherwise each pixel stores
/// singly. No residual is part of the stem contract
macro_rules! stem3x3 {
    (
        $(#[$doc:meta])*
        $name:ident,
        threads = $threads:literal,
        pixels = $pixels:literal,
        store = $store:ident $(,)?
    ) => {
        $(#[$doc])*
        #[kernel]
        #[launch_bounds($threads)]
        pub fn $name(
            x: &[f32],
            weight: &[f32],
            bias: &[f32],
            h: u32,
            w: u32,
            vector: u32,
            mut output: DisjointSlice<f32>,
        ) {
            const THREADS: u32 = $threads;
            const PIXELS: usize = $pixels;
            // nine weights and the bias per channel, padded to 12 words so each channel's
            // block is 16-byte aligned for uniform vector loads
            static mut STEM_TAPS: SharedArray<f32, { 32 * STEM_CHANNEL_WORDS as usize }, 16> =
                SharedArray::UNINIT;

            let tid = thread::threadIdx_x();
            // safety: this kernel's shared tile; only its address is taken
            let smem = unsafe { cvta_generic_to_shared_u32(&raw const STEM_TAPS as *const u8) };
            if weight.len() < 32 * 9 || bias.len() < 32 {
                return;
            }
            let mut k = tid;
            while k < 32 * STEM_CHANNEL_WORDS {
                let channel = k / STEM_CHANNEL_WORDS;
                let slot = k % STEM_CHANNEL_WORDS;
                let value = if slot < 9 {
                    weight[(channel * 9 + slot) as usize]
                } else if slot == 9 {
                    bias[channel as usize]
                } else {
                    0.0
                };
                // safety: `k` indexes the shared tile; the barrier below publishes it
                unsafe { sts1(smem + k * 4, value) };
                k += THREADS;
            }
            thread::sync_threads();

            let plane = h * w;
            let item = thread::blockIdx_y();
            let f0 = (thread::blockIdx_x() * THREADS + tid) * PIXELS as u32;
            if f0 >= plane {
                return;
            }
            let base = item * plane;
            let mut window = [[0.0f32; 9]; PIXELS];
            let mut p = 0;
            #[unroll]
            while p < PIXELS {
                let f = f0 + p as u32;
                if f < plane {
                    let row = f / w;
                    let col = f - row * w;
                    let mut tap = 0;
                    #[unroll]
                    while tap < 9 {
                        let iy = row as i32 + (tap / 3) as i32 - 1;
                        let ix = col as i32 + (tap % 3) as i32 - 1;
                        if iy >= 0 && iy < h as i32 && ix >= 0 && ix < w as i32 {
                            window[p][tap] = x[(base + iy as u32 * w + ix as u32) as usize];
                        }
                        tap += 1;
                    }
                }
                p += 1;
            }

            let out = output.as_mut_ptr();
            let mut c = 0;
            #[unroll]
            while c < 32 {
                let block = smem + c * STEM_CHANNEL_WORDS * 4;
                // safety: uniform aligned addresses inside the published tile
                let (t0, t1, t2) = unsafe { (lds4(block), lds4(block + 16), lds4(block + 32)) };
                let taps = [
                    t0[0], t0[1], t0[2], t0[3], t1[0], t1[1], t1[2], t1[3], t2[0],
                ];
                let mut value = [0.0f32; PIXELS];
                let mut p = 0;
                #[unroll]
                while p < PIXELS {
                    let mut sum = 0.0f32;
                    let mut tap = 0;
                    #[unroll]
                    while tap < 9 {
                        sum += window[p][tap] * taps[tap];
                        tap += 1;
                    }
                    let sum = sum + t2[1];
                    value[p] = if sum < 0.0 { 0.0 } else { sum };
                    p += 1;
                }
                let index = ((item * 32 + c) * plane + f0) as usize;
                // safety: the host validates the NCHW output and `vector`; each pixel of each
                // plane has one writer
                unsafe {
                    if vector != 0 {
                        let mut q = 0;
                        #[unroll]
                        while q < PIXELS {
                            $store(out.add(index + q), [value[q], value[q + 1], value[q + 2], value[q + 3]]);
                            q += 4;
                        }
                    } else {
                        let mut p = 0;
                        #[unroll]
                        while p < PIXELS {
                            if f0 + (p as u32) < plane {
                                *out.add(index + p) = value[p];
                            }
                            p += 1;
                        }
                    }
                }
                c += 1;
            }
        }
    };
}

stem3x3! {
    /// Stem with four pixels per thread of a 128-thread CTA: a warp's stores cover 512
    /// contiguous bytes per channel
    ///
    /// Launch 128 threads, grid `(ceil(h * w / 512), batch)`
    spk_wideconv_stem,
    threads = 128,
    pixels = 4,
    store = stg4,
}

stem3x3! {
    /// Stem with eight pixels per thread of a 256-thread CTA and streaming stores
    /// (`st.global.cs`), for grids of many CTAs: the output is not read again by this
    /// launch. On a 4060 Ti at batch 32 it ran 1.255 ms against 1.303 ms for
    /// `spk_wideconv_stem` and cuDNN's 1.288 ms, with the DRAM floor near 1.23 ms; at
    /// batch 1 its 39 CTAs left the SMs latency-bound (0.032 against 0.013 ms)
    ///
    /// Launch 256 threads, grid `(ceil(h * w / 2048), batch)`
    spk_wideconv_stem_wide,
    threads = 256,
    pixels = 8,
    store = stg4_streaming,
}

/// Stores two floats at an 8-byte aligned global address
#[inline(always)]
unsafe fn stg2(pointer: *mut f32, value: [f32; 2]) {
    // safety: the caller passes an aligned pointer to two floats only this thread writes
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

/// Shared words per stem output channel: nine weights, the bias and two pad words
const STEM_CHANNEL_WORDS: u32 = 12;

/// Stores four floats at a 16-byte aligned global address
#[inline(always)]
unsafe fn stg4(pointer: *mut f32, value: [f32; 4]) {
    // safety: the caller passes an aligned pointer to four floats only this thread writes
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

/// Stores four floats at a 16-byte aligned global address with the streaming hint
#[inline(always)]
unsafe fn stg4_streaming(pointer: *mut f32, value: [f32; 4]) {
    // safety: the caller passes an aligned pointer to four floats only this thread writes
    unsafe {
        ptx_asm!(
            "{ .reg .u64 g; cvta.to.global.u64 g, %0; st.global.cs.v4.f32 [g], {%1, %2, %3, %4}; }",
            in("l") pointer as u64,
            in("f") value[0],
            in("f") value[1],
            in("f") value[2],
            in("f") value[3],
            clobber("memory"),
        );
    }
}

macro_rules! fixed {
    ($index:ident in [$($value:literal),*] $body:block) => { $( { let $index = $value; $body } )* };
}

// eight padding words give the four TF32 K fragments distinct bank groups
const GEMM_WEIGHT_STRIDE: u32 = 40;
const GEMM_WEIGHT_WORDS: u32 = 32 * GEMM_WEIGHT_STRIDE;

/// Copies one float asynchronously on sm80, with scalar publication on sm75
#[inline(always)]
unsafe fn stage_scalar(dst: u32, src: *const f32) {
    #[cfg(feature = "tier-sm80")]
    // safety: the caller supplies valid four-byte aligned shared and global addresses
    unsafe {
        ptx_asm!("{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.ca.shared.global [%0], [g], 4; }",
            in("r") dst, in("l") src as u64, clobber("memory"));
    }
    #[cfg(not(feature = "tier-sm80"))]
    // safety: the caller supplies valid scalar addresses
    unsafe {
        sts1(dst, *src);
    }
}

/// Copies four contiguous weights without a register staging array
#[inline(always)]
unsafe fn stage_quad(dst: u32, src: *const f32) {
    #[cfg(feature = "tier-sm80")]
    // safety: the caller supplies valid 16-byte aligned shared and global addresses
    unsafe {
        ptx_asm!("{ .reg .u64 g; cvta.to.global.u64 g, %1; cp.async.ca.shared.global [%0], [g], 16; }",
            in("r") dst, in("l") src as u64, clobber("memory"));
    }
    #[cfg(not(feature = "tier-sm80"))]
    // safety: the caller supplies valid aligned four-float regions
    unsafe {
        sts4(dst, ldg4(src));
    }
}

#[inline(always)]
fn commit() {
    #[cfg(feature = "tier-sm80")]
    // safety: commits this lane's staged copies, without making them visible to other lanes
    unsafe {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
fn wait(last: bool) {
    #[cfg(feature = "tier-sm80")]
    // safety: one or two committed groups remain; the caller publishes them with a CTA barrier
    unsafe {
        if last {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
        } else {
            ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
        }
    }
    #[cfg(not(feature = "tier-sm80"))]
    let _ = last;
}

#[inline(always)]
unsafe fn gemm_stage(
    buffer: u32,
    tid: u32,
    stage: u32,
    start: u32,
    end: u32,
    ci: u32,
    co: u32,
    h: u32,
    w: u32,
    stride: u32,
    item: u32,
    channel0: u32,
    position0: u32,
    ow: u32,
    oh: u32,
    x: *const f32,
    weight: *const f32,
    shortcut: u32,
) {
    fixed!(i in [0, 1] {
        let v = tid + i * 128;
        let red = start + stage * 32 + v / 8;
        let channel = channel0 + v % 8 * 4;
        let dst = buffer + (v / 8 * GEMM_WEIGHT_STRIDE + v % 8 * 4) * 4;
        if red < end {
            // safety: complete aligned output-channel vectors lie inside packed weights
            unsafe {
                stage_quad(dst, weight.add((red * co + channel) as usize));
            }
        } else {
            // safety: every masked weight vector has one writer before publication
            unsafe {
                sts4(dst, [0.0; 4]);
            }
        }
    });
    fixed!(i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] {
        let e = tid + i * 128;
        let red = start + stage * 32 + e / 64;
        let position = position0 + e % 64;
        let (iy, ix, input_channel) = if shortcut == 0 {
            ((position / ow * stride + red % 9 / 3) as i32 - 1,
             (position % ow * stride + red % 3) as i32 - 1, red / 9)
        } else { ((position / ow * stride) as i32, (position % ow * stride) as i32, red) };
        let dst = buffer + (GEMM_WEIGHT_WORDS + e / 64 * 68 + e % 64) * 4;
        if red < end && position < oh * ow && iy >= 0 && iy < h as i32 && ix >= 0 && ix < w as i32 {
            let offset = (item * ci + input_channel) * h * w + iy as u32 * w + ix as u32;
            // safety: padding and the partition boundary are checked before the copy
            unsafe {
                stage_scalar(dst, x.add(offset as usize));
            }
        } else {
            // safety: every masked input has one writer before publication
            unsafe {
                sts1(dst, 0.0);
            }
        }
    });
    commit();
}

/// Implicit GEMM, 32 output channels by 64 flattened spatial positions per CTA
///
/// Launch 128 threads, grid `(ceil(oh * ow / 64), 1, batch * cout / 32 * splits)`
/// sm75 always uses FP32 SIMT; sm80 uses TF32 MMA only when `tf32` is nonzero
/// Two copy stages have distinct ownership, and reduction planes never use atomics
#[kernel]
#[launch_bounds(128, 3)]
pub fn spk_wideconv_gemm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    residual: &[f32],
    add_residual: u32,
    batch: u32,
    ci: u32,
    co: u32,
    h: u32,
    w: u32,
    stride: u32,
    splits: u32,
    tf32: u32,
    shortcut: u32,
    mut output: DisjointSlice<f32>,
) {
    const STAGE: u32 = GEMM_WEIGHT_WORDS + 32 * 68;
    static mut TILE: SharedArray<f32, { STAGE as usize * 2 }, 16> = SharedArray::UNINIT;
    // safety: this address belongs to the current CTA's static shared allocation
    let smem = unsafe { cvta_generic_to_shared_u32(&raw const TILE as *const u8) };
    let tid = thread::threadIdx_x();
    let lane = tid % 32;
    let warp = tid / 32;
    let z = thread::blockIdx_z();
    let channel0 = z % (co / 32) * 32;
    let item = z / (co / 32) % batch;
    let split = z / (co / 32) / batch;
    let taps = if shortcut == 0 { 9 } else { 1 };
    let start = split * (ci / splits) * taps;
    let end = start + ci / splits * taps;
    let stages = (end - start).div_ceil(32);
    let oh = (h - 1) / stride + 1;
    let ow = (w - 1) / stride + 1;
    let position0 = thread::blockIdx_x() * 64;
    // safety: the host validates complete input and packed weight allocations
    unsafe {
        gemm_stage(
            smem,
            tid,
            0,
            start,
            end,
            ci,
            co,
            h,
            w,
            stride,
            item,
            channel0,
            position0,
            ow,
            oh,
            x.as_ptr(),
            weight.as_ptr(),
            shortcut,
        );
        if stages > 1 {
            gemm_stage(
                smem + STAGE * 4,
                tid,
                1,
                start,
                end,
                ci,
                co,
                h,
                w,
                stride,
                item,
                channel0,
                position0,
                ow,
                oh,
                x.as_ptr(),
                weight.as_ptr(),
                shortcut,
            );
        }
    }
    let mut acc = [[0.0f32; 4]; 4];
    let mut stage = 0;
    while stage < stages {
        wait(stage + 1 == stages);
        thread::sync_threads();
        let buffer = opaque(smem + stage % 2 * STAGE * 4);
        #[cfg(feature = "tier-sm80")]
        if tf32 != 0 {
            let row = warp / 2 * 16 + lane / 4;
            let col = warp % 2 * 32;
            let t = lane % 4;
            let mut k = 0;
            #[unroll]
            while k < 4 {
                // safety: this stage's weight and input slots were copied or zero-filled
                let a = unsafe {
                    [
                        cuda_device::convert::cvt_rna_tf32_f32(lds1(
                            buffer + ((k * 8 + t) * GEMM_WEIGHT_STRIDE + row) * 4,
                        )),
                        cuda_device::convert::cvt_rna_tf32_f32(lds1(
                            buffer + ((k * 8 + t) * GEMM_WEIGHT_STRIDE + row + 8) * 4,
                        )),
                        cuda_device::convert::cvt_rna_tf32_f32(lds1(
                            buffer + ((k * 8 + t + 4) * GEMM_WEIGHT_STRIDE + row) * 4,
                        )),
                        cuda_device::convert::cvt_rna_tf32_f32(lds1(
                            buffer + ((k * 8 + t + 4) * GEMM_WEIGHT_STRIDE + row + 8) * 4,
                        )),
                    ]
                };
                let mut n = 0;
                #[unroll]
                while n < 4 {
                    // operand B is column-major: lane group chooses its spatial column, t chooses K
                    // safety: both fragment operands lie inside the published input tile
                    let b = unsafe {
                        [
                            cuda_device::convert::cvt_rna_tf32_f32(lds1(
                                buffer
                                    + (GEMM_WEIGHT_WORDS
                                        + (k * 8 + t) * 68
                                        + col
                                        + n as u32 * 8
                                        + lane / 4)
                                        * 4,
                            )),
                            cuda_device::convert::cvt_rna_tf32_f32(lds1(
                                buffer
                                    + (GEMM_WEIGHT_WORDS
                                        + (k * 8 + t + 4) * 68
                                        + col
                                        + n as u32 * 8
                                        + lane / 4)
                                        * 4,
                            )),
                        ]
                    };
                    acc[n] = unsafe { cuda_device::wmma::mma_m16n8k8_f32_tf32(acc[n], a, b) };
                    n += 1;
                }
                k += 1;
            }
        } else {
            simt_tile(buffer, tid, &mut acc);
        }
        #[cfg(not(feature = "tier-sm80"))]
        {
            let _ = (tf32, lane, warp);
            simt_tile(buffer, tid, &mut acc);
        }
        // consumers finish before this stage slot is reused by stage + 2
        thread::sync_threads();
        if stage + 2 < stages {
            // safety: the barrier closed all consumers of this slot
            unsafe {
                gemm_stage(
                    buffer,
                    tid,
                    stage + 2,
                    start,
                    end,
                    ci,
                    co,
                    h,
                    w,
                    stride,
                    item,
                    channel0,
                    position0,
                    ow,
                    oh,
                    x.as_ptr(),
                    weight.as_ptr(),
                    shortcut,
                );
            }
        }
        stage += 1;
    }
    let mut m = 0;
    #[unroll]
    while m < 4 {
        let mut n = 0;
        #[unroll]
        while n < 4 {
            let channel = channel0 + tid / 16 * 4 + n as u32;
            let position = position0 + tid % 16 * 4 + m as u32;
            #[cfg(feature = "tier-sm80")]
            let (channel, position) = if tf32 != 0 {
                (
                    channel0 + warp / 2 * 16 + lane / 4 + n as u32 / 2 * 8,
                    position0 + warp % 2 * 32 + m as u32 * 8 + lane % 4 * 2 + n as u32 % 2,
                )
            } else {
                (channel, position)
            };
            if position < oh * ow {
                let index = ((split * batch + item) * co + channel) * oh * ow + position;
                let mut value = acc[m][n];
                if splits == 1 {
                    if add_residual != 0 {
                        value += residual[index as usize];
                    }
                    value += bias[channel as usize];
                    if shortcut == 0 && value < 0.0 {
                        value = 0.0;
                    }
                }
                // safety: this SIMT or MMA lane owns one distinct output position/channel
                unsafe {
                    *output.get_unchecked_mut(index as usize) = value;
                }
            }
            n += 1;
        }
        m += 1;
    }
}

#[inline(always)]
fn simt_tile(buffer: u32, tid: u32, acc: &mut [[f32; 4]; 4]) {
    fixed!(k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] {
        // safety: all vectors are aligned and inside the current published tile
        let (weights, input) = unsafe {
            (
                lds4(buffer + (k * GEMM_WEIGHT_STRIDE + tid / 16 * 4) * 4),
                lds4(buffer + (GEMM_WEIGHT_WORDS + k * 68 + tid % 16 * 4) * 4),
            )
        };
        fixed!(m in [0, 1, 2, 3] {
            fixed!(n in [0, 1, 2, 3] {
                acc[m][n] += input[m] * weights[n];
            });
        });
    });
}
