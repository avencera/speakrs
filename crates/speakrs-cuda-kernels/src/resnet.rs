//! ResNet34 3x3 convolutions of the first two embedding stages, with the folded
//! bias, optional residual add and ReLU fused into one kernel
//!
//! Three fixed shapes cover every eligible convolution of `layer1` and `layer2`:
//!
//! - [`spk_resnet_conv3x3_c32`]: 32 -> 32 channels, stride 1
//! - [`spk_resnet_conv3x3_c64`]: 64 -> 64 channels, stride 1
//! - [`spk_resnet_conv3x3_c32s2`]: 32 -> 64 channels, stride 2
//!
//! Activations stay NCHW. A block computes 64 output columns of
//! `threads * channels / (8 * cout)` output rows for every output channel of one batch
//! item.
//! The `_small` kernels split the same work into more blocks or more threads, so a
//! single item still spreads over every SM. Each
//! thread owns 8 adjacent output columns of one row and 8 output channels, the
//! channels `4g..4g + 4` and `cout / 2 + 4g..cout / 2 + 4g + 4` for its channel
//! group `g`, so the shared-memory reads of a warp hit 32 distinct banks. The small
//! 64-channel kernel gives each thread only `4g..4g + 4`, which doubles its warps for
//! the same work; every output still sums its products in the same order
//!
//! The input tile with its one-pixel halo and the matching weight slice are staged
//! in shared memory, a few input channels at a time. Each stage's global loads are
//! issued into registers before the previous stage's arithmetic, so their latency
//! overlaps it. Most kernels alternate two shared buffers, which measured faster; the
//! small 64-channel kernel uses one, so that enough of its blocks fit on an SM. A
//! barrier closes every stage either way. Padding
//! becomes exact zeros in the staged tile. For every input
//! channel and kernel row a thread reads one window of input columns and 24
//! weights and issues 192 FMAs, so loads stay a small share of the issued
//! instructions. Accumulation is plain FP32 FMA in a fixed order, the same in both
//! host math modes, with no atomics
//!
//! The epilogue stages the accumulators through shared memory again so that every
//! warp stores, and reads the residual, along contiguous output rows. It adds the
//! residual, then bias, then applies a ReLU that keeps NaN and negative zero
//!
//! [`spk_resnet_pack_weights`] packs the weights once per plan as
//! `[cin][ky][kx][cout]`, so one stage is a contiguous run that loads with 16-byte
//! transactions
//!
//! Index arithmetic is 32-bit: the host rejects tensors with more than `u32::MAX`
//! elements

use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, ptx_asm, thread};

// only the sm80 variant exports the TF32 tensor-core entries, so the sm75 PTX that the
// capability 12.0 production binding pins keeps its exact bytes
#[cfg(feature = "tier-sm80")]
pub mod tensor;

/// Output columns per block
pub const CONV_TILE_COLS: u32 = 64;

/// Packs folded weights `[cout][cin][3][3]` into `[cin][3][3][cout]`
///
/// Launch one thread per element of `packed`, which has as many elements as `weight`
#[kernel]
pub fn spk_resnet_pack_weights(
    weight: &[f32],
    cin: u32,
    cout: u32,
    mut packed: DisjointSlice<f32>,
) {
    let idx = thread::index_1d();
    let i = idx.get() as u32;
    let channel_out = i % cout;
    let rest = i / cout;
    let tap = rest % 9;
    let channel_in = rest / 9;
    if let Some(out) = packed.get_mut(idx) {
        *out = weight[((channel_out * cin + channel_in) * 9 + tap) as usize];
    }
}

/// Hides a shared address from LLVM, which otherwise proves the low bits are zero
/// and rewrites later `base + offset` additions as `base | offset`; `ptxas` folds
/// only the additions into the load and store immediates
#[inline(always)]
fn opaque(value: u32) -> u32 {
    let out: u32;
    // SAFETY: a register move with no memory access
    unsafe {
        ptx_asm!("mov.b32 %0, %1;", out("=r") out, in("r") value, options(register_only));
    }
    out
}

/// Four floats from a 16-byte aligned shared address
#[inline(always)]
unsafe fn lds4(address: u32) -> [f32; 4] {
    let (a, b, c, d): (f32, f32, f32, f32);
    // SAFETY: the caller passes an aligned address inside this block's shared tile
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
    // SAFETY: the caller passes an aligned address inside this block's shared tile
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
    // SAFETY: the caller passes an address inside this block's shared tile
    unsafe {
        ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") a, in("r") address);
    }
    a
}

/// Stores one float at a shared address
#[inline(always)]
unsafe fn sts1(address: u32, value: f32) {
    // SAFETY: the caller passes an address inside this block's shared tile that no
    // other thread accesses until the next barrier
    unsafe {
        ptx_asm!("st.shared.f32 [%0], %1;", in("r") address, in("f") value);
    }
}

/// Stores four floats at a 16-byte aligned shared address
#[inline(always)]
unsafe fn sts4(address: u32, value: [f32; 4]) {
    // SAFETY: as for `sts1`, with an aligned address
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
    // SAFETY: the caller passes an aligned pointer to four readable floats of a
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
            mut y: DisjointSlice<f32>,
        ) {
            const THREADS: u32 = $threads;
            const WARPS: u32 = THREADS / 32;
            const CIN: u32 = $cin;
            const COUT: u32 = $cout;
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
            const STAGES: u32 = CIN / CHUNK;
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
            let item = thread::blockIdx_z();
            let tile_y0 = thread::blockIdx_y() * TILE_ROWS;
            let tile_x0 = thread::blockIdx_x() * TILE_COLS;

            // the host sizes every buffer; a mismatch must not touch other memory
            let x_base = item * CIN * in_plane;
            let y_base = item * COUT * out_plane;
            if (x_base + CIN * in_plane) as usize > x.len()
                || (y_base + COUT * out_plane) as usize > y.len()
                || (add_residual != 0 && (y_base + COUT * out_plane) as usize > residual.len())
                || (CIN * 9 * COUT) as usize > weight.len()
                || COUT as usize > bias.len()
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

            // SAFETY: the static is this kernel's shared tile; only its address is taken
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
                        // SAFETY: inside the item's input, which the length check covers
                        unsafe { *x_ptr.add((x_base + seg_global[k] + col_x[j]) as usize) }
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
                    // SAFETY: inside the first stage of the packed weights
                    staged_w[i] = unsafe { ldg4(w_ptr.add((vector * 4) as usize)) };
                }
                i += 1;
            }

            let mut acc = [[0.0f32; CHANNELS]; 8];
            let in_offset = ((row * STRIDE) * ROW_STRIDE + col_group * 8 * STRIDE) * 4;
            let w_offset = IN_STAGE * 4 + group * LOW as u32 * 4;

            let mut stage = 0;
            while stage < STAGES {
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
                            // SAFETY: inside this buffer's input region; each
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
                        // SAFETY: inside this buffer's weight region, one vector per thread
                        unsafe { sts4(buffer + IN_STAGE * 4 + vector * 16, staged_w[i]) };
                    }
                    i += 1;
                }
                thread::sync_threads();

                // fetch the next stage while this one computes
                if stage + 1 < STAGES {
                    let channel_offset = x_base + (stage + 1) * CHUNK * in_plane;
                    let mut k = 0;
                    #[unroll]
                    while k < SEG_ITERS {
                        let mut j = 0;
                        #[unroll]
                        while j < COL_ITERS {
                            staged[k][j] = if seg_inside[k] && col_inside[j] {
                                // SAFETY: as for the first stage
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
                            // SAFETY: inside the next stage of the packed weights
                            staged_w[i] =
                                unsafe { ldg4(w_ptr.add((w_stage + vector * 4) as usize)) };
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
                            // SAFETY: the window lies inside the staged row
                            let quad = unsafe { lds4(row_address + v as u32 * 4) };
                            window[v] = quad[0];
                            window[v + 1] = quad[1];
                            window[v + 2] = quad[2];
                            window[v + 3] = quad[3];
                            v += 4;
                        }
                        if WINDOW - v >= 2 {
                            // SAFETY: as above, 8-byte aligned
                            let pair = unsafe { lds2(row_address + v as u32 * 4) };
                            window[v] = pair[0];
                            window[v + 1] = pair[1];
                            v += 2;
                        }
                        if WINDOW - v == 1 {
                            // SAFETY: as above
                            window[v] = unsafe { lds1(row_address + v as u32 * 4) };
                        }

                        let mut kx = 0;
                        #[unroll]
                        while kx < 3 {
                            let tap = (channel * 9 + ky * 3 + kx) * COUT * 4;
                            let low = if LOW == 4 {
                                // SAFETY: inside this buffer's weight region, 16-byte
                                // aligned
                                unsafe { lds4(w_base + tap) }
                            } else {
                                // SAFETY: as above, 8-byte aligned
                                let pair = unsafe { lds2(w_base + tap) };
                                [pair[0], pair[1], 0.0, 0.0]
                            };
                            let high = if HIGH {
                                // SAFETY: as above
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
                    if add_residual != 0 && oy < h_out && ox < w_out {
                        let channel = channel_base + (local / 2) * LOW as u32 + local % 2;
                        let index = y_base + channel * out_plane + oy * w_out + ox;
                        // SAFETY: inside the item's residual, which the length check
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
                    // SAFETY: inside the epilogue region; each (channel, position)
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
                        let channel = channel_base + (local / 2) * LOW as u32 + local % 2;
                        let index = (y_base + channel * out_plane + oy * w_out + ox) as usize;
                        // SAFETY: the epilogue slot was written before the barrier
                        let sum = unsafe { lds1(smem + (local * EPI_STRIDE + position) * 4) };
                        // the residual goes in before the bias, as cuDNN's fused call
                        // adds them, so the shared FMA order gives the same result
                        let value = if add_residual != 0 {
                            sum + shortcut[m]
                        } else {
                            sum
                        };
                        // SAFETY: `channel < COUT`, which the length check covers;
                        // the channel is the same across the block, so this hits cache
                        let value = value + unsafe { *bias_ptr.add(channel as usize) };
                        // a NaN and negative zero fail the comparison and pass through
                        let value = if value < 0.0 { 0.0 } else { value };
                        // SAFETY: inside the item's output; each element has one writer
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
    /// `y = relu(conv3x3(x, weight) + bias [+ residual])` for 32 -> 32 channels,
    /// stride 1, padding 1
    ///
    /// `x` is `[b, 32, h, w]`, `weight` the packed `[32][3][3][32]` weights,
    /// `bias` `[32]`, and `y` and `residual` `[b, 32, h, w]`. `residual` is read
    /// only when `add_residual` is nonzero. Launch 256 threads with
    /// `grid = (ceil(w / 64), ceil(h / 8), b)`
    spk_resnet_conv3x3_c32,
    threads = 256,
    min_blocks = 2,
    cin = 32,
    cout = 32,
    stride = 1,
    channels = 8,
    groups = 4,
    warp_cols = 4,
    warp_rows = 2,
    chunk = 2,
    buffers = 2,
    row_stride = 68,
    window = 10,
}

conv3x3! {
    /// `y = relu(conv3x3(x, weight) + bias [+ residual])` for 64 -> 64 channels,
    /// stride 1, padding 1
    ///
    /// `x` is `[b, 64, h, w]`, `weight` the packed `[64][3][3][64]` weights,
    /// `bias` `[64]`, and `y` and `residual` `[b, 64, h, w]`. `residual` is read
    /// only when `add_residual` is nonzero. Launch 256 threads with
    /// `grid = (ceil(w / 64), ceil(h / 4), b)`
    spk_resnet_conv3x3_c64,
    threads = 256,
    min_blocks = 2,
    cin = 64,
    cout = 64,
    stride = 1,
    channels = 8,
    groups = 8,
    warp_cols = 4,
    warp_rows = 1,
    chunk = 4,
    buffers = 2,
    row_stride = 68,
    window = 10,
}

conv3x3! {
    /// [`spk_resnet_conv3x3_c64`] in blocks of one output row, with 4 output channels
    /// per thread, for batches too small to fill the GPU with 4-row blocks
    ///
    /// Launch 128 threads with `grid = (ceil(w / 64), h, b)`
    spk_resnet_conv3x3_c64_small,
    threads = 128,
    min_blocks = 5,
    cin = 64,
    cout = 64,
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
    /// `y = relu(conv3x3(x, weight) + bias [+ residual])` for 32 -> 64 channels,
    /// stride 2, padding 1
    ///
    /// `x` is `[b, 32, h, w]`, `weight` the packed `[32][3][3][64]` weights,
    /// `bias` `[64]`, and `y` and `residual` `[b, 64, (h + 1) / 2, (w + 1) / 2]`.
    /// `residual` is read only when `add_residual` is nonzero. Launch 256 threads
    /// with `grid = (ceil(w_out / 64), ceil(h_out / 4), b)`
    spk_resnet_conv3x3_c32s2,
    threads = 256,
    min_blocks = 2,
    cin = 32,
    cout = 64,
    stride = 2,
    channels = 8,
    groups = 8,
    warp_cols = 2,
    warp_rows = 2,
    chunk = 2,
    buffers = 2,
    row_stride = 132,
    window = 17,
}

conv3x3! {
    /// [`spk_resnet_conv3x3_c32s2`] in blocks of two output rows, for batches too
    /// small to fill the GPU with the 256-thread blocks
    ///
    /// Launch 128 threads with `grid = (ceil(w_out / 64), ceil(h_out / 2), b)`
    spk_resnet_conv3x3_c32s2_small,
    threads = 128,
    min_blocks = 4,
    cin = 32,
    cout = 64,
    stride = 2,
    channels = 8,
    groups = 8,
    warp_cols = 2,
    warp_rows = 2,
    chunk = 2,
    buffers = 2,
    row_stride = 132,
    window = 17,
}
