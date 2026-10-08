use cudarc::driver::{CudaFunction, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};

use super::super::error::check_len;
use super::super::{CudaError, CudaRuntime, KernelModule, PtxTier};

const EMBEDDING_FBANK_TRANSPOSE: &str = "embedding_fbank_transpose";
const EMBEDDING_BIAS: &str = "embedding_bias";
const EMBEDDING_BROADCAST_ROWS: &str = "embedding_broadcast_rows";
const EMBEDDING_MASK_POOL: &str = "embedding_mask_pool";

/// Kernel entries loaded by this host plan
#[cfg(test)]
pub(crate) const REQUIRED_KERNELS: [&str; 4] = [
    EMBEDDING_FBANK_TRANSPOSE,
    EMBEDDING_BIAS,
    EMBEDDING_BROADCAST_ROWS,
    EMBEDDING_MASK_POOL,
];

/// Threads per block for every embedding kernel; a multiple of the warp size, as
/// the pooling kernel requires
const BLOCK_THREADS: u32 = 256;

/// Lanes per warp; the pooling kernel runs one warp per output column
const WARP_THREADS: usize = 32;

/// Elements per thread of the bias kernel; must equal `EPILOGUE_ITEMS` in the
/// kernel crate
const EPILOGUE_ITEMS: u32 = 4;

/// CUDA's limit on `gridDim.y`, which the bias kernel uses for `n * channels`
const MAX_GRID_Y: usize = 65_535;

/// Host launchers for the cuda-oxide kernels in `crates/speakrs-cuda-kernels/src/embedding.rs`
///
/// cuda-oxide passes each slice as a device pointer followed by a `u64` element
/// count, so every slice argument pushes the view and then its length
#[derive(Debug, Clone)]
pub(super) struct EmbeddingKernels {
    tier: PtxTier,
    fbank_transpose: CudaFunction,
    bias: CudaFunction,
    broadcast_rows: CudaFunction,
    mask_pool: CudaFunction,
}

/// Bias layout of an NCHW tensor: `channels` biases, each repeated over a `plane`
/// of `h * w` elements
#[derive(Debug, Clone, Copy)]
pub(super) struct ChannelBias {
    pub(super) channels: usize,
    pub(super) plane: usize,
}

/// Dimensions of the multi-mask pooling
#[derive(Debug, Clone, Copy)]
pub(super) struct PoolShape {
    pub(super) chunks: usize,
    pub(super) speakers: usize,
    /// channels times frequency bins of the trunk output
    pub(super) columns: usize,
    /// time frames of the trunk output
    pub(super) frames: usize,
    /// frames of each speaker mask before resizing
    pub(super) mask_frames: usize,
}

impl EmbeddingKernels {
    /// Loads the embedding module and looks up its kernels
    pub(super) fn load(runtime: &CudaRuntime) -> Result<Self, CudaError> {
        let kernels = runtime.load_kernels(KernelModule::Embedding)?;
        Ok(Self {
            tier: kernels.tier(),
            fbank_transpose: kernels.function(EMBEDDING_FBANK_TRANSPOSE)?,
            bias: kernels.function(EMBEDDING_BIAS)?,
            broadcast_rows: kernels.function(EMBEDDING_BROADCAST_ROWS)?,
            mask_pool: kernels.function(EMBEDDING_MASK_POOL)?,
        })
    }

    /// The PTX tier the module was loaded from
    pub(super) fn tier(&self) -> PtxTier {
        self.tier
    }

    /// `[b, frames, bins]` to `[b, 1, bins, frames]`
    pub(super) fn fbank_transpose(
        &self,
        runtime: &CudaRuntime,
        fbank: &CudaView<'_, f32>,
        frames: usize,
        bins: usize,
        out: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let len = out.len();
        check_len("embedding fbank transpose", len, fbank.len())?;
        let config = elementwise(len)?;
        let frames = to_u32(frames)?;
        let bins = to_u32(bins)?;

        let len = len as u64;
        #[cfg(all(test, feature = "_cuda-libraries"))]
        let _fixed = super::super::test_support::fixed("embedding_fbank_transpose");
        let mut launch = runtime.stream().launch_builder(&self.fbank_transpose);
        launch
            .arg(fbank)
            .arg(&len)
            .arg(&frames)
            .arg(&bins)
            .arg(out)
            .arg(&len);

        // SAFETY: the arguments match `embedding_fbank_transpose(fbank: &[f32],
        // frames: u32, bins: u32, out: DisjointSlice<f32>)`, both buffers hold the
        // same number of elements, and there is one thread per output element
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// `y = y + bias` in place over an NCHW tensor
    pub(super) fn bias(
        &self,
        runtime: &CudaRuntime,
        bias: &CudaView<'_, f32>,
        layout: ChannelBias,
        y: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        check_len("embedding conv bias", layout.channels, bias.len())?;
        check_channel_layout(layout, y.len())?;
        // one row of blocks per (item, channel) plane
        let rows = y.len() / layout.plane;
        if rows > MAX_GRID_Y {
            return Err(CudaError::DimensionOverflow {
                context: "embedding bias grid",
                value: rows,
            });
        }
        let channels = to_u32(layout.channels)?;
        let plane = to_u32(layout.plane)?;
        let config = LaunchConfig {
            grid_dim: (
                plane.div_ceil(BLOCK_THREADS * EPILOGUE_ITEMS),
                rows as u32,
                1,
            ),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };

        let bias_len = bias.len() as u64;
        let y_len = y.len() as u64;
        #[cfg(all(test, feature = "_cuda-libraries"))]
        let _fixed = super::super::test_support::fixed("embedding_bias");
        let mut launch = runtime.stream().launch_builder(&self.bias);
        launch
            .arg(bias)
            .arg(&bias_len)
            .arg(&channels)
            .arg(&plane)
            .arg(y)
            .arg(&y_len);

        // SAFETY: the arguments match `embedding_bias(bias, channels: u32, plane:
        // u32, y)`, `bias` holds one value per channel and `y` a whole number of
        // channel planes, and the grid has one row of blocks per plane covering it
        // `EPILOGUE_ITEMS` elements per thread
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Writes `bias` into every row of the row-major `out`
    pub(super) fn broadcast_rows(
        &self,
        runtime: &CudaRuntime,
        bias: &CudaView<'_, f32>,
        out: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let cols = bias.len();
        if cols == 0 || !out.len().is_multiple_of(cols) {
            return Err(CudaError::BufferLength {
                context: "embedding output rows",
                expected: cols * (out.len() / cols.max(1)),
                actual: out.len(),
            });
        }
        let config = elementwise(out.len())?;
        let cols_u32 = to_u32(cols)?;

        let bias_len = cols as u64;
        let out_len = out.len() as u64;
        #[cfg(all(test, feature = "_cuda-libraries"))]
        let _fixed = super::super::test_support::fixed("embedding_broadcast_rows");
        let mut launch = runtime.stream().launch_builder(&self.broadcast_rows);
        launch
            .arg(bias)
            .arg(&bias_len)
            .arg(&cols_u32)
            .arg(out)
            .arg(&out_len);

        // SAFETY: the arguments match `embedding_broadcast_rows(bias, cols: u32,
        // out)`, `out` holds whole rows of `cols`, and there is one thread per element
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// Multi-mask weighted mean and standard deviation pooling into
    /// `[chunks * speakers, 2 * columns]`
    pub(super) fn mask_pool(
        &self,
        runtime: &CudaRuntime,
        features: &CudaView<'_, f32>,
        masks: &CudaView<'_, f32>,
        shape: PoolShape,
        pooled: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let rows = shape.chunks * shape.speakers;
        check_len(
            "embedding pool features",
            shape.chunks * shape.columns * shape.frames,
            features.len(),
        )?;
        check_len(
            "embedding pool masks",
            rows * shape.mask_frames,
            masks.len(),
        )?;
        check_len(
            "embedding pool output",
            rows * 2 * shape.columns,
            pooled.len(),
        )?;

        let threads = to_u32(shape.chunks * shape.columns * WARP_THREADS)?;
        let config = LaunchConfig {
            grid_dim: (threads.div_ceil(BLOCK_THREADS), 1, 1),
            block_dim: (BLOCK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let speakers = to_u32(shape.speakers)?;
        let columns = to_u32(shape.columns)?;
        let frames = to_u32(shape.frames)?;
        let mask_frames = to_u32(shape.mask_frames)?;

        let features_len = features.len() as u64;
        let masks_len = masks.len() as u64;
        let pooled_len = pooled.len() as u64;
        #[cfg(all(test, feature = "_cuda-libraries"))]
        let _fixed = super::super::test_support::fixed("embedding_mask_pool");
        let mut launch = runtime.stream().launch_builder(&self.mask_pool);
        launch
            .arg(features)
            .arg(&features_len)
            .arg(masks)
            .arg(&masks_len)
            .arg(&speakers)
            .arg(&columns)
            .arg(&frames)
            .arg(&mask_frames)
            .arg(pooled)
            .arg(&pooled_len);

        // SAFETY: the arguments match `embedding_mask_pool(features, masks, speakers,
        // columns, frames, mask_frames: u32, pooled)`, the buffer lengths match the
        // shape checked above, and the grid has one full warp per (chunk, column)
        // in blocks that are a multiple of 32 threads
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

/// One thread per element in blocks of [`BLOCK_THREADS`]; the kernels use 32-bit
/// indices, so the element count must fit in a `u32`
fn elementwise(len: usize) -> Result<LaunchConfig, CudaError> {
    let threads = to_u32(len)?;
    Ok(LaunchConfig {
        grid_dim: (threads.div_ceil(BLOCK_THREADS).max(1), 1, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn check_channel_layout(layout: ChannelBias, len: usize) -> Result<(), CudaError> {
    let group = layout.channels * layout.plane;
    if group == 0 || !len.is_multiple_of(group) {
        return Err(CudaError::BufferLength {
            context: "embedding conv output",
            expected: group * (len / group.max(1)),
            actual: len,
        });
    }

    Ok(())
}

fn to_u32(value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow {
        context: "embedding kernel launch",
        value,
    })
}
