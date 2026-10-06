use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::super::error::check_len;
use super::super::{CudaError, CudaRuntime, KernelModule, PtxTier};
use super::shape::CLASSES;

const SEGMENTATION_POOL_NORM: &str = "segmentation_pool_norm";
const SEGMENTATION_BIAS_LEAKY: &str = "segmentation_bias_leaky";
const SEGMENTATION_BIAS_LOG_SOFTMAX: &str = "segmentation_bias_log_softmax";

/// Kernel entries loaded by this host plan
#[cfg(test)]
pub(crate) const REQUIRED_KERNELS: [&str; 3] = [
    SEGMENTATION_POOL_NORM,
    SEGMENTATION_BIAS_LEAKY,
    SEGMENTATION_BIAS_LOG_SOFTMAX,
];

/// Threads per block of the row reductions; multiples of 32
const NORM_THREADS: u32 = 256;
/// Long rows (the waveform normalization: one block per 160000-sample window) need
/// more warps in flight to keep memory busy
const NORM_THREADS_LONG: u32 = 1024;
/// Output length from which a row gets [`NORM_THREADS_LONG`]
const LONG_ROW: usize = 16 * 1024;

/// Where one pooled and normalized row goes, as element strides of the output
#[derive(Debug, Clone, Copy)]
pub(super) struct RowLayout {
    pub batch_stride: usize,
    pub channel_stride: usize,
    pub time_stride: usize,
}

impl RowLayout {
    /// `[batch, channels, len]`, the ONNX NCW layout
    pub(super) fn channels_first(channels: usize, len: usize) -> Self {
        Self {
            batch_stride: channels * len,
            channel_stride: len,
            time_stride: 1,
        }
    }

    /// `[batch, len, channels]`, the batch-major LSTM input
    pub(super) fn time_major_rows(channels: usize, len: usize) -> Self {
        Self {
            batch_stride: channels * len,
            channel_stride: 1,
            time_stride: channels,
        }
    }
}

/// One `segmentation_pool_norm` launch
#[derive(Debug, Clone, Copy)]
pub(super) struct PoolNorm {
    pub batch: usize,
    pub channels: usize,
    pub in_len: usize,
    pub pool: usize,
    /// Take `abs` of the input before pooling (after the SincNet convolution)
    pub abs_input: bool,
    /// LeakyReLU slope; 1 makes the activation the identity
    pub slope: f32,
    pub epsilon: f32,
    pub layout: RowLayout,
}

impl PoolNorm {
    pub(super) fn out_len(&self) -> usize {
        self.in_len / self.pool
    }
}

/// Launchers for the segmentation kernels in `segmentation.<tier>.ptx`
#[derive(Debug, Clone)]
pub(super) struct SegmentationKernels {
    tier: PtxTier,
    pool_norm: CudaFunction,
    bias_leaky: CudaFunction,
    bias_log_softmax: CudaFunction,
}

impl SegmentationKernels {
    pub(super) fn load(runtime: &CudaRuntime) -> Result<Self, CudaError> {
        let kernels = runtime.load_kernels(KernelModule::Segmentation)?;
        Ok(Self {
            tier: kernels.tier(),
            pool_norm: kernels.function(SEGMENTATION_POOL_NORM)?,
            bias_leaky: kernels.function(SEGMENTATION_BIAS_LEAKY)?,
            bias_log_softmax: kernels.function(SEGMENTATION_BIAS_LOG_SOFTMAX)?,
        })
    }

    pub(super) fn tier(&self) -> PtxTier {
        self.tier
    }

    /// Pooling, instance normalization and LeakyReLU over every `(batch, channel)` row
    ///
    /// `conv_bias` is added before pooling; `gamma`, `beta` and `conv_bias` hold one
    /// value per channel
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pool_norm(
        &self,
        runtime: &CudaRuntime,
        spec: PoolNorm,
        input: &CudaSlice<f32>,
        conv_bias: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let context = "segmentation pool norm";
        let rows = spec.batch * spec.channels;
        let out_len = spec.out_len();
        check_len(context, rows * spec.in_len, input.len())?;
        check_len(context, spec.channels, gamma.len())?;
        check_len(context, spec.channels, beta.len())?;
        check_len(context, rows * out_len, out.len())?;
        if conv_bias.len() < spec.channels {
            return Err(CudaError::BufferLength {
                context,
                expected: spec.channels,
                actual: conv_bias.len(),
            });
        }

        let layout = spec.layout;
        let blocks = to_u32(context, rows)?;
        let channels = to_u32(context, spec.channels)?;
        let in_len = to_u32(context, spec.in_len)?;
        let out_len = to_u32(context, out_len)?;
        let pool = to_u32(context, spec.pool)?;
        let abs_input = u32::from(spec.abs_input);
        let batch_stride = to_u32(context, layout.batch_stride)?;
        let channel_stride = to_u32(context, layout.channel_stride)?;
        let time_stride = to_u32(context, layout.time_stride)?;
        let lens = [
            input.len(),
            conv_bias.len(),
            gamma.len(),
            beta.len(),
            out.len(),
        ]
        .map(|len| len as u64);

        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        let _fixed = super::super::test_support::fixed("segmentation_pool_norm");
        let mut launch = runtime.stream().launch_builder(&self.pool_norm);
        launch
            .arg(input)
            .arg(&lens[0])
            .arg(conv_bias)
            .arg(&lens[1])
            .arg(gamma)
            .arg(&lens[2])
            .arg(beta)
            .arg(&lens[3])
            .arg(out)
            .arg(&lens[4])
            .arg(&channels)
            .arg(&in_len)
            .arg(&out_len)
            .arg(&pool)
            .arg(&abs_input)
            .arg(&spec.slope)
            .arg(&spec.epsilon)
            .arg(&batch_stride)
            .arg(&channel_stride)
            .arg(&time_stride);

        let threads = if spec.out_len() >= LONG_ROW {
            NORM_THREADS_LONG
        } else {
            NORM_THREADS
        };
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments follow the PTX signature of `segmentation_pool_norm`
        // (pointer and length per slice, then ten scalars), the lengths are checked
        // above, and the grid has one block per row as the kernel expects
        unsafe { launch.launch(config) }?;
        Ok(())
    }

    /// In place `x = leaky_relu(x + bias)` over a row-major matrix with
    /// `bias.len()` columns
    pub(super) fn bias_leaky(
        &self,
        runtime: &CudaRuntime,
        bias: &CudaSlice<f32>,
        slope: f32,
        x: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let context = "segmentation bias leaky";
        if bias.is_empty() || !x.len().is_multiple_of(bias.len()) {
            return Err(CudaError::BufferLength {
                context,
                expected: bias.len(),
                actual: x.len(),
            });
        }

        let threads = to_u32(context, x.len())?;
        let bias_len = bias.len() as u64;
        let x_len = x.len() as u64;
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        let _fixed = super::super::test_support::fixed("segmentation_bias_leaky");
        let mut launch = runtime.stream().launch_builder(&self.bias_leaky);
        launch
            .arg(bias)
            .arg(&bias_len)
            .arg(&slope)
            .arg(x)
            .arg(&x_len);

        // SAFETY: the arguments follow the PTX signature of `segmentation_bias_leaky`,
        // and the 1-D grid has one thread per element of `x`
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;
        Ok(())
    }

    /// In place `rows = log_softmax(rows + bias)` over `[rows, 7]` logits
    pub(super) fn bias_log_softmax(
        &self,
        runtime: &CudaRuntime,
        bias: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let context = "segmentation log softmax";
        check_len(context, CLASSES, bias.len())?;
        if !logits.len().is_multiple_of(CLASSES) {
            return Err(CudaError::BufferLength {
                context,
                expected: CLASSES,
                actual: logits.len(),
            });
        }

        let rows = logits.len() / CLASSES;
        let threads = to_u32(context, rows)?;
        let bias_len = bias.len() as u64;
        // the kernel takes `[f32; 7]` elements, so its length counts rows
        let rows_len = rows as u64;
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        let _fixed = super::super::test_support::fixed("segmentation_bias_log_softmax");
        let mut launch = runtime.stream().launch_builder(&self.bias_log_softmax);
        launch.arg(bias).arg(&bias_len).arg(logits).arg(&rows_len);

        // SAFETY: the arguments follow the PTX signature of
        // `segmentation_bias_log_softmax`, `logits` holds `rows * 7` values, and the
        // 1-D grid has one thread per row
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;
        Ok(())
    }
}

fn to_u32(context: &'static str, value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow { context, value })
}
