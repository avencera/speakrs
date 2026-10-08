//! The cuda-oxide Sinc producer: band-pass convolution, `abs` and max pool in one
//! kernel, writing only the pooled `[batch, 80, pooled]` tensor; a small kernel packs
//! the filters for it once per plan
//!
//! The locked dispatch then runs the shared normalization pass over the pooled values
//! with pool 1 and no `abs`, which keeps the Library pass's thread assignment and its
//! two-pass mean and variance order

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg,
};

use super::{
    Batches, Coverage, CoverageEntry, Maths, Phases, PlanError, SincCandidate, SincInputs,
    SincOutput, SincSpec,
};
use crate::inference::cuda::error::check_len;
use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, KernelModule};

const SPK_SINCNET_PACK_FILTERS: &str = "spk_sincnet_pack_filters";
const SPK_SINCNET_CONV_ABS_POOL: &str = "spk_sincnet_conv_abs_pool";

/// Kernel entries loaded by this host plan
pub(crate) const REQUIRED_KERNELS: [&str; 2] =
    [SPK_SINCNET_PACK_FILTERS, SPK_SINCNET_CONV_ABS_POOL];

/// The boundary name
const LAYER: &str = "sincnet.conv0.abs_pool";
/// The error context of every check
const CONTEXT: &str = "sinc producer";
/// SincNet filters
const CHANNELS: usize = 80;
/// Taps of every filter
const TAPS: usize = 251;
/// Convolution stride in samples
const STRIDE: usize = 10;
/// Max pool kernel and stride
const POOL: usize = 3;
/// Channels of one block; `GROUP` in the kernel crate's `sincnet` module
const GROUP: usize = 16;
/// Pooled outputs of one block; `TILE` in the kernel crate
const TILE: usize = 256;
/// Threads of one block; `THREADS` in the kernel crate
const THREADS: u32 = 256;

/// Threads of one filter-packing block
const PACK_THREADS: u32 = 256;

/// The producer kernel, the filters packed for it and the launch shape of one batch
/// size
#[derive(Debug)]
pub(crate) struct Oxide {
    produce: CudaFunction,
    /// `[80 / GROUP, 251, GROUP]`, packed from the plan's filters
    packed: CudaSlice<f32>,
    batch: usize,
    samples: usize,
    pooled: usize,
}

impl SincCandidate for Oxide {
    // only FP32 triples were accepted by qualification
    const COVERAGE: Coverage = Coverage(&[CoverageEntry {
        layers: &[LAYER],
        batches: Batches::All,
        maths: Maths::Only(&[CudaMath::Fp32]),
    }]);
    const OUTPUT: SincOutput = SincOutput::Pooled;

    fn plan(runtime: &CudaRuntime, spec: SincSpec<'_>) -> Result<Self, PlanError> {
        check_len(CONTEXT, CHANNELS * TAPS, spec.filters.len())?;
        // every valid pooled output reads only samples of its own row when the pooled
        // length is the pooled length of a valid convolution
        check_len(
            CONTEXT,
            (spec.samples.saturating_sub(TAPS) / STRIDE + 1) / POOL,
            spec.pooled,
        )?;

        // packing once per plan measured slightly faster than packing on every call;
        // every plan packs the filters it is given
        let kernels = runtime.load_kernels(KernelModule::Sincnet)?;
        let mut packed = runtime.stream().alloc_zeros(CHANNELS * TAPS)?;
        pack(
            runtime.stream(),
            &kernels.function(SPK_SINCNET_PACK_FILTERS)?,
            &spec.filters.as_view(),
            &mut packed,
        )?;
        // the producer may run on another stream than the packing
        runtime.synchronize()?;

        Ok(Self {
            produce: kernels.function(SPK_SINCNET_CONV_ABS_POOL)?,
            packed,
            batch: spec.batch,
            samples: spec.samples,
            pooled: spec.pooled,
        })
    }

    fn enqueue(
        &self,
        inputs: SincInputs<'_, '_>,
        output: &mut CudaViewMut<'_, f32>,
        _phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        // the producer reads the filters packed in `plan`, which are the model's
        // generated filters
        check_len(CONTEXT, self.batch * self.samples, inputs.waveform.len())?;
        check_len(CONTEXT, self.batch * CHANNELS * self.pooled, output.len())?;
        self.produce(stream, inputs.waveform, output)
    }
}

/// Queues `spk_sincnet_pack_filters` from `filters` `[80, 251]` into `packed`
fn pack(
    stream: &CudaStream,
    function: &CudaFunction,
    filters: &CudaView<'_, f32>,
    packed: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let elements = to_u32(packed.len())?;
    let filters_len = filters.len() as u64;
    let packed_len = packed.len() as u64;

    let mut launch = stream.launch_builder(function);
    launch
        .arg(filters)
        .arg(&filters_len)
        .arg(packed)
        .arg(&packed_len);

    let config = LaunchConfig {
        grid_dim: (elements.div_ceil(PACK_THREADS), 1, 1),
        block_dim: (PACK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: the arguments follow the PTX signature of `spk_sincnet_pack_filters`
    // (pointer and length per slice), both lengths are `80 * 251`, checked by `plan`
    // and its allocation, and the grid has at least one thread per packed element
    unsafe { launch.launch(config) }?;
    Ok(())
}

impl Oxide {
    /// Queues `spk_sincnet_conv_abs_pool` over `wave` `[batch, samples]` with the
    /// packed filters into `output` `[batch, 80, pooled]`
    fn produce(
        &self,
        stream: &CudaStream,
        wave: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), CudaError> {
        let tiles = to_u32(self.pooled.div_ceil(TILE))?;
        let groups = to_u32(CHANNELS / GROUP)?;
        let batch = to_u32(self.batch)?;
        let samples = to_u32(self.samples)?;
        let pool_len = to_u32(self.pooled)?;
        let wave_len = wave.len() as u64;
        // the kernel takes the filters as float quads
        let quads = (self.packed.len() / 4) as u64;
        let output_len = output.len() as u64;

        let mut launch = stream.launch_builder(&self.produce);
        launch
            .arg(wave)
            .arg(&wave_len)
            .arg(&self.packed)
            .arg(&quads)
            .arg(output)
            .arg(&output_len)
            .arg(&samples)
            .arg(&pool_len);

        let config = LaunchConfig {
            grid_dim: (tiles, groups, batch),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the arguments follow the PTX signature of
        // `spk_sincnet_conv_abs_pool` (pointer and length per slice, then two
        // scalars), the caller and `plan` check the lengths and the pooled length, the
        // packed filter buffer starts 256-byte aligned for the quad view, and the grid
        // matches the kernel's tile, group and row mapping
        unsafe { launch.launch(config) }?;
        Ok(())
    }
}

fn to_u32(value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow {
        context: CONTEXT,
        value,
    })
}
