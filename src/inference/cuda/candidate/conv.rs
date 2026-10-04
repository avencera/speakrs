//! The ResNet convolution candidate: fused cuda-oxide 3x3 convolutions for the 14
//! eligible layers of the first two stages
//!
//! The kernels in `crates/speakrs-cuda-kernels/src/resnet.rs` compute the whole
//! `relu(conv(x) + bias [+ residual])` operator in one launch on the same NCHW buffers
//! as the cuDNN plan, so no layout conversion surrounds them. Three fixed shapes cover
//! the 14 layers: 32 -> 32 and 64 -> 64 channels at stride 1, and 32 -> 64 channels at
//! stride 2. A plan packs its layer's weights `[cin][ky][kx][cout]` once, on the
//! device; packing them on every call measurably slowed the b1 calls. When a batch is too small for the 256-thread grid to fill the GPU, as a
//! single item is, the plan picks the shape's small-block kernel instead
//!
//! Every PTX tier computes the full operator in FP32 FMA whatever the boundary's
//! `CudaMath`. The accumulation order and the residual-then-bias epilogue match
//! cuDNN's implicit GEMM, which gives the same bits wherever cuDNN picks that
//! algorithm and fewer rounding errors where it picks Winograd or TF32

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg,
};

use super::{
    Batches, ConvCandidate, ConvInputs, ConvLayerSpec, Coverage, CoverageEntry, Maths, Op, Phases,
};
use crate::inference::cuda::dnn::Conv2d;
use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, KernelModule};

/// The 32 -> 32 convolutions of `layer1` and the strided 32 -> 64 one of `layer2`
const C32_AND_STRIDED: [&str; 7] = [
    "resnet.layer1.0.conv1",
    "resnet.layer1.0.conv2",
    "resnet.layer1.1.conv1",
    "resnet.layer1.1.conv2",
    "resnet.layer1.2.conv1",
    "resnet.layer1.2.conv2",
    "resnet.layer2.0.conv1",
];

/// The 64 -> 64 convolutions of `layer2`
const C64: [&str; 7] = [
    "resnet.layer2.0.conv2",
    "resnet.layer2.1.conv1",
    "resnet.layer2.1.conv2",
    "resnet.layer2.2.conv1",
    "resnet.layer2.2.conv2",
    "resnet.layer2.3.conv1",
    "resnet.layer2.3.conv2",
];

/// Output columns per block; must equal `CONV_TILE_COLS` in the kernel crate
const TILE_COLS: usize = 64;

/// Threads per block of the weight packing kernel
const PACK_THREADS: u32 = 256;

/// Below this many 256-thread blocks per SM, a shape with small blocks uses them
const SMALL_BATCH_WAVES: usize = 2;

/// The convolution shapes that have a fused kernel: 3x3, padding 1, no dilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// 32 -> 32 channels, stride 1
    C32,
    /// 64 -> 64 channels, stride 1
    C64,
    /// 32 -> 64 channels, stride 2
    C32Stride2,
}

impl Shape {
    /// The fused shape of `conv`, if it has one
    fn of(conv: &Conv2d) -> Option<Self> {
        if conv.kernel != [3, 3] || conv.padding != [1, 1] || conv.dilation != [1, 1] {
            return None;
        }

        match (conv.in_channels, conv.out_channels, conv.stride) {
            (32, 32, [1, 1]) => Some(Self::C32),
            (64, 64, [1, 1]) => Some(Self::C64),
            (32, 64, [2, 2]) => Some(Self::C32Stride2),
            _ => None,
        }
    }

    fn stride(self) -> usize {
        match self {
            Self::C32 | Self::C64 => 1,
            Self::C32Stride2 => 2,
        }
    }

    /// The 256-thread kernel, and the small-block one where it exists
    fn tilings(self) -> (Tiling, Option<Tiling>) {
        match self {
            Self::C32 => (Tiling::new("spk_resnet_conv3x3_c32", 256, 8), None),
            Self::C64 => (
                Tiling::new("spk_resnet_conv3x3_c64", 256, 4),
                Some(Tiling::new("spk_resnet_conv3x3_c64_small", 128, 1)),
            ),
            Self::C32Stride2 => (
                Tiling::new("spk_resnet_conv3x3_c32s2", 256, 4),
                Some(Tiling::new("spk_resnet_conv3x3_c32s2_small", 128, 2)),
            ),
        }
    }
}

/// One kernel entry with its fixed block: `threads` per block covering 64 output
/// columns of `rows` output rows; both must match the kernel crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tiling {
    entry: &'static str,
    threads: u32,
    rows: usize,
}

impl Tiling {
    const fn new(entry: &'static str, threads: u32, rows: usize) -> Self {
        Self {
            entry,
            threads,
            rows,
        }
    }

    /// Blocks covering the output `[h, w]` of each of `batch` items
    fn grid(self, batch: usize, output: [usize; 2]) -> Result<(u32, u32, u32), CudaError> {
        let [h, w] = output;
        Ok((
            to_u32(w.div_ceil(TILE_COLS))?,
            to_u32(h.div_ceil(self.rows))?,
            to_u32(batch)?,
        ))
    }

    fn blocks(self, batch: usize, output: [usize; 2]) -> usize {
        let [h, w] = output;
        w.div_ceil(TILE_COLS) * h.div_ceil(self.rows) * batch
    }
}

/// One layer's fused convolution at one batch size: its kernel, its fixed sizes and
/// its packed weights
#[derive(Debug)]
pub(crate) struct Oxide {
    function: CudaFunction,
    tiling: Tiling,
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    /// input height and width of one item
    input: [usize; 2],
    /// output height and width of one item
    output: [usize; 2],
    /// `[cin][ky][kx][cout]`, written once in `plan`
    packed: CudaSlice<f32>,
}

impl Oxide {
    fn input_len(&self) -> usize {
        self.batch * self.in_channels * self.input[0] * self.input[1]
    }

    fn output_len(&self) -> usize {
        self.batch * self.out_channels * self.output[0] * self.output[1]
    }
}

impl ConvCandidate for Oxide {
    // cuDNN runs the 64-channel layers on TF32 tensor cores at b1 in TF32 mode, where
    // this FP32 kernel is only 1-3% faster, inside the timing noise bound
    const COVERAGE: Coverage = Coverage(&[
        CoverageEntry {
            layers: &C32_AND_STRIDED,
            batches: Batches::All,
            maths: Maths::All,
        },
        CoverageEntry {
            layers: &C64,
            batches: Batches::Only(&[7, 32, 33, 64]),
            maths: Maths::All,
        },
        CoverageEntry {
            layers: &C64,
            batches: Batches::Only(&[1]),
            maths: Maths::Only(&[CudaMath::Fp32]),
        },
    ]);

    fn plan(runtime: &CudaRuntime, layer: ConvLayerSpec<'_>) -> Result<Self, CudaError> {
        let conv = layer.conv;
        let shape = Shape::of(&conv).ok_or_else(|| CudaError::Unsupported {
            context: "fused conv3x3 plan",
            reason: format!(
                "{} has no fused kernel for {} -> {} channels, kernel {:?}, stride {:?}",
                layer.name, conv.in_channels, conv.out_channels, conv.kernel, conv.stride
            ),
        })?;
        let output = conv
            .input
            .map(|size| size.saturating_sub(1) / shape.stride() + 1);

        // a batch whose 256-thread grid would leave SMs idle or doubly loaded uses
        // the small blocks, which spread the same work evenly
        let (large, small) = shape.tilings();
        let tiling = match small {
            Some(small)
                if large.blocks(conv.batch, output)
                    < SMALL_BATCH_WAVES * runtime.multiprocessor_count()? =>
            {
                small
            }
            _ => large,
        };
        let weight_len = conv.out_channels * conv.in_channels * 9;
        check_len("fused conv3x3 weights", weight_len, layer.weight.len())?;
        let kernels = runtime.load_kernels(KernelModule::Resnet)?;
        let pack = kernels.function("spk_resnet_pack_weights")?;
        let mut packed = runtime.stream().alloc_zeros::<f32>(weight_len)?;
        pack_weights(
            runtime.stream(),
            &pack,
            [conv.in_channels, conv.out_channels],
            layer.weight,
            &mut packed,
        )?;

        let plan = Self {
            function: kernels.function(tiling.entry)?,
            tiling,
            batch: conv.batch,
            in_channels: conv.in_channels,
            out_channels: conv.out_channels,
            input: conv.input,
            output,
            packed,
        };
        // every index in the kernels is 32-bit
        to_u32(plan.input_len().max(plan.output_len()))?;
        Ok(plan)
    }

    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError> {
        let output_len = self.output_len();
        check_len("fused conv3x3 input", self.input_len(), inputs.x.len())?;
        check_len("fused conv3x3 bias", self.out_channels, inputs.bias.len())?;
        check_len("fused conv3x3 output", output_len, y.len())?;
        let (residual, add_residual) = match inputs.residual {
            Some(value) => {
                check_len("fused conv3x3 residual", output_len, value.len())?;
                (value, 1u32)
            }
            // never read, so any valid view stands in for the residual operand
            None => (inputs.x, 0u32),
        };

        let config = LaunchConfig {
            grid_dim: self.tiling.grid(self.batch, self.output)?,
            block_dim: (self.tiling.threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let h_in = to_u32(self.input[0])?;
        let w_in = to_u32(self.input[1])?;
        let lengths = [
            inputs.x.len() as u64,
            self.packed.len() as u64,
            inputs.bias.len() as u64,
            residual.len() as u64,
            y.len() as u64,
        ];
        let mut launch = stream.launch_builder(&self.function);
        launch
            .arg(inputs.x)
            .arg(&lengths[0])
            .arg(&self.packed)
            .arg(&lengths[1])
            .arg(inputs.bias)
            .arg(&lengths[2])
            .arg(residual)
            .arg(&lengths[3])
            .arg(&add_residual)
            .arg(&h_in)
            .arg(&w_in)
            .arg(y)
            .arg(&lengths[4]);

        phases.op(Op::Main, || {
            // SAFETY: the arguments match `spk_resnet_conv3x3_*(x: &[f32], weight: &[f32],
            // bias: &[f32], residual: &[f32], add_residual: u32, h_in: u32, w_in: u32,
            // y: DisjointSlice<f32>)`; the checks above size every buffer for this plan's
            // convolution, whose shape selected the kernel, and the grid covers the output
            // with that kernel's fixed tile and block size
            unsafe { launch.launch(config) }?;
            Ok(())
        })
    }
}

/// Packs `weight` `[cout][cin][3][3]` into `packed` `[cin][3][3][cout]`, both
/// holding `cout * cin * 9` elements
fn pack_weights(
    stream: &CudaStream,
    pack: &CudaFunction,
    [cin, cout]: [usize; 2],
    weight: &CudaSlice<f32>,
    packed: &mut CudaSlice<f32>,
) -> Result<(), CudaError> {
    let len = (cout * cin * 9) as u64;
    let threads = to_u32(cout * cin * 9)?;
    let (cin, cout) = (to_u32(cin)?, to_u32(cout)?);
    let mut launch = stream.launch_builder(pack);
    launch
        .arg(weight)
        .arg(&len)
        .arg(&cin)
        .arg(&cout)
        .arg(packed)
        .arg(&len);
    // SAFETY: the arguments match `spk_resnet_pack_weights(weight: &[f32], cin: u32,
    // cout: u32, packed: DisjointSlice<f32>)`, the caller sized both buffers to
    // `cout * cin * 9` elements, and there is one thread per packed element
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (threads.div_ceil(PACK_THREADS), 1, 1),
            block_dim: (PACK_THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

fn check_len(context: &'static str, expected: usize, actual: usize) -> Result<(), CudaError> {
    if expected == actual {
        return Ok(());
    }

    Err(CudaError::BufferLength {
        context,
        expected,
        actual,
    })
}

fn to_u32(value: usize) -> Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::DimensionOverflow {
        context: "fused conv3x3 launch",
        value,
    })
}
