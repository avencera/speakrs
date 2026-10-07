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
    Batches, ConvCandidate, ConvInputs, ConvKernel, ConvLayerSpec, ConvPin, ConvShape, Coverage,
    CoverageEntry, FiniteContract, GeometryError, InfinityContract, Maths, NanContract, Op, Phases,
    PlanError, SignedZeroContract, SpecialValues,
};
use crate::inference::cuda::geometry::Conv2d;
use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, LoadedKernels};

const SPK_RESNET_PACK_WEIGHTS: &str = "spk_resnet_pack_weights";

/// Kernel entries loaded by this host plan
pub(crate) const REQUIRED_KERNELS: [&str; 1] = [SPK_RESNET_PACK_WEIGHTS];

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
pub(super) const SMALL_BATCH_WAVES: usize = 2;

impl ConvShape {
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
    pub(super) fn tilings(self) -> (Tiling, Option<Tiling>) {
        match self {
            Self::C32 => (ConvKernel::C32.tiling(), None),
            Self::C64 => (
                ConvKernel::C64.tiling(),
                Some(ConvKernel::C64Small.tiling()),
            ),
            Self::C32Stride2 => (
                ConvKernel::C32Stride2.tiling(),
                Some(ConvKernel::C32Stride2Small.tiling()),
            ),
        }
    }
}

impl ConvKernel {
    /// The entry's fixed block and tile
    fn tiling(self) -> Tiling {
        match self {
            Self::C32 => Tiling::new("spk_resnet_conv3x3_c32", 256, 8),
            Self::C64 => Tiling::new("spk_resnet_conv3x3_c64", 256, 4),
            Self::C64Small => Tiling::new("spk_resnet_conv3x3_c64_small", 128, 1),
            Self::C32Stride2 => Tiling::new("spk_resnet_conv3x3_c32s2", 256, 4),
            Self::C32Stride2Small => Tiling::new("spk_resnet_conv3x3_c32s2_small", 128, 2),
        }
    }
}

/// One kernel entry with its fixed block: `threads` per block covering 64 output
/// columns of `rows` output rows; both must match the kernel crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Tiling {
    pub(super) entry: &'static str,
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

    pub(super) fn blocks(self, batch: usize, output: [usize; 2]) -> usize {
        let [h, w] = output;
        w.div_ceil(TILE_COLS) * h.div_ceil(self.rows) * batch
    }
}

pub(super) fn select_tiling(
    large: Tiling,
    small: Option<Tiling>,
    batch: usize,
    output: [usize; 2],
    multiprocessors: usize,
) -> Tiling {
    match small {
        Some(small) if large.blocks(batch, output) < SMALL_BATCH_WAVES * multiprocessors => small,
        _ => large,
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
    epilogue: super::Epilogue,
    math: CudaMath,
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
    type Pin = ConvPin;
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

    // the ReLU is `if value < 0.0 { 0.0 } else { value }` after FP32 FMA sums, so NaN
    // and negative zero pass through it
    const SPECIAL_VALUES: SpecialValues = SpecialValues {
        finite: FiniteContract::AbsoluteSum { headroom: 2 },
        nan: NanContract::Propagates,
        infinity: InfinityContract::Ieee,
        signed_zero: SignedZeroContract::ReluKeepsNegative,
    };

    fn implemented_pin(layer: &ConvLayerSpec<'_>) -> Result<ConvPin, PlanError> {
        if layer.epilogue == super::Epilogue::Bias {
            return Err(PlanError::Geometry(GeometryError::Unimplemented {
                context: "fused conv3x3 plan",
                reason: "the fused kernel requires ReLU".into(),
            }));
        }

        ConvShape::of(&layer.conv)
            .map(ConvPin::LegacyWaves)
            .ok_or_else(|| {
                PlanError::Geometry(GeometryError::Unimplemented {
                    context: "fused conv3x3 plan",
                    reason: unfused(layer),
                })
            })
    }

    fn plan(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        layer: ConvLayerSpec<'_>,
        pin: ConvPin,
    ) -> Result<Self, PlanError> {
        if layer.epilogue == super::Epilogue::Bias {
            return Err(PlanError::Geometry(GeometryError::Unimplemented {
                context: "fused conv3x3 plan",
                reason: "the fused kernel requires ReLU".into(),
            }));
        }

        let conv = layer.conv;
        let shape = ConvShape::of(&conv).ok_or_else(|| {
            PlanError::Geometry(GeometryError::Invalid {
                context: "fused conv3x3 plan",
                reason: unfused(&layer),
            })
        })?;
        if shape != pin.shape() {
            return Err(PlanError::Geometry(GeometryError::Invalid {
                context: "fused conv3x3 plan",
                reason: format!("{} is {shape:?}, but its pin is {pin:?}", layer.name),
            }));
        }
        let output = conv
            .input
            .map(|size| size.saturating_sub(1) / shape.stride() + 1);

        let tiling = match pin {
            ConvPin::Kernel(kernel) => kernel.tiling(),
            ConvPin::LegacyWaves(_) => {
                // a batch whose 256-thread grid would leave SMs idle or doubly loaded
                // uses the small blocks, which spread the same work evenly
                let (large, small) = shape.tilings();
                let multiprocessors = if small.is_some() {
                    runtime.multiprocessor_count()?
                } else {
                    0
                };
                select_tiling(large, small, conv.batch, output, multiprocessors)
            }
        };
        let weight_len = conv.out_channels * conv.in_channels * 9;
        check_len("fused conv3x3 weights", weight_len, layer.weight.len())?;
        let pack = kernels.function(SPK_RESNET_PACK_WEIGHTS)?;
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
            epilogue: layer.epilogue,
            math: conv.math,
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
        if inputs.residual.is_some() != (self.epilogue == super::Epilogue::BiasReluResidual) {
            return Err(CudaError::CandidateGeometry {
                area: "resnet",
                boundary: "fused conv3x3 residual".into(),
                batch: self.batch,
                math: self.math,
                error: GeometryError::Invalid {
                    context: "fused conv3x3 residual",
                    reason: "residual input does not match the planned epilogue".into(),
                },
            });
        }

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

fn unfused(layer: &ConvLayerSpec<'_>) -> String {
    let conv = layer.conv;
    format!(
        "{} has no fused kernel for {} -> {} channels, kernel {:?}, stride {:?}",
        layer.name, conv.in_channels, conv.out_channels, conv.kernel, conv.stride
    )
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

impl super::DriverCandidate for Oxide {
    const AREA: super::KernelModule = super::KernelModule::Resnet;

    fn driver_coverage(tier: super::PtxTier) -> Coverage {
        <Self as ConvCandidate>::coverage(tier)
    }

    fn driver_pin(
        boundary: super::super::implementation::BoundaryId,
        batch: usize,
        _math: CudaMath,
        device: &super::super::device::DeviceAttributes,
        _tier: super::PtxTier,
    ) -> Result<super::ConfigPin, PlanError> {
        let name = boundary.name();
        let shape = if name == "resnet.layer2.0.conv1" {
            ConvShape::C32Stride2
        } else if C32_AND_STRIDED.contains(&name) {
            ConvShape::C32
        } else if C64.contains(&name) {
            ConvShape::C64
        } else {
            return Err(PlanError::Geometry(GeometryError::Unimplemented {
                context: "driver convolution pin",
                reason: name.to_owned(),
            }));
        };
        // these are the fixed model outputs, not a shape supplied by the caller
        let output = if shape == ConvShape::C32 {
            [80, 998]
        } else {
            [40, 499]
        };
        let (large, small) = shape.tilings();
        let tiling = select_tiling(
            large,
            small,
            batch,
            output,
            device.multiprocessors().get() as usize,
        );
        let kernel = match (shape, tiling.threads) {
            (ConvShape::C32, _) => ConvKernel::C32,
            (ConvShape::C64, 128) => ConvKernel::C64Small,
            (ConvShape::C64, _) => ConvKernel::C64,
            (ConvShape::C32Stride2, 128) => ConvKernel::C32Stride2Small,
            (ConvShape::C32Stride2, _) => ConvKernel::C32Stride2,
        };
        Ok(super::ConfigPin::Conv(ConvPin::Kernel(kernel)))
    }
}
