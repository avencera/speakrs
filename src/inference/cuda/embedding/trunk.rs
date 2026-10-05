use super::super::dnn::Conv2d;
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
use super::super::implementation::Choice;
use super::super::{CudaError, CudaMath, CudaRuntime, DeviceTensor, SafetensorsFile};

/// Channels of the stem convolution
const STEM_CHANNELS: usize = 32;

/// Output channels and basic-block count of the four residual stages
const STAGES: [(usize, usize); 4] = [(32, 3), (64, 4), (128, 6), (256, 3)];

/// Trunk buffer the stem writes, which the first block reads
pub(super) const STEM_SLOT: usize = 0;

/// Per-item element counts of an embedding batch's activation buffers
#[derive(Debug, Clone, Copy)]
pub(super) struct BufferLens {
    pub(super) trunk: [usize; 2],
    pub(super) hidden: usize,
    pub(super) shortcut: usize,
}

/// Geometry of one square trunk convolution for a single item
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ConvShape {
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    /// input height and width
    input: [usize; 2],
}

impl ConvShape {
    /// The cuDNN convolution for `batch` items in `math` precision
    pub(super) fn conv(self, batch: usize, math: CudaMath) -> Conv2d {
        // 3x3 convolutions pad by one, the 1x1 shortcuts not at all
        let padding = self.kernel / 2;
        Conv2d {
            batch,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            input: self.input,
            kernel: [self.kernel; 2],
            padding: [padding; 2],
            stride: [self.stride; 2],
            dilation: [1, 1],
            math,
        }
    }
}

/// One trunk convolution with its bias; batch norm is already folded into both
#[derive(Debug)]
pub(super) struct ConvLayer {
    weight: DeviceTensor,
    bias: DeviceTensor,
    shape: ConvShape,
    /// layer identity and qualification-only override
    plan: LayerPlan,
}

/// Per-layer ownership is separate from the shared cuDNN shape plans
#[derive(Debug)]
struct LayerPlan {
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    override_choice: Option<Choice>,
    name: String,
}

impl ConvLayer {
    /// Uploads `<prefix>.weight` and `<prefix>.weight_bias`, the names the ONNX export
    /// gives a convolution whose batch norm it folded
    fn load(
        runtime: &CudaRuntime,
        weights: &SafetensorsFile,
        prefix: &str,
        shape: ConvShape,
    ) -> Result<Self, CudaError> {
        let ConvShape {
            in_channels,
            out_channels,
            kernel,
            ..
        } = shape;
        let weight = weights.upload(
            runtime,
            &format!("{prefix}.weight"),
            &[out_channels, in_channels, kernel, kernel],
        )?;
        let bias = weights.upload(runtime, &format!("{prefix}.weight_bias"), &[out_channels])?;

        Ok(Self {
            weight,
            bias,
            shape,
            plan: LayerPlan {
                #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
                override_choice: None,
                name: prefix.to_owned(),
            },
        })
    }

    /// The cuDNN convolution for `batch` items in `math` precision
    pub(super) fn conv(&self, batch: usize, math: CudaMath) -> Conv2d {
        self.shape.conv(batch, math)
    }

    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    pub(super) fn override_choice(&self) -> Option<Choice> {
        self.plan.override_choice
    }

    pub(super) fn name(&self) -> &str {
        &self.plan.name
    }

    pub(super) fn weight(&self) -> &DeviceTensor {
        &self.weight
    }

    pub(super) fn bias(&self) -> &DeviceTensor {
        &self.bias
    }

    pub(super) fn out_channels(&self) -> usize {
        self.shape.out_channels
    }

    /// Output height and width
    pub(super) fn output(&self) -> [usize; 2] {
        self.conv(1, CudaMath::Fp32).output()
    }

    /// Elements of one item's output
    pub(super) fn output_len(&self) -> usize {
        let [h, w] = self.output();
        self.shape.out_channels * h * w
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod test_support;

/// A ResNet basic block: two 3x3 convolutions and a residual connection, with a
/// strided 1x1 shortcut when the block changes resolution
///
/// The forward pass keeps block inputs and outputs in two trunk buffers and
/// alternates between them: a block reads one and writes the other, because cuDNN
/// reads the residual while it writes the output
#[derive(Debug)]
pub(super) struct BasicBlock {
    pub(super) conv1: ConvLayer,
    pub(super) conv2: ConvLayer,
    pub(super) shortcut: Option<ConvLayer>,
    /// trunk buffer holding the block input
    pub(super) input_slot: usize,
}

impl BasicBlock {
    /// Trunk buffer that receives the block output
    pub(super) fn output_slot(&self) -> usize {
        1 - self.input_slot
    }
}

/// The WeSpeaker ResNet34 trunk: a 3x3 stem and four stages of basic blocks
#[derive(Debug)]
pub(super) struct Trunk {
    pub(super) stem: ConvLayer,
    pub(super) blocks: Vec<BasicBlock>,
}

impl Trunk {
    /// Uploads every trunk convolution for an input of `bins` frequency bins by
    /// `frames` time frames
    pub(super) fn load(
        runtime: &CudaRuntime,
        weights: &SafetensorsFile,
        bins: usize,
        frames: usize,
    ) -> Result<Self, CudaError> {
        let stem_shape = ConvShape {
            in_channels: 1,
            out_channels: STEM_CHANNELS,
            kernel: 3,
            stride: 1,
            input: [bins, frames],
        };
        let stem = ConvLayer::load(runtime, weights, "resnet.conv1", stem_shape)?;

        let mut blocks = Vec::with_capacity(STAGES.iter().map(|(_, count)| count).sum());
        let mut channels = STEM_CHANNELS;
        let mut size = stem.output();
        for (stage, &(out_channels, count)) in STAGES.iter().enumerate() {
            for index in 0..count {
                // the first block of every stage after the first halves the resolution
                let stride = if index == 0 && stage > 0 { 2 } else { 1 };
                let prefix = format!("resnet.layer{}.{index}", stage + 1);

                let conv1_shape = ConvShape {
                    in_channels: channels,
                    out_channels,
                    kernel: 3,
                    stride,
                    input: size,
                };
                let conv1 =
                    ConvLayer::load(runtime, weights, &format!("{prefix}.conv1"), conv1_shape)?;
                let block_output = conv1.output();

                let conv2_shape = ConvShape {
                    in_channels: out_channels,
                    out_channels,
                    kernel: 3,
                    stride: 1,
                    input: block_output,
                };
                let conv2 =
                    ConvLayer::load(runtime, weights, &format!("{prefix}.conv2"), conv2_shape)?;

                let shortcut = if stride != 1 || channels != out_channels {
                    let shortcut_shape = ConvShape {
                        in_channels: channels,
                        out_channels,
                        kernel: 1,
                        stride,
                        input: size,
                    };
                    Some(ConvLayer::load(
                        runtime,
                        weights,
                        &format!("{prefix}.shortcut.0"),
                        shortcut_shape,
                    )?)
                } else {
                    None
                };

                let input_slot = blocks.last().map_or(STEM_SLOT, BasicBlock::output_slot);
                blocks.push(BasicBlock {
                    conv1,
                    conv2,
                    shortcut,
                    input_slot,
                });
                channels = out_channels;
                size = block_output;
            }
        }

        Ok(Self { stem, blocks })
    }

    /// Every convolution and whether its epilogue adds a residual
    pub(super) fn layers(&self) -> impl Iterator<Item = (&ConvLayer, bool)> {
        std::iter::once((&self.stem, false)).chain(self.blocks.iter().flat_map(|block| {
            [
                Some((&block.conv1, false)),
                Some((&block.conv2, true)),
                block.shortcut.as_ref().map(|layer| (layer, false)),
            ]
            .into_iter()
            .flatten()
        }))
    }

    /// Elements per item each buffer must hold: the two trunk buffers, the hidden
    /// activation between a block's convolutions, and the shortcut output
    ///
    /// The trunk buffer the stem does not write also serves as the stem's unused
    /// residual operand, so it holds at least the stem output as well
    pub(super) fn buffer_lens(&self) -> BufferLens {
        let stem = self.stem.output_len();
        let mut lens = BufferLens {
            trunk: [stem; 2],
            hidden: 0,
            shortcut: 0,
        };
        for block in &self.blocks {
            let output = block.conv2.output_len();
            let slot = block.output_slot();
            lens.trunk[slot] = lens.trunk[slot].max(output);
            lens.hidden = lens.hidden.max(block.conv1.output_len());
            if let Some(shortcut) = &block.shortcut {
                lens.shortcut = lens.shortcut.max(shortcut.output_len());
            }
        }

        lens
    }

    /// Trunk buffer holding the trunk output
    pub(super) fn output_slot(&self) -> usize {
        self.blocks
            .last()
            .map_or(STEM_SLOT, BasicBlock::output_slot)
    }

    /// Output `[channels, height, width]` of one item
    pub(super) fn output_shape(&self) -> [usize; 3] {
        let last = self.blocks.last().map_or(&self.stem, |block| &block.conv2);
        let [h, w] = last.output();
        [last.out_channels(), h, w]
    }
}
