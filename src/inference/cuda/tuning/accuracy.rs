//! Reviewed algorithm accuracy, independent of device identity and speed

use crate::inference::cuda::CudaMath;
use crate::inference::cuda::candidate::{
    ConfigPin, ConvKernel, ConvPin, FbankPin, LstmPin, LstmProjection, SegdenseEntry, SincPin,
    WideconvAlgorithm, WideconvPin, WideconvProducts, WideconvTensorKernel,
};
use crate::inference::cuda::implementation::BoundaryId;

/// End-to-end evidence for an algorithm in the requested pipeline math mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Approval {
    DirectFp32,
    DirectTf32,
    ThreeProductSegmentation,
    StagedWinograd,
    C64FfmaWinograd,
    Fp16Trunk,
}

impl Approval {
    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::DirectFp32 => "reviewed direct FP32 end-to-end accuracy",
            Self::ThreeProductSegmentation => {
                "reviewed A100 three-product segmentation end-to-end accuracy"
            }
            Self::DirectTf32 => "reviewed direct TF32 end-to-end accuracy",
            Self::StagedWinograd => "reviewed wtp1 end-to-end accuracy",
            Self::Fp16Trunk => {
                "FP16 trunk: T4 e1 DER gates pass; A100 a100-e4/pcie-e1 identical RTTMs; 4060 Ti identical RTTMs"
            }
            Self::C64FfmaWinograd => "reviewed C64 FFMA Winograd end-to-end accuracy",
        }
    }
}

/// Runtime accuracy rules depend on whether a selection can fall back to Library
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimePolicy {
    /// Keep the reviewed pipeline math limits when Library can serve a refusal
    Strict,
    /// Without Library, an independently approved exact FP32 kernel may serve TF32
    ExactFp32,
}

impl RuntimePolicy {
    /// Whether runtime may substitute an exact FP32 kernel for a TF32 request
    pub(crate) const fn allows_exact_fp32(self) -> bool {
        matches!(self, Self::ExactFp32)
    }

    /// Approve a runtime choice without changing the tuner's reviewed algorithm set
    pub(crate) fn approve(
        self,
        boundary: BoundaryId,
        math: CudaMath,
        pin: ConfigPin,
    ) -> Option<Approval> {
        Policy::approve(boundary, math, pin).or_else(|| {
            if !self.allows_exact_fp32() || math != CudaMath::Tf32 {
                return None;
            }

            Policy::approve(boundary, CudaMath::Fp32, pin)
                .filter(|approval| *approval == Approval::DirectFp32)
        })
    }
}

/// The tuner cannot promote implementation coverage or a recipe into approval
pub(crate) struct Policy;

impl Policy {
    // bump when the reviewed algorithm set or its math-mode limits change
    pub(crate) const IDENTITY: &'static str = "end-to-end-algorithms-v6";

    /// PR 3 and the C128/T4 branch reports establish unchanged per-file DER or
    /// byte-identical RTTMs for these algorithms; device support is checked separately
    pub(crate) fn approve(
        boundary: BoundaryId,
        math: CudaMath,
        pin: ConfigPin,
    ) -> Option<Approval> {
        let approval = match pin {
            ConfigPin::Conv(ConvPin::Kernel(kernel)) => match kernel {
                ConvKernel::C32
                | ConvKernel::C64
                | ConvKernel::C64Small
                | ConvKernel::C32Stride2
                | ConvKernel::C32Stride2Small => Approval::DirectFp32,
                ConvKernel::C32Tensor
                | ConvKernel::C64Tensor
                | ConvKernel::C64TensorSlim
                | ConvKernel::C32Stride2Tensor => Approval::DirectTf32,
            },
            ConfigPin::Wideconv(WideconvPin::Configured(config)) => match config.algorithm {
                // layerwins evidence/e1: T4 stride-2, subset/hard/test30/dev216
                // DER gates pass; dev216 7.0118 unchanged, jynhe +0.0026
                // layerwins a100-e4 and pcie-e1: subset/hard/test30 RTTMs
                // are identical and all DER gates pass for the FP16 wide trunk
                WideconvAlgorithm::Fp16(_) => Approval::Fp16Trunk,
                WideconvAlgorithm::Spatial | WideconvAlgorithm::WideStem => Approval::DirectFp32,
                WideconvAlgorithm::TensorCore(
                    WideconvTensorKernel::Tf32 | WideconvTensorKernel::Tf32Slim,
                ) => Approval::DirectTf32,
                WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1Staged) => {
                    Approval::StagedWinograd
                }
                WideconvAlgorithm::Winograd(WideconvProducts::Fp32)
                    if [
                        BoundaryId::named("resnet.layer2.0.conv2"),
                        BoundaryId::named("resnet.layer2.1.conv1"),
                        BoundaryId::named("resnet.layer2.1.conv2"),
                        BoundaryId::named("resnet.layer2.2.conv1"),
                        BoundaryId::named("resnet.layer2.2.conv2"),
                        BoundaryId::named("resnet.layer2.3.conv1"),
                        BoundaryId::named("resnet.layer2.3.conv2"),
                    ]
                    .contains(&boundary) =>
                {
                    Approval::C64FfmaWinograd
                }
                _ => return None,
            },
            ConfigPin::Lstm(LstmPin::Projected(LstmProjection::Small | LstmProjection::Large))
            | ConfigPin::Fbank(FbankPin::FftMelAccurate) => Approval::DirectFp32,
            // segmentation end-to-end evidence uses FP32, even on TF32-capable devices
            ConfigPin::Sinc(SincPin::ConvAbsPool) if math == CudaMath::Fp32 => Approval::DirectFp32,
            ConfigPin::Segdense(pin) => match pin.entry() {
                // unmeasured cc 8.0/9.0 FP32 defaults use these three-product kernels
                // the A100 check established identical RTTMs and per-file DER
                SegdenseEntry::Conv1B32X3 | SegdenseEntry::Conv2B32X3 => {
                    Approval::ThreeProductSegmentation
                }
                SegdenseEntry::Conv1B1
                | SegdenseEntry::Conv1B32
                | SegdenseEntry::Conv2B1
                | SegdenseEntry::Conv2B32
                | SegdenseEntry::Linear0B1
                | SegdenseEntry::Linear0B32
                | SegdenseEntry::Linear1B1
                | SegdenseEntry::Linear1B32
                | SegdenseEntry::ClassifierB1
                | SegdenseEntry::ClassifierB32
                | SegdenseEntry::EmbedB1
                | SegdenseEntry::EmbedB32 => Approval::DirectFp32,
                SegdenseEntry::EmbedB32Tf32
                | SegdenseEntry::EmbedB32Tf32K2
                | SegdenseEntry::EmbedB32Tf32E64 => Approval::DirectTf32,
                _ => return None,
            },
            _ => return None,
        };
        match (math, approval) {
            (_, Approval::DirectFp32) | (CudaMath::Fp32, Approval::ThreeProductSegmentation) => {
                Some(approval)
            }
            (CudaMath::Tf32, Approval::ThreeProductSegmentation) => None,
            (CudaMath::Tf32, _) => Some(approval),
            _ => None,
        }
    }
}
