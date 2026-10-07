//! The PR #36 ResNet convolution binding on capability 12.0

use super::{FP32, LEGACY_SCOPE, TF32, legacy_proof};
use crate::inference::cuda::candidate::{ConfigPin, ConvPin, ConvShape};
use crate::inference::cuda::implementation::{Binding, BoundaryId, RecordHash, TupleProof};
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{CudaMath, KernelModule, PtxTier};

/// SHA256 of qualify-resnet-Oxide-20261004T064208.284837Z.json.gz
pub(crate) const RECORD: RecordHash =
    RecordHash::from_hex("8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758");

const fn proof(name: &str, shape: ConvShape, batch: usize, math: CudaMath) -> TupleProof {
    legacy_proof(
        BoundaryId::named(name),
        batch,
        math,
        ConfigPin::Conv(ConvPin::LegacyWaves(shape)),
        RECORD,
    )
}

/// Pinned evidence, not candidate declarations
const PROOFS: [TupleProof; 48] = [
    // 32 -> 32 channels: both batches and both modes
    proof("resnet.layer1.0.conv1", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.0.conv1", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.0.conv1", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.0.conv1", ConvShape::C32, 32, TF32),
    proof("resnet.layer1.0.conv2", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.0.conv2", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.0.conv2", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.0.conv2", ConvShape::C32, 32, TF32),
    proof("resnet.layer1.1.conv1", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.1.conv1", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.1.conv1", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.1.conv1", ConvShape::C32, 32, TF32),
    proof("resnet.layer1.1.conv2", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.1.conv2", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.1.conv2", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.1.conv2", ConvShape::C32, 32, TF32),
    proof("resnet.layer1.2.conv1", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.2.conv1", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.2.conv1", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.2.conv1", ConvShape::C32, 32, TF32),
    proof("resnet.layer1.2.conv2", ConvShape::C32, 1, FP32),
    proof("resnet.layer1.2.conv2", ConvShape::C32, 1, TF32),
    proof("resnet.layer1.2.conv2", ConvShape::C32, 32, FP32),
    proof("resnet.layer1.2.conv2", ConvShape::C32, 32, TF32),
    // the 36-SM RTX 5060 Ti control recorded a hard "candidate slower than the faster
    // Library process" failure for b1 FP32, then unresolved noise (min pair 0.989,
    // spread 0.025); the legacy record used a 70-SM RTX 5070 Ti, and a recorded hard
    // failure cannot be outweighed by a later pass
    proof("resnet.layer2.0.conv1", ConvShape::C32Stride2, 32, FP32),
    proof("resnet.layer2.0.conv1", ConvShape::C32Stride2, 32, TF32),
    proof("resnet.layer2.0.conv1", ConvShape::C32Stride2, 1, TF32),
    // 64 -> 64 channels: cuDNN's b1 TF32 tensor-core path is within the noise bound
    proof("resnet.layer2.0.conv2", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.0.conv2", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.1.conv1", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.1.conv1", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.1.conv2", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.1.conv2", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.2.conv1", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.2.conv1", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.2.conv2", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.2.conv2", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.3.conv1", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.3.conv1", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.3.conv2", ConvShape::C64, 32, FP32),
    proof("resnet.layer2.3.conv2", ConvShape::C64, 32, TF32),
    proof("resnet.layer2.0.conv2", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.1.conv1", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.1.conv2", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.2.conv1", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.2.conv2", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.3.conv1", ConvShape::C64, 1, FP32),
    proof("resnet.layer2.3.conv2", ConvShape::C64, 1, FP32),
];

pub(crate) const BINDING: Binding = Binding {
    scope: LEGACY_SCOPE,
    module: ModuleRequest::new(
        KernelModule::Resnet,
        PtxTier::Sm75,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "dd6449c0129f9a03bf691c0338611b50b651ab714d5803caedea87de3c72b6b7",
            ),
        },
    ),
    proofs: &PROOFS,
};
