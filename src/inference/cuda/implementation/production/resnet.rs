//! The measured ResNet artifact for the 36-SM RTX 5060 Ti

use super::super::ModuleBinding;
use crate::inference::cuda::candidate::ConvOxide;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{ComputeCapability, KernelModule, PtxTier};

/// Both modes use the same sm80 cubin; only TF32 selects the tensor-core entries
///
/// The FP32 entries match the legacy path within 0.04% in the b1/b32 trunk checks
/// All 144 sm80 checks pass against cuDNN and sampled f64 truth
pub(crate) const BINDING: ModuleBinding = ModuleBinding::new(
    ConvOxide::RTX50_SCOPE,
    ModuleRequest::new(
        KernelModule::Resnet,
        PtxTier::Sm80,
        LoadedArtifact::Cubin {
            arch: ComputeCapability::new(12, 0),
            sha256: ArtifactHash::from_hex(
                "0950b0d84cd9fa3d9053cd30399fce14a6aa6c3ff8777485598dd8deeba89078",
            ),
        },
    ),
);
