//! The measured wideconv artifact for the RTX 5090

use super::super::ModuleBinding;
use super::super::policy::Recipe;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{ComputeCapability, KernelModule, PtxTier};

/// The sm120 cubin, whose FP16 trunk kernels pair taps into m16n8k16 products and which
/// adds FP16 Winograd kernels for the same-channel 128- and 256-channel layers
///
/// Only this card loads the sm120 variant: every other device stays on the sm80 one,
/// whose kernels the sm120 build compiles differently and which lacks the Winograd
/// entries. On voxconverse-dev the paired taps cut the RTX 5090 Cuda run from 34.0 to
/// 30.9 s, with DER unchanged at 7.0123% and one of 216 RTTMs changed by 0.003 points
pub(crate) const BINDING: ModuleBinding = ModuleBinding::new(
    Recipe::Rtx5090.scope(),
    ModuleRequest::new(
        KernelModule::Wideconv,
        PtxTier::Sm120,
        LoadedArtifact::Cubin {
            arch: ComputeCapability::new(12, 0),
            sha256: ArtifactHash::from_hex(
                "06f574e49d8a2de6fd3aaef187c475678f6bf6b244fe2e3daac00549bb35447c",
            ),
        },
    ),
);
