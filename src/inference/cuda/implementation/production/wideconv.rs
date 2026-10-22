//! The measured wideconv artifact for the RTX 5090

use super::super::ModuleBinding;
use super::super::policy::Recipe;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{ComputeCapability, KernelModule, PtxTier};

/// The sm120 cubin, whose FP16 trunk kernels pair taps into m16n8k16 products
///
/// Only this card loads the sm120 variant: every other device stays on the sm80 one,
/// whose kernels the sm120 build compiles differently. On voxconverse-dev it cut the
/// RTX 5090 Cuda run from 34.0 to 30.9 s, with DER unchanged at 7.0123% and one of 216
/// RTTMs changed by 0.003 points
pub(crate) const BINDING: ModuleBinding = ModuleBinding::new(
    Recipe::Rtx5090.scope(),
    ModuleRequest::new(
        KernelModule::Wideconv,
        PtxTier::Sm120,
        LoadedArtifact::Cubin {
            arch: ComputeCapability::new(12, 0),
            sha256: ArtifactHash::from_hex(
                "9e39c5e3e8cae4920da8a2bab3fd20e3010d51f95515c40006c5ee2afab86185",
            ),
        },
    ),
);
