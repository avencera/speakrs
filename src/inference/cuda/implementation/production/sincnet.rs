//! The PR #36 Sinc producer binding on capability 12.0

use super::{FP32, LEGACY_SCOPE, legacy_proof};
use crate::inference::cuda::candidate::{ConfigPin, SincPin};
use crate::inference::cuda::implementation::{Binding, BoundaryId, RecordHash, TupleProof};
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{KernelModule, PtxTier};

/// SHA256 of the archived SincNet accuracy and speed record
pub(crate) const RECORD: RecordHash =
    RecordHash::from_hex("a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675");

const PIN: ConfigPin = ConfigPin::Sinc(SincPin::ConvAbsPool);

/// Only FP32 was accepted
const PROOFS: [TupleProof; 2] = [
    legacy_proof(
        BoundaryId::named("sincnet.conv0.abs_pool"),
        1,
        FP32,
        PIN,
        RECORD,
    ),
    legacy_proof(
        BoundaryId::named("sincnet.conv0.abs_pool"),
        32,
        FP32,
        PIN,
        RECORD,
    ),
];

pub(crate) const BINDING: Binding = Binding {
    scope: LEGACY_SCOPE,
    module: ModuleRequest::new(
        KernelModule::Sincnet,
        PtxTier::Sm75,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "967bc6893f80da84d8d4d288f2cf1ca3336ca09c4386495cab722beb0ac87247",
            ),
        },
    ),
    proofs: &PROOFS,
};
