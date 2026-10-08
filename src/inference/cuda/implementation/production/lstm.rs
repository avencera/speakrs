//! The PR #36 LSTM stack binding on capability 12.0

use super::{FP32, LEGACY_SCOPE, legacy_proof};
use crate::inference::cuda::candidate::{ConfigPin, LstmPin};
use crate::inference::cuda::implementation::{Binding, BoundaryId, RecordHash, TupleProof};
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{KernelModule, PtxTier};

/// SHA256 of qualify-lstm-Oxide-20261004T095844.546766Z.json.gz
pub(crate) const RECORD: RecordHash =
    RecordHash::from_hex("3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8");

const PIN: ConfigPin = ConfigPin::Lstm(LstmPin::LegacyCooperative);

/// Only FP32 was accepted
const PROOFS: [TupleProof; 2] = [
    legacy_proof(BoundaryId::named("lstm.stack"), 1, FP32, PIN, RECORD),
    legacy_proof(BoundaryId::named("lstm.stack"), 32, FP32, PIN, RECORD),
];

pub(crate) const BINDING: Binding = Binding {
    scope: LEGACY_SCOPE,
    module: ModuleRequest::new(
        KernelModule::Lstm,
        PtxTier::Sm75,
        LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "72945743a3c1b915c05d8ea21b438dfd860fa9487401c447fb24fd48802916fa",
            ),
        },
    ),
    proofs: &PROOFS,
};
