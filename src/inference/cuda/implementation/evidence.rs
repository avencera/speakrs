//! Production evidence: module bindings, accuracy-qualified tuple proofs and the
//! separate speed status of each proof
//!
//! Three coverage layers stay distinct. A candidate's implemented coverage is its
//! trait declaration. A [`TupleProof`] is accuracy acceptance of one execution
//! identity. Its [`SpeedStatus`] says whether speed was measured for a device scope.
//! An absent or unmeasured speed record never implies speed qualification

use std::fmt;

use super::BoundaryId;
use super::boundary::same as same_text;
use crate::inference::cuda::candidate::ConfigPin;
use crate::inference::cuda::device::DeviceAttributes;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

/// SHA-256 of one immutable record in the outside-tree cache
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RecordHash(ArtifactHash);

impl RecordHash {
    /// A canonical lowercase hash; invalid pins fail to compile
    pub(crate) const fn from_hex(text: &str) -> Self {
        Self(ArtifactHash::from_hex(text))
    }
}

impl fmt::Display for RecordHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Debug for RecordHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RecordHash({self})")
    }
}

/// The worst measured speedup on one architecture, in thousandths
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ArchitectureSpeed {
    pub(crate) capability: ComputeCapability,
    pub(crate) minimum_speedup_milli: u32,
}

/// Structural speed evidence accepted for every supported device
///
/// Construction requires two distinct architectures and at least 1.05x in every
/// measured case. The summary names the structural reason and the supporting reports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BroadEvidence {
    measurements: &'static [ArchitectureSpeed],
    summary: &'static str,
    minimum: ComputeCapability,
    minimum_tier: PtxTier,
}

impl BroadEvidence {
    /// The oldest architecture to which these structural wins apply
    pub(crate) const fn with_minimum(
        measurements: &'static [ArchitectureSpeed],
        summary: &'static str,
        minimum: ComputeCapability,
    ) -> Self {
        Self::with_limits(measurements, summary, minimum, PtxTier::Sm75)
    }

    /// Separate device coverage from the oldest artifact with the measured algorithm
    pub(crate) const fn with_limits(
        measurements: &'static [ArchitectureSpeed],
        summary: &'static str,
        minimum: ComputeCapability,
        minimum_tier: PtxTier,
    ) -> Self {
        let tier_minimum = minimum_tier.min_capability();
        assert!(
            minimum.major > tier_minimum.major
                || minimum.major == tier_minimum.major && minimum.minor >= tier_minimum.minor,
            "broad device scope must support its artifact tier"
        );
        assert!(
            measurements.len() >= 2 && !summary.is_empty(),
            "broad evidence needs two architectures and a summary"
        );
        let mut distinct_architecture = false;
        let mut index = 0;
        while index < measurements.len() {
            assert!(
                measurements[index].minimum_speedup_milli >= 1050,
                "every architecture must win by at least 1.05x"
            );
            let mut other = index + 1;
            while other < measurements.len() {
                assert!(
                    !same_capability(
                        measurements[index].capability,
                        measurements[other].capability
                    ),
                    "broad evidence needs distinct architectures"
                );
                other += 1;
            }
            if !same_architecture(measurements[0].capability, measurements[index].capability) {
                distinct_architecture = true;
            }
            index += 1;
        }
        assert!(
            distinct_architecture,
            "broad evidence needs two architectures, not two capabilities of one architecture"
        );
        Self {
            measurements,
            summary,
            minimum,
            minimum_tier,
        }
    }

    pub(crate) const fn measurements(self) -> &'static [ArchitectureSpeed] {
        self.measurements
    }
    pub(crate) const fn summary(self) -> &'static str {
        self.summary
    }

    const fn same(self, other: Self) -> bool {
        if self.minimum_tier as u8 != other.minimum_tier as u8
            || !same_capability(self.minimum, other.minimum)
            || !same_text(self.summary, other.summary)
            || self.measurements.len() != other.measurements.len()
        {
            return false;
        }
        let mut index = 0;
        while index < self.measurements.len() {
            let left = self.measurements[index];
            let right = other.measurements[index];
            if !same_capability(left.capability, right.capability)
                || left.minimum_speedup_milli != right.minimum_speedup_milli
            {
                return false;
            }
            index += 1;
        }
        true
    }
}

/// The devices one measurement, and the module binding built on it, covers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeedScope {
    /// A structural winner accepted on every supported GPU, with explicit evidence
    AllDevices(&'static BroadEvidence),
    /// One measured card: its capability, SM count and driver-reported name
    // the first new record adds a point binding; until then only tests construct one
    #[cfg_attr(not(all(test, feature = "_cuda-libraries")), allow(dead_code))]
    Point {
        capability: ComputeCapability,
        multiprocessors: u32,
        device_name: &'static str,
    },
    /// Development speed measured for a capability, without a legacy record claim
    MeasuredCapability { capability: ComputeCapability },
    /// A PR #36 record approved for a whole capability before SM count was a key;
    /// never copied into point evidence
    LegacyCapability { capability: ComputeCapability },
}

impl SpeedScope {
    pub(crate) const fn capability(self) -> ComputeCapability {
        match self {
            Self::AllDevices(evidence) => evidence.minimum,
            Self::Point { capability, .. }
            | Self::LegacyCapability { capability }
            | Self::MeasuredCapability { capability } => capability,
        }
    }

    /// Whether this exact device is inside the scope
    pub(crate) fn contains(self, device: &DeviceAttributes) -> bool {
        match self {
            Self::AllDevices(evidence) => device.capability() >= evidence.minimum,
            Self::Point {
                capability,
                multiprocessors,
                device_name,
            } => {
                device.capability() == capability
                    && device.multiprocessors().get() == multiprocessors
                    && device.name() == device_name
            }
            Self::LegacyCapability { capability } | Self::MeasuredCapability { capability } => {
                device.capability() == capability
            }
        }
    }

    /// Whether this artifact tier contains the algorithm covered by the speed evidence
    pub(crate) fn allows_tier(self, tier: PtxTier) -> bool {
        match self {
            Self::AllDevices(evidence) => tier >= evidence.minimum_tier,
            _ => true,
        }
    }

    /// Whether speed was measured on this capability, distinct from broad acceptance
    pub(crate) fn measured_on_device(self, capability: ComputeCapability) -> bool {
        match self {
            Self::AllDevices(evidence) => evidence
                .measurements()
                .iter()
                .any(|speed| speed.capability == capability),
            _ => self.capability() == capability,
        }
    }

    /// Whether one device could be inside both scopes
    const fn overlaps(self, other: Self) -> bool {
        if matches!(self, Self::AllDevices(_)) || matches!(other, Self::AllDevices(_)) {
            return true;
        }
        if !same_capability(self.capability(), other.capability()) {
            return false;
        }
        match (self, other) {
            (
                Self::Point {
                    multiprocessors: left_sms,
                    device_name: left_name,
                    ..
                },
                Self::Point {
                    multiprocessors: right_sms,
                    device_name: right_name,
                    ..
                },
            ) => left_sms == right_sms && same_text(left_name, right_name),
            _ => true,
        }
    }

    const fn same(self, other: Self) -> bool {
        match (self, other) {
            (Self::AllDevices(left), Self::AllDevices(right)) => left.same(*right),
            (
                Self::Point {
                    capability: left,
                    multiprocessors: left_sms,
                    device_name: left_name,
                },
                Self::Point {
                    capability: right,
                    multiprocessors: right_sms,
                    device_name: right_name,
                },
            ) => {
                same_capability(left, right)
                    && left_sms == right_sms
                    && same_text(left_name, right_name)
            }
            (
                Self::MeasuredCapability { capability: left },
                Self::MeasuredCapability { capability: right },
            ) => same_capability(left, right),
            (
                Self::LegacyCapability { capability: left },
                Self::LegacyCapability { capability: right },
            ) => same_capability(left, right),
            _ => false,
        }
    }
}

/// A speed record for one device scope, with the integrated DER evidence that
/// accepted the complete configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpeedEvidence {
    pub(crate) scope: SpeedScope,
    pub(crate) record: RecordHash,
    pub(crate) integrated: RecordHash,
}

/// Whether a proof's speed was measured; absence is never qualification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeedStatus {
    Measured(SpeedEvidence),
    /// Accuracy-qualified only; production uses Library while libraries exist
    // no accepted proof is speed-unmeasured yet; tests construct one
    #[cfg_attr(not(all(test, feature = "_cuda-libraries")), allow(dead_code))]
    Unmeasured,
}

/// One accuracy-accepted execution identity of a binding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TupleProof {
    pub(crate) boundary: BoundaryId,
    pub(crate) batch: usize,
    pub(crate) math: CudaMath,
    pub(crate) pin: ConfigPin,
    pub(crate) accuracy: RecordHash,
    pub(crate) speed: SpeedStatus,
}

impl TupleProof {
    pub(crate) fn covers(&self, boundary: BoundaryId, batch: usize, math: CudaMath) -> bool {
        self.boundary == boundary && self.batch == batch && self.math == math
    }
}

/// One module binding: the request an area loads on one device scope, with every
/// tuple proof that uses it
///
/// The runtime caches one module per area, so bindings for one area on overlapping
/// scopes must request the same module
#[derive(Debug, Clone, Copy)]
pub(crate) struct Binding {
    pub(crate) scope: SpeedScope,
    pub(crate) module: ModuleRequest,
    pub(crate) proofs: &'static [TupleProof],
}

impl Binding {
    pub(crate) const fn area(&self) -> KernelModule {
        self.module.area()
    }

    pub(crate) fn proof(
        &self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
    ) -> Option<&'static TupleProof> {
        self.proofs
            .iter()
            .find(|proof| proof.covers(boundary, batch, math))
    }
}

/// Reject tables that could load two modules for one area or misstate their evidence
///
/// `always_on` areas keep their own policy and cannot be bound. `precedence` orders
/// the routes that may implement one boundary and must name every bound area once
pub(crate) const fn validate(
    bindings: &[Binding],
    always_on: &[KernelModule],
    precedence: &[KernelModule],
) {
    let mut index = 0;
    while index < precedence.len() {
        let mut other = index + 1;
        while other < precedence.len() {
            assert!(
                precedence[index] as u8 != precedence[other] as u8,
                "route precedence names an area twice"
            );
            other += 1;
        }
        index += 1;
    }

    let mut index = 0;
    while index < bindings.len() {
        validate_binding(&bindings[index], always_on, precedence);
        let mut other = index + 1;
        while other < bindings.len() {
            validate_pair(&bindings[index], &bindings[other]);
            other += 1;
        }
        index += 1;
    }
}

const fn validate_binding(
    binding: &Binding,
    always_on: &[KernelModule],
    precedence: &[KernelModule],
) {
    let area = binding.area() as u8;
    assert!(
        !contains_area(always_on, area),
        "always-on areas cannot have record bindings"
    );
    assert!(
        contains_area(precedence, area),
        "every bound area needs a route precedence"
    );
    let module = binding.module;
    let capability = binding.scope.capability();
    let minimum = module.tier().min_capability();
    assert!(
        capability.major > minimum.major
            || capability.major == minimum.major && capability.minor >= minimum.minor,
        "a binding's device cannot run its tier"
    );
    if let LoadedArtifact::Cubin { arch, .. } = module.artifact() {
        assert!(
            !matches!(binding.scope, SpeedScope::AllDevices(_)),
            "all-device bindings require portable PTX, not a device cubin"
        );
        assert!(
            same_capability(arch, capability),
            "a cubin binding must target its exact device"
        );
    }
    if let SpeedScope::Point {
        multiprocessors,
        device_name,
        ..
    } = binding.scope
    {
        assert!(multiprocessors > 0 && !device_name.is_empty());
    }
    assert!(!binding.proofs.is_empty(), "a binding needs a tuple proof");

    let mut index = 0;
    while index < binding.proofs.len() {
        let proof = &binding.proofs[index];
        assert!(
            proof.batch > 0 && proof.boundary.batches().contains(proof.batch),
            "a proof batch must be a production batch of its boundary"
        );
        assert!(
            proof.pin.area() as u8 == area,
            "a proof pin must run on its binding's module"
        );
        if proof.pin.is_device_rule() {
            assert!(
                matches!(binding.scope, SpeedScope::LegacyCapability { .. }),
                "point evidence must pin one fixed configuration"
            );
        }
        if let SpeedStatus::Measured(evidence) = proof.speed {
            assert!(
                evidence.scope.same(binding.scope),
                "speed evidence must cover exactly its binding's scope"
            );
        }
        let mut other = index + 1;
        while other < binding.proofs.len() {
            assert!(
                !same_tuple(proof, &binding.proofs[other]),
                "a binding proves each tuple once"
            );
            other += 1;
        }
        index += 1;
    }
}

const fn validate_pair(left: &Binding, right: &Binding) {
    if left.area() as u8 != right.area() as u8 || !left.scope.overlaps(right.scope) {
        return;
    }
    assert!(
        same_module(left.module, right.module),
        "conflicting module bindings for one area and device"
    );
    // one module may carry several records, but each tuple has one proof per device
    let mut index = 0;
    while index < left.proofs.len() {
        let mut other = 0;
        while other < right.proofs.len() {
            assert!(
                !same_tuple(&left.proofs[index], &right.proofs[other]),
                "overlapping bindings prove one tuple twice"
            );
            other += 1;
        }
        index += 1;
    }
}

const fn contains_area(areas: &[KernelModule], area: u8) -> bool {
    let mut index = 0;
    while index < areas.len() {
        if areas[index] as u8 == area {
            return true;
        }
        index += 1;
    }
    false
}

const fn same_tuple(left: &TupleProof, right: &TupleProof) -> bool {
    left.boundary.same(right.boundary)
        && left.batch == right.batch
        && left.math as u8 == right.math as u8
}

const fn same_capability(left: ComputeCapability, right: ComputeCapability) -> bool {
    left.major == right.major && left.minor == right.minor
}

const fn same_module(left: ModuleRequest, right: ModuleRequest) -> bool {
    left.area() as u8 == right.area() as u8
        && left.tier() as u8 == right.tier() as u8
        && same_artifact(left.artifact(), right.artifact())
}

const fn same_artifact(left: LoadedArtifact, right: LoadedArtifact) -> bool {
    match (left, right) {
        (LoadedArtifact::PtxJit { sha256: left }, LoadedArtifact::PtxJit { sha256: right }) => {
            left.const_eq(right)
        }
        (
            LoadedArtifact::Cubin {
                arch: left_arch,
                sha256: left,
            },
            LoadedArtifact::Cubin {
                arch: right_arch,
                sha256: right,
            },
        ) => same_capability(left_arch, right_arch) && left.const_eq(right),
        _ => false,
    }
}

// Ada (8.9) and Ampere (8.0/8.6/8.7) share a capability major but are separate architectures
const fn same_architecture(left: ComputeCapability, right: ComputeCapability) -> bool {
    left.major == right.major && (left.major != 8 || (left.minor == 9) == (right.minor == 9))
}
