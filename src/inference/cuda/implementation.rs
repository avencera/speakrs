//! Qualification-backed selection before any optional library state is created

// runtime wiring is deferred until override policy and input ownership are defined
#[allow(dead_code)]
pub(crate) mod overrides;

use super::kernels::{ArtifactHash, ArtifactRequest, LoadedArtifact};

use super::candidate::{
    Batches, ConvCandidate, ConvLayerSpec, ConvOxide, Coverage, CoverageEntry, LstmCandidate,
    LstmOxide, LstmSpec, Maths, PlanError, SincCandidate, SincOxide, SincSpec,
};
use super::{
    ComputeCapability, CudaError, CudaLibrary, CudaMath, CudaRuntime, KernelModule, PtxTier,
};

/// The variant an area loads and the exact device that executes it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Target {
    pub tier: PtxTier,
    pub device: ComputeCapability,
    pub artifact: LoadedArtifact,
}

impl Target {
    /// Resolve the area's actual variant, not the runtime's upper tier limit
    pub(crate) fn for_area(runtime: &CudaRuntime, area: KernelModule) -> Result<Self, CudaError> {
        let loaded = runtime.load_kernels(area)?;
        Ok(Self {
            tier: loaded.tier(),
            artifact: loaded.artifact(),
            device: runtime.compute_capability(),
        })
    }
}

/// An embedded variant and device, without claiming that an artifact was loaded
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AreaTarget {
    pub tier: PtxTier,
    pub device: ComputeCapability,
}

impl AreaTarget {
    /// Resolve Library diagnostics without loading an unused candidate module
    pub(crate) fn for_area(runtime: &CudaRuntime, area: KernelModule) -> Result<Self, CudaError> {
        Ok(Self {
            tier: runtime.area_ptx(area)?.0,
            device: runtime.compute_capability(),
        })
    }
}

/// Why a candidate was selected; only production permits a device fallback
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Selection {
    /// An accepted production-table entry
    Production,
    /// An explicit or qualification request
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Explicit,
}

/// Implementation requests used only by the qualification controls
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Choice {
    #[default]
    Library,
    Oxide(Selection),
    /// The pinned FP32 production path with a stage-only timing fault
    StageTail,
    /// The same pinned FP32 production path without the timing fault
    StageTailControl,
    Mutant(super::test_support::Mutant),
}

/// One selected owner; an Oxide token can only come from this module
#[derive(Debug)]
pub(crate) enum Selected {
    Library,
    Oxide(Qualified),
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Mutant(super::test_support::Mutant),
}

/// An accepted production record, separate from candidate implementation coverage
#[derive(Debug)]
struct Production {
    area: KernelModule,
    coverage: Coverage,
    tier: PtxTier,
    devices: &'static [ComputeCapability],
    artifact: LoadedArtifact,
    record: &'static str,
    der: &'static str,
}

impl Production {
    fn matches_target(&self, target: Target) -> bool {
        self.tier == target.tier
            && self.devices.contains(&target.device)
            && self.artifact == target.artifact
    }
}

/// SHA256 of qualify-resnet-Oxide-20261004T064208.284837Z.json.gz
const RESNET_RECORD: &str = "8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758";
/// SHA256 of qualify-lstm-Oxide-20261004T095844.546766Z.json.gz
const LSTM_RECORD: &str = "3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8";
/// SHA256 of qualify-sincnet-Oxide-20261004T093054.587886Z.json.gz
const SINC_RECORD: &str = "a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675";
/// SHA256 of int-k/ab-summary.json for the integrated configuration
const INTEGRATED_DER: &str = "8066268031afba058d93e305206d5b8225e40c6ab2ebb1052874d607e055646f";

/// Production model batches, excluding the harness stress classes
pub(crate) const MODEL_BATCHES: [usize; 2] = [1, 32];
const DEVICES: &[ComputeCapability] = &[ComputeCapability::new(12, 0)];

// these layers and modes are pinned evidence, not candidate declarations
const C32: &[&str] = &[
    "resnet.layer1.0.conv1",
    "resnet.layer1.0.conv2",
    "resnet.layer1.1.conv1",
    "resnet.layer1.1.conv2",
    "resnet.layer1.2.conv1",
    "resnet.layer1.2.conv2",
];
const C64: &[&str] = &[
    "resnet.layer2.0.conv2",
    "resnet.layer2.1.conv1",
    "resnet.layer2.1.conv2",
    "resnet.layer2.2.conv1",
    "resnet.layer2.2.conv2",
    "resnet.layer2.3.conv1",
    "resnet.layer2.3.conv2",
];

const PRODUCTION: &[Production] = &[
    Production {
        area: KernelModule::Resnet,
        coverage: Coverage(&[
            CoverageEntry {
                layers: C32,
                batches: Batches::Only(&MODEL_BATCHES),
                maths: Maths::All,
            },
            // the 36-SM RTX 5060 Ti control recorded a hard "candidate slower than the
            // faster Library process" failure, then unresolved noise (min pair 0.989,
            // spread 0.025); the legacy record used a 70-SM RTX 5070 Ti
            // a recorded hard failure cannot be outweighed by a later pass
            CoverageEntry {
                layers: &["resnet.layer2.0.conv1"],
                batches: Batches::Only(&[32]),
                maths: Maths::All,
            },
            CoverageEntry {
                layers: &["resnet.layer2.0.conv1"],
                batches: Batches::Only(&[1]),
                maths: Maths::Only(&[CudaMath::Tf32]),
            },
            CoverageEntry {
                layers: C64,
                batches: Batches::Only(&[32]),
                maths: Maths::All,
            },
            CoverageEntry {
                layers: C64,
                batches: Batches::Only(&[1]),
                maths: Maths::Only(&[CudaMath::Fp32]),
            },
        ]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        artifact: LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "dd6449c0129f9a03bf691c0338611b50b651ab714d5803caedea87de3c72b6b7",
            ),
        },
        record: RESNET_RECORD,
        der: INTEGRATED_DER,
    },
    Production {
        area: KernelModule::Lstm,
        coverage: Coverage(&[CoverageEntry {
            layers: &["lstm.stack"],
            batches: Batches::Only(&MODEL_BATCHES),
            maths: Maths::Only(&[CudaMath::Fp32]),
        }]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        artifact: LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "72945743a3c1b915c05d8ea21b438dfd860fa9487401c447fb24fd48802916fa",
            ),
        },
        record: LSTM_RECORD,
        der: INTEGRATED_DER,
    },
    Production {
        area: KernelModule::Sincnet,
        coverage: Coverage(&[CoverageEntry {
            layers: &["sincnet.conv0.abs_pool"],
            batches: Batches::Only(&MODEL_BATCHES),
            maths: Maths::Only(&[CudaMath::Fp32]),
        }]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        artifact: LoadedArtifact::PtxJit {
            sha256: ArtifactHash::from_hex(
                "967bc6893f80da84d8d4d288f2cf1ca3336ca09c4386495cab722beb0ac87247",
            ),
        },
        record: SINC_RECORD,
        der: INTEGRATED_DER,
    },
];

// one owner per area/tier/device prevents conflicting artifact requests at compile time
const _: () = validate_production_owners(PRODUCTION);

const fn validate_production_owners(entries: &[Production]) {
    let mut i = 0;
    while i < entries.len() {
        let mut j = i + 1;
        while j < entries.len() {
            if entries[i].area as u8 == entries[j].area as u8
                && entries[i].tier as u8 == entries[j].tier as u8
            {
                let mut a = 0;
                while a < entries[i].devices.len() {
                    let mut b = 0;
                    while b < entries[j].devices.len() {
                        let left = entries[i].devices[a];
                        let right = entries[j].devices[b];
                        assert!(
                            left.major != right.major || left.minor != right.minor,
                            "production area/tier/device must have one artifact owner"
                        );
                        b += 1;
                    }
                    a += 1;
                }
            }
            j += 1;
        }
        i += 1;
    }
}

/// Production artifact policy, independent of an explicit qualification request
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProductionArtifactOwner {
    /// Preserve the always-on kernels' driver-JIT behavior on every device
    AlwaysOnPtxJit,
    /// Use exactly the artifact accepted by a production record
    Qualified(LoadedArtifact),
}

impl ProductionArtifactOwner {
    /// Resolve the policy with the bytes embedded in this binary
    pub(crate) fn request(self, ptx: &str) -> ArtifactRequest {
        let artifact = match self {
            Self::AlwaysOnPtxJit => LoadedArtifact::PtxJit {
                sha256: ArtifactHash::of(ptx.as_bytes()),
            },
            Self::Qualified(artifact) => artifact,
        };
        ArtifactRequest::Pinned(artifact)
    }
}

// always-on areas retain the artifact policy used before cubins were shipped
const ALWAYS_ON: &[KernelModule] = &[
    KernelModule::Fbank,
    KernelModule::Embedding,
    KernelModule::Segmentation,
];

const _: () = {
    let mut index = 0;
    while index < PRODUCTION.len() {
        let mut always = 0;
        while always < ALWAYS_ON.len() {
            assert!(
                PRODUCTION[index].area as u8 != ALWAYS_ON[always] as u8,
                "always-on policy and record owner must not overlap"
            );
            always += 1;
        }
        index += 1;
    }
};

/// Every production module load must have an owner before it reaches the driver
pub(crate) fn artifact_owner(
    area: KernelModule,
    location: AreaTarget,
) -> Option<ProductionArtifactOwner> {
    if ALWAYS_ON.contains(&area) {
        return Some(ProductionArtifactOwner::AlwaysOnPtxJit);
    }
    production_artifact(area, location).map(ProductionArtifactOwner::Qualified)
}

/// The record owns the artifact request before any module is loaded
pub(crate) fn production_artifact(
    area: KernelModule,
    location: AreaTarget,
) -> Option<LoadedArtifact> {
    production_owner(PRODUCTION, area, location).map(|entry| entry.artifact)
}

fn production_owner(
    entries: &[Production],
    area: KernelModule,
    location: AreaTarget,
) -> Option<&Production> {
    entries.iter().find(|entry| {
        entry.area == area
            && entry.tier == location.tier
            && entry.devices.contains(&location.device)
    })
}

/// A boundary accepted by a pinned record, or an explicit qualification control
#[derive(Debug)]
pub(crate) struct Qualified {
    boundary: String,
    batch: usize,
    math: CudaMath,
    target: Target,
    record: &'static str,
    der: &'static str,
    selection: Selection,
}

impl Qualified {
    fn check(
        &self,
        runtime: &CudaRuntime,
        area: KernelModule,
        boundary: &str,
        batch: usize,
        math: CudaMath,
    ) -> Result<(), CudaError> {
        if self.boundary != boundary
            || self.batch != batch
            || self.math != math
            || self.target != Target::for_area(runtime, area)?
        {
            return Err(CudaError::Unsupported {
                context: "qualified plan",
                reason: "qualification token does not match the requested plan".to_owned(),
            });
        }
        debug_assert!(!self.record.is_empty() && !self.der.is_empty());
        Ok(())
    }

    /// Resolve a refusal without constructing a dormant Library plan
    pub(super) fn finish<T>(
        &self,
        area: KernelModule,
        driver_only: bool,
        result: Result<T, PlanError>,
    ) -> Result<Option<T>, CudaError> {
        match result {
            Ok(plan) => Ok(Some(plan)),
            Err(PlanError::Cuda(error)) => Err(error),
            Err(PlanError::DeviceUnsupported { reason }) => {
                if self.selection == Selection::Production && !driver_only {
                    tracing::warn!(
                        boundary = self.boundary,
                        batch = self.batch,
                        "CUDA candidate unavailable reason={reason}; using Library"
                    );
                    return Ok(None);
                }

                Err(CudaError::CandidateDeviceUnsupported {
                    area: area.name(),
                    boundary: self.boundary.clone(),
                    batch: self.batch,
                    math: self.math,
                    tier: self.target.tier,
                    device: self.target.device,
                    reason,
                })
            }
        }
    }

    /// Build exactly the accepted convolution
    pub(crate) fn conv(
        self,
        runtime: &CudaRuntime,
        spec: ConvLayerSpec<'_>,
    ) -> Result<Option<ConvOxide>, CudaError> {
        self.check(
            runtime,
            KernelModule::Resnet,
            spec.name,
            spec.conv.batch,
            spec.conv.math,
        )?;
        self.finish(
            KernelModule::Resnet,
            super::driver_only(),
            ConvOxide::plan(runtime, spec),
        )
    }

    /// Build exactly the accepted Sinc producer
    pub(crate) fn sinc(
        self,
        runtime: &CudaRuntime,
        spec: SincSpec<'_>,
    ) -> Result<Option<SincOxide>, CudaError> {
        self.check(
            runtime,
            KernelModule::Sincnet,
            "sincnet.conv0.abs_pool",
            spec.batch,
            spec.math,
        )?;
        self.finish(
            KernelModule::Sincnet,
            super::driver_only(),
            SincOxide::plan(runtime, spec),
        )
    }

    /// Build the accepted stack; qualification forbids nested library calls
    pub(crate) fn lstm(
        self,
        runtime: &CudaRuntime,
        spec: LstmSpec<'_>,
    ) -> Result<Option<LstmOxide>, CudaError> {
        self.check(
            runtime,
            KernelModule::Lstm,
            "lstm.stack",
            spec.batch,
            spec.math,
        )?;
        self.finish(
            KernelModule::Lstm,
            super::driver_only(),
            LstmOxide::plan(runtime, spec),
        )
    }
}

/// Invalid selection requests cannot create an Oxide token
#[derive(Debug, thiserror::Error)]
#[error("cannot select a CUDA boundary with an empty name or batch zero")]
pub(crate) struct SelectionError;

/// Select a pinned tuple; uncovered tuples use a Library plan
pub(crate) fn select(
    boundary: &str,
    batch: usize,
    math: CudaMath,
    target: Target,
) -> Result<Selected, SelectionError> {
    if boundary.is_empty() || batch == 0 {
        return Err(SelectionError);
    }
    let entry = PRODUCTION.iter().find(|entry| {
        entry.matches_target(target)
            && MODEL_BATCHES.contains(&batch)
            && entry.coverage.covers(boundary, batch, math)
            && boundary.split('.').next() == Some(entry.area.name())
    });
    Ok(match entry {
        Some(entry) => Selected::Oxide(Qualified {
            boundary: boundary.to_owned(),
            batch,
            math,
            target,
            record: entry.record,
            der: entry.der,
            selection: Selection::Production,
        }),
        None => Selected::Library,
    })
}

/// Whether a forced tier has any production evidence on this exact device
pub(crate) fn tier_qualified(area: KernelModule, target: Target) -> bool {
    PRODUCTION
        .iter()
        .any(|entry| entry.area == area && entry.matches_target(target))
}

/// Declare legacy fixture coverage without loading an otherwise unused module
///
/// This only enumerates test tuples. Actual plans still need the successful
/// loaded artifact to obtain a production token through `plan_selection`
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
pub(crate) fn legacy_fixture_coverage(
    area: KernelModule,
    location: AreaTarget,
    embedded_ptx_sha256: ArtifactHash,
) -> Coverage {
    PRODUCTION
        .iter()
        .find(|entry| {
            entry.area == area
                && entry.tier == location.tier
                && entry.devices.contains(&location.device)
                && entry.artifact
                    == LoadedArtifact::PtxJit {
                        sha256: embedded_ptx_sha256,
                    }
        })
        .map_or(Coverage::NONE, |entry| entry.coverage)
}

/// Selection for a concrete plan, with test controls kept outside production
pub(crate) fn plan_selection(
    runtime: &CudaRuntime,
    area: KernelModule,
    boundary: &str,
    batch: usize,
    math: CudaMath,
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))] override_choice: Option<
        Choice,
    >,
) -> Result<Selected, CudaError> {
    let request = PlanRequest::Production;
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    let request = override_choice.map_or_else(
        || match super::test_support::default_choice(Choice::Oxide(Selection::Production)) {
            Choice::Oxide(Selection::Production) => request,
            choice => PlanRequest::Qualification(choice),
        },
        PlanRequest::Qualification,
    );
    request.resolve(
        area,
        boundary,
        batch,
        math,
        AreaTarget::for_area(runtime, area)?,
        |request| {
            let loaded = runtime.load_requested_kernels(area, request)?;
            Ok(Target {
                tier: loaded.tier(),
                device: runtime.compute_capability(),
                artifact: loaded.artifact(),
            })
        },
    )
}

/// Selection intent precedes artifact loading; only an Oxide token needs artifact identity
#[derive(Debug, Clone, Copy)]
enum PlanRequest {
    Production,
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    Qualification(Choice),
}

impl PlanRequest {
    fn resolve(
        self,
        area: KernelModule,
        boundary: &str,
        batch: usize,
        math: CudaMath,
        location: AreaTarget,
        load: impl FnOnce(ArtifactRequest) -> Result<Target, CudaError>,
    ) -> Result<Selected, CudaError> {
        if boundary.is_empty() || batch == 0 {
            return Err(CudaError::Unsupported {
                context: "CUDA selection",
                reason: SelectionError.to_string(),
            });
        }

        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        if let Self::Qualification(choice) = self {
            match choice {
                Choice::Library => return Ok(Selected::Library),
                Choice::Mutant(mutant) => return Ok(Selected::Mutant(mutant)),
                Choice::StageTail | Choice::StageTailControl if math != CudaMath::Fp32 => {
                    return Ok(Selected::Library);
                }
                Choice::Oxide(Selection::Explicit) => {
                    if !candidate_coverage(area, location.tier).covers(boundary, batch, math) {
                        return Ok(Selected::Library);
                    }
                    return qualification_selection(
                        choice,
                        area,
                        boundary,
                        batch,
                        math,
                        load(ArtifactRequest::EmbeddedExact)?,
                    );
                }
                _ => {}
            }
        }

        // uncovered production tuples cannot need a loaded artifact, even for diagnostics
        let Some(entry) = production_owner(PRODUCTION, area, location).filter(|entry| {
            MODEL_BATCHES.contains(&batch) && entry.coverage.covers(boundary, batch, math)
        }) else {
            return Ok(Selected::Library);
        };
        let target = match load(ArtifactRequest::Pinned(entry.artifact)) {
            Ok(target) => target,
            Err(error) => {
                return artifact_refusal(
                    error,
                    matches!(self, Self::Production) && !super::driver_only(),
                );
            }
        };
        select(boundary, batch, math, target).map_err(|error| CudaError::Unsupported {
            context: "CUDA selection",
            reason: error.to_string(),
        })
    }
}

/// Only production may use the established Library policy after an artifact refusal
fn artifact_refusal(error: CudaError, library_allowed: bool) -> Result<Selected, CudaError> {
    if library_allowed
        && matches!(
            error,
            CudaError::ArtifactLoad { .. } | CudaError::ArtifactUnavailable { .. }
        )
    {
        tracing::warn!(%error, "CUDA qualified artifact unavailable; using Library");
        return Ok(Selected::Library);
    }
    Err(error)
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
fn candidate_coverage(area: KernelModule, tier: PtxTier) -> Coverage {
    match area {
        KernelModule::Resnet => ConvOxide::coverage(tier),
        KernelModule::Lstm => LstmOxide::coverage(tier),
        KernelModule::Sincnet => SincOxide::coverage(tier),
        _ => Coverage::NONE,
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
fn qualification_selection(
    choice: Choice,
    area: KernelModule,
    boundary: &str,
    batch: usize,
    math: CudaMath,
    target: Target,
) -> Result<Selected, CudaError> {
    let coverage = candidate_coverage(area, target.tier);
    Ok(match choice {
        Choice::Oxide(selection) if coverage.covers(boundary, batch, math) => {
            Selected::Oxide(Qualified {
                boundary: boundary.to_owned(),
                batch,
                math,
                target,
                record: "qualification-control",
                der: "qualification-control",
                selection,
            })
        }
        Choice::Mutant(mutant) => Selected::Mutant(mutant),
        Choice::StageTail | Choice::StageTailControl if math == CudaMath::Fp32 => {
            // isolated segmentation selectors use this owner directly, without plan_selection
            select(boundary, batch, math, target).map_err(|error| CudaError::Unsupported {
                context: "CUDA selection",
                reason: error.to_string(),
            })?
        }
        Choice::Library | Choice::Oxide(_) | Choice::StageTail | Choice::StageTailControl => {
            Selected::Library
        }
    })
}

/// A required library at a specific boundary, including nested products
#[derive(Debug, Clone)]
pub(crate) struct LibraryNeed {
    area: KernelModule,
    boundary: String,
    batch: usize,
    math: CudaMath,
    target: AreaTarget,
    library: CudaLibrary,
}

impl LibraryNeed {
    /// Describe required Library state without constructing a candidate artifact key
    pub(crate) fn new(
        area: KernelModule,
        boundary: &str,
        batch: usize,
        math: CudaMath,
        target: AreaTarget,
        library: CudaLibrary,
    ) -> Self {
        Self {
            area,
            boundary: boundary.to_owned(),
            batch,
            math,
            target,
            library,
        }
    }

    pub(crate) fn error(&self) -> CudaError {
        CudaError::NotDriverOnly {
            area: self.area.name(),
            boundary: self.boundary.clone(),
            batch: self.batch,
            math: self.math,
            tier: self.target.tier,
            device: self.target.device,
            library: self.library,
        }
    }

    /// Resolve optional libraries before buffers, forward execution or graph capture
    pub(crate) fn prepare(&self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(self.error());
        }
        #[cfg(feature = "cuda")]
        return runtime.prepare_library(self.library);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = runtime;
            Err(self.error())
        }
    }

    /// Prepare selected handles before model buffers, leaving NVRTC to its Library RNN plan
    pub(crate) fn prepare_handle(&self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        match self.library {
            CudaLibrary::Nvrtc if !super::driver_only() => Ok(()),
            _ => self.prepare(runtime),
        }
    }
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
mod tests;
