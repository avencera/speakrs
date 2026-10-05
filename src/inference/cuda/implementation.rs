//! Qualification-backed selection before any optional library state is created

use super::candidate::{
    Batches, ConvCandidate, ConvLayerSpec, ConvOxide, Coverage, CoverageEntry, LstmCandidate,
    LstmOxide, LstmSpec, SincCandidate, SincOxide, SincSpec,
};
use super::{
    ComputeCapability, CudaError, CudaLibrary, CudaMath, CudaRuntime, KernelModule, PtxTier,
};

/// The variant an area loads and the exact device that executes it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Target {
    pub tier: PtxTier,
    pub device: ComputeCapability,
}

impl Target {
    /// Resolve the area's actual variant, not the runtime's upper tier limit
    pub(crate) fn for_area(runtime: &CudaRuntime, area: KernelModule) -> Result<Self, CudaError> {
        Ok(Self {
            tier: runtime.area_ptx(area)?.0,
            device: runtime.compute_capability(),
        })
    }
}

/// Implementation requests used only by the qualification controls
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Choice {
    #[default]
    Library,
    Oxide,
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
    coverage: Coverage,
    tier: PtxTier,
    devices: &'static [ComputeCapability],
    record: &'static str,
    der: &'static str,
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

const fn production_entry(coverage: CoverageEntry) -> CoverageEntry {
    CoverageEntry {
        batches: Batches::Only(&MODEL_BATCHES),
        ..coverage
    }
}

const PRODUCTION: &[Production] = &[
    Production {
        coverage: Coverage(&[
            production_entry(ConvOxide::COVERAGE.0[0]),
            // b1 TF32 C64 was not accepted
            CoverageEntry {
                batches: Batches::Only(&[32]),
                ..ConvOxide::COVERAGE.0[1]
            },
            ConvOxide::COVERAGE.0[2],
        ]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        record: RESNET_RECORD,
        der: INTEGRATED_DER,
    },
    Production {
        coverage: Coverage(&[production_entry(LstmOxide::COVERAGE.0[0])]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        record: LSTM_RECORD,
        der: INTEGRATED_DER,
    },
    Production {
        coverage: Coverage(&[production_entry(SincOxide::COVERAGE.0[0])]),
        tier: PtxTier::Sm75,
        devices: DEVICES,
        record: SINC_RECORD,
        der: INTEGRATED_DER,
    },
];

/// A boundary accepted by a pinned record, or an explicit qualification control
#[derive(Debug)]
pub(crate) struct Qualified {
    boundary: String,
    batch: usize,
    math: CudaMath,
    target: Target,
    record: &'static str,
    der: &'static str,
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

    /// Build exactly the accepted convolution
    pub(crate) fn conv(
        self,
        runtime: &CudaRuntime,
        spec: ConvLayerSpec<'_>,
    ) -> Result<ConvOxide, CudaError> {
        self.check(
            runtime,
            KernelModule::Resnet,
            spec.name,
            spec.conv.batch,
            spec.conv.math,
        )?;
        ConvOxide::plan(runtime, spec)
    }

    /// Build exactly the accepted Sinc producer
    pub(crate) fn sinc(
        self,
        runtime: &CudaRuntime,
        spec: SincSpec<'_>,
    ) -> Result<SincOxide, CudaError> {
        self.check(
            runtime,
            KernelModule::Sincnet,
            "sincnet.conv0.abs_pool",
            spec.batch,
            spec.math,
        )?;
        SincOxide::plan(runtime, spec)
    }

    /// Build the accepted stack, including its required library projections
    pub(crate) fn lstm(
        self,
        runtime: &CudaRuntime,
        spec: LstmSpec<'_>,
    ) -> Result<LstmOxide, CudaError> {
        self.check(
            runtime,
            KernelModule::Lstm,
            "lstm.stack",
            spec.batch,
            spec.math,
        )?;
        LibraryNeed::new(
            KernelModule::Lstm,
            "lstm.stack.input_proj",
            spec.batch,
            spec.math,
            self.target,
            CudaLibrary::Cublas,
        )
        .prepare(runtime)?;
        LstmOxide::plan(runtime, spec)
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
        entry.tier == target.tier
            && entry.devices.contains(&target.device)
            && MODEL_BATCHES.contains(&batch)
            && entry.coverage.covers(boundary, batch, math)
    });
    Ok(match entry {
        Some(entry) => Selected::Oxide(Qualified {
            boundary: boundary.to_owned(),
            batch,
            math,
            target,
            record: entry.record,
            der: entry.der,
        }),
        None => Selected::Library,
    })
}

/// Whether a forced tier has any production evidence on this exact device
pub(crate) fn tier_qualified(target: Target) -> bool {
    PRODUCTION
        .iter()
        .any(|entry| entry.tier == target.tier && entry.devices.contains(&target.device))
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
    let target = Target::for_area(runtime, area)?;
    let selected =
        select(boundary, batch, math, target).map_err(|error| CudaError::Unsupported {
            context: "CUDA selection",
            reason: error.to_string(),
        })?;
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    {
        let choice = override_choice.unwrap_or_else(|| {
            super::test_support::default_choice(match selected {
                Selected::Oxide(_) => Choice::Oxide,
                _ => Choice::Library,
            })
        });
        qualification_selection(choice, area, boundary, batch, math, target)
    }
    #[cfg(not(all(test, feature = "cuda", not(feature = "cuda-driver-only"))))]
    Ok(selected)
}

#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
pub(crate) fn qualification_selection(
    choice: Choice,
    area: KernelModule,
    boundary: &str,
    batch: usize,
    math: CudaMath,
    target: Target,
) -> Result<Selected, CudaError> {
    let coverage = match area {
        KernelModule::Resnet => ConvOxide::COVERAGE,
        KernelModule::Lstm => LstmOxide::COVERAGE,
        KernelModule::Sincnet => SincOxide::COVERAGE,
        _ => Coverage::NONE,
    };
    Ok(match choice {
        Choice::Oxide if coverage.covers(boundary, batch, math) => Selected::Oxide(Qualified {
            boundary: boundary.to_owned(),
            batch,
            math,
            target,
            record: "qualification-control",
            der: "qualification-control",
        }),
        Choice::Mutant(mutant) => Selected::Mutant(mutant),
        _ => Selected::Library,
    })
}

/// A required library at a specific boundary, including nested products
#[derive(Debug, Clone)]
pub(crate) struct LibraryNeed {
    area: KernelModule,
    boundary: String,
    batch: usize,
    math: CudaMath,
    target: Target,
    library: CudaLibrary,
}

impl LibraryNeed {
    pub(crate) fn new(
        area: KernelModule,
        boundary: &str,
        batch: usize,
        math: CudaMath,
        target: Target,
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
