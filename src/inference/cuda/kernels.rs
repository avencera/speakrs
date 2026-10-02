use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};

use super::{CudaError, PtxTier};

/// Embeds a higher-tier PTX file only when its Cargo feature is on, so a default
/// build neither contains nor can load code for newer GPUs
///
/// `cargo xtask cuda-kernels check` requires every variant above the baseline to be
/// embedded through this macro with the feature of its own tier
// only the test-only probe area ships a higher tier today
#[cfg_attr(not(test), allow(unused_macros))]
macro_rules! tier_ptx {
    ($feature:literal, $path:literal) => {{
        #[cfg(feature = $feature)]
        let ptx = Some(include_str!($path));
        #[cfg(not(feature = $feature))]
        let ptx = None;
        ptx
    }};
}

/// A cuda-oxide kernel area, embedded as committed PTX
///
/// `cargo xtask cuda-kernels build` regenerates `ptx/<area>.<tier>.ptx` on a GPU box,
/// and `cargo xtask cuda-kernels check` fails when that PTX is stale. Every area ships
/// an `sm75` baseline, which the driver JIT-compiles for Turing and every newer GPU,
/// and may ship higher tiers behind the `cuda-sm80`, `cuda-sm90` and `cuda-sm120`
/// features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KernelModule {
    /// Toolchain probe kernels, which only the tier-dispatch tests load
    #[cfg(test)]
    Probe,
    /// Filterbank front-end kernels
    Fbank,
    /// ResNet34 multi-mask embedding kernels
    Embedding,
    /// Segmentation kernels
    Segmentation,
}

impl KernelModule {
    /// The area name, which is also the PTX file stem
    pub const fn name(self) -> &'static str {
        match self {
            #[cfg(test)]
            Self::Probe => "probe",
            Self::Fbank => "fbank",
            Self::Embedding => "embedding",
            Self::Segmentation => "segmentation",
        }
    }

    /// The PTX variants embedded in this build
    pub const fn variants(self) -> AreaPtx {
        match self {
            #[cfg(test)]
            Self::Probe => AreaPtx {
                sm75: include_str!("ptx/probe.sm75.ptx"),
                sm80: tier_ptx!("cuda-sm80", "ptx/probe.sm80.ptx"),
                sm90: None,
                sm120: None,
            },
            Self::Fbank => AreaPtx::baseline(include_str!("ptx/fbank.sm75.ptx")),
            Self::Embedding => AreaPtx::baseline(include_str!("ptx/embedding.sm75.ptx")),
            Self::Segmentation => AreaPtx::baseline(include_str!("ptx/segmentation.sm75.ptx")),
        }
    }
}

/// The PTX variants of one area: the `sm75` baseline every area ships, plus each
/// higher tier the area ships and this build compiles in
///
/// Every variant exports the same kernels with the same parameters, which
/// `cargo xtask cuda-kernels check` verifies, so the host can load any of them
#[derive(Debug, Clone, Copy)]
pub struct AreaPtx {
    sm75: &'static str,
    sm80: Option<&'static str>,
    sm90: Option<&'static str>,
    sm120: Option<&'static str>,
}

impl AreaPtx {
    const fn baseline(sm75: &'static str) -> Self {
        Self {
            sm75,
            sm80: None,
            sm90: None,
            sm120: None,
        }
    }

    /// The embedded variants, lowest tier first
    pub fn iter(&self) -> impl Iterator<Item = (PtxTier, &'static str)> {
        [
            (PtxTier::Sm75, Some(self.sm75)),
            (PtxTier::Sm80, self.sm80),
            (PtxTier::Sm90, self.sm90),
            (PtxTier::Sm120, self.sm120),
        ]
        .into_iter()
        .filter_map(|(tier, ptx)| Some((tier, ptx?)))
    }

    /// The highest embedded variant at or below `limit`
    pub fn select(&self, limit: PtxTier) -> (PtxTier, &'static str) {
        self.iter()
            .filter(|(tier, _)| *tier <= limit)
            .last()
            .unwrap_or((PtxTier::Sm75, self.sm75))
    }
}

/// A kernel module loaded into a CUDA context
#[derive(Debug, Clone)]
pub struct LoadedKernels {
    module: KernelModule,
    tier: PtxTier,
    inner: Arc<CudaModule>,
}

impl LoadedKernels {
    pub(super) fn new(module: KernelModule, tier: PtxTier, inner: Arc<CudaModule>) -> Self {
        Self {
            module,
            tier,
            inner,
        }
    }

    /// The PTX tier of the variant that was loaded
    pub fn tier(&self) -> PtxTier {
        self.tier
    }

    /// Looks up a kernel by its PTX entry name, which is the Rust function name
    pub fn function(&self, kernel: &str) -> Result<CudaFunction, CudaError> {
        self.inner
            .load_function(kernel)
            .map_err(|source| CudaError::KernelMissing {
                module: self.module.name(),
                kernel: kernel.to_string(),
                source,
            })
    }
}
