use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule};

use super::{ComputeCapability, CudaError, PtxTier};

/// Embeds a variant under exactly the GPU targets for which it is the best shipped PTX
///
/// The kernel check verifies these masks so tier-only builds include each area's
/// fallback without also including variants they do not need
macro_rules! tier_ptx {
    ([$($feature:literal),+], $path:literal) => {{
        #[cfg(any($(feature = $feature),+))]
        let ptx = Some(include_str!($path));
        #[cfg(not(any($(feature = $feature),+)))]
        let ptx = None;
        ptx
    }};
}

/// A cuda-oxide kernel area, embedded as committed PTX
///
/// `cargo xtask cuda-kernels build` regenerates `ptx/<area>.<tier>.ptx` on a GPU box,
/// and `cargo xtask cuda-kernels check` fails when that PTX is stale. Each GPU tier
/// feature embeds the best shipped variant of each area for that target. The `cuda`
/// and `cuda-driver-only` features enable all four targets and embed all variants
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
    /// Candidate kernels for the ResNet convolutions, kept apart from the Library-owned
    /// areas so the harness can tell them apart
    // the candidate areas are unused until a candidate plan loads one
    #[allow(dead_code)]
    Resnet,
    /// Candidate kernels for the LSTM stack
    #[allow(dead_code)]
    Lstm,
    /// Candidate kernels for the Sinc producer
    #[allow(dead_code)]
    Sincnet,
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
            Self::Resnet => "resnet",
            Self::Lstm => "lstm",
            Self::Sincnet => "sincnet",
        }
    }

    /// The PTX variants embedded in this build
    pub const fn variants(self) -> AreaPtx {
        match self {
            #[cfg(test)]
            Self::Probe => AreaPtx {
                sm75: tier_ptx!(["cuda-sm75"], "ptx/probe.sm75.ptx"),
                sm80: tier_ptx!(
                    ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                    "ptx/probe.sm80.ptx"
                ),
                sm90: None,
                sm120: None,
            },
            Self::Fbank => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/fbank.sm75.ptx"
            )),
            Self::Embedding => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/embedding.sm75.ptx"
            )),
            Self::Segmentation => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/segmentation.sm75.ptx"
            )),
            Self::Resnet => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/resnet.sm75.ptx"
            )),
            Self::Lstm => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/lstm.sm75.ptx"
            )),
            Self::Sincnet => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/sincnet.sm75.ptx"
            )),
        }
    }
}

/// The PTX variants of one area embedded by this build
///
/// A tier feature may embed an older variant when that is the best the area ships
///
/// Every variant exports the same kernels with the same parameters, which
/// `cargo xtask cuda-kernels check` verifies, so the host can load any of them
#[derive(Debug, Clone, Copy)]
pub struct AreaPtx {
    sm75: Option<&'static str>,
    sm80: Option<&'static str>,
    sm90: Option<&'static str>,
    sm120: Option<&'static str>,
}

impl AreaPtx {
    const fn baseline(sm75: Option<&'static str>) -> Self {
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
            (PtxTier::Sm75, self.sm75),
            (PtxTier::Sm80, self.sm80),
            (PtxTier::Sm90, self.sm90),
            (PtxTier::Sm120, self.sm120),
        ]
        .into_iter()
        .filter_map(|(tier, ptx)| Some((tier, ptx?)))
    }

    /// The highest embedded variant at or below `limit`
    pub fn select(&self, limit: PtxTier) -> Option<(PtxTier, &'static str)> {
        self.iter().filter(|(tier, _)| *tier <= limit).last()
    }

    /// Resolve a runnable variant or name the target feature that is missing
    pub(super) fn resolve(
        &self,
        area: KernelModule,
        limit: PtxTier,
        device: ComputeCapability,
    ) -> Result<(PtxTier, &'static str), CudaError> {
        self.select(limit)
            .filter(|(tier, _)| tier.min_capability() <= device)
            .ok_or(CudaError::AreaTierNotCompiledIn {
                area: area.name(),
                tier: limit,
                device,
                feature: limit.feature(),
            })
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

#[cfg(test)]
mod tests {
    use super::{AreaPtx, ComputeCapability, CudaError, KernelModule, PtxTier};

    #[test]
    fn gpu_targets_embed_each_areas_best_shipped_variant() {
        let production = KernelModule::Fbank.variants();
        assert_eq!(
            production.iter().map(|(tier, _)| tier).collect::<Vec<_>>(),
            if PtxTier::compiled_in().next().is_some() {
                vec![PtxTier::Sm75]
            } else {
                vec![]
            }
        );
        let probe = KernelModule::Probe.variants();
        assert_eq!(probe.sm75.is_some(), cfg!(feature = "cuda-sm75"));
        assert_eq!(
            probe.sm80.is_some(),
            cfg!(any(
                feature = "cuda-sm80",
                feature = "cuda-sm90",
                feature = "cuda-sm120"
            ))
        );
        assert_eq!(PtxTier::Sm75.is_compiled_in(), cfg!(feature = "cuda-sm75"));
        let selected = probe.select(PtxTier::Sm120).map(|(tier, _)| tier);
        assert_eq!(
            selected,
            if probe.sm80.is_some() {
                Some(PtxTier::Sm80)
            } else if probe.sm75.is_some() {
                Some(PtxTier::Sm75)
            } else {
                None
            }
        );
    }

    #[test]
    fn missing_area_variant_names_area_device_and_target_feature() {
        let area = AreaPtx {
            sm75: None,
            sm80: Some("test PTX"),
            sm90: None,
            sm120: None,
        };
        let device = ComputeCapability::new(7, 5);
        let error = area
            .resolve(KernelModule::Probe, PtxTier::Sm75, device)
            .unwrap_err();
        assert!(matches!(error, CudaError::AreaTierNotCompiledIn {
            area: "probe", tier: PtxTier::Sm75, device: actual, feature: "cuda-sm75"
        } if actual == device));
        assert!(
            error.to_string().contains("probe")
                && error.to_string().contains("7.5")
                && error.to_string().contains("cuda-sm75")
        );
    }
}
