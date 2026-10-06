use std::sync::Arc;

use sha2::{Digest, Sha256};

use cudarc::driver::{CudaFunction, CudaModule};

use super::{ComputeCapability, CudaError, PtxTier};

/// Embeds a variant under exactly the GPU targets for which it is the best shipped PTX
///
/// The kernel check verifies these masks so tier-only builds include each area's
/// fallback without also including variants they do not need
macro_rules! tier_ptx {
    ([$($feature:literal),+], $stem:literal, [$($arch:literal),+]) => {{
        #[cfg(any($(feature = $feature),+))]
        const CUBINS: &[EmbeddedCubin] = &[$(EmbeddedCubin {
            arch: ComputeCapability::new($arch / 10, $arch % 10),
            bytes: include_bytes!(concat!($stem, ".sm_", stringify!($arch), ".cubin")),
        }),+];
        #[cfg(any($(feature = $feature),+))]
        let ptx = Some(EmbeddedPtx {
            text: include_str!(concat!($stem, ".ptx")),
            cubins: CUBINS,
        });
        #[cfg(not(any($(feature = $feature),+)))]
        let ptx = None;
        ptx
    }};
}

/// Set to `1` to disable cubins and use embedded PTX JIT for diagnosis
pub const FORCE_PTX_JIT_ENV: &str = "SPEAKRS_CUDA_FORCE_PTX_JIT";

/// Content identity of bytes embedded in the running binary
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArtifactHash([u8; 32]);

impl ArtifactHash {
    /// Hash the actual artifact bytes, not a source-tree file or manifest claim
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    /// Read a canonical pinned hash; invalid pins fail at compile time in constants
    pub const fn from_hex(text: &str) -> Self {
        const fn digit(byte: u8) -> u8 {
            match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("hash must use lowercase hexadecimal"),
            }
        }
        assert!(
            text.len() == 64,
            "hash must contain 64 hexadecimal characters"
        );
        let mut bytes = [0; 32];
        let mut index = 0;
        while index < bytes.len() {
            bytes[index] =
                digit(text.as_bytes()[index * 2]) * 16 + digit(text.as_bytes()[index * 2 + 1]);
            index += 1;
        }
        Self(bytes)
    }
}

impl std::fmt::Display for ArtifactHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// The exact artifact accepted by the driver for a kernel area
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadedArtifact {
    /// Ready SASS compiled for this exact device capability
    Cubin {
        /// Exact cubin architecture, never an older compatible architecture
        arch: ComputeCapability,
        /// Hash of the binary bytes handed to the driver
        sha256: ArtifactHash,
    },
    /// Embedded PTX compiled by the device driver
    PtxJit {
        /// Hash of the PTX text handed to the driver
        sha256: ArtifactHash,
    },
}

/// One embedded binary, available only alongside its source PTX variant
#[derive(Debug, Clone, Copy)]
pub(super) struct EmbeddedCubin {
    pub arch: ComputeCapability,
    pub bytes: &'static [u8],
}

/// The PTX fallback and exact-architecture binaries for one tier
#[derive(Debug, Clone, Copy)]
pub(super) struct EmbeddedPtx {
    pub text: &'static str,
    pub cubins: &'static [EmbeddedCubin],
}

impl EmbeddedPtx {
    pub fn cubin(self, device: ComputeCapability) -> Option<EmbeddedCubin> {
        self.cubins
            .iter()
            .copied()
            .find(|cubin| cubin.arch == device)
    }
}

/// Load one exact candidate binary, then PTX if the driver refuses it
pub(super) fn load_artifact<T, E>(
    cubin: Option<EmbeddedCubin>,
    ptx_sha256: ArtifactHash,
    load_cubin: impl FnOnce(&[u8]) -> Result<T, E>,
    load_ptx: impl FnOnce() -> Result<T, E>,
) -> Result<(T, LoadedArtifact), E> {
    if let Some(cubin) = cubin
        && let Ok(loaded) = load_cubin(cubin.bytes)
    {
        return Ok((
            loaded,
            LoadedArtifact::Cubin {
                arch: cubin.arch,
                sha256: ArtifactHash::of(cubin.bytes),
            },
        ));
    }
    load_ptx().map(|loaded| (loaded, LoadedArtifact::PtxJit { sha256: ptx_sha256 }))
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

    /// The build metadata embedded beside this area's artifact bytes
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    pub(crate) const fn manifest(self) -> &'static str {
        match self {
            #[cfg(test)]
            Self::Probe => include_str!("ptx/probe.manifest"),
            Self::Fbank => include_str!("ptx/fbank.manifest"),
            Self::Embedding => include_str!("ptx/embedding.manifest"),
            Self::Segmentation => include_str!("ptx/segmentation.manifest"),
            Self::Resnet => include_str!("ptx/resnet.manifest"),
            Self::Lstm => include_str!("ptx/lstm.manifest"),
            Self::Sincnet => include_str!("ptx/sincnet.manifest"),
        }
    }

    /// The PTX variants embedded in this build
    pub const fn variants(self) -> AreaPtx {
        match self {
            #[cfg(test)]
            Self::Probe => AreaPtx {
                sm75: tier_ptx!(["cuda-sm75"], "ptx/probe.sm75", [75, 80, 86, 89, 90, 120]),
                sm80: tier_ptx!(
                    ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                    "ptx/probe.sm80",
                    [80, 86, 89, 90, 120]
                ),
                sm90: None,
                sm120: None,
            },
            Self::Fbank => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/fbank.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::Embedding => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/embedding.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::Segmentation => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/segmentation.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::Resnet => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/resnet.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::Lstm => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/lstm.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::Sincnet => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/sincnet.sm75",
                [75, 80, 86, 89, 90, 120]
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
    sm75: Option<EmbeddedPtx>,
    sm80: Option<EmbeddedPtx>,
    sm90: Option<EmbeddedPtx>,
    sm120: Option<EmbeddedPtx>,
}

impl AreaPtx {
    const fn baseline(sm75: Option<EmbeddedPtx>) -> Self {
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
        .filter_map(|(tier, ptx)| Some((tier, ptx?.text)))
    }

    /// The embedded bytes for one specific tier
    pub(super) fn embedded(&self, tier: PtxTier) -> Option<EmbeddedPtx> {
        match tier {
            PtxTier::Sm75 => self.sm75,
            PtxTier::Sm80 => self.sm80,
            PtxTier::Sm90 => self.sm90,
            PtxTier::Sm120 => self.sm120,
        }
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
    artifact: LoadedArtifact,
    ptx_sha256: ArtifactHash,
}

impl LoadedKernels {
    pub(super) fn new(
        module: KernelModule,
        tier: PtxTier,
        inner: Arc<CudaModule>,
        artifact: LoadedArtifact,
        ptx_sha256: ArtifactHash,
    ) -> Self {
        Self {
            module,
            tier,
            inner,
            artifact,
            ptx_sha256,
        }
    }

    /// The PTX tier of the variant that was loaded
    pub fn tier(&self) -> PtxTier {
        self.tier
    }

    /// Identity of the artifact that the driver successfully loaded
    pub fn artifact(&self) -> LoadedArtifact {
        self.artifact
    }

    /// Identity of the fallback PTX text embedded alongside the loaded artifact
    pub fn ptx_sha256(&self) -> ArtifactHash {
        self.ptx_sha256
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
    fn binary_loading_uses_only_the_exact_architecture_and_keeps_jit_identity() {
        use super::{ArtifactHash, EmbeddedCubin, EmbeddedPtx, LoadedArtifact, load_artifact};
        let cubins = [
            EmbeddedCubin {
                arch: ComputeCapability::new(8, 9),
                bytes: b"cubin89",
            },
            EmbeddedCubin {
                arch: ComputeCapability::new(12, 0),
                bytes: b"cubin120",
            },
        ];
        let ptx = EmbeddedPtx {
            text: "ptx",
            cubins: &[],
        };
        let hash = ArtifactHash::of(ptx.text.as_bytes());
        for device in [ComputeCapability::new(8, 9), ComputeCapability::new(12, 0)] {
            let cubin = cubins
                .iter()
                .copied()
                .find(|cubin| cubin.arch == device)
                .unwrap();
            let (loaded, artifact) = load_artifact(
                Some(cubin),
                hash,
                |bytes| Ok::<_, ()>(bytes.to_vec()),
                || panic!("accepted cubin must not JIT"),
            )
            .unwrap();
            assert_eq!(loaded, cubin.bytes);
            assert_eq!(
                artifact,
                LoadedArtifact::Cubin {
                    arch: device,
                    sha256: ArtifactHash::of(cubin.bytes)
                }
            );
            let (loaded, artifact) = load_artifact(
                Some(cubin),
                hash,
                |_| Err("driver refusal"),
                || Ok::<_, &str>("jit"),
            )
            .unwrap();
            assert_eq!(loaded, "jit");
            assert_eq!(artifact, LoadedArtifact::PtxJit { sha256: hash });
        }
        let (loaded, artifact) = load_artifact(
            None,
            hash,
            |_| panic!("no binary candidate"),
            || Ok::<_, ()>("jit"),
        )
        .unwrap();
        assert_eq!(loaded, "jit");
        assert_eq!(artifact, LoadedArtifact::PtxJit { sha256: hash });
        assert_eq!(
            load_artifact::<(), _>(None, hash, |_| unreachable!(), || Err("PTX rejected")),
            Err("PTX rejected")
        );
    }

    #[test]
    fn unknown_device_never_uses_an_older_cubin() {
        let Some(tier) = KernelModule::Fbank
            .variants()
            .iter()
            .next()
            .map(|(tier, _)| tier)
        else {
            return;
        };
        let embedded = KernelModule::Fbank.variants().embedded(tier).unwrap();
        for device in [ComputeCapability::new(8, 9), ComputeCapability::new(12, 0)] {
            assert_eq!(embedded.cubin(device).unwrap().arch, device);
        }
        assert!(embedded.cubin(ComputeCapability::new(12, 1)).is_none());
        assert!(embedded.cubin(ComputeCapability::new(10, 0)).is_none());
    }

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
            sm80: Some(super::EmbeddedPtx {
                text: "test PTX",
                cubins: &[],
            }),
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
