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

impl ArtifactHash {
    /// Byte equality usable in compile-time table validation
    pub(crate) const fn const_eq(self, other: Self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != other.0[index] {
                return false;
            }
            index += 1;
        }
        true
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

/// A complete module identity, resolved before the driver sees any bytes
///
/// Loading, the runtime's module cache and qualification tokens compare whole
/// requests: the area, the PTX tier whose bytes are loaded, and the exact artifact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ModuleRequest {
    area: KernelModule,
    tier: PtxTier,
    artifact: LoadedArtifact,
}

impl ModuleRequest {
    /// A request whose cubin, if any, can run the tier; invalid constants fail to compile
    pub(crate) const fn new(area: KernelModule, tier: PtxTier, artifact: LoadedArtifact) -> Self {
        if let LoadedArtifact::Cubin { arch, .. } = artifact {
            let minimum = tier.min_capability();
            assert!(
                arch.major > minimum.major
                    || arch.major == minimum.major && arch.minor >= minimum.minor,
                "a cubin cannot be older than its PTX tier"
            );
        }
        Self {
            area,
            tier,
            artifact,
        }
    }

    pub(crate) const fn area(self) -> KernelModule {
        self.area
    }

    pub(crate) const fn tier(self) -> PtxTier {
        self.tier
    }

    pub(crate) const fn artifact(self) -> LoadedArtifact {
        self.artifact
    }

    /// Reject a different cached execution identity without replacing its module
    pub(crate) fn check_cached(self, cached: Self) -> Result<(), CudaError> {
        if self != cached {
            return Err(CudaError::ArtifactUnavailable {
                module: self.area.name(),
                artifact: self.artifact,
            });
        }
        Ok(())
    }

    /// The same area and tier, loaded through driver JIT of the tier's embedded PTX
    pub(crate) const fn ptx_jit(self, sha256: ArtifactHash) -> Self {
        Self::new(self.area, self.tier, LoadedArtifact::PtxJit { sha256 })
    }
}

/// Failure to load the requested artifact, without substituting another artifact
#[derive(Debug, PartialEq, Eq)]
pub(super) enum ArtifactLoadError<E> {
    Unavailable,
    Driver(E),
}

/// Load exactly the requested bytes; driver rejection never tries the other format
pub(super) fn load_artifact<T, E>(
    requested: LoadedArtifact,
    cubin: Option<EmbeddedCubin>,
    ptx_sha256: ArtifactHash,
    load_cubin: impl FnOnce(&[u8]) -> Result<T, E>,
    load_ptx: impl FnOnce() -> Result<T, E>,
) -> Result<(T, LoadedArtifact), ArtifactLoadError<E>> {
    let loaded = match requested {
        LoadedArtifact::PtxJit { sha256 } if sha256 == ptx_sha256 => load_ptx(),
        LoadedArtifact::Cubin { arch, sha256 } => {
            let Some(cubin) =
                cubin.filter(|cubin| cubin.arch == arch && ArtifactHash::of(cubin.bytes) == sha256)
            else {
                return Err(ArtifactLoadError::Unavailable);
            };
            load_cubin(cubin.bytes)
        }
        _ => return Err(ArtifactLoadError::Unavailable),
    };
    loaded
        .map(|loaded| (loaded, requested))
        .map_err(ArtifactLoadError::Driver)
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
    /// Record-owned filterbank DFT producer, separate from always-on Fbank
    FbankDft,
    /// Record-owned segmentation dense operators
    Segdense,
    /// Record-owned wide convolution operators
    Wideconv,
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
            Self::FbankDft => "fbankdft",
            Self::Segdense => "segdense",
            Self::Wideconv => "wideconv",
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
            // no candidate artifact exists until the separate kernel port
            Self::FbankDft | Self::Segdense | Self::Wideconv => "",
        }
    }

    /// The PTX variants embedded in this build
    pub const fn variants(self) -> AreaPtx {
        match self {
            Self::FbankDft | Self::Segdense | Self::Wideconv => AreaPtx::baseline(None),
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
    /// Embedded variants without cubins, for host-only tests such as a fake higher tier
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    pub(crate) fn fixture(variants: &[(PtxTier, &'static str)]) -> Self {
        let mut area = Self::baseline(None);
        for (tier, text) in variants {
            let ptx = Some(EmbeddedPtx { text, cubins: &[] });
            match tier {
                PtxTier::Sm75 => area.sm75 = ptx,
                PtxTier::Sm80 => area.sm80 = ptx,
                PtxTier::Sm90 => area.sm90 = ptx,
                PtxTier::Sm120 => area.sm120 = ptx,
            }
        }
        area
    }

    const fn baseline(sm75: Option<EmbeddedPtx>) -> Self {
        Self {
            sm75,
            sm80: None,
            sm90: None,
            sm120: None,
        }
    }

    /// The embedded variants, lowest tier first
    #[cfg(test)]
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
    ///
    /// Production never resolves a tier this way: a binding names its tier. Only tests
    /// and explicit qualification requests ask for the best embedded variant
    #[cfg(test)]
    pub fn select(&self, limit: PtxTier) -> Option<(PtxTier, &'static str)> {
        self.iter().filter(|(tier, _)| *tier <= limit).last()
    }

    /// Resolve a runnable variant or name the target feature that is missing
    #[cfg(test)]
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
    request: ModuleRequest,
    inner: Arc<CudaModule>,
    ptx_sha256: ArtifactHash,
}

impl LoadedKernels {
    pub(super) fn new(
        request: ModuleRequest,
        inner: Arc<CudaModule>,
        ptx_sha256: ArtifactHash,
    ) -> Self {
        Self {
            request,
            inner,
            ptx_sha256,
        }
    }

    /// The complete identity the driver accepted
    pub(crate) fn request(&self) -> ModuleRequest {
        self.request
    }

    /// The PTX tier of the variant that was loaded
    pub fn tier(&self) -> PtxTier {
        self.request.tier
    }

    /// Identity of the artifact that the driver successfully loaded
    #[cfg(test)]
    pub fn artifact(&self) -> LoadedArtifact {
        self.request.artifact
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
                module: self.request.area.name(),
                kernel: kernel.to_string(),
                source,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{AreaPtx, ComputeCapability, CudaError, KernelModule, PtxTier};

    #[test]
    #[ignore = "GPU artifact proof; run under the shared GPU flock"]
    fn driver_artifacts_resolve_every_ptx_entry() -> Result<(), CudaError> {
        use super::{ArtifactHash, LoadedArtifact, LoadedKernels, load_artifact};
        use crate::inference::cuda::CudaRuntime;
        use cudarc::nvrtc::Ptx;
        use std::fs::{OpenOptions, TryLockError};

        // the proof process must be serialized, including context creation and drops
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/workspace/gpu-bench.lock")
            .expect("proof runs under the shared GPU lock");
        assert!(matches!(lock.try_lock(), Err(TryLockError::WouldBlock)));
        let runtime = CudaRuntime::new(0)?;
        let device = runtime.compute_capability();
        let force_jit =
            std::env::var_os(super::FORCE_PTX_JIT_ENV).is_some_and(|value| value == "1");
        for area in [
            KernelModule::Probe,
            KernelModule::Fbank,
            KernelModule::Embedding,
            KernelModule::Segmentation,
            KernelModule::Resnet,
            KernelModule::Lstm,
            KernelModule::Sincnet,
        ] {
            for (tier, ptx) in area.variants().iter() {
                if tier.min_capability() > device {
                    continue;
                }
                let embedded = area.variants().embedded(tier).expect("embedded variant");
                let cubin = embedded
                    .cubin(device)
                    .expect("proof device has exact cubins");
                let cubin_key = LoadedArtifact::Cubin {
                    arch: device,
                    sha256: ArtifactHash::of(cubin.bytes),
                };
                let ptx_hash = ArtifactHash::of(ptx.as_bytes());
                let (module, artifact) = load_artifact(
                    if force_jit {
                        LoadedArtifact::PtxJit { sha256: ptx_hash }
                    } else {
                        cubin_key
                    },
                    Some(cubin),
                    ptx_hash,
                    |bytes| {
                        runtime
                            .context()
                            .load_module(Ptx::from_binary(bytes.to_vec()))
                    },
                    || runtime.context().load_module(Ptx::from_src(ptx)),
                )
                .map_err(|error| match error {
                    super::ArtifactLoadError::Driver(source) => CudaError::Driver(source),
                    super::ArtifactLoadError::Unavailable => CudaError::ArtifactUnavailable {
                        module: area.name(),
                        artifact: cubin_key,
                    },
                })?;
                if force_jit {
                    assert_eq!(artifact, LoadedArtifact::PtxJit { sha256: ptx_hash });
                    assert_ne!(artifact, cubin_key);
                } else {
                    // this proof must fail, not silently qualify a rejected cubin's JIT
                    assert_eq!(artifact, cubin_key);
                }
                let loaded = LoadedKernels::new(
                    super::ModuleRequest::new(area, tier, artifact),
                    module,
                    ptx_hash,
                );
                let mut entries = 0;
                for line in ptx.lines() {
                    let Some(entry) = line.trim().strip_prefix(".visible .entry ") else {
                        continue;
                    };
                    let name = entry.split(['(', ' ', '\t']).next().expect("entry name");
                    loaded.function(name)?;
                    entries += 1;
                }
                assert!(entries > 0, "proof requires declared entry points");
                println!(
                    "artifact_proof area={} tier={tier} device={device} artifact={artifact:?} embedded_ptx_sha256={ptx_hash} entries={entries}",
                    area.name()
                );
            }
            let loaded = runtime.load_kernels(area)?;
            assert_eq!(loaded.artifact(), runtime.load_kernels(area)?.artifact());
        }
        runtime.synchronize()
    }

    #[test]
    fn pinned_loading_never_substitutes_artifacts() {
        use super::{
            ArtifactHash, ArtifactLoadError, EmbeddedCubin, LoadedArtifact, load_artifact,
        };
        let hash = ArtifactHash::of(b"ptx");
        let jit = LoadedArtifact::PtxJit { sha256: hash };
        for device in [ComputeCapability::new(8, 9), ComputeCapability::new(12, 0)] {
            let cubin = EmbeddedCubin {
                arch: device,
                bytes: b"cubin",
            };
            let binary = LoadedArtifact::Cubin {
                arch: device,
                sha256: ArtifactHash::of(cubin.bytes),
            };
            let result = load_artifact(
                binary,
                Some(cubin),
                hash,
                |bytes| Ok::<_, &str>(bytes.to_vec()),
                || panic!("cubin request cannot JIT"),
            )
            .unwrap();
            assert_eq!(result, (cubin.bytes.to_vec(), binary));
            assert_eq!(
                load_artifact::<(), _>(
                    binary,
                    Some(cubin),
                    hash,
                    |_| Err("driver refusal"),
                    || panic!("refused cubin must not JIT")
                ),
                Err(ArtifactLoadError::Driver("driver refusal"))
            );
            assert_eq!(
                load_artifact::<(), &str>(
                    binary,
                    None,
                    hash,
                    |_| panic!("missing cubin"),
                    || panic!("missing cubin must not JIT")
                ),
                Err(ArtifactLoadError::Unavailable)
            );
            assert_eq!(
                load_artifact(
                    jit,
                    Some(cubin),
                    hash,
                    |_| panic!("PTX request cannot load cubin"),
                    || Ok::<_, &str>("jit")
                ),
                Ok(("jit", jit))
            );
            assert_eq!(
                load_artifact::<(), _>(
                    jit,
                    Some(cubin),
                    hash,
                    |_| panic!("refused PTX cannot load cubin"),
                    || Err("PTX rejected")
                ),
                Err(ArtifactLoadError::Driver("PTX rejected"))
            );
        }
        let wrong = LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(b"stale ptx"),
        };
        assert_eq!(
            load_artifact::<(), &str>(
                wrong,
                None,
                hash,
                |_| panic!("wrong hash"),
                || panic!("wrong hash")
            ),
            Err(ArtifactLoadError::Unavailable)
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
