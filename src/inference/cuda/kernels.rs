use cudakit::{KernelArea, Profile};

// identity types speakrs's plans and records pin, kept under the kernel registry
use cudakit::PtxTier;
pub(crate) use cudakit::{AreaPtx, ArtifactHash, LoadedArtifact};

/// A cuda-oxide area's loaded module
pub type LoadedKernels = cudakit::LoadedKernels<KernelModule>;

/// The exact identity of a speakrs kernel module
pub(crate) type ModuleRequest = cudakit::ModuleRequest<KernelModule>;

/// Embeds a variant under exactly the GPU targets for which it is the best shipped PTX
///
/// The kernel check verifies these masks so tier-only builds include each area's
/// fallback without also including variants they do not need
macro_rules! tier_ptx {
    ($($variant:tt)*) => {
        cudakit::embed_tier!(shipped_ptx, shipped_cubin; $($variant)*)
    };
}

/// A shipped PTX file, from the `speakrs-ptx-<tier>` crate that carries its tier
///
/// Each tier ships in its own crate so every published crate stays under the crates.io
/// upload limit. Call this only in const context outside tests: a run-time call would
/// link every file of every enabled tier crate into the binary
#[cfg(any(
    feature = "cuda-sm75",
    feature = "cuda-sm80",
    feature = "cuda-sm90",
    feature = "cuda-sm120"
))]
pub(crate) const fn shipped_ptx(path: &str) -> &'static str {
    if let Some(text) = speakrs_ptx_sm75::ptx(path) {
        return text;
    }

    #[cfg(any(feature = "cuda-sm80", feature = "cuda-sm90", feature = "cuda-sm120"))]
    if let Some(text) = speakrs_ptx_sm80::ptx(path) {
        return text;
    }

    #[cfg(feature = "cuda-sm120")]
    if let Some(text) = speakrs_ptx_sm120::ptx(path) {
        return text;
    }

    panic!("no enabled speakrs-ptx crate ships this PTX file")
}

/// A shipped cubin, from the crate that carries its PTX tier; const context only, like
/// [`shipped_ptx`]
#[cfg(any(
    feature = "cuda-sm75",
    feature = "cuda-sm80",
    feature = "cuda-sm90",
    feature = "cuda-sm120"
))]
const fn shipped_cubin(path: &str) -> &'static [u8] {
    if let Some(bytes) = speakrs_ptx_sm75::cubin(path) {
        return bytes;
    }

    #[cfg(any(feature = "cuda-sm80", feature = "cuda-sm90", feature = "cuda-sm120"))]
    if let Some(bytes) = speakrs_ptx_sm80::cubin(path) {
        return bytes;
    }

    #[cfg(feature = "cuda-sm120")]
    if let Some(bytes) = speakrs_ptx_sm120::cubin(path) {
        return bytes;
    }

    panic!("no enabled speakrs-ptx crate ships this cubin")
}

/// Set to `1` to disable cubins and use embedded PTX JIT for diagnosis
pub const FORCE_PTX_JIT_ENV: &str = "SPEAKRS_CUDA_FORCE_PTX_JIT";

/// A cuda-oxide kernel area, embedded as committed PTX
///
/// `cargo xtask cuda-kernels build` regenerates `ptx/<area>.<tier>.ptx` in the
/// `speakrs-ptx-<tier>` crate on a GPU box,
/// and `cargo xtask cuda-kernels check` fails when that PTX is stale. Each GPU tier
/// feature embeds the best shipped variant of each area for that target. The `cuda`
/// feature enables all four targets and embeds all shipped variants
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
    /// areas so module identities cannot overlap
    // the candidate areas are unused until a candidate plan loads one
    #[allow(dead_code)]
    Resnet,
    /// Candidate kernels for the LSTM stack
    #[allow(dead_code)]
    Lstm,
    /// Candidate kernels for the Sinc producer
    #[allow(dead_code)]
    Sincnet,
    /// The LSTM stack with its own input projections, which needs no library
    LstmProj,
    /// Record-owned filterbank DFT producer, separate from always-on Fbank
    FbankDft,
    /// Record-owned segmentation dense operators
    Segdense,
    /// Wide trunk convolutions: the stem, the 1x1 shortcuts, the strided layers from
    /// 64 channels and the 128- and 256-channel layers
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
            Self::LstmProj => "lstmproj",
            Self::FbankDft => "fbankdft",
            Self::Segdense => "segdense",
            Self::Wideconv => "wideconv",
        }
    }

    /// The build metadata embedded beside this area's artifact bytes
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
            Self::Segdense => include_str!("ptx/segdense.manifest"),
            Self::FbankDft => include_str!("ptx/fbankdft.manifest"),
            Self::Wideconv => include_str!("ptx/wideconv.manifest"),
            Self::LstmProj => include_str!("ptx/lstmproj.manifest"),
        }
    }

    /// The PTX variants embedded in this build
    // without calls rustc treats this as inlinable and gives every calling codegen unit
    // a local copy, each with its own copy of every embedded PTX and cubin
    #[inline(never)]
    pub const fn variants(self) -> AreaPtx {
        match self {
            #[cfg(test)]
            Self::Probe => AreaPtx::EMPTY
                .with(
                    PtxTier::Sm75,
                    tier_ptx!(["cuda-sm75"], "ptx/probe.sm75", [75, 80, 86, 89, 90, 120]),
                )
                .with(
                    PtxTier::Sm80,
                    tier_ptx!(
                        ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                        "ptx/probe.sm80",
                        [80, 86, 89, 90, 120]
                    ),
                ),
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
            // the capability 12.0 binding pins the sm75 bytes, so every target embeds
            // them; the sm80 tier adds the TF32 tensor-core entries
            Self::Resnet => AreaPtx::EMPTY
                .with(
                    PtxTier::Sm75,
                    tier_ptx!(
                        ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                        "ptx/resnet.sm75",
                        [75, 80, 86, 89, 90, 120]
                    ),
                )
                .with(
                    PtxTier::Sm80,
                    tier_ptx!(
                        ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                        "ptx/resnet.sm80",
                        [80, 86, 89, 90, 120]
                    ),
                ),
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
            // the sm80 tier adds tensor-core kernels; on cc 8.0 and newer it replaces
            // the sm75 variant, whose tensor-core entries are trapping stubs
            Self::Segdense => AreaPtx::EMPTY
                .with(
                    PtxTier::Sm75,
                    tier_ptx!(
                        ["cuda-sm75"],
                        "ptx/segdense.sm75",
                        [75, 80, 86, 89, 90, 120]
                    ),
                )
                .with(
                    PtxTier::Sm80,
                    tier_ptx!(
                        ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                        "ptx/segdense.sm80",
                        [80, 86, 89, 90, 120]
                    ),
                ),
            Self::FbankDft => AreaPtx::baseline(tier_ptx!(
                ["cuda-sm75", "cuda-sm80", "cuda-sm90", "cuda-sm120"],
                "ptx/fbankdft.sm75",
                [75, 80, 86, 89, 90, 120]
            )),
            Self::LstmProj => AreaPtx::EMPTY
                .with(
                    PtxTier::Sm75,
                    tier_ptx!(
                        ["cuda-sm75"],
                        "ptx/lstmproj.sm75",
                        [75, 80, 86, 89, 90, 120]
                    ),
                )
                .with(
                    PtxTier::Sm80,
                    tier_ptx!(
                        ["cuda-sm80", "cuda-sm90"],
                        "ptx/lstmproj.sm80",
                        [80, 86, 89, 90, 120]
                    ),
                )
                .with(
                    PtxTier::Sm120,
                    tier_ptx!(["cuda-sm120"], "ptx/lstmproj.sm120", [120]),
                ),
            Self::Wideconv => AreaPtx::EMPTY
                .with(
                    PtxTier::Sm75,
                    tier_ptx!(
                        ["cuda-sm75"],
                        "ptx/wideconv.sm75",
                        [75, 80, 86, 89, 90, 120]
                    ),
                )
                // other cc 12.x devices keep this variant in sm120 builds; only the
                // RTX 5090 binds the sm120 one
                .with(
                    PtxTier::Sm80,
                    tier_ptx!(
                        ["cuda-sm80", "cuda-sm90", "cuda-sm120"],
                        "ptx/wideconv.sm80",
                        [80, 86, 89, 90, 120]
                    ),
                )
                .with(
                    PtxTier::Sm120,
                    tier_ptx!(["cuda-sm120"], "ptx/wideconv.sm120", [120]),
                ),
        }
    }
}

impl KernelArea for KernelModule {
    const PROFILE: &'static Profile = &super::tier::PROFILE;

    fn name(self) -> &'static str {
        KernelModule::name(self)
    }

    fn variants(self) -> AreaPtx {
        KernelModule::variants(self)
    }
}

#[cfg(test)]
mod tests {
    use super::KernelModule;
    use crate::inference::cuda::tier::test_support::BuildTiers;
    use crate::inference::cuda::{ComputeCapability, CudaError, PtxTier};

    #[test]
    #[ignore = "GPU artifact proof; run under the shared GPU flock"]
    fn driver_artifacts_resolve_every_ptx_entry() -> Result<(), CudaError> {
        use cudakit::{ArtifactHash, ArtifactLoadError, LoadedArtifact, load_artifact};
        use cudarc::nvrtc::Ptx;

        use super::{LoadedKernels, ModuleRequest};
        use crate::inference::cuda::CudaRuntime;
        use std::fs::{OpenOptions, TryLockError};

        // the proof process must be serialized, including context creation and drops;
        // the lock path follows prove-driver-only.sh
        let path = std::env::var_os("SPEAKRS_GPU_LOCK")
            .unwrap_or_else(|| "/workspace/gpu-bench.lock".into());
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
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
            KernelModule::Segdense,
            KernelModule::FbankDft,
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
                    ArtifactLoadError::Driver(source) => CudaError::Driver(source),
                    ArtifactLoadError::Unavailable => CudaError::ArtifactUnavailable {
                        module: area.name(),
                        artifact: cubin_key,
                    },
                })?;
                if force_jit {
                    assert_eq!(artifact, LoadedArtifact::PtxJit { sha256: ptx_hash });
                    assert_ne!(artifact, cubin_key);
                } else {
                    // this proof must fail, not silently accept a rejected cubin's JIT
                    assert_eq!(artifact, cubin_key);
                }
                let loaded = LoadedKernels::from_module(
                    runtime.context().clone(),
                    ModuleRequest::new(area, tier, artifact),
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
            // a record-owned area without a binding on this device has no production load
            if runtime.production_module(area)?.is_some() {
                let loaded = runtime.load_kernels(area)?;
                assert_eq!(loaded.artifact(), runtime.load_kernels(area)?.artifact());
            }
        }
        runtime.synchronize()
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
        assert_eq!(
            probe.embedded(PtxTier::Sm75).is_some(),
            cfg!(feature = "cuda-sm75")
        );
        assert_eq!(
            probe.embedded(PtxTier::Sm80).is_some(),
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
            if probe.embedded(PtxTier::Sm80).is_some() {
                Some(PtxTier::Sm80)
            } else if probe.embedded(PtxTier::Sm75).is_some() {
                Some(PtxTier::Sm75)
            } else {
                None
            }
        );
    }
}
