use std::ops::Deref;

use super::CudaMath;
#[cfg(feature = "_cuda-libraries")]
use super::error::CudaLibrary;
use super::implementation::BoundaryId;
use super::kernels::ModuleRequest;
use super::tier::TierPolicy;
use super::tuning::{ApprovedChoice, BenchKind, BoundaryGraph, TuneControl};
use super::{CudaError, KernelModule, LoadedKernels, PtxTier};

mod tune_hint;

/// One CUDA device context and stream with speakrs's kernel areas, plus the model-load
/// policy: forced Library routing, recipe mode and tuning state
///
/// Everything issued through one runtime runs in order on its stream. Use one runtime
/// per worker thread; device buffers from one runtime must not be used on another
/// runtime's stream without explicit synchronization
///
/// Each kernel area loads the module request its production policy names, never a
/// tier inferred from what is embedded. Dereferences to cudakit's runtime for the
/// context, stream, device attributes, library handles, GEMM and graph capture
#[derive(Debug)]
pub struct CudaRuntime {
    kit: cudakit::Runtime<KernelModule>,
    force_library: bool,
    recipe_mode: super::implementation::policy::RecipeMode,
    tuning: Option<TuneControl>,
}

impl CudaRuntime {
    /// Opens device `ordinal` and creates a stream without loading optional libraries
    ///
    /// Fails with [`CudaError::LibraryUnavailable`] instead of panicking when a CUDA
    /// shared library is missing. [`PTX_TIER_ENV`](super::tier::PTX_TIER_ENV) may force
    /// a lower compiled-in PTX tier; see [`Self::with_ptx_tier`]
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        Self::with_ptx_tier(ordinal, PtxTier::from_env()?)
    }

    /// Like [`Self::new`], with an explicit PTX tier instead of reading the environment
    ///
    /// `None` picks the highest compiled-in tier the device supports. Fails with
    /// [`CudaError::UnsupportedDevice`] below the `sm_75` baseline, and when
    /// `requested` is not compiled in or is above what the device supports
    pub fn with_ptx_tier(ordinal: usize, requested: Option<PtxTier>) -> Result<Self, CudaError> {
        Self::open(ordinal, requested, true)
    }

    fn open(
        ordinal: usize,
        requested: Option<PtxTier>,
        read_tune_file: bool,
    ) -> Result<Self, CudaError> {
        let kit = cudakit::Runtime::open(
            ordinal,
            cudakit::Options {
                tier: requested,
                force_ptx_jit: force_ptx_jit_from_env(),
            },
        )?;
        let mut runtime = Self {
            kit,
            force_library: !super::driver_only() && force_library_from_env(),
            recipe_mode: super::implementation::policy::RecipeMode::Disabled,
            tuning: None,
        };
        if read_tune_file {
            runtime.tuning = TuneControl::load(&runtime);
        }
        Ok(runtime)
    }

    /// Suggest explicit tuning only when this model has no measured device choices
    pub(crate) fn log_tune_hint(&self) {
        tune_hint::log(self.device(), self.ptx_tier(), self.tuning.is_some());
    }

    /// The tuner never reads old measurements or inherits the force-Library policy
    pub(crate) fn for_tuning(
        ordinal: usize,
        kind: BenchKind,
        include_library: bool,
    ) -> Result<Self, CudaError> {
        if force_ptx_jit_from_env() {
            return Err(super::tuning::invalid(
                "unset SPEAKRS_CUDA_FORCE_PTX_JIT before tuning",
            ));
        }
        let mut runtime = Self::open(ordinal, PtxTier::from_env()?, false)?;
        runtime.force_library = false;
        runtime.tuning = Some(TuneControl::benchmark(
            kind,
            include_library,
            runtime.device(),
            runtime.ptx_tier(),
        )?);
        Ok(runtime)
    }

    pub(crate) fn tuned_choice(
        &self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
    ) -> Option<ApprovedChoice> {
        self.tuning
            .as_ref()
            .and_then(|tuning| tuning.plan_choice(boundary, batch, math))
    }

    pub(crate) fn tuning_library_versions(
        &self,
    ) -> Result<super::tuning::LibraryVersions, CudaError> {
        #[cfg(feature = "_cuda-libraries")]
        {
            let versions = self.kit.library_versions()?;
            Ok(super::tuning::LibraryVersions::Hybrid {
                cudnn: versions.cudnn,
                cublas: versions.cublas,
            })
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            Err(super::tuning::invalid(
                "Library tuning requires the cuda feature",
            ))
        }
    }

    pub(crate) fn is_tuning(&self) -> bool {
        self.tuning.as_ref().is_some_and(TuneControl::is_benchmark)
    }

    pub(crate) fn begin_tune_capture(&self) -> Result<(), CudaError> {
        self.tuning
            .as_ref()
            .ok_or_else(|| super::tuning::invalid("no explicit tuner"))?
            .begin_capture()
    }

    pub(crate) fn take_tune_graphs(&self) -> Result<Vec<BoundaryGraph>, CudaError> {
        self.tuning
            .as_ref()
            .ok_or_else(|| super::tuning::invalid("no explicit tuner"))?
            .take_graphs()
    }

    /// Normal forward execution performs no capture, allocation or tuning work
    pub(crate) fn record_boundary(
        &self,
        boundary: BoundaryId,
        batch: usize,
        math: CudaMath,
        run: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        match &self.tuning {
            Some(tuning) => tuning.record(self, boundary, batch, math, run),
            None => run(),
        }
    }

    /// Bind whole-pipeline precision before any model plans are built
    pub(crate) fn with_recipe_mode(
        mut self,
        mode: super::implementation::policy::RecipeMode,
    ) -> Self {
        self.recipe_mode = mode;
        self
    }

    /// Whole-plan recipe eligibility, fixed for this model session
    pub(crate) fn recipe_mode(&self) -> super::implementation::policy::RecipeMode {
        self.recipe_mode
    }

    /// The model-load policy snapshot, shared by all boundary plans
    pub(crate) fn force_library(&self) -> bool {
        self.force_library
    }

    /// Prepare a library during construction, never during forward or capture
    #[cfg(feature = "_cuda-libraries")]
    pub(super) fn prepare_library(&self, library: CudaLibrary) -> Result<(), CudaError> {
        Ok(self.kit.prepare(library)?)
    }

    /// The module production loads for `module` on this device, resolved before any
    /// load; bindings take precedence over the best runnable embedded artifact
    pub(crate) fn production_module(
        &self,
        module: KernelModule,
    ) -> Result<Option<ModuleRequest>, CudaError> {
        super::implementation::production_module(
            module,
            self.device(),
            self.ptx_tier(),
            module.variants(),
        )?
        .map(|request| self.effective_request(request))
        .transpose()
    }

    /// Resolve the diagnostic artifact before selection fixes a token or cache identity
    pub(crate) fn effective_request(
        &self,
        request: ModuleRequest,
    ) -> Result<ModuleRequest, CudaError> {
        Ok(self.kit.effective_request(request)?)
    }

    /// The same area artifact for explicit and production plans, so their cache
    /// identities cannot conflict
    #[cfg(all(test, feature = "_cuda-libraries"))]
    pub(crate) fn embedded_exact_request(
        &self,
        module: KernelModule,
    ) -> Result<ModuleRequest, CudaError> {
        self.production_module(module)?
            .ok_or(CudaError::AreaTierNotCompiledIn {
                area: module.name(),
                tier: self.ptx_tier(),
                device: self.compute_capability(),
                feature: self.ptx_tier().feature(),
            })
    }

    /// The area's production module, reusing only a cache entry with the same identity
    ///
    /// Plan selection skips uncovered areas before a candidate plan calls this. A
    /// development load cannot change what a later production request loads
    pub fn load_kernels(&self, module: KernelModule) -> Result<LoadedKernels, CudaError> {
        let request = self
            .production_module(module)?
            .ok_or(CudaError::TierNotQualified {
                tier: PtxTier::BASELINE,
                device: self.compute_capability(),
            })?;
        self.load_module(request)
    }

    /// Load exactly `request`; a cached module with another identity is an error, never
    /// a replacement, and a driver refusal never tries the other artifact format
    pub(crate) fn load_module(&self, request: ModuleRequest) -> Result<LoadedKernels, CudaError> {
        Ok(self.kit.load(request)?)
    }

    /// Blocks until all work queued on the runtime's stream has finished
    pub fn synchronize(&self) -> Result<(), CudaError> {
        Ok(self.kit.synchronize()?)
    }
}

impl Deref for CudaRuntime {
    type Target = cudakit::Runtime<KernelModule>;

    fn deref(&self) -> &Self::Target {
        &self.kit
    }
}

/// Read once per runtime so graph capture and execution do not consult the environment
pub(super) fn force_library_from_env() -> bool {
    std::env::var_os("SPEAKRS_CUDA_FORCE_LIBRARY").is_some_and(|value| value == "1")
}

fn force_ptx_jit_from_env() -> bool {
    std::env::var_os(super::kernels::FORCE_PTX_JIT_ENV).is_some_and(|value| value == "1")
}

#[cfg(test)]
mod direct_request_tests {
    use super::super::error::CudaLibrary;
    use super::CudaError;

    #[test]
    fn forbidden_direct_requests_do_not_invent_model_selection_context() {
        for library in [CudaLibrary::Cublas, CudaLibrary::Cudnn, CudaLibrary::Nvrtc] {
            let error = CudaError::from(cudakit::Error::LibraryForbidden { library });
            assert!(matches!(
                error,
                CudaError::LibraryForbidden { library: requested } if requested == library
            ));
            let message = error.to_string();
            assert!(message.contains(&library.to_string()));
            for invented in ["runtime/", "b1", "Fp32", "PTX tier"] {
                assert!(!message.contains(invented));
            }
        }
    }
}

#[cfg(all(test, feature = "_cuda-libraries"))]
mod module_policy_tests {
    use cudakit::{ArtifactHash, LoadedArtifact};

    use super::{CudaError, KernelModule, ModuleRequest, PtxTier};
    use crate::inference::cuda::ComputeCapability;

    #[test]
    fn cached_module_must_match_every_part_of_the_production_request() {
        let artifact = LoadedArtifact::PtxJit {
            sha256: ArtifactHash::of(b"qualified"),
        };
        let cached = ModuleRequest::new(KernelModule::Lstm, PtxTier::Sm75, artifact);
        cached.check_cached(cached).unwrap();
        for request in [
            ModuleRequest::new(KernelModule::Resnet, PtxTier::Sm75, artifact),
            ModuleRequest::new(KernelModule::Lstm, PtxTier::Sm80, artifact),
            ModuleRequest::new(
                KernelModule::Lstm,
                PtxTier::Sm75,
                LoadedArtifact::PtxJit {
                    sha256: ArtifactHash::of(b"different"),
                },
            ),
            ModuleRequest::new(
                KernelModule::Lstm,
                PtxTier::Sm75,
                LoadedArtifact::Cubin {
                    arch: ComputeCapability::new(12, 0),
                    sha256: ArtifactHash::of(b"qualified"),
                },
            ),
        ] {
            assert!(
                matches!(request.check_cached(cached).map_err(CudaError::from),
                Err(CudaError::ArtifactUnavailable { module, artifact })
                    if module == request.area().name() && artifact == request.artifact())
            );
            assert_eq!(cached.artifact(), artifact);
        }
    }
}
