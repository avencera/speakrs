use std::collections::HashMap;
#[cfg(feature = "_cuda-libraries")]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex};

#[cfg(feature = "_cuda-libraries")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "_cuda-libraries")]
use cudarc::cudnn::Cudnn;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use tracing::debug;

use super::CudaMath;
use super::device::DeviceAttributes;
use super::error::CudaLibrary;
use super::implementation::BoundaryId;
use super::kernels::{ArtifactLoadError, ModuleRequest};
#[cfg(feature = "_cuda-libraries")]
use super::libraries::Libraries;
use super::tuning::{ApprovedChoice, BenchKind, BoundaryGraph, TuneControl};
use super::{ComputeCapability, CudaError, KernelModule, LoadedKernels, PtxTier};

/// One CUDA device context and stream, with optional libraries prepared by plans
///
/// Everything issued through one runtime runs in order on its stream. Use one runtime
/// per worker thread; device buffers from one runtime must not be used on another
/// runtime's stream without explicit synchronization
///
/// The runtime queries the device's attributes once. Each kernel area loads the module
/// request its production policy names, never a tier inferred from what is embedded
#[derive(Debug)]
pub struct CudaRuntime {
    // plans live in the session state; library handles must drop before driver state
    #[cfg(feature = "_cuda-libraries")]
    libraries: Libraries,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    device: DeviceAttributes,
    ptx_tier: PtxTier,
    modules: Mutex<HashMap<KernelModule, LoadedKernels>>,
    force_library: bool,
    recipe_mode: super::implementation::policy::RecipeMode,
    tuning: Option<TuneControl>,
    force_ptx_jit: bool,
}

impl CudaRuntime {
    /// Opens device `ordinal` and creates a stream without loading optional libraries
    ///
    /// Fails with [`CudaError::LibraryUnavailable`] instead of panicking when a CUDA
    /// shared library is missing. [`PTX_TIER_ENV`](super::PTX_TIER_ENV) may force a
    /// lower compiled-in PTX tier; see [`Self::with_ptx_tier`]
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
        ensure_driver()?;

        let count = device_count()?;
        if ordinal >= count {
            return Err(CudaError::NoDevice { ordinal, count });
        }

        let context = CudaContext::new(ordinal)?;
        let device = DeviceAttributes::query(&context)?;
        let capability = device.capability();
        // target features must support the device even when an override lowers the limit
        if super::driver_only() {
            let compiled = PtxTier::select(capability, None)?;
            let native = PtxTier::native(capability);
            if compiled < native {
                return Err(CudaError::TierNotCompiledIn {
                    tier: native,
                    device: capability,
                    feature: native.feature(),
                });
            }
        }
        let ptx_tier = PtxTier::select(capability, requested)?;
        debug!(
            ordinal,
            %capability,
            %ptx_tier,
            ?requested,
            "CUDA device opened"
        );

        // a dedicated stream rather than the legacy default stream, so workers on
        // separate runtimes do not serialize against each other
        let stream = context.new_stream()?;

        let mut runtime = Self {
            #[cfg(feature = "_cuda-libraries")]
            libraries: Libraries::default(),
            context,
            stream,
            device,
            ptx_tier,
            modules: Mutex::new(HashMap::new()),
            force_library: !super::driver_only() && force_library_from_env(),
            recipe_mode: super::implementation::policy::RecipeMode::Disabled,
            tuning: None,
            force_ptx_jit: std::env::var_os(super::kernels::FORCE_PTX_JIT_ENV)
                .is_some_and(|value| value == "1"),
        };
        if read_tune_file {
            runtime.tuning = TuneControl::load(&runtime);
        }
        debug!(
            device_name = runtime.device.name(),
            sm_count = runtime.device.multiprocessors(),
            l2_bytes = runtime.device.l2_bytes(),
            shared_optin_bytes = runtime.device.shared_optin_bytes(),
            "CUDA device properties"
        );
        Ok(runtime)
    }

    /// The tuner never reads old measurements or inherits the force-Library policy
    pub(crate) fn for_tuning(ordinal: usize, kind: BenchKind) -> Result<Self, CudaError> {
        if std::env::var_os(super::kernels::FORCE_PTX_JIT_ENV).is_some_and(|value| value == "1") {
            return Err(super::tuning::invalid(
                "unset SPEAKRS_CUDA_FORCE_PTX_JIT before tuning",
            ));
        }
        let mut runtime = Self::open(ordinal, PtxTier::from_env()?, false)?;
        runtime.force_library = false;
        runtime.tuning = Some(TuneControl::benchmark(
            kind,
            &runtime.device,
            runtime.ptx_tier,
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
            self.context.bind_to_thread()?;
            self.libraries.versions(&self.stream)
        }
        #[cfg(not(feature = "_cuda-libraries"))]
        {
            Ok(super::tuning::LibraryVersions::DriverOnly)
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

    /// The device's compute capability
    pub fn compute_capability(&self) -> ComputeCapability {
        self.device.capability()
    }

    /// The device attributes queried when the runtime opened
    pub(crate) fn device(&self) -> &DeviceAttributes {
        &self.device
    }

    /// The PTX tier limit: a production binding whose tier is above it is not loaded
    pub fn ptx_tier(&self) -> PtxTier {
        self.ptx_tier
    }

    /// The device context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// The stream that every operation of this runtime runs on
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// A direct request has no selected area, boundary, batch, math or loaded tier
    pub(super) fn library_forbidden(library: CudaLibrary) -> CudaError {
        CudaError::LibraryForbidden { library }
    }

    #[cfg(feature = "_cuda-libraries")]
    fn library_policy(&self, library: CudaLibrary) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(Self::library_forbidden(library));
        }
        Ok(())
    }

    /// Prepare a library during construction, never during forward or capture
    #[cfg(feature = "_cuda-libraries")]
    pub(super) fn prepare_library(&self, library: CudaLibrary) -> Result<(), CudaError> {
        self.library_policy(library)?;
        self.libraries.prepare(library, &self.stream)
    }

    /// The already prepared cuBLAS handle
    #[cfg(feature = "_cuda-libraries")]
    pub fn blas(&self) -> Result<&CudaBlas, CudaError> {
        self.library_policy(CudaLibrary::Cublas)?;
        self.libraries.blas()
    }

    /// Keep cuBLAS mode selection and enqueue atomic
    #[cfg(feature = "_cuda-libraries")]
    pub(super) fn lock_blas(&self, math: CudaMath) -> Result<MutexGuard<'_, CudaMath>, CudaError> {
        self.library_policy(CudaLibrary::Cublas)?;
        self.libraries.lock_blas(math)
    }

    /// The already prepared cuDNN handle
    #[cfg(feature = "_cuda-libraries")]
    pub fn dnn(&self) -> Result<&Arc<Cudnn>, CudaError> {
        self.library_policy(CudaLibrary::Cudnn)?;
        self.libraries.dnn()
    }

    /// The module production loads for `module` on this device, resolved before any
    /// load; bindings take precedence over the best runnable embedded artifact
    pub(crate) fn production_module(
        &self,
        module: KernelModule,
    ) -> Result<Option<ModuleRequest>, CudaError> {
        super::implementation::production_module(
            module,
            &self.device,
            self.ptx_tier,
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
        request.diagnostic_request(self.force_ptx_jit)
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
                tier: self.ptx_tier,
                device: self.device.capability(),
                feature: self.ptx_tier.feature(),
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
                device: self.device.capability(),
            })?;
        self.load_module(request)
    }

    /// Load exactly `request`; a cached module with another identity is an error, never
    /// a replacement, and a driver refusal never tries the other artifact format
    pub(crate) fn load_module(&self, request: ModuleRequest) -> Result<LoadedKernels, CudaError> {
        let module = request.area();
        let tier = request.tier();
        let unavailable = |artifact| CudaError::ArtifactUnavailable {
            module: module.name(),
            artifact,
        };
        let embedded = module
            .variants()
            .embedded(tier)
            .ok_or_else(|| unavailable(request.artifact()))?;
        let ptx = embedded.text;
        let ptx_sha256 = embedded.sha256();
        let cubin = embedded.cubin(self.device.capability());
        let requested = request.artifact();
        let mut modules = self
            .modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(loaded) = modules.get(&module) {
            request.check_cached(loaded.request())?;
            return Ok(loaded.clone());
        }

        let (inner, artifact) = super::kernels::load_artifact(
            requested,
            cubin,
            ptx_sha256,
            |bytes| self.context.load_module(Ptx::from_binary(bytes.to_vec())),
            || self.context.load_module(Ptx::from_src(ptx)),
        )
        .map_err(|error| match error {
            ArtifactLoadError::Unavailable => unavailable(requested),
            ArtifactLoadError::Driver(source) => CudaError::ArtifactLoad {
                module: module.name(),
                artifact: requested,
                source,
            },
        })?;
        debug!(area = module.name(), %tier, capability = %self.device.capability(), ?artifact, "Loaded CUDA artifact");
        let loaded = LoadedKernels::new(
            ModuleRequest::new(module, tier, artifact),
            inner,
            ptx_sha256,
        );
        debug!(embedded_ptx_sha256 = %loaded.ptx_sha256(), "CUDA embedded PTX identity");

        modules.insert(module, loaded.clone());
        Ok(loaded)
    }

    /// Streaming multiprocessors this context may use: the device's, or the client's
    /// share under MPS active-thread limits, as queried when the runtime opened
    pub fn multiprocessor_count(&self) -> Result<usize, CudaError> {
        Ok(self.device.multiprocessors().get() as usize)
    }

    /// Whether the device supports cooperative kernel launches
    pub fn supports_cooperative_launch(&self) -> Result<bool, CudaError> {
        let supported = self.context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH,
        )?;
        Ok(supported != 0)
    }

    /// Blocks of `function` that one cooperative launch keeps resident at once: the
    /// occupancy API's active blocks per SM for `block_threads` threads and
    /// `dynamic_smem` bytes of dynamic shared memory, times the SM count, or 0 when the
    /// device has no cooperative launch
    ///
    /// Grids that run concurrently, for example on a side stream, share this capacity,
    /// so their sum must fit; size them with [`Self::concurrent_cooperative_capacity`]
    pub fn cooperative_capacity(
        &self,
        function: &cudarc::driver::CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<usize, CudaError> {
        let per_sm = self.cooperative_blocks_per_sm(function, block_threads, dynamic_smem)?;
        Ok(per_sm * self.multiprocessor_count()?)
    }

    /// Blocks of `function` that cooperative grids running at the same time can keep
    /// resident together in this context, or `None` when the context's SM limit is
    /// unknown, in which case grids must not run concurrently
    ///
    /// The driver checks co-residency per cooperative launch, not for grids launched on
    /// different streams, and Ampere and Ada start a cooperative grid before all its
    /// blocks fit. MPS active-thread limits already shrink the SM count attribute, but a
    /// context created with an SM-count execution affinity is limited further, so the
    /// joint budget also honors that limit. Green contexts, static MPS SM partitions and
    /// other spinning work on the device, such as a second pipeline, are not covered
    pub fn concurrent_cooperative_capacity(
        &self,
        function: &cudarc::driver::CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<Option<usize>, CudaError> {
        let per_sm = self.cooperative_blocks_per_sm(function, block_threads, dynamic_smem)?;
        let device_sms = self.multiprocessor_count()?;
        let budget_sms = match self.affinity_sm_limit()? {
            SmLimit::Unlimited => device_sms,
            SmLimit::Limited(sms) => sms.min(device_sms),
            SmLimit::Unknown => return Ok(None),
        };

        Ok(Some(per_sm * budget_sms))
    }

    /// Active blocks per SM for a cooperative launch of `function`, or 0 when the device
    /// has no cooperative launch
    fn cooperative_blocks_per_sm(
        &self,
        function: &cudarc::driver::CudaFunction,
        block_threads: u32,
        dynamic_smem: usize,
    ) -> Result<usize, CudaError> {
        if !self.supports_cooperative_launch()? {
            return Ok(0);
        }

        let per_sm = function.occupancy_max_active_blocks_per_multiprocessor(
            block_threads,
            dynamic_smem,
            None,
        )?;
        Ok(usize::try_from(per_sm).unwrap_or(0))
    }

    /// The SM-count execution affinity of this context
    fn affinity_sm_limit(&self) -> Result<SmLimit, CudaError> {
        use cudarc::driver::sys::{
            CUexecAffinityParam, CUexecAffinityParam_st__bindgen_ty_1, CUexecAffinitySmCount,
            CUexecAffinityType, cuCtxGetExecAffinity,
        };

        self.context.bind_to_thread()?;
        let mut param = CUexecAffinityParam {
            type_: CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT,
            param: CUexecAffinityParam_st__bindgen_ty_1 {
                smCount: CUexecAffinitySmCount { val: 0 },
            },
        };

        // SAFETY: `param` is a valid out-pointer for the call, and the context is bound
        // to this thread above; the driver writes only `param`
        let status = unsafe {
            cuCtxGetExecAffinity(
                &mut param,
                CUexecAffinityType::CU_EXEC_AFFINITY_TYPE_SM_COUNT,
            )
        };

        match status {
            CUresult::CUDA_SUCCESS => {
                // SAFETY: the SM-count query fills the `smCount` member
                let count = unsafe { param.param.smCount.val };
                Ok(usize::try_from(count)
                    .ok()
                    .filter(|count| *count > 0)
                    .map_or(SmLimit::Unknown, SmLimit::Limited))
            }
            // affinity is an MPS feature, so a context without it has no extra limit
            CUresult::CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY => Ok(SmLimit::Unlimited),
            status => {
                debug!(?status, "Context SM limit unavailable");
                Ok(SmLimit::Unknown)
            }
        }
    }

    /// Blocks until all work queued on [`Self::stream`] has finished
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.stream.synchronize()?;
        Ok(())
    }
}

/// An SM-count execution-affinity limit on a context
enum SmLimit {
    /// The context has no SM-count affinity
    Unlimited,
    /// The context may use this many SMs
    Limited(usize),
    /// The driver didn't report the limit
    Unknown,
}

/// Only the driver is needed before a model is selected
fn ensure_driver() -> Result<(), CudaError> {
    // SAFETY: driver library initializers have no caller preconditions
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Err(CudaError::LibraryUnavailable {
            library: CudaLibrary::Driver,
        });
    }
    Ok(())
}

fn device_count() -> Result<usize, CudaError> {
    match CudaContext::device_count() {
        Ok(count) => Ok(usize::try_from(count).unwrap_or(0)),
        // a driver with no visible GPU fails initialization instead of reporting zero
        Err(DriverError(CUresult::CUDA_ERROR_NO_DEVICE)) => Ok(0),
        Err(error) => Err(error.into()),
    }
}

/// Read once per runtime so graph capture and execution do not consult the environment
pub(super) fn force_library_from_env() -> bool {
    std::env::var_os("SPEAKRS_CUDA_FORCE_LIBRARY").is_some_and(|value| value == "1")
}

#[cfg(test)]
mod direct_request_tests {
    use super::{CudaError, CudaLibrary, CudaRuntime};

    #[test]
    fn forbidden_direct_requests_do_not_invent_model_selection_context() {
        for library in [CudaLibrary::Cublas, CudaLibrary::Cudnn, CudaLibrary::Nvrtc] {
            let error = CudaRuntime::library_forbidden(library);
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
    use super::{ComputeCapability, CudaError, KernelModule, ModuleRequest, PtxTier};
    use crate::inference::cuda::kernels::ArtifactHash;
    use crate::inference::cuda::kernels::LoadedArtifact;

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
            assert!(matches!(request.check_cached(cached),
                Err(CudaError::ArtifactUnavailable { module, artifact })
                    if module == request.area().name() && artifact == request.artifact()));
            assert_eq!(cached.artifact(), artifact);
        }
    }
}
