use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::cudnn::Cudnn;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use tracing::debug;

use super::device::DeviceAttributes;
use super::error::CudaLibrary;
#[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
use super::kernels::LoadedArtifact;
use super::kernels::{ArtifactHash, ArtifactLoadError, ModuleRequest};
use super::{ComputeCapability, CudaError, KernelModule, LoadedKernels, PtxTier};
#[cfg(feature = "cuda")]
use super::{CudaMath, libraries::Libraries};

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
    #[cfg(feature = "cuda")]
    libraries: Libraries,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    device: DeviceAttributes,
    ptx_tier: PtxTier,
    modules: Mutex<HashMap<KernelModule, LoadedKernels>>,
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

        let runtime = Self {
            #[cfg(feature = "cuda")]
            libraries: Libraries::default(),
            context,
            stream,
            device,
            ptx_tier,
            modules: Mutex::new(HashMap::new()),
        };
        // a forced tier in driver-only mode must leave every candidate area a binding
        if super::driver_only() && requested.is_some() {
            for area in [
                KernelModule::Resnet,
                KernelModule::Lstm,
                KernelModule::Sincnet,
            ] {
                let request =
                    runtime
                        .production_module(area)?
                        .ok_or(CudaError::TierNotQualified {
                            tier: PtxTier::BASELINE,
                            device: capability,
                        })?;
                runtime.load_module(request)?;
            }
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

    #[cfg(feature = "cuda")]
    fn library_policy(&self, library: CudaLibrary) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(Self::library_forbidden(library));
        }
        Ok(())
    }

    /// Prepare a library during construction, never during forward or capture
    #[cfg(feature = "cuda")]
    pub(super) fn prepare_library(&self, library: CudaLibrary) -> Result<(), CudaError> {
        self.library_policy(library)?;
        self.libraries.prepare(library, &self.stream)
    }

    /// The already prepared cuBLAS handle
    #[cfg(feature = "cuda")]
    pub fn blas(&self) -> Result<&CudaBlas, CudaError> {
        self.library_policy(CudaLibrary::Cublas)?;
        self.libraries.blas()
    }

    /// Keep cuBLAS mode selection and enqueue atomic
    #[cfg(feature = "cuda")]
    pub(super) fn lock_blas(&self, math: CudaMath) -> Result<MutexGuard<'_, CudaMath>, CudaError> {
        self.library_policy(CudaLibrary::Cublas)?;
        self.libraries.lock_blas(math)
    }

    /// The already prepared cuDNN handle
    #[cfg(feature = "cuda")]
    pub fn dnn(&self) -> Result<&Arc<Cudnn>, CudaError> {
        self.library_policy(CudaLibrary::Cudnn)?;
        self.libraries.dnn()
    }

    /// The module production loads for `module` on this device, resolved before any
    /// load; `None` when no binding covers this device within the tier limit
    pub(crate) fn production_module(
        &self,
        module: KernelModule,
    ) -> Result<Option<ModuleRequest>, CudaError> {
        super::implementation::production_module(
            module,
            &self.device,
            self.ptx_tier,
            module.variants(),
        )
    }

    /// The best embedded artifact an explicit qualification asks for: the highest
    /// embedded variant within the tier limit, as the device's exact cubin when
    /// embedded, otherwise PTX JIT
    #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
    pub(crate) fn embedded_exact_request(
        &self,
        module: KernelModule,
    ) -> Result<ModuleRequest, CudaError> {
        let capability = self.device.capability();
        let variants = module.variants();
        let (tier, ptx) = variants.resolve(module, self.ptx_tier, capability)?;
        let embedded = variants.embedded(tier).expect("resolved embedded tier");
        let artifact = embedded.cubin(capability).map_or(
            LoadedArtifact::PtxJit {
                sha256: ArtifactHash::of(ptx.as_bytes()),
            },
            |cubin| LoadedArtifact::Cubin {
                arch: cubin.arch,
                sha256: ArtifactHash::of(cubin.bytes),
            },
        );
        Ok(ModuleRequest::new(module, tier, artifact))
    }

    /// The cached module, or the area's production module
    ///
    /// Plan selection skips uncovered areas before a candidate plan calls this. A
    /// cached module retains its actual identity
    pub fn load_kernels(&self, module: KernelModule) -> Result<LoadedKernels, CudaError> {
        if let Some(loaded) = self
            .modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&module)
        {
            return Ok(loaded.clone());
        }
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
        let ptx_sha256 = ArtifactHash::of(ptx.as_bytes());
        let cubin = embedded.cubin(self.device.capability());
        let force_jit =
            std::env::var_os(super::kernels::FORCE_PTX_JIT_ENV).is_some_and(|value| value == "1");
        // the diagnostic override keeps the tier and changes only the artifact format
        let request = if force_jit {
            request.ptx_jit(ptx_sha256)
        } else {
            request
        };
        let requested = request.artifact();
        let mut modules = self
            .modules
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(loaded) = modules.get(&module) {
            return if loaded.request() == request {
                Ok(loaded.clone())
            } else {
                Err(unavailable(requested))
            };
        }
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::test_support::assert_module_load_allowed();
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
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::test_support::record_artifact(
            module,
            tier,
            ptx,
            loaded.artifact(),
            loaded.ptx_sha256(),
        );
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
