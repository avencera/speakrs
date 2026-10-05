use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::MutexGuard;

#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::cudnn::Cudnn;
use cudarc::driver::sys::CUresult;
use cudarc::driver::{CudaContext, CudaStream, DriverError};
use cudarc::nvrtc::Ptx;
use tracing::debug;

use super::error::CudaLibrary;
use super::{ComputeCapability, CudaError, KernelModule, LoadedKernels, PtxTier};
#[cfg(feature = "cuda")]
use super::{CudaMath, libraries::Libraries};

/// One CUDA device context and stream, with optional libraries prepared by plans
///
/// Everything issued through one runtime runs in order on its stream. Use one runtime
/// per worker thread; device buffers from one runtime must not be used on another
/// runtime's stream without explicit synchronization
///
/// The runtime reads the device's compute capability once and loads, for each kernel
/// area, the highest compiled-in PTX tier the device supports
#[derive(Debug)]
pub struct CudaRuntime {
    // plans live in the session state; library handles must drop before driver state
    #[cfg(feature = "cuda")]
    libraries: Libraries,
    stream: Arc<CudaStream>,
    context: Arc<CudaContext>,
    capability: ComputeCapability,
    ptx_tier: PtxTier,
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
        let (major, minor) = context.compute_capability()?;
        let capability = ComputeCapability::new(
            u32::try_from(major).unwrap_or(0),
            u32::try_from(minor).unwrap_or(0),
        );
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
        if super::driver_only() && requested.is_some() {
            // qualification follows the area variant, not the GPU target feature
            for area in [
                KernelModule::Resnet,
                KernelModule::Lstm,
                KernelModule::Sincnet,
            ] {
                let (tier, _) = area.variants().resolve(area, ptx_tier, capability)?;
                if !super::implementation::tier_qualified(super::implementation::Target {
                    tier,
                    device: capability,
                }) {
                    return Err(CudaError::TierNotQualified {
                        tier,
                        device: capability,
                    });
                }
            }
        }
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

        Ok(Self {
            #[cfg(feature = "cuda")]
            libraries: Libraries::default(),
            context,
            stream,
            capability,
            ptx_tier,
        })
    }

    /// The device's compute capability
    pub fn compute_capability(&self) -> ComputeCapability {
        self.capability
    }

    /// The highest PTX tier this runtime loads; each area loads its highest variant at
    /// or below it
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

    #[cfg(feature = "cuda")]
    fn library_policy(&self, library: CudaLibrary) -> Result<(), CudaError> {
        if super::driver_only() {
            return Err(CudaError::NotDriverOnly {
                area: "runtime",
                boundary: "library request".to_owned(),
                batch: 1,
                math: CudaMath::Fp32,
                tier: self.ptx_tier,
                device: self.capability,
                library,
            });
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

    /// Resolve an area's variant before selection or module loading
    pub(super) fn area_ptx(
        &self,
        module: KernelModule,
    ) -> Result<(PtxTier, &'static str), CudaError> {
        module
            .variants()
            .resolve(module, self.ptx_tier, self.capability)
    }

    /// Loads the highest embedded PTX variant of `module` at or below
    /// [`Self::ptx_tier`]; the driver JIT-compiles it for this device
    ///
    /// Load each module once per runtime and keep the result: JIT compilation is
    /// cached by the driver but loading still costs time
    pub fn load_kernels(&self, module: KernelModule) -> Result<LoadedKernels, CudaError> {
        let (tier, ptx) = self.area_ptx(module)?;
        debug!(
            area = module.name(),
            %tier,
            limit = %self.ptx_tier,
            capability = %self.capability,
            "Loading CUDA PTX variant"
        );

        // the harness allow-list comes from the exact bytes handed to the driver
        #[cfg(all(test, feature = "cuda", not(feature = "cuda-driver-only")))]
        super::test_support::record_module(module.name(), &tier.to_string(), ptx);
        let loaded = self
            .context
            .load_module(Ptx::from_src(ptx))
            .map_err(|source| CudaError::ModuleLoad {
                module: module.name(),
                source,
            })?;

        Ok(LoadedKernels::new(module, tier, loaded))
    }

    /// Streaming multiprocessors this context may use: the device's, or the client's
    /// share under MPS active-thread limits
    pub fn multiprocessor_count(&self) -> Result<usize, CudaError> {
        // under per-context MPS partitioning the attribute follows the current context
        self.context.bind_to_thread()?;
        let count = self.context.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )?;
        Ok(usize::try_from(count).unwrap_or(0))
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
