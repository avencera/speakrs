use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};

use super::error::check_len;
use super::{CudaError, CudaRuntime, KernelModule, PtxTier};

const PROBE_SCALE_ADD: &str = "probe_scale_add";

/// Kernel entries loaded by the probe host plan
pub(super) const REQUIRED_KERNELS: [&str; 1] = [PROBE_SCALE_ADD];

/// Host launcher for the toolchain probe kernel; also the template for area launchers
///
/// cuda-oxide passes each `&[T]` or `DisjointSlice<T>` parameter as two PTX
/// parameters, a `u64` device pointer followed by a `u64` element count, so every
/// slice pushes the buffer and then its length
#[derive(Debug, Clone)]
pub struct ProbeKernels {
    tier: PtxTier,
    scale_add: CudaFunction,
}

impl ProbeKernels {
    /// Loads the probe module and looks up its kernels
    pub fn load(runtime: &CudaRuntime) -> Result<Self, CudaError> {
        // the toolchain probe is an explicit test request, never a production owner
        #[cfg(all(feature = "cuda", not(feature = "cuda-driver-only")))]
        let request = super::kernels::ArtifactRequest::EmbeddedExact;
        #[cfg(any(not(feature = "cuda"), feature = "cuda-driver-only"))]
        let request =
            super::kernels::ArtifactRequest::Pinned(super::kernels::LoadedArtifact::PtxJit {
                sha256: super::kernels::ArtifactHash::of(
                    runtime.area_ptx(KernelModule::Probe)?.1.as_bytes(),
                ),
            });
        let kernels = runtime.load_requested_kernels(KernelModule::Probe, request)?;
        Ok(Self {
            tier: kernels.tier(),
            scale_add: kernels.function(REQUIRED_KERNELS[0])?,
        })
    }

    /// The PTX tier the probe module was loaded from
    pub fn tier(&self) -> PtxTier {
        self.tier
    }

    /// `out[i] = alpha * x[i] + y[i]`, with all three buffers the same length
    pub fn scale_add(
        &self,
        runtime: &CudaRuntime,
        alpha: f32,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let len = out.len();
        check_len("probe x", len, x.len())?;
        check_len("probe y", len, y.len())?;
        let threads = u32::try_from(len).map_err(|_| CudaError::DimensionOverflow {
            context: "probe launch",
            value: len,
        })?;

        let slice_len = len as u64;
        let mut launch = runtime.stream().launch_builder(&self.scale_add);
        launch
            .arg(&alpha)
            .arg(x)
            .arg(&slice_len)
            .arg(y)
            .arg(&slice_len)
            .arg(out)
            .arg(&slice_len);

        // SAFETY: the arguments match the PTX signature of `probe_scale_add` (f32,
        // then pointer and length for x, y and out), the lengths are the real buffer
        // lengths, and the 1-D grid has one thread per element as the kernel expects
        unsafe { launch.launch(LaunchConfig::for_num_elems(threads)) }?;
        Ok(())
    }
}
