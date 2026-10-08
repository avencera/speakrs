//! Private diagnostics for the FP16 operand experiment

use cudarc::driver::{CudaFunction, CudaView, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::super::{CudaError, CudaMath, CudaRuntime};
use super::trunk::ConvLayer;

#[derive(Debug)]
pub(super) struct RangeProbe(CudaFunction);

impl RangeProbe {
    pub(super) fn load(runtime: &CudaRuntime) -> Result<Option<Self>, CudaError> {
        let Some(path) = std::env::var_os("SPEAKRS_FP16_RANGE_PTX") else {
            return Ok(None);
        };
        let source = std::fs::read_to_string(path).map_err(|error| CudaError::Unsupported {
            context: "FP16 range experiment",
            reason: error.to_string(),
        })?;
        let module = runtime.context().load_module(Ptx::from_src(source))?;
        Ok(Some(Self(module.load_function("range_stats")?)))
    }

    pub(super) fn observe(
        &self,
        runtime: &CudaRuntime,
        layer: &ConvLayer,
        batch: usize,
        values: &CudaView<'_, f32>,
        weight: bool,
    ) -> Result<(), CudaError> {
        let spec = layer.conv(batch, CudaMath::Tf32);
        let [h, w] = spec.input.map(|v| v as i32);
        let transformed = !weight
            && spec.in_channels >= 128
            && spec.in_channels == spec.out_channels
            && spec.stride == [1, 1]
            && spec.kernel == [3, 3];
        for transform in 0..=i32::from(transformed) {
            let mut partial = runtime.stream().alloc_zeros::<u64>(4096 * 8)?;
            let len = values.len() as u64;
            let mut launch = runtime.stream().launch_builder(&self.0);
            launch
                .arg(values)
                .arg(&len)
                .arg(&h)
                .arg(&w)
                .arg(&transform)
                .arg(&mut partial);
            // safety: the external diagnostic has this ABI and writes eight words per thread
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (16, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            let partial = runtime.stream().clone_dtoh(&partial)?;
            let mut maximum = 0.0f32;
            let mut counts = [0u64; 6];
            let mut minimum = f32::INFINITY;
            for row in partial.as_chunks::<8>().0 {
                maximum = maximum.max(f32::from_bits(row[0] as u32));
                minimum = minimum.min(f32::from_bits(row[7] as u32));
                for (count, value) in counts.iter_mut().zip(&row[1..7]) {
                    *count += value;
                }
            }
            let kind = if weight {
                "weight"
            } else if transform != 0 {
                "winograd_input"
            } else {
                "input"
            };
            let name = layer.name();
            let [small, overflow, zero, nonfinite, total, scaled_small] = counts;
            eprintln!(
                "FP16_RANGE {name} {kind} {maximum} {small} {overflow} {zero} {nonfinite} {total} {scaled_small} {minimum}"
            );
        }
        Ok(())
    }
}
