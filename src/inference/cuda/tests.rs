//! GPU tests of the native CUDA backend's internals
//!
//! Each test skips with a message when the machine has no NVIDIA driver or GPU, or no
//! reference tensors. Set `SPEAKRS_REQUIRE_GPU=1` to turn every skip into a failure,
//! so a GPU host proves that each test really ran. The references are the ONNX
//! Runtime CPU outputs written by `scripts/cuda/make_reference.py`, read from
//! `SPEAKRS_CUDA_REF`, else `/workspace/ref` on the GPU box, else
//! `~/Library/Caches/speakrs-cuda-ref`
//!
//! The benchmarks are ignored by default; run them with the GPU lock held, for example
//! `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib
//! segmentation_benchmark -- --ignored --nocapture`

mod driver_sinc;
mod embedding;
mod fbank;
mod lstmproj;
mod resnet;
mod runtime;
mod segdense;
mod segmentation;
mod wideconv;

use std::path::PathBuf;

use tracing_subscriber::EnvFilter;

use super::{CudaError, CudaRuntime, PtxTier};

/// Whether `SPEAKRS_REQUIRE_GPU` turns skips into failures
fn required() -> bool {
    std::env::var_os("SPEAKRS_REQUIRE_GPU").is_some_and(|value| value != "0")
}

/// Skips `test` for `reason`, or fails it when the GPU is required
fn skip(test: &str, reason: &str) {
    assert!(!required(), "{test}: {reason}");
    eprintln!("skipping {test}: {reason}");
}

/// A runtime on device 0 that honours `SPEAKRS_CUDA_PTX_TIER`, or `None` to skip
fn runtime(test: &str) -> Option<CudaRuntime> {
    start_runtime(test, || CudaRuntime::new(0))
}

/// A runtime on device 0 with an explicit PTX tier, independent of
/// `SPEAKRS_CUDA_PTX_TIER`
fn runtime_with_tier(test: &str, tier: Option<PtxTier>) -> Option<CudaRuntime> {
    start_runtime(test, || CudaRuntime::with_ptx_tier(0, tier))
}

fn start_runtime(
    test: &str,
    start: impl FnOnce() -> Result<CudaRuntime, CudaError>,
) -> Option<CudaRuntime> {
    // shows the PTX variant and the convolution plans when run with --nocapture
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("speakrs::inference::cuda=debug"))
        .with_test_writer()
        .try_init();

    match start() {
        Ok(runtime) => Some(runtime),
        Err(error) if error.is_device_unavailable() => {
            skip(test, &format!("no usable NVIDIA GPU ({error})"));
            None
        }
        Err(error) => panic!("{test}: CUDA runtime failed to start: {error}"),
    }
}

/// The reference directory that holds `area`, or `None` to skip
fn reference_dir(test: &str, area: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(|home| PathBuf::from(home).join("Library/Caches/speakrs-cuda-ref"));
    let found = std::env::var_os("SPEAKRS_CUDA_REF")
        .map(PathBuf::from)
        .into_iter()
        .chain([PathBuf::from("/workspace/ref")])
        .chain(home)
        .find(|dir| dir.join(area).is_dir());

    if found.is_none() {
        skip(
            test,
            &format!("no `{area}` reference tensors (set SPEAKRS_CUDA_REF)"),
        );
    }
    found
}
