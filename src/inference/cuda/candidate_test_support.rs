//! Harness scopes compiled only in CUDA tests

use super::{CudaStream, Direction};

/// A locked harness sub-scope in the qualification binary; production opens none
pub(super) fn sub_scope(name: impl FnOnce() -> String) -> super::super::test_support::Scope {
    super::super::test_support::sub_scope(&name())
}

pub(super) fn projection_scope(
    stream: &CudaStream,
    layer: usize,
    direction: Direction,
) -> super::super::test_support::Scope {
    super::super::test_support::projection(stream, layer, direction)
}

use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::{Value, json};

use crate::inference::cuda::{CudaError, CudaLibrary, CudaRuntime};

/// Read versions from the actual loaded APIs, not package names on disk
pub(crate) fn device(runtime: &CudaRuntime) -> Result<Value, CudaError> {
    runtime.prepare_library(CudaLibrary::Cublas)?;
    runtime.prepare_library(CudaLibrary::Cudnn)?;
    let mut driver = 0;
    let mut blas = 0;
    // SAFETY: both APIs write one initialized integer and the prepared handle is live
    unsafe {
        cudarc::driver::sys::cuDriverGetVersion(&mut driver).result()?;
        cudarc::cublas::sys::cublasGetVersion_v2(*runtime.blas()?.handle(), &mut blas).result()?;
    }
    // SAFETY: version queries take no pointers and the libraries were prepared above
    let (dnn, cuda) = unsafe {
        (
            cudarc::cudnn::sys::cudnnGetVersion(),
            cudarc::cublas::sys::cublasGetCudartVersion(),
        )
    };
    let driver_release = Command::new("nvidia-smi")
        .args([
            "--id=0",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .expect("driver version query");
    assert!(
        driver_release.status.success(),
        "driver version query failed"
    );
    let driver_release = String::from_utf8(driver_release.stdout)
        .expect("driver version text")
        .trim()
        .to_owned();
    Ok(
        json!({"name": runtime.context().name()?, "compute_capability": runtime.compute_capability().to_string(),
              "sm_count": runtime.multiprocessor_count()?, "l2_bytes": runtime.l2_cache_size()?, "driver_api_version": driver, "driver_version":driver_release,
              "cuda_version": cuda, "cudnn_version": dnn, "cublas_version": blas}),
    )
}

/// A joined sampler; no process survives the timed driver's GPU lock
pub(crate) struct Clocks {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<u32>>>,
    worker: Option<JoinHandle<()>>,
}

impl Clocks {
    pub(crate) fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let worker_stop = Arc::clone(&stop);
        let worker_samples = Arc::clone(&samples);
        let worker = thread::spawn(move || {
            while !worker_stop.load(Ordering::Acquire) {
                // each query exits before the next one; there is no detached nvidia-smi poller
                let output = Command::new("nvidia-smi")
                    .args([
                        "--id=0",
                        "--query-gpu=clocks.current.sm",
                        "--format=csv,noheader,nounits",
                    ])
                    .output()
                    .expect("SM clock query");
                assert!(output.status.success(), "SM clock query failed");
                let clock = String::from_utf8(output.stdout)
                    .expect("clock text")
                    .trim()
                    .parse()
                    .expect("numeric SM clock");
                worker_samples.lock().expect("clock samples").push(clock);
                thread::sleep(Duration::from_millis(100));
            }
        });
        Self {
            stop,
            samples,
            worker: Some(worker),
        }
    }

    pub(crate) fn finish(mut self) -> Value {
        self.join();
        let samples = self.samples.lock().expect("clock samples");
        assert!(!samples.is_empty(), "observed clocks required");
        json!({"min_mhz": samples.iter().min(), "max_mhz": samples.iter().max(), "samples": samples.len(), "poll_ms": 100})
    }

    fn join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().expect("clock sampler completed");
        }
    }
}

impl Drop for Clocks {
    fn drop(&mut self) {
        self.join();
    }
}
