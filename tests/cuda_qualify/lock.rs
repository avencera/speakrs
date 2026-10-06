//! Shared GPU ownership with unlocked, host-only qualification work

use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::fs::{File, OpenOptions};
use std::marker::PhantomData;
use std::path::Path;
use std::rc::Rc;

const GPU_LOCK: &str = "/workspace/gpu-bench.lock";

/// One prepared truth case, whether or not its candidate tuple is declared
pub(crate) struct TruthCase(String);

impl TruthCase {
    /// Bind truth preparation and the emitted secret row to the same case
    pub(crate) fn new(math: CudaMath, batch: usize, layer: &str) -> Self {
        assert!(batch > 0 && !layer.is_empty(), "valid truth case");
        let mode = match math {
            CudaMath::Fp32 => "fp32",
            CudaMath::Tf32 => "tf32",
        };
        Self(format!("{mode}/secret/b{batch}/{layer}"))
    }

    /// Return the shared truth-preparation and result-row identity
    pub(crate) fn id(&self) -> &str {
        &self.0
    }
}

/// CPU work that must not hold the shared GPU lock
#[derive(Clone, Copy)]
pub(crate) enum CpuWork<'a> {
    F64(&'a TruthCase),
    Tf32Draws,
}

impl CpuWork<'_> {
    fn name(self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::Tf32Draws => "tf32_draws",
        }
    }
}

enum Ownership {
    Child(File),
    Parent,
}

struct State {
    ownership: Ownership,
    cpu_sections: Vec<Value>,
    gpu_sections: usize,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

/// Holds GPU ownership until all local and test TLS CUDA values have been dropped
pub(crate) struct GpuLock(PhantomData<Rc<()>>);

impl GpuLock {
    pub(crate) fn from_environment(phase: &str) -> Self {
        let owner = std::env::var("SPEAKRS_QUALIFY_LOCK_OWNER").expect("GPU lock owner");
        match owner.as_str() {
            "child" => {
                assert_eq!(phase, "numeric", "only numeric can release the GPU lock");
                Self::acquire(Path::new(GPU_LOCK))
            }
            "parent" => {
                let file = open(Path::new(GPU_LOCK));
                assert!(
                    matches!(file.try_lock(), Err(std::fs::TryLockError::WouldBlock)),
                    "parent must hold the GPU lock"
                );
                Self::install(Ownership::Parent)
            }
            _ => panic!("unknown GPU lock owner"),
        }
    }

    fn acquire(path: &Path) -> Self {
        STATE.with(|cell| assert!(cell.borrow().is_none(), "GPU lock owner must not nest"));
        let file = open(path);
        file.lock().expect("acquire shared GPU lock");
        Self::install(Ownership::Child(file))
    }

    fn install(ownership: Ownership) -> Self {
        STATE.with(|cell| {
            let mut state = cell.borrow_mut();
            assert!(state.is_none(), "GPU lock owner must not nest");
            *state = Some(State {
                ownership,
                cpu_sections: Vec::new(),
                gpu_sections: 1,
            });
        });
        Self(PhantomData)
    }

    pub(crate) fn evidence(&self) -> Value {
        STATE.with(|cell| {
            let state = cell.borrow();
            let state = state.as_ref().expect("live GPU lock owner");
            let owner = match state.ownership {
                Ownership::Child(_) => "child",
                Ownership::Parent => "parent",
            };
            json!({"path": GPU_LOCK, "owner": owner,
                "cpu_sections": state.cpu_sections, "gpu_sections": state.gpu_sections})
        })
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        // CUDA functions and slices retain contexts beyond the runtime's local scope
        super::paired::clear_tail();
        super::super::clear_device_state();
        STATE.with(|cell| {
            let state = cell.borrow_mut().take().expect("live GPU lock owner");
            if let Ownership::Child(file) = state.ownership {
                file.unlock()
                    .expect("release shared GPU lock after CUDA drops");
            }
        });
    }
}

fn open(path: &Path) -> File {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .expect("open shared GPU lock")
}

struct Relock<'a>(&'a File);

impl Drop for Relock<'_> {
    fn drop(&mut self) {
        // failure must not unwind into CUDA destructors without GPU ownership
        if self.0.lock().is_err() {
            std::process::abort();
        }
    }
}

/// Runs host-only work after all GPU streams are idle, and relocks before any unwind
pub(crate) fn cpu<T>(
    runtime: &CudaRuntime,
    work: CpuWork<'_>,
    compute: impl FnOnce() -> T,
) -> Result<T, CudaError> {
    // cuDNN and candidates can use side streams, so the whole context must be idle
    runtime.context().synchronize()?;
    Ok(unlocked(work, compute))
}

fn unlocked<T>(work: CpuWork<'_>, compute: impl FnOnce() -> T) -> T {
    let result = STATE.with(|cell| {
        let state = cell.borrow();
        let state = state.as_ref().expect("CPU work requires a GPU lock owner");
        let Ownership::Child(file) = &state.ownership else {
            panic!("parent-owned GPU processes must not run CPU qualification work");
        };
        file.unlock().expect("release GPU lock for host-only work");
        let relock = Relock(file);
        let result = compute();
        drop(relock);
        result
    });
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let state = state.as_mut().expect("live GPU lock owner");
        let mut evidence = json!({"work": work.name(), "locked": false});
        if let CpuWork::F64(case) = work {
            evidence["case"] = json!(case.id());
        }
        state.cpu_sections.push(evidence);
        state.gpu_sections += 1;
    });
    result
}

#[cfg(test)]
mod tests {
    use super::{CpuWork, GpuLock, TruthCase, open, unlocked};
    use crate::inference::cuda::CudaMath;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn cpu_work_releases_lock_and_relocks_on_return_and_panic() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("speakrs-gpu-lock-{}-{unique}", std::process::id()));
        let owner = GpuLock::acquire(&path);
        let competing = open(&path);
        assert!(competing.try_lock().is_err());
        let case = TruthCase::new(CudaMath::Fp32, 1, "lstm.stack");
        let value = unlocked(CpuWork::F64(&case), || {
            competing.try_lock().expect("CPU section does not own lock");
            competing.unlock().expect("release competing lock");
            73
        });
        assert_eq!(value, 73);
        assert!(competing.try_lock().is_err());
        assert_eq!(owner.evidence()["cpu_sections"][0]["locked"], false);
        assert_eq!(owner.evidence()["cpu_sections"][0]["case"], case.id());
        let panic = catch_unwind(AssertUnwindSafe(|| {
            unlocked(CpuWork::Tf32Draws, || panic!("CPU failure"));
        }));
        assert!(panic.is_err());
        assert!(competing.try_lock().is_err());
        drop(owner);
        competing
            .try_lock()
            .expect("owner released lock after drop");
        drop(competing);
        std::fs::remove_file(path).expect("remove test lock");
    }
}
