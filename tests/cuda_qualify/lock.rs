//! Shared GPU ownership with unlocked, host-only qualification work

use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime};
use serde_json::{Value, json};
use std::cell::{OnceCell, RefCell};
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

/// One complete deterministic fbank stage reference, bound before any GPU replay
pub(crate) struct StageTruthCase<'a> {
    case: String,
    batch: usize,
    input: &'a [f32],
    binding: OnceCell<Value>,
}

impl<'a> StageTruthCase<'a> {
    /// Own an immutable view of the exact case input, without a sampling seed
    pub(crate) fn new(case: &str, batch: usize, input: &'a [f32]) -> Self {
        assert!((1..=32).contains(&batch));
        assert_eq!(input.len(), batch * 160_000);
        assert!(case.ends_with("/stage") || case.ends_with("/stage/switched"));
        let parts: Vec<_> = case.split('/').collect();
        assert!(parts.len() == 4 || (parts.len() == 5 && parts[4] == "switched"));
        assert!(matches!(parts[0], "fp32" | "tf32"));
        assert_eq!(parts[2], format!("b{batch}"));
        assert_eq!(parts[3], "stage");
        assert!(match parts[1] {
            "first" | "last" => batch == 1,
            "short" => matches!(batch, 1 | 7),
            "mixed" => batch >= 2,
            _ => false,
        });
        Self {
            case: case.to_owned(),
            batch,
            input,
            binding: OnceCell::new(),
        }
    }

    /// Return the completed unrounded f64 reference identity for the emitted row
    pub(crate) fn binding(&self) -> &Value {
        self.binding
            .get()
            .expect("complete stage truth prepared under CPU ownership")
    }

    fn bind(&self, values: &[f64], constants: Value) {
        assert_eq!(values.len(), self.batch * 998 * 80);
        assert!(values.iter().all(|value| value.is_finite()));
        let parts: Vec<_> = self.case.split('/').collect();
        assert!(matches!(parts[0], "fp32" | "tf32"));
        assert_eq!(parts[2], format!("b{}", self.batch));
        let source = if self.case.ends_with("/switched") {
            if parts[1] == "short" {
                "first"
            } else {
                "short"
            }
        } else {
            parts[1]
        };
        let fixture_rows: Vec<_> = (0..self.batch)
            .map(|row| match source {
                "first" => 0,
                "last" => 17,
                "short" => 18,
                "mixed" => row,
                _ => panic!("fixed fbank stage input set"),
            })
            .collect();
        self.binding
            .set(json!({
                "case": self.case, "evaluation": "complete-deterministic", "dtype": "f64",
                "definition": super::reference::fbank_truth::DEFINITION, "constants": constants,
                "input_shape": [self.batch, 160_000], "shape": [self.batch, 998, 80],
                "fixture_rows":fixture_rows,
                "input_length": self.batch * 160_000, "length": values.len(),
                "input_sha256": super::sha(self.input),
                "truth_sha256": super::reference::fbank_truth::sha(values),
            }))
            .expect("stage truth is prepared once per case");
    }
}

/// Prepare a complete stage reference only while the shared GPU lock is released
pub(crate) fn stage_truth(
    runtime: &CudaRuntime,
    case: &StageTruthCase<'_>,
    compute: impl FnOnce() -> (Vec<f64>, Value),
) -> Result<Vec<f64>, CudaError> {
    cpu(runtime, CpuWork::StageTruth(case), || {
        assert!(case.input.iter().all(|value| value.is_finite()));
        let (values, constants) = compute();
        case.bind(&values, constants);
        values
    })
}

/// One host draw preparation, bound to the replay that consumes it
pub(crate) struct DrawCase {
    case: String,
    layer: String,
    seed: u32,
    length: usize,
}

impl DrawCase {
    /// Bind the unmodified band seed and output length before deriving draws
    pub(crate) fn new(case: &str, layer: &str, seed: u32, length: usize) -> Self {
        assert!(
            !case.is_empty() && !layer.is_empty() && length > 0,
            "valid draw case"
        );
        Self {
            case: case.to_owned(),
            layer: layer.to_owned(),
            seed,
            length,
        }
    }
}

/// CPU work that must not hold the shared GPU lock
#[derive(Clone, Copy)]
pub(crate) enum CpuWork<'a> {
    F64(&'a TruthCase),
    StageTruth(&'a StageTruthCase<'a>),
    Tf32Draws(&'a DrawCase),
}

impl CpuWork<'_> {
    fn name(self) -> &'static str {
        match self {
            Self::F64(_) => "f64",
            Self::StageTruth(_) => "stage_f64",
            Self::Tf32Draws(_) => "tf32_draws",
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
    cpu_wall_seconds: [f64; 2],
    cpu_byte_identity: Vec<Value>,
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
                cpu_wall_seconds: [0.0; 2],
                cpu_byte_identity: Vec::new(),
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
                "cpu_sections": state.cpu_sections, "gpu_sections": state.gpu_sections,
                "cpu_mode": super::cpu::Mode::from_environment().name(),
                "cpu_wall_seconds": {"f64":state.cpu_wall_seconds[0], "tf32_draws":state.cpu_wall_seconds[1]},
                "cpu_byte_identity":state.cpu_byte_identity})
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
    let (result, wall_seconds, proofs) = STATE.with(|cell| {
        let state = cell.borrow();
        let state = state.as_ref().expect("CPU work requires a GPU lock owner");
        let Ownership::Child(file) = &state.ownership else {
            panic!("parent-owned GPU processes must not run CPU qualification work");
        };
        file.unlock().expect("release GPU lock for host-only work");
        let relock = Relock(file);
        assert!(
            super::cpu::take_proofs().is_empty(),
            "no unbound CPU evidence proof"
        );
        let start = std::time::Instant::now();
        let result = compute();
        let wall_seconds = start.elapsed().as_secs_f64();
        let proofs = super::cpu::take_proofs();
        drop(relock);
        (result, wall_seconds, proofs)
    });
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let state = state.as_mut().expect("live GPU lock owner");
        let mut evidence = json!({"work": work.name(), "locked": false});
        match work {
            CpuWork::F64(case) => evidence["case"] = json!(case.id()),
            CpuWork::StageTruth(case) => {
                evidence
                    .as_object_mut()
                    .expect("CPU section")
                    .extend(case.binding().as_object().expect("stage binding").clone());
            }
            CpuWork::Tf32Draws(draw) => {
                evidence["case"] = json!(draw.case);
                evidence["layer"] = json!(draw.layer);
                evidence["seed"] = json!(draw.seed);
                evidence["length"] = json!(draw.length);
            }
        }
        let index = match work {
            CpuWork::F64(_) | CpuWork::StageTruth(_) => 0,
            CpuWork::Tf32Draws(_) => 1,
        };
        state.cpu_wall_seconds[index] += wall_seconds;
        for mut proof in proofs {
            proof["binding"] = evidence.clone();
            state.cpu_byte_identity.push(proof);
        }
        state.cpu_sections.push(evidence);
        state.gpu_sections += 1;
    });
    result
}

#[cfg(test)]
mod tests {
    use super::{CpuWork, DrawCase, GpuLock, StageTruthCase, TruthCase, open, unlocked};
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
        let draw = DrawCase::new("tf32/first/b1/stage", "lstm.stack", 11, 589 * 256);
        let value = unlocked(CpuWork::Tf32Draws(&draw), || 91);
        assert_eq!(value, 91);
        assert_eq!(
            owner.evidence()["cpu_sections"][1],
            serde_json::json!({
                "work": "tf32_draws", "locked": false, "case": "tf32/first/b1/stage",
                "layer": "lstm.stack", "seed": 11, "length": 589 * 256,
            })
        );
        let proof_case = TruthCase::new(CudaMath::Fp32, 7, "fbank.dft");
        let value = unlocked(CpuWork::F64(&proof_case), || {
            competing
                .try_lock()
                .expect("proof also runs outside GPU lock");
            competing.unlock().expect("release competing lock");
            super::super::cpu::evaluate_mode(super::super::cpu::Mode::Verify, |mode| {
                super::super::cpu::ordered_map(mode, 31, |index| index as u8)
            })
        });
        assert_eq!(value, (0..31).collect::<Vec<u8>>());
        let proof = owner.evidence()["cpu_byte_identity"].clone();
        assert_eq!(proof[0]["binding"], owner.evidence()["cpu_sections"][2]);
        assert_eq!(proof[0]["binding"]["case"], proof_case.id());
        assert_eq!(proof[0]["serial_parallel_equal"], true);
        assert!(
            owner.evidence()["cpu_wall_seconds"]["f64"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        let audio = vec![0.25; 160_000];
        let stage = StageTruthCase::new("fp32/short/b1/stage/switched", 1, &audio);
        assert!(catch_unwind(AssertUnwindSafe(|| stage.binding())).is_err());
        assert!(
            catch_unwind(AssertUnwindSafe(
                || stage.bind(&[0.0], serde_json::json!({}))
            ))
            .is_err()
        );
        let values = unlocked(CpuWork::StageTruth(&stage), || {
            competing
                .try_lock()
                .expect("stage truth runs without GPU ownership");
            competing.unlock().expect("release competing lock");
            let values =
                super::super::cpu::evaluate_mode(super::super::cpu::Mode::Verify, |mode| {
                    super::super::cpu::ordered_map(mode, 998 * 80, |index| index as f64 * 0.125)
                });
            stage.bind(&values, super::super::reference::fbank_truth::constants());
            values
        });
        let binding = stage.binding();
        assert_eq!(binding["fixture_rows"], serde_json::json!([0]));
        assert_eq!(binding["input_sha256"], super::super::sha(&audio));
        assert_eq!(
            binding["truth_sha256"],
            super::super::reference::fbank_truth::sha(&values)
        );
        assert_eq!(binding["length"], 998 * 80);
        assert_eq!(binding["evaluation"], "complete-deterministic");
        let evidence = owner.evidence();
        assert_eq!(evidence["cpu_sections"][3]["work"], "stage_f64");
        assert_eq!(
            evidence["cpu_byte_identity"][1]["binding"],
            evidence["cpu_sections"][3]
        );
        assert_eq!(evidence["cpu_byte_identity"][1]["bytes"], 8 + 998 * 80 * 8);
        assert!(competing.try_lock().is_err());
        let panic = catch_unwind(AssertUnwindSafe(|| {
            unlocked(CpuWork::Tf32Draws(&draw), || panic!("CPU failure"));
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
