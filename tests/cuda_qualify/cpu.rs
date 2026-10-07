//! Ordered host-only evidence work, independent of all CUDA owners

use super::reference::Sample;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;

/// Execution policy for an immutable evidence snapshot
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Serial,
    Parallel,
    Verify,
}

impl Mode {
    pub(crate) fn from_environment() -> Self {
        match std::env::var("SPEAKRS_QUALIFY_CPU_MODE").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("parallel") => Self::Parallel,
            Ok("serial") => Self::Serial,
            Ok("verify") => Self::Verify,
            _ => panic!("CPU evidence mode must be serial, parallel or verify"),
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Parallel => "parallel",
            Self::Verify => "verify",
        }
    }
}

/// Exact bytes of the evidence, including sample indices and floating-point bits
pub(crate) trait Evidence {
    fn bytes(&self) -> Vec<u8>;
}

impl Evidence for Sample {
    fn bytes(&self) -> Vec<u8> {
        assert_eq!(self.indices.len(), self.values.len());
        let mut bytes = Vec::with_capacity(8 + self.indices.len() * 16);
        bytes.extend_from_slice(&(self.indices.len() as u64).to_le_bytes());
        for (&index, value) in self.indices.iter().zip(&self.values) {
            bytes.extend_from_slice(&(index as u64).to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

impl Evidence for Vec<u8> {
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.len());
        bytes.extend_from_slice(&(self.len() as u64).to_le_bytes());
        bytes.extend_from_slice(self);
        bytes
    }
}

thread_local! { static PROOFS: RefCell<Vec<Value>> = const { RefCell::new(Vec::new()) }; }

/// Drain proofs on the owning CPU thread before binding them to its lock section
pub(crate) fn take_proofs() -> Vec<Value> {
    PROOFS.with(|proofs| proofs.take())
}

/// Evaluate the same immutable snapshot with the selected policy
pub(crate) fn evaluate<R: Evidence>(compute: impl Fn(Mode) -> R) -> R {
    evaluate_mode(Mode::from_environment(), compute)
}

pub(super) fn evaluate_mode<R: Evidence>(mode: Mode, compute: impl Fn(Mode) -> R) -> R {
    if mode != Mode::Verify {
        return compute(mode);
    }
    let serial = compute(Mode::Serial).bytes();
    let parallel = compute(Mode::Parallel);
    let parallel_bytes = parallel.bytes();
    assert!(
        serial == parallel_bytes,
        "serial and parallel evidence bytes differ on one snapshot"
    );
    let digest = format!("{:x}", Sha256::digest(&serial));
    PROOFS.with(|proofs| {
        proofs.borrow_mut().push(json!({
            "serial_parallel_equal":true, "bytes":serial.len(), "sha256":digest,
        }))
    });
    parallel
}

/// Partition independent outputs and join in index order, never reduce in parallel
pub(crate) fn ordered_map<R: Send>(
    mode: Mode,
    len: usize,
    compute: impl Fn(usize) -> R + Sync,
) -> Vec<R> {
    assert_ne!(
        mode,
        Mode::Verify,
        "verify runs serial and parallel policies separately"
    );
    if mode == Mode::Serial || len == 0 {
        return (0..len).map(compute).collect();
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(len);
    let chunk = len.div_ceil(workers);
    std::thread::scope(|scope| {
        let compute = &compute;
        let handles: Vec<_> = (0..len)
            .step_by(chunk)
            .map(|start| {
                scope.spawn(move || {
                    (start..(start + chunk).min(len))
                        .map(compute)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("CPU evidence worker panicked"))
            .collect()
    })
}

#[test]
fn ordered_snapshot_proof_preserves_indices_and_float_bits() {
    let _ = take_proofs();
    let indices = [7, 2, 19, 0];
    let values = [f64::from_bits(0x7ff8_0000_0000_0042), -0.0, 3.25, 1.0];
    let result = evaluate_mode(Mode::Verify, |mode| Sample {
        indices: indices.to_vec(),
        values: ordered_map(mode, values.len(), |index| values[index]),
    });
    assert_eq!(result.indices, indices);
    assert_eq!(
        result
            .values
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    let proof = take_proofs();
    assert_eq!(proof.len(), 1);
    assert_eq!(proof[0]["bytes"], 72);
    assert_eq!(
        proof[0]["sha256"],
        format!("{:x}", Sha256::digest(result.bytes()))
    );
}

#[test]
fn proof_rejects_an_index_or_signed_zero_change() {
    for (indices, value) in [(vec![9], -0.0), (vec![8], 0.0)] {
        assert!(
            std::panic::catch_unwind(|| evaluate_mode(Mode::Verify, |mode| Sample {
                indices: if mode == Mode::Serial {
                    vec![8]
                } else {
                    indices.clone()
                },
                values: vec![if mode == Mode::Serial { -0.0 } else { value }],
            }))
            .is_err()
        );
    }
}
