# Phase 2c-1 proof carry-forward

## Rule

Phase 2c-1 changes infrastructure, not candidate arithmetic. Its proof requires
three full Library controls, one live run of each of the 13 standard mutant kinds,
and matched StageTail control/fault pairs for ResNet and SincNet. Each record
keeps its actual device and compiled lock. The reduced set does not qualify a
new candidate or a different device. Run all 39 standard mutants at the final
phase 2c seal, after phase 2c-2. No gate, bound, sample count or seed changes.
A mutant that does not reach its intended gate stops this phase.

## Exact source evidence

The audited baseline is `8952f419928d91ecb67fadd8715b3ef19ede63fe`. The collection diff ends at `cc9b225e7d66ef3bd2f486a700e478d1c66b329f`.
`evidence/phase2c1-gate-decisions.diff` is empty: `gates.py` and `verdict.py`
have no changed bytes. Its SHA256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
`evidence/phase2c1-collection-injection.diff.gz` holds every changed Rust, Python,
PTX, manifest, shell and Cargo path that can collect or inject evidence, plus the
changed loader, selection and kernel tooling. Its SHA256 is `bd855bdc418c52795e03a579e62ccf9ef1f183c18a651dab287695006a266a7d`.
Decompress it to inspect the exact diff. The JSON plan lists each exact path. The diff excludes no changed injection
function. Production PTX and kernel arithmetic have no changes. The control PTX
only receives the same host-prepared integer draw choices. Frozen vectors and
all eight seeds verify the old integer hash and raw-bit ULP rule.

The full controls cover all numeric, timing, paired, profile and sanitizer paths:
LSTM and ResNet on cc 12.0; SincNet on cc 8.9. Their raw hashes are pinned in the
JSON plan. They prove device/artifact evidence and unlocked f64/draw sections on
both devices. The actual cc 12.0 and cc 8.9 cubin/JIT all-entry proofs cover the
unchanged loader bytes. Host tests cover all embedded kernel names, exact artifact
key refusal, failed-cubin JIT fallback, record negatives and typed overrides.
Static lints cover the exact eight-site shared-address fixture.

## Live mutant allocation

| Mutant | Area / box | Changed collection or injection path |
| --- | --- | --- |
| Lookup | resnet / 5060 | Host weight snapshots and case-bound secret f64 truth in embedding.rs and probe.rs |
| StageAccuracy | lstm / 4060 | TF32 stage truth, cloned LSTM weights, and host-prepared perturbation draws |
| Precision | lstm / 4060 | Input restoration across isolated LSTM plan selection and unlocked truth work |
| Shape | sincnet / 4060 | Batch coverage and direct isolated SincNet selection through the shared plan owner |
| Tail | sincnet / 4060 | Partial-batch output path through Sinc producer and locked pool consumer |
| UninitShared | lstm / 4060 | Recorded PTX identity and shared-load inspection after the artifact evidence change |
| Atomic | lstm / 4060 | Launched entry/atomic evidence with changed module metadata and GPU ownership |
| Fallback | lstm / 4060 | Nested Library-call evidence in the changed segmentation selection owner |
| Unlisted | sincnet / 4060 | Artifact-aware loaded-entry allow-list and unrecorded launch evidence |
| Unscoped | sincnet / 4060 | Scope/launch evidence across the changed module cache and cleanup owner |
| PhaseCheat | lstm / 4060 | Numeric-to-timing output identity after case-bound CPU truth and lock release |
| Slow | sincnet / 4060 | Separate-process operator timing with the changed selection and lock owners |
| StageSlow | sincnet / 4060 | Same-process paired replay and GPU-state cleanup after CPU lock release |

## Path coverage

- `embedding.rs`, its dispatcher, and test `embedding.rs`: ResNet Library, Lookup,
  and the ResNet matched pair cover Library selection, host snapshots and real
  candidate selection
- `segmentation.rs`, its dispatcher, and test `segmentation.rs`: LSTM and SincNet
  Library controls, the allocated mutants, and the SincNet matched pair cover both
  host reference variants, isolated selection and Library diagnostics
- `probe.rs`, `lock.rs`, `test_support.rs`, and control PTX: all three controls
  and the allocated truth, scope, PTX, timing and paired faults cover preparation,
  artifact identity, lock ownership and cleanup; lock panic and exact draw-vector
  host tests cover paths not caused by the live controls
- `kernels.rs`, `runtime.rs`, manifests, area kernel owners and host inventory:
  exact-device cubin/JIT all-entry proofs and Mac inventory tests cover the load
  path; controls pin the bytes actually embedded in each measured binary
- `implementation.rs` and its override types: live controls, Lookup and matched
  real-candidate pairs cover plan intent; host tests cover exact artifact keys,
  unsupported tuples, driver-only errors and speed-unqualified overrides
- `qualify.py`, `artifacts.py`, and `records.py`: all controls and mutants exercise
  collection and binding; strict Python negatives and both table modes cover
  stale identities, missing fields, legacy mapping and embed-mask loadability
- Candidate host inventory constants, kernel tooling, setup script, and other
  changed build/test paths have no altered gate decision or fault arithmetic;
  kernel check/rebuild, lint fixtures, host tests and restart-path checks cover them

The exact diff and file list permit an audit of this mapping. A mapping is not a
passed result. Final proof receipts must name each actual mutant record, its
intended caught gate, and both matched comparisons. The old 39-run phase 2b
records remain unchanged and are carried forward only for unchanged gate logic.
