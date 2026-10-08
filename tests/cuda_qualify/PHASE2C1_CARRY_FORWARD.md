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

The audited baseline is `8952f419928d91ecb67fadd8715b3ef19ede63fe`. The collection diff ends at `4fc0a5fac4297599151b1d1c8bc1df7859161151`.
`evidence/phase2c1-gate-decisions.diff` is empty: `gates.py` and `verdict.py`
have no changed bytes. Its SHA256 is `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
`evidence/phase2c1-collection-injection.diff.gz` holds every changed Rust, Python,
PTX, manifest, shell and Cargo path that can collect or inject evidence, plus the
changed loader, selection and kernel tooling. The decompressed diff SHA256 is `a2506a7dbe2f949abdd2d386ae649969f4e3d42d7a81cfef37eb58c094737b17`.
The gzip SHA256 is `94bbca2974d14e011e9db25c96bab7d5e6142f6729ddbabf082c0b5623d01669`.
Decompress it to inspect the exact diff. The JSON plan lists each exact path. The diff excludes no changed injection
function. Production PTX and kernel arithmetic have no changes. The control PTX
only receives the same host-prepared integer draw choices. Frozen vectors and
all eight seeds verify the old integer hash and raw-bit ULP rule.

The full controls cover all numeric, timing, paired, profile and sanitizer paths:
LSTM and ResNet on cc 12.0; SincNet on cc 8.9. Their raw hashes are pinned in the
JSON plan. They prove device/artifact evidence and unlocked f64/draw sections on
both devices. The actual cc 12.0 and cc 8.9 cubin/JIT all-entry proofs cover the
unchanged artifact bytes and first-seal load policy. The short default-production
follow-up proof below covers the new strict loader policy. Host tests cover all
embedded kernel names, exact artifact key refusal, requested-artifact refusal
without format substitution, record negatives and typed overrides.
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

## Completed live proof

`evidence/phase2c1-results.json` pins three passed full Library controls, all 13
caught standard mutant kinds, and both matched comparisons. It keeps each raw
record hash, original compiled lock, exact device and artifact proof receipt.
`PROOF.json` and `AUDIT_PROOF.json` link that receipt without changing their
historical phase 2b results. Full command receipts, raw archives, comparison sets,
noise flips and timing spreads remain in the hash-addressed outside-tree evidence
store named by the receipt. Raw JSON records also remain in the qualification
record cache. No record is relabelled with the final owner lock.

The SincNet matched control now passes every check. Its fault fails only the
intended mixed-b32 margin check. The earlier rejected control remains evidence:
three b1 paired checks had tight intervals below 1, and its TF32 process loaded
SincNet with no declared triple. These were real failures, not dismissed noise.
The common selection owner already resolves tuple intent before module loading.
The missed path was coverage discovery: it loaded PTX before that owner was
called. The declaration now uses `AreaTarget` and the embedded PTX hash without
loading an artifact or making a token. Actual plans still need the successfully
loaded artifact and the exact production token. A Mac test calls the same Sinc
plan owner for both fixtures, covered FP32, uncovered batches, TF32 and cc 8.9.

The correction changed no stage graph, candidate code, PTX bytes or paired replay
schedule. Both old and new Sinc controls used the same PTX-JIT hashes and the
same 32 recorded graph scope/kernel sequences. Selection and key work happen
before capture; timed replay only launches the captured graph. Paired processes
hold the parent lock throughout. CPU unlocks occur in separate numeric children.
The corrected b1 ratios are about 1.010, versus about 0.998 in the failed record.
This isolates eager module loading during coverage discovery as the changed
execution path. The root attributes the earlier b1 slowdown to that fault, fixed
by `79de5eb`; no cubin-versus-JIT attribution run is needed.
The driver allocation/cache mechanism is not proved; retain this as an audit
residual, not a claim that the old record passed.

The ResNet pair keeps its real measured 292 lock. Both raw records have the same
stable-module failure from the old coverage-discovery path. The unchanged matched
criterion passes: the intended control margin passes, the fault margin fails,
and no new effective or non-timing failure appears. This is a fault-injection
proof, not a passing control verdict or new production qualification. The common
coverage fix is exercised by the new Sinc pair and both-area host coverage tests;
actual ResNet candidate arithmetic and the captured timing path are unchanged.

The three approved Library controls run the unchanged Library choice, not this
fixture-only branch. The 13 standard mutants also do not enter that branch.
Their original hashes and compiled locks remain valid proof of those unchanged
paths. The frozen 93 control and its LSTM projection baseline keep their original
archive/device identity. The later 03 source archive is used for the corrected
matched pair; no projection baseline is relabelled or assumed for that archive.
Any later control that needs a different baseline identity must collect it.

The exact cubin/JIT entry proofs keep their old f84 lock. At the first seal,
PTX, cubin, manifest, runtime-loader and `load_artifact` source bytes had no changes
since those proofs. Those later changes concerned unloaded intent and fixture
declarations. The production artifact ownership follow-up below changes the loader
policy, not the PTX, cubin or manifest bytes.
Host tests cover their key checks; there is no new binary-load acceptance claim.
Run the full 39 standard mutants at the final phase 2c seal after phase 2c-2.


## Production artifact ownership follow-up

The first seal used cubin-first loading even for legacy JIT-qualified production
entries. This changed production selection to Library fallback on cc 12.0 and
was a regression. The follow-up makes the production record own the artifact
request before loading. Legacy pins again select JIT without a diagnostic override.
Explicit qualification requests resolve their declared embedded artifact before
loading; a driver refusal never changes the artifact being measured. Gates,
bounds, samples, seeds, production coverage and production artifact pins do not
change. The new refusal policy affects failed loads only, not any retained successful
control or mutant. The short default-production GPU proof passed for this
follow-up on cc 12.0. Earlier preflight receipts prove the unchanged artifact
bytes, not the new loader policy. They retain their original source and compiled-lock identities.

The default build selects `sincnet.conv0.abs_pool` and `resnet.layer1.0.conv1`,
batch 1 FP32, with their PR #36 sm75 PTX-JIT pins. No force-JIT or forced-tier
environment was present; exactly two modules were recorded, both JIT. The proof
ran under the shared flock, not a timing window.
`evidence/phase2c1-production-artifacts.json` pins the actual measured 42 lock,
source commit, two successful production selections, 433 source hashes, command
logs and independent Mac receipt. The final owner lock does not relabel that binary.
The host test covers all 52 current production tuples, exact cubin requests, driver
refusal without substitution, conflicting owners and the typed Library policy.
Both table modes and all required Mac checks pass.

## Grok audit fixes and new controls

`evidence/phase2c1-grok-fixes.json` pins the five-area default-build functional
proof and two new full Library controls. The cc 12.0 proof records PTX JIT for
fbank, embedding and segmentation and the pinned PR #36 SincNet and ResNet
candidates, with no cubin or force-JIT environment. Typed always-on owners keep
PR #36 artifact policy on every device. Host tests cover cc 12.0 and cc 8.9;
the owner does not branch on device beyond existing tier/capability resolution.
The functional proof keeps its actual 39102c3 source and a2 development lock.
Its exact carry-forward diff to the control source contains only lint, fixture,
control archive pin and lock changes, not runtime or artifact bytes.

The new SincNet and ResNet controls use 4fc0a5f and the real 1b development lock.
Both passed all numeric, timing, paired, profile and sanitizer phases: 214 and
1,159 checks, with no failed check and no replacement acceptance. Both ran on
the 5060 Ti in the same root-granted quiet window. Each finished record and its
logs were copied and independently checked before the next control started.
The actual control archive is 1914adb9; the old d701, 03 and 93 archives keep
their real identities. The final owner lock does not relabel any of these runs.

These controls cover the changed gate-bearing evidence paths: TF32 draw sections
now bind to case, layer, seed and length; each timed process prepares all required
modules before clocks start. Runtime and harness module-load guards cover both
clock and event intervals. The control record binds every candidate-area PTX,
cubin and manifest even when no candidate module loads. The new table negatives
reject stale unloaded artifacts and stale shipped cubin pins in both modes.

No draw generation, f64 arithmetic, layer seeds, sample count, timing schedule,
replay, injection arithmetic or numeric bound changes. `gates.py` and `verdict.py`
still have the exact empty diff above. CPU-evidence validation and artifact
binding are stricter, not waived. LSTM has concrete geometry and binding negatives
and unchanged integer-draw golden vectors. Fresh review found no extra LSTM
live-path gap; the root required new full controls for SincNet and ResNet only.
A later full LSTM control needs a baseline measured with its actual archive; the
93 archive projection baseline cannot be relabelled as the new control.

Fresh review found three more taint-lint gaps: dollar register names, wide/vector
result widths, and narrowing below 32 bits. The fix has a 12-site failing fixture,
six passing lint tests, all 155 xtask tests, the unchanged safe fixture, and a
successful check of every committed PTX and cubin. The original eight-site LSTM
fixture is unchanged. This is static tooling proof, not new kernel qualification.

The first new control staging used an extra directory level. The unchanged path
guard refused it before GPU work, so it has no control verdict. Its failed logs
remain pinned. The successful controls use a real `/workspace/<task>/tree` and
real owned disk targets; the proof scripts check this path before qualification.
The guard was not changed. No new full matrix is claimed: all 39 standard mutants
remain required at the final phase 2c seal after phase 2c-2. Always-on cubin
adoption remains phase 5, with end-to-end A/B startup, RTFx and bit-identical outputs.
