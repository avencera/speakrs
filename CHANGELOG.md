# Changelog

## [unreleased]

- Add `QueueConfig` so queue channel capacity is configurable at construction (default 64; capacity 0 is rejected)
- Add non-blocking `QueueSender::try_push`, which returns typed `QueueError::Full` with the rejected request when the queue is at capacity
- Remove `QueueSender::push`; callers submit work with `try_push`
- Remove `QueueError::Terminal`; a finished worker reports `Closed` or `WorkerPanicked`
- Replace split `PipelineConfig` clustering and activity fields with checked `ClusteringConfig` and `ActivityCleanup`

## [0.5.0] - 2026-07-07

- Clean up the public API for 0.5.0: expose tuning config types, make selected enums non-exhaustive, rename custom segment conversion to `to_segments_with`, and make `Segment` display human-readable text instead of RTTM.
- Return errors instead of panicking when queues end unexpectedly or ORT returns malformed inference output.
- Modernize shipped examples around `OwnedDiarizationPipeline`/`PipelineBuilder`, including a queued sender-clone example.
- Add dependency policy, MSRV, no-default-features, docs, README, package, and release-readiness checks.

## [0.4.2] - 2026-05-29

- Fix the published crate manifest so `ort`, `crossbeam-channel`, `rayon`, `thiserror`, `tracing`, and `libloading` are available on `x86_64`
- Add GitHub Actions checks for the packaged crate artifact to catch manifest dependency-scope regressions before publishing

## [0.4.1] - 2026-05-26

- Fix embedding weight preparation so shorter masks clear stale tail values before reuse
- Validate BLAS backend feature selection at compile time, including no-backend and multi-backend configurations
- Reject `coreml` builds on unsupported targets at compile time and clarify macOS CoreML support in the docs
- Keep only the fixture files still used by the crate tests

## [0.4.0] - 2026-04-15

- Default `ndarray-linalg` to Intel MKL on `x86_64` and OpenBLAS elsewhere, which avoids OpenBLAS CPU-target issues on `x86_64`
- Add explicit `intel-mkl`, `openblas-static`, and `openblas-system` feature flags for users who disable default features and need to choose a BLAS backend
- Route PLDA linear algebra through an internal backend shim, update the generated docs for the new backend options, and remove the stale OpenBLAS override from the GPU Docker build

## [0.3.2] - 2026-04-14

- Require native CoreML bundles for `CoreMl` and `CoreMlFast` modes instead of silently falling back to ORT CPU, with clearer errors for missing or invalid compiled assets and updated model manifests for segmentation, fbank, tail, and chunk models
- Reduce default `info` log noise across the diarization pipeline by moving stage-completion logs to `debug`
- Upgrade dependencies

## [0.3.1] - 2026-03-26

- Fix docs.rs build: replace removed `doc_auto_cfg` feature with `doc_cfg`

## [0.3.0] - 2026-03-26

- Split `QueuedDiarizationPipeline` into `QueueSender` and `QueueReceiver`, enabling cloneable senders for multi-threaded push
- Add `QueueError::Closed` variant to distinguish clean shutdown from worker panics
- `QueueReceiver` now joins the worker thread on drain, surfacing panics as `QueueError::WorkerPanicked`
- Remove `push_batch` and `finish` in favor of drop-based signaling and iterator drain
- Move `make_exclusive` from a free function to a method on `DiscreteDiarization`
- Move the main documentation (benchmarks, pipeline diagram, comparison tables) into `lib.rs` and generate the README with `cargo-rdme`
- Fix `repository` URL in Cargo.toml pointing to wrong GitHub org
