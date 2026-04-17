# Changelog

## [unreleased]

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
