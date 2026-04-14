# Changelog

## [unreleased]

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
- Move full documentation (benchmarks, pipeline diagram, comparison tables) into `lib.rs` and generate README via `cargo-rdme`
- Fix `repository` URL in Cargo.toml pointing to wrong GitHub org
