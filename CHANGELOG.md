# Changelog

## [Unreleased]

- Split `QueuedDiarizationPipeline` into `QueueSender` and `QueueReceiver`, enabling cloneable senders for multi-threaded push
- Add `QueueError::Closed` variant to distinguish clean shutdown from worker panics
- `QueueReceiver` now joins the worker thread on drain, surfacing panics as `QueueError::WorkerPanicked`
- Remove `push_batch` and `finish` in favor of drop-based signaling and iterator drain
