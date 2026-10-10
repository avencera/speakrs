# Changelog

## [0.6.0] - 2026-10-09

- Make ONNX Runtime optional (breaking): no inference backend is enabled by default, so enable `coreml` (macOS), `cuda` (NVIDIA), `migraphx` (AMD), or the new `cpu` feature; a build without one fails with a compile error naming these choices, and a `coreml`-only build no longer compiles, links, or downloads ONNX Runtime
- Run `ExecutionMode::Cpu` with native Rust PyanNet, WeSpeaker, and filterbank models instead of ONNX Runtime (breaking); the `cpu` feature no longer compiles, links, or downloads ONNX Runtime, and CPU shared clones reuse immutable weights with private workspaces
- Load the two native safetensors weight files, PLDA files, and embedding metadata for CPU mode; canonical ONNX paths remain family selectors, but arbitrary ONNX models are not supported. MIGraphX still uses ONNX Runtime, and `load-dynamic` retains the optional external ONNX session helpers
- Require the `cpu` feature for `ExecutionMode::Cpu`, including `SegmentationModel::new` and `EmbeddingModel::new`; `load-dynamic` now needs `cpu` or `migraphx` alongside it, and a build with `load-dynamic` but neither fails with a compile error
- Run `ExecutionMode::Cuda` and `CudaFast` on a native NVIDIA backend instead of ONNX Runtime (breaking): the `cuda` feature no longer enables ONNX Runtime and builds without a CUDA toolkit, because it loads the driver, cuBLAS, and cuDNN 9 at run time; it needs a Turing (compute capability 7.5) or newer GPU. The filterbank, ResNet34 embedding, and segmentation all run on the GPU, and the filterbank features stay on the device. On an RTX 4090, `cuda` runs VoxConverse dev at 978x real time, up from 59x with the ONNX Runtime backend, with matching DER
- Load `segmentation-3.0.safetensors` and `wespeaker-multimask-tail.safetensors` in CUDA modes instead of ONNX models (breaking); `ModelManager` downloads only those, the PLDA files, and the embedding metadata for CUDA modes, `scripts/cuda/export_weights.py --runtime-assets <dir>` exports them, and a missing file is a `ModelLoadError::MissingCudaWeights` or `CudaAssetsUnavailable` that names the export command
- Run every layer that has a speakrs CUDA kernel on that kernel, on every GPU; cuDNN and cuBLAS run only the layers without one. The kernels include TF32 tensor-core and Winograd ResNet trunk convolutions, FP16 tensor-core trunk convolutions on Turing and Ada in TF32 mode, a tiled LSTM, and an FFT filterbank
- Pick each layer's kernel by batch size and math mode from, in order, a tune file, a built-in recipe measured on the T4, A100, RTX 4060 Ti, RTX 4090, or RTX 5060 Ti, and a default for the device class; `SPEAKRS_CUDA_FORCE_LIBRARY=1` forces cuDNN and cuBLAS where the library has an implementation
- Add the `cuda-sm75`, `cuda-sm80`, `cuda-sm90`, and `cuda-sm120` GPU target features and the `cuda-rtx20`, `cuda-rtx30`, `cuda-rtx40`, `cuda-a100`, and `cuda-rtx50` aliases; `cuda` enables every target plus cuDNN and cuBLAS, while a target feature with `default-features = false` builds a driver-only backend that never links or loads cuDNN or cuBLAS, and `SPEAKRS_CUDA_PTX_TIER` forces a lower compiled-in tier
- Add the `speakrs cuda tune` command and the `speakrs::inference::cuda::tune_cuda` API, which time each layer's kernel candidates on the current GPU and write a tune file that later runs load automatically; tune files are keyed by GPU, driver, kernel artifacts, speakrs version, and accuracy policy, `--dry-run` writes nothing, `--include-library` also compares cuDNN and cuBLAS, and `SPEAKRS_CUDA_TUNE_FILE` sets a custom path
- Add CUDA options to `RuntimeConfig`: `cuda_segmentation_math` (`CudaMath::Fp32` by default; TF32 worsened one VoxConverse-dev file's DER by 4.5 points) and `cuda_embedding_math` (`CudaMath::Tf32` by default, DER-neutral on VoxConverse-dev), `cuda_lstm_algorithm` (`CudaLstmAlgorithm::PersistStaticSmallH` by default; `PersistDynamic` falls back to `Standard` with a warning when NVRTC is missing), and `cuda_graphs` (`CudaGraphs::Enabled` by default); the filterbank always runs in FP32
- Add `SegmentationModel::with_mode_and_config`, so segmentation picks up the CUDA options outside the pipeline builder
- Add `InferenceError::Cuda`, `ModelLoadError::Cuda`, and the public `CudaError`, `CudaLibrary`, `GeometryError`, `WeightFault`, `ComputeCapability`, and `PtxTier` types
- `OwnedDiarizationPipeline::clone_shared` in CUDA modes loads a second copy of the models on its own CUDA stream instead of sharing ONNX Runtime sessions
- `with_execution_mode` maps the CUDA modes to the ONNX Runtime CPU provider, as it already did for CoreML, because speakrs no longer builds ONNX Runtime CUDA sessions
- Overlap multi-file batches with inference: up to four workers decode audio ahead of the current file, and clustering for one file runs while the next file is in inference. Results keep input order, a failed stream returns no partial batch, and the pipeline stays reusable
- Decode a lookahead file early only when it is a regular file, so pipes and devices are read in turn, and reject WAV `fmt` chunks larger than 64 KiB
- Remove the `default-linalg`, `intel-mkl`, `openblas-static`, and `openblas-system` features: PLDA setup now uses a small built-in Rust solver, so builds no longer link MKL or OpenBLAS
- Replace `ort::Error` in the public API with the new `InferenceError`: `EmbeddingModel` methods return it, `SegmentationModel::run` returns `SegmentationError`, and `SegmentationError::Ort` and `PipelineError::Ort` become `SegmentationError::Inference` and `PipelineError::Inference`; native CoreML failures surface as the typed `CoreMlError` and shape failures as `TensorShapeError`
- Gate `ModelLoadError::Ort`, `ModelLoadError::Runtime`, `OrtRuntimeError`, `DynamicRuntimeError`, and `with_execution_mode` behind the ONNX Runtime backends; `with_execution_mode` now returns `ModelLoadError`
- Stop building ONNX Runtime sessions in CoreML modes: `EmbeddingModel::embed`, `embed_masked`, and `embed_batch` run on the native CoreML filterbank and tail, following `RuntimeConfig::chunk_emb_compute_units` (use `CpuOnly` for the closest match to the CPU path)
- Stop downloading ONNX files for CoreML modes in `ModelManager`; CoreML modes need only the compiled bundles, PLDA files, and embedding metadata
- Add `QueueConfig` so queue channel capacity is configurable at construction (default 64; capacity 0 is rejected)
- Add non-blocking `QueueSender::try_push`, which returns typed `QueueError::Full` with the rejected request when the queue is at capacity
- Remove `QueueSender::push`; callers submit work with `try_push`
- Remove `QueueError::Terminal`; a finished worker reports `Closed` or `WorkerPanicked`
- Replace split `PipelineConfig` clustering and activity fields with checked `ClusteringConfig` and `ActivityCleanup`; `BinarizeConfig` and the free `binarize` function are removed
- Make `AhcConfig` and `VbxConfig` fields private behind checked constructors and accessors
- Return `Result` from `ModelBundle::from_dir` and `PipelineBuilder::from_dir`, and report `ModelBundle` loading failures as `ModelLoadError` instead of `hf_hub` errors
- Remove the unused `RuntimeConfig::chunk_emb_workers` field
- Add activation-aware `DiarizationResult::exclusive_segments` and remove the obsolete `DiscreteDiarization::make_exclusive` binary tie-breaker
- Fix exclusive diarization so overlaps resolve by activation score instead of cluster index
- Add checked filterbank session-pool and thread settings to `RuntimeConfig`
- Add `OwnedDiarizationPipeline::clone_shared` for non-CoreML concurrent pipelines that share model sessions
- Pool the ONNX Runtime filterbank sessions, and speed up clustering with vectorized VBx and bounded parallel AHC distances (`SPEAKRS_AHC_THREADS` overrides the worker count)
- Use available cores for embedding ONNX session intra-op threads
- Fix multi-mask embedding batching so the batched tail model loads at the multi-mask batch size
- Fix chunk embedding returning all-zero filterbank features when the 30s filterbank model is absent
- Pad non-empty audio shorter than one window into a single segmentation window instead of producing no output
- Match CoreML diarization accuracy with CUDA by fixing chunk filterbank stitching and using a 1 second segmentation step
- Validate tensor shapes, PLDA dimensions, and inference output geometry before computation
- Run the GPU benchmark without S3 credentials: fetch public models and datasets, keep results locally, and print DER and RTFx in the captured log

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
