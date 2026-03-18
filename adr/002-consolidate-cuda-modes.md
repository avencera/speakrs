# Consolidate CUDA Execution Modes

## Date: 2026-03-17

## Context

Benchmarks on VoxConverse dev (216 files, 1218 min audio) compared two CUDA modes:

- **Old Cuda**: fused model on GPU, sequential segmentation-then-embedding
- **CudaHybrid**: fused model on GPU, concurrent seg+emb via crossbeam channel

Despite the naming, CudaHybrid was already using the fused embedding model on GPU — not split-tail. The `use_split_backend` flag in `embedding.rs` only activated for CoreMl/CoreMlFast modes. The entire speed win came from the concurrent crossbeam pipeline overlapping segmentation and embedding.

## Benchmark Results

| Subset | Files | Audio | Old Cuda (seq) | CudaHybrid (concurrent) | Speedup |
|--------|-------|-------|----------------|------------------------|---------|
| 10 min | 6 | 10 min | 17.3s | 10.7s | 1.62x |
| 60 min | 29 | 60 min | 248s | 169s | 1.47x |
| Full | 216 | 1218 min | 2243s | 1430s | 1.57x |

DER is identical at 7.0% for both modes across all subsets.

For reference, pyannote CUDA on the full dataset: 7.2% DER, 1450s. CudaHybrid matches pyannote throughput while beating its DER.

## Key Findings

1. **Concurrent pipeline always wins.** The crossbeam channel overlaps segmentation and embedding, giving 1.5-1.6x speedup with no DER impact

2. **Old CudaHybrid was already using fused model.** The split-tail path (CPU fbank + GPU tail) was only wired for CoreML modes. CudaHybrid was just "fused model + concurrent pipeline" — a misnomer

3. **Sequential mode is strictly dominated.** Same DER, always slower. No hardware profile benefits from it

4. **Step 2 has negligible DER impact.** SpeakerKit/SDBench stride ablation confirms stride 2 = negligible DER degradation, matching our CoreMlFast findings (ADR 001: F16+1s and F16+2s produce nearly identical DER)

## Decision

### Remove old sequential Cuda mode
Strictly dominated — same DER, 1.5x slower.

### Rename CudaHybrid → Cuda
This is now the default CUDA mode: fused model on GPU, concurrent pipeline, 1s step.

### ~~Add CudaHybrid (true split-tail)~~ — Removed

Added as a benchmark experiment: CPU fbank extraction + CUDA ResNet tail, concurrent pipeline, 1s step. Benchmarks showed it was ~8x slower per-file than fused CUDA on short runs (10 min test: 161.8s vs 19.6s). The CPU fbank extraction is a bottleneck that negates any benefit from offloading the tail. Mode removed entirely in favor of fused Cuda.

### Add CudaFast
Fused model on GPU, concurrent pipeline, 2s step. Same relationship as CoreMlFast to CoreMl — trades temporal resolution for ~2x speed.

### Summary of new modes

| Mode | Embedding | Pipeline | Step | Use Case |
|------|-----------|----------|------|----------|
| Cuda | Fused GPU | Concurrent | 1s | Default CUDA — best accuracy |
| CudaFast | Fused GPU | Concurrent | 2s | Speed-optimized, acceptable DER |

All three share the same CUDA EP config: tf32=false, Exhaustive conv search, max workspace, SameAsRequested arena strategy.

## Supersedes

- `_notes/cuda-execution-modes.md` — original design doc for the two-mode split
