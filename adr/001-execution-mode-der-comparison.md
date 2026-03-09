# Execution Mode DER Comparison

## Date: 2026-03-09

## Context

We expanded DER testing from 20 to 39 VoxConverse files (~53 min audio) and compared
three CoreML execution modes. The original 20-file set showed FP32 and FP16 producing
near-identical DER, but broader coverage revealed significant divergence.

## Results (39 files)

| Mode              | Avg DER | Total Time | Notes                          |
|-------------------|---------|------------|--------------------------------|
| FP32 + 1s step    | 6.8%    | 84.2s      | Best accuracy                  |
| FP16 + 1s step    | 27.1%   | 56.5s      | Same step, worse DER           |
| Lite (FP16 + 2s)  | 27.0%   | 29.9s      | Same DER as F16+1s, 2x faster  |

## Key Findings

1. **FP16 embedding drift breaks clustering on harder files.** Clean files (few speakers,
   clear turns) produce identical DER across all modes. But files with many speakers or
   overlapping speech show massive confusion spikes (40-77% DER) under FP16. The
   root cause is FP16 embedding tail precision — small numeric differences compound
   through PLDA + VBx clustering.

2. **The 2s step doesn't degrade DER.** F16+1s and F16+2s produce nearly identical
   avg DER (27.1% vs 27.0%). The coarser step skips chunks but doesn't introduce
   additional clustering errors.

3. **F16+1s has no value.** Same DER as Lite but twice as slow. It offers neither the
   accuracy of FP32 nor the speed of Lite.

## Decision

Support two execution modes:

- **CoreMl (FP32 + 1s step)** — default, best accuracy (6.8% avg DER)
- **CoreMlLite (FP16 + 2s step)** — fast mode, ~3x faster, acceptable for real-time/preview

Remove the FP16+1s combination — it's strictly dominated by the other two modes.

## Per-file DER comparison

Files where FP32 and FP16 diverge most (confusion-dominated):

| File  | FP32  | F16+1s | Lite  |
|-------|-------|--------|-------|
| qpylu | 5.5%  | 39.0%  | 38.9% |
| fxgvy | 2.9%  | 44.2%  | 44.3% |
| syiwe | 2.7%  | 49.1%  | 56.2% |
| hiyis | 0.6%  | 44.7%  | 42.5% |
| lknjp | 7.8%  | 62.2%  | 61.1% |
| iqtde | 0.4%  | 62.7%  | 62.7% |
| jsmbi | 4.4%  | 66.4%  | 65.5% |
| exymw | 9.9%  | 77.3%  | 77.0% |
| qjgpl | 42.5% | 57.2%  | 60.2% |

Files where all modes agree (clean speech):

| File  | FP32  | F16+1s | Lite  |
|-------|-------|--------|-------|
| hqyok | 4.3%  | 4.3%   | 4.3%  |
| tfvyr | 2.0%  | 2.0%   | 2.0%  |
| qrzjk | 0.8%  | 0.8%   | 0.8%  |
| usbgm | 0.4%  | 0.4%   | 0.4%  |
| qydmg | 0.1%  | 0.1%   | 0.1%  |
| ysgbf | 6.1%  | 6.1%   | 6.9%  |
| atgpi | 5.0%  | 6.1%   | 5.0%  |
