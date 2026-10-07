# Locked CUDA qualification

The harness decides whether a custom kernel may replace a cuDNN or cuBLAS call at one of
three boundaries: the 14 eligible ResNet 3x3 convolutions, the Sinc producer, and the
complete four-layer bidirectional LSTM stack. New results authorize a replacement only when `accepts_replacement: true`.
The production table check also evaluates the raw checks in pinned records with the
locked evaluator. A blocked raw timing check is not an acceptance by itself.

## Running

Run from your own task tree on the GPU box. Any `/workspace/<task>/tree` whose locked
files match the owner's digest works; results go to `/workspace/<task>/results` and
builds to `/workspace/<task>/target`.

```sh
cd /workspace/<task>/tree
source /workspace/env.sh
export SPEAKRS_QUALIFY_OWNER_DIGEST=<digest-kept-outside-the-tree>
export SPEAKRS_CUDA_PTX_TIER=sm75
cargo xtask cuda-qualify <resnet|lstm|sincnet> <implementation>
```

Implementations are `Library` (the sanity control, never a replacement), `Oxide` (the
registered candidate) and the thirteen planted faults listed under
[Mutation proof](#mutation-proof). Do not wrap the command in `flock`: the harness takes
`/workspace/gpu-bench.lock` for every GPU child process and refuses unlocked ones, so
an outer `flock` on the same file would deadlock. Numeric children own the lock.
They release it for CPU f64 truth and TF32 draw preparation, then take it again for
GPU replay. Other phases and tool launches use the parent lock. Each timing and
paired process covers one math mode, so kernel workers can interleave.

Exit codes: 0 pass, 1 rejected, 2 refused invocation, 3 blocked (the evidence cannot
decide), 4 a mutant escaped its intended check. `cargo xtask` fails on anything but 0.
`SPEAKRS_QUALIFY_DIAGNOSTICS_ONLY=1` skips Compute Sanitizer, always writes
**blocked** and cannot authorize a replacement.

## Candidate interface

A candidate is a plan type behind one trait per boundary, defined in the locked
`src/inference/cuda/candidate.rs`. Locked dispatch (`embedding/dispatch.rs` and
`segmentation/dispatch.rs`) creates the plan when a batch class is set up, outside
every timed and traced interval, then calls `enqueue` inside a harness scope it owns.
The candidate never chooses its scope, its timing or its fallback.

### Where candidate files go

| What | Path |
| --- | --- |
| Host plan for ResNet convolutions | `src/inference/cuda/candidate/conv.rs` and `candidate/conv/**` |
| Host plan for the Sinc producer | `src/inference/cuda/candidate/sinc.rs` and `candidate/sinc/**` |
| Host plan for the LSTM stack | `src/inference/cuda/candidate/lstm.rs` and `candidate/lstm/**` |
| Device kernels | `crates/speakrs-cuda-kernels/src/{resnet,sincnet,lstm}.rs` and `src/{resnet,sincnet,lstm}/**` |
| Committed PTX | `src/inference/cuda/ptx/{resnet,sincnet,lstm}.<tier>.ptx` and `.manifest`, from `cargo xtask cuda-kernels build <area>` |

Each host file keeps its `pub(crate) struct Oxide` and implements the trait; split
larger code into submodules under the matching directory. Load kernels only with
`runtime.load_kernels(KernelModule::Resnet)` (or `Lstm`, `Sincnet`) in `plan`, then
`kernels.function("<entry>")`; the locked loader records the exact PTX bytes. Prefix
entry names with `spk_<area>_`, as in `spk_resnet_conv3x3_c32`. Everything else under `src/inference/cuda/` is locked.
Qualification builds use `--features cuda`: all compiled tiers and the libraries are
available. Tier-only features are not qualification builds. `SPEAKRS_CUDA_PTX_TIER`
sets the tier limit for one run: `sm75`, `sm80`, `sm90` or `sm120`. Direct Python
invocation also accepts `--tier`. The device must support that tier. The candidate's
own loaded area must use the requested tier; a lower-tier candidate rejects the
run. Library-owned and glue areas use the production loader: the highest embedded
variant at or below both the limit and device capability. A lower tier in those
areas is valid and remains recorded. A Library control has no candidate area. Fixed
`qualify` and `controls` test instrumentation has its own sm75 PTX and is identified
separately. A tier label alone is not proof of matching loaded PTX.

The loader can register tier variants for ResNet, LSTM, SincNet and future areas.
`cargo xtask cuda-kernels check` must pass before qualification. It verifies the
committed PTX manifests and equal entry-point ABI across each area's tier variants.
A record for sm75 does not qualify sm80, sm90 or sm120.

`compiled_tier_fixture` is a test-only proof with the committed probe sm75 and sm80
variants. Run it in one fresh process per requested tier with the GPU lock held.
Both variants must produce the same output hash. Their recorded PTX bytes and tier
must match the requested tier; a deliberately mismatched request must reject.
The host-only coverage fixture also tests different batch/math triples per tier.
These fixtures prove the generic harness path. They do not qualify a production
candidate at a tier it does not ship.

### Traits

```rust
pub(crate) enum Batches { All, Only(&'static [usize]) }
pub(crate) enum Maths { All, Only(&'static [CudaMath]) }
pub(crate) struct CoverageEntry {
    pub layers: &'static [&'static str],
    pub batches: Batches,
    pub maths: Maths,
}
pub(crate) struct Coverage(pub &'static [CoverageEntry]); // the union of the entries

pub(crate) trait ConvCandidate: Sized {
    const COVERAGE: Coverage;
    fn coverage(tier: PtxTier) -> Coverage { Self::COVERAGE }
    fn plan(runtime: &CudaRuntime, layer: ConvLayerSpec<'_>) -> Result<Self, PlanError>;
    fn enqueue(
        &self,
        inputs: ConvInputs<'_, '_>,
        y: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;
}

pub(crate) trait SincCandidate: Sized {
    const COVERAGE: Coverage;
    fn coverage(tier: PtxTier) -> Coverage { Self::COVERAGE }
    const OUTPUT: SincOutput; // RawConv or Pooled
    fn plan(runtime: &CudaRuntime, spec: SincSpec<'_>) -> Result<Self, PlanError>;
    fn enqueue(
        &self,
        inputs: SincInputs<'_, '_>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &Phases,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;
}

pub(crate) trait LstmCandidate: Sized {
    const COVERAGE: Coverage;
    fn coverage(tier: PtxTier) -> Coverage { Self::COVERAGE }
    fn plan(runtime: &CudaRuntime, spec: LstmSpec<'_>) -> Result<Self, PlanError>;
    fn enqueue(
        &self,
        input: &CudaView<'_, f32>,
        output: &mut CudaViewMut<'_, f32>,
        phases: &LstmPhases<'_>,
        stream: &CudaStream,
    ) -> Result<(), CudaError>;
    fn diagnostic_layers(&self, stream: &CudaStream) -> Vec<(usize, Vec<f32>)> { Vec::new() }
}

// locked handles
impl Phases {
    fn op<T>(&self, op: Op, enqueue: impl FnOnce() -> Result<T, CudaError>) -> Result<T, CudaError>;
}
impl LstmPhases<'_> {
    fn op<T>(&self, op: Op, enqueue: impl FnOnce() -> Result<T, CudaError>) -> Result<T, CudaError>;
    fn input_proj<T>(&self, layer: usize, direction: Direction,
        enqueue: impl FnOnce(&Projection<'_>) -> Result<T, CudaError>) -> Result<T, CudaError>;
    fn recurrence<T>(&self, layer: usize, direction: Direction,
        enqueue: impl FnOnce() -> Result<T, CudaError>) -> Result<T, CudaError>;
}
impl<T: DeviceRepr + ValidAsZeroBits> Scratch<T> {
    fn zeros(runtime: &CudaRuntime, len: usize) -> Result<Self, CudaError>; // in plan
    fn from_host(runtime: &CudaRuntime, values: &[T]) -> Result<Self, CudaError>;
    fn get(&self) -> RefMut<'_, CudaSlice<T>>; // read or write during enqueue
}
impl SideStream {
    fn new(runtime: &CudaRuntime) -> Result<Self, CudaError>; // in plan
    fn stream(&self) -> &Arc<CudaStream>;
    fn split(&self, parent: &CudaStream) -> Result<(), CudaError>; // side waits for parent
    fn merge(&self, parent: &CudaStream) -> Result<(), CudaError>; // parent waits for side
}
// Op is one of Prologue, Pack, Main, Reduce, Epilogue
```

- `ConvLayerSpec` gives the layer name, the `Conv2d` shape (with batch and math
  mode), whether the layer adds a residual, and the folded weight `[k, c, 3, 3]` and
  bias `[k]`. `ConvInputs` gives `x`, the optional residual, the weight and the bias.
  The candidate computes `y = relu(conv(x, w) + bias [+ residual])`.
- `SincSpec` gives batch, samples (160000), `sinc` (15975), `pooled` (5325), the math
  mode and the generated filters `[80, 1, 251]`. With `SincOutput::Pooled` the
  candidate writes `max(|conv|)` over each window of 3, `[n, 80, 5325]`, and the locked
  dispatch runs the unchanged shared `segmentation_pool_norm` consumer with pool 1 and
  no `abs`. With `RawConv` it writes `[n, 80, 15975]` and the consumer pools by 3 with
  `abs`, as on the Library path. The consumer is Library-owned work.
- `LstmSpec` gives batch, frames (589), the math mode and the four layers' host
  weights in ONNX layout (`W [2, 512, in]`, `R [2, 512, 128]`, `B [2, 1024]`, gates
  `[i, o, f, c]`). Input is `[n, 589, 60]`; output `[n, 589, 256]`, forward direction
  first. Every input projection runs inside `phases.input_proj(layer, direction, |p|
  ...)` and every recurrence inside `phases.recurrence(layer, direction, || ...)`:
  the profile requires all eight of each, one per layer and direction, each with a
  launch, none overlapping on the host.  Oxide input projections and recurrences must both use custom kernels. No cuBLAS
  or cuDNN call is allowed anywhere inside an Oxide candidate region. The locked
  projection handle remains for Library controls and planted faults, not as an
  exception for Oxide.
- `phases.op(Op::Pack, || ...)` and the other `Op` names open optional locked
  sub-scopes for any target. Scope names come from these fixed methods only; a
  candidate cannot push its own range text.
- Plan-owned device buffers that `enqueue(&self)` writes are `Scratch<T>`: only
  device memory changes after `plan`.
- `enqueue` enqueues device work on `stream` and on its `SideStream`s. Create side
  streams with `SideStream::new(runtime)` in `plan`; in `enqueue`, `split` from the
  given stream, launch on `side.stream()`, and `merge` back before returning. All of
  it must be graph-capturable: the production graph captures the branches. Cooperative
  launches (`launch_builder(..).launch_cooperative(..)`) are allowed and pass the
  allow-list and sanitizer like any other launch. Size cooperative grids with the
  locked `runtime.cooperative_capacity(function, block_threads, dynamic_smem)`
  (active blocks per SM times `runtime.multiprocessor_count()`, 0 without
  `runtime.supports_cooperative_launch()`), in `plan`. Cooperative grids that can run
  at the same time, such as one per direction on a side stream, must fit the capacity
  together: their block counts must sum to at most the capacity. `enqueue` must write every element
  of its output.

### Coverage

`coverage(tier)` declares the (layer, batch, math mode) triples at the requested
tier. Its default is `COVERAGE`; a candidate can override it to return a different
coverage for each tier. The declaration is the
union of product entries. For example, 32-channel layers at every batch and mode, plus
64-channel layers at b7, b32, b33 and b64 in both modes, plus 64-channel layers at b1
in FP32 only:

```rust
const COVERAGE: Coverage = Coverage(&[
    CoverageEntry { layers: C32, batches: Batches::All, maths: Maths::All },
    CoverageEntry { layers: C64, batches: Batches::Only(&[7, 32, 33, 64]), maths: Maths::All },
    CoverageEntry { layers: C64, batches: Batches::Only(&[1]), maths: Maths::Only(&[CudaMath::Fp32]) },
]);
```

Dispatch runs the candidate for exactly the union's triples and the Library path for
every other one. Production selection uses the locked `implementation::PRODUCTION`
table. The table selects the accepted ResNet, LSTM, and SincNet coverage for
exactly their declared triples. Every other triple runs the Library path. The
production-table test pins those triples independently of the declarations.
Qualification starts all other boundaries on the Library path, so production
defaults cannot change a Library control or another candidate's input.
Layers must be boundary names of the target (`resnet.layer1.0.conv1` ...
`resnet.layer2.3.conv2`, `sincnet.conv0.abs_pool`, `lstm.stack`). Batches must be
harness batches: 1, 7, 32, 33 or 64. `All` means all positive batches for candidate
selection; qualification samples all five harness batches. Only model batches 1 and
32 grant production coverage. Batches 7, 33 and 64 are required stress cases when
declared, but cannot enter the production table. Production dispatch runs the
Library path at all other model batches. A declaration outside these, or an
`Oxide` that declares nothing, is refused. The result records the declaration.

The harness qualifies exactly the declared triples: layer parity, speed, profile and
sanitizer run on them. Undeclared triples must produce output bit-identical to the
frozen Library control (`library_dispatch:` checks), and their trace scopes may not
contain any entry from a candidate area. Stage gates always run on the mix in place.

### What the static scan refuses

`scan.py` runs before anything is built. It strips comments and scans string literals.
Every compiled Rust file outside the candidate tree is locked, so the candidate tree is
the only unlocked code in the qualification binary; an unlocked Rust file anywhere
else under `src/` is refused.

Candidate host code may reach only its own tree and an explicit set of names, after
resolving `use` trees, renames, `self::`/`super::` and full paths:

- `crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, KernelModule, PtxTier,
  LoadedKernels}`, `dnn::Conv2d`, and `error::{check_len, element_count, to_c_int}`;
  `CudaRuntime` methods are reached through the `runtime` the interface passes in, not
  by path (`load_kernels`, `cooperative_capacity`, `multiprocessor_count`,
  `supports_cooperative_launch`, `stream`, `ptx_tier`)
- the interface in `candidate.rs` (`super::` from the candidate files)
- `cudarc::driver::{CudaStream, CudaView, CudaViewMut, CudaSlice, CudaFunction,
  LaunchConfig, LaunchArgs, PushKernelArg, DeviceRepr, ValidAsZeroBits, DevicePtr,
  DevicePtrMut, DeviceSlice}` and the `sys::CUfunction_attribute` and
  `sys::CUfunc_cache` enums for `set_attribute`
- `std`, `core` and `alloc` modules without I/O, clocks, threads or shared state
  (`ops`, `mem`, `cmp`, `iter`, `fmt`, `slice`, `array`, `num`, `vec`, `sync::Arc`
  and similar)

Anything else, including other `crate::` functions, `test_support`,
`SafetensorsFile`, glob imports from outside the tree and other dependencies, is
refused. The candidate host files also refuse:

- environment reads (`env::`, `std::env`, `option_env!`; `env!` only with a `CARGO_`
  name), and the `fs`, `net`, `process`, `thread`, `os`, `io` and `time` modules,
  `File`, `Command`, `TcpStream`, `Instant` and similar
- `SPEAKRS_QUALIFY`, `/workspace`, `/proc/`, `/dev/`, `.safetensors` and `ref/`
- every `static` item, and `OnceLock`, `OnceCell`, `LazyLock`, `lazy_static`,
  `thread_local`, atomics, `Cell`, `RefCell`, `UnsafeCell`, `Mutex`, `RwLock`, so plan
  state cannot change after `plan` (`enqueue` takes `&self`)
- `test_support`, NVTX, `libloading` and `qualify_` names
- cuDNN or cuBLAS: `cudnn`, `cublas`, `sgemm`, `.blas()`, `.dnn()`, `ConvPlanner`,
  `ConvPlan`, `forward_bias_relu`
- foreign code and module loading: `extern`, `#[link]`, `#[no_mangle]`,
  `link_section`, `include!`/`include_str!`/`include_bytes!`, `asm!`, `load_module`,
  `Ptx`, NVRTC, `sys::`, raw `cu*()` driver calls, `dlopen`, `transmute`
- capture, stream, event and timing introspection outside `SideStream`: `capture`,
  `new_stream`, `new_event`, `.fork(`, `.join(`, `.wait(`, `.record(`,
  `record_event`, `elapsed_ms`, `context()`
- `cfg(test)` and `cfg(debug_assertions)`, so tests and production run one code path

In candidate device files it refuses the same paths and harness names, library names,
foreign code, and every `static` other than `static mut NAME: SharedArray<..>` shared
memory. Across all of `src/` it refuses code that runs without being called
(`link_section`, `#[used]`, `no_mangle`, `export_name`, `global_allocator`). It also
refuses a root `build.rs` and any Cargo configuration outside the locked
`.cargo/config.toml`. The build drops `RUSTC`, `RUSTDOC`, `RUSTUP_TOOLCHAIN`,
`RUSTFLAGS`, `RUSTC_*` and `CARGO_TARGET_*` overrides, records `rustc -vV`, and builds
the non-test library from the same tree, so test-only code cannot pass.

## Coverage and gates

Both FP32 and TF32 use b1 first, last partial and short-audio references, and b7, b32,
b33 and b64 rows from the mixed b32 reference, plus short b7. Rows above 32 use a
shifted reference-row order. Every case also runs a second input set on the same
allocations.

- **Layer parity** (declared triples): the Library error is the floor. Per-case L2
  ratio <= 1.10 and max-abs ratio <= 2.00, geometric means <= 1.00, and a zero Library
  error needs a zero candidate error. A failure names every failing case.
- **Determinism**: two runs of every row must have identical FP32 bits, and launched
  candidate entries may not contain floating-point atomics.
- **Secret input** (`secret:`): each numeric process builds fresh models from
  seeded, in-memory perturbed weights. It mixes two different fixture waveform
  windows with a random mixture weight in [0.1, 0.9], applies a circular time shift
  and gain in [0.5, 1.5), and adds zero-mean noise at 30–40 dB SNR. Samples and
  perturbed weights are never written to disk. The process records its seed only
  after the candidate runs; the static scan blocks candidate access to seeds and
  results. The locked FP32 Library front end produces the operator inputs: waveform
  normalization for Sinc, SincNet features for LSTM, and fbank plus preceding
  convolutions for ResNet (including the real residual).
  The truth is an independent CPU f64 definition on these exact boundary inputs
  and perturbed weights. Convolution and Sinc use seeded samples of at least 4096
  output elements per case, covering all batch rows, channels and spatial edges.
  Convolution includes bias, residual and ReLU; Sinc includes absolute value and
  max-pooling. LSTM uses the full four-layer recurrence in both directions over all
  589 steps for two seeded batch rows (the single available row at b1).
  The result records the sample indices and errors, not the input or truth values.
  Candidate relative L2 must be at most 1.10 times the Library-in-mode relative L2;
  max-abs must be at most 2.00 times the Library-in-mode max-abs on the same f64 sample.
  Library-in-mode is FP32 Library in FP32 mode and TF32 Library in TF32 mode.
  An exactly zero Library error fails closed. No fixture or ULP floor is used.
  Eager and captured replay outputs must both pass and must match bit for bit.
  Library controls run this check too. Baked fixture answers and plan-time tables
  fail because neither the input nor the perturbed weights existed before the run.
  The LSTM Library plan for the layer and secret gates uses the production default,
  `PersistStaticSmallH`, recorded in the result.
- **FP32 stage**: embedding minimum cosine may drop by at most 1e-9; segmentation
  logits error ratios <= 1.10 and argmax flips may not increase.
- **TF32 stage truth**: compare the candidate stage with an FP32 Library stage on
  the same input. Record the FP32 truth hash and require it to match between the
  candidate, Library and perturbation records. Compare relative L2, max-abs and
  minimum cosine error. Each candidate error must be no larger than the Library
  TF32 error bound: the maximum of its unperturbed error and the errors from the
  same 8 independent seeded 1-ulp perturbation draws. Perturb exactly the declared
  layer outputs. Apply this bound to every implementation, including controls.
  For cosine, compare `1 - minimum cosine`. Keep each candidate error, unperturbed
  Library error, all draw errors and their maximum in the result, including failed
  checks. The old upper-95th-percentile value is diagnostic only. This avoids
  rejecting an exactly Library-accurate output when all perturbations improve its
  error. A zero bound requires zero error; there is no absolute error floor.
  Existing f64 operator truth gates remain unchanged.
  The old drift band against Library TF32 is a reported diagnostic, not the stage
  accuracy gate. Segmentation's per-case and total argmax-flip bounds remain gates.
  TF32 coverage also needs a DER A/B at integration, which the root runs.

### Timing

Each math mode runs in four fresh processes in Library, candidate, Library, candidate
order, each covering every case of that mode.
Operators and stages are timed as CUDA graph replays the locked driver captures and
launches itself. Each of 20 samples is one CUDA event pair around a burst of replays
sized so the sample takes at least 20 ms, after 5 single warm-up replays. Operator
bursts alternate the two input sets launch by launch; stage samples alternate them
sample by sample, uploaded outside the timed interval.

### Operator timing and the locked noise rule

The raw operator timing check is blocked when the two Library process medians differ
by more than 1.0%. Otherwise it retains the strict 0% rule: the candidate must win
both A/B pairs, and each candidate median must be no larger than the faster Library
median. Undeclared operators have no speed gate; their outputs must match Library.
Separate-process stage timing remains a diagnostic for new records.

`verdict.py` evaluates a noise-blocked timing check from its recorded data. It accepts
that check only when:

```text
min(pair_speedups) >= 1 + 3 * max(Library_spread_fraction, locked_bound)
```

It recomputes both speedups and spread from the four medians and verifies the stored
values and locked bound. Every other check must pass. A hard failure rejects, even
when the noise rule accepts other checks. The raw `passed: false, blocked: true`
check stays in the result. `verdict_evaluation.noise_rule` records the medians,
speedups, spread, bound, threshold and decision so a reviewer can repeat the calculation.
Below-threshold timing remains blocked. No status or device mapping bypasses a gate.
The rule applies to new operator timing and to stored pre-schema-4 stage timing
(with its locked 0.3% bound). It never resolves a new paired-stage block.

### Paired stage guard

Library and candidate replay in one process, in ABBA order at single-replay
resolution. Each block uses the same input on both sides. There are 5 warm-up blocks
and at least 256 measured ABBA blocks (512 pairs). Eligible operators run beside the
stage with the same schedule. Uploads stay outside CUDA event timing.

ResNet keeps one isolated Library/candidate operator pair live at a time to bound
GPU memory at stress batches. Each eligible layer gets the same number of measured
blocks, rounded up to a multiple of 16, and 5 warm-up blocks. The stage runs beside
that layer in every ABBA block. The saving is the sum of per-layer mean savings, not
the mean across layers. The record keeps the raw operator times and the layer label
for every block; the evaluator requires complete, equal, contiguous layer strata.
This uses the same process and stage replay set, not separate operator timings.

The point estimate is the sum of Library stage times divided by the sum of candidate
stage times. The bootstrap uses 4096 seeded draws. ResNet resamples whole, equal,
contiguous operator strata (32 ABBA blocks per stratum for the full 14-layer set),
never 16-block halves of a stratum. Unstratified stages resample contiguous groups
of 32 ABBA blocks. At least two whole groups are required. This retains correlation
within the measured stratum and adjacent replays instead of treating them as
independent. The result records the actual block length, group count and design.

Both acceptance paths use the one-sided 95% lower confidence bound (the bootstrap
5th percentile). The 2.5th/97.5th percentiles remain a two-sided diagnostic only.
The primary path passes if the one-sided lower bound is at least 1.0. Otherwise
the time-unit non-inferiority margin path passes only if all three conditions hold:

- the point estimate is at least 1.0;
- the one-sided 95% lower speedup bound is at least `1 / (1 + delta)`, where
  `delta = sum(operator saving_ms) / Library stage_ms`, measured in this process
  and replay set;
- every eligible operator passed the locked operator timing rule, including its
  noise-rule evaluation when applicable.

A point estimate below 1.0 rejects. A failed non-inferiority margin rejects. A margin
without a passed operator gate blocks. No CI half-width comparison grants acceptance.
Saving measured in a separate process cannot resolve this gate. The result keeps all stage and operator
ABBA observations, bootstrap settings, CI and measured saving. Each paired output
hash must match the numeric phase for the same input. Library controls validate this
measurement contract without requiring Library to be faster than itself.

After its bursts, every timing process replays each case once on each input set and
records both output hashes. Both must equal the numeric phase's outputs for the same
inputs, so a no-op, phase-dependent or cached result fails (`timing_output:`).

### Profile

One eager nsys trace covers every case. The locked driver and dispatch push NVTX
ranges named `qualify.<nonce>.<kind>.<name>`; the nonce is random per run and passed
only to locked code. A range with another nonce is refused.

- Between each input upload and its output download (a `window`), every kernel, copy
  and memset on any stream must be launched inside a `candidate`, `library`, `call`
  (locked cuDNN or cuBLAS call site), `projection` or `fixed` (locked Library-owned
  kernel) scope. Attribution uses the host launch call, so work on a side stream
  belongs to the candidate scope active when it was launched. Candidate work must be
  on the qualification stream or on a side stream that carries only candidate work;
  everything else must be on the qualification stream, apart from vendor-internal
  streams inside a locked call scope.
- `phase` scopes must sit inside a candidate scope; the projection helper must sit
  inside the `input_proj` phase of the same layer and direction; LSTM stacks need all
  16 phases as described above.
- Kernels a candidate launches from `plan` (the `plan` scope) must be on the
  allow-list too.
- In a candidate scope, a kernel's mangled and demangled names must both be entries
  of the PTX bytes the process loaded. Oxide candidates must make zero Library
  calls, including calls inside the locked projection helper.
  Host-device copies are refused.
- A Library path may not launch a candidate-area entry.
- Every declared boundary must have a candidate scope with kernels.

The numeric and timing processes also track library calls made while a candidate
scope is open, and, during graph capture, enumerate the nodes each candidate scope
adds to the captured graph, side-stream branches included, and resolve kernel names with `cuFuncGetName` /
`cuKernelGetName`. Library kernels in any candidate scope, unlisted kernels,
host copies and host or event nodes fail (`profile:captured_library_calls`,
`profile:graph_nodes`). The driver labels every capture with its case, and
`profile:graph_nodes` also fails unless each case's captured candidate-kernel multiset
equals that case's eager-profile candidate-kernel multiset. Graph mode is set and verified
by the locked driver. Only Library controls use projection-node permissions. These
permissions reset when the driver capture ID changes, so node addresses reused after
graph destruction cannot hide kernels.
The ignored `sequential_capture_keeps_candidate_kernels` regression uses a real
Oxide ResNet convolution plan and runs fresh captures at b1, b1, b7, b32, b33, b64 and b7 in one process with scope tracking on.
For each capture, the same plan first records an eager launch trace on the same
input. The harness derives the expected kernel multiset from that trace and requires
capture == eager, including duplicate launches. There is no fixed kernel count and
no candidate-declared count. The ignored `secret_library_algorithms` proof measures Standard and
PersistStaticSmallH separately against f64 on the same secret input. Both must have
FP32 rounding-scale error (below 1e-4 relative L2), and a one-element, one-ULP nudge
must change the SmallH error by at most 10%. The SmallH/Standard error ratio is
information only; the proof does not require one algorithm to pass the other
algorithm's gate. The capture and soundness tests need the qualification environment
and GPU lock. The ignored `f64_reference_matches_fixture_rounding` and
`conv_f64_matches_fixture_rounding` tests use fixture intermediates and run on CPU.

### Loaded artifacts and embedded PTX

`cargo xtask cuda-kernels build-cubins` builds each committed PTX tier for each
exact architecture in 75, 80, 86, 89, 90 and 120 at or above that tier. The manifest
pins the cubin hash, its source PTX hash, and the ptxas version and flags.
`cargo xtask cuda-kernels check` checks these pins and the feature embed masks.
`check --rebuild` checks committed PTX pins and byte-identical cubin rebuilds
with the pinned CUDA 13.0.88 ptxas. It does not regenerate PTX. A CPU test checks each host plan's possible kernel names
against every PTX tier that it can use. The PTX lint rejects generic shared
addresses from `cvta.shared.u64` that reach `cvt.u32.u64`, including through
64-bit add, subtract and move instructions.

The loader first selects the PTX tier for each area. The production table then
owns the artifact request for that area, tier and exact device capability. A
`PtxJit` pin loads the pinned embedded PTX; a `Cubin { arch, sha256 }` pin loads
only that exact architecture and hash. A missing or driver-rejected requested
artifact is a typed refusal, not a request to try the other format. Production
uses the existing Library fallback where allowed; driver-only mode returns the
typed refusal. Uncovered production tuples load no candidate module.

An explicit qualification triple requests the exact embedded artifact it declares
before loading: the device's exact cubin when embedded, otherwise PTX JIT. A
rejected cubin does not silently become a JIT measurement. The typed successful
artifact is `Cubin { arch, sha256 }` or `PtxJit { sha256 }`.
`SPEAKRS_CUDA_FORCE_PTX_JIT=1` is a diagnostic override only. Actual loaded bytes
still form the selection key, so this override cannot authorize a cubin pin.
The legacy StageTail fixture retains its explicit JIT policy. Current PR #36
production pins request JIT without any environment override, including cc 12.0.

One table owner binds each area, tier and exact capability to one artifact.
A compile-time assertion and both `--check-table` modes reject duplicate owners;
coverage for one owner must be combined in that entry. Cached modules cannot be
replaced by a request for a different artifact.
The selection key includes the tier, exact device capability and loaded artifact.
A cubin-qualified table entry cannot match JIT, another cubin architecture, or
other bytes. A mismatch uses Library where allowed, or the typed driver-only
error. The legacy PR #36 entries remain explicit sm75, cc 12.0, PTX-JIT entries.
They do not authorize the new cubins.

Selection intent and coverage come before artifact loading. Library controls,
Library-backed faults, and uncovered tuples do not load a candidate module for
selection or diagnostics. Library diagnostics use only the embedded tier and
device context. A covered Oxide request loads its artifact before it can receive
a qualification token. The exact candidate-area loading check remains required.

The allow-list uses the actual PTX text embedded in the running binary, not a
source-tree hash assigned after the run. Each module records
`embedded_ptx_sha256`. An accepting verdict requires it to match the pinned PTX
file, the module hash, and the cubin's source PTX hash where applicable. A stale
binary cannot accept. Parsed entries must also match the PTX. PTX in a
subdirectory of `ptx/` is refused. Cubin records include exact architecture, actual
binary hash, source PTX hash, and embedded ptxas version and flags.

Schema-5 records include the device name, exact capability, positive SM count and
L2 size, driver release and API version, cuDNN and cuBLAS versions, and
`cuda_version` from `cublasGetCudartVersion`. Device name, SM count and L2 are
physical-device evidence, not extra selection-key fields. All processes must
report consistent device and artifact data. Timing and paired replay use a joined
SM clock sampler. The sampler and its child processes stop before unlock.

### CPU work and GPU ownership

GPU preparation produces the exact FP32 front-end inputs and copies the operator
weights under the lock. Host-only snapshots then compute f64 truth without the
lock. Each TF32 draw uses the same seed, layer-name hash, per-index integer hash
and one-ULP rule as before. The CPU prepares the draw choices without the lock;
the GPU applies them after the lock is taken again. Gates, thresholds, samples
and seeds are unchanged. A context-wide synchronization precedes each CPU section.
A guard takes the lock again before normal return or panic unwind, so CUDA object
drops remain locked. Records state lock ownership and each unlocked CPU section.

### Override types

The plan types can request Library or one configuration from an enumerated set
per area. Each custom configuration has its own pinned accuracy and deterministic
execution evidence for an exact boundary, batch, math mode and target artifact.
A private token prevents reuse for another configuration or tuple. These overrides
are marked **unqualified for speed** in logs and diagnostics. This phase provides
types and tests only; it adds no CLI or file format.

### Sanitizer

Every Library control and candidate run proves the exact include filter with three
planted faults:
`qualify_oob` (memcheck, invalid global write), `qualify_race` (racecheck, shared
memory hazard) and `qualify_round` reading uninitialized memory (initcheck). Each must
exit 86 with its fault reported.

A candidate then runs memcheck, racecheck and initcheck with `--kernel-name
kne=<entry>` for every loaded entry, on both input sets of its declared batches (b7 and b33 when it
declares every batch), both math modes. Each tool must complete with zero findings
within 20 minutes after lock acquisition. A ResNet timeout is split into one process
per declared convolution and batch; LSTM and Sinc timeouts block.

Oxide LSTM candidates must use custom input projections and make zero Library projection
calls. The previous K2 record remains pinned in production, but K2 cannot pass this new profile rule. The locked projection baseline runs all three tools, unfiltered, on every
shape the helper can issue at a harness batch (m = 589 x 1, 7, 32, 33, 64; n = 128 to
512; k = 60 and 256; both modes) and is retained under `tests/cuda_qualify/baselines/`.
These shapes constrain the Library baseline only. They do not permit any Library
call in an Oxide candidate region.

Shared-memory initialization is outside all three tools: initcheck covers global
memory only and racecheck covers hazards. Two checks cover it instead:

- `ptx:shared_initialization` runs a must-dataflow pass over every entry of the
  loaded PTX. Each `ld.shared` or `ldmatrix` must be preceded, on every path from the
  entry, by a shared store (`st.shared`, `cp.async` to shared, `atom.shared`,
  `red.shared`, `stmatrix`) or by a CTA barrier that some shared store in the entry can
  reach (one thread reading what another wrote). It is address-insensitive and does
  not follow generic pointers converted from shared addresses.
- Before candidate launches in the numeric phase, the locked dispatch launches
  `qualify_poison`, which fills the largest dynamic shared allocation on every SM with
  a NaN pattern. This is best effort: the hardware does not promise a later kernel sees
  that residue.

**Shared-memory initialization stays an explicit item of every kernel audit.**

Library controls run numeric, timing, profile and the positive controls. Mutants do
not run the sanitizer tools.

## Mutation proof

### Phase 2c infrastructure proof and final seal

Phase 2c-1 requires three full Library controls (one per area), one live mutant
per each of the 13 standard mutant kinds on the area with the most changed
collection or injection path, and matched ResNet and SincNet StageTail pairs.
Record each actual device and compiled lock. The exact unchanged gate-decision
code and every changed collection/injection path are bound in
`tests/cuda_qualify/PHASE2C1_CARRY_FORWARD.md` and its evidence files. This reduced
infrastructure proof grants no new production acceptance. If any mutant fails to
reach its intended gate, stop; do not change its gate, bounds, samples or seeds.

Run all 39 standard mutants at the final phase 2c seal, after phase 2c-2. That
full area matrix is required even when this infrastructure subset passes. Keep
each completed run and its verification receipt, copy it off the GPU box at once,
and skip only verified matching completed runs after a restart. Matched controls
and faults must run on the same device in the same quiet window. Legacy StageTail
production fixtures remain cc 12.0 PTX-JIT only.

The phase 2c-1 results are pinned in
`tests/cuda_qualify/evidence/phase2c1-results.json`, linked by `PROOF.json` and
`AUDIT_PROOF.json`. They preserve the measured locks and full raw evidence; the
final owner lock does not relabel a measured binary. Fixture coverage discovery
reads embedded PTX identity without loading an unused module. The plan owner
still requires actual loaded-artifact identity before it makes a production token.
The SincNet matched control passes every check. The ResNet matched proof has a
shared control/fault stable-module failure in its earlier records; its matched
criterion passed, but those records grant no production acceptance. The exact
carry-forward document states this limit and the unchanged paths.


A mutant is caught only by its exact tier-stripped check name and reason:

| Mutant | Planted defect | Caught by |
| --- | --- | --- |
| `Precision` | rounds real inputs to BF16 | `layer:` with reason `layer parity` |
| `Shape` | correct only at b32 | `layer:` `layer parity`, failing exactly the non-b32 rows |
| `Tail` | zeroes the partial tile | `layer:` `layer parity`, failing every partial-tile row |
| `Fallback` | library work in the candidate scope | `profile` with `forbidden library kernels` |
| `Atomic` | floating-point atomic accumulation | `determinism:fixed_reduction_order` with `floating-point atomic in launched custom entry` |
| `Slow` | the library operation three times | `speed:` with `candidate slower than the faster Library process` |
| `StageSlow` | replays the candidate stage twice, without adding isolated operator work | `paired_stage:` with `stage regression exceeds zero slowdown` |
| `StageTail` | adds an intermittent captured GPU spin after a real pinned FP32 candidate stage, with unchanged operators | `paired_stage:` with `non-inferiority margin not established` |
| `StageAccuracy` | rounds the real TF32 stage output to BF16, without changing isolated operators | `stage_truth:` with `candidate less accurate than Library TF32` |
| `PhaseCheat` | skips its work in the timing phase | `timing_output:` with `final output differs` |
| `Unscoped` | launches after its scope closes | `profile` with `outside a candidate or library range` |
| `Unlisted` | launches a kernel from unrecorded PTX | `profile` with `not on the loaded PTX allow-list` |
| `Lookup` | answers from the first weights or input it saw, like a baked table | `secret:` with `differs from Library on a fresh input` |
| `UninitShared` | launches an entry that loads shared memory no thread stored | `ptx:shared_initialization` with `shared load` |

An escaped mutant is written as `escaped` and exits 4. A mutant runs only the phases
its gate reads (always numeric, which includes the secret-input and PTX checks;
profile for `Fallback`, `Atomic`, `Unscoped` and `Unlisted`; timing for `Slow` and
`PhaseCheat`; paired replay for `StageSlow` and `StageTail`), and no sanitizer tools. Library
controls run every phase plus the three positive sanitizer controls; candidates run
every phase, the positive controls and the candidate sanitizer. The static scan has its own
proof: `tests/cuda_qualify/scan_fixtures/phase_cheat.rs` reads `SPEAKRS_QUALIFY_PHASE`,
and `unscanned_call.rs` hands its work to code outside the candidate tree; both must be
refused, in the unit tests and live, copied over the LSTM candidate in a scratch tree.
A new module elsewhere under `src/` fails the lock before Python starts. `PhaseCheat`
is the phase cheat with the scan bypassed.

`StageTail` differs from the Library-backed fault seam. It uses only the FP32
tuples selected by the pinned production table on the actual tier and device.
Other tuples, including stress batches and TF32, retain Library identity. The
operators run the real accepted candidate, in both separate-process timing and
paired replay. A failed margin requires the existing operator gate to have passed;
neither planted timings nor a forced operator pass can authorize that branch.

The fault applies only to `fp32/first/b1` on ResNet and `fp32/mixed/b32` on SincNet.
The first 16 collected ABBA blocks use a second captured stage graph; every other
block uses the normal candidate graph. The delayed graph runs the same real stage
on the same buffers, then one thread spins on the [PTX device timer](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer-globaltimer-lo-globaltimer-hi).
CUDA events measure this GPU work. There is no host sleep or alteration of timing
data. The extra duration is 1.7 times the median of five in-process Library stage
replays for ResNet, and 1.4 times for SincNet. These fixed fault sizes keep the point
at least 1 while testing the lower tail; they do not change a gate threshold.
The record holds the calibration replays, requested duration, measured spin time,
group size/count, and delayed output hashes on both inputs. Both normal and delayed
hashes must match numeric evidence. ResNet still bootstraps whole operator strata;
SincNet still uses whole 32-block bootstrap groups, not the shorter fault group.

Hardware proof covers ResNet and SincNet. LSTM StageTail is refused until phase
2c-2 integrates the Library-free LSTM candidate: the current pinned implementation
uses cuBLAS projections inside Oxide. This deferral does not exempt those calls
from the zero-library rule. Shared gate logic also has the exact asymmetric-tail
regression case and a clearly labelled, altered recorded-data fixture.

## Production table and record cache

Run `python3 scripts/cuda/qualify/qualify.py --check-table`. This host-only check runs
the locked Rust table-export test with the `cuda` feature. It uses evaluated table
values, not a parser for Rust source expressions. `--table <json>` checks a scratch
copy for negative proof. It does not run a GPU or change production selection.

Pinned JSON or gzip JSON records live outside the tree, under
`$SPEAKRS_QUALIFY_CACHE/records/<sha256>`. The default Mac cache is
`~/Library/Caches/speakrs-cuda-qualify`. Import exact raw bytes with:

```sh
python3 scripts/cuda/qualify/assets.py --record-sha256 <sha256> --import-file <file>
python3 scripts/cuda/qualify/qualify.py --check-table
```

The default check uses the small owner-locked `ACCEPTANCE.json` summary and needs
no raw records. Each entry must be a subset of its summarized accepted tuples,
with matching tier, device capability, area and DER hash. Current area source,
manifest, PTX and full tested coverage hashes must equal the summary. Missing
summaries, extra production tuples, changed files or declarations fail closed.
The pinned tier must also be one that production can load on that device. For
some tier feature supported by the device, it must be the highest variant the
area ships at or below that feature. This permits a minimal tier build even when
an all-tier build selects a higher variant. A forced qualification tier that no
production build can select on the pinned device cannot authorize an entry.
The summary contains the original record hash, harness lock digest, accepted tuples
and complete noise-rule inputs, threshold and result. CI runs this offline mode
and the Python suite. It does not download or publish raw evidence.

For a full local evidence check, use the outside-tree cache root:

```sh
python3 scripts/cuda/qualify/qualify.py --check-table --records "$SPEAKRS_QUALIFY_CACHE"
```

This reads `<cache>/records/<sha256>`, verifies exact raw file hashes and re-runs
the locked evaluator. It re-derives the complete canonical summary; its bytes must
match the committed summary exactly. The full mode also checks all offline drift
rules. The DER evidence hash must resolve to a nonempty JSON object in this mode.
No raw record or generated verdict can silently change the shipped summary.

The table and its derived summary replace `QUALIFIED.json` and `qualified.py`.
There is one locked noise evaluator. It agrees with the old evaluator on all three
pinned records, including 20 ResNet and three LSTM noise-blocked cases. Full record
checks validate phase completion, exact aggregate checks, and required numeric,
secret, speed, stage and paired evidence for each declared case. Stress batches
must pass but never grant production coverage.

New records retain source-file hashes. The three older records retain a driver
binary hash, not source hashes. Their fixed source, manifest and PTX bindings were
copied from PR #36's acceptance manifest into the explicit locked legacy mapping.
Both the original shared candidate-interface hash and its amended hash are reported.
The original amendment covers `PlanError`, per-tier coverage and production/stress
batches. Separate fixed old/new pins cover host kernel enumeration and cubin
manifest metadata. Each pin states the reason. Neither amendment changes candidate
arithmetic or claims new GPU evidence. It is not GPU evidence for a changed operator. Added,
removed or changed bound files reject. The summary retains this source evidence gap.

Owners generate summaries with `records.write_summaries` from the evaluated Rust
table and exact raw cache. This is an explicit owner action, followed by full
comparison, drift tests and a new lock. Qualification proof uses the full record
mode before the GPU matrix. Summary generation does not run a GPU or approve a
new candidate. No writable acceptance manifest or second noise-rule copy remains.

### Explicit legacy hardware mapping

Only these three PR #36 records use the legacy hardware mapping in `records.LEGACY`:

| Area | Record SHA-256 | Hardware source report |
| --- | --- | --- |
| ResNet (K1) | `8f8fa3e3c158771e354aad83f4e42fca6fac998aa192b39a966067a4b0035758` | `qualification/k1/REPORT-K1-requalify.md` |
| LSTM (K2) | `3badc1aec939b0e8f7312786d695bec6445de1dacb1f85e44124bf3ac20356f8` | `qualification/k2/REPORT-final-lock.md` |
| SincNet (K3) | `a4d1a2692b78a814cd2da16f084801c3641bfd11c6d095d610f3a8182ad6f675` | `qualification/k3/REPORT-K3-requalify.md` |

The reports are in the `speakrs-native-cuda-records-2026-10-04` research archive.
They state sm75 PTX on an RTX 5070 Ti, capability 12.0. This is not Turing hardware
qualification. The mapping supplies only the missing tier/device identity and names
older binary-only code provenance explicitly. The derived summary also records the
legacy device name from those reports. It does not bypass a blocked status.
K1 and K2 keep their raw blocked outcomes; the same general noise evaluator used for
new operator timing must accept their recorded speedups. The PR #36 noise decision
is documented in `decision-log.tsv`, at 1:54:43 PM CDT on October 3, 2026, and the K1
and K2 decisions at 4:47:33 AM and 5:08:33 AM CDT on October 4, 2026.

## Lock

`SCOPE.json` defines the locked inventory, and `lock.py` and the xtask entry point
both read it: `scripts/cuda/qualify/`, `tests/cuda_qualify/`, everything under `src/`
except `src/inference/cuda/candidate/` and the three candidate PTX areas
(`ptx/resnet.*`, `ptx/lstm.*`, `ptx/sincnet.*`), plus `Cargo.toml`, `Cargo.lock`,
`.cargo/config.toml` and the xtask entry points. The Library-owned PTX is locked.
Added, removed and changed files change the lock; symlinks are refused. `lock.py`
refuses to write a lock unless `cargo fmt --check` and `ruff format --check` pass, so
routine formatting never breaks it. Rust and Python both require the owner's
outside-tree digest before GPU work, and Python checks the lock again after
collection. Any change elsewhere under `src/`, for example an integration edit, needs
a new owner lock.

The Library source is frozen as the external asset `tests/cuda_qualify/control.tar.gz` (the harness's own
Python and scan fixtures are left out, since they never build it), and the control
binary is built from it, so candidate edits cannot change the control. The projection
baseline binds to the archive hash and the exact `rustc -vV` toolchain; the control
binary's own hash depends on its build path, so it is recorded but not bound. Before
recording a new owner digest:

```sh
python3 -m unittest discover -s tests/cuda_qualify -p 'test_*.py'
python3 scripts/cuda/qualify/freeze_control.py
python3 scripts/cuda/qualify/lock.py
```

To retain the projection baseline, use a development lock to run `lstm Library`, copy
the result and its evidence directory back, run `freeze_baselines.py <result-json>`,
then make the final lock. Do not change the control archive after collecting the
baseline. Changing a locked file needs a new owner review. Real Turing hardware is
untested: the sm75 tier is forced on the sm120 GPU.

### External qualification assets

Files over 1 MB stay outside the repository. `SCOPE.json` lists their logical names
and SHA-256 hashes, which are also retained in `LOCK`. The frozen control archive and
LSTM source reports use `<cache>/assets/<sha256>` with no filename extension.
Set `SPEAKRS_QUALIFY_CACHE` to an outside-tree cache directory. The default is
`~/Library/Caches/speakrs-cuda-qualify` on macOS and
`$XDG_CACHE_HOME/speakrs-cuda-qualify` (or `~/.cache/speakrs-cuda-qualify`) on Linux.

Import a retained owner file with:

```sh
python3 scripts/cuda/qualify/assets.py --import-file /path/to/owner-file \
  --name tests/cuda_qualify/control.tar.gz
```

Use the logical `tests/cuda_qualify/baselines/lstm-source-<sha256>.json` name for an
LSTM source report. Imports check the owner hash before writing. Missing or changed
assets stop qualification with the required hash and import command. No URL or
credential is needed. To regenerate, run `freeze_control.py` or
`freeze_baselines.py <Library-result-json>`; these write cache assets and update
`SCOPE.json`. A regenerated asset needs a new lock and owner review. Retain the
original control archive when reusing its existing projection baseline.

The runtime defaults are segmentation FP32, embedding TF32,
`CudaLstmAlgorithm::PersistStaticSmallH`, and enabled CUDA graphs. The frozen control
archive retains the exact Library source for the current lock. Renew it when the
locked Rust harness changes, before collecting a new baseline.

## Phase 2a ownership and production policy

Qualification builds use `cuda`, never `cuda-driver-only`. Library controls construct
Library plans; candidates construct Oxide plans from a test-only qualification token.
Production selection uses the boundary, batch, math, actual area PTX tier, and exact
device capability. Production model batches are 1 and 32; 7, 33 and 64 remain stress
cases. The production table pins accepted record and integrated DER hashes. Each result records its tier/device, and the host-only record check validates the
production entries.

## Candidate plan refusals

Candidate traits return `Result<Self, PlanError>`. A validated production token
may receive `DeviceUnsupported` when a device cannot host a candidate. With the
Library policy allowed, dispatch then builds only the Library plan for that boundary
and loads its library at that point. It does not build a dormant second plan. With
the driver-only policy, the refusal returns `CandidateDeviceUnsupported`, with the
area, boundary, batch, math, actual tier, device capability and reason. Explicit and
qualification selections return that error and never fall back. Real CUDA or model
errors always propagate. A successful candidate remains the only plan owner.

### Harness proof records

`tests/cuda_qualify/PROOF.json` pins the raw hashes for the Library controls and
mutation proof. The owner lock covers this small receipt. It does not grant
production coverage: Library controls and mutants cannot authorize a replacement.
The receipt preserves each record's original harness digest, tier and device.
A later owner lock can retain proof only with an explicit, reviewed comparison
of the changed files; it cannot silently relabel the original record.

Raw proof records remain outside git, named by their exact SHA-256. The phase 2b
Mac proof uses `~/.local/share/speakrs-cuda-qualify/records/<sha256>`. To read an
imported production record from that cache, pass its root with `--records`:

```sh
python3 scripts/cuda/qualify/qualify.py --check-table --records "$HOME/.local/share/speakrs-cuda-qualify"
```

The full tool logs and archive hash are retained with the proof report. Raw file
hashes identify the original JSON bytes, not a reserialized copy.

### Audit decisions for phase 2b

The TF32 truth ceilings are componentwise: cosine, max-abs and relative L2 may take
their maxima from different perturbation draws. This is an accepted residual of
the root-decided max(unperturbed Library, eight draws) rule. The Grok audit measured
a lift of at most 6.7% in cosine error and about 1.9% in SincNet max-abs. These lifts
are not a new tolerance or a change to the gate.

The owner lock includes `.github/workflows/ci.yml`. Offline CI validates the locked
acceptance summaries and current shipped bindings; it does not open raw records.
Full raw-record re-derivation (`qualify.py --check-table --records <cache>`) runs on
the GPU box at every re-lock. No record publication or download URL is required.

Production loadability follows the locked `AreaPtx` feature embed masks and each
tier feature. A PTX file on disk with no matching embed mask cannot make a table
entry loadable. The cubin/JIT artifact must also match the qualified entry.
The current production areas ship only sm75, so their current table proof is not
a claim that an unembedded future variant is production-loadable.

### StageTail matched-candidate control

`StageTailControl` is a test-only, non-accepting control. It selects exactly the
same pinned FP32 production plans and coverage as `StageTail`, runs the same
numeric, operator timing and paired stage phases, and loads the same recorded
spin module. It does not capture or enqueue a delay. Neither choice grants
production acceptance. In the same quiet device window, the proof requires the
mutant to fail its intended paired-stage margin check and that same check to pass
in the control. After the unchanged locked noise evaluator is applied to both
records, the mutant must have no new failure except that margin. It must also
have no new non-timing failure that the control passes. Raw blocked checks can
change in both directions between runs of the same binary; the proof records
these flips, both raw and evaluated failure sets, and every Library spread.
The comparison changes no qualification gate, noise bound or record verdict.

The locked `tests/cuda_qualify/AUDIT_PROOF.json` pins the two matched pairs and
the offline replay of all 42 prior control and mutant verdicts. All 39 prior
intended-gate catches are unchanged. ResNet and SincNet StageTail both reach the
margin path with a point estimate above 1, positive measured operator saving,
and a passed operator gate, then fail its one-sided lower-bound requirement.
The same paired check passes in each no-fault control. Both pairs have no new
non-timing failure. ResNet has raw operator noise flips; the receipt retains
the raw sets, all noise evaluations, and a hash for every timing spread row.
The receipt grants no production coverage.

The ResNet pair keeps its original compiled lock. Its later direct-selector
correction is used by the isolated SincNet test only: ResNet's unchanged
`plan_selection` returns the pinned token before that branch. The receipt pins
the exact source diff and both source snapshots. SincNet uses the corrected
snapshot. Final receipt and documentation changes do not relabel either
compiled digest or alter measured evidence. Large archives and raw records
remain outside git in the SHA-addressed qualification cache.
