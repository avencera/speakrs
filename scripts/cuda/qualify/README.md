# Locked CUDA qualification

The harness decides whether a custom kernel may replace a cuDNN or cuBLAS call at one of
three boundaries: the 14 eligible ResNet 3x3 convolutions, the Sinc producer, and the
complete four-layer bidirectional LSTM stack. An accepted qualification record
authorizes a replacement. The acceptance command below checks a normal pass or the
strict Library-noise rule.

## Running

Run from your own task tree on the GPU box. Any `/workspace/<task>/tree` whose locked
files match the owner's digest works; results go to `/workspace/<task>/results` and
builds to `/workspace/<task>/target`.

```sh
cd /workspace/<task>/tree
source /workspace/env.sh
export SPEAKRS_QUALIFY_OWNER_DIGEST=<digest-kept-outside-the-tree>
cargo xtask cuda-qualify <resnet|lstm|sincnet> <implementation>
```

Implementations are `Library` (the sanity control, never a replacement), `Oxide` (the
registered candidate) and the nine planted faults listed under
[Mutation proof](#mutation-proof). Do not wrap the command in `flock`: the harness takes
`/workspace/gpu-bench.lock` for every GPU child process and refuses unlocked ones, so
an outer `flock` on the same file would deadlock. Each GPU process holds the lock for
one phase and, for numeric and timing, one math mode, so kernel workers interleave.

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

All three candidate traits return `PlanError`. `DeviceUnsupported` means that the
device cannot host the plan. Shared dispatch uses the already planned Library path
for a production selection and logs the boundary, batch and reason once per plan.
An explicit harness selection fails for the same outcome. `PlanError::Cuda` always
propagates, including ordinary `CudaError::Unsupported` errors. Coverage still
selects the Library path for undeclared triples.


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
Adding a higher PTX tier for a candidate area is a one-line locked change to
`kernels.rs` and `AREAS` in `xtask/src/commands/cuda_kernels.rs`, so the owner
re-locks.

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
  launch, none overlapping on the host. `p.project(layer, direction, ProjectionGemm
  { n, weight_transposed, beta }, a, weight, c)` is the only library call a candidate
  may make, and it exists only inside `input_proj`: cuBLAS with `m = batch * 589`,
  `k = 60` for layer 0 and 256 above, `n` in 128, 256, 384, 512, `beta` 0 or 1, and
  the boundary's math mode. It refuses other shapes. Recurrence is library-free.
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

`COVERAGE` declares the (layer, batch, math mode) triples the candidate ships, as the
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
harness batches: 1, 7, 32, 33 or 64, or `All`, which means exactly those five;
production dispatch runs a candidate only at a qualified batch and the Library path
at any other. A declaration outside these, or an
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
  `ConvPlan`, `forward_bias_relu`; use the projection helper instead
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
- **TF32 stage noise band**: when the candidate declares TF32 layers at a case's batch,
  the frozen Library control reruns that stage 8 times, each with seeded random 1-ulp
  perturbations (`qualify_perturb`) on the outputs of exactly the declared layers. The
  band of each case is that case's own largest drop in minimum cosine (embedding), or
  its largest increase in logits L2 error, max-abs error and flips (segmentation) over
  the 8 seeds. Per case the candidate's change from the Library stage must be within
  that case's band. Across all TF32 stage cases the candidate's mean cosine must be at
  least the Library mean (segmentation: mean logits error at most the Library mean,
  total flips at most the Library total plus the sum of the cases' largest
  single-seed flip increases). The per-case bands and seeds are in the result.
  TF32 coverage also needs a DER A/B at integration, which the root runs.

### Timing

Each math mode runs in four fresh processes in Library, candidate, Library, candidate
order, each covering every case of that mode.
Operators and stages are timed as CUDA graph replays the locked driver captures and
launches itself. Each of 20 samples is one CUDA event pair around a burst of replays
sized so the sample takes at least 20 ms, after 5 single warm-up replays. Operator
bursts alternate the two input sets launch by launch; stage samples alternate them
sample by sample, uploaded outside the timed interval.

A case is **blocked**, not passed, when the two Library processes' medians differ by
more than 0.3% (stages) or 1.0% (operators). Otherwise:

- **Declared operator triples** get the strict 0% test: the candidate wins both A/B
  pairs, and each candidate median is at most the faster Library median.
- **Stage cases that run declared layers** get a regression guard, because a stage
  saving can sit at the stage noise: in both pairs the candidate stage median is at
  most its Library process's median times 1.003, and the case's mean speedup (Library
  over candidate, averaged over both pairs) is at least 1, so no case offsets another.
- **Undeclared operator and stage cases** have no speed gate. Their outputs must be
  bit-identical to the frozen Library control (`library_dispatch:`), and the profile
  must show no candidate kernels on their Library paths.

Each timing row records the basis of its decision in `gate`, with the bound, the
observed spread and a measurability count.

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
  of the PTX bytes the process loaded. A library call is allowed only inside the
  locked projection helper, as cuBLAS, at a shape the projection baseline covers.
  Host-device copies are refused.
- A Library path may not launch a candidate-area entry.
- Every declared boundary must have a candidate scope with kernels.

The numeric and timing processes also track library calls made while a candidate
scope is open, and, during graph capture, enumerate the nodes each candidate scope
adds to the captured graph, side-stream branches included, and resolve kernel names with `cuFuncGetName` /
`cuKernelGetName`. Library kernels outside the projection helper, unlisted kernels,
host copies and host or event nodes fail (`profile:captured_library_calls`,
`profile:graph_nodes`). The driver labels every capture with its case, and
`profile:graph_nodes` also fails unless each case's captured candidate-kernel set
equals that case's eager-profile candidate-kernel set. Graph mode is set and verified
by the locked driver. Projection-node permissions are reset when the driver capture
ID changes, so node addresses reused after graph destruction cannot hide kernels.
The ignored `sequential_capture_keeps_candidate_kernels` regression runs fresh
captures at b1, b1, b7, b32, b33, b64 and b7 in one process with scope tracking on.
Every capture must report exactly nine candidate kernels, excluding its one
projection kernel. The ignored `secret_library_algorithms` proof measures Standard and
PersistStaticSmallH separately against f64 on the same secret input. Both must have
FP32 rounding-scale error (below 1e-4 relative L2), and a one-element, one-ULP nudge
must change the SmallH error by at most 10%. The SmallH/Standard error ratio is
information only; the proof does not require one algorithm to pass the other
algorithm's gate. The capture and soundness tests need the qualification environment
and GPU lock. The ignored `f64_reference_matches_fixture_rounding` and
`conv_f64_matches_fixture_rounding` tests use fixture intermediates and run on CPU.

### Loaded PTX

The allow-list is built from the exact bytes the locked loader handed to the driver.
Each loaded module must match a committed file under `src/inference/cuda/ptx/` or
`tests/cuda_qualify/device/` byte for byte and by area name, and its parsed entries
must match. Any PTX in a subdirectory of `ptx/` is refused. The result records the
module hashes and the areas loaded by each numeric, timing and profile process.
Non-candidate modules must have identical bytes in every process. Each candidate
area must have identical bytes in every process that loads it. A process that runs
at least one declared triple must load its candidate area. A process that runs no
declared triple must load no candidate area. Thus FP32-only coverage loads candidate
PTX in FP32 processes, but not in TF32 processes.

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

LSTM candidates may call cuBLAS projections, which the include filter does not
instrument. The locked projection baseline runs all three tools, unfiltered, on every
shape the helper can issue at a harness batch (m = 589 x 1, 7, 32, 33, 64; n = 128 to
512; k = 60 and 256; both modes) and is retained under `tests/cuda_qualify/baselines/`.
The profile rule allows only those shapes.

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

A mutant is caught only by its exact tier-stripped check name and reason:

| Mutant | Planted defect | Caught by |
| --- | --- | --- |
| `Precision` | rounds real inputs to BF16 | `layer:` with reason `layer parity` |
| `Shape` | correct only at b32 | `layer:` `layer parity`, failing exactly the non-b32 rows |
| `Tail` | zeroes the partial tile | `layer:` `layer parity`, failing every partial-tile row |
| `Fallback` | library work in the candidate scope | `profile` with `forbidden library kernels` |
| `Atomic` | floating-point atomic accumulation | `determinism:fixed_reduction_order` with `floating-point atomic in launched custom entry` |
| `Slow` | the library operation three times | `speed:` with `candidate slower than the faster Library process` |
| `PhaseCheat` | skips its work in the timing phase | `timing_output:` with `final output differs` |
| `Unscoped` | launches after its scope closes | `profile` with `outside a candidate or library range` |
| `Unlisted` | launches a kernel from unrecorded PTX | `profile` with `not on the loaded PTX allow-list` |
| `Lookup` | answers from the first weights or input it saw, like a baked table | `secret:` with `differs from Library on a fresh input` |
| `UninitShared` | launches an entry that loads shared memory no thread stored | `ptx:shared_initialization` with `shared load` |

An escaped mutant is written as `escaped` and exits 4. A mutant runs only the phases
its gate reads (always numeric, which includes the secret-input and PTX checks;
profile for `Fallback`, `Atomic`, `Unscoped` and `Unlisted`; timing for `Slow` and
`PhaseCheat`), and no sanitizer tools. Library
controls run every phase plus the three positive sanitizer controls; candidates run
every phase, the positive controls and the candidate sanitizer. The static scan has its own
proof: `tests/cuda_qualify/scan_fixtures/phase_cheat.rs` reads `SPEAKRS_QUALIFY_PHASE`,
and `unscanned_call.rs` hands its work to code outside the candidate tree; both must be
refused, in the unit tests and live, copied over the LSTM candidate in a scratch tree.
A new module elsewhere under `src/` fails the lock before Python starts. `PhaseCheat`
is the phase cheat with the scan bypassed.

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
archive retains the exact source used for the original qualification; it is not
rewritten during integration.

## Production qualification manifest

CI runs `python3 scripts/cuda/qualify/qualified.py check` on every push. This CPU-only
check binds each `implementation::PRODUCTION` candidate area to `QUALIFIED.json`.
It checks every shipped PTX tier, its area `.manifest`, all candidate host and
kernel area source modules, the shared `candidate.rs` execution helpers, and the
exact coverage declared in the code. Added or removed files and missing production entries fail. Unfamiliar coverage syntax fails
closed and needs a checker update.

A kernel change needs a new qualification record, then acceptance, then a manifest
commit:

```sh
cargo xtask cuda-qualify <resnet|lstm|sincnet> Oxide
python3 scripts/cuda/qualify/qualified.py accept /archive/qualify-<area>-Oxide-<run>.json
python3 scripts/cuda/qualify/qualified.py check
python3 scripts/cuda/qualify/lock.py
git add scripts/cuda/qualify/QUALIFIED.json scripts/cuda/qualify/LOCK
```

The accept command also reads `.json.gz` archives. It verifies that the record's
loaded candidate PTX hashes match every shipped tier and that recorded coverage
matches the code. It stores the record filename and SHA-256 of its exact file bytes,
the original qualification lock digest, source hashes, coverage and verdict basis.
The original lock digest is historical evidence; it does not claim that the current
harness has the same digest. A manifest update changes the current lock, so re-lock
after acceptance. Do not rewrite the frozen Library control or the old records.

Each tier must first contain complete phase evidence and a completed harness verdict.
Acceptance then requires `accepts_replacement: true` with no failed checks, or this
rule: all non-passing checks must be speed checks blocked only by Library process spread,
with no hard failures. For each blocked case, the smaller of the two Library over
candidate speedups must be at least `1 + 3 * max(Library spread, spread bound)`.
The command recomputes the spread and both speedups from the four recorded medians.
The manifest retains the table of blocked cases and their margins. The qualification
measurement gates remain unchanged; this command records the acceptance decision.

### Retained qualification records

The initial entries use these outside-repository archives under
`~/code/research/cuda-kernel-opportunities/speakrs-native-cuda-records-2026-10-04/qualification/`:

| Area | Archive | Verdict basis | Original lock prefix |
| --- | --- | --- | --- |
| resnet | `k1/qualify-resnet-Oxide-20261004T064208.284837Z.json.gz` | Library-noise rule, 20 blocked cases | `368389d2` |
| lstm | `k2/qualify-lstm-Oxide-20261004T095844.546766Z.json.gz` | Library-noise rule, 3 blocked cases | `4bb7818f` |
| sincnet | `k3/qualify-sincnet-Oxide-20261004T093054.587886Z.json.gz` | `accepts_replacement` | `368389d2` |

**Source evidence gap:** schema-3 records contain loaded PTX hashes and coverage, but
not host-source, kernel-source, or PTX build-manifest hashes. The initial manifest
captures these hashes at acceptance, including the device-capacity fallback change.
These hashes prevent later unrecorded changes; they do not prove that the old run
used the same source or build-manifest bytes. Keep this gap visible in each entry. Do not treat a source-only acceptance of
an old record as a new qualification. Future host changes need a new run under the
updated harness, as do kernel and PTX changes. The archived PTX hashes match the
initial shipped files exactly.
