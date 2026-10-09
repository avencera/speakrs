//! Development check of the segdense plans: each boundary against an f64 truth and
//! against the library call it replaces, on the sm75 and sm80 PTX tiers, with a timed
//! library A/B on the real model shapes
//!
//! This is evidence for a driver-only build, not a qualification. Run it under the
//! GPU lock with `--ignored --nocapture`; it reads the native weights from
//! `SEGDENSE_WEIGHTS` (default `/workspace/speakrs-cuda-ref/models-native`) and the
//! intermediate tensors from the reference directory. `SEGDENSE_SITE`,
//! `SEGDENSE_BATCH` and `SEGDENSE_TIERS` (such as `sm80`) narrow the cases, and
//! `SEGDENSE_HARDWARE=a100` picks the configurations an A100 would get, which checks
//! their correctness on another GPU without timing them

use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaFunction, CudaGraph, CudaSlice, LaunchConfig, PushKernelArg};

use super::super::candidate::{
    DenseCandidate, DenseOxide, DenseSite, DenseSpec, Phases, SegConvCandidate, SegConvOxide,
    SegConvSite, SegConvSpec, SegdensePin,
};
use super::super::device::DeviceAttributes;
use super::super::device::test_support::Builder;
use super::super::dnn::ConvPlan;
use super::super::{
    ComputeCapability, Conv2d, ConvPlanner, CudaError, CudaLibrary, CudaMath, CudaRuntime,
    KernelModule, LoadedKernels, PtxTier, SafetensorsFile, Sgemm,
};
use super::{reference_dir, runtime_with_tier};

/// One boundary of the area, with the record name the evidence uses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Site {
    Conv(SegConvSite),
    Dense(DenseSite),
}

impl Site {
    const ALL: [(Self, &'static str); 6] = [
        (Self::Conv(SegConvSite::Conv1), "conv1"),
        (Self::Conv(SegConvSite::Conv2), "conv2"),
        (Self::Dense(DenseSite::Linear0), "linear0"),
        (Self::Dense(DenseSite::Linear1), "linear1"),
        (Self::Dense(DenseSite::Classifier), "linear2"),
        (Self::Dense(DenseSite::Embedding), "embedding"),
    ];

    /// `(rows per batch item, columns, reduction, NCW input length)`; convolution
    /// rows are output positions and their input length is nonzero
    fn dimensions(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Conv(SegConvSite::Conv1) => (5321, 60, 400, 5325),
            Self::Conv(SegConvSite::Conv2) => (1769, 60, 300, 1773),
            Self::Dense(DenseSite::Linear0) => (589, 128, 256, 0),
            Self::Dense(DenseSite::Linear1) => (589, 128, 128, 0),
            Self::Dense(DenseSite::Classifier) => (589, 7, 128, 0),
            Self::Dense(DenseSite::Embedding) => (3, 256, 5120, 0),
        }
    }
}

struct Data {
    input: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
}

fn read(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, CudaError> {
    let shape = file.shape(name).ok_or_else(|| CudaError::MissingTensor {
        path: file.path().into(),
        name: name.into(),
    })?;
    file.read_f32(name, shape)
}

fn load(reference: &std::path::Path, site: Site, batch: usize) -> Result<Data, CudaError> {
    let embedding = site == Site::Dense(DenseSite::Embedding);
    let model = if embedding {
        "wespeaker-multimask-tail"
    } else {
        "segmentation-3.0"
    };
    let weights_dir = std::env::var("SEGDENSE_WEIGHTS")
        .unwrap_or_else(|_| "/workspace/speakrs-cuda-ref/models-native".into());
    let (dir, case) = if batch == 1 {
        (model.to_string(), "test_first_b1")
    } else {
        (format!("{model}-b32"), "test_and_short_b32")
    };
    let weights = SafetensorsFile::open(format!("{weights_dir}/{model}.safetensors"))?;
    let reference = SafetensorsFile::open(reference.join(dir).join(format!("{case}.safetensors")))?;
    let input_name = match site {
        Site::Conv(SegConvSite::Conv1) => "tensor//sincnet/LeakyRelu_output_0",
        Site::Conv(SegConvSite::Conv2) => "tensor//sincnet/LeakyRelu_1_output_0",
        Site::Dense(DenseSite::Linear0) => "tensor//lstm/Transpose_5_output_0",
        Site::Dense(DenseSite::Linear1) => "tensor//LeakyRelu_output_0",
        Site::Dense(DenseSite::Classifier) => "tensor//LeakyRelu_1_output_0",
        Site::Dense(DenseSite::Embedding) => "tensor/where_1",
    };

    let mut matmuls: Vec<_> = weights
        .names()
        .into_iter()
        .filter(|name| name.starts_with("onnx::MatMul_"))
        .collect();
    matmuls.sort_by_key(|name| {
        name.rsplit('_')
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    });

    let (weight, bias) = match site {
        Site::Conv(SegConvSite::Conv1) => (read(&weights, "sincnet.conv1d.1.weight")?, Vec::new()),
        Site::Conv(SegConvSite::Conv2) => (read(&weights, "sincnet.conv1d.2.weight")?, Vec::new()),
        Site::Dense(DenseSite::Linear0) => (
            read(&weights, &matmuls[0])?,
            read(&weights, "linear.0.bias")?,
        ),
        Site::Dense(DenseSite::Linear1) => (
            read(&weights, &matmuls[1])?,
            read(&weights, "linear.1.bias")?,
        ),
        Site::Dense(DenseSite::Classifier) => (
            read(&weights, &matmuls[2])?,
            read(&weights, "classifier.bias")?,
        ),
        Site::Dense(DenseSite::Embedding) => (
            read(&weights, "resnet.seg_1.weight")?,
            read(&weights, "resnet.seg_1.bias")?,
        ),
    };

    Ok(Data {
        input: read(&reference, input_name)?,
        weight,
        bias,
    })
}

/// The library calls a segdense plan replaces, with the area epilogue kernels they
/// need around them
struct Library {
    conv: Option<ConvPlan>,
    workspace: CudaSlice<u8>,
    gemm: Sgemm,
    site: Site,
    epilogue: CudaFunction,
}

impl Library {
    fn plan(
        runtime: &CudaRuntime,
        site: Site,
        batch: usize,
        math: CudaMath,
    ) -> Result<Self, CudaError> {
        let (rows, n, k, input_len) = site.dimensions();
        runtime.prepare_library(CudaLibrary::Cublas)?;
        let conv = match site {
            Site::Conv(_) => Some(ConvPlanner::new(runtime)?.plan(Conv2d {
                batch,
                in_channels: k / 5,
                out_channels: n,
                input: [1, input_len],
                kernel: [1, 5],
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                math,
            })?),
            Site::Dense(_) => None,
        };
        let workspace = runtime
            .stream()
            .alloc_zeros(conv.as_ref().map_or(0, ConvPlan::workspace_bytes))?;
        let (module, name) = match site {
            Site::Dense(DenseSite::Embedding) => {
                (KernelModule::Embedding, "embedding_broadcast_rows")
            }
            Site::Dense(DenseSite::Classifier) => {
                (KernelModule::Segmentation, "segmentation_bias_log_softmax")
            }
            _ => (KernelModule::Segmentation, "segmentation_bias_leaky"),
        };
        let epilogue = runtime.load_kernels(module)?.function(name)?;
        let embedding = site == Site::Dense(DenseSite::Embedding);
        let gemm = Sgemm {
            math,
            b_transposed: embedding,
            beta: if embedding { 1.0 } else { 0.0 },
            ..Sgemm::new(rows * batch, n, k)
        };

        Ok(Self {
            conv,
            workspace,
            gemm,
            site,
            epilogue,
        })
    }

    fn enqueue(
        &mut self,
        runtime: &CudaRuntime,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        if let Some(conv) = &self.conv {
            return conv.forward(
                &mut self.workspace.as_view_mut(),
                &input.as_view(),
                &weight.as_view(),
                &mut output.as_view_mut(),
            );
        }

        let embedding = self.site == Site::Dense(DenseSite::Embedding);
        if embedding {
            self.epilogue(runtime, bias, output)?;
        }
        runtime.sgemm(self.gemm, input, weight, output)?;
        if !embedding {
            self.epilogue(runtime, bias, output)?;
        }

        Ok(())
    }

    fn epilogue(
        &self,
        runtime: &CudaRuntime,
        bias: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let classifier = self.site == Site::Dense(DenseSite::Classifier);
        let lens = [
            bias.len() as u64,
            if classifier {
                (output.len() / 7) as u64
            } else {
                output.len() as u64
            },
        ];
        let cols = self.gemm.n as u32;
        let slope = 0.01f32;
        let mut launch = runtime.stream().launch_builder(&self.epilogue);
        launch.arg(bias).arg(&lens[0]);
        match self.site {
            Site::Dense(DenseSite::Embedding) => {
                launch.arg(&cols);
            }
            Site::Dense(DenseSite::Classifier) => {}
            _ => {
                launch.arg(&slope);
            }
        }
        launch.arg(output).arg(&lens[1]);
        // safety: the arguments match the area kernel's slice ABI; classifier output
        // is viewed as seven-element rows, with its length adjusted accordingly
        unsafe { launch.launch(LaunchConfig::for_num_elems(lens[1] as u32)) }?;
        Ok(())
    }
}

/// A planned segdense boundary of either trait
enum Plan {
    Conv(SegConvOxide),
    Dense(DenseOxide),
}

impl Plan {
    fn new(
        runtime: &CudaRuntime,
        kernels: &LoadedKernels,
        device: &DeviceAttributes,
        site: Site,
        batch: usize,
        math: CudaMath,
        case: &Case,
    ) -> Result<(Self, SegdensePin), CudaError> {
        let refused = |error| CudaError::Unsupported {
            context: "segdense dev check",
            reason: format!("{site:?} batch {batch} {math:?}: {error}"),
        };
        match site {
            Site::Conv(site) => {
                let spec = SegConvSpec::new(site, batch, math).map_err(refused)?;
                let pin =
                    SegConvOxide::implemented_pin(spec, kernels.tier(), device).map_err(refused)?;
                let plan = SegConvOxide::plan(runtime, kernels, spec, &case.weight, pin)
                    .map_err(refused)?;
                Ok((Self::Conv(plan), pin))
            }
            Site::Dense(site) => {
                let spec = DenseSpec::new(site, batch, math).map_err(refused)?;
                let pin =
                    DenseOxide::implemented_pin(spec, kernels.tier(), device).map_err(refused)?;
                let plan = DenseOxide::plan(runtime, kernels, spec, &case.weight, &case.bias, pin)
                    .map_err(refused)?;
                Ok((Self::Dense(plan), pin))
            }
        }
    }

    fn enqueue(
        &self,
        runtime: &CudaRuntime,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        let phases = Phases::new();
        match self {
            Self::Conv(plan) => plan.enqueue(
                &input.as_view(),
                &mut output.as_view_mut(),
                &phases,
                runtime,
            ),
            Self::Dense(plan) => plan.enqueue(
                &input.as_view(),
                &mut output.as_view_mut(),
                &phases,
                runtime,
            ),
        }
    }
}

fn capture(
    runtime: &CudaRuntime,
    repetitions: usize,
    mut enqueue: impl FnMut() -> Result<(), CudaError>,
) -> Result<CudaGraph, CudaError> {
    let stream = runtime.stream();
    stream.synchronize()?;
    let context = runtime.context();
    let tracking = context.is_event_tracking();
    // safety: all buffers use this one synchronized stream throughout capture
    unsafe { context.disable_event_tracking() };
    let result = stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(CudaError::from)
        .and_then(|()| {
            let result = (0..repetitions).try_for_each(|_| enqueue());
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            result?;
            Ok(graph?)
        });
    if tracking {
        // safety: restore the context state after capture has ended
        unsafe { context.enable_event_tracking() };
    }

    result?.ok_or_else(|| CudaError::Unsupported {
        context: "segdense capture",
        reason: "empty graph".into(),
    })
}

/// Milliseconds per enqueue of ten replays of `graph`, after five warm-up replays
fn timed(
    runtime: &CudaRuntime,
    graph: &CudaGraph,
    repetitions: usize,
) -> Result<Vec<f64>, CudaError> {
    for _ in 0..5 {
        graph.launch()?;
    }

    runtime.synchronize()?;
    (0..10)
        .map(|_| {
            let stream = runtime.stream();
            let start = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            graph.launch()?;
            let end = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            end.synchronize()?;
            Ok(f64::from(start.elapsed_ms(&end)?) / repetitions as f64)
        })
        .collect()
}

/// Output indices with an f64 truth: an even spread including the first and last
/// element, one element of every (batch item, channel) pair, and the first and last
/// position or row of every channel in the first and last batch item, which lie in
/// the final partial tile
fn samples(site: Site, batch: usize) -> Vec<usize> {
    let (rows, n, _, input_len) = site.dimensions();
    let len = batch * rows * n;
    let index = |b: usize, row: usize, column: usize| {
        if input_len == 0 {
            (b * rows + row) * n + column
        } else {
            (b * n + column) * rows + row
        }
    };

    let spread = len.min(1024);
    let mut picked: Vec<usize> = (0..spread)
        .map(|sample| sample * (len - 1) / (spread - 1))
        .collect();
    for b in 0..batch {
        for column in 0..n {
            // a fixed scatter over the rows, so no two pairs share a row pattern
            let row = (b * 7919 + column * 104_729) % rows;
            picked.push(index(b, row, column));
        }
    }
    for b in [0, batch - 1] {
        for column in 0..n {
            picked.push(index(b, 0, column));
            picked.push(index(b, rows - 1, column));
        }
    }

    picked.sort_unstable();
    picked.dedup();
    picked
}

/// Max-abs and relative-L2 error of `actual` against an f64 truth at `indices`
fn truth_errors(site: Site, data: &Data, indices: &[usize], actual: &[f32]) -> (f64, f64) {
    let (rows, n, k, input_len) = site.dimensions();
    let (mut max, mut squares, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for &index in indices {
        let (row, column, b) = if input_len == 0 {
            (index / n, index % n, 0)
        } else {
            (index % rows, (index / rows) % n, index / (rows * n))
        };
        let dot = |column: usize| {
            let mut sum = 0.0f64;
            for q in 0..k {
                let a = if input_len == 0 {
                    row * k + q
                } else {
                    (b * (k / 5) + q / 5) * input_len + row + q % 5
                };
                let transposed = input_len != 0 || site == Site::Dense(DenseSite::Embedding);
                let w = if transposed {
                    column * k + q
                } else {
                    q * n + column
                };
                sum += f64::from(data.input[a]) * f64::from(data.weight[w]);
            }
            if input_len == 0 {
                sum += f64::from(data.bias[column]);
            }
            sum
        };

        let mut truth = dot(column);
        match site {
            Site::Dense(DenseSite::Linear0 | DenseSite::Linear1) if truth < 0.0 => {
                truth *= f64::from(0.01f32);
            }
            Site::Dense(DenseSite::Classifier) => {
                let logits = std::array::from_fn::<_, 7, _>(dot);
                let largest = logits.into_iter().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = logits.into_iter().map(|v| (v - largest).exp()).sum();
                truth = logits[column] - largest - sum.ln();
            }
            _ => {}
        }

        let error = f64::from(actual[index]) - truth;
        max = max.max(error.abs());
        squares += error * error;
        scale += truth * truth;
    }

    (max, (squares / scale.max(f64::MIN_POSITIVE)).sqrt())
}

/// Max-abs and relative-L2 difference of `actual` from `expected` over every element
fn difference(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    let (mut max, mut squares, mut scale) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &e) in actual.iter().zip(expected) {
        let error = f64::from(a) - f64::from(e);
        max = max.max(error.abs());
        squares += error * error;
        scale += f64::from(e) * f64::from(e);
    }
    (max, (squares / scale.max(f64::MIN_POSITIVE)).sqrt())
}

/// Per-site floors cover rounding when the library happens to be exact
fn numerical_pass(site: Site, math: CudaMath, ours: (f64, f64), library: (f64, f64)) -> bool {
    if ![ours.0, ours.1, library.0, library.1]
        .into_iter()
        .all(f64::is_finite)
    {
        return false;
    }
    let (absolute, relative) = match (site, math) {
        (Site::Conv(_), CudaMath::Fp32) => (1e-4, 1e-5),
        (Site::Dense(DenseSite::Classifier), CudaMath::Fp32) => (1e-5, 1e-6),
        (Site::Dense(DenseSite::Embedding), CudaMath::Fp32) => (1e-3, 1e-4),
        (Site::Dense(_), CudaMath::Fp32) => (1e-4, 1e-5),
        (Site::Conv(_), CudaMath::Tf32) => (1e-2, 2e-3),
        (Site::Dense(DenseSite::Classifier), CudaMath::Tf32) => (1e-3, 2e-3),
        (Site::Dense(DenseSite::Embedding), CudaMath::Tf32) => (1e-2, 2e-3),
        (Site::Dense(_), CudaMath::Tf32) => (1e-2, 2e-3),
    };
    // a bad library control cannot make a grossly wrong kernel acceptable
    let ceiling = match math {
        CudaMath::Fp32 => 1e-3,
        CudaMath::Tf32 => 2e-2,
    };
    ours.0 <= 10.0 * library.0 + absolute && ours.1 <= (10.0 * library.1 + relative).min(ceiling)
}

#[test]
fn numerical_limits_reject_zero_for_nonzero_f64_truth_at_every_site() {
    for (site, name) in Site::ALL {
        let (rows, n, k, input_len) = site.dimensions();
        let data = Data {
            input: vec![
                1.0;
                if input_len == 0 {
                    rows * k
                } else {
                    k / 5 * input_len
                }
            ],
            weight: vec![1.0; k * n],
            bias: vec![0.0; n],
        };
        let truth = if site == Site::Dense(DenseSite::Classifier) {
            -7f32.ln()
        } else {
            k as f32
        };
        let correct = vec![truth; rows * n];
        let zero = vec![0.0; correct.len()];
        let indices = [0, correct.len() - 1];
        let library = truth_errors(site, &data, &indices, &correct);
        let wrong = truth_errors(site, &data, &indices, &zero);
        assert!(wrong.0 > 0.0 && wrong.1 > 0.9, "{name}: nonzero reference");
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            assert!(
                numerical_pass(site, math, library, library),
                "{name} {math:?}: correct result"
            );
            assert!(
                !numerical_pass(site, math, wrong, library),
                "{name} {math:?}: zero result"
            );
            assert!(
                !numerical_pass(site, math, wrong, wrong),
                "{name} {math:?}: bad control"
            );
            for invalid in [f64::NAN, f64::INFINITY] {
                assert!(!numerical_pass(site, math, (invalid, 0.0), library));
                assert!(!numerical_pass(site, math, library, (0.0, invalid)));
            }
        }
    }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.iter()
        .map(|v| v.to_bits())
        .eq(b.iter().map(|v| v.to_bits()))
}

/// Old eager values must not hide missing stores in a captured replay
fn poison(runtime: &CudaRuntime, output: &mut CudaSlice<f32>) -> Result<(), CudaError> {
    runtime
        .stream()
        .memcpy_htod(&vec![f32::NAN; output.len()], output)?;
    Ok(())
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// Device copies of one case's operands and outputs
struct Case {
    data: Data,
    indices: Vec<usize>,
    input: CudaSlice<f32>,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    actual: CudaSlice<f32>,
    expected: CudaSlice<f32>,
}

impl Case {
    fn load(
        runtime: &CudaRuntime,
        reference: &std::path::Path,
        site: Site,
        batch: usize,
    ) -> Result<Self, CudaError> {
        let data = load(reference, site, batch)?;
        let stream = runtime.stream();
        let (rows, n, _, _) = site.dimensions();
        Ok(Self {
            indices: samples(site, batch),
            input: stream.clone_htod(&data.input)?,
            weight: stream.clone_htod(&data.weight)?,
            // the convolutions take no bias; keep one element so the copy is valid
            bias: stream.clone_htod(if data.bias.is_empty() {
                &[0.0][..]
            } else {
                &data.bias[..]
            })?,
            actual: stream.alloc_zeros(batch * rows * n)?,
            expected: stream.alloc_zeros(batch * rows * n)?,
            data,
        })
    }
}

/// The tiers `SEGDENSE_TIERS` names, by default sm75 and sm80
fn tiers() -> Vec<PtxTier> {
    std::env::var("SEGDENSE_TIERS").map_or(vec![PtxTier::Sm75, PtxTier::Sm80], |tiers| {
        tiers
            .split(',')
            .map(|tier| tier.parse().expect("SEGDENSE_TIERS lists PTX tiers"))
            .collect()
    })
}

/// The site and batch filters of `SEGDENSE_SITE` and `SEGDENSE_BATCH`
fn selected() -> impl Iterator<Item = (Site, &'static str, usize)> {
    Site::ALL.into_iter().flat_map(|(site, name)| {
        [1, 32].into_iter().filter_map(move |batch| {
            let site_ok = std::env::var("SEGDENSE_SITE").map_or(true, |v| v == name);
            let batch_ok = std::env::var("SEGDENSE_BATCH").map_or(true, |v| v == batch.to_string());
            (site_ok && batch_ok).then_some((site, name, batch))
        })
    })
}

/// The runtime's device, or with `SEGDENSE_HARDWARE=a100` an A100-SXM4's attributes
fn device(runtime: &CudaRuntime) -> (DeviceAttributes, bool) {
    match std::env::var("SEGDENSE_HARDWARE").as_deref() {
        Ok("a100") => (
            Builder::new(ComputeCapability::new(8, 0))
                .multiprocessors(108)
                .shared_optin_bytes(163 * 1024)
                .name("forced A100-SXM4")
                .build(),
            true,
        ),
        _ => (runtime.device().clone(), false),
    }
}

#[test]
#[ignore = "requires GPU and reference tensors; hold gpu-bench.lock"]
fn segdense_dev_check() -> Result<(), CudaError> {
    let test = "segdense_dev_check";
    let Some(reference) = reference_dir(test, "segmentation-3.0") else {
        return Ok(());
    };
    for tier in tiers() {
        let Some(runtime) = runtime_with_tier(test, Some(tier)) else {
            return Ok(());
        };
        let request = runtime.embedded_exact_request(KernelModule::Segdense)?;
        assert_eq!(
            request.tier(),
            tier,
            "the build embeds the {tier} segdense PTX"
        );
        let kernels = runtime.load_module(request)?;
        let (device, forced) = device(&runtime);
        eprintln!(
            "MODULE\ttier={tier}\tdevice={}\tsms={}\tartifact={:?}\tptx_sha256={}\tforced={forced}",
            runtime.compute_capability(),
            runtime.device().multiprocessors(),
            kernels.artifact(),
            kernels.ptx_sha256()
        );
        for (site, name, batch) in selected() {
            let mut case = Case::load(&runtime, &reference, site, batch)?;
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                run_case(
                    &runtime,
                    &kernels,
                    &device,
                    forced,
                    &mut case,
                    (site, name, batch, math),
                )?;
            }
        }
    }

    Ok(())
}

fn run_case(
    runtime: &CudaRuntime,
    kernels: &LoadedKernels,
    device: &DeviceAttributes,
    forced: bool,
    case: &mut Case,
    (site, name, batch, math): (Site, &str, usize, CudaMath),
) -> Result<(), CudaError> {
    let stream = runtime.stream();
    let tier = kernels.tier();
    let (plan, pin) = Plan::new(runtime, kernels, device, site, batch, math, case)?;
    let mut library = Library::plan(runtime, site, batch, math)?;
    let Case {
        data,
        indices,
        input,
        weight,
        bias,
        actual,
        expected,
    } = case;

    poison(runtime, actual)?;
    plan.enqueue(runtime, input, actual)?;
    poison(runtime, expected)?;
    library.enqueue(runtime, input, weight, bias, expected)?;
    let eager = stream.clone_dtoh(actual)?;
    let library_values = stream.clone_dtoh(expected)?;
    assert!(
        eager.iter().chain(&library_values).all(|v| v.is_finite()),
        "{name} b{batch} {math:?} {tier}: nonfinite output"
    );

    let ours = truth_errors(site, data, indices, &eager);
    let theirs = truth_errors(site, data, indices, &library_values);
    let versus = difference(&eager, &library_values);
    assert!(
        numerical_pass(site, math, ours, theirs) && versus.0.is_finite() && versus.1.is_finite(),
        "{name} b{batch} {math:?} {tier}: f64 error {ours:?}, library {theirs:?}, versus {versus:?}"
    );
    let repetitions = if std::env::var_os("SEGDENSE_SANITIZE").is_some() {
        1
    } else {
        20
    };
    let graph = capture(runtime, repetitions, || {
        plan.enqueue(runtime, input, actual)
    })?;
    // a replay must store every element and repeat the eager bits exactly
    for check in ["eager/graph", "repeat"] {
        poison(runtime, actual)?;
        graph.launch()?;
        assert!(
            same(&eager, &stream.clone_dtoh(actual)?),
            "{check} differs: {name} b{batch} {math:?} {tier}"
        );
    }

    let config = pin.config();
    let speed = if forced || std::env::var_os("SEGDENSE_SANITIZE").is_some() {
        "untimed".to_owned()
    } else {
        let library_graph = capture(runtime, repetitions, || {
            library.enqueue(runtime, input, weight, bias, expected)
        })?;
        let (mut lib, mut own) = (Vec::new(), Vec::new());
        // interleaved arms, so drift and clock changes hit both alike
        for _ in 0..3 {
            lib.extend(timed(runtime, &library_graph, repetitions)?);
            own.extend(timed(runtime, &graph, repetitions)?);
        }
        let (lib, own) = (median(&lib), median(&own));
        format!(
            "library_ms={lib:.4}\tours_ms={own:.4}\tspeedup={:.3}",
            lib / own
        )
    };
    eprintln!(
        "CASE\t{name}\tb{batch}\t{math:?}\t{tier}\tkernel={}\tsplits={:?}\t{speed}\t\
         f64_library={:.3e}/{:.3e}\tf64_ours={:.3e}/{:.3e}\tvs_library={:.3e}/{:.3e}\t\
         samples={}\tgraph=equal",
        config.kernel,
        pin.splits(),
        theirs.0,
        theirs.1,
        ours.0,
        ours.1,
        versus.0,
        versus.1,
        indices.len()
    );

    Ok(())
}
