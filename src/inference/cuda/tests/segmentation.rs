//! Parity and speed of the native CUDA segmentation model against the ONNX Runtime
//! CPU references written by `scripts/cuda/make_reference.py`
//!
//! The references live in `SPEAKRS_CUDA_REF`, else `/workspace/ref` on the GPU box,
//! else `~/Library/Caches/speakrs-cuda-ref`. Tests skip without a GPU or without the
//! references; `SPEAKRS_REQUIRE_GPU=1` turns both skips into failures
//!
//! The benchmark is ignored by default. Run it with the GPU lock held:
//! `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib segmentation_benchmark -- --ignored --nocapture`
use std::time::{Duration, Instant};

use super::super::segmentation::SegmentationTensor;
use super::super::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime, CudaSegmentation, SafetensorsFile,
    SegmentationOptions,
};
use super::{reference_dir, runtime};

const CLASSES: usize = CudaSegmentation::CLASSES;
const WINDOW_SAMPLES: usize = CudaSegmentation::WINDOW_SAMPLES;

/// FP32 logits may differ from the CPU reference by summation order only; this bounds
/// the drift through four LSTM layers over 589 steps
const FP32_LOGIT_TOLERANCE: f32 = 2e-3;
/// TF32 rounds GEMM and convolution inputs to 10 mantissa bits; this only catches a
/// broken TF32 path, whether TF32 is acceptable is decided by a DER run
const TF32_LOGIT_TOLERANCE: f32 = 0.5;

/// FP32 with the standard LSTM and no graph, the configuration closest to the references
const FP32_EAGER: SegmentationOptions = SegmentationOptions {
    math: CudaMath::Fp32,
    lstm_algo: CudaLstmAlgorithm::Standard,
    cuda_graph: false,
};

/// One reference case: the model export it came from and its batch
#[derive(Debug, Clone, Copy)]
struct Case {
    model: &'static str,
    name: &'static str,
    batch: usize,
}

const CASES: [Case; 4] = [
    Case {
        model: "segmentation-3.0",
        name: "test_first_b1",
        batch: 1,
    },
    Case {
        model: "segmentation-3.0",
        name: "test_last_partial_b1",
        batch: 1,
    },
    Case {
        model: "segmentation-3.0",
        name: "test_short_partial_b1",
        batch: 1,
    },
    Case {
        model: "segmentation-3.0-b32",
        name: "test_and_short_b32",
        batch: 32,
    },
];

struct Loaded {
    weights: SafetensorsFile,
    reference: SafetensorsFile,
}

impl Case {
    fn load(&self, dir: &std::path::Path) -> Result<Loaded, CudaError> {
        let model_dir = dir.join(self.model);
        Ok(Loaded {
            weights: SafetensorsFile::open(model_dir.join(format!("{}.safetensors", self.model)))?,
            reference: SafetensorsFile::open(model_dir.join(format!("{}.safetensors", self.name)))?,
        })
    }
}

fn read(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, CudaError> {
    let shape = file
        .shape(name)
        .ok_or_else(|| CudaError::MissingTensor {
            path: file.path().to_path_buf(),
            name: name.to_owned(),
        })?
        .to_vec();
    file.read_f32(name, &shape)
}

fn max_abs_diff(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs())
        .fold(0.0, f32::max)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0
}

/// `(batch row, frame)` of every frame whose powerset argmax differs
fn argmax_flips(actual: &[f32], expected: &[f32], frames: usize) -> Vec<(usize, usize)> {
    actual
        .chunks(CLASSES)
        .zip(expected.chunks(CLASSES))
        .enumerate()
        .filter(|(_, (a, e))| argmax(a) != argmax(e))
        .map(|(row, _)| (row / frames, row % frames))
        .collect()
}

/// `[b, c, t]` to `[b, t, c]`
fn transpose_last(values: &[f32], batch: usize, channels: usize, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; values.len()];
    for b in 0..batch {
        for c in 0..channels {
            for t in 0..len {
                out[(b * len + t) * channels + c] = values[(b * channels + c) * len + t];
            }
        }
    }
    out
}

/// Adds a per-channel bias to `[b, c, t]`
fn add_channel_bias(values: &[f32], bias: &[f32], len: usize) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| value + bias[(index / len) % bias.len()])
        .collect()
}

struct Parity {
    max_error: f32,
    flips: Vec<(usize, usize)>,
    layers: Vec<(&'static str, f32)>,
}

/// Runs one case and compares the output and every intermediate the model keeps
fn run_case(
    runtime: &CudaRuntime,
    case: Case,
    loaded: &Loaded,
    options: SegmentationOptions,
    runs: usize,
) -> Result<Parity, CudaError> {
    let reference = &loaded.reference;
    let mut model = CudaSegmentation::new(runtime, &loaded.weights, options)?;
    let input = read(reference, "input/input")?;

    // later runs replay the captured graph when it is on
    let mut output = Vec::new();
    for _ in 0..runs {
        output = model.run(runtime, case.batch, &input)?;
    }
    let workspace = model
        .find_workspace(case.batch, WINDOW_SAMPLES)
        .expect("run allocates the workspace");
    let shape = workspace.shape();
    let expected = read(reference, "tensor/output")?;
    let stream = runtime.stream();
    let download = |tensor| workspace.tensor(tensor).download(stream);

    let conv1_bias = read(&loaded.weights, "sincnet.conv1d.1.bias")?;
    let conv2_bias = read(&loaded.weights, "sincnet.conv1d.2.bias")?;
    let sinc_abs: Vec<f32> = download(SegmentationTensor::SincConv)?
        .iter()
        .map(|value| value.abs())
        .collect();
    let filters = stream.clone_dtoh(model.sinc_filters())?;
    let layers = vec![
        (
            "sinc filters",
            max_abs_diff(
                &filters,
                &read(reference, "tensor//sincnet/conv1d.0/Concat_2_output_0")?,
            ),
        ),
        (
            "wav norm",
            max_abs_diff(
                &download(SegmentationTensor::WaveNorm)?,
                &read(
                    reference,
                    "tensor//sincnet/wav_norm1d/InstanceNormalization_output_0",
                )?,
            ),
        ),
        (
            "|sinc conv|",
            max_abs_diff(&sinc_abs, &read(reference, "tensor//sincnet/Abs_output_0")?),
        ),
        (
            "stage 0",
            max_abs_diff(
                &download(SegmentationTensor::Stage0)?,
                &read(reference, "tensor//sincnet/LeakyRelu_output_0")?,
            ),
        ),
        (
            "conv 1",
            max_abs_diff(
                &add_channel_bias(
                    &download(SegmentationTensor::Conv1)?,
                    &conv1_bias,
                    shape.conv1,
                ),
                &read(reference, "tensor//sincnet/conv1d.1/Conv_output_0")?,
            ),
        ),
        (
            "stage 1",
            max_abs_diff(
                &download(SegmentationTensor::Stage1)?,
                &read(reference, "tensor//sincnet/LeakyRelu_1_output_0")?,
            ),
        ),
        (
            "conv 2",
            max_abs_diff(
                &add_channel_bias(
                    &download(SegmentationTensor::Conv2)?,
                    &conv2_bias,
                    shape.conv2,
                ),
                &read(reference, "tensor//sincnet/conv1d.2/Conv_output_0")?,
            ),
        ),
        (
            "lstm input",
            max_abs_diff(
                &download(SegmentationTensor::LstmInput)?,
                &transpose_last(
                    &read(reference, "tensor//sincnet/LeakyRelu_2_output_0")?,
                    case.batch,
                    60,
                    shape.frames,
                ),
            ),
        ),
        (
            "lstm output",
            max_abs_diff(
                &download(SegmentationTensor::LstmOutput)?,
                &read(reference, "tensor//lstm/Transpose_5_output_0")?,
            ),
        ),
        (
            "linear 0",
            max_abs_diff(
                &download(SegmentationTensor::Linear0)?,
                &read(reference, "tensor//LeakyRelu_output_0")?,
            ),
        ),
        (
            "linear 1",
            max_abs_diff(
                &download(SegmentationTensor::Linear1)?,
                &read(reference, "tensor//LeakyRelu_1_output_0")?,
            ),
        ),
    ];

    Ok(Parity {
        max_error: max_abs_diff(&output, &expected),
        flips: argmax_flips(&output, &expected, shape.frames),
        layers,
    })
}

fn report(case: Case, options: SegmentationOptions, parity: &Parity) {
    eprintln!(
        "{} ({}, {:?}, lstm {:?}, graph {}): logits max abs error {:e}, argmax flips {} {:?}",
        case.name,
        case.model,
        options.math,
        options.lstm_algo,
        options.cuda_graph,
        parity.max_error,
        parity.flips.len(),
        parity.flips
    );
    for (layer, error) in &parity.layers {
        eprintln!("    {layer:>12}: max abs error {error:e}");
    }
}

/// Every reference case in both precisions with every LSTM algorithm, eager and
/// through a CUDA graph
#[test]
fn segmentation_matches_reference() -> Result<(), CudaError> {
    let test = "segmentation_matches_reference";
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let Some(dir) = reference_dir(test, "segmentation-3.0") else {
        return Ok(());
    };

    let algos = [
        CudaLstmAlgorithm::Standard,
        CudaLstmAlgorithm::PersistStaticSmallH,
        CudaLstmAlgorithm::PersistDynamic,
    ];
    for case in CASES {
        let loaded = case.load(&dir)?;
        for (math, tolerance) in [
            (CudaMath::Fp32, FP32_LOGIT_TOLERANCE),
            (CudaMath::Tf32, TF32_LOGIT_TOLERANCE),
        ] {
            for lstm_algo in algos {
                for cuda_graph in [false, true] {
                    let options = SegmentationOptions {
                        math,
                        lstm_algo,
                        cuda_graph,
                    };
                    // three runs: the graph is captured on the first and replayed after
                    let parity = run_case(&runtime, case, &loaded, options, 3)?;
                    report(case, options, &parity);
                    assert!(
                        parity.max_error <= tolerance,
                        "{}: {options:?} logits error {} above {tolerance}",
                        case.name,
                        parity.max_error
                    );
                    if math == CudaMath::Fp32 {
                        assert!(
                            parity.flips.is_empty(),
                            "{}: {options:?} argmax flips {:?}",
                            case.name,
                            parity.flips
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

/// The two exports carry the same weights under different generated names
#[test]
fn segmentation_exports_resolve_to_same_output() -> Result<(), CudaError> {
    let test = "segmentation_exports_resolve_to_same_output";
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let Some(dir) = reference_dir(test, "segmentation-3.0") else {
        return Ok(());
    };

    let case = CASES[0];
    let loaded = case.load(&dir)?;
    let input = read(&loaded.reference, "input/input")?;
    let b32_weights =
        SafetensorsFile::open(dir.join("segmentation-3.0-b32/segmentation-3.0-b32.safetensors"))?;

    let mut outputs = Vec::new();
    for weights in [&loaded.weights, &b32_weights] {
        let mut model = CudaSegmentation::new(&runtime, weights, FP32_EAGER)?;
        outputs.push(model.run(&runtime, 1, &input)?);
    }

    assert_eq!(outputs[0], outputs[1]);
    Ok(())
}

/// Too-short windows and empty batches are rejected; a 5 s window gives fewer frames
#[test]
fn segmentation_rejects_and_sizes_windows() -> Result<(), CudaError> {
    let test = "segmentation_rejects_and_sizes_windows";
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let Some(dir) = reference_dir(test, "segmentation-3.0") else {
        return Ok(());
    };

    let loaded = CASES[0].load(&dir)?;
    let mut model = CudaSegmentation::new(&runtime, &loaded.weights, FP32_EAGER)?;
    assert!(model.workspace(&runtime, 1, 300).is_err());
    assert!(model.workspace(&runtime, 0, WINDOW_SAMPLES).is_err());

    // the 10 s reference window split into two 5 s windows
    let input = read(&loaded.reference, "input/input")?;
    let output = model.run(&runtime, 2, &input)?;
    let shape = model.workspace(&runtime, 2, 80_000)?.shape();
    assert_eq!(shape.frames, 293);
    assert_eq!(output.len(), 2 * 293 * CLASSES);
    assert!(
        output
            .iter()
            .all(|value| value.is_finite() && *value <= 0.0)
    );

    let uneven = model.run(&runtime, 2, &input[..1001]);
    assert!(matches!(uneven, Err(CudaError::BufferLength { .. })));
    Ok(())
}

/// `(pid, MiB)` of every process using the GPU, from `nvidia-smi`
fn gpu_processes() -> Vec<(u32, u32)> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };

    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (pid, mib) = line.split_once(',')?;
            Some((pid.trim().parse().ok()?, mib.trim().parse().ok()?))
        })
        .collect()
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Median latency per batch, precision, LSTM algorithm and graph mode; copies are
/// timed apart from GPU compute. Hold `/workspace/gpu-bench.lock` while it runs
#[test]
#[ignore = "benchmark; run with the GPU lock held"]
fn segmentation_benchmark() -> Result<(), CudaError> {
    let test = "segmentation_benchmark";
    let Some(runtime) = runtime(test) else {
        return Ok(());
    };
    let Some(dir) = reference_dir(test, "segmentation-3.0") else {
        return Ok(());
    };

    let runs = std::env::var("SPEAKRS_BENCH_RUNS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30_usize);
    let warmup = 5;
    // `batch/math/algo/graph`, for example `32/Fp32/PersistStaticSmallH/true`; run one
    // configuration per process to read its GPU memory cleanly
    let only = std::env::var("SPEAKRS_BENCH_ONLY").ok();
    let algos = [
        CudaLstmAlgorithm::Standard,
        CudaLstmAlgorithm::PersistStaticSmallH,
        CudaLstmAlgorithm::PersistDynamic,
    ];
    eprintln!("other GPU processes: {:?}", gpu_processes());
    eprintln!(
        "| batch | math | lstm | graph | upload ms | compute ms | download ms | windows/s (compute) | logits err | flips | process GPU MiB | buffers MiB |"
    );
    eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|");

    for case in [CASES[0], CASES[3]] {
        let loaded = case.load(&dir)?;
        let input = read(&loaded.reference, "input/input")?;
        let expected = read(&loaded.reference, "tensor/output")?;
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            for lstm_algo in algos {
                for cuda_graph in [false, true] {
                    let key = format!("{}/{math:?}/{lstm_algo:?}/{cuda_graph}", case.batch);
                    if only.as_ref().is_some_and(|only| *only != key) {
                        continue;
                    }

                    let options = SegmentationOptions {
                        math,
                        lstm_algo,
                        cuda_graph,
                    };
                    let row = bench_one(
                        &runtime, case, &loaded, options, &input, &expected, warmup, runs,
                    );
                    match row {
                        Ok(row) => eprintln!("{row}"),
                        Err(error) => eprintln!(
                            "| {} | {math:?} | {lstm_algo:?} | {cuda_graph} | unsupported: {error} |",
                            case.batch
                        ),
                    }
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench_one(
    runtime: &CudaRuntime,
    case: Case,
    loaded: &Loaded,
    options: SegmentationOptions,
    input: &[f32],
    expected: &[f32],
    warmup: usize,
    runs: usize,
) -> Result<String, CudaError> {
    let mut model = CudaSegmentation::new(runtime, &loaded.weights, options)?;
    let (batch, samples) = (case.batch, WINDOW_SAMPLES);

    let mut output = Vec::new();
    for _ in 0..warmup {
        output = model.run(runtime, batch, input)?;
    }
    runtime.synchronize()?;
    let process_mib = gpu_processes()
        .into_iter()
        .find(|(pid, _)| *pid == std::process::id())
        .map_or(-1, |(_, mib)| i64::from(mib));

    let mut upload = Vec::with_capacity(runs);
    let mut compute = Vec::with_capacity(runs);
    let mut download = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        model
            .workspace(runtime, batch, samples)?
            .upload_input(runtime, input)?;
        runtime.synchronize()?;
        let uploaded = Instant::now();
        model.forward(runtime, batch, samples)?;
        runtime.synchronize()?;
        let computed = Instant::now();
        output = model
            .workspace(runtime, batch, samples)?
            .download_output(runtime)?;
        let downloaded = Instant::now();
        upload.push(uploaded - start);
        compute.push(computed - uploaded);
        download.push(downloaded - computed);
    }

    let workspace = model.workspace(runtime, batch, samples)?;
    let frames = workspace.shape().frames;
    let compute = median(compute);
    Ok(format!(
        "| {} | {:?} | {:?} | {} | {:.3} | {:.3} | {:.3} | {:.0} | {:.2e} | {} | {} | {:.0} |",
        case.batch,
        options.math,
        options.lstm_algo,
        options.cuda_graph,
        millis(median(upload)),
        millis(compute),
        millis(median(download)),
        case.batch as f64 / compute.as_secs_f64(),
        max_abs_diff(&output, expected),
        argmax_flips(&output, expected, frames).len(),
        process_mib,
        (workspace.activation_bytes() + workspace.lstm_workspace_bytes()) as f64
            / (1024.0 * 1024.0),
    ))
}
