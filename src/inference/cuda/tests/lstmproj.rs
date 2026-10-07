//! Development check of the library-free LSTM stack (`lstmproj`) against the cuDNN
//! stack and an f64 reference, with a rough eager timing A/B
//!
//! Ignored by default; run on a GPU box with the lock held:
//! `flock /workspace/gpu-bench.lock cargo test --release --features cuda --lib
//! lstmproj_dev -- --ignored --nocapture`. `SPEAKRS_LSTMPROJ_BATCHES` (default
//! `1,7,32,33,64`) and `SPEAKRS_LSTMPROJ_MODES` (`fp32,tf32`) narrow the run, and
//! `SPEAKRS_CUDA_PTX_TIER` forces a lower `lstmproj` tier. Each case prints one
//! `LSTMPROJ_CASE` JSON line
//!
//! The f64 errors pool the first and last windows of the batch. The check fails only on
//! a gross mismatch: a candidate error above twice the worse cuDNN algorithm's

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUevent_flags;
use serde_json::json;

use super::super::candidate::{
    LstmCandidate, LstmLayerWeights, LstmPhases, LstmPin, LstmProjOxide, LstmProjection, LstmSpec,
    Projection,
};
use super::super::test_support::select_lstm;
use super::super::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime, CudaSegmentation, KernelModule,
    LoadedKernels, SafetensorsFile, SegmentationOptions,
};
use super::{reference_dir, runtime};

type DevResult<T> = Result<T, Box<dyn std::error::Error>>;

const STEPS: usize = 589;
const FEATURES: usize = 60;
const OUTPUT: usize = 256;
const SAMPLES: usize = 160_000;

fn env_list(name: &str, default: &str) -> Vec<String> {
    std::env::var(name)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(str::to_owned)
        .collect()
}

fn modes() -> Vec<CudaMath> {
    env_list("SPEAKRS_LSTMPROJ_MODES", "fp32,tf32")
        .iter()
        .map(|mode| match mode.as_str() {
            "fp32" => CudaMath::Fp32,
            "tf32" => CudaMath::Tf32,
            other => panic!("unknown mode {other}"),
        })
        .collect()
}

fn batches() -> Vec<usize> {
    env_list("SPEAKRS_LSTMPROJ_BATCHES", "1,7,32,33,64")
        .iter()
        .map(|batch| batch.parse().expect("batch"))
        .collect()
}

/// Host weights of one layer in ONNX layout
struct HostLayer {
    input: usize,
    w: Vec<f32>,
    r: Vec<f32>,
    b: Vec<f32>,
}

fn read(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, CudaError> {
    let shape = file.shape(name).expect("tensor").to_vec();
    file.read_f32(name, &shape)
}

/// The four layers in export order: generated `onnx::LSTM_*` names, numbered B, W, R
fn host_layers(file: &SafetensorsFile) -> Result<Vec<HostLayer>, CudaError> {
    let mut names: Vec<(u64, String)> = file
        .names()
        .into_iter()
        .filter_map(|name| {
            let number = name.strip_prefix("onnx::LSTM_")?.parse().ok()?;
            Some((number, name))
        })
        .collect();
    names.sort();
    assert_eq!(names.len(), 12, "three LSTM tensors per layer");
    (0..4)
        .map(|layer| {
            let [b, w, r] = [0, 1, 2].map(|k| names[3 * layer + k].1.clone());
            Ok(HostLayer {
                input: if layer == 0 { FEATURES } else { OUTPUT },
                w: read(file, &w)?,
                r: read(file, &r)?,
                b: read(file, &b)?,
            })
        })
        .collect()
}

/// Batch-major `[batch, 589, 60]` LSTM input from the reference windows; batches
/// above the fixture's repeat its windows in a shifted order
fn fixture_input(dir: &std::path::Path, batch: usize) -> Result<Vec<f32>, CudaError> {
    let file = if batch == 1 {
        SafetensorsFile::open(dir.join("segmentation-3.0/test_first_b1.safetensors"))?
    } else {
        SafetensorsFile::open(dir.join("segmentation-3.0-b32/test_and_short_b32.safetensors"))?
    };
    let channels_first = read(&file, "tensor//sincnet/LeakyRelu_2_output_0")?;
    let source_batch = channels_first.len() / (FEATURES * STEPS);
    let mut input = vec![0.0; batch * STEPS * FEATURES];
    for b in 0..batch {
        let source = (b + 17 * (b / source_batch)) % source_batch;
        for t in 0..STEPS {
            for c in 0..FEATURES {
                input[(b * STEPS + t) * FEATURES + c] =
                    channels_first[(source * FEATURES + c) * STEPS + t];
            }
        }
    }
    Ok(input)
}

/// The stack in f64 for one window: `x` is `[589, 60]`, the result `[589, 256]`
fn f64_stack(layers: &[HostLayer], x: &[f32]) -> Vec<f64> {
    const H: usize = 128;
    let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
    let mut input: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    for layer in layers {
        let width = layer.input;
        let mut output = vec![0.0f64; STEPS * 2 * H];
        for direction in 0..2 {
            let w = &layer.w[direction * 4 * H * width..(direction + 1) * 4 * H * width];
            let r = &layer.r[direction * 4 * H * H..(direction + 1) * 4 * H * H];
            let b = &layer.b[direction * 8 * H..(direction + 1) * 8 * H];
            let mut h = vec![0.0f64; H];
            let mut c = vec![0.0f64; H];
            for step in 0..STEPS {
                let t = if direction == 0 {
                    step
                } else {
                    STEPS - 1 - step
                };
                let xt = &input[t * width..(t + 1) * width];
                let mut gates = vec![0.0f64; 4 * H];
                for (row, gate) in gates.iter_mut().enumerate() {
                    let mut sum = f64::from(b[row]) + f64::from(b[4 * H + row]);
                    for (k, x) in xt.iter().enumerate() {
                        sum += f64::from(w[row * width + k]) * x;
                    }
                    for (k, h) in h.iter().enumerate() {
                        sum += f64::from(r[row * H + k]) * h;
                    }
                    *gate = sum;
                }
                // ONNX gate blocks [i, o, f, c]
                for u in 0..H {
                    let i = sigmoid(gates[u]);
                    let o = sigmoid(gates[H + u]);
                    let f = sigmoid(gates[2 * H + u]);
                    let g = gates[3 * H + u].tanh();
                    c[u] = f * c[u] + i * g;
                    h[u] = o * c[u].tanh();
                    output[t * 2 * H + direction * H + u] = h[u];
                }
            }
        }
        input = output;
    }
    input
}

/// Pooled relative L2 and max abs error of the truth rows against f64
fn truth_error(output: &[f32], rows: &[usize], truth: &[Vec<f64>]) -> (f64, f64) {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut max_abs = 0.0f64;
    for (row, expected) in rows.iter().zip(truth) {
        let got = &output[row * STEPS * OUTPUT..(row + 1) * STEPS * OUTPUT];
        for (&a, &e) in got.iter().zip(expected) {
            let d = f64::from(a) - e;
            numerator += d * d;
            denominator += e * e;
            max_abs = max_abs.max(d.abs());
        }
    }
    ((numerator / denominator).sqrt(), max_abs)
}

/// Relative L2 and max abs difference of two complete outputs
fn difference(actual: &[f32], expected: &[f32]) -> (f64, f64) {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let d = f64::from(a) - f64::from(e);
        numerator += d * d;
        denominator += f64::from(e) * f64::from(e);
        max_abs = max_abs.max(d.abs());
    }
    ((numerator / denominator).sqrt(), max_abs)
}

/// Median and spread in ms of 15 event-timed eager runs after 5 warm-ups
fn time(runtime: &CudaRuntime, mut run: impl FnMut() -> DevResult<()>) -> DevResult<(f32, f32)> {
    let flags = Some(CUevent_flags::CU_EVENT_DEFAULT);
    let mut samples = Vec::new();
    for i in 0..20 {
        let start = runtime.stream().record_event(flags)?;
        run()?;
        let end = runtime.stream().record_event(flags)?;
        end.synchronize()?;
        if i >= 5 {
            samples.push(start.elapsed_ms(&end)?);
        }
    }
    samples.sort_by(f32::total_cmp);
    Ok((
        samples[samples.len() / 2],
        samples[samples.len() - 1] - samples[0],
    ))
}

/// One implementation's output, timing and repeatability
struct Run {
    output: Vec<f32>,
    time: (f32, f32),
    bitwise: bool,
}

fn library(
    runtime: &CudaRuntime,
    weights: &SafetensorsFile,
    math: CudaMath,
    lstm_algo: CudaLstmAlgorithm,
    batch: usize,
    input: &[f32],
) -> DevResult<Run> {
    let options = SegmentationOptions {
        math,
        lstm_algo,
        cuda_graph: false,
    };
    let mut model = CudaSegmentation::new(runtime, weights, options)?;
    select_lstm(&mut model, runtime, [batch, SAMPLES], "Library")?;
    let mut op = model.isolated(runtime, batch, "lstm", [input, input])?;
    model.isolated_run(runtime, &mut op, 0)?;
    let output = model.isolated_output(runtime, &op)?;
    model.isolated_run(runtime, &mut op, 1)?;
    let bitwise = bits_equal(&output, &model.isolated_output(runtime, &op)?);
    let time = time(runtime, || Ok(model.isolated_run(runtime, &mut op, 0)?))?;
    Ok(Run {
        output,
        time,
        bitwise,
    })
}

fn spec(layers: &[HostLayer], batch: usize, math: CudaMath) -> LstmSpec<'_> {
    LstmSpec {
        batch,
        frames: STEPS,
        math,
        layers: [0, 1, 2, 3].map(|index| LstmLayerWeights {
            input: layers[index].input,
            w: &layers[index].w,
            r: &layers[index].r,
            b: &layers[index].b,
        }),
    }
}

fn candidate(
    runtime: &CudaRuntime,
    kernels: &LoadedKernels,
    spec: LstmSpec<'_>,
    pin: LstmPin,
    input: &[f32],
) -> DevResult<Run> {
    let (batch, math) = (spec.batch, spec.math);
    let plan = LstmProjOxide::plan(runtime, kernels, spec, pin)?;
    let stream = runtime.stream();
    let input = stream.clone_htod(input)?;
    let mut output = stream.alloc_zeros::<f32>(batch * STEPS * OUTPUT)?;
    // the library projection helper is never called by this candidate
    let phases = LstmPhases::new(Projection::new(runtime, batch * STEPS, math));
    let run = |output: &mut CudaSlice<f32>| -> DevResult<()> {
        plan.enqueue(&input.as_view(), &mut output.as_view_mut(), &phases, stream)?;
        Ok(())
    };
    run(&mut output)?;
    let first = stream.clone_dtoh(&output)?;
    run(&mut output)?;
    let bitwise = bits_equal(&first, &stream.clone_dtoh(&output)?);
    let time = time(runtime, || run(&mut output))?;
    Ok(Run {
        output: first,
        time,
        bitwise,
    })
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "GPU development check; hold /workspace/gpu-bench.lock"]
fn lstmproj_dev() {
    let Some(runtime) = runtime("lstmproj_dev") else {
        return;
    };
    let Some(dir) = reference_dir("lstmproj_dev", "segmentation-3.0") else {
        return;
    };
    // dropping cuDNN handles after a sticky device error panics and hides the error
    if let Err(error) = dev(&runtime, &dir) {
        eprintln!("FAILED: {error}");
        std::process::exit(1);
    }
}

fn dev(runtime: &CudaRuntime, dir: &std::path::Path) -> DevResult<()> {
    let weights = SafetensorsFile::open(dir.join("segmentation-3.0/segmentation-3.0.safetensors"))?;
    let layers = host_layers(&weights)?;
    let kernels = runtime.load_module(runtime.embedded_exact_request(KernelModule::LstmProj)?)?;
    let device = runtime.device();
    println!(
        "device={} cc={} sms={} lstmproj tier={} artifact={:?}",
        device.name(),
        device.capability(),
        device.multiprocessors(),
        kernels.tier(),
        kernels.artifact()
    );

    for math in modes() {
        for batch in batches() {
            let input = fixture_input(dir, batch)?;
            let rows = [0, batch - 1];
            let truth: Vec<Vec<f64>> = rows
                .iter()
                .map(|&row| {
                    f64_stack(
                        &layers,
                        &input[row * STEPS * FEATURES..][..STEPS * FEATURES],
                    )
                })
                .collect();

            let spec = spec(&layers, batch, math);
            let pin = LstmProjOxide::device_pin(device, kernels.tier(), &spec)?;
            let standard = library(
                runtime,
                &weights,
                math,
                CudaLstmAlgorithm::Standard,
                batch,
                &input,
            )?;
            let smallh = library(
                runtime,
                &weights,
                math,
                CudaLstmAlgorithm::PersistStaticSmallH,
                batch,
                &input,
            )?;
            let oxide = candidate(runtime, &kernels, spec, pin, &input)?;

            let standard_truth = truth_error(&standard.output, &rows, &truth);
            let smallh_truth = truth_error(&smallh.output, &rows, &truth);
            let oxide_truth = truth_error(&oxide.output, &rows, &truth);
            let vs_standard = difference(&oxide.output, &standard.output);

            // the tensor projection the rule avoids or uses, for the accuracy evidence
            let tensor = LstmPin::Projected(LstmProjection::Tensor);
            let alternative = if math == CudaMath::Tf32 && pin != tensor {
                candidate(runtime, &kernels, spec, tensor, &input)
                    .ok()
                    .map(|run| (truth_error(&run.output, &rows, &truth), run.time.0))
            } else {
                None
            };

            let case = json!({
                "device": device.name(),
                "cc": device.capability().to_string(),
                "tier": kernels.tier().to_string(),
                "math": format!("{math:?}"),
                "batch": batch,
                "pin": format!("{pin:?}"),
                "f64_rows": rows,
                "standard": {"l2": standard_truth.0, "abs": standard_truth.1, "ms": standard.time.0, "spread": standard.time.1, "bitwise": standard.bitwise},
                "smallh": {"l2": smallh_truth.0, "abs": smallh_truth.1, "ms": smallh.time.0, "spread": smallh.time.1, "bitwise": smallh.bitwise},
                "oxide": {"l2": oxide_truth.0, "abs": oxide_truth.1, "ms": oxide.time.0, "spread": oxide.time.1, "bitwise": oxide.bitwise},
                "oxide_vs_standard": {"l2": vs_standard.0, "abs": vs_standard.1},
                "tensor_alternative": alternative.map(|((l2, abs), ms)| json!({"l2": l2, "abs": abs, "ms": ms})),
            });
            println!("LSTMPROJ_CASE {case}");
            println!(
                "{math:?} b{batch} {pin:?}: f64 l2 standard={:.3e} smallh={:.3e} oxide={:.3e} | abs {:.3e} {:.3e} {:.3e} | ms {:.3} {:.3} {:.3} | vs standard l2={:.3e} abs={:.3e} | bitwise {}",
                standard_truth.0,
                smallh_truth.0,
                oxide_truth.0,
                standard_truth.1,
                smallh_truth.1,
                oxide_truth.1,
                standard.time.0,
                smallh.time.0,
                oxide.time.0,
                vs_standard.0,
                vs_standard.1,
                oxide.bitwise,
            );

            let worse_l2 = standard_truth.0.max(smallh_truth.0);
            let worse_abs = standard_truth.1.max(smallh_truth.1);
            assert!(oxide.bitwise, "{math:?} b{batch}: repeated passes differ");
            assert!(
                oxide_truth.0 <= 2.0 * worse_l2 && oxide_truth.1 <= 2.0 * worse_abs,
                "{math:?} b{batch}: gross mismatch against f64"
            );
        }
    }
    Ok(())
}
