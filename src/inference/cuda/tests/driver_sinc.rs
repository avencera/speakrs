//! Development check of the driver-only Sinc producer: the PR #36 fused convolution,
//! `abs` and max pool against cuDNN's convolution pooled on the host and a sampled f64
//! truth, on real normalized waveforms and filters, at the model and stress batches in
//! both math modes
//!
//! Run on a GPU box with the segmentation references under the GPU lock:
//! `cargo test --release --features cuda --lib driver_sinc -- --ignored --nocapture`.
//! Each case prints one `SINC` JSON line; `TRUNK_BATCHES` and `TRUNK_MATHS` filter
//! cases and `TRUNK_TIMING=1` adds graph-timed medians at batches 1 and 32

use std::sync::Arc;

use cudarc::driver::{CudaGraph, CudaStream, sys};
use serde_json::json;

use super::super::candidate::{Phases, SincCandidate, SincInputs, SincOxide, SincPin, SincSpec};
use super::super::dnn::ConvPlanner;
use super::super::geometry::Conv2d;
use super::super::{CudaError, CudaMath, KernelModule, SafetensorsFile};
use super::{reference_dir, runtime};

const SAMPLES: usize = 160_000;
const CHANNELS: usize = 80;
const TAPS: usize = 251;
const STRIDE: usize = 10;
const SINC: usize = 15_975;
const POOLED: usize = 5_325;
const INPUT: &str = "tensor//sincnet/wav_norm1d/InstanceNormalization_output_0";
const FILTERS: &str = "tensor//sincnet/conv1d.0/Concat_2_output_0";

fn read(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, CudaError> {
    let shape = file.shape(name).expect(name).to_vec();
    file.read_f32(name, &shape)
}

fn capture(
    stream: &Arc<CudaStream>,
    mut enqueue: impl FnMut() -> Result<(), CudaError>,
) -> Result<CudaGraph, CudaError> {
    stream.synchronize()?;
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let enqueued = enqueue();
    let captured = stream.end_capture(sys::CUgraphInstantiate_flags(0));
    enqueued?;
    Ok(captured?.expect("nonempty graph"))
}

/// Median milliseconds per replay of 10 samples of 10 replays after 100 ms of warm-up
fn timed(graph: &CudaGraph, stream: &CudaStream) -> Result<f64, CudaError> {
    let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    loop {
        graph.launch()?;
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        if start.elapsed_ms(&end)? >= 100.0 {
            break;
        }
    }
    let mut samples = Vec::new();
    for _ in 0..10 {
        let start = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..10 {
            graph.launch()?;
        }
        let end = stream.record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        end.synchronize()?;
        samples.push(f64::from(start.elapsed_ms(&end)?) / 10.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok((samples[4] + samples[5]) / 2.0)
}

/// `max(|raw|)` over each window of three convolution steps
fn pool(raw: &[f32], batch: usize) -> Vec<f32> {
    (0..batch * CHANNELS)
        .flat_map(|row| {
            let raw = &raw[row * SINC..][..SINC];
            (0..POOLED).map(move |p| {
                raw[3 * p..3 * p + 3]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()))
            })
        })
        .collect()
}

/// The f64 pooled output at 4096 spread samples
fn truth(x: &[f32], filters: &[f32], batch: usize) -> Vec<(usize, f64)> {
    let len = batch * CHANNELS * POOLED;
    (0..4096)
        .map(|i| (i * 1_000_003 + 11) % len)
        .map(|index| {
            let item = index / (CHANNELS * POOLED);
            let channel = index / POOLED % CHANNELS;
            let p = index % POOLED;
            let wave = &x[item * SAMPLES..][..SAMPLES];
            let filter = &filters[channel * TAPS..][..TAPS];
            let value = (3 * p..3 * p + 3)
                .map(|step| {
                    let window = &wave[step * STRIDE..][..TAPS];
                    window
                        .iter()
                        .zip(filter)
                        .map(|(&x, &w)| f64::from(x) * f64::from(w))
                        .sum::<f64>()
                        .abs()
                })
                .fold(0.0, f64::max);
            (index, value)
        })
        .collect()
}

fn error(actual: &[f32], truth: &[(usize, f64)]) -> (f64, f64) {
    let (mut max, mut diff, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for &(i, expected) in truth {
        assert!(actual[i].is_finite(), "non-finite output at {i}");
        let delta = f64::from(actual[i]) - expected;
        max = max.max(delta.abs());
        diff += delta * delta;
        norm += expected * expected;
    }
    (max, (diff / norm.max(f64::MIN_POSITIVE)).sqrt())
}

#[test]
#[ignore = "development check; run on a GPU box with the references under the GPU lock"]
fn driver_sinc_matches_library() -> Result<(), CudaError> {
    let Some(runtime) = runtime("driver_sinc_matches_library") else {
        return Ok(());
    };
    let Some(root) = reference_dir("driver_sinc_matches_library", "segmentation-3.0") else {
        return Ok(());
    };
    // safety: this test uses one stream for every allocation, transfer, launch and replay
    unsafe { runtime.context().disable_event_tracking() };
    let stream = runtime.stream();
    let timing = std::env::var_os("TRUNK_TIMING").is_some();
    let kernels = runtime.load_module(runtime.embedded_exact_request(KernelModule::Sincnet)?)?;
    let batches = std::env::var("TRUNK_BATCHES").unwrap_or_else(|_| "1,7,32,33".into());
    let mut failures = Vec::new();
    for batch in batches
        .split(',')
        .map(|b| b.parse::<usize>().expect("TRUNK_BATCHES"))
    {
        let (model, case, items) = if batch == 1 {
            ("segmentation-3.0", "test_first_b1", 1)
        } else {
            ("segmentation-3.0-b32", "test_and_short_b32", 32)
        };
        let reference =
            SafetensorsFile::open(root.join(model).join(format!("{case}.safetensors")))?;
        let waves = read(&reference, INPUT)?;
        let x: Vec<f32> = (0..batch)
            .flat_map(|item| &waves[item % items * SAMPLES..][..SAMPLES])
            .copied()
            .collect();
        let filters = read(&reference, FILTERS)?;
        let xd = stream.clone_htod(&x)?;
        let fd = stream.clone_htod(&filters)?;
        let truth = truth(&x, &filters, batch);
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            if std::env::var("TRUNK_MATHS")
                .is_ok_and(|m| !m.split(',').any(|m| m == format!("{math:?}")))
            {
                continue;
            }
            let spec = SincSpec {
                batch,
                samples: SAMPLES,
                sinc: SINC,
                pooled: POOLED,
                math,
                filters: &fd,
            };
            let covered =
                SincOxide::coverage(kernels.tier()).covers("sincnet.conv0.abs_pool", batch, math);
            let candidate = SincOxide::plan(&runtime, &kernels, spec, SincPin::ConvAbsPool)
                .map_err(|error| CudaError::Unsupported {
                    context: "driver sinc plan",
                    reason: error.to_string(),
                })?;
            let mut out = stream.alloc_zeros::<f32>(batch * CHANNELS * POOLED)?;
            let mut enqueue = || {
                candidate.enqueue(
                    SincInputs {
                        waveform: &xd.as_view(),
                        filters: &fd.as_view(),
                    },
                    &mut out.as_view_mut(),
                    &Phases::new(),
                    stream,
                )
            };
            enqueue()?;
            let graph = capture(stream, &mut enqueue)?;
            let eager = stream.clone_dtoh(&out)?;
            graph.launch()?;
            let bitwise = eager == stream.clone_dtoh(&out)?;

            let conv = Conv2d {
                batch,
                in_channels: 1,
                out_channels: CHANNELS,
                input: [1, SAMPLES],
                kernel: [1, TAPS],
                padding: [0, 0],
                stride: [1, STRIDE],
                dilation: [1, 1],
                math,
            };
            let library = ConvPlanner::new(&runtime)?.plan(conv)?;
            let mut work = stream.alloc_zeros::<u8>(library.workspace_bytes().max(1))?;
            let mut raw = stream.alloc_zeros::<f32>(batch * CHANNELS * SINC)?;
            let mut enqueue_library = || {
                library.forward(
                    &mut work.as_view_mut(),
                    &xd.as_view(),
                    &fd.as_view(),
                    &mut raw.as_view_mut(),
                )
            };
            enqueue_library()?;
            let lib_graph = capture(stream, &mut enqueue_library)?;
            let lib_pooled = pool(&stream.clone_dtoh(&raw)?, batch);

            let kernel_error = error(&eager, &truth);
            let library_error = error(&lib_pooled, &truth);
            let floor = if math == CudaMath::Tf32 { 1e-4 } else { 1e-6 };
            let pass = bitwise && kernel_error.1 <= 10.0 * library_error.1 + floor;
            let (library_ms, kernel_ms) = if timing && [1, 32].contains(&batch) {
                (
                    Some(timed(&lib_graph, stream)?),
                    Some(timed(&graph, stream)?),
                )
            } else {
                (None, None)
            };
            eprintln!(
                "SINC {}",
                json!({
                    "batch": batch, "math": format!("{math:?}"),
                    "tier": kernels.tier().to_string(), "covered": covered,
                    "kernel_error": kernel_error, "library_error": library_error,
                    "bitwise": bitwise, "pass": pass,
                    "library_conv_ms": library_ms, "kernel_ms": kernel_ms,
                })
            );
            if !pass {
                failures.push(format!("b{batch} {math:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "gross mismatches: {failures:?}");
    Ok(())
}
