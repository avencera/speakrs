//! Live qualification entry point, compiled only in the library test binary
//!
//! One process measures one phase. Numeric and timing processes measure one math mode
//! and case each, so the shared GPU lock is held briefly; profile and sanitizer
//! processes cover every case they need. Every timed sample is a burst of CUDA graph
//! replays inside one event pair, and every timing process reports the hash of its
//! final output so a no-op or cached result cannot pass

use super::{
    graph_evidence, graph_violations, library_call_violations, loaded_modules, prepare,
    registered_side_streams, sanitizer_control, secret_seed, select_conv, select_lstm, select_sinc,
    set_band, set_label, window,
};
use crate::inference::cuda::candidate::{
    Batches, ConvCandidate, ConvOxide, Coverage, CoverageEntry, Maths, Projection,
};
use crate::inference::cuda::embedding::test_support::{HostInputs, Operator};
use crate::inference::cuda::weights::uniform;
use crate::inference::cuda::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime, CudaSegmentation, KernelModule,
    ResNetEmbedding, SafetensorsFile, SegmentationOptions,
};
use cudarc::driver::CudaGraph;
use cudarc::driver::sys::{CUevent_flags, CUgraphInstantiate_flags, CUstreamCaptureMode};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "paired.rs"]
mod paired;
#[path = "reference.rs"]
pub(crate) mod reference;
#[path = "truth.rs"]
mod truth;

const CASES: [(&str, usize); 8] = [
    ("first", 1),
    ("last", 1),
    ("short", 1),
    ("mixed", 7),
    ("mixed", 32),
    ("mixed", 33),
    ("mixed", 64),
    ("short", 7),
];
/// Batch sizes the harness can qualify; a coverage declaration may only name these
const BATCHES: [usize; 5] = [1, 7, 32, 33, 64];
/// Seeds of the TF32 stage noise band
const BAND_SEEDS: [u32; 8] = [11, 23, 37, 41, 53, 67, 79, 97];
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
/// Each timed sample is a burst of at least this many milliseconds
const SAMPLE_MS: f32 = 20.0;
const WINDOW: usize = 160_000;

fn source_row(row: usize, source_batch: usize) -> usize {
    // the second batch tile must not repeat the first tile at the same offset
    (row + 17 * (row / source_batch)) % source_batch
}

pub(crate) fn read_batch(
    file: &SafetensorsFile,
    name: &str,
    batch: usize,
) -> Result<Vec<f32>, CudaError> {
    let values = file.read_f32(name, file.shape(name).expect("fixed reference tensor"))?;
    let source_batch = file
        .shape("input/fbank")
        .or_else(|| file.shape("input/input"))
        .expect("input shape")[0];
    assert_eq!(values.len() % source_batch, 0);
    let width = values.len() / source_batch;
    Ok((0..batch)
        .flat_map(|row| {
            let source = source_row(row, source_batch);
            values[source * width..(source + 1) * width].iter().copied()
        })
        .collect())
}

fn reference(target: &str, case: &str) -> Result<SafetensorsFile, CudaError> {
    let model = if target == "resnet" {
        "wespeaker-multimask-tail"
    } else {
        "segmentation-3.0"
    };
    let (suffix, name) = match case {
        "first" => ("", "test_first_b1"),
        "last" => ("", "test_last_partial_b1"),
        "short" => ("", "test_short_partial_b1"),
        "mixed" => ("-b32", "test_and_short_b32"),
        _ => panic!("fixed case"),
    };
    SafetensorsFile::open(format!("/workspace/ref/{model}{suffix}/{name}.safetensors"))
}

/// SHA-256 of the little-endian FP32 bytes
fn sha(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn metrics(actual: &[f32], expected: &[f32], width: usize) -> Value {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&a, &r) in actual.iter().zip(expected) {
        let (x, y) = (f64::from(a), f64::from(r));
        numerator += (x - y).powi(2);
        denominator += y.powi(2);
        max_abs = max_abs.max((x - y).abs());
    }
    assert!(denominator > 0.0);
    let mut cosine = 1.0f64;
    let mut cosine_sum = 0.0f64;
    let mut flips = 0;
    let mut rows = 0;
    if width > 1 {
        for (a, r) in actual.chunks_exact(width).zip(expected.chunks_exact(width)) {
            let dot = a
                .iter()
                .zip(r)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum::<f64>();
            let norm = a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt()
                * r.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let row = if norm > 0.0 { dot / norm } else { -1.0 };
            cosine = cosine.min(row);
            cosine_sum += row;
            rows += 1;
            let argmax = |v: &[f32]| {
                v.iter()
                    .enumerate()
                    .fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
            };
            flips += usize::from(argmax(a) != argmax(r));
        }
    }
    let mean_cosine = if rows > 0 {
        cosine_sum / rows as f64
    } else {
        1.0
    };
    json!({"relative_l2":(numerator/denominator).sqrt(),"max_abs":max_abs,"minimum_cosine":cosine,"mean_cosine":mean_cosine,"argmax_flips":flips,"sha256":sha(actual),"elements":actual.len()})
}

#[allow(clippy::too_many_arguments)]
fn record(
    rows: &mut Vec<Value>,
    id: &str,
    first: Vec<f32>,
    second: Vec<f32>,
    expected: &[f32],
    width: usize,
    declared: bool,
) {
    let a = metrics(&first, expected, width);
    let b = metrics(&second, expected, width);
    rows.push(json!({"id":id,"first":a,"second":b,"bitwise_equal":first.iter().zip(&second).all(|(x,y)| x.to_bits()==y.to_bits()),"declared":declared}));
}

/// Captures `enqueue` on the runtime's stream as a graph the driver replays itself
fn capture(
    runtime: &CudaRuntime,
    enqueue: impl FnOnce() -> Result<(), CudaError>,
) -> Result<CudaGraph, CudaError> {
    let stream = runtime.stream();
    stream.synchronize()?;
    let context = runtime.context();
    let tracking = context.is_event_tracking();
    // SAFETY: every buffer lives on this one stream, so stream order alone keeps them
    // consistent while per-buffer event tracking is off during the capture
    unsafe { context.disable_event_tracking() };
    let captured = stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(CudaError::from)
        .and_then(|()| {
            let enqueued = enqueue();
            let graph = stream.end_capture(CUgraphInstantiate_flags(0));
            enqueued?;
            Ok(graph?)
        });
    if tracking {
        // SAFETY: restores the tracking state from before the capture
        unsafe { context.enable_event_tracking() };
    }
    Ok(captured?.expect("a qualification operator enqueues at least one node"))
}

/// Timed bursts of launches, with per-launch milliseconds
struct Timing {
    per_launch_ms: Vec<f32>,
    warmup_ms: Vec<f32>,
    burst: usize,
    launches: usize,
}

fn median(values: &[f32]) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted[sorted.len() / 2]
}

/// Times [`SAMPLES`] bursts, each long enough to take at least [`SAMPLE_MS`]
///
/// `prepare(sample)` runs before each burst, outside the timed interval; `launch(i)`
/// gets the global launch index, so operators alternate input sets launch by launch
fn bursts(
    runtime: &CudaRuntime,
    mut prepare: impl FnMut(usize) -> Result<(), CudaError>,
    mut launch: impl FnMut(usize) -> Result<(), CudaError>,
) -> Result<Timing, CudaError> {
    let stream = runtime.stream();
    let event = || stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT));
    let mut index = 0;
    prepare(0)?;
    let mut warmup_ms = Vec::new();
    for _ in 0..WARMUP {
        let start = event()?;
        launch(index)?;
        index += 1;
        let end = event()?;
        end.synchronize()?;
        warmup_ms.push(start.elapsed_ms(&end)?);
    }

    let single = median(&warmup_ms).max(1e-3);
    let burst = ((SAMPLE_MS / single).ceil() as usize).clamp(1, 4096);
    let mut per_launch_ms = Vec::new();
    for sample in 0..SAMPLES {
        prepare(sample)?;
        let start = event()?;
        for _ in 0..burst {
            launch(index)?;
            index += 1;
        }
        let end = event()?;
        end.synchronize()?;
        per_launch_ms.push(start.elapsed_ms(&end)? / burst as f32);
    }

    Ok(Timing {
        per_launch_ms,
        warmup_ms,
        burst,
        launches: index,
    })
}

/// One timed case, with the output hash of one replay on each input set after the
/// bursts, so neither graph can be a no-op or answer from a cache
fn timing_row(id: &str, timing: &Timing, outputs: [&[f32]; 2], declared: bool) -> Value {
    json!({"id":id,"milliseconds":timing.per_launch_ms,"warmup":WARMUP,"warmup_ms":timing.warmup_ms,"burst":timing.burst,"launches":timing.launches,"output_sha256":[sha(outputs[0]), sha(outputs[1])],"cuda_graph":true,"declared":declared})
}

fn math_name(math: CudaMath) -> &'static str {
    match math {
        CudaMath::Fp32 => "fp32",
        _ => "tf32",
    }
}

pub(super) fn entry_json(entry: &CoverageEntry) -> Value {
    let batches = match entry.batches {
        Batches::All => json!("all"),
        Batches::Only(batches) => json!(batches),
    };
    let maths = match entry.maths {
        Maths::All => json!("all"),
        Maths::Only(maths) => json!(
            maths
                .iter()
                .map(|math| math_name(*math))
                .collect::<Vec<_>>()
        ),
    };
    json!({"layers": entry.layers, "batches": batches, "maths": maths})
}

pub(crate) fn coverage_json(coverage: Coverage) -> Value {
    json!({"entries": coverage.entries().iter().map(entry_json).collect::<Vec<_>>()})
}

/// The coverage the selected implementation declares: everything for planted faults,
/// nothing for Library
fn declared_coverage(
    target: &str,
    implementation: &str,
    tier: crate::inference::cuda::PtxTier,
) -> Value {
    if implementation == "Library" {
        return coverage_json(Coverage::NONE);
    }
    if implementation != "Oxide" {
        return json!({"entries": [{"layers": "all", "batches": "all", "maths": "all"}]});
    }
    if target == "resnet" {
        return coverage_json(ConvOxide::coverage(tier));
    }
    coverage_json(CudaSegmentation::coverage(target, tier))
}

fn lstm_diagnostics(
    model: &CudaSegmentation,
    runtime: &CudaRuntime,
    file: &SafetensorsFile,
    batch: usize,
    stack: &[f32],
) -> Result<Vec<Value>, CudaError> {
    let taps = model.diagnostic_lstm_outputs(runtime, batch);
    let mut rows = Vec::new();
    for layer in 0..4 {
        let name = if layer == 0 {
            "tensor//lstm/LSTM_output_0".to_owned()
        } else {
            format!("tensor//lstm/LSTM_{layer}_output_0")
        };
        let actual = if layer == 3 {
            Some(stack)
        } else {
            taps.iter()
                .find(|(index, _)| *index == layer)
                .map(|(_, values)| values.as_slice())
        };
        let Some(actual) = actual else {
            rows.push(json!({"layer":layer,"reference":name,"gate":false,"available":false,"reason":"implementation provides no intermediate tap; Library is monolithic"}));
            continue;
        };
        let shape = file.shape(&name).expect("per-layer ONNX reference");
        assert_eq!(shape[0], 589);
        assert_eq!(shape[1], 2);
        assert_eq!(shape[3], 128);
        let raw = file.read_f32(&name, shape)?;
        let mut expected = vec![0.0; batch * 589 * 256];
        for b in 0..batch {
            for t in 0..589 {
                for direction in 0..2 {
                    for h in 0..128 {
                        expected[(b * 589 + t) * 256 + direction * 128 + h] = raw
                            [((t * 2 + direction) * shape[2] + source_row(b, shape[2])) * 128 + h];
                    }
                }
            }
        }
        if actual.len() != expected.len() || !actual.iter().all(|x| x.is_finite()) {
            rows.push(json!({"layer":layer,"reference":name,"gate":false,"available":false,"reason":"invalid diagnostic output"}));
            continue;
        }
        rows.push(json!({"layer":layer,"reference":name,"gate":false,"available":true,"error":metrics(actual,&expected,256)}));
    }
    Ok(rows)
}

struct Run<'a> {
    runtime: &'a CudaRuntime,
    files: [&'a SafetensorsFile; 2],
    batch: usize,
    target: &'a str,
    choice: &'a str,
    phase: &'a str,
    id: String,
    math: CudaMath,
    /// layers whose Library outputs get 1-ulp noise in the TF32 stage band
    band: Option<Vec<String>>,
}

impl Run<'_> {
    fn key(&self, layer: &str, which: usize) -> String {
        let switched = if which == 1 { "/switched" } else { "" };
        format!("{}/{layer}{switched}", self.id)
    }

    fn embedding(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let runtime = self.runtime;
        let batch = self.batch;
        let weights =
            SafetensorsFile::open("/workspace/models-native/wespeaker-multimask-tail.safetensors")?;
        let mut model = ResNetEmbedding::load(runtime, &weights, self.math)?;
        for (stage, count) in [(1, 3), (2, 4)] {
            for block in 0..count {
                for conv in [1, 2] {
                    assert!(select_conv(
                        &mut model,
                        &format!("resnet.layer{stage}.{block}.conv{conv}"),
                        self.choice
                    ));
                }
            }
        }

        let mut any_declared = false;
        for block in 0..7 {
            for second in [false, true] {
                if self.phase == "sanitize" {
                    let stage = if block < 3 { 1 } else { 2 };
                    let index = if block < 3 { block } else { block - 3 };
                    let conv = if second { 2 } else { 1 };
                    let name = format!("resnet.layer{stage}.{index}.conv{conv}");
                    if std::env::var("SPEAKRS_QUALIFY_SANITIZER_LAYER")
                        .is_ok_and(|selected| selected != name)
                    {
                        continue;
                    }
                }
                let mut op = Operator::new(&model, runtime, self.files, batch, block, second)?;
                let declared = op.declared();
                any_declared |= declared;
                let name = op.name().to_owned();
                self.operator(rows, &name, declared, &mut OperatorRun::Conv(&mut op))?;
            }
        }

        if self.phase == "sanitize" {
            return Ok(());
        }

        let mut buffers = model.batch(runtime, batch)?;
        let inputs = self.files.map(|file| -> Result<_, CudaError> {
            Ok((
                read_batch(file, "input/fbank", batch)?,
                read_batch(file, "input/masks", batch)?,
                read_batch(file, "tensor/output", batch)?,
            ))
        });
        let [first_input, second_input] = inputs;
        let inputs = [first_input?, second_input?];
        let upload = |buffers: &mut crate::inference::cuda::EmbeddingBatch,
                      which: usize|
         -> Result<(), CudaError> {
            let (fbank, masks, _) = &inputs[which];
            buffers
                .fbank_mut()
                .copy_from_host(runtime.stream(), fbank)?;
            buffers.masks_mut().copy_from_host(runtime.stream(), masks)
        };

        if self.phase == "profile" {
            for which in 0..2 {
                upload(&mut buffers, which)?;
                let key = self.key("stage", which);
                {
                    let _window = window(&key);
                    buffers.forward(runtime)?;
                }
                buffers.download_output(runtime)?;
            }
            return Ok(());
        }

        upload(&mut buffers, 0)?;
        set_label(Some(self.key("stage", 0)));
        let captured = buffers.capture_graph(runtime);
        set_label(None);
        captured?;
        assert!(
            buffers.has_graph(),
            "the locked driver times a captured stage"
        );
        if self.phase == "timing" {
            let timing = {
                let cell = std::cell::RefCell::new(&mut buffers);
                bursts(
                    runtime,
                    |sample| upload(&mut cell.borrow_mut(), sample % 2),
                    |_| {
                        let buffers = cell.borrow();
                        let graph = buffers.graph().expect("captured stage graph");
                        Ok(graph.launch()?)
                    },
                )?
            };
            let mut outputs = Vec::new();
            for which in 0..2 {
                upload(&mut buffers, which)?;
                buffers.graph().expect("captured stage graph").launch()?;
                outputs.push(buffers.download_output(runtime)?);
            }
            rows.push(timing_row(
                &format!("{}/stage", self.id),
                &timing,
                [&outputs[0], &outputs[1]],
                any_declared,
            ));
            return Ok(());
        }

        let truth = (self.math == CudaMath::Tf32)
            .then(|| self.truth())
            .transpose()?;
        for (which, (_, _, reference)) in inputs.iter().enumerate() {
            upload(&mut buffers, which)?;
            let graph = buffers.graph().expect("captured stage graph");
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                buffers.qualification_round_stage(runtime)?;
            }
            let first = buffers.download_output(runtime)?;
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                buffers.qualification_round_stage(runtime)?;
            }
            let second = buffers.download_output(runtime)?;
            let truth_metrics = truth
                .as_ref()
                .map(|outputs| metrics(&first, &outputs[which], 256));
            let key = self.key("stage", which);
            record(rows, &key, first, second, reference, 256, any_declared);
            rows.last_mut().expect("stage row")["truth"] = json!(truth_metrics);
            rows.last_mut().expect("stage row")["truth_sha256"] =
                json!(truth.as_ref().map(|outputs| sha(&outputs[which])));
            let Some(layers) = &self.band else {
                continue;
            };
            let mut band = Vec::new();
            let mut truth_draws = Vec::new();
            for seed in BAND_SEEDS {
                set_band(Some((seed, layers.clone())));
                upload(&mut buffers, which)?;
                buffers.forward_with_taps(runtime, &mut |_, _| Ok(()))?;
                let output = buffers.download_output(runtime)?;
                set_band(None);
                band.push(metrics(&output, reference, 256));
                truth_draws.push(metrics(
                    &output,
                    &truth.as_ref().expect("TF32 truth")[which],
                    256,
                ));
            }
            rows.push(json!({"id": format!("{key}/band"), "seeds": BAND_SEEDS, "layers": layers, "metrics": band, "truth_draws": truth_draws, "truth_sha256": sha(&truth.as_ref().expect("TF32 truth")[which])}));
        }
        Ok(())
    }

    /// Measures one isolated boundary in the current phase
    fn operator(
        &self,
        rows: &mut Vec<Value>,
        name: &str,
        declared: bool,
        op: &mut OperatorRun<'_, '_>,
    ) -> Result<(), CudaError> {
        let runtime = self.runtime;
        match self.phase {
            "numeric" => {
                for which in 0..2 {
                    op.restore(runtime, which)?;
                    op.run(runtime, which)?;
                    let first = op.output(runtime)?;
                    op.restore(runtime, which)?;
                    op.run(runtime, which)?;
                    let second = op.output(runtime)?;
                    let diagnostics =
                        op.diagnostics(runtime, self.files[which], self.batch, &first)?;
                    let expected = op.reference(which);
                    record(
                        rows,
                        &self.key(name, which),
                        first,
                        second,
                        &expected,
                        1,
                        declared,
                    );
                    if let Some(diagnostics) = diagnostics {
                        rows.last_mut().expect("recorded layer")["diagnostics"] =
                            json!(diagnostics);
                    }
                }
            }
            "timing" => {
                op.restore(runtime, 0)?;
                op.restore(runtime, 1)?;
                set_label(Some(self.key(name, 0)));
                let first = capture(runtime, || op.run(runtime, 0));
                set_label(Some(self.key(name, 1)));
                let second = capture(runtime, || op.run(runtime, 1));
                set_label(None);
                let graphs = [first?, second?];
                let timing = bursts(runtime, |_| Ok(()), |index| Ok(graphs[index % 2].launch()?))?;
                graphs[0].launch()?;
                let first = op.output(runtime)?;
                graphs[1].launch()?;
                let second = op.output(runtime)?;
                rows.push(timing_row(
                    &self.key(name, 0),
                    &timing,
                    [&first, &second],
                    declared,
                ));
            }
            "profile" => {
                for which in 0..2 {
                    op.restore(runtime, which)?;
                    {
                        let _window = window(&self.key(name, which));
                        op.run(runtime, which)?;
                    }
                    op.output(runtime)?;
                }
            }
            "sanitize" => {
                for which in 0..2 {
                    op.restore(runtime, which)?;
                    op.run(runtime, which)?;
                }
                runtime.synchronize()?;
                rows.push(json!({"id":self.key(name, 0),"sanitized":true,"declared":declared}));
                println!("sanitized {}", self.key(name, 0));
            }
            other => panic!("operator phase {other}"),
        }
        Ok(())
    }

    fn segmentation(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let runtime = self.runtime;
        let batch = self.batch;
        let target = self.target;
        let weights =
            SafetensorsFile::open("/workspace/models-native/segmentation-3.0.safetensors")?;
        let options = SegmentationOptions {
            math: self.math,
            lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
            cuda_graph: self.phase != "profile",
        };
        let mut model = CudaSegmentation::new(runtime, &weights, options)?;
        if target == "lstm" {
            select_lstm(&mut model, runtime, [batch, WINDOW], self.choice)?;
        } else {
            select_sinc(&mut model, runtime, [batch, WINDOW], self.choice)?;
        }
        let isolated_inputs = |file: &SafetensorsFile| -> Result<_, CudaError> {
            Ok(if target == "lstm" {
                let cf = read_batch(file, "tensor//sincnet/LeakyRelu_2_output_0", batch)?;
                let mut input = vec![0.0; cf.len()];
                for b in 0..batch {
                    for t in 0..589 {
                        for c in 0..60 {
                            input[(b * 589 + t) * 60 + c] = cf[(b * 60 + c) * 589 + t];
                        }
                    }
                }
                (
                    input,
                    read_batch(file, "tensor//lstm/Transpose_5_output_0", batch)?,
                )
            } else {
                let input = read_batch(
                    file,
                    "tensor//sincnet/wav_norm1d/InstanceNormalization_output_0",
                    batch,
                )?;
                let abs = read_batch(file, "tensor//sincnet/Abs_output_0", batch)?;
                let pooled = abs
                    .as_chunks::<15975>()
                    .0
                    .iter()
                    .flat_map(|row| {
                        row.as_chunks::<3>()
                            .0
                            .iter()
                            .map(|x| x[0].max(x[1]).max(x[2]))
                    })
                    .collect::<Vec<_>>();
                (input, pooled)
            })
        };
        let (first_input, first_expected) = isolated_inputs(self.files[0])?;
        let (second_input, second_expected) = isolated_inputs(self.files[1])?;
        let mut isolated = model.isolated(
            runtime,
            batch,
            target,
            [first_input.as_slice(), second_input.as_slice()],
        )?;
        let declared = model.isolated_declared(batch, target);
        let name = if target == "lstm" {
            "lstm.stack"
        } else {
            "sincnet.conv0.abs_pool"
        };
        {
            let mut op = OperatorRun::Segmentation {
                model: &mut model,
                op: &mut isolated,
                references: [first_expected, second_expected],
            };
            self.operator(rows, name, declared, &mut op)?;
        }

        if self.phase == "sanitize" {
            return Ok(());
        }

        let inputs = [
            read_batch(self.files[0], "input/input", batch)?,
            read_batch(self.files[1], "input/input", batch)?,
        ];
        let references = [
            read_batch(self.files[0], "tensor/output", batch)?,
            read_batch(self.files[1], "tensor/output", batch)?,
        ];
        let download = |model: &CudaSegmentation| {
            model
                .find_workspace(batch, WINDOW)
                .expect("workspace")
                .download_output(runtime)
        };
        if self.phase == "profile" {
            for (which, input) in inputs.iter().enumerate() {
                model
                    .workspace(runtime, batch, WINDOW)?
                    .upload_input(runtime, input)?;
                {
                    let _window = window(&self.key("stage", which));
                    model.forward_eager(runtime, batch, WINDOW)?;
                }
                download(&model)?;
            }
            return Ok(());
        }

        // the first forward captures the production graph
        model
            .workspace(runtime, batch, WINDOW)?
            .upload_input(runtime, &inputs[0])?;
        set_label(Some(self.key("stage", 0)));
        let captured = model.forward(runtime, batch, WINDOW);
        set_label(None);
        captured?;
        assert!(
            model.qualification_has_graph(batch),
            "the locked driver times a captured stage"
        );
        if self.phase == "timing" {
            let timing = {
                let cell = std::cell::RefCell::new(&mut model);
                bursts(
                    runtime,
                    |sample| {
                        cell.borrow_mut()
                            .workspace(runtime, batch, WINDOW)?
                            .upload_input(runtime, &inputs[sample % 2])
                    },
                    |_| {
                        let model = cell.borrow();
                        let graph = model
                            .qualification_graph(batch)
                            .expect("captured stage graph");
                        Ok(graph.launch()?)
                    },
                )?
            };
            let mut outputs = Vec::new();
            for input in &inputs {
                model
                    .workspace(runtime, batch, WINDOW)?
                    .upload_input(runtime, input)?;
                let graph = model
                    .qualification_graph(batch)
                    .expect("captured stage graph");
                graph.launch()?;
                outputs.push(download(&model)?);
            }
            rows.push(timing_row(
                &format!("{}/stage", self.id),
                &timing,
                [&outputs[0], &outputs[1]],
                declared,
            ));
            return Ok(());
        }

        let truth = (self.math == CudaMath::Tf32)
            .then(|| self.truth())
            .transpose()?;
        for (which, input) in inputs.iter().enumerate() {
            model
                .workspace(runtime, batch, WINDOW)?
                .upload_input(runtime, input)?;
            let graph = model
                .qualification_graph(batch)
                .expect("captured stage graph");
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                model.qualification_round_stage(runtime, batch)?;
            }
            let first = download(&model)?;
            let graph = model
                .qualification_graph(batch)
                .expect("captured stage graph");
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                model.qualification_round_stage(runtime, batch)?;
            }
            let second = download(&model)?;
            let truth_metrics = truth
                .as_ref()
                .map(|outputs| metrics(&first, &outputs[which], 7));
            let key = self.key("stage", which);
            record(rows, &key, first, second, &references[which], 7, declared);
            rows.last_mut().expect("stage row")["truth"] = json!(truth_metrics);
            rows.last_mut().expect("stage row")["truth_sha256"] =
                json!(truth.as_ref().map(|outputs| sha(&outputs[which])));
            let Some(layers) = &self.band else {
                continue;
            };
            let mut band = Vec::new();
            let mut truth_draws = Vec::new();
            for seed in BAND_SEEDS {
                set_band(Some((seed, layers.clone())));
                model
                    .workspace(runtime, batch, WINDOW)?
                    .upload_input(runtime, input)?;
                model.forward_eager(runtime, batch, WINDOW)?;
                let output = download(&model)?;
                set_band(None);
                band.push(metrics(&output, &references[which], 7));
                truth_draws.push(metrics(
                    &output,
                    &truth.as_ref().expect("TF32 truth")[which],
                    7,
                ));
            }
            rows.push(json!({"id": format!("{key}/band"), "seeds": BAND_SEEDS, "layers": layers, "metrics": band, "truth_draws": truth_draws, "truth_sha256": sha(&truth.as_ref().expect("TF32 truth")[which])}));
        }
        Ok(())
    }
}

/// One isolated boundary of either stage
enum OperatorRun<'m, 'o> {
    Conv(&'o mut Operator<'m>),
    Segmentation {
        model: &'o mut CudaSegmentation,
        op: &'o mut crate::inference::cuda::segmentation::test_support::Isolated,
        references: [Vec<f32>; 2],
    },
}

impl OperatorRun<'_, '_> {
    fn restore(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        match self {
            Self::Conv(op) => op.restore(runtime, which),
            Self::Segmentation { model, op, .. } => model.isolated_restore(runtime, op, which),
        }
    }

    fn run(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        match self {
            Self::Conv(op) => op.run(runtime, which),
            Self::Segmentation { model, op, .. } => model.isolated_run(runtime, op, which),
        }
    }

    fn output(&self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        match self {
            Self::Conv(op) => op.output(runtime),
            Self::Segmentation { model, op, .. } => model.isolated_output(runtime, op),
        }
    }

    fn reference(&self, which: usize) -> Vec<f32> {
        match self {
            Self::Conv(op) => op.references[which].clone(),
            Self::Segmentation { references, .. } => references[which].clone(),
        }
    }

    /// Per-layer LSTM taps for the case file, never gated
    fn diagnostics(
        &self,
        runtime: &CudaRuntime,
        file: &SafetensorsFile,
        batch: usize,
        stack: &[f32],
    ) -> Result<Option<Vec<Value>>, CudaError> {
        match self {
            Self::Segmentation { model, op, .. } if op.is_lstm() => {
                Ok(Some(lstm_diagnostics(model, runtime, file, batch, stack)?))
            }
            _ => Ok(None),
        }
    }
}

/// Errors over exactly the elements computed by the independent f64 reference
fn difference(actual: &[f32], expected: &reference::Sample) -> Value {
    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut finite = true;
    for (&index, &y) in expected.indices.iter().zip(&expected.values) {
        let x = f64::from(actual[index]);
        finite &= x.is_finite() && y.is_finite();
        numerator += (x - y).powi(2);
        denominator += y.powi(2);
        max_abs = max_abs.max((x - y).abs());
    }
    let relative_l2 = (numerator / denominator).sqrt();
    finite &= relative_l2.is_finite();
    json!({"relative_l2": if finite { relative_l2 } else { -1.0 },
        "max_abs": if finite { max_abs } else { -1.0 }, "finite": finite,
        "elements": expected.indices.len()})
}

/// Same-input f64 truth and Library errors, without an artificial floor
struct SecretReference {
    truth: reference::Sample,
    library: Vec<f32>,
    nudged: Vec<f32>,
}

impl SecretReference {
    fn row(&self, id: String, seed: u64, eager: &[f32], replay: &[f32]) -> Value {
        let same = eager.len() == replay.len()
            && eager
                .iter()
                .zip(replay)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let mut hash = Sha256::new();
        for value in &self.truth.values {
            hash.update(value.to_le_bytes());
        }
        json!({"id": id, "secret": true, "seed": seed, "truth": "f64",
            "sample_indices": self.truth.indices, "truth_sha256": format!("{:x}", hash.finalize()),
            "library_algorithm": "PersistStaticSmallH",
            "library": difference(&self.library, &self.truth),
            "fp32_input_ulp": difference(&self.nudged, &self.truth),
            "eager": difference(eager, &self.truth),
            "replay": difference(replay, &self.truth), "replay_equals_eager": same})
    }
}

/// Reuses public fixture files instead of rereading gigabytes for each secret layer
struct BakedFixtures<'a> {
    target: &'a str,
    files: [SafetensorsFile; 2],
}

impl<'a> BakedFixtures<'a> {
    fn new(target: &'a str) -> Result<Self, CudaError> {
        Ok(Self {
            target,
            files: [reference(target, "mixed")?, reference(target, "short")?],
        })
    }

    fn errors(
        &self,
        expected: &SecretReference,
        batch: usize,
        name: &str,
    ) -> Result<Value, CudaError> {
        let mut errors = Vec::new();
        for (case, file) in ["mixed", "short"].into_iter().zip(&self.files) {
            let answer = if self.target == "sincnet" {
                let abs = read_batch(file, "tensor//sincnet/Abs_output_0", batch)?;
                abs.as_chunks::<3>()
                    .0
                    .iter()
                    .map(|window| window[0].max(window[1]).max(window[2]))
                    .collect()
            } else {
                read_batch(file, name, batch)?
            };
            errors.push(json!({"fixture": case, "error": difference(&answer, &expected.truth)}));
        }

        Ok(json!(errors))
    }
}

/// Moves each boundary input by one ULP in a seeded, independent direction
fn nudge(input: &[f32], state: &mut u64) -> Vec<f32> {
    input
        .iter()
        .map(|value| {
            if uniform(state) < 0.5 {
                value.next_down()
            } else {
                value.next_up()
            }
        })
        .collect()
}

/// Mixes two fixture windows, then shifts, scales and adds noise at 30–40 dB SNR
///
/// Only the seed and output errors are recorded, never the resulting samples
fn transformed_audio(fixture: &[f32], batch: usize, state: &mut u64) -> Vec<f32> {
    assert_eq!(fixture.len() % WINDOW, 0);
    let count = fixture.len() / WINDOW;
    assert!(count >= 2);
    let mut result = Vec::with_capacity(batch * WINDOW);
    for _ in 0..batch {
        let first = (uniform(state) * count as f32) as usize;
        let offset = 1 + (uniform(state) * (count - 1) as f32) as usize;
        let second = (first + offset) % count;
        let shift = (uniform(state) * WINDOW as f32) as usize;
        let mix = 0.1 + 0.8 * uniform(state);
        let gain = 0.5 + uniform(state);
        let snr = 30.0 + 10.0 * uniform(state);
        let mut row: Vec<f32> = (0..WINDOW)
            .map(|i| {
                let index = (i + shift) % WINDOW;
                gain * (mix * fixture[first * WINDOW + index]
                    + (1.0 - mix) * fixture[second * WINDOW + index])
            })
            .collect();
        let power = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / WINDOW as f64;
        // uniform noise has variance 1/3 on [-1, 1], so its amplitude sets the SNR
        let amplitude = (3.0 * power / 10.0f64.powf(f64::from(snr) / 10.0)).sqrt() as f32;
        for sample in &mut row {
            *sample += amplitude * (2.0 * uniform(state) - 1.0);
        }
        result.extend(row);
    }
    result
}

/// Secret operator inputs from a locked FP32 Library front end on transformed audio
fn embedding_secret_inputs(
    runtime: &CudaRuntime,
    library: &ResNetEmbedding,
    audio: &[f32],
    batch: usize,
) -> Result<Vec<HostInputs>, CudaError> {
    use crate::inference::cuda::CudaFbank;
    use crate::inference::cuda::embedding::EmbeddingTap;
    let fbank = CudaFbank::new(runtime, CudaMath::Fp32)?;
    let features = {
        let mut buffers = fbank.buffers(runtime, batch)?;
        let waveforms: Vec<&[f32]> = audio
            .as_chunks::<WINDOW>()
            .0
            .iter()
            .map(<[f32; WINDOW]>::as_slice)
            .collect();
        let features = fbank.compute_host(runtime, &waveforms, &mut buffers)?;
        runtime.stream().clone_dtoh(&features)?
    };
    let mut front = library.batch(runtime, batch)?;
    front
        .fbank_mut()
        .copy_from_host(runtime.stream(), &features)?;
    // masks affect only pooling after the last eligible convolution
    front
        .masks_mut()
        .copy_from_host(runtime.stream(), &vec![1.0; batch * 3 * 589])?;
    let mut inputs = Vec::new();
    let mut previous = Vec::new();
    front.forward_with_taps(runtime, &mut |tap, values| {
        match tap {
            EmbeddingTap::Stem => previous = runtime.stream().clone_dtoh(values)?,
            EmbeddingTap::Hidden { block } if block < 7 => {
                inputs.push(HostInputs {
                    input: previous.clone(),
                    residual: None,
                });
                inputs.push(HostInputs {
                    input: runtime.stream().clone_dtoh(values)?,
                    residual: Some(previous.clone()),
                });
            }
            EmbeddingTap::Shortcut { block } if block < 7 => {
                inputs[2 * block + 1].residual = Some(runtime.stream().clone_dtoh(values)?);
            }
            EmbeddingTap::Block { block } if block < 6 => {
                previous = runtime.stream().clone_dtoh(values)?
            }
            _ => {}
        }
        Ok(())
    })?;
    assert_eq!(inputs.len(), 14);
    Ok(inputs)
}

fn conv_secret_output(
    runtime: &CudaRuntime,
    model: &ResNetEmbedding,
    input: &HostInputs,
    batch: usize,
    block: usize,
    second: bool,
) -> Result<Vec<f32>, CudaError> {
    let mut op = Operator::from_host(
        model,
        runtime,
        std::slice::from_ref(input),
        Vec::new(),
        batch,
        block,
        second,
    )?;
    op.run(runtime, 0)?;
    op.output(runtime)
}

/// Selects the declared candidate or a Library control
#[derive(Clone, Copy)]
enum SecretPlan<'a> {
    Candidate(&'a str),
    Library(CudaLstmAlgorithm),
}

fn segmentation_secret(
    runtime: &CudaRuntime,
    target: &str,
    plan: SecretPlan<'_>,
    math: CudaMath,
    rows: &mut Vec<Value>,
) -> Result<(), CudaError> {
    use crate::inference::cuda::segmentation::SegmentationTensor;
    let seed = secret_seed();
    let mut state = seed;
    let weights = SafetensorsFile::open("/workspace/models-native/segmentation-3.0.safetensors")?
        .perturbed(seed, 0.01);
    let options = SegmentationOptions {
        math,
        lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
        cuda_graph: false,
    };
    let fp32 = SegmentationOptions {
        math: CudaMath::Fp32,
        ..options
    };
    let fixture = read_batch(&reference(target, "mixed")?, "input/input", 32)?;
    let baked = BakedFixtures::new(target)?;
    let layer = if target == "lstm" {
        "lstm.stack"
    } else {
        "sincnet.conv0.abs_pool"
    };
    for batch in BATCHES {
        // each batch releases its front-end workspaces before the next batch
        let mut truth = CudaSegmentation::new(runtime, &weights, fp32)?;
        let mut library = CudaSegmentation::new(runtime, &weights, options)?;
        let candidate_options = match plan {
            SecretPlan::Candidate(_) => options,
            SecretPlan::Library(lstm_algo) => SegmentationOptions {
                lstm_algo,
                ..options
            },
        };
        let mut candidate = CudaSegmentation::new(runtime, &weights, candidate_options)?;
        if let SecretPlan::Candidate(choice) = plan {
            if target == "lstm" {
                select_lstm(&mut candidate, runtime, [batch, WINDOW], choice)?;
            } else {
                select_sinc(&mut candidate, runtime, [batch, WINDOW], choice)?;
            }
            if !candidate.isolated_declared(batch, target) {
                continue;
            }
        }
        let audio = transformed_audio(&fixture, batch, &mut state);
        truth
            .workspace(runtime, batch, WINDOW)?
            .upload_input(runtime, &audio)?;
        truth.forward_eager(runtime, batch, WINDOW)?;
        let tensor = if target == "lstm" {
            SegmentationTensor::LstmInput
        } else {
            SegmentationTensor::WaveNorm
        };
        let input = truth
            .find_workspace(batch, WINDOW)
            .expect("front-end workspace")
            .tensor(tensor)
            .download(runtime.stream())?;
        let output = |model: &mut CudaSegmentation, input: &[f32]| -> Result<Vec<f32>, CudaError> {
            let mut op = model.isolated(runtime, batch, target, [input, input])?;
            model.isolated_run(runtime, &mut op, 0)?;
            model.isolated_output(runtime, &op)
        };
        let expected = SecretReference {
            truth: truth.f64_reference(runtime, &input, batch, target, &mut state)?,
            library: output(&mut library, &input)?,
            nudged: output(&mut truth, &nudge(&input, &mut state))?,
        };
        let mut op = candidate.isolated(runtime, batch, target, [&input, &input])?;
        candidate.isolated_run(runtime, &mut op, 0)?;
        let eager = candidate.isolated_output(runtime, &op)?;
        candidate.isolated_restore(runtime, &mut op, 0)?;
        let graph = capture(runtime, || candidate.isolated_run(runtime, &mut op, 0))?;
        candidate.isolated_restore(runtime, &mut op, 0)?;
        graph.launch()?;
        let replay = candidate.isolated_output(runtime, &op)?;
        let mut row = expected.row(
            format!("{}/secret/b{batch}/{layer}", math_name(math)),
            seed,
            &eager,
            &replay,
        );
        row["library_algorithm"] = json!(if target == "lstm" {
            "PersistStaticSmallH"
        } else {
            "cuDNN convolution planner"
        });
        row["baked_answers"] =
            baked.errors(&expected, batch, "tensor//lstm/Transpose_5_output_0")?;
        rows.push(row);
    }
    Ok(())
}

/// Fresh weights and realistic inputs stay in memory and are inaccessible to candidates
fn secret(
    runtime: &CudaRuntime,
    target: &str,
    choice: &str,
    math: CudaMath,
    rows: &mut Vec<Value>,
) -> Result<(), CudaError> {
    if target != "resnet" {
        return segmentation_secret(
            runtime,
            target,
            if choice == "Library" {
                SecretPlan::Library(CudaLstmAlgorithm::PersistStaticSmallH)
            } else {
                SecretPlan::Candidate(choice)
            },
            math,
            rows,
        );
    }
    let seed = secret_seed();
    let mut state = seed;
    let weights =
        SafetensorsFile::open("/workspace/models-native/wespeaker-multimask-tail.safetensors")?
            .perturbed(seed, 0.01);
    let truth = ResNetEmbedding::load(runtime, &weights, CudaMath::Fp32)?;
    let library = ResNetEmbedding::load(runtime, &weights, math)?;
    let mut candidate = ResNetEmbedding::load(runtime, &weights, math)?;
    for (stage, count) in [(1, 3), (2, 4)] {
        for block in 0..count {
            for conv in [1, 2] {
                assert!(select_conv(
                    &mut candidate,
                    &format!("resnet.layer{stage}.{block}.conv{conv}"),
                    choice
                ));
            }
        }
    }
    let fixture =
        SafetensorsFile::open("/workspace/ref/wespeaker-fbank-b32/test_and_short_b32.safetensors")?;
    let fixture = fixture.read_f32("input/waveform", &[32, 1, WINDOW])?;
    let baked = BakedFixtures::new(target)?;
    for batch in BATCHES {
        let audio = transformed_audio(&fixture, batch, &mut state);
        let inputs = embedding_secret_inputs(runtime, &truth, &audio, batch)?;
        for (index, input) in inputs.into_iter().enumerate() {
            let block = index / 2;
            let second = index % 2 == 1;
            let nudged = HostInputs {
                input: nudge(&input.input, &mut state),
                residual: input
                    .residual
                    .as_ref()
                    .map(|values| nudge(values, &mut state)),
            };
            let expected = SecretReference {
                truth: Operator::from_host(
                    &truth,
                    runtime,
                    std::slice::from_ref(&input),
                    Vec::new(),
                    batch,
                    block,
                    second,
                )?
                .f64_reference(runtime, &input, &mut state)?,
                library: conv_secret_output(runtime, &library, &input, batch, block, second)?,
                nudged: conv_secret_output(runtime, &truth, &nudged, batch, block, second)?,
            };
            drop(nudged);
            let mut op = Operator::from_host(
                &candidate,
                runtime,
                &[input],
                Vec::new(),
                batch,
                block,
                second,
            )?;
            if choice != "Library" && !op.declared() {
                continue;
            }
            op.run(runtime, 0)?;
            let eager = op.output(runtime)?;
            op.restore(runtime, 0)?;
            let graph = capture(runtime, || op.run(runtime, 0))?;
            op.restore(runtime, 0)?;
            graph.launch()?;
            let replay = op.output(runtime)?;
            let mut row = expected.row(
                format!("{}/secret/b{batch}/{}", math_name(math), op.name()),
                seed,
                &eager,
                &replay,
            );
            let output_name = format!("tensor/relu_{}", 2 * block + if second { 2 } else { 1 });
            row["library_algorithm"] = json!("cuDNN convolution planner");
            row["baked_answers"] = baked.errors(&expected, batch, &output_name)?;
            rows.push(row);
        }
    }
    Ok(())
}

fn parse_mode(text: &str) -> CudaMath {
    match text {
        "fp32" => CudaMath::Fp32,
        "tf32" => CudaMath::Tf32,
        _ => panic!("math mode"),
    }
}

/// `<batch>:<layer>,<layer>;<batch>:...`, the declared layers the TF32 noise band
/// perturbs at each batch size
fn parse_band(text: &str) -> Vec<(usize, Vec<String>)> {
    text.split(';')
        .filter(|item| !item.is_empty())
        .map(|item| {
            let (batch, layers) = item.split_once(':').expect("batch:layers");
            let batch = batch.parse().expect("band batch");
            assert!(
                BATCHES.contains(&batch),
                "band batch from the fixed inventory"
            );
            (batch, layers.split(',').map(str::to_owned).collect())
        })
        .collect()
}

/// The mode, case and batch triples this process measures
fn selected_cases(phase: &str) -> Vec<(CudaMath, &'static str, usize)> {
    let all = [CudaMath::Fp32, CudaMath::Tf32]
        .into_iter()
        .flat_map(|math| CASES.map(|(case, batch)| (math, case, batch)));
    match phase {
        "numeric" | "timing" | "paired" => {
            let mode = std::env::var("SPEAKRS_QUALIFY_MODE").expect("one math mode per process");
            let math = parse_mode(&mode);
            all.filter(|(item, _, _)| *item == math).collect()
        }
        "sanitize" => {
            let batches: Vec<usize> = std::env::var("SPEAKRS_QUALIFY_SANITIZE_BATCHES")
                .expect("sanitizer batches")
                .split(',')
                .map(|batch| batch.parse().expect("sanitizer batch"))
                .collect();
            assert!(batches.iter().all(|batch| BATCHES.contains(batch)));
            let only = std::env::var("SPEAKRS_QUALIFY_SANITIZER_BATCH")
                .ok()
                .map(|batch| batch.parse::<usize>().expect("sanitizer batch"));
            all.filter(|(_, case, batch)| {
                let canonical = if *batch == 1 { "first" } else { "mixed" };
                *case == canonical
                    && batches.contains(batch)
                    && only.is_none_or(|selected| selected == *batch)
            })
            .collect()
        }
        _ => all.collect(),
    }
}

fn write(result: &Value) {
    std::fs::write(
        std::env::var("SPEAKRS_QUALIFY_OUTPUT").expect("output path"),
        serde_json::to_vec_pretty(result).expect("finite JSON"),
    )
    .expect("write evidence");
}

#[test]
#[ignore = "locked CUDA qualification entry point"]
fn qualification_driver() -> Result<(), CudaError> {
    let target = std::env::var("SPEAKRS_QUALIFY_TARGET").expect("target");
    assert!(["resnet", "sincnet", "lstm"].contains(&target.as_str()));
    let implementation = std::env::var("SPEAKRS_QUALIFY_IMPL").expect("implementation");
    let choice = implementation.as_str();
    let phase = std::env::var("SPEAKRS_QUALIFY_PHASE").expect("phase");
    assert!(
        [
            "coverage",
            "numeric",
            "timing",
            "profile",
            "paired",
            "sanitize",
            "projection_baseline",
            "filter_proof"
        ]
        .contains(&phase.as_str())
    );
    let sanitizer_layer = std::env::var("SPEAKRS_QUALIFY_SANITIZER_LAYER").ok();
    if let Some(name) = &sanitizer_layer {
        assert_eq!(phase, "sanitize");
        assert_eq!(target, "resnet");
        assert!((1..=2).any(|stage| {
            (0..if stage == 1 { 3 } else { 4 }).any(|block| {
                (1..=2).any(|conv| name == &format!("resnet.layer{stage}.{block}.conv{conv}"))
            })
        }));
    }
    // integration can enable production kernels, but this driver owns all choices
    assert_eq!(
        super::default_choice(super::Choice::Oxide(
            crate::inference::cuda::implementation::Selection::Production
        )),
        super::Choice::Library
    );
    let tier = std::env::var("SPEAKRS_CUDA_PTX_TIER")
        .expect("requested tier")
        .parse()
        .expect("known tier");
    let coverage = if matches!(choice, "StageTail" | "StageTailControl") {
        assert!(matches!(target.as_str(), "resnet" | "sincnet"));
        let runtime = CudaRuntime::new(0)?;
        let area = if target == "resnet" {
            KernelModule::Resnet
        } else {
            KernelModule::Sincnet
        };
        let target = crate::inference::cuda::implementation::Target::for_area(&runtime, area)?;
        let pinned = crate::inference::cuda::implementation::production_coverage(area, target);
        let entries: Vec<_> = pinned
            .entries()
            .iter()
            .filter(|entry| match entry.maths {
                Maths::All => true,
                Maths::Only(maths) => maths.contains(&CudaMath::Fp32),
            })
            .map(|entry| {
                let mut entry = entry_json(entry);
                entry["maths"] = json!(["fp32"]);
                entry
            })
            .collect();
        assert!(
            !entries.is_empty(),
            "StageTail requires pinned FP32 production plans"
        );
        json!({"entries": entries})
    } else {
        declared_coverage(&target, choice, tier)
    };
    if phase == "coverage" {
        write(
            &json!({"target":target,"implementation":implementation,"phase":phase,"coverage":coverage}),
        );
        return Ok(());
    }

    let runtime = CudaRuntime::new(0)?;
    if phase == "filter_proof" {
        let control = std::env::var("SPEAKRS_QUALIFY_CONTROL").expect("sanitizer control");
        return sanitizer_control(&runtime, &control);
    }
    if phase == "projection_baseline" {
        assert_eq!(target, "lstm");
        assert_eq!(choice, "Library");
        return projection_baseline(&runtime);
    }

    prepare(&runtime)?;
    if matches!(choice, "StageTail" | "StageTailControl") {
        paired::prepare_tail(&runtime)?;
    }
    let device = crate::inference::cuda::candidate::test_support::device(&runtime)?;
    let clocks = matches!(phase.as_str(), "timing" | "paired")
        .then(crate::inference::cuda::candidate::test_support::Clocks::start);
    if target == "resnet" {
        // the secret front end needs fbank only in numeric, but fixed module bytes
        // must have the same inventory in timing and eager-profile processes
        runtime.load_kernels(KernelModule::Fbank)?;
    }

    let band = std::env::var("SPEAKRS_QUALIFY_BAND_LAYERS")
        .map(|text| parse_band(&text))
        .unwrap_or_default();
    assert!(band.is_empty() || (phase == "numeric" && choice == "Library"));
    let mut rows = Vec::new();
    let cases = selected_cases(&phase);
    assert!(!cases.is_empty(), "the process measures at least one case");
    for (math, case, batch) in &cases {
        let name = math_name(*math);
        println!("qualify target={target} phase={phase} mode={name} case={case} batch={batch}");
        let file = reference(&target, case)?;
        let alternate = reference(&target, if *case == "short" { "first" } else { "short" })?;
        let run = Run {
            runtime: &runtime,
            files: [&file, &alternate],
            batch: *batch,
            target: &target,
            choice,
            phase: &phase,
            id: format!("{name}/{case}/b{batch}"),
            math: *math,
            band: band
                .iter()
                .find(|(declared, _)| declared == batch && *math == CudaMath::Tf32)
                .map(|(_, layers)| layers.clone()),
        };
        if phase == "paired" {
            run.paired(&mut rows)?;
        } else if target == "resnet" {
            run.embedding(&mut rows)?;
        } else {
            run.segmentation(&mut rows)?;
        }
    }
    if phase == "numeric" {
        let math = cases.first().map_or(CudaMath::Fp32, |(math, _, _)| *math);
        secret(&runtime, &target, choice, math, &mut rows)?;
    }
    runtime.synchronize()?;
    let mode = std::env::var("SPEAKRS_QUALIFY_MODE").ok();
    write(&json!({
        "target": target,
        "implementation": implementation,
        "phase": phase,
        "mode": mode,
        "pid": std::process::id(),
        "rows": rows,
        "tier": runtime.ptx_tier().to_string(),
        "device_sm": runtime.compute_capability().to_string(),
        "device": device,
        "observed_sm_clock": clocks.map(crate::inference::cuda::candidate::test_support::Clocks::finish),
        "library_call_violations": library_call_violations(),
        "graph_violations": graph_violations(),
        "graph_evidence": graph_evidence(),
        "loaded_modules": loaded_modules(),
        "side_streams": registered_side_streams(),
        "coverage": coverage,
        "lstm_algorithm": "PersistStaticSmallH",
        "sinc_timing_boundary": "producer + unchanged shared instance-norm consumer",
    }));
    Ok(())
}

/// cuBLAS input projections at every shape the locked helper can issue in the harness
fn projection_baseline(runtime: &CudaRuntime) -> Result<(), CudaError> {
    runtime.prepare_library(crate::inference::cuda::CudaLibrary::Cublas)?;
    use crate::inference::cuda::blas::Sgemm;
    let mut shapes = Vec::new();
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        for k in [60, 256] {
            for batch in BATCHES {
                for n in Projection::COLUMNS {
                    let m = batch * 589;
                    let values = |len| {
                        (0..len)
                            .map(|i| (i % 31) as f32 / 31.0 - 0.5)
                            .collect::<Vec<_>>()
                    };
                    let a = runtime.stream().clone_htod(&values(m * k))?;
                    let b = runtime.stream().clone_htod(&values(n * k))?;
                    let mut c = runtime.stream().alloc_zeros::<f32>(m * n)?;
                    let spec = Sgemm {
                        b_transposed: true,
                        math,
                        ..Sgemm::new(m, n, k)
                    };
                    let mode = math_name(math);
                    println!("projection baseline mode={mode} batch={batch} m={m} n={n} k={k}");
                    runtime.sgemm(spec, &a, &b, &mut c)?;
                    runtime.synchronize()?;
                    shapes.push(json!({"mode":mode,"batch":batch,"m":m,"n":n,"k":k}));
                }
            }
        }
    }
    write(
        &json!({"phase":"projection_baseline","device_sm":runtime.compute_capability().to_string(),"shapes":shapes}),
    );
    Ok(())
}

#[test]
fn second_tile_uses_different_reference_rows() {
    for row in 0..32 {
        assert_ne!(source_row(row, 32), source_row(row + 32, 32));
    }

    assert_eq!(source_row(32, 32), 17);
    assert_eq!(source_row(0, 1), 0);
}

#[test]
fn band_layers_parse_per_batch() {
    let band = parse_band("7:lstm.stack;64:lstm.stack");
    assert_eq!(band.len(), 2);
    assert_eq!(band[1], (64, vec!["lstm.stack".to_owned()]));
    assert!(std::panic::catch_unwind(|| parse_band("48:lstm.stack")).is_err());
}

#[test]
#[ignore = "requires the GPU lock and qualification environment"]
fn sequential_capture_keeps_candidate_kernels() -> Result<(), CudaError> {
    assert!(
        super::phase().is_some(),
        "qualification scope tracking must be on"
    );
    let runtime = CudaRuntime::new(0)?;
    prepare(&runtime)?;
    // use a current library-free candidate; this tests shared scope tracking, not
    // whether the old LSTM candidate qualifies under the new projection rule
    let weights =
        SafetensorsFile::open("/workspace/models-native/wespeaker-multimask-tail.safetensors")?;
    let mut model = ResNetEmbedding::load(&runtime, &weights, CudaMath::Fp32)?;
    assert!(select_conv(&mut model, "resnet.layer1.0.conv1", "Oxide"));
    let file = reference("resnet", "mixed")?;
    for (position, batch) in [1, 1, 7, 32, 33, 64, 7].into_iter().enumerate() {
        let mut op = Operator::new(&model, &runtime, [&file, &file], batch, 0, false)?;
        let key = format!("capture-regression/{position}/b{batch}");
        {
            let _window = window(&key);
            op.run(&runtime, 0)?;
        }
        runtime.synchronize()?;
        set_label(Some(key));
        let graph = capture(&runtime, || op.run(&runtime, 0));
        set_label(None);
        graph?.launch()?;
        runtime.synchronize()?;
        // Python checks the complete launch multiset against this same plan's eager
        // nsys window after export; no candidate declaration or fixed count is used
    }
    let device = crate::inference::cuda::candidate::test_support::device(&runtime)?;
    write(
        &json!({"phase":"profile","requested_tier":runtime.ptx_tier().to_string(),
                  "tier":runtime.ptx_tier().to_string(),"device_sm":runtime.compute_capability().to_string(),
                  "device":device,"loaded_modules":loaded_modules(),
                  "graph_evidence":graph_evidence(),"graph_violations":graph_violations()}),
    );
    assert!(graph_violations().is_empty());
    Ok(())
}

/// Exercise requested compiled tiers through real PTX, identity and clock evidence
#[test]
#[ignore = "requires the GPU lock and qualification environment"]
fn compiled_tier_fixture() -> Result<(), CudaError> {
    use crate::inference::cuda::DeviceTensor;
    use crate::inference::cuda::probe::ProbeKernels;
    let runtime = CudaRuntime::new(0)?;
    let device = crate::inference::cuda::candidate::test_support::device(&runtime)?;
    let clocks = crate::inference::cuda::candidate::test_support::Clocks::start();
    let probe = ProbeKernels::load(&runtime)?;
    assert_eq!(probe.tier(), runtime.ptx_tier());
    let input: Vec<f32> = (0..1007).map(|i| i as f32).collect();
    let x = DeviceTensor::upload(runtime.stream(), &input, &[input.len()])?;
    let y = DeviceTensor::upload(runtime.stream(), &input, &[input.len()])?;
    let mut output = DeviceTensor::<f32>::zeros(runtime.stream(), &[input.len()])?;
    let graph = capture(&runtime, || {
        probe.scale_add(&runtime, 2.0, x.data(), y.data(), output.data_mut())
    })?;
    let timing = bursts(&runtime, |_| Ok(()), |_| Ok(graph.launch()?))?;
    let actual = output.download(runtime.stream())?;
    let expected: Vec<f32> = input.iter().map(|x| 3.0 * x).collect();
    assert_eq!(actual, expected);
    write(
        &json!({"phase":"timing","requested_tier":runtime.ptx_tier().to_string(),
                  "tier":probe.tier().to_string(),"device_sm":runtime.compute_capability().to_string(),
                  "device":device,"loaded_modules":loaded_modules(),"pid":std::process::id(),
                  "observed_sm_clock":clocks.finish(),"output_sha256":sha(&actual),
                  "warmup":WARMUP,"samples_ms":timing.per_launch_ms}),
    );
    Ok(())
}

#[test]
#[ignore = "requires the GPU lock and qualification environment"]
fn secret_library_algorithms() -> Result<(), CudaError> {
    let runtime = CudaRuntime::new(0)?;
    prepare(&runtime)?;
    let seed = secret_seed();
    let mut state = seed;
    let weights = SafetensorsFile::open("/workspace/models-native/segmentation-3.0.safetensors")?
        .perturbed(seed, 0.01);
    let options = SegmentationOptions {
        math: CudaMath::Fp32,
        lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
        cuda_graph: false,
    };
    let fixture = read_batch(&reference("lstm", "mixed")?, "input/input", 32)?;
    let mut rows = Vec::new();
    for batch in BATCHES {
        use crate::inference::cuda::segmentation::SegmentationTensor;
        let mut small = CudaSegmentation::new(&runtime, &weights, options)?;
        let mut standard = CudaSegmentation::new(
            &runtime,
            &weights,
            SegmentationOptions {
                lstm_algo: CudaLstmAlgorithm::Standard,
                ..options
            },
        )?;
        let audio = transformed_audio(&fixture, batch, &mut state);
        small
            .workspace(&runtime, batch, WINDOW)?
            .upload_input(&runtime, &audio)?;
        small.forward_eager(&runtime, batch, WINDOW)?;
        let input = small
            .find_workspace(batch, WINDOW)
            .expect("front-end workspace")
            .tensor(SegmentationTensor::LstmInput)
            .download(runtime.stream())?;
        let truth = small.f64_reference(&runtime, &input, batch, "lstm", &mut state)?;
        let run = |model: &mut CudaSegmentation, input: &[f32]| -> Result<Vec<f32>, CudaError> {
            let mut op = model.isolated(&runtime, batch, "lstm", [input, input])?;
            model.isolated_run(&runtime, &mut op, 0)?;
            model.isolated_output(&runtime, &op)
        };
        let a = difference(&run(&mut standard, &input)?, &truth);
        let b = difference(&run(&mut small, &input)?, &truth);
        // one sampled boundary element moves one ULP, matching the chaos diagnostic
        let mut nudged = input.clone();
        let index = truth.indices[0] / (589 * 256) * 589 * 60
            + (uniform(&mut state) * (589 * 60) as f32) as usize;
        nudged[index] = nudged[index].next_up();
        let n = difference(&run(&mut small, &nudged)?, &truth);
        rows.push(json!({"batch": batch, "seed": seed, "Standard": a,
            "PersistStaticSmallH": b, "fp32_input_ulp": n,
            "smallh_standard_l2_ratio": b["relative_l2"].as_f64().expect("finite") / a["relative_l2"].as_f64().expect("finite"),
            "nudge_l2_ratio": n["relative_l2"].as_f64().expect("finite") / b["relative_l2"].as_f64().expect("finite")}));
    }
    write(&json!({"truth": "f64", "rows": rows}));
    for row in rows {
        for name in ["Standard", "PersistStaticSmallH", "fp32_input_ulp"] {
            assert_eq!(row[name]["finite"], true);
            assert!(
                row[name]["relative_l2"].as_f64().expect("finite") < 1e-4,
                "not FP32 rounding scale: {row}"
            );
        }
        assert!(
            (0.9..=1.1).contains(&row["nudge_l2_ratio"].as_f64().expect("finite")),
            "one-ULP sensitivity is not marginal: {row}"
        );
    }
    Ok(())
}

#[test]
fn secret_audio_is_seeded_and_not_a_fixture_lookup() {
    let fixture: Vec<f32> = (0..2 * WINDOW)
        .map(|i| (i as f32 * 0.007).sin() * 0.2)
        .collect();
    let first = transformed_audio(&fixture, 2, &mut 17);
    assert_eq!(first, transformed_audio(&fixture, 2, &mut 17));
    assert_ne!(first, transformed_audio(&fixture, 2, &mut 18));
    assert!(first.iter().all(|value| value.is_finite()));
    assert_ne!(first, fixture);
    assert_ne!(&first[..WINDOW], &first[WINDOW..]);
}
