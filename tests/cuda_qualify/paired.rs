//! Same-process replay-level ABBA evidence for the stage and its eligible operators

use super::{Run, WARMUP, WINDOW, capture, read_batch, sha};
use crate::inference::cuda::embedding::test_support::Operator;
use crate::inference::cuda::test_support::{select_conv, select_lstm, select_sinc};
use crate::inference::cuda::{
    CudaError, CudaLstmAlgorithm, CudaRuntime, CudaSegmentation, ResNetEmbedding, SafetensorsFile,
    SegmentationOptions,
};
use cudarc::driver::{CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::driver::{CudaGraph, sys::CUevent_flags};
use serde_json::{Value, json};
use std::cell::RefCell;

pub(super) const BLOCKS: usize = 256;
const TAIL_BLOCKS: usize = 16;

thread_local! {
    static TAIL_KERNEL: RefCell<Option<CudaFunction>> = const { RefCell::new(None) };
}

/// Releases test CUDA functions while the shared GPU lock is held
pub(super) fn clear_tail() {
    TAIL_KERNEL.with(|cell| *cell.borrow_mut() = None);
}

/// Keep the recorded module inventory identical in every StageTail phase
pub(super) fn prepare_tail(runtime: &CudaRuntime) -> Result<(), CudaError> {
    let module = super::super::load_recorded(
        runtime,
        "stage_tail",
        include_str!("device/stage_tail.sm75.ptx"),
    )?;
    TAIL_KERNEL.with(|cell| {
        *cell.borrow_mut() = Some(module.load_function("qualify_stage_tail")?);
        Ok(())
    })
}

/// A real candidate stage followed by a device delay, captured as one graph
struct Tail {
    graph: CudaGraph,
    evidence: Value,
}

impl Tail {
    fn capture(
        runtime: &CudaRuntime,
        library: &CudaGraph,
        factor: f32,
        mut candidate_stage: impl FnMut() -> Result<(), CudaError>,
    ) -> Result<Self, CudaError> {
        let mut calibration = Vec::new();
        for _ in 0..WARMUP {
            calibration.push(elapsed(runtime, || Ok(library.launch()?))?);
        }
        let stage_ms = super::median(&calibration);
        assert!(stage_ms.is_finite() && stage_ms > 0.0);
        let duration_ns = (f64::from(stage_ms) * f64::from(factor) * 1_000_000.0).ceil() as u64;
        let kernel =
            TAIL_KERNEL.with(|cell| cell.borrow().as_ref().expect("prepared tail").clone());
        let spin = || -> Result<(), CudaError> {
            let _scope = super::super::fixed("qualify_stage_tail");
            let mut launch = runtime.stream().launch_builder(&kernel);
            launch.arg(&duration_ns);
            // SAFETY: one thread reads only a duration and the device timer, no pointers
            unsafe {
                launch.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            Ok(())
        };
        let spin_ms = elapsed(runtime, spin)?;
        // event resolution can round down by a tick; the fault must still be real GPU work
        assert!(f64::from(spin_ms) >= 0.99 * duration_ns as f64 / 1_000_000.0);
        let graph = capture(runtime, || {
            candidate_stage()?;
            spin()
        })?;
        Ok(Self {
            graph,
            evidence: json!({"entry":"qualify_stage_tail", "timer":"globaltimer_ns",
                "calibration_library_ms":calibration, "calibration_stage_ms":stage_ms,
                "extra_stage_fraction":factor, "duration_ns":duration_ns,
                "measured_spin_ms":spin_ms, "first_block":0, "group_abba_blocks":TAIL_BLOCKS,
                "candidate_replays_per_block":2, "operators_delayed":false,
                "captured_with_real_candidate_stage":true}),
        })
    }

    fn for_block(&self, block: Option<usize>) -> Option<&CudaGraph> {
        block
            .filter(|index| *index < TAIL_BLOCKS)
            .map(|_| &self.graph)
    }
}

/// Each observation times exactly one replay, never a burst of one implementation
fn elapsed(
    runtime: &CudaRuntime,
    launch: impl FnOnce() -> Result<(), CudaError>,
) -> Result<f32, CudaError> {
    let _interval = super::super::TimedInterval::start();
    let start = runtime
        .stream()
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    launch()?;
    let end = runtime
        .stream()
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
    end.synchronize()?;
    Ok(start.elapsed_ms(&end)?)
}

/// Warm-up and collection share the same alternating-input, replay-level schedule
pub(super) fn replay_set(
    blocks: usize,
    mut observe: impl FnMut(usize, Option<usize>) -> Result<([f32; 4], [f32; 4]), CudaError>,
) -> Result<Value, CudaError> {
    let mut stage = Vec::with_capacity(blocks);
    let mut operator = Vec::with_capacity(blocks);
    for block in 0..WARMUP + blocks {
        let (stage_ms, operator_ms) = observe(block % 2, block.checked_sub(WARMUP))?;
        if block >= WARMUP {
            stage.push(stage_ms);
            operator.push(operator_ms);
        }
    }
    Ok(json!({"order":"ABBA","warmup":WARMUP,"stage_abba_ms":stage,
              "operator_abba_ms":operator,"cuda_graph":true,"declared":true,
              "pid":std::process::id()}))
}

pub(super) fn observe(
    runtime: &CudaRuntime,
    stages: [&CudaGraph; 2],
    operators: [&[[CudaGraph; 2]]; 2],
    which: usize,
    slow: bool,
    tail: Option<&CudaGraph>,
) -> Result<([f32; 4], [f32; 4]), CudaError> {
    let mut stage_ms = [0.0; 4];
    let mut operator_ms = [0.0; 4];
    for (index, side) in [0, 1, 1, 0].into_iter().enumerate() {
        operator_ms[index] = elapsed(runtime, || {
            for graph in operators[side] {
                graph[which].launch()?;
            }
            Ok(())
        })?;
        stage_ms[index] = elapsed(runtime, || {
            let graph = if side == 1 {
                tail.unwrap_or(stages[side])
            } else {
                stages[side]
            };
            graph.launch()?;
            if side == 1 && slow {
                stages[side].launch()?;
            }
            Ok(())
        })?;
    }
    Ok((stage_ms, operator_ms))
}

fn conv_choices(model: &mut ResNetEmbedding, choice: &str) {
    for (stage, count) in [(1, 3), (2, 4)] {
        for block in 0..count {
            for conv in [1, 2] {
                assert!(select_conv(
                    model,
                    &format!("resnet.layer{stage}.{block}.conv{conv}"),
                    choice
                ));
            }
        }
    }
}

impl Run<'_> {
    pub(super) fn paired(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        if self.target == "resnet" {
            self.paired_embedding(rows)
        } else {
            self.paired_segmentation(rows)
        }
    }

    fn paired_embedding(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let runtime = self.runtime;
        let weights =
            SafetensorsFile::open("/workspace/models-native/wespeaker-multimask-tail.safetensors")?;
        let library = ResNetEmbedding::load(runtime, &weights, self.math)?;
        let mut candidate = ResNetEmbedding::load(runtime, &weights, self.math)?;
        conv_choices(&mut candidate, self.choice);
        let mut eligible = Vec::new();
        for block in 0..7 {
            for second in [false, true] {
                if self.choice == "Library"
                    || Operator::declared_at(&candidate, runtime, self.batch, block, second)?
                {
                    eligible.push((block, second));
                }
            }
        }
        if eligible.is_empty() {
            return Ok(());
        }
        // equal strata preserve an unbiased sum of per-layer savings without keeping
        // every layer's two input sets, outputs and workspaces live at batch 64
        let blocks_per_layer = BLOCKS.div_ceil(16 * eligible.len()) * 16;
        let mut library_stage = library.batch(runtime, self.batch)?;
        let mut candidate_stage = candidate.batch(runtime, self.batch)?;
        let inputs = self.files.map(|file| -> Result<_, CudaError> {
            Ok((
                read_batch(file, "input/fbank", self.batch)?,
                read_batch(file, "input/masks", self.batch)?,
            ))
        });
        let [first, second] = inputs;
        let inputs = [first?, second?];
        let upload = |buffers: &mut crate::inference::cuda::EmbeddingBatch,
                      which: usize|
         -> Result<(), CudaError> {
            buffers
                .fbank_mut()
                .copy_from_host(runtime.stream(), &inputs[which].0)?;
            buffers
                .masks_mut()
                .copy_from_host(runtime.stream(), &inputs[which].1)
        };
        upload(&mut library_stage, 0)?;
        upload(&mut candidate_stage, 0)?;
        library_stage.capture_graph(runtime)?;
        candidate_stage.capture_graph(runtime)?;
        let mut stages = [library_stage, candidate_stage];
        let tail = if self.choice == "StageTail" && self.id == "fp32/first/b1" {
            let (library, candidate) = stages.split_at_mut(1);
            Some(Tail::capture(
                runtime,
                library[0].graph().expect("Library graph"),
                1.7,
                || candidate[0].forward_with_taps(runtime, &mut |_, _| Ok(())),
            )?)
        } else {
            None
        };
        let mut stage_blocks = Vec::new();
        let mut operator_blocks = Vec::new();
        let mut operator_layers = Vec::new();
        let mut block_layers = Vec::new();
        let mut operator_outputs = Vec::new();
        for (block, second) in eligible {
            let mut library_op =
                Operator::new(&library, runtime, self.files, self.batch, block, second)?;
            let mut candidate_op =
                Operator::new(&candidate, runtime, self.files, self.batch, block, second)?;
            assert!(candidate_op.declared() || self.choice == "Library");
            let capture_op = |op: &mut Operator<'_>| -> Result<[CudaGraph; 2], CudaError> {
                op.restore(runtime, 0)?;
                op.restore(runtime, 1)?;
                Ok([
                    capture(runtime, || op.run(runtime, 0))?,
                    capture(runtime, || op.run(runtime, 1))?,
                ])
            };
            let library_graphs = capture_op(&mut library_op)?;
            let candidate_graphs = capture_op(&mut candidate_op)?;
            let offset = stage_blocks.len();
            let collected = replay_set(blocks_per_layer, |which, block| {
                for buffers in &mut stages {
                    upload(buffers, which)?;
                }
                observe(
                    runtime,
                    [
                        stages[0].graph().expect("Library graph"),
                        stages[1].graph().expect("candidate graph"),
                    ],
                    [
                        std::slice::from_ref(&library_graphs),
                        std::slice::from_ref(&candidate_graphs),
                    ],
                    which,
                    self.choice == "StageSlow",
                    tail.as_ref()
                        .and_then(|tail| tail.for_block(block.map(|block| offset + block))),
                )
            })?;
            stage_blocks.extend(
                collected["stage_abba_ms"]
                    .as_array()
                    .expect("blocks")
                    .clone(),
            );
            operator_blocks.extend(
                collected["operator_abba_ms"]
                    .as_array()
                    .expect("blocks")
                    .clone(),
            );
            let layer = candidate_op.name().to_owned();
            block_layers.extend(std::iter::repeat_n(layer.clone(), blocks_per_layer));
            operator_layers.push(layer);
            operator_outputs.extend(
                op_outputs(
                    runtime,
                    std::slice::from_ref(&library_graphs),
                    std::slice::from_ref(&candidate_graphs),
                    std::slice::from_ref(&library_op),
                    std::slice::from_ref(&candidate_op),
                )?
                .as_array()
                .expect("operator outputs")
                .clone(),
            );
        }
        let mut row = json!({
            "order":"ABBA", "warmup":WARMUP, "warmup_per_operator":WARMUP,
            "stage_abba_ms":stage_blocks, "operator_abba_ms":operator_blocks,
            "operator_layers":operator_layers, "operator_layer_by_block":block_layers,
            "cuda_graph":true, "declared":true, "pid":std::process::id(),
        });
        row["id"] = json!(self.key("stage", 0));
        if let Some(tail) = &tail {
            row["stage_tail"] = tail.evidence.clone();
            row["stage_tail"]["group_count"] =
                json!(row["stage_abba_ms"].as_array().expect("blocks").len() / TAIL_BLOCKS);
        }
        let mut hashes = [Vec::new(), Vec::new()];
        for which in 0..2 {
            for (side, buffers) in stages.iter_mut().enumerate() {
                upload(buffers, which)?;
                buffers.graph().expect("stage graph").launch()?;
                hashes[side].push(sha(&buffers.download_output(runtime)?));
            }
        }
        row["output_sha256"] = json!(hashes);
        if let Some(tail) = &tail {
            let mut delayed = Vec::new();
            for which in 0..2 {
                upload(&mut stages[1], which)?;
                tail.graph.launch()?;
                delayed.push(sha(&stages[1].download_output(runtime)?));
            }
            row["stage_tail"]["delayed_output_sha256"] = json!(delayed);
        }
        row["operator_outputs"] = json!(operator_outputs);
        rows.push(row);
        Ok(())
    }

    fn paired_segmentation(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let runtime = self.runtime;
        let weights =
            SafetensorsFile::open("/workspace/models-native/segmentation-3.0.safetensors")?;
        let options = SegmentationOptions {
            math: self.math,
            lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
            cuda_graph: true,
        };
        let mut library = CudaSegmentation::new(runtime, &weights, options)?;
        let mut candidate = CudaSegmentation::new(runtime, &weights, options)?;
        if self.target == "lstm" {
            select_lstm(&mut candidate, runtime, [self.batch, WINDOW], self.choice)?;
        } else {
            select_sinc(&mut candidate, runtime, [self.batch, WINDOW], self.choice)?;
        }
        if !candidate.isolated_declared(self.batch, self.target) && self.choice != "Library" {
            return Ok(());
        }
        let isolated = |file: &SafetensorsFile| -> Result<Vec<f32>, CudaError> {
            if self.target != "lstm" {
                return read_batch(
                    file,
                    "tensor//sincnet/wav_norm1d/InstanceNormalization_output_0",
                    self.batch,
                );
            }
            let cf = read_batch(file, "tensor//sincnet/LeakyRelu_2_output_0", self.batch)?;
            let mut input = vec![0.0; cf.len()];
            for b in 0..self.batch {
                for t in 0..589 {
                    for c in 0..60 {
                        input[(b * 589 + t) * 60 + c] = cf[(b * 60 + c) * 589 + t];
                    }
                }
            }
            Ok(input)
        };
        let inputs = [isolated(self.files[0])?, isolated(self.files[1])?];
        let mut library_op =
            library.isolated(runtime, self.batch, self.target, [&inputs[0], &inputs[1]])?;
        let mut candidate_op =
            candidate.isolated(runtime, self.batch, self.target, [&inputs[0], &inputs[1]])?;
        let library_graphs = [
            capture(runtime, || {
                library.isolated_run(runtime, &mut library_op, 0)
            })?,
            capture(runtime, || {
                library.isolated_run(runtime, &mut library_op, 1)
            })?,
        ];
        let candidate_graphs = [
            capture(runtime, || {
                candidate.isolated_run(runtime, &mut candidate_op, 0)
            })?,
            capture(runtime, || {
                candidate.isolated_run(runtime, &mut candidate_op, 1)
            })?,
        ];
        let audio = [
            read_batch(self.files[0], "input/input", self.batch)?,
            read_batch(self.files[1], "input/input", self.batch)?,
        ];
        for model in [&mut library, &mut candidate] {
            model
                .workspace(runtime, self.batch, WINDOW)?
                .upload_input(runtime, &audio[0])?;
            model.forward(runtime, self.batch, WINDOW)?;
        }
        // graph objects are borrowed only after uploads; both sides use the same input
        let tail = if self.choice == "StageTail" && self.id == "fp32/mixed/b32" {
            Some(Tail::capture(
                runtime,
                library
                    .qualification_graph(self.batch)
                    .expect("Library graph"),
                1.4,
                || candidate.forward_eager(runtime, self.batch, WINDOW),
            )?)
        } else {
            None
        };
        let mut row = replay_set(BLOCKS, |which, block| {
            for model in [&mut library, &mut candidate] {
                model
                    .workspace(runtime, self.batch, WINDOW)?
                    .upload_input(runtime, &audio[which])?;
            }
            observe(
                runtime,
                [
                    library
                        .qualification_graph(self.batch)
                        .expect("Library graph"),
                    candidate
                        .qualification_graph(self.batch)
                        .expect("candidate graph"),
                ],
                [
                    std::slice::from_ref(&library_graphs),
                    std::slice::from_ref(&candidate_graphs),
                ],
                which,
                self.choice == "StageSlow",
                tail.as_ref().and_then(|tail| tail.for_block(block)),
            )
        })?;
        let mut hashes = [Vec::new(), Vec::new()];
        let mut op_hashes = [Vec::new(), Vec::new()];
        for which in 0..2 {
            for (side, model) in [&mut library, &mut candidate].into_iter().enumerate() {
                model
                    .workspace(runtime, self.batch, WINDOW)?
                    .upload_input(runtime, &audio[which])?;
                model
                    .qualification_graph(self.batch)
                    .expect("stage graph")
                    .launch()?;
                hashes[side].push(sha(&model
                    .find_workspace(self.batch, WINDOW)
                    .expect("workspace")
                    .download_output(runtime)?));
            }
            library_graphs[which].launch()?;
            op_hashes[0].push(sha(&library.isolated_output(runtime, &library_op)?));
            candidate_graphs[which].launch()?;
            op_hashes[1].push(sha(&candidate.isolated_output(runtime, &candidate_op)?));
        }
        let layer = if self.target == "lstm" {
            "lstm.stack"
        } else {
            "sincnet.conv0.abs_pool"
        };
        row["id"] = json!(self.key("stage", 0));
        if let Some(tail) = &tail {
            row["stage_tail"] = tail.evidence.clone();
            row["stage_tail"]["group_count"] = json!(BLOCKS / TAIL_BLOCKS);
        }
        row["output_sha256"] = json!(hashes);
        if let Some(tail) = &tail {
            let mut delayed = Vec::new();
            for audio in &audio {
                candidate
                    .workspace(runtime, self.batch, WINDOW)?
                    .upload_input(runtime, audio)?;
                tail.graph.launch()?;
                delayed.push(sha(&candidate
                    .find_workspace(self.batch, WINDOW)
                    .expect("workspace")
                    .download_output(runtime)?));
            }
            row["stage_tail"]["delayed_output_sha256"] = json!(delayed);
        }
        row["operator_outputs"] = json!([{"id":self.key(layer,0),"output_sha256":op_hashes}]);
        rows.push(row);
        Ok(())
    }
}

fn op_outputs(
    runtime: &CudaRuntime,
    library_graphs: &[[CudaGraph; 2]],
    candidate_graphs: &[[CudaGraph; 2]],
    library: &[Operator<'_>],
    candidate: &[Operator<'_>],
) -> Result<Value, CudaError> {
    let mut rows = Vec::new();
    for index in 0..candidate.len() {
        let mut hashes = [Vec::new(), Vec::new()];
        for which in 0..2 {
            library_graphs[index][which].launch()?;
            hashes[0].push(sha(&library[index].output(runtime)?));
            candidate_graphs[index][which].launch()?;
            hashes[1].push(sha(&candidate[index].output(runtime)?));
        }
        rows.push(json!({"layer":candidate[index].name(),"output_sha256":hashes}));
    }
    Ok(json!(rows))
}
