//! Bounded graph timing of approved choices on production model shapes

use cudarc::driver::sys::CUevent_flags;
use tracing::info;

use super::{BenchmarkMeasurement, BoundaryGraph, Catalogue, CudaTuneError, CudaTuneOptions};
use crate::inference::cuda::embedding::{
    EmbeddingBatch, MASK_FRAMES, ResNetEmbedding, SPEAKERS_PER_CHUNK,
};
use crate::inference::cuda::fbank::{
    CudaFbank, FBANK_FRAMES, FBANK_MEL_BINS, FBANK_WINDOW_SAMPLES, FbankBuffers,
};
use crate::inference::cuda::segmentation::{CudaSegmentation, SegmentationOptions};
use crate::inference::cuda::{CudaError, CudaRuntime, SafetensorsFile};

const WARMUP_PASSES: usize = 2;
const GRAPH_WARMUPS: usize = 3;
const ROUNDS: usize = 7;
const REPEATS: usize = 3;

#[derive(Clone, Copy)]
enum Stage {
    Segmentation,
    Fbank,
    Embedding,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Segmentation => "segmentation",
            Self::Fbank => "fbank",
            Self::Embedding => "embedding",
        }
    }

    fn batches(self) -> Vec<usize> {
        match self {
            Self::Fbank => (1..=32).collect(),
            Self::Segmentation => vec![1, 32],
            Self::Embedding => super::super::embedding::EmbeddingBatchClass::ALL
                .into_iter()
                .map(|class| class.chunks())
                .collect(),
        }
    }

    fn model(
        self,
        runtime: &CudaRuntime,
        options: &CudaTuneOptions,
        batch: usize,
        segmentation_weights: &SafetensorsFile,
        embedding_weights: &SafetensorsFile,
    ) -> Result<ModelState, CudaError> {
        match self {
            Self::Segmentation => {
                let mut model = CudaSegmentation::new(
                    runtime,
                    segmentation_weights,
                    SegmentationOptions {
                        math: options.segmentation_math,
                        #[cfg(feature = "_cuda-libraries")]
                        lstm_algo: Default::default(),
                        cuda_graph: false,
                    },
                )?;
                model
                    .workspace(runtime, batch, FBANK_WINDOW_SAMPLES)?
                    .upload_input(runtime, &waveform(batch))?;
                Ok(ModelState::Segmentation {
                    model: Box::new(model),
                    batch,
                })
            }
            Self::Fbank => {
                let model = CudaFbank::new(runtime, super::super::CudaMath::Fp32)?;
                let mut buffers = model.buffers(runtime, batch)?;
                let waveforms = waveform(batch);
                let rows: Vec<_> = waveforms
                    .as_chunks::<FBANK_WINDOW_SAMPLES>()
                    .0
                    .iter()
                    .map(|row| row.as_slice())
                    .collect();
                buffers.upload(runtime, &rows)?;
                Ok(ModelState::Fbank {
                    model: Box::new(model),
                    buffers: Box::new(buffers),
                })
            }
            Self::Embedding => {
                let model =
                    ResNetEmbedding::load(runtime, embedding_weights, options.embedding_math)?;
                // each captured class owns storage on its candidate runtime
                let activations = model.activations(runtime, batch)?;
                let mut model = model.batch_with_activations(runtime, batch, activations)?;
                let features: Vec<_> = (0..batch * FBANK_FRAMES * FBANK_MEL_BINS)
                    .map(|index| ((index % 157) as f32 - 78.0) / 80.0)
                    .collect();
                model
                    .fbank_mut()
                    .copy_from_host(runtime.stream(), &features)?;
                let masks: Vec<_> = (0..batch * SPEAKERS_PER_CHUNK * MASK_FRAMES)
                    .map(|index| 0.2 + (index % 13) as f32 / 20.0)
                    .collect();
                model.masks_mut().copy_from_host(runtime.stream(), &masks)?;
                Ok(ModelState::Embedding(Box::new(model)))
            }
        }
    }
}

/// Model weights, plans and buffers used by a set of captured boundary graphs
// models run eager passes only; no whole-pass graph can hide a boundary capture
enum ModelState {
    Segmentation {
        model: Box<CudaSegmentation>,
        batch: usize,
    },
    Fbank {
        model: Box<CudaFbank>,
        buffers: Box<FbankBuffers>,
    },
    Embedding(Box<EmbeddingBatch>),
}

impl ModelState {
    fn enqueue(&mut self, runtime: &CudaRuntime) -> Result<(), CudaError> {
        match self {
            Self::Segmentation { model, batch } => {
                model.enqueue_for_tuning(runtime, *batch, FBANK_WINDOW_SAMPLES)
            }
            Self::Fbank { model, buffers } => {
                model.compute_uploaded(runtime, buffers)?;
                Ok(())
            }
            Self::Embedding(model) => model.forward(runtime),
        }
    }
}

/// Drops graphs before their raw-pointer owners and waits before either can be freed
struct CapturedModel<'a> {
    graphs: Vec<BoundaryGraph>,
    model: ModelState,
    runtime: &'a CudaRuntime,
}

impl<'a> CapturedModel<'a> {
    fn capture(runtime: &'a CudaRuntime, model: ModelState) -> Result<Self, CudaError> {
        let mut captured = Self {
            graphs: Vec::new(),
            model,
            runtime,
        };
        runtime.context().bind_to_thread()?;
        for _ in 0..WARMUP_PASSES {
            captured.model.enqueue(runtime)?;
        }
        runtime.synchronize()?;

        runtime.begin_tune_capture()?;
        let enqueued = captured.model.enqueue(runtime);
        // drain even after an enqueue error so the runtime cannot retain graphs to
        // buffers which this function is about to drop
        captured.graphs = runtime.take_tune_graphs()?;
        enqueued?;
        runtime.synchronize()?;
        Ok(captured)
    }
}

impl Drop for CapturedModel<'_> {
    fn drop(&mut self) {
        // graph launches bypass cudarc's buffer events, so buffer drop cannot serve
        // as the only wait when a launch or timing call fails
        let _ = self.runtime.synchronize();
    }
}

struct TimedChoice<'a> {
    graph: &'a BoundaryGraph,
    runtime: &'a CudaRuntime,
    samples: [f64; ROUNDS],
}

impl TimedChoice<'_> {
    fn warmup(&self) -> Result<(), CudaError> {
        self.runtime.context().bind_to_thread()?;
        for _ in 0..GRAPH_WARMUPS {
            self.graph.graph.launch()?;
        }
        self.runtime.synchronize()
    }

    fn sample(&mut self, round: usize) -> Result<(), CudaError> {
        self.runtime.context().bind_to_thread()?;
        let context = self.runtime.context();
        let start = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = context.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        let stream = self.runtime.stream();
        start.record(stream)?;
        for _ in 0..REPEATS {
            self.graph.graph.launch()?;
        }
        end.record(stream)?;
        let elapsed = f64::from(start.elapsed_ms(&end)?) / REPEATS as f64;
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err(CudaError::Unsupported {
                context: "CUDA tuning timer",
                reason: format!("invalid elapsed time {elapsed}"),
            });
        }
        self.samples[round] = elapsed;
        Ok(())
    }

    fn measurement(mut self) -> BenchmarkMeasurement {
        self.samples.sort_by(f64::total_cmp);
        BenchmarkMeasurement {
            boundary: self.graph.boundary,
            batch: self.graph.batch,
            math: self.graph.math,
            choice: self.graph.choice.clone(),
            median_ms: self.samples[ROUNDS / 2],
        }
    }
}

/// Measures each distinct approved choice with alternating candidate order
fn measure(models: &[CapturedModel<'_>]) -> Result<Vec<BenchmarkMeasurement>, CudaTuneError> {
    let mut groups: Vec<Vec<TimedChoice<'_>>> = Vec::new();
    for model in models {
        for graph in &model.graphs {
            let index = groups.iter().position(|group| {
                let first = group[0].graph;
                (first.boundary, first.batch, first.math)
                    == (graph.boundary, graph.batch, graph.math)
            });
            let index = match index {
                Some(index) => index,
                None => {
                    groups.push(Vec::new());
                    groups.len() - 1
                }
            };
            let group = &mut groups[index];
            if group.iter().any(|timed| timed.graph.choice == graph.choice) {
                return Err(CudaTuneError::Invalid(
                    "duplicate candidate identity in captured workloads".into(),
                ));
            }
            group.push(TimedChoice {
                graph,
                runtime: model.runtime,
                samples: [0.0; ROUNDS],
            });
        }
    }

    let mut measurements = Vec::new();
    for mut group in groups {
        for timed in &group {
            timed.warmup()?;
        }
        for round in 0..ROUNDS {
            if round % 2 == 0 {
                for timed in &mut group {
                    timed.sample(round)?;
                }
            } else {
                for timed in group.iter_mut().rev() {
                    timed.sample(round)?;
                }
            }
        }
        measurements.extend(group.into_iter().map(TimedChoice::measurement));
    }
    Ok(measurements)
}

/// Uses real weights, fixed model batches and a bounded number of CUDA event samples
pub(super) fn run(
    options: &CudaTuneOptions,
    first_runtime: CudaRuntime,
    catalogue: &Catalogue,
) -> Result<Vec<BenchmarkMeasurement>, CudaTuneError> {
    let segmentation_weights =
        SafetensorsFile::open(options.model_dir.join("segmentation-3.0.safetensors"))?;
    let embedding_weights = SafetensorsFile::open(
        options
            .model_dir
            .join("wespeaker-multimask-tail.safetensors"),
    )?;
    let mut runtimes = vec![first_runtime];
    for kind in catalogue.benchmark_kinds(options.include_library).skip(1) {
        runtimes.push(CudaRuntime::for_tuning(
            options.device,
            kind,
            options.include_library,
        )?);
    }

    let mut measurements = Vec::new();
    for stage in [Stage::Segmentation, Stage::Fbank, Stage::Embedding] {
        for batch in stage.batches() {
            info!(stage = stage.name(), batch, "Timing approved CUDA choices");
            let mut models = Vec::new();
            for runtime in &runtimes {
                runtime
                    .context()
                    .bind_to_thread()
                    .map_err(CudaError::from)?;
                let model = stage.model(
                    runtime,
                    options,
                    batch,
                    &segmentation_weights,
                    &embedding_weights,
                )?;
                models.push(CapturedModel::capture(runtime, model)?);
            }
            measurements.extend(measure(&models)?);
            // this drops every graph before its model and releases each stage's
            // largest activations before the next batch is allocated
            drop(models);
        }
    }
    Ok(measurements)
}

fn waveform(batch: usize) -> Vec<f32> {
    (0..batch * FBANK_WINDOW_SAMPLES)
        .map(|index| {
            let sample = (index % FBANK_WINDOW_SAMPLES) as f32;
            0.2 * (sample * 0.071).sin() + 0.08 * (sample * 0.023).cos()
        })
        .collect()
}
