//! Independent collection paths for each segmentation and wide-convolution operation

use super::{
    BAND_SEEDS, BATCHES, Run, WINDOW, bursts, capture, metrics, paired, read_batch, record, sha,
    timing_row,
};
use crate::inference::cuda::candidate::{
    ConvCandidate, ConvInputs, ConvLayerSpec, DenseSite, DenseSpec, Epilogue, Phases, SegConvSite,
    SegConvSpec,
};
use crate::inference::cuda::dnn::Conv2d;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::segmentation::harness::{DenseLibrary, SegConvLibrary};
use crate::inference::cuda::test_support::{
    self,
    boundaries::Owner,
    candidate_seam::{Operation, Slices},
};
use crate::inference::cuda::{
    CudaError, CudaLstmAlgorithm, CudaMath, CudaRuntime, CudaSegmentation, EmbeddingBatch,
    ResNetEmbedding, SafetensorsFile, SegmentationOptions,
};
use cudarc::driver::{CudaGraph, CudaSlice};
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy)]
enum Kind {
    Dense(DenseSpec),
    Temporal(SegConvSpec),
    Spatial(Conv2d, Epilogue),
}

impl Kind {
    fn operation(self, name: &'static str) -> Operation {
        match self {
            Self::Dense(spec) => Operation::Dense(spec),
            Self::Temporal(spec) => Operation::Temporal(spec),
            Self::Spatial(conv, epilogue) => Operation::Spatial {
                boundary: BoundaryId::named(name),
                conv,
                epilogue,
            },
        }
    }
}

/// Fixed input, output and residual identities, not candidate declarations
#[derive(Clone)]
struct Path {
    name: &'static str,
    input: String,
    output: String,
    residual: Option<String>,
    kind: Kind,
}

pub(super) fn is_new(target: &str) -> bool {
    matches!(
        target,
        "wideconv"
            | "segdense-conv1"
            | "segdense-conv2"
            | "segdense-linear0"
            | "segdense-linear1"
            | "segdense-classifier"
            | "segdense-embedding"
    )
}
pub(super) fn embedding(target: &str) -> bool {
    matches!(target, "wideconv" | "segdense-embedding")
}
/// Candidate declarations for one collection path, from its registered typed factory
pub(super) fn declared_coverage(target: &str, tier: crate::inference::cuda::PtxTier) -> Value {
    let paths = paths(target, 1, CudaMath::Fp32).expect("fixed collection path");
    let family = paths[0].kind.operation(paths[0].name).family();
    let coverage = test_support::candidate_seam::coverage(family, tier);
    coverage_for_paths(&paths, coverage)
}

fn coverage_for_paths(
    paths: &[Path],
    coverage: crate::inference::cuda::candidate::Coverage,
) -> Value {
    let entries: Vec<_> = coverage
        .entries()
        .iter()
        .filter_map(|entry| {
            let names: Vec<_> = entry
                .layers
                .iter()
                .filter(|name| paths.iter().any(|path| path.name == **name))
                .copied()
                .collect();
            if names.is_empty() {
                return None;
            }
            let mut row = super::entry_json(entry);
            row["layers"] = json!(names);
            Some(row)
        })
        .collect();
    json!({"entries":entries})
}

/// Merge legacy owner pins and prepared candidate receipts without changing either
pub(super) fn configurations() -> Result<Value, CudaError> {
    let Value::Array(mut legacy) = test_support::configuration::planned() else {
        return Err(error("legacy configuration owner did not export an array"));
    };
    let Value::Array(prepared) = test_support::candidate_seam::planned() else {
        return Err(error(
            "prepared configuration owner did not export an array",
        ));
    };
    legacy.extend(prepared);
    Ok(Value::Array(legacy))
}

/// Preload the registered module outside every timed interval
pub(super) fn preload_candidates(
    runtime: &CudaRuntime,
    target: &str,
    cases: &[(CudaMath, &str, usize)],
) -> Result<(), CudaError> {
    for (math, _, batch) in cases {
        for path in paths(target, *batch, *math)? {
            test_support::candidate_seam::preload(runtime, path.kind.operation(path.name))?;
        }
    }
    Ok(())
}

fn error(error: impl std::fmt::Display) -> CudaError {
    CudaError::Unsupported {
        context: "new boundary collection",
        reason: error.to_string(),
    }
}

fn paths(target: &str, batch: usize, math: CudaMath) -> Result<Vec<Path>, CudaError> {
    if target != "wideconv" {
        let (name, input, output, kind) = match target {
            "segdense-conv1" => (
                "sincnet.conv1",
                "tensor//sincnet/LeakyRelu_output_0",
                "tensor//sincnet/conv1d.1/Conv_output_0",
                Kind::Temporal(SegConvSpec::new(SegConvSite::Conv1, batch, math).map_err(error)?),
            ),
            "segdense-conv2" => (
                "sincnet.conv2",
                "tensor//sincnet/LeakyRelu_1_output_0",
                "tensor//sincnet/conv1d.2/Conv_output_0",
                Kind::Temporal(SegConvSpec::new(SegConvSite::Conv2, batch, math).map_err(error)?),
            ),
            "segdense-linear0" => (
                "linear0",
                "tensor//lstm/Transpose_5_output_0",
                "tensor//LeakyRelu_output_0",
                Kind::Dense(DenseSpec::new(DenseSite::Linear0, batch, math).map_err(error)?),
            ),
            "segdense-linear1" => (
                "linear1",
                "tensor//LeakyRelu_output_0",
                "tensor//LeakyRelu_1_output_0",
                Kind::Dense(DenseSpec::new(DenseSite::Linear1, batch, math).map_err(error)?),
            ),
            "segdense-classifier" => (
                "linear2",
                "tensor//LeakyRelu_1_output_0",
                "tensor/output",
                Kind::Dense(DenseSpec::new(DenseSite::Classifier, batch, math).map_err(error)?),
            ),
            "segdense-embedding" => (
                "resnet.seg_1",
                "tensor/where_1",
                "tensor/output",
                Kind::Dense(DenseSpec::new(DenseSite::Embedding, batch, math).map_err(error)?),
            ),
            _ => unreachable!("fixed new collection"),
        };
        return Ok(vec![Path {
            name,
            input: input.into(),
            output: output.into(),
            residual: None,
            kind,
        }]);
    }
    let mut sites = vec![("resnet.conv1".to_owned(), 0, false, false, 1, 32, [80, 998])];
    for (stage, count, base, channels, input) in
        [(3, 6, 7, 128, [40, 499]), (4, 3, 13, 256, [20, 250])]
    {
        let out = [input[0] / 2, usize::div_ceil(input[1], 2)];
        for block in 0..count {
            for second in [false, true] {
                let down = block == 0 && !second;
                sites.push((
                    format!(
                        "resnet.layer{stage}.{block}.conv{}",
                        if second { 2 } else { 1 }
                    ),
                    base + block,
                    second,
                    false,
                    if down { channels / 2 } else { channels },
                    channels,
                    if down { input } else { out },
                ));
            }
        }
    }
    for (stage, block, cin, cout, input) in [
        (2, 3, 32, 64, [80, 998]),
        (3, 7, 64, 128, [40, 499]),
        (4, 13, 128, 256, [20, 250]),
    ] {
        sites.push((
            format!("resnet.layer{stage}.0.shortcut.0"),
            block,
            false,
            true,
            cin,
            cout,
            input,
        ));
    }
    Ok(sites
        .into_iter()
        .map(|(name, block, second, shortcut, cin, cout, input)| {
            let down = shortcut || (cin != cout && cin != 1);
            let input_name = if cin == 1 {
                "tensor/unsqueeze".into()
            } else {
                format!("tensor/relu_{}", 2 * block + usize::from(second))
            };
            let residual = second.then(|| {
                if block == 7 {
                    "tensor/getitem_54".into()
                } else if block == 13 {
                    "tensor/getitem_93".into()
                } else {
                    format!("tensor/relu_{}", 2 * block)
                }
            });
            let output = if shortcut {
                match block {
                    3 => "tensor/getitem_27".into(),
                    7 => "tensor/getitem_54".into(),
                    13 => "tensor/getitem_93".into(),
                    _ => unreachable!(),
                }
            } else if cin == 1 {
                "tensor/relu".into()
            } else {
                format!("tensor/relu_{}", 2 * block + if second { 2 } else { 1 })
            };
            let epilogue = if shortcut {
                Epilogue::Bias
            } else if second {
                Epilogue::BiasReluResidual
            } else {
                Epilogue::BiasRelu
            };
            Path {
                name: BoundaryId::parse(&name)
                    .expect("fixed model boundary")
                    .name(),
                input: input_name,
                output,
                residual,
                kind: Kind::Spatial(
                    Conv2d {
                        batch,
                        in_channels: cin,
                        out_channels: cout,
                        input,
                        kernel: [if shortcut { 1 } else { 3 }; 2],
                        padding: [usize::from(!shortcut); 2],
                        stride: [if down { 2 } else { 1 }; 2],
                        dilation: [1, 1],
                        math,
                    },
                    epilogue,
                ),
            }
        })
        .collect())
}

#[derive(Clone)]
struct Host {
    input: Vec<f32>,
    residual: Option<Vec<f32>>,
}
struct Inputs {
    input: CudaSlice<f32>,
    residual: Option<CudaSlice<f32>>,
}
enum LibraryPlan {
    Dense(DenseLibrary),
    Temporal(SegConvLibrary),
    Spatial(crate::inference::cuda::embedding::wideconv_library::Library),
}
struct Operator {
    path: Path,
    plan: LibraryPlan,
    owner: Owner,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    host: [Host; 2],
    inputs: [Inputs; 2],
    output: CudaSlice<f32>,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

fn weights(
    file: &SafetensorsFile,
    target: &str,
    path: &Path,
) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
    if target == "wideconv" {
        let Kind::Spatial(spec, _) = path.kind else {
            unreachable!()
        };
        return Ok((
            file.read_f32(&format!("{}.weight", path.name), &spec.filter_shape())?,
            file.read_f32(&format!("{}.weight_bias", path.name), &[spec.out_channels])?,
        ));
    }
    if target == "segdense-embedding" {
        return Ok((
            file.read_f32("resnet.seg_1.weight", &[256, 5120])?,
            file.read_f32("resnet.seg_1.bias", &[256])?,
        ));
    }
    crate::inference::cuda::segmentation::harness::weights(
        file,
        target.trim_start_matches("segdense-"),
    )
}

impl Operator {
    fn fixture(file: &SafetensorsFile, path: &Path, batch: usize) -> Result<Host, CudaError> {
        Ok(Host {
            input: read_batch(file, &path.input, batch)?,
            residual: path
                .residual
                .as_ref()
                .map(|name| read_batch(file, name, batch))
                .transpose()?,
        })
    }
    fn new(
        runtime: &CudaRuntime,
        target: &str,
        path: Path,
        choice: &str,
        host: [Host; 2],
        source: &SafetensorsFile,
    ) -> Result<Self, CudaError> {
        let _scope = test_support::plan(path.name);
        let (weights, biases) = weights(source, target, &path)?;
        let weight = runtime.stream().clone_htod(&weights)?;
        let bias = runtime.stream().clone_htod(if biases.is_empty() {
            &[0.0][..]
        } else {
            &biases[..]
        })?;
        let (plan, len, batch) = match path.kind {
            Kind::Dense(spec) => (
                LibraryPlan::Dense(DenseLibrary::new(runtime, spec).map_err(error)?),
                spec.output_len(),
                spec.batch(),
            ),
            Kind::Temporal(spec) => (
                LibraryPlan::Temporal(SegConvLibrary::new(runtime, spec).map_err(error)?),
                spec.output_len(),
                spec.batch(),
            ),
            Kind::Spatial(spec, epilogue) => (
                LibraryPlan::Spatial(
                    crate::inference::cuda::embedding::wideconv_library::Library::new(
                        runtime,
                        ConvLayerSpec {
                            name: path.name,
                            conv: spec,
                            epilogue,
                            weight: &weight,
                            bias: &bias,
                        },
                    )
                    .map_err(error)?,
                ),
                spec.output_shape().iter().product(),
                spec.batch,
            ),
        };
        let inputs = host.each_ref().map(|host| -> Result<_, CudaError> {
            Ok(Inputs {
                input: runtime.stream().clone_htod(&host.input)?,
                residual: host
                    .residual
                    .as_ref()
                    .map(|r| runtime.stream().clone_htod(r))
                    .transpose()?,
            })
        });
        let [a, b] = inputs;
        let owner = Owner::new(runtime, path.name, batch, "Library", || unreachable!())?;
        let mut op = Self {
            path,
            plan,
            owner,
            weight,
            bias,
            host,
            inputs: [a?, b?],
            output: runtime.stream().alloc_zeros(len)?,
            weights,
            biases,
        };
        // finish vendor first-call setup on the control route, before any graph capture
        op.run(runtime, 0)?;
        runtime.synchronize()?;
        if choice == "Lookup" {
            // compute only pinned fixture answers, with the original fixture weights
            let file = super::reference(target, "mixed")?;
            let fixed = Self::fixture(&file, &op.path, batch)?;
            let fixed_weights = SafetensorsFile::open(model_path(target))?;
            let mut fixture = Self::new(
                runtime,
                target,
                op.path.clone(),
                "Library",
                [fixed.clone(), fixed],
                &fixed_weights,
            )?;
            fixture.run(runtime, 0)?;
            let cached = fixture.output(runtime)?;
            op.owner = Owner::prepare(
                runtime,
                op.path.kind.operation(op.path.name),
                choice,
                &op.weight,
                &op.bias,
                || Ok(cached),
            )?;
        } else {
            op.owner = Owner::prepare(
                runtime,
                op.path.kind.operation(op.path.name),
                choice,
                &op.weight,
                &op.bias,
                || unreachable!(),
            )?;
        }
        Ok(op)
    }
    fn restore(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        runtime
            .stream()
            .memcpy_htod(&self.host[which].input, &mut self.inputs[which].input)?;
        Ok(())
    }
    fn run(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        let inputs = &self.inputs[which];
        self.owner.run_slices(
            runtime,
            self.path.kind.operation(self.path.name),
            Slices {
                input: &inputs.input,
                weight: &self.weight,
                bias: (!matches!(self.path.kind, Kind::Temporal(_))).then_some(&self.bias),
                residual: inputs.residual.as_ref(),
            },
            &mut self.output,
            |output| match &self.plan {
                LibraryPlan::Dense(plan) => plan.enqueue(
                    &inputs.input,
                    &self.weight,
                    &self.bias,
                    output,
                    &Phases::new(),
                    runtime,
                ),
                LibraryPlan::Temporal(plan) => {
                    plan.enqueue(&inputs.input, &self.weight, output, &Phases::new(), runtime)
                }
                LibraryPlan::Spatial(plan) => plan.enqueue(
                    ConvInputs {
                        x: &inputs.input.as_view(),
                        weight: &self.weight.as_view(),
                        bias: &self.bias.as_view(),
                        residual: inputs.residual.as_ref().map(CudaSlice::as_view).as_ref(),
                    },
                    &mut output.as_view_mut(),
                    &Phases::new(),
                    runtime.stream(),
                ),
            },
        )
    }
    fn output(&self, runtime: &CudaRuntime) -> Result<Vec<f32>, CudaError> {
        Ok(runtime.stream().clone_dtoh(&self.output)?)
    }
    fn graphs(&mut self, runtime: &CudaRuntime) -> Result<[CudaGraph; 2], CudaError> {
        self.restore(runtime, 0)?;
        self.restore(runtime, 1)?;
        Ok([
            capture(runtime, || self.run(runtime, 0))?,
            capture(runtime, || self.run(runtime, 1))?,
        ])
    }
}
fn model_path(target: &str) -> &'static str {
    if embedding(target) {
        "/workspace/models-native/wespeaker-multimask-tail.safetensors"
    } else {
        "/workspace/models-native/segmentation-3.0.safetensors"
    }
}

enum Stage {
    Segmentation(Box<CudaSegmentation>),
    Embedding(Box<EmbeddingBatch>),
}
impl Stage {
    fn new(
        runtime: &CudaRuntime,
        target: &str,
        batch: usize,
        math: CudaMath,
        choice: &str,
    ) -> Result<Self, CudaError> {
        let weights = SafetensorsFile::open(model_path(target))?;
        let fixture = super::reference(target, "mixed")?;
        let mut routes = Vec::new();
        for path in paths(target, batch, math)? {
            let host = Operator::fixture(&fixture, &path, batch)?;
            let mut op = Operator::new(
                runtime,
                target,
                path.clone(),
                "Library",
                [host.clone(), host],
                &weights,
            )?;
            op.run(runtime, 0)?;
            routes.push((path.kind.operation(path.name), op.output(runtime)?));
        }
        if embedding(target) {
            let model = ResNetEmbedding::load(runtime, &weights, math)?;
            let mut buffers = model.batch(runtime, batch)?;
            for (operation, fixture) in routes {
                buffers.install_boundary(runtime, operation, choice, fixture)?;
            }
            return Ok(Self::Embedding(Box::new(buffers)));
        }
        let mut model = CudaSegmentation::new(
            runtime,
            &weights,
            SegmentationOptions {
                math,
                lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
                cuda_graph: false,
            },
        )?;
        model.workspace(runtime, batch, WINDOW)?;
        for (operation, fixture) in routes {
            model.install_boundary(runtime, batch, operation, choice, fixture)?;
        }
        Ok(Self::Segmentation(Box::new(model)))
    }
    fn upload(
        &mut self,
        runtime: &CudaRuntime,
        file: &SafetensorsFile,
        batch: usize,
    ) -> Result<(), CudaError> {
        match self {
            Self::Segmentation(model) => model
                .workspace(runtime, batch, WINDOW)?
                .upload_input(runtime, &read_batch(file, "input/input", batch)?),
            Self::Embedding(buffers) => {
                buffers
                    .fbank_mut()
                    .copy_from_host(runtime.stream(), &read_batch(file, "input/fbank", batch)?)?;
                buffers
                    .masks_mut()
                    .copy_from_host(runtime.stream(), &read_batch(file, "input/masks", batch)?)
            }
        }
    }
    fn run(&mut self, runtime: &CudaRuntime, batch: usize) -> Result<(), CudaError> {
        match self {
            Self::Segmentation(model) => model.forward_eager(runtime, batch, WINDOW),
            Self::Embedding(buffers) => buffers.forward_with_taps(runtime, &mut |_, _| Ok(())),
        }
    }
    fn output(&self, runtime: &CudaRuntime, batch: usize) -> Result<Vec<f32>, CudaError> {
        match self {
            Self::Segmentation(model) => model
                .find_workspace(batch, WINDOW)
                .expect("prepared stage")
                .download_output(runtime),
            Self::Embedding(buffers) => buffers.download_output(runtime),
        }
    }
    fn graphs(
        &mut self,
        runtime: &CudaRuntime,
        files: [&SafetensorsFile; 2],
        batch: usize,
    ) -> Result<[CudaGraph; 2], CudaError> {
        self.upload(runtime, files[0], batch)?;
        self.run(runtime, batch)?;
        runtime.synchronize()?;
        self.upload(runtime, files[0], batch)?;
        let first = capture(runtime, || self.run(runtime, batch))?;
        self.upload(runtime, files[1], batch)?;
        let second = capture(runtime, || self.run(runtime, batch))?;
        Ok([first, second])
    }
    fn round(&self, runtime: &CudaRuntime, batch: usize) -> Result<(), CudaError> {
        match self {
            Self::Segmentation(model) => model.qualification_round_stage(runtime, batch),
            Self::Embedding(buffers) => buffers.qualification_round_stage(runtime),
        }
    }
}

impl Run<'_> {
    pub(super) fn new_boundaries(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let weights = SafetensorsFile::open(model_path(self.target))?;
        if self.phase == "paired" {
            let mut stage = Stage::new(
                self.runtime,
                self.target,
                self.batch,
                self.math,
                self.choice,
            )?;
            return self.paired_new(rows, &mut stage);
        }
        for path in paths(self.target, self.batch, self.math)? {
            let host = [
                Operator::fixture(self.files[0], &path, self.batch)?,
                Operator::fixture(self.files[1], &path, self.batch)?,
            ];
            let mut op = Operator::new(
                self.runtime,
                self.target,
                path.clone(),
                self.choice,
                host,
                &weights,
            )?;
            self.collect_operator(rows, &mut op)?;
            // release each isolated operator before the full stage is allocated
        }
        if self.phase == "sanitize" {
            return Ok(());
        }
        let mut stage = Stage::new(
            self.runtime,
            self.target,
            self.batch,
            self.math,
            self.choice,
        )?;
        self.collect_stage(rows, &mut stage)
    }
    fn collect_operator(&self, rows: &mut Vec<Value>, op: &mut Operator) -> Result<(), CudaError> {
        let name = op.path.name;
        let declared = op.owner.declared();
        if self.phase == "profile" {
            for which in 0..2 {
                op.restore(self.runtime, which)?;
                super::profile_case(
                    self.runtime,
                    &self.key(name, which),
                    self.choice,
                    name,
                    || op.run(self.runtime, which),
                )?;
                op.output(self.runtime)?;
            }
            return Ok(());
        }
        let graphs = op.graphs(self.runtime)?;
        if self.phase == "timing" {
            let cell = RefCell::new(&mut *op);
            let timing = bursts(
                self.runtime,
                |sample| cell.borrow_mut().restore(self.runtime, sample % 2),
                |launch| Ok(graphs[launch % 2].launch()?),
            )?;
            let mut outputs = Vec::new();
            for (which, graph) in graphs.iter().enumerate() {
                cell.borrow_mut().restore(self.runtime, which)?;
                graph.launch()?;
                outputs.push(cell.borrow().output(self.runtime)?);
            }
            rows.push(timing_row(
                &self.key(name, 0),
                &timing,
                [&outputs[0], &outputs[1]],
                declared,
            ));
            return Ok(());
        }
        if self.phase == "sanitize" {
            for graph in &graphs {
                graph.launch()?;
            }
            self.runtime.synchronize()?;
            rows.push(json!({"id":self.key(name,0),"sanitized":true,"declared":declared}));
            println!("sanitized {}", self.key(name, 0));
            return Ok(());
        }
        for (which, graph) in graphs.iter().enumerate() {
            op.restore(self.runtime, which)?;
            graph.launch()?;
            let first = op.output(self.runtime)?;
            op.restore(self.runtime, which)?;
            graph.launch()?;
            let second = op.output(self.runtime)?;
            let mut expected = read_batch(self.files[which], &op.path.output, self.batch)?;
            if let Kind::Temporal(spec) = op.path.kind {
                // ONNX Conv includes bias; the qualified producer leaves it to the pool consumer
                let model = SafetensorsFile::open(model_path(self.target))?;
                let suffix = if spec.site() == SegConvSite::Conv1 {
                    1
                } else {
                    2
                };
                let bias = model.read_f32(&format!("sincnet.conv1d.{suffix}.bias"), &[60])?;
                for (index, value) in expected.iter_mut().enumerate() {
                    *value -= bias[index / spec.output_steps() % 60];
                }
            }
            record(
                rows,
                &self.key(name, which),
                first,
                second,
                &expected,
                1,
                declared,
            );
        }
        Ok(())
    }
    fn collect_stage(&self, rows: &mut Vec<Value>, stage: &mut Stage) -> Result<(), CudaError> {
        let width = if embedding(self.target) { 256 } else { 7 };
        let declared = self.choice != "Library";
        let layer = paths(self.target, self.batch, self.math)?[0].name;
        if self.phase == "profile" {
            for which in 0..2 {
                stage.upload(self.runtime, self.files[which], self.batch)?;
                super::profile_case(
                    self.runtime,
                    &self.key("stage", which),
                    self.choice,
                    layer,
                    || stage.run(self.runtime, self.batch),
                )?;
                stage.output(self.runtime, self.batch)?;
            }
            return Ok(());
        }
        let graphs = stage.graphs(self.runtime, self.files, self.batch)?;
        if self.phase == "timing" {
            let cell = RefCell::new(&mut *stage);
            let selected = Cell::new(0);
            let timing = bursts(
                self.runtime,
                |sample| {
                    selected.set(sample % 2);
                    cell.borrow_mut()
                        .upload(self.runtime, self.files[selected.get()], self.batch)
                },
                |_| Ok(graphs[selected.get()].launch()?),
            )?;
            let mut outputs = Vec::new();
            for (which, graph) in graphs.iter().enumerate() {
                cell.borrow_mut()
                    .upload(self.runtime, self.files[which], self.batch)?;
                graph.launch()?;
                outputs.push(cell.borrow().output(self.runtime, self.batch)?);
            }
            rows.push(timing_row(
                &self.key("stage", 0),
                &timing,
                [&outputs[0], &outputs[1]],
                declared,
            ));
            return Ok(());
        }
        let mut truth = if self.math == CudaMath::Tf32 {
            Some(Stage::new(
                self.runtime,
                self.target,
                self.batch,
                CudaMath::Fp32,
                "Library",
            )?)
        } else {
            None
        };
        for (which, graph) in graphs.iter().enumerate() {
            let expected = read_batch(self.files[which], "tensor/output", self.batch)?;
            let truth = truth
                .as_mut()
                .map(|truth| -> Result<_, CudaError> {
                    truth.upload(self.runtime, self.files[which], self.batch)?;
                    truth.run(self.runtime, self.batch)?;
                    truth.output(self.runtime, self.batch)
                })
                .transpose()?;
            stage.upload(self.runtime, self.files[which], self.batch)?;
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                stage.round(self.runtime, self.batch)?;
            }
            let first = stage.output(self.runtime, self.batch)?;
            let truth_metrics = truth.as_ref().map(|truth| metrics(&first, truth, width));
            stage.upload(self.runtime, self.files[which], self.batch)?;
            graph.launch()?;
            if self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                stage.round(self.runtime, self.batch)?;
            }
            let second = stage.output(self.runtime, self.batch)?;
            let key = self.key("stage", which);
            record(rows, &key, first, second, &expected, width, declared);
            rows.last_mut().expect("stage row")["truth"] = json!(truth_metrics);
            rows.last_mut().expect("stage row")["truth_sha256"] =
                json!(truth.as_ref().map(|values| sha(values)));
            if let Some(layers) = &self.band {
                let mut band = Vec::new();
                let mut draws = Vec::new();
                for seed in BAND_SEEDS {
                    test_support::set_band(Some((key.clone(), seed, layers.clone())));
                    stage.upload(self.runtime, self.files[which], self.batch)?;
                    stage.run(self.runtime, self.batch)?;
                    test_support::set_band(None);
                    let output = stage.output(self.runtime, self.batch)?;
                    band.push(metrics(&output, &expected, width));
                    draws.push(metrics(&output, truth.as_ref().expect("TF32 truth"), width));
                }
                rows.push(json!({"id":format!("{key}/band"),"seeds":BAND_SEEDS,"layers":layers,"metrics":band,"truth_draws":draws,"truth_sha256":sha(truth.as_ref().expect("TF32 truth"))}));
            }
        }
        Ok(())
    }
    fn paired_new(
        &self,
        rows: &mut Vec<Value>,
        candidate_stage: &mut Stage,
    ) -> Result<(), CudaError> {
        let source = SafetensorsFile::open(model_path(self.target))?;
        let mut library_stage =
            Stage::new(self.runtime, self.target, self.batch, self.math, "Library")?;
        let stages = [
            library_stage.graphs(self.runtime, self.files, self.batch)?,
            candidate_stage.graphs(self.runtime, self.files, self.batch)?,
        ];
        let paths = paths(self.target, self.batch, self.math)?;
        let stratified = paths.len() > 1;
        // preserve the existing equal-stratum design with one live operator pair
        let blocks_per_layer = if stratified {
            paired::BLOCKS.div_ceil(16 * paths.len()) * 16
        } else {
            paired::BLOCKS
        };
        let mut stage_blocks = Vec::new();
        let mut operator_blocks = Vec::new();
        let mut operator_layers = Vec::new();
        let mut block_layers = Vec::new();
        let mut outputs = Vec::new();
        for path in paths {
            let host = [
                Operator::fixture(self.files[0], &path, self.batch)?,
                Operator::fixture(self.files[1], &path, self.batch)?,
            ];
            let name = path.name;
            let mut control = Operator::new(
                self.runtime,
                self.target,
                path.clone(),
                "Library",
                host.clone(),
                &source,
            )?;
            let mut candidate =
                Operator::new(self.runtime, self.target, path, self.choice, host, &source)?;
            let operator_graphs = [
                control.graphs(self.runtime)?,
                candidate.graphs(self.runtime)?,
            ];
            let collected = paired::replay_set(blocks_per_layer, |which, _| {
                // uploads stay outside the CUDA event intervals on both sides
                library_stage.upload(self.runtime, self.files[which], self.batch)?;
                candidate_stage.upload(self.runtime, self.files[which], self.batch)?;
                paired::observe(
                    self.runtime,
                    [&stages[0][which], &stages[1][which]],
                    [
                        std::slice::from_ref(&operator_graphs[0]),
                        std::slice::from_ref(&operator_graphs[1]),
                    ],
                    which,
                    self.choice == "StageSlow",
                    None,
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
            operator_layers.push(name);
            block_layers.extend(std::iter::repeat_n(name, blocks_per_layer));
            let mut hashes = Vec::new();
            for (side, op) in [&mut control, &mut candidate].into_iter().enumerate() {
                let mut pair = Vec::new();
                for (which, graph) in operator_graphs[side].iter().enumerate() {
                    op.restore(self.runtime, which)?;
                    graph.launch()?;
                    pair.push(sha(&op.output(self.runtime)?));
                }
                hashes.push(pair);
            }
            outputs.push(json!({"layer":name,"output_sha256":hashes}));
        }
        let mut row = json!({
            "order":"ABBA", "warmup":super::WARMUP,
            "stage_abba_ms":stage_blocks, "operator_abba_ms":operator_blocks,
            "cuda_graph":true, "declared":true, "pid":std::process::id(),
        });
        if stratified {
            row["warmup_per_operator"] = json!(super::WARMUP);
            row["operator_layers"] = json!(operator_layers);
            row["operator_layer_by_block"] = json!(block_layers);
        }
        let mut stage_hashes = Vec::new();
        for (side, stage) in [&mut library_stage, candidate_stage]
            .into_iter()
            .enumerate()
        {
            let mut hashes = Vec::new();
            for (which, graph) in stages[side].iter().enumerate() {
                stage.upload(self.runtime, self.files[which], self.batch)?;
                graph.launch()?;
                hashes.push(sha(&stage.output(self.runtime, self.batch)?));
            }
            stage_hashes.push(hashes);
        }
        row["id"] = json!(self.key("stage", 0));
        row["output_sha256"] = json!(stage_hashes);
        row["operator_outputs"] = json!(outputs);
        rows.push(row);
        Ok(())
    }
}

fn sample_shape(kind: Kind) -> Vec<usize> {
    match kind {
        Kind::Dense(spec) => {
            let (m, n, _) = spec.dimensions();
            vec![spec.batch() * m, n]
        }
        Kind::Temporal(spec) => vec![spec.batch(), spec.out_channels(), 1, spec.output_steps()],
        Kind::Spatial(spec, _) => spec.output_shape().to_vec(),
    }
}

fn dense_value(spec: DenseSpec, input: &Host, weight: &[f32], bias: &[f32], index: usize) -> f64 {
    let (_, n, k) = spec.dimensions();
    let row = index / n;
    let column = index % n;
    let dot = |col: usize| {
        let mut sum = 0f64;
        for term in 0..k {
            let w = if spec.transposed_weights() {
                col * k + term
            } else {
                term * n + col
            };
            sum += f64::from(input.input[row * k + term]) * f64::from(weight[w]);
        }
        sum + f64::from(bias[col])
    };
    let value = dot(column);
    match spec.site() {
        DenseSite::Linear0 | DenseSite::Linear1 => {
            if value < 0.0 {
                value * 0.01
            } else {
                value
            }
        }
        DenseSite::Embedding => value,
        DenseSite::Classifier => {
            let logits: Vec<_> = (0..n).map(dot).collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            value - max - logits.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
        }
    }
}

fn temporal_value(spec: SegConvSpec, input: &Host, weight: &[f32], index: usize) -> f64 {
    let step = index % spec.output_steps();
    let channel = index / spec.output_steps() % 60;
    let batch = index / (spec.output_steps() * 60);
    let mut sum = 0f64;
    for cin in 0..spec.in_channels() {
        for tap in 0..5 {
            let i = (batch * spec.in_channels() + cin) * spec.input_steps() + step + tap;
            let w = (channel * spec.in_channels() + cin) * 5 + tap;
            sum += f64::from(input.input[i]) * f64::from(weight[w]);
        }
    }
    sum
}

fn evaluate(
    path: &Path,
    input: &Host,
    weight: &[f32],
    bias: &[f32],
    state: &mut u64,
) -> super::reference::Sample {
    let shape = sample_shape(path.kind);
    let indices = super::reference::indices(&shape, state);
    super::cpu::evaluate(|mode| {
        let values = super::cpu::ordered_map(mode, indices.len(), |position| {
            let index = indices[position];
            match path.kind {
                Kind::Dense(spec) => dense_value(spec, input, weight, bias, index),
                Kind::Temporal(spec) => temporal_value(spec, input, weight, index),
                Kind::Spatial(spec, epilogue) => {
                    let [oh, ow] = spec.output();
                    let [ih, iw] = spec.input;
                    let x = index % ow;
                    let y = index / ow % oh;
                    let channel = index / (ow * oh) % spec.out_channels;
                    let batch = index / (ow * oh * spec.out_channels);
                    let mut sum = 0f64;
                    for cin in 0..spec.in_channels {
                        for ky in 0..spec.kernel[0] {
                            let yy = (y * spec.stride[0] + ky) as isize - spec.padding[0] as isize;
                            if !(0..ih as isize).contains(&yy) {
                                continue;
                            }
                            for kx in 0..spec.kernel[1] {
                                let xx =
                                    (x * spec.stride[1] + kx) as isize - spec.padding[1] as isize;
                                if !(0..iw as isize).contains(&xx) {
                                    continue;
                                }
                                let i = ((batch * spec.in_channels + cin) * ih + yy as usize) * iw
                                    + xx as usize;
                                let w = ((channel * spec.in_channels + cin) * spec.kernel[0] + ky)
                                    * spec.kernel[1]
                                    + kx;
                                sum += f64::from(input.input[i]) * f64::from(weight[w]);
                            }
                        }
                    }
                    sum += f64::from(bias[channel]);
                    if let Some(residual) = &input.residual {
                        sum += f64::from(residual[index]);
                    }
                    if epilogue == Epilogue::Bias {
                        sum
                    } else {
                        sum.max(0.0)
                    }
                }
            }
        });
        super::reference::Sample {
            indices: indices.clone(),
            values,
        }
    })
}

/// Use the same transformed-audio rule as existing collection paths, then run a Library front end
fn secret_inputs(
    runtime: &CudaRuntime,
    target: &str,
    source: &SafetensorsFile,
    audio: &[f32],
    batch: usize,
    paths: &[Path],
) -> Result<Vec<Host>, CudaError> {
    if !embedding(target) {
        use crate::inference::cuda::segmentation::SegmentationTensor;
        let mut model = CudaSegmentation::new(
            runtime,
            source,
            SegmentationOptions {
                math: CudaMath::Fp32,
                lstm_algo: CudaLstmAlgorithm::PersistStaticSmallH,
                cuda_graph: false,
            },
        )?;
        model
            .workspace(runtime, batch, WINDOW)?
            .upload_input(runtime, audio)?;
        model.forward_eager(runtime, batch, WINDOW)?;
        let tensor = match target {
            "segdense-conv1" => SegmentationTensor::Stage0,
            "segdense-conv2" => SegmentationTensor::Stage1,
            "segdense-linear0" => SegmentationTensor::LstmOutput,
            "segdense-linear1" => SegmentationTensor::Linear0,
            "segdense-classifier" => SegmentationTensor::Linear1,
            _ => unreachable!(),
        };
        return Ok(vec![Host {
            input: model
                .find_workspace(batch, WINDOW)
                .expect("secret front end")
                .tensor(tensor)
                .download(runtime.stream())?,
            residual: None,
        }]);
    }
    use crate::inference::cuda::{CudaFbank, embedding::EmbeddingTap};
    let fbank = CudaFbank::new(runtime, CudaMath::Fp32)?;
    let mut buffers = fbank.buffers(runtime, batch)?;
    let waveforms: Vec<_> = audio
        .as_chunks::<WINDOW>()
        .0
        .iter()
        .map(<[f32; WINDOW]>::as_slice)
        .collect();
    let features =
        runtime
            .stream()
            .clone_dtoh(&fbank.compute_host(runtime, &waveforms, &mut buffers)?)?;
    let model = ResNetEmbedding::load(runtime, source, CudaMath::Fp32)?;
    let mut front = model.batch(runtime, batch)?;
    front
        .fbank_mut()
        .copy_from_host(runtime.stream(), &features)?;
    front
        .masks_mut()
        .copy_from_host(runtime.stream(), &vec![1.0; batch * 3 * 589])?;
    let mut inputs = std::collections::BTreeMap::new();
    let mut stem = vec![0f32; features.len()];
    for b in 0..batch {
        for t in 0..998 {
            for c in 0..80 {
                stem[(b * 80 + c) * 998 + t] = features[(b * 998 + t) * 80 + c];
            }
        }
    }
    inputs.insert("tensor/unsqueeze".to_owned(), stem);
    front.forward_with_taps(runtime, &mut |tap, values| {
        let name = match tap {
            EmbeddingTap::Stem => "tensor/relu".to_owned(),
            EmbeddingTap::Hidden { block } => format!("tensor/relu_{}", 2 * block + 1),
            EmbeddingTap::Block { block } => format!("tensor/relu_{}", 2 * block + 2),
            EmbeddingTap::Shortcut { block } => match block {
                3 => "tensor/getitem_27".into(),
                7 => "tensor/getitem_54".into(),
                13 => "tensor/getitem_93".into(),
                _ => unreachable!(),
            },
            EmbeddingTap::Pooled => "tensor/where_1".into(),
            _ => return Ok(()),
        };
        if paths
            .iter()
            .any(|path| path.input == name || path.residual.as_ref() == Some(&name))
        {
            inputs.insert(name, runtime.stream().clone_dtoh(values)?);
        }
        Ok(())
    })?;
    Ok(paths
        .iter()
        .map(|path| Host {
            input: inputs
                .get(&path.input)
                .expect("secret boundary input tap")
                .clone(),
            residual: path
                .residual
                .as_ref()
                .map(|name| inputs.get(name).expect("secret residual tap").clone()),
        })
        .collect())
}

pub(super) fn secret(
    runtime: &CudaRuntime,
    target: &str,
    choice: &str,
    math: CudaMath,
    rows: &mut Vec<Value>,
) -> Result<(), CudaError> {
    let seed = test_support::secret_seed();
    let mut state = seed;
    let source = SafetensorsFile::open(model_path(target))?.perturbed(seed, 0.01);
    let fixture_file = if embedding(target) {
        super::reference("fbankdft", "mixed")?
    } else {
        super::reference(target, "mixed")?
    };
    let audio = if embedding(target) {
        fixture_file.read_f32("input/waveform", &[32, 1, WINDOW])?
    } else {
        read_batch(&fixture_file, "input/input", 32)?
    };
    for batch in BATCHES {
        let paths = paths(target, batch, math)?;
        let waveform = super::transformed_audio(&audio, batch, &mut state);
        let inputs = secret_inputs(runtime, target, &source, &waveform, batch, &paths)?;
        for (path, input) in paths.into_iter().zip(inputs) {
            if test_support::phase() == Some("profile") {
                let name = path.name;
                let mut candidate = Operator::new(
                    runtime,
                    target,
                    path,
                    choice,
                    [input.clone(), input],
                    &source,
                )?;
                {
                    let _window = test_support::window(&format!(
                        "lifecycle/secret/{}/{name}/b{batch}/fresh",
                        super::math_name(math)
                    ));
                    candidate.run(runtime, 0)?;
                }
                candidate.output(runtime)?;
                continue;
            }
            let mut control = Operator::new(
                runtime,
                target,
                path.clone(),
                "Library",
                [input.clone(), input.clone()],
                &source,
            )?;
            let case = super::lock::TruthCase::new(math, batch, path.name);
            let truth = super::lock::cpu(runtime, super::lock::CpuWork::F64(&case), || {
                evaluate(&path, &input, &control.weights, &control.biases, &mut state)
            })?;
            control.run(runtime, 0)?;
            let library = control.output(runtime)?;
            let nudged = Host {
                input: super::nudge(&input.input, &mut state),
                residual: input.residual.as_ref().map(|r| super::nudge(r, &mut state)),
            };
            let mut fp32 = path.clone();
            match &mut fp32.kind {
                Kind::Dense(spec) => {
                    *spec = DenseSpec::new(spec.site(), batch, CudaMath::Fp32).map_err(error)?
                }
                Kind::Temporal(spec) => {
                    *spec = SegConvSpec::new(spec.site(), batch, CudaMath::Fp32).map_err(error)?
                }
                Kind::Spatial(spec, _) => spec.math = CudaMath::Fp32,
            };
            let mut nudge = Operator::new(
                runtime,
                target,
                fp32,
                "Library",
                [nudged.clone(), nudged],
                &source,
            )?;
            nudge.run(runtime, 0)?;
            let expected = super::SecretReference {
                truth,
                library,
                nudged: nudge.output(runtime)?,
            };
            drop(control);
            drop(nudge);
            let mut candidate = Operator::new(
                runtime,
                target,
                path,
                choice,
                [input.clone(), input],
                &source,
            )?;
            candidate.run(runtime, 0)?;
            let eager = candidate.output(runtime)?;
            candidate.restore(runtime, 0)?;
            let graph = capture(runtime, || candidate.run(runtime, 0))?;
            candidate.restore(runtime, 0)?;
            graph.launch()?;
            let replay = candidate.output(runtime)?;
            let mut row = expected.row(case.id().to_owned(), seed, &eager, &replay);
            row["library_algorithm"] =
                json!("locked model Library operation and independent f64 definition");
            rows.push(row);
        }
    }
    Ok(())
}

#[test]
fn path_topology_has_distinct_shortcut_and_residual_epilogues() {
    let paths = paths("wideconv", 1, CudaMath::Fp32).expect("paths");
    assert_eq!(paths.len(), 22);
    let shortcut = paths
        .iter()
        .find(|path| path.name == "resnet.layer3.0.shortcut.0")
        .unwrap();
    let Kind::Spatial(spec, epilogue) = shortcut.kind else {
        panic!("spatial")
    };
    assert_eq!(spec.kernel, [1, 1]);
    assert_eq!(spec.stride, [2, 2]);
    assert_eq!(epilogue, Epilogue::Bias);
    assert!(shortcut.residual.is_none());
    let second = paths
        .iter()
        .find(|path| path.name == "resnet.layer3.0.conv2")
        .unwrap();
    assert_eq!(second.residual.as_deref(), Some("tensor/getitem_54"));
    let stem = paths
        .iter()
        .find(|path| path.name == "resnet.conv1")
        .unwrap();
    let Kind::Spatial(spec, epilogue) = stem.kind else {
        panic!("spatial")
    };
    assert_eq!(spec.in_channels, 1);
    assert_eq!(epilogue, Epilogue::BiasRelu);
}

#[test]
fn independent_truth_applies_bias_without_shortcut_relu() {
    let path = Path {
        name: "resnet.layer2.0.shortcut.0",
        input: String::new(),
        output: String::new(),
        residual: None,
        kind: Kind::Spatial(
            Conv2d {
                batch: 1,
                in_channels: 1,
                out_channels: 1,
                input: [1, 1],
                kernel: [1, 1],
                padding: [0, 0],
                stride: [1, 1],
                dilation: [1, 1],
                math: CudaMath::Fp32,
            },
            Epilogue::Bias,
        ),
    };
    let host = Host {
        input: vec![2.0],
        residual: None,
    };
    let truth = evaluate(&path, &host, &[-3.0], &[1.0], &mut 7);
    assert_eq!(truth.indices, [0]);
    assert_eq!(truth.values, [-5.0]);
}

#[test]
fn dense_definition_uses_bias_leaky_relu_and_both_weight_layouts() {
    let spec = DenseSpec::new(DenseSite::Linear0, 1, CudaMath::Fp32).unwrap();
    let (_, n, k) = spec.dimensions();
    let mut input = Host {
        input: vec![0.0; spec.input_len()],
        residual: None,
    };
    input.input[0] = 2.0;
    input.input[1] = 1.0;
    input.input[k] = -2.0;
    let mut weight = vec![0.0; spec.weight_len()];
    weight[0] = -3.0;
    weight[n] = 2.0;
    weight[1] = 4.0;
    weight[n + 1] = 6.0;
    let mut bias = vec![0.0; n];
    bias[0] = 1.0;
    bias[1] = -2.0;
    assert_eq!(dense_value(spec, &input, &weight, &bias, 0), -0.03);
    assert_eq!(dense_value(spec, &input, &weight, &bias, 1), 12.0);
    assert_eq!(dense_value(spec, &input, &weight, &bias, n), 7.0);

    let spec = DenseSpec::new(DenseSite::Embedding, 1, CudaMath::Fp32).unwrap();
    let (_, n, k) = spec.dimensions();
    let mut input = Host {
        input: vec![0.0; spec.input_len()],
        residual: None,
    };
    input.input[0] = 2.0;
    input.input[k] = -1.0;
    let mut weight = vec![0.0; spec.weight_len()];
    weight[0] = -3.0;
    weight[k] = 4.0;
    let mut bias = vec![0.0; n];
    bias[0] = 1.0;
    bias[1] = -1.0;
    assert_eq!(dense_value(spec, &input, &weight, &bias, 0), -5.0);
    assert_eq!(dense_value(spec, &input, &weight, &bias, 1), 7.0);
    assert_eq!(dense_value(spec, &input, &weight, &bias, n), 4.0);
}

#[test]
fn classifier_definition_applies_bias_and_row_log_softmax() {
    let spec = DenseSpec::new(DenseSite::Classifier, 1, CudaMath::Fp32).unwrap();
    let (_, n, _) = spec.dimensions();
    let mut input = Host {
        input: vec![0.0; spec.input_len()],
        residual: None,
    };
    input.input[0] = 1.0;
    let mut weight = vec![0.0; spec.weight_len()];
    weight[..n].copy_from_slice(&[-6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0]);
    let bias = [6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0];
    let expected = [
        -6.457762847404243,
        -5.457762847404243,
        -4.457762847404243,
        -3.4577628474042426,
        -2.4577628474042426,
        -1.4577628474042426,
        -0.45776284740424256,
    ];
    let values: Vec<_> = (0..n)
        .map(|index| dense_value(spec, &input, &weight, &bias, index))
        .collect();
    for (value, expected) in values.iter().zip(expected) {
        assert!((value - expected).abs() < 2e-15);
    }
    assert!((values.iter().map(|value| value.exp()).sum::<f64>() - 1.0).abs() < 2e-15);
    assert!((dense_value(spec, &input, &weight, &bias, n) + 0.14541262633979457).abs() < 2e-15);
    assert!(
        (dense_value(spec, &input, &weight, &bias, 2 * n - 1) + 12.145412626339795).abs() < 2e-15
    );
}

#[test]
fn temporal_definition_is_raw_five_tap_cross_correlation_with_batch_and_channel_edges() {
    for site in [SegConvSite::Conv1, SegConvSite::Conv2] {
        let spec = SegConvSpec::new(site, 2, CudaMath::Fp32).unwrap();
        let mut input = Host {
            input: vec![0.0; spec.input_len()],
            residual: None,
        };
        input.input[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let last_channel = (spec.in_channels() - 1) * spec.input_steps();
        input.input[last_channel..last_channel + 6]
            .copy_from_slice(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let tail = (2 * spec.in_channels() - 1) * spec.input_steps() + spec.output_steps() - 1;
        input.input[tail..tail + 5].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut weight = vec![0.0; spec.weight_len()];
        weight[..5].copy_from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0]);
        let w = (spec.in_channels() - 1) * 5;
        weight[w..w + 5].copy_from_slice(&[-1.0, 2.0, -3.0, 4.0, -5.0]);
        let w = (spec.out_channels() - 1) * spec.in_channels() * 5;
        weight[w..w + 5].fill(-1.0);
        assert_eq!(temporal_value(spec, &input, &weight, 0), 9.0);
        assert_eq!(temporal_value(spec, &input, &weight, 1), 15.0);
        assert_eq!(
            temporal_value(
                spec,
                &input,
                &weight,
                (spec.out_channels() - 1) * spec.output_steps()
            ),
            -15.0
        );
        assert_eq!(
            temporal_value(
                spec,
                &input,
                &weight,
                spec.out_channels() * spec.output_steps() + spec.output_steps() - 1
            ),
            -15.0
        );
    }
}

#[test]
fn dense_sampler_retains_matrix_shape_minimum_and_complete_row_column_coverage() {
    use std::collections::BTreeSet;
    for batch in [1, 32, 64] {
        for site in [
            DenseSite::Linear0,
            DenseSite::Linear1,
            DenseSite::Classifier,
            DenseSite::Embedding,
        ] {
            let spec = DenseSpec::new(site, batch, CudaMath::Fp32).unwrap();
            let (rows, columns, _) = spec.dimensions();
            let shape = sample_shape(Kind::Dense(spec));
            assert_eq!(shape, [batch * rows, columns]);
            let indices = super::reference::indices(&shape, &mut 73);
            let total = batch * rows * columns;
            assert!(indices.len() >= 4096.min(total));
            assert!(indices.iter().all(|index| *index < total));
            assert!(indices.windows(2).all(|pair| pair[0] < pair[1]));
            assert_eq!(
                indices
                    .iter()
                    .map(|index| index / columns)
                    .collect::<BTreeSet<_>>()
                    .len(),
                batch * rows
            );
            assert_eq!(
                indices
                    .iter()
                    .map(|index| index % columns)
                    .collect::<BTreeSet<_>>()
                    .len(),
                columns
            );
            if batch == 64 && site != DenseSite::Embedding {
                assert!(indices.len() >= 37_696);
                assert!(indices.len() <= 37_696 + columns);
            }
            println!(
                "dense sampler site={site:?} batch={batch} shape={shape:?} count={}",
                indices.len()
            );
        }
    }
}

#[test]
fn candidate_metadata_queries_do_not_invent_coverage_without_a_port() {
    assert_eq!(
        declared_coverage("segdense-linear0", crate::inference::cuda::PtxTier::Sm75),
        json!({"entries":[]})
    );
    assert_eq!(
        declared_coverage("wideconv", crate::inference::cuda::PtxTier::Sm80),
        json!({"entries":[]})
    );
    assert_eq!(
        configurations().unwrap(),
        test_support::configuration::planned()
    );
    let _preload = preload_candidates;
}

#[test]
fn candidate_coverage_query_keeps_only_the_requested_operation_path() {
    use crate::inference::cuda::candidate::{Batches, Coverage, CoverageEntry, Maths};
    let coverage = Coverage(&[CoverageEntry {
        layers: &["linear0", "sincnet.conv1", "resnet.layer3.0.conv2"],
        batches: Batches::Only(&[1, 32]),
        maths: Maths::Only(&[CudaMath::Fp32]),
    }]);
    for (target, boundary) in [
        ("segdense-linear0", "linear0"),
        ("segdense-conv1", "sincnet.conv1"),
        ("wideconv", "resnet.layer3.0.conv2"),
    ] {
        let paths = paths(target, 1, CudaMath::Fp32).unwrap();
        assert_eq!(
            coverage_for_paths(&paths, coverage),
            json!({"entries":[{"layers":[boundary],"batches":[1,32],"maths":["fp32"]}]})
        );
    }
}
