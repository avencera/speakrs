//! Locked collection path for the filterbank energy producer and full stage

use std::cell::RefCell;

use super::{
    BAND_SEEDS, Run, bursts, capture, metrics, metrics_f64, paired, record, sha, timing_row,
};
use crate::inference::cuda::candidate::{FbankCandidate, FbankSpec, Phases};
use crate::inference::cuda::fbank::test_support::Library;
use crate::inference::cuda::test_support::{self, Mutant};
use crate::inference::cuda::{CudaError, CudaMath, CudaRuntime, SafetensorsFile};
use cudarc::driver::CudaSlice;
use serde_json::{Value, json};

const LAYER: &str = "fbank.dft";
const ELEMENTS: usize = 998 * 80;
const WAVEFORM_SAMPLES: usize = 160_000;

/// A prefix of the pinned B32 fixture, never the current operator's input
struct LookupFixture(Vec<f32>);

impl LookupFixture {
    fn load(spec: FbankSpec) -> Result<Self, CudaError> {
        let file = super::reference("fbankdft", "mixed")?;
        let waveform = file.read_f32("input/waveform", &[32, 1, WAVEFORM_SAMPLES])?;
        Self::from_b32(spec, &waveform)
    }

    fn from_b32(spec: FbankSpec, waveform: &[f32]) -> Result<Self, CudaError> {
        let expected = 32 * WAVEFORM_SAMPLES;
        if waveform.len() != expected {
            return Err(CudaError::BufferLength {
                context: "fbank Lookup B32 fixture",
                expected,
                actual: waveform.len(),
            });
        }
        Ok(Self(waveform[..spec.batch() * WAVEFORM_SAMPLES].to_vec()))
    }
}

/// Own energies computed from the fixture before capture, independent of live inputs
struct LookupPlan<T>(T);

impl<T> LookupPlan<T> {
    fn prepare(
        spec: FbankSpec,
        fixture: LookupFixture,
        memorize: impl FnOnce(&[f32], usize) -> Result<T, CudaError>,
    ) -> Result<Self, CudaError> {
        let expected = spec.batch() * WAVEFORM_SAMPLES;
        if fixture.0.len() != expected {
            return Err(CudaError::BufferLength {
                context: "fbank Lookup snapshot",
                expected,
                actual: fixture.0.len(),
            });
        }
        // preparation receives only fixture data; secret inputs cannot refresh the answer
        memorize(&fixture.0, spec.batch() * ELEMENTS)
            .map(Self)
            .map_err(|error| Self::error("memorize fixture energies before capture", error))
    }

    fn replay(&self, copy: impl FnOnce(&T) -> Result<(), CudaError>) -> Result<(), CudaError> {
        copy(&self.0).map_err(|error| Self::error("replay cached fixture energies", error))
    }

    fn error(call: &'static str, error: CudaError) -> CudaError {
        CudaError::Unsupported {
            context: "fbank Lookup injection",
            reason: format!("{call}: {error}"),
        }
    }
}

/// Keep capture diagnostics local to the Lookup injection, not other harness routes
fn capture_lookup(
    runtime: &CudaRuntime,
    enqueue: impl FnOnce() -> Result<(), CudaError>,
) -> Result<cudarc::driver::CudaGraph, CudaError> {
    use cudarc::driver::sys::{CUgraphInstantiate_flags, CUstreamCaptureMode};

    let stream = runtime.stream();
    stream
        .synchronize()
        .map_err(CudaError::from)
        .map_err(|error| {
            LookupPlan::<()>::error("CudaStream::synchronize before capture", error)
        })?;
    let context = runtime.context();
    let tracking = context.is_event_tracking();
    // SAFETY: all buffers use this stream and remain owned by the operator
    unsafe { context.disable_event_tracking() };
    let captured = stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .map_err(CudaError::from)
        .map_err(|error| LookupPlan::<()>::error("cuStreamBeginCapture", error))
        .and_then(|()| {
            let _trace = super::super::CaptureTrace::start();
            let enqueued = enqueue();
            // end capture even after enqueue fails, as on the ordinary harness path
            let graph = stream
                .end_capture(CUgraphInstantiate_flags(0))
                .map_err(CudaError::from)
                .map_err(|error| {
                    LookupPlan::<()>::error(
                        "CudaStream::end_capture (cuStreamEndCapture/cuGraphInstantiate)",
                        error,
                    )
                });
            enqueued?;
            graph
        });
    if tracking {
        // SAFETY: restore the tracking state from before capture, including on error
        unsafe { context.enable_event_tracking() };
    }
    Ok(captured?.expect("Lookup enqueues a cached energy copy"))
}

fn plan_error(error: impl std::fmt::Display) -> CudaError {
    CudaError::Unsupported {
        context: "fbank.dft qualification",
        reason: error.to_string(),
    }
}

/// Identify mel matrix multiplication by its operation and complete output shape
fn energy_tensor(file: &SafetensorsFile) -> String {
    let names = file.names();
    mel_energy_name(
        names
            .iter()
            .filter_map(|name| file.shape(name).map(|shape| (name.as_str(), shape))),
    )
}

fn mel_energy_name<'a>(tensors: impl IntoIterator<Item = (&'a str, &'a [usize])>) -> String {
    let names: Vec<_> = tensors
        .into_iter()
        .filter(|(name, shape)| {
            name.starts_with("tensor/")
                && name.to_ascii_lowercase().contains("/matmul")
                && *shape == [32, 998, 80]
        })
        .map(|(name, _)| name)
        .collect();
    assert_eq!(
        names.len(),
        1,
        "one mel matrix-multiply output must match [32,998,80]"
    );
    names[0].to_owned()
}

fn fixture_row(case: &str, row: usize) -> usize {
    assert!(row < 32, "fbank fixture batch domain");
    match case {
        "first" => 0,
        "last" => 17,
        "short" => 18,
        "mixed" => row,
        _ => panic!("fixed fbank fixture case"),
    }
}

/// Select matching audio and reference rows from the one hash-pinned B32 snapshot
fn fixture(
    file: &SafetensorsFile,
    tensor: &str,
    batch: usize,
    case: &str,
) -> Result<Vec<f32>, CudaError> {
    let shape = file.shape(tensor).expect("pinned fbank reference tensor");
    assert_eq!(shape[0], 32);
    let values = file.read_f32(tensor, shape)?;
    let width = values.len() / 32;
    Ok((0..batch)
        .flat_map(|row| {
            let source = fixture_row(case, row);
            values[source * width..(source + 1) * width].iter().copied()
        })
        .collect())
}

enum Route<C> {
    Library,
    Candidate(C),
    Lookup(LookupPlan<CudaSlice<f32>>),
    Mutant(Mutant),
}

/// The replacement producer owns its plan; the locked consumer is identical on both routes
struct Operator<C> {
    library: Library,
    route: Route<C>,
    spec: FbankSpec,
    audio: [CudaSlice<f32>; 2],
    host: [Vec<f32>; 2],
    energies: CudaSlice<f32>,
    features: CudaSlice<f32>,
}

impl<C: FbankCandidate> Operator<C>
where
    C::Pin: test_support::configuration::FbankPinEvidence,
{
    fn new(
        runtime: &CudaRuntime,
        spec: FbankSpec,
        choice: &str,
        host: [Vec<f32>; 2],
    ) -> Result<Self, CudaError> {
        let _plan = test_support::plan(LAYER);
        let library = Library::plan(runtime, spec, ()).map_err(plan_error)?;
        let route = match choice {
            "Library" => Route::Library,
            "Lookup" => {
                let fixture = LookupFixture::load(spec)?;
                let lookup = LookupPlan::prepare(spec, fixture, |waveform, len| {
                    let input = runtime
                        .stream()
                        .clone_htod(waveform)
                        .map_err(CudaError::from)
                        .map_err(|error| {
                            LookupPlan::<()>::error("upload fixture waveform", error)
                        })?;
                    let mut energies = runtime
                        .stream()
                        .alloc_zeros(len)
                        .map_err(CudaError::from)
                        .map_err(|error| {
                        LookupPlan::<()>::error("allocate cached energies", error)
                    })?;
                    let _scope = test_support::library(runtime.stream(), LAYER);
                    library
                        .enqueue(
                            &input.as_view(),
                            &mut energies.as_view_mut(),
                            &Phases::new(),
                            runtime,
                        )
                        .map_err(|error| {
                            LookupPlan::<()>::error("compute fixture energies", error)
                        })?;
                    runtime.synchronize().map_err(|error| {
                        LookupPlan::<()>::error("finish fixture computation", error)
                    })?;
                    Ok(energies)
                })?;
                Route::Lookup(lookup)
            }
            "Oxide" => {
                assert!(
                    C::coverage(runtime.ptx_tier()).covers(LAYER, spec.batch(), spec.math()),
                    "fbank candidate must declare this plan"
                );
                let pin = C::implemented_pin(spec).map_err(plan_error)?;
                let candidate = C::plan(runtime, spec, pin).map_err(plan_error)?;
                let identity = test_support::configuration::FbankPinEvidence::configuration(pin)
                    .expect("a candidate plan has a complete configuration pin");
                test_support::configuration::record(LAYER, spec.batch(), spec.math(), identity);
                Route::Candidate(candidate)
            }
            name => Route::Mutant(Mutant::parse(name).expect("applicable fbank mutant")),
        };
        let audio = [
            runtime.stream().clone_htod(&host[0])?,
            runtime.stream().clone_htod(&host[1])?,
        ];
        let len = spec.batch() * ELEMENTS;
        Ok(Self {
            library,
            route,
            spec,
            audio,
            host,
            energies: runtime.stream().alloc_zeros(len)?,
            features: runtime.stream().alloc_zeros(len)?,
        })
    }

    fn declared(&self) -> bool {
        !matches!(self.route, Route::Library)
    }

    fn restore(&mut self, runtime: &CudaRuntime, which: usize) -> Result<(), CudaError> {
        runtime
            .stream()
            .memcpy_htod(&self.host[which], &mut self.audio[which])?;
        Ok(())
    }

    fn run(&mut self, runtime: &CudaRuntime, which: usize, stage: bool) -> Result<(), CudaError> {
        let phases = Phases::new();
        match &self.route {
            Route::Library => {
                let _scope = test_support::library(runtime.stream(), LAYER);
                if stage && !test_support::band_enabled(LAYER) {
                    return self.library.full(
                        runtime,
                        &self.audio[which].as_view(),
                        &mut self.features.as_view_mut(),
                    );
                }
                self.library.enqueue(
                    &self.audio[which].as_view(),
                    &mut self.energies.as_view_mut(),
                    &phases,
                    runtime,
                )?;
                test_support::perturb(runtime, LAYER, &mut self.energies)?;
            }
            Route::Candidate(candidate) => {
                let _scope = test_support::candidate(runtime.stream(), LAYER);
                candidate.enqueue(
                    &self.audio[which].as_view(),
                    &mut self.energies.as_view_mut(),
                    &phases,
                    runtime,
                )?;
            }
            Route::Lookup(lookup) => {
                test_support::poison(runtime)
                    .map_err(|error| LookupPlan::<()>::error("poison launch", error))?;
                let _scope = test_support::mutant_scope(runtime.stream(), LAYER, Mutant::Lookup);
                lookup.replay(|energies| {
                    Ok(runtime.stream().memcpy_dtod(energies, &mut self.energies)?)
                })?;
                test_support::post(
                    runtime,
                    &mut self.energies,
                    self.spec.batch(),
                    Mutant::Lookup,
                )
                .map_err(|error| LookupPlan::<()>::error("post output pointer", error))?;
                if stage {
                    let _scope = test_support::library(runtime.stream(), "fbank.log_cmn");
                    self.library
                        .consume(
                            runtime,
                            &self.energies.as_view(),
                            &mut self.features.as_view_mut(),
                        )
                        .map_err(|error| LookupPlan::<()>::error("log_cmn launch", error))?;
                }
                return Ok(());
            }
            Route::Mutant(mutant) => {
                test_support::poison(runtime)?;
                let _scope = test_support::mutant_scope(runtime.stream(), LAYER, *mutant);
                if !mutant.skips() {
                    if *mutant == Mutant::Precision {
                        test_support::round_input(runtime, &self.audio[which])?;
                    }
                    let input = &self.audio[which];
                    let repeats = if *mutant == Mutant::Slow { 3 } else { 1 };
                    for _ in 0..repeats {
                        self.library.enqueue(
                            &input.as_view(),
                            &mut self.energies.as_view_mut(),
                            &phases,
                            runtime,
                        )?;
                    }
                    test_support::post(runtime, &mut self.energies, self.spec.batch(), *mutant)?;
                }
            }
        }
        if stage {
            let _scope = test_support::library(runtime.stream(), "fbank.log_cmn");
            self.library.consume(
                runtime,
                &self.energies.as_view(),
                &mut self.features.as_view_mut(),
            )?;
            if matches!(self.route, Route::Mutant(Mutant::StageAccuracy))
                && self.spec.math() == CudaMath::Tf32
                && test_support::phase() == Some("numeric")
            {
                test_support::round_input(runtime, &self.features)?;
            }
        }
        if matches!(self.route, Route::Mutant(Mutant::Unscoped)) {
            test_support::unscoped(runtime, Mutant::Unscoped)?;
        }
        Ok(())
    }

    fn output(&self, runtime: &CudaRuntime, stage: bool) -> Result<Vec<f32>, CudaError> {
        Ok(runtime.stream().clone_dtoh(if stage {
            &self.features
        } else {
            &self.energies
        })?)
    }
}

impl Run<'_> {
    pub(super) fn fbank(&self, rows: &mut Vec<Value>) -> Result<(), CudaError> {
        let case = self.id.split('/').nth(1).expect("case id");
        let alternate = if case == "short" { "first" } else { "short" };
        let cases = [case, alternate];
        let audio = [
            fixture(self.files[0], "input/waveform", self.batch, cases[0])?,
            fixture(self.files[1], "input/waveform", self.batch, cases[1])?,
        ];
        assert_ne!(
            sha(&audio[0]),
            sha(&audio[1]),
            "switched fbank inputs must differ"
        );
        let spec = FbankSpec::new(self.batch, self.math).map_err(plan_error)?;
        let mut op = Operator::<Library>::new(self.runtime, spec, self.choice, audio.clone())?;
        if self.phase == "paired" {
            return self.paired_fbank(rows, &mut op, spec, audio);
        }
        let truth = if self.phase == "numeric" {
            let mut values = Vec::new();
            for (which, input) in audio.iter().enumerate() {
                let case =
                    super::lock::StageTruthCase::new(&self.key("stage", which), self.batch, input);
                let full = super::lock::stage_truth(self.runtime, &case, || {
                    let full = super::reference::fbank_truth::stage(input);
                    (full, super::reference::fbank_truth::constants())
                })?;
                values.push((full, case.binding().clone()));
            }
            Some(values)
        } else {
            None
        };
        for stage in [false, true] {
            if stage && self.phase == "sanitize" {
                continue;
            }
            let layer = if stage { "stage" } else { LAYER };
            if self.phase == "profile" {
                for which in 0..2 {
                    op.restore(self.runtime, which)?;
                    super::profile_case(
                        self.runtime,
                        &self.key(layer, which),
                        self.choice,
                        LAYER,
                        || op.run(self.runtime, which, stage),
                    )?;
                    op.output(self.runtime, stage)?;
                }
                continue;
            }
            op.restore(self.runtime, 0)?;
            test_support::set_label(Some(self.key(layer, 0)));
            let first = if self.choice == "Lookup" {
                capture_lookup(self.runtime, || op.run(self.runtime, 0, stage))
            } else {
                capture(self.runtime, || op.run(self.runtime, 0, stage))
            };
            test_support::set_label(None);
            let graphs = [
                first?,
                if self.choice == "Lookup" {
                    capture_lookup(self.runtime, || op.run(self.runtime, 1, stage))?
                } else {
                    capture(self.runtime, || op.run(self.runtime, 1, stage))?
                },
            ];
            if self.phase == "timing" {
                let cell = RefCell::new(&mut op);
                let timing = bursts(
                    self.runtime,
                    |sample| cell.borrow_mut().restore(self.runtime, sample % 2),
                    |launch| Ok(graphs[launch % 2].launch()?),
                )?;
                let mut outputs = Vec::new();
                for (which, graph) in graphs.iter().enumerate() {
                    cell.borrow_mut().restore(self.runtime, which)?;
                    graph.launch()?;
                    outputs.push(cell.borrow().output(self.runtime, stage)?);
                }
                rows.push(timing_row(
                    &self.key(layer, 0),
                    &timing,
                    [&outputs[0], &outputs[1]],
                    cell.borrow().declared(),
                ));
                continue;
            }
            if self.phase == "sanitize" {
                for (which, graph) in graphs.iter().enumerate() {
                    op.restore(self.runtime, which)?;
                    graph.launch()?;
                }
                self.runtime.synchronize()?;
                rows.push(
                    json!({"id":self.key(LAYER, 0),"sanitized":true,"declared":op.declared()}),
                );
                println!("sanitized {}", self.key(LAYER, 0));
                continue;
            }
            for which in 0..2 {
                let tensor = if stage {
                    "tensor/fbank".to_owned()
                } else {
                    energy_tensor(self.files[which])
                };
                let expected = fixture(self.files[which], &tensor, self.batch, cases[which])?;
                op.restore(self.runtime, which)?;
                graphs[which].launch()?;
                if stage && self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                    test_support::round_input(self.runtime, &op.features)?;
                }
                let first = op.output(self.runtime, stage)?;
                op.restore(self.runtime, which)?;
                graphs[which].launch()?;
                if stage && self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                    test_support::round_input(self.runtime, &op.features)?;
                }
                let second = op.output(self.runtime, stage)?;
                let key = self.key(layer, which);
                if stage {
                    let (full, binding) =
                        &truth.as_ref().expect("numeric CPU f64 stage truth")[which];
                    let first_truth = metrics_f64(&first, full, 80);
                    let second_truth = metrics_f64(&second, full, 80);
                    rows.push(json!({
                        "id":key, "first":first_truth, "second":second_truth,
                        "truth":first_truth, "stage_truth":binding,
                        "truth_sha256":binding["truth_sha256"], "declared":op.declared(),
                        "bitwise_equal":first.iter().zip(&second).all(|(x,y)| x.to_bits()==y.to_bits()),
                        "fixture_diagnostic": {"reference":"ORT FP32 tensor/fbank", "acceptance_bound":false,
                            "first":metrics(&first, &expected, 80), "second":metrics(&second, &expected, 80)},
                    }));
                } else {
                    record(rows, &key, first, second, &expected, 1, op.declared());
                }
                if stage && let Some(layers) = &self.band {
                    let mut band = Vec::new();
                    let mut draws = Vec::new();
                    for seed in BAND_SEEDS {
                        test_support::set_band(Some((key.clone(), seed, layers.clone())));
                        op.restore(self.runtime, which)?;
                        let result = op.run(self.runtime, which, true);
                        test_support::set_band(None);
                        result?;
                        let output = op.output(self.runtime, true)?;
                        let full = &truth.as_ref().expect("CPU f64 stage truth")[which].0;
                        let measured = metrics_f64(&output, full, 80);
                        band.push(measured.clone());
                        draws.push(measured);
                    }
                    rows.push(json!({"id":format!("{key}/band"),"seeds":BAND_SEEDS,"layers":layers,"metrics":band,"truth_draws":draws,"truth_sha256":truth.as_ref().expect("CPU f64 stage truth")[which].1["truth_sha256"],"stage_truth":truth.as_ref().expect("CPU f64 stage truth")[which].1}));
                }
            }
        }
        Ok(())
    }

    fn paired_fbank(
        &self,
        rows: &mut Vec<Value>,
        candidate: &mut Operator<Library>,
        spec: FbankSpec,
        audio: [Vec<f32>; 2],
    ) -> Result<(), CudaError> {
        let mut library = Operator::<Library>::new(self.runtime, spec, "Library", audio)?;
        let mut stages = Vec::new();
        let mut operators = Vec::new();
        for op in [&mut library, &mut *candidate] {
            operators.push([
                capture(self.runtime, || op.run(self.runtime, 0, false))?,
                capture(self.runtime, || op.run(self.runtime, 1, false))?,
            ]);
            stages.push([
                capture(self.runtime, || op.run(self.runtime, 0, true))?,
                capture(self.runtime, || op.run(self.runtime, 1, true))?,
            ]);
        }
        let mut row = paired::replay_set(paired::BLOCKS, |which, _| {
            paired::observe(
                self.runtime,
                [&stages[0][which], &stages[1][which]],
                [&operators[0..1], &operators[1..2]],
                which,
                self.choice == "StageSlow",
                None,
            )
        })?;
        let mut stage_hashes = Vec::new();
        let mut operator_hashes = Vec::new();
        for (side, op) in [&mut library, candidate].into_iter().enumerate() {
            let mut stage = Vec::new();
            let mut operator = Vec::new();
            for which in 0..2 {
                op.restore(self.runtime, which)?;
                operators[side][which].launch()?;
                operator.push(sha(&op.output(self.runtime, false)?));
                op.restore(self.runtime, which)?;
                stages[side][which].launch()?;
                if side == 1 && self.choice == "StageAccuracy" && self.math == CudaMath::Tf32 {
                    test_support::round_input(self.runtime, &op.features)?;
                }
                stage.push(sha(&op.output(self.runtime, true)?));
            }
            stage_hashes.push(stage);
            operator_hashes.push(operator);
        }
        row["id"] = json!(self.key("stage", 0));
        row["output_sha256"] = json!(stage_hashes);
        row["operator_outputs"] = json!([{"layer":LAYER,"output_sha256":operator_hashes}]);
        rows.push(row);
        Ok(())
    }
}

/// Fresh in-memory audio, independent f64 energies, and identical eager/replay plans
pub(super) fn secret(
    runtime: &CudaRuntime,
    choice: &str,
    math: CudaMath,
    rows: &mut Vec<Value>,
) -> Result<(), CudaError> {
    let seed = test_support::secret_seed();
    let mut state = seed;
    let file = super::reference("fbankdft", "mixed")?;
    let fixture = file.read_f32("input/waveform", &[32, 1, 160_000])?;
    let constants = crate::inference::cuda::fbank::FbankConstants::new();
    for batch in 1..=32 {
        let audio = super::transformed_audio(&fixture, batch, &mut state);
        if test_support::phase() == Some("profile") {
            let spec = FbankSpec::new(batch, math).map_err(plan_error)?;
            let mut candidate =
                Operator::<Library>::new(runtime, spec, choice, [audio.clone(), audio])?;
            {
                let _window = test_support::window(&format!(
                    "lifecycle/secret/{}/{LAYER}/b{batch}/fresh",
                    super::math_name(math)
                ));
                candidate.run(runtime, 0, false)?;
            }
            candidate.output(runtime, false)?;
            continue;
        }
        let case = super::lock::TruthCase::new(math, batch, LAYER);
        let truth = super::lock::cpu(runtime, super::lock::CpuWork::F64(&case), || {
            let indices = super::reference::indices(&[batch, 80, 998], &mut state)
                .into_iter()
                .map(|index| {
                    let row = index / (80 * 998);
                    let bin = index / 998 % 80;
                    let frame = index % 998;
                    (row * 998 + frame) * 80 + bin
                })
                .collect();
            super::reference::fbank(&audio, constants.mel(), indices)
        })?;
        let spec = FbankSpec::new(batch, math).map_err(plan_error)?;
        let mut library =
            Operator::<Library>::new(runtime, spec, "Library", [audio.clone(), audio.clone()])?;
        library.run(runtime, 0, false)?;
        let library = library.output(runtime, false)?;
        let nudged = super::nudge(&audio, &mut state);
        let fp32 = FbankSpec::new(batch, CudaMath::Fp32).map_err(plan_error)?;
        let mut control =
            Operator::<Library>::new(runtime, fp32, "Library", [nudged.clone(), nudged])?;
        control.run(runtime, 0, false)?;
        let expected = super::SecretReference {
            truth,
            library,
            nudged: control.output(runtime, false)?,
        };
        let mut candidate =
            Operator::<Library>::new(runtime, spec, choice, [audio.clone(), audio])?;
        candidate.run(runtime, 0, false)?;
        let eager = candidate.output(runtime, false)?;
        candidate.restore(runtime, 0)?;
        let graph = capture(runtime, || candidate.run(runtime, 0, false))?;
        candidate.restore(runtime, 0)?;
        graph.launch()?;
        let replay = candidate.output(runtime, false)?;
        let mut row = expected.row(case.id().to_owned(), seed, &eager, &replay);
        row["library_algorithm"] = json!("frame/window, cuBLAS DFT, sparse mel");
        rows.push(row);
    }
    Ok(())
}

#[test]
fn lookup_memorizes_only_fixture_outputs_for_every_batch() {
    use std::cell::Cell;

    use crate::inference::cuda::implementation::BoundaryId;

    let fixture: Vec<f32> = (0..32 * WAVEFORM_SAMPLES)
        .map(|index| {
            let row = index / WAVEFORM_SAMPLES;
            let sample = index % WAVEFORM_SAMPLES;
            ((sample as f32 * (0.034 + row as f32 * 0.013)).sin()
                + 0.2 * (sample as f32 * 0.27).cos())
                * (0.02 + row as f32 * 0.003)
        })
        .collect();
    let constants = crate::inference::cuda::fbank::FbankConstants::new();
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        let batches = BoundaryId::named(LAYER).batches();
        for batch in (0..=64).filter(|batch| batches.contains(*batch)) {
            let spec = FbankSpec::new(batch, math).expect("declared batch");
            let source = LookupFixture::from_b32(spec, &fixture).expect("fixture prefix");
            assert_eq!(source.0, fixture[..batch * WAVEFORM_SAMPLES]);
            let indices: Vec<_> = (0..batch)
                .map(|row| row * ELEMENTS + 37 * 80 + 31)
                .collect();
            let fixture_output =
                super::reference::fbank(&source.0, constants.mel(), indices.clone());
            let preparations = Cell::new(0);
            let plan = LookupPlan::prepare(spec, source, |memorized, len| {
                preparations.set(preparations.get() + 1);
                assert_eq!(memorized, &fixture[..batch * WAVEFORM_SAMPLES]);
                assert_eq!(len, batch * ELEMENTS);
                let mut output = vec![0.0f64; len];
                for (&index, &value) in indices.iter().zip(&fixture_output.values) {
                    output[index] = value;
                }
                Ok(output)
            })
            .expect("memorize fixture energies");
            let address = plan.0.as_ptr();
            for mut seed in [17, 18] {
                // use the actual secret generator and independent Library DFT definition
                // driver capture and the CUDA Library itself require the later box run
                let fresh = super::transformed_audio(&fixture, batch, &mut seed);
                assert_ne!(&fixture[..batch * WAVEFORM_SAMPLES], fresh);
                let fresh_library =
                    super::reference::fbank(&fresh, constants.mel(), indices.clone());
                let mut replay = Vec::new();
                for _input in 0..2 {
                    for _stage in [false, true] {
                        plan.replay(|cached| {
                            assert_eq!(cached.as_ptr(), address);
                            replay = indices.iter().map(|&index| cached[index]).collect();
                            Ok(())
                        })
                        .expect("capture enqueue only borrows cached output");
                        assert_eq!(replay, fixture_output.values);
                        assert_ne!(replay, fresh_library.values);
                        let drift: f64 = replay
                            .iter()
                            .zip(&fresh_library.values)
                            .map(|(a, b)| (a - b).abs())
                            .sum();
                        let scale: f64 = fresh_library.values.iter().map(|value| value.abs()).sum();
                        assert!(
                            drift > 0.01 * scale,
                            "fixture answers must not solve a fresh input"
                        );
                        assert_eq!(preparations.get(), 1, "replay cannot memorize fresh inputs");
                    }
                }
            }
        }
    }
}

#[test]
fn lookup_rejects_nonfixture_geometry_and_reports_replay_errors() {
    let spec = FbankSpec::new(1, CudaMath::Fp32).expect("batch");
    assert!(matches!(
        LookupFixture::from_b32(spec, &[0.0]),
        Err(CudaError::BufferLength {
            context: "fbank Lookup B32 fixture",
            expected: 5_120_000,
            actual: 1,
        })
    ));
    let result = LookupPlan::prepare(
        spec,
        LookupFixture(vec![0.0]),
        |_, _| -> Result<(), CudaError> { panic!("invalid fixture must not be memorized") },
    );
    assert!(matches!(
        result,
        Err(CudaError::BufferLength {
            context: "fbank Lookup snapshot",
            expected: WAVEFORM_SAMPLES,
            actual: 1,
        })
    ));
    let error = LookupPlan(()).replay(|_| {
        Err(CudaError::BufferLength {
            context: "injected copy error",
            expected: 2,
            actual: 1,
        })
    });
    let message = error.expect_err("copy failure is retained").to_string();
    assert!(message.contains("fbank Lookup injection"));
    assert!(message.contains("replay cached fixture energies"));
    assert!(message.contains("injected copy error"));
}

#[test]
fn mel_reference_selection_requires_unique_operation_and_shape() {
    let candidates = [
        ("tensor//MatMul_output_0", &[32, 998, 512][..]),
        ("tensor/fbank", &[32, 998, 80][..]),
        ("tensor//MatMul_1_output_0", &[32, 998, 80][..]),
    ];
    assert_eq!(mel_energy_name(candidates), "tensor//MatMul_1_output_0");
    assert!(std::panic::catch_unwind(|| mel_energy_name(candidates[..2].iter().copied())).is_err());

    // the hash-pinned snapshot names its PyTorch operation in lowercase
    let snapshot = [
        ("tensor/clamp_min", &[32, 998, 80][..]),
        ("tensor/fbank", &[32, 998, 80][..]),
        ("tensor/log", &[32, 998, 80][..]),
        ("tensor/matmul", &[32, 998, 80][..]),
    ];
    assert_eq!(mel_energy_name(snapshot), "tensor/matmul");

    let duplicate = [
        ("tensor/matmul", &[32, 998, 80][..]),
        ("tensor//MatMul_2_output_0", &[32, 998, 80][..]),
    ];
    assert!(std::panic::catch_unwind(|| mel_energy_name(duplicate)).is_err());
}

#[test]
fn fixture_rows_keep_inputs_and_references_in_the_same_window() {
    assert_eq!(
        (0..7)
            .map(|row| fixture_row("last", row))
            .collect::<Vec<_>>(),
        vec![17; 7]
    );
    assert_eq!(
        (0..7)
            .map(|row| fixture_row("short", row))
            .collect::<Vec<_>>(),
        vec![18; 7]
    );
    assert_eq!(
        (0..32)
            .map(|row| fixture_row("mixed", row))
            .collect::<Vec<_>>(),
        (0..32).collect::<Vec<_>>()
    );
    assert_ne!(fixture_row("first", 0), fixture_row("short", 0));
    assert!(std::panic::catch_unwind(|| fixture_row("mixed", 32)).is_err());
}
