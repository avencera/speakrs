use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use color_eyre::eyre::{Context, Result, ensure, eyre};
use serde::Serialize;
use speakrs::inference::ExecutionMode;
use speakrs::pipeline::{
    ClusteringBackend, OwnedDiarizationPipeline, PipelineBuilder, RuntimeConfig,
};

use crate::cmd::project_root;
use crate::commands::benchmark::{DerAccumulation, PerFileDerResult};
use crate::wav::load_wav_samples;

use super::domain::{
    CandidateClustering, CheckedCandidate, ComparisonSide, CoreMlMode, ExecutableExperiment,
};
use super::record::{
    ChunkInferenceStageTimings, ClusteringBackendDiagnostics, ClusteringDiagnostics,
    ExperimentRecord, SphereAhcInitializationDiagnostics, SphereInitializationDiagnostics,
    StageTimings,
};
use super::store::{ManifestFile, RunStore};

const REQUIRED_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Serialize)]
struct RepetitionRecord {
    status: &'static str,
    resumed: bool,
    whole_process_seconds: f64,
    peak_rss_bytes: Option<u64>,
    completed_at: String,
}

pub(super) fn run_managed(store: &RunStore) -> Result<()> {
    let worker = build_worker_binary()?;
    run_managed_with_worker(&worker, store)
}

pub(super) fn run_managed_with_worker(worker: &Path, store: &RunStore) -> Result<()> {
    let executable = store.experiment().to_executable()?;
    store.assert_worker(worker)?;
    store.assert_input_digests()?;
    let executed = match &executable.spec().baseline_run {
        Some(baseline_run) => run_abba_comparison(worker, store, &executable, baseline_run)?,
        None => run_isolated_repetitions(worker, store, &executable)?,
    };

    store.rebuild_projections()?;
    if !executed {
        println!("all experiment records are already complete");
    }
    Ok(())
}

fn run_isolated_repetitions(
    worker: &Path,
    store: &RunStore,
    experiment: &ExecutableExperiment,
) -> Result<bool> {
    let mut executed = false;
    for run in experiment.schedule().runs() {
        if store.repetition_complete(run.repetition)? {
            store.ensure_unmeasured_repetition_record(run.repetition)?;
            continue;
        }
        executed = true;
        store.assert_worker(worker)?;
        run_worker_command(
            worker,
            &[
                "mac-experiment",
                "worker",
                &store.run_dir().to_string_lossy(),
                "--repetition",
                &run.repetition.to_string(),
            ],
            &format!("experiment repetition {}", run.repetition),
            &store.process_log_path(run.repetition),
        )?;
        sleep_between_processes(experiment);
    }
    Ok(executed)
}

fn run_abba_comparison(
    worker: &Path,
    store: &RunStore,
    experiment: &ExecutableExperiment,
    baseline_run: &Path,
) -> Result<bool> {
    let baseline_run = if baseline_run.is_absolute() {
        baseline_run.to_path_buf()
    } else {
        project_root().join(baseline_run)
    };
    let baseline_store = RunStore::open(&baseline_run)?;
    let comparison_store = store.comparison_baseline_store(&baseline_store)?;
    let mut executed = false;

    for run in experiment.schedule().runs() {
        let complete_store = match run.side {
            ComparisonSide::Baseline => &comparison_store,
            ComparisonSide::Candidate => store,
        };
        if complete_store.repetition_complete(run.repetition)? {
            complete_store.ensure_unmeasured_repetition_record(run.repetition)?;
            continue;
        }
        executed = true;
        store.assert_worker(worker)?;
        match run.side {
            ComparisonSide::Baseline => {
                run_worker_command(
                    worker,
                    &[
                        "mac-experiment",
                        "comparison-worker",
                        &baseline_run.to_string_lossy(),
                        store.run_dir().to_string_lossy().as_ref(),
                        "--repetition",
                        &run.repetition.to_string(),
                    ],
                    &format!("baseline repetition {}", run.repetition),
                    &comparison_store.process_log_path(run.repetition),
                )?;
            }
            ComparisonSide::Candidate => {
                run_worker_command(
                    worker,
                    &[
                        "mac-experiment",
                        "worker",
                        store.run_dir().to_string_lossy().as_ref(),
                        "--repetition",
                        &run.repetition.to_string(),
                    ],
                    &format!("candidate repetition {}", run.repetition),
                    &store.process_log_path(run.repetition),
                )?;
            }
        }
        sleep_between_processes(experiment);
    }
    Ok(executed)
}

fn run_worker_command(worker: &Path, args: &[&str], label: &str, log_path: &Path) -> Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    writeln!(log, "=== {} {label} ===", chrono::Utc::now().to_rfc3339())?;
    log.flush()?;
    let stdout = log.try_clone()?;
    let status = Command::new(worker)
        .args(args)
        .current_dir(project_root())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(log.try_clone()?))
        .status()
        .wrap_err_with(|| format!("failed to start {label} with {}", worker.display()))?;
    log.sync_all()?;
    ensure!(
        status.success(),
        "{label} failed with {status}; see {}",
        log_path.display()
    );
    Ok(())
}

fn sleep_between_processes(experiment: &ExecutableExperiment) {
    if experiment.performance().sleep_seconds() > 0 {
        std::thread::sleep(Duration::from_secs(
            experiment.performance().sleep_seconds(),
        ));
    }
}

pub(super) fn run_worker(run_dir: &Path, repetition: u32) -> Result<()> {
    ensure!(
        cfg!(feature = "coreml"),
        "the macOS experiment worker must be built with the xtask coreml feature"
    );
    let worker_start = Instant::now();
    let store = RunStore::open(run_dir)?;
    let experiment = store.experiment().to_executable()?;
    execute_store_repetition(&store, &experiment, repetition, worker_start)
}

pub(super) fn run_comparison_worker(
    baseline_run_dir: &Path,
    candidate_run_dir: &Path,
    repetition: u32,
) -> Result<()> {
    ensure!(
        cfg!(feature = "coreml"),
        "the comparison worker must be built with the xtask coreml feature"
    );
    let worker_start = Instant::now();
    let baseline_store = RunStore::open(baseline_run_dir)?;
    let candidate_store = RunStore::open(candidate_run_dir)?;
    let store = candidate_store.comparison_baseline_store(&baseline_store)?;
    let experiment = store.experiment().to_executable()?;
    execute_store_repetition(&store, &experiment, repetition, worker_start)
}

fn execute_store_repetition(
    store: &RunStore,
    experiment: &ExecutableExperiment,
    repetition: u32,
    worker_start: Instant,
) -> Result<()> {
    if store.repetition_complete(repetition)? {
        store.ensure_unmeasured_repetition_record(repetition)?;
        return Ok(());
    }
    let resumed = store.repetition_started(repetition)?;

    let mut executor = ExperimentExecutor::new(experiment, worker_start)?;
    executor.warm_up(store.manifest_files().first())?;
    store.execute_repetition(repetition, |_, file, candidates| {
        executor.execute_file(file, candidates)
    })?;
    ensure!(
        store.repetition_complete(repetition)?,
        "repetition {repetition} ended with missing durable records"
    );
    store.write_repetition_record(
        repetition,
        &RepetitionRecord {
            status: "complete",
            resumed,
            whole_process_seconds: worker_start.elapsed().as_secs_f64(),
            peak_rss_bytes: peak_rss_bytes(),
            completed_at: chrono::Utc::now().to_rfc3339(),
        },
    )?;
    Ok(())
}

pub(super) fn run_profile_worker(
    run_dir: &Path,
    file_index: usize,
    stage_output: &Path,
) -> Result<()> {
    ensure!(
        cfg!(feature = "coreml"),
        "the macOS profile worker must be built with the xtask coreml feature"
    );
    let worker_start = Instant::now();
    let store = RunStore::open(run_dir)?;
    let experiment = store.experiment().to_executable()?;
    let file = store
        .manifest_files()
        .get(file_index)
        .ok_or_else(|| eyre!("profile file index {file_index} is outside the manifest"))?;
    let mut executor = ExperimentExecutor::new(&experiment, worker_start)?;
    executor.warm_up(Some(file))?;
    let candidate_ids = experiment
        .candidates()
        .iter()
        .map(|candidate| candidate.id.clone())
        .collect::<Vec<_>>();
    let records = executor.execute_file(file, &candidate_ids)?;
    ensure!(
        records
            .iter()
            .all(|(_, record)| matches!(record, ExperimentRecord::Complete { .. })),
        "profile workload produced a failed candidate record"
    );
    let output = ProfileStageOutput {
        schema_version: 1,
        file_id: file.id.clone(),
        candidates: records
            .into_iter()
            .map(|(candidate_id, record)| ProfileCandidateOutput {
                candidate_id,
                record,
            })
            .collect(),
    };
    let mut bytes = serde_json::to_vec_pretty(&output)?;
    bytes.push(b'\n');
    super::store::atomic_write(stage_output, &bytes)?;
    Ok(())
}

#[derive(Serialize)]
struct ProfileStageOutput {
    schema_version: u32,
    file_id: String,
    candidates: Vec<ProfileCandidateOutput>,
}

#[derive(Serialize)]
struct ProfileCandidateOutput {
    candidate_id: String,
    record: ExperimentRecord,
}

struct ExperimentExecutor<'a> {
    experiment: &'a ExecutableExperiment,
    pipeline: OwnedDiarizationPipeline,
    worker_start: Instant,
}

impl<'a> ExperimentExecutor<'a> {
    fn new(experiment: &'a ExecutableExperiment, worker_start: Instant) -> Result<Self> {
        let inference = experiment.inference();
        let mode = execution_mode(inference.mode());
        let inference_config = inference.to_pipeline()?;
        let runtime = RuntimeConfig::default().with_experiment(inference_config);
        let pipeline = PipelineBuilder::from_dir(experiment.models_dir(), mode)?
            .runtime(runtime)
            .build()
            .map_err(|error| eyre!("failed to build experiment pipeline: {error}"))?;

        ensure!(
            (pipeline.segmentation_step() - inference.step_seconds()).abs() <= 1e-9,
            "pipeline step {} does not match experiment layout step {}",
            pipeline.segmentation_step(),
            inference.step_seconds()
        );

        Ok(Self {
            experiment,
            pipeline,
            worker_start,
        })
    }

    fn warm_up(&mut self, file: Option<&ManifestFile>) -> Result<()> {
        let Some(file) = file else {
            return Ok(());
        };
        if self.experiment.performance().warmups() == 0 {
            return Ok(());
        }

        let (audio, sample_rate) = load_audio(file)?;
        ensure_sample_rate(file, sample_rate)?;
        let candidate = self
            .experiment
            .candidates()
            .first()
            .ok_or_else(|| eyre!("experiment has no post-inference candidate"))?;
        let config = candidate_config(&self.pipeline, candidate)?;
        for _ in 0..self.experiment.performance().warmups() {
            let artifacts = self
                .pipeline
                .run_inference_only(&audio)
                .map_err(|error| eyre!("warm-up inference failed: {error}"))?;
            self.pipeline
                .finish_post_inference(artifacts, &config)
                .map_err(|error| eyre!("warm-up post-inference failed: {error}"))?;
        }
        Ok(())
    }

    fn execute_file(
        &mut self,
        file: &ManifestFile,
        candidate_ids: &[String],
    ) -> Result<Vec<(String, ExperimentRecord)>> {
        let audio_seconds = file.duration_seconds;
        let (audio, sample_rate) = match load_audio(file) {
            Ok(value) => value,
            Err(error) => {
                return Ok(self.failure_records(candidate_ids, audio_seconds, None, error));
            }
        };
        if let Err(error) = ensure_sample_rate(file, sample_rate) {
            return Ok(self.failure_records(candidate_ids, audio_seconds, None, error));
        }

        let inference_start = Instant::now();
        let artifacts = match self.pipeline.run_inference_only(&audio) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                return Ok(self.failure_records(
                    candidate_ids,
                    audio_seconds,
                    None,
                    eyre!("inference failed: {error}"),
                ));
            }
        };
        let inference_seconds = inference_start.elapsed().as_secs_f64();
        let chunk_inference = artifacts.stage_timings().map(Into::into);

        let mut records = Vec::with_capacity(candidate_ids.len());
        for candidate_id in candidate_ids {
            let record = self.execute_candidate(
                file,
                candidate_id,
                audio_seconds,
                inference_seconds,
                chunk_inference,
                artifacts.clone(),
            );
            records.push((candidate_id.clone(), record));
        }
        Ok(records)
    }

    fn execute_candidate(
        &self,
        file: &ManifestFile,
        candidate_id: &str,
        audio_seconds: f64,
        inference_seconds: f64,
        chunk_inference: Option<ChunkInferenceStageTimings>,
        artifacts: speakrs::pipeline::InferenceArtifacts,
    ) -> ExperimentRecord {
        let result = (|| -> Result<ExperimentRecord> {
            let candidate = self
                .experiment
                .candidates()
                .iter()
                .find(|candidate| candidate.id == candidate_id)
                .ok_or_else(|| eyre!("unknown post-inference candidate '{candidate_id}'"))?;
            let config = candidate_config(&self.pipeline, candidate)?;
            let usable_training_embeddings =
                artifacts.usable_training_embedding_count(config.clean_frame_duration);
            let post_start = Instant::now();
            let output = self
                .pipeline
                .finish_post_inference(artifacts, &config)
                .map_err(|error| eyre!("post-inference failed: {error}"))?;
            let post_inference_seconds = post_start.elapsed().as_secs_f64();
            let hypothesis_rttm = output.rttm(&file.id);
            let der = score_file(file, &hypothesis_rttm)?;
            let wall_seconds = inference_seconds + post_inference_seconds;
            let backend = match config.clustering_backend() {
                ClusteringBackend::GaussianVbx(vbx) => ClusteringBackendDiagnostics::GaussianVbx {
                    fa: vbx.fa(),
                    fb: vbx.fb(),
                    max_iters: vbx.max_iters(),
                },
                ClusteringBackend::SphereVbxPf(sphere) => {
                    let initialization = match sphere.initialization() {
                        speakrs::pipeline::SphereVbxInitialization::Hard => {
                            SphereInitializationDiagnostics::Hard
                        }
                        speakrs::pipeline::SphereVbxInitialization::Smoothed(smoothing) => {
                            SphereInitializationDiagnostics::Smoothed {
                                scale: smoothing.get(),
                            }
                        }
                        _ => return Err(eyre!("unsupported SphereVBx-PF initialization")),
                    };
                    let ahc_initialization = match sphere.ahc_initialization() {
                        speakrs::pipeline::SphereVbxAhcInitialization::Cosine => {
                            SphereAhcInitializationDiagnostics::Cosine
                        }
                        speakrs::pipeline::SphereVbxAhcInitialization::PldaTransformed => {
                            SphereAhcInitializationDiagnostics::PldaTransformed
                        }
                    };
                    ClusteringBackendDiagnostics::SphereVbxPf {
                        fa: sphere.fa(),
                        fb: sphere.fb(),
                        max_iters: sphere.max_iters(),
                        responsibility_tolerance: sphere.responsibility_tolerance().get(),
                        initialization,
                        ahc_initialization,
                    }
                }
                _ => return Err(eyre!("unsupported clustering backend")),
            };

            Ok(ExperimentRecord::Complete {
                audio_seconds,
                wall_seconds,
                rtfx: audio_seconds / wall_seconds,
                worker_elapsed_seconds: self.worker_start.elapsed().as_secs_f64(),
                stage_timings: StageTimings {
                    inference_seconds,
                    post_inference_seconds,
                    chunk_inference,
                },
                clustering: Some(ClusteringDiagnostics {
                    usable_training_embeddings,
                    clean_frame_seconds: config.clean_frame_duration.seconds(),
                    backend: Some(backend),
                }),
                peak_rss_bytes: peak_rss_bytes(),
                der: Box::new(der),
                hypothesis_rttm,
                fallback_events: Vec::new(),
            })
        })();

        result.unwrap_or_else(|error| ExperimentRecord::Failed {
            error: format!("{error:#}"),
            audio_seconds,
            worker_elapsed_seconds: self.worker_start.elapsed().as_secs_f64(),
            inference_seconds: Some(inference_seconds),
            peak_rss_bytes: peak_rss_bytes(),
        })
    }

    fn failure_records(
        &self,
        candidate_ids: &[String],
        audio_seconds: f64,
        inference_seconds: Option<f64>,
        error: color_eyre::Report,
    ) -> Vec<(String, ExperimentRecord)> {
        let error = format!("{error:#}");
        candidate_ids
            .iter()
            .map(|candidate_id| {
                (
                    candidate_id.clone(),
                    ExperimentRecord::Failed {
                        error: error.clone(),
                        audio_seconds,
                        worker_elapsed_seconds: self.worker_start.elapsed().as_secs_f64(),
                        inference_seconds,
                        peak_rss_bytes: peak_rss_bytes(),
                    },
                )
            })
            .collect()
    }
}

fn candidate_config(
    pipeline: &OwnedDiarizationPipeline,
    candidate: &CheckedCandidate,
) -> Result<speakrs::pipeline::PipelineConfig> {
    apply_candidate_config(pipeline.pipeline_config(), candidate)
}

fn apply_candidate_config(
    mut config: speakrs::pipeline::PipelineConfig,
    candidate: &CheckedCandidate,
) -> Result<speakrs::pipeline::PipelineConfig> {
    match candidate.clustering {
        CandidateClustering::Gaussian(overrides) => {
            let mut vbx = match config.clustering_backend() {
                ClusteringBackend::GaussianVbx(vbx) => vbx,
                _ => speakrs::VbxConfig::default(),
            };
            if let Some(max_iters) = overrides.max_iters {
                vbx = vbx
                    .with_max_iters(max_iters.get())
                    .map_err(|error| eyre!(error))?;
            }
            if let Some(fb) = overrides.fb {
                vbx = vbx.with_fb(fb).map_err(|error| eyre!(error))?;
            }
            config = config.with_clustering(ClusteringBackend::GaussianVbx(vbx));
        }
        CandidateClustering::Sphere(sphere) => {
            config = config.with_clustering(ClusteringBackend::SphereVbxPf(sphere));
        }
    }
    if let Some(duration) = candidate.clean_frame_duration {
        config.clean_frame_duration = duration;
    }
    Ok(config)
}

fn load_audio(file: &ManifestFile) -> Result<(Vec<f32>, u32)> {
    load_wav_samples(&file.wav.to_string_lossy())
        .wrap_err_with(|| format!("failed to load WAV {}", file.wav.display()))
}

fn ensure_sample_rate(file: &ManifestFile, sample_rate: u32) -> Result<()> {
    ensure!(
        sample_rate == REQUIRED_SAMPLE_RATE,
        "WAV {} uses {sample_rate} Hz; expected {REQUIRED_SAMPLE_RATE} Hz",
        file.wav.display()
    );
    Ok(())
}

fn score_file(file: &ManifestFile, hypothesis_rttm: &str) -> Result<PerFileDerResult> {
    let files = vec![(file.wav.clone(), file.rttm.clone())];
    let hypotheses = HashMap::from([(file.id.clone(), hypothesis_rttm.to_owned())]);
    let accumulation = DerAccumulation::compute(&files, &hypotheses)?;
    accumulation
        .per_file()
        .first()
        .cloned()
        .ok_or_else(|| eyre!("DER calculation returned no result for '{}'", file.id))
}

const fn execution_mode(mode: CoreMlMode) -> ExecutionMode {
    match mode {
        CoreMlMode::CoreMl => ExecutionMode::CoreMl,
        CoreMlMode::CoreMlFast => ExecutionMode::CoreMlFast,
    }
}

pub(super) fn build_worker_binary() -> Result<PathBuf> {
    let root = project_root();
    let target_dir = root.join("target/mac-experiment");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--package",
            "xtask",
            "--features",
            "coreml",
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .current_dir(&root)
        .status()
        .wrap_err("failed to build the macOS experiment worker")?;
    ensure!(
        status.success(),
        "macOS experiment worker build failed with {status}"
    );
    let worker = target_dir.join("release/xtask");
    ensure!(
        worker.is_file(),
        "worker binary does not exist: {}",
        worker.display()
    );
    Ok(worker)
}

fn peak_rss_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // safety: getrusage writes the complete rusage value for the current process
        let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if status != 0 {
            return None;
        }
        // safety: a successful getrusage call initialized the value
        let usage = unsafe { usage.assume_init() };
        #[cfg(target_os = "macos")]
        return u64::try_from(usage.ru_maxrss).ok();
        #[cfg(not(target_os = "macos"))]
        return u64::try_from(usage.ru_maxrss).ok()?.checked_mul(1024);
    }
    #[cfg(not(unix))]
    None
}

#[cfg(test)]
mod tests {
    use super::super::domain::PostInferenceVariant;
    use super::super::domain::{
        CandidateClustering, EmbeddingComputeUnits, ExecutableInference, FbankNormalizationScope,
        FbankPreparationWorkers, GaussianOverrides, InferenceWorkers, SegmentationWorkers,
        parse_checked_candidate,
    };
    use super::*;

    fn default_candidate() -> CheckedCandidate {
        parse_checked_candidate(&PostInferenceVariant {
            id: "default".to_owned(),
            vbx_max_iters: None,
            vbx_fb: None,
            clean_frame_seconds: None,
            ahc_stopping: None,
            clustering_backend: None,
            documented_der_outliers: Vec::new(),
        })
        .unwrap()
    }

    #[test]
    fn omitted_vbx_iterations_preserve_fast_product_default() {
        let product = speakrs::pipeline::PipelineConfig::for_mode(ExecutionMode::CoreMlFast);
        let config = apply_candidate_config(product, &default_candidate()).unwrap();

        match config.clustering_backend() {
            ClusteringBackend::GaussianVbx(vbx) => assert_eq!(vbx.max_iters(), 3),
            _ => panic!("expected gaussian backend"),
        }
        assert!(matches!(
            default_candidate().clustering,
            CandidateClustering::Gaussian(GaussianOverrides {
                max_iters: None,
                fb: None
            })
        ));
    }

    #[test]
    fn cpu_only_experiment_value_reaches_runtime_config() {
        let inference = ExecutableInference::PerWindowOneSecond {
            workers: InferenceWorkers {
                segmentation_workers: SegmentationWorkers::Automatic,
                filterbank_preparation_workers: FbankPreparationWorkers::Two,
                filterbank_normalization_scope: FbankNormalizationScope::Chunk,
                embedding_compute_units: EmbeddingComputeUnits::CpuOnly,
            },
        };
        assert_eq!(
            inference.to_pipeline().unwrap().embedding_compute_units,
            speakrs::inference::CoreMlComputeUnits::CpuOnly
        );
    }
}
