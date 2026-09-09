use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::{Context, Result, bail, ensure, eyre};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::cmd::project_root;
use crate::commands::benchmark::discover_files;
use crate::path::file_stem_string;

use super::ValidatedExperiment;
use super::identity::{HostIdentity, digest_paths, digest_paths_cached};
use super::statistics::{
    FileDerObservation, dataset_noise_margin, material_speed_rule, median,
    median_absolute_deviation, pair_file_observations, paired_file_bootstrap,
    speed_improvement_percent,
};

const MANIFEST_SCHEMA_VERSION: u32 = 1;
const RECORD_SCHEMA_VERSION: u32 = 1;
const RECORDS_DIR: &str = "records";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ManifestFile {
    pub id: String,
    #[serde(alias = "wav_path")]
    pub wav: PathBuf,
    #[serde(alias = "rttm_path")]
    pub rttm: PathBuf,
    pub duration_seconds: f64,
}

impl ManifestFile {
    fn record_component(&self) -> String {
        path_component(&self.id)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ManifestIdentity {
    #[serde(default)]
    host: Option<HostIdentity>,
    #[serde(default)]
    model_sha256: String,
    #[serde(default)]
    worker_sha256: String,
    #[serde(default)]
    dataset_sha256: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    cache_state: String,
    #[serde(default)]
    run_order: RunOrder,
    #[serde(default)]
    fallback_events: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunOrder {
    repetitions: Vec<u32>,
    files: Vec<String>,
    candidates: Vec<String>,
    #[serde(default)]
    comparison_processes: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunManifest {
    schema_version: u32,
    spec: super::MacExperimentSpec,
    files: Vec<ManifestFile>,
    #[serde(default)]
    identity: Option<ManifestIdentity>,
}

pub(super) struct RunStore {
    run_dir: PathBuf,
    manifest: RunManifest,
}

impl RunStore {
    pub(super) fn create(experiment: &ValidatedExperiment, worker: &Path) -> Result<Self> {
        let root = project_root();
        let files = resolve_dataset_files(experiment)?;
        let identity = build_identity(experiment, &files, worker, &root)?;
        let manifest = RunManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            spec: experiment.spec().clone(),
            files,
            identity: Some(identity),
        };
        let run_dir = root.join("_benchmarks").join("macos").join(experiment.id());
        let parent = run_dir
            .parent()
            .ok_or_else(|| eyre!("run directory has no parent: {}", run_dir.display()))?;
        fs::create_dir_all(parent).wrap_err_with(|| {
            format!("failed to create benchmark directory {}", parent.display())
        })?;
        create_run_dir(&run_dir)?;
        write_manifest(&run_dir, &manifest)?;
        Ok(Self { run_dir, manifest })
    }

    pub(super) fn open(run_dir: &Path) -> Result<(Self, ValidatedExperiment)> {
        ensure!(
            run_dir.is_dir(),
            "run directory does not exist: {}",
            run_dir.display()
        );
        let manifest_path = run_dir.join("manifest.json");
        let text = fs::read_to_string(&manifest_path)
            .wrap_err_with(|| format!("failed to read manifest {}", manifest_path.display()))?;
        let manifest: RunManifest = serde_json::from_str(&text)
            .wrap_err_with(|| format!("invalid run manifest {}", manifest_path.display()))?;
        ensure!(
            manifest.schema_version == MANIFEST_SCHEMA_VERSION,
            "manifest schema_version {} is not supported; expected {MANIFEST_SCHEMA_VERSION}",
            manifest.schema_version
        );
        let experiment = ValidatedExperiment::from_spec(manifest.spec.clone())?;
        let directory_id = run_dir.file_name().and_then(OsStr::to_str).ok_or_else(|| {
            eyre!(
                "run directory has no valid experiment id: {}",
                run_dir.display()
            )
        })?;
        ensure!(
            experiment.id() == directory_id,
            "run directory identity '{}' does not match manifest experiment_id '{}'",
            directory_id,
            experiment.id()
        );
        validate_manifest_files(&manifest, &experiment)?;
        let store = Self {
            run_dir: run_dir.to_path_buf(),
            manifest,
        };
        // resume and summarize must reject model or dataset drift before mixing records
        store.assert_input_digests(&experiment)?;
        Ok((store, experiment))
    }

    #[cfg(test)]
    pub(super) fn execute<F, T>(
        &self,
        experiment: ValidatedExperiment,
        mut executor: F,
    ) -> Result<()>
    where
        F: FnMut(u32, &ManifestFile, &[String]) -> Result<Vec<(String, T)>>,
        T: Serialize,
    {
        self.validate_experiment(&experiment)?;
        for repetition in 0..experiment.performance().repetitions() {
            self.execute_repetition(&experiment, repetition, &mut executor)?;
        }
        self.rebuild_projections(&experiment)
    }

    pub(super) fn execute_repetition<F, T>(
        &self,
        experiment: &ValidatedExperiment,
        repetition: u32,
        mut executor: F,
    ) -> Result<()>
    where
        F: FnMut(u32, &ManifestFile, &[String]) -> Result<Vec<(String, T)>>,
        T: Serialize,
    {
        self.validate_experiment(experiment)?;
        ensure!(
            repetition < experiment.performance().repetitions(),
            "repetition {repetition} is outside the manifest protocol"
        );

        for (file_index, file) in self.manifest.files.iter().enumerate() {
            let missing = self.missing_candidate_ids(repetition, file_index, experiment)?;
            if missing.is_empty() {
                continue;
            }
            let payloads = executor(repetition, file, &missing).wrap_err_with(|| {
                format!(
                    "execution failed for repetition {repetition}, file '{}'",
                    file.id
                )
            })?;
            validate_payload_ids(&missing, &payloads, repetition, file_index)?;
            for (candidate_id, payload) in payloads {
                self.write_record(repetition, file_index, &candidate_id, &payload)?;
            }
        }

        Ok(())
    }

    pub(super) fn rebuild_projections(&self, experiment: &ValidatedExperiment) -> Result<()> {
        self.validate_experiment(experiment)?;
        let records = self.read_records(experiment)?;
        let projections = ProjectionSummary::from_records(self, &records)?;
        let files_jsonl = records
            .iter()
            .map(|record| serde_json::to_string(&record.value))
            .collect::<serde_json::Result<Vec<_>>>();
        let files_jsonl = files_jsonl?.join("\n");
        let files_jsonl = if files_jsonl.is_empty() {
            String::new()
        } else {
            format!("{files_jsonl}\n")
        };
        atomic_write(&self.run_dir.join("files.jsonl"), files_jsonl.as_bytes())?;
        let mut summary = serde_json::to_vec_pretty(&projections)?;
        summary.push(b'\n');
        atomic_write(&self.run_dir.join("summary.json"), &summary)?;
        let report = projections.report(&self.manifest);
        atomic_write(&self.run_dir.join("report.md"), report.as_bytes())
    }

    pub(super) fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    pub(super) fn manifest_files(&self) -> &[ManifestFile] {
        &self.manifest.files
    }

    pub(super) fn process_log_path(&self, repetition: u32) -> PathBuf {
        self.run_dir
            .join("process-logs")
            .join(format!("r{repetition:03}.log"))
    }

    pub(super) fn assert_worker(&self, worker: &Path) -> Result<()> {
        let identity = self.require_identity()?;
        ensure!(
            !identity.worker_sha256.is_empty(),
            "run manifest has no worker digest; create a new experiment run"
        );
        let actual = digest_paths(&project_root(), &[worker.to_path_buf()])?;
        ensure!(
            actual == identity.worker_sha256,
            "experiment worker changed after manifest creation"
        );
        Ok(())
    }

    pub(super) fn assert_input_digests(&self, experiment: &ValidatedExperiment) -> Result<()> {
        let identity = self.require_identity()?;
        ensure!(
            !identity.model_sha256.is_empty(),
            "run manifest has no model digest; create a new experiment run"
        );
        ensure!(
            !identity.dataset_sha256.is_empty(),
            "run manifest has no dataset digest; create a new experiment run"
        );
        let root = project_root();
        let actual_model = model_digest(experiment, &root)?;
        ensure!(
            actual_model == identity.model_sha256,
            "experiment models changed after manifest creation"
        );
        let actual_dataset = dataset_digest(&self.manifest.files, &root)?;
        ensure!(
            actual_dataset == identity.dataset_sha256,
            "experiment dataset changed after manifest creation"
        );
        Ok(())
    }

    pub(super) fn comparison_baseline_store(
        &self,
        source: &RunStore,
    ) -> Result<(RunStore, ValidatedExperiment)> {
        let run_dir = self.run_dir.join("comparison-baseline");
        let manifest = if run_dir.is_dir() {
            let manifest_path = run_dir.join("manifest.json");
            let stored: RunManifest = serde_json::from_reader(BufReader::new(
                File::open(&manifest_path).wrap_err_with(|| {
                    format!(
                        "failed to open comparison manifest {}",
                        manifest_path.display()
                    )
                })?,
            ))?;
            stored
        } else {
            fs::create_dir_all(&run_dir)?;
            let mut manifest = source.manifest.clone();
            let source_files: HashMap<_, _> = manifest
                .files
                .iter()
                .cloned()
                .map(|file| (file.id.clone(), file))
                .collect();
            manifest.files = self
                .manifest
                .files
                .iter()
                .map(|candidate_file| {
                    source_files
                        .get(&candidate_file.id)
                        .cloned()
                        .ok_or_else(|| {
                            eyre!(
                                "baseline run has no file '{}' required by the candidate",
                                candidate_file.id
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            manifest.spec.dataset.files =
                manifest.files.iter().map(|file| file.id.clone()).collect();
            manifest.spec.dataset.max_files = manifest.files.len() as u32;
            manifest.spec.dataset.max_minutes = self.manifest.spec.dataset.max_minutes;
            manifest.spec.performance = self.manifest.spec.performance.clone();
            // keep baseline asset digests; pin only the candidate seed and comparison run-order
            if let Some(identity) = &mut manifest.identity {
                if let Some(candidate_identity) = &self.manifest.identity {
                    identity.seed = candidate_identity.seed;
                }
                identity.run_order.repetitions =
                    (0..manifest.spec.performance.repetitions).collect();
                identity.run_order.files =
                    manifest.files.iter().map(|file| file.id.clone()).collect();
                identity.run_order.candidates = manifest
                    .spec
                    .post_inference
                    .iter()
                    .map(|candidate| candidate.id.clone())
                    .collect();
                identity.run_order.comparison_processes.clear();
                identity.created_at = chrono::Utc::now().to_rfc3339();
            }
            write_manifest(&run_dir, &manifest)?;
            manifest
        };
        ensure!(
            manifest.spec.performance.repetitions == self.manifest.spec.performance.repetitions,
            "comparison baseline repetition count does not match the candidate"
        );
        let experiment = ValidatedExperiment::from_spec(manifest.spec.clone())?;

        Ok((RunStore { run_dir, manifest }, experiment))
    }

    pub(super) fn record_path(
        &self,
        repetition: u32,
        file_index: usize,
        candidate_id: &str,
    ) -> PathBuf {
        let file = &self.manifest.files[file_index];
        self.run_dir
            .join(RECORDS_DIR)
            .join(format!("r{repetition:03}"))
            .join(file.record_component())
            .join(format!("{candidate_id}.json"))
    }

    pub(super) fn record_exists(
        &self,
        repetition: u32,
        file_index: usize,
        candidate_id: &str,
    ) -> bool {
        self.record_path(repetition, file_index, candidate_id)
            .is_file()
    }

    pub(super) fn repetition_complete(
        &self,
        repetition: u32,
        experiment: &ValidatedExperiment,
    ) -> Result<bool> {
        self.validate_experiment(experiment)?;
        ensure!(
            repetition < experiment.performance().repetitions(),
            "repetition {repetition} is outside the manifest protocol"
        );

        Ok(self
            .manifest
            .files
            .iter()
            .enumerate()
            .all(|(file_index, _)| {
                experiment
                    .post_inference()
                    .iter()
                    .all(|candidate| self.record_exists(repetition, file_index, &candidate.id))
            }))
    }

    pub(super) fn repetition_started(
        &self,
        repetition: u32,
        experiment: &ValidatedExperiment,
    ) -> Result<bool> {
        self.validate_experiment(experiment)?;
        ensure!(
            repetition < experiment.performance().repetitions(),
            "repetition {repetition} is outside the manifest protocol"
        );

        Ok(self
            .manifest
            .files
            .iter()
            .enumerate()
            .any(|(file_index, _)| {
                experiment
                    .post_inference()
                    .iter()
                    .any(|candidate| self.record_exists(repetition, file_index, &candidate.id))
            }))
    }

    pub(super) fn write_repetition_record<T: Serialize>(
        &self,
        repetition: u32,
        payload: &T,
    ) -> Result<()> {
        ensure!(
            repetition < self.manifest.spec.performance.repetitions,
            "repetition {repetition} is outside the manifest protocol"
        );
        let path = self
            .run_dir
            .join("repetitions")
            .join(format!("r{repetition:03}.json"));
        if path.is_file() {
            return Ok(());
        }

        let mut value = serde_json::to_value(payload)?;
        ensure!(
            value.is_object(),
            "repetition payload must be a JSON object"
        );
        let object = value
            .as_object_mut()
            .ok_or_else(|| eyre!("repetition payload must be a JSON object"))?;
        object.insert(
            "repetition".to_owned(),
            Value::Number(serde_json::Number::from(repetition)),
        );
        object.insert(
            "schema_version".to_owned(),
            Value::Number(serde_json::Number::from(RECORD_SCHEMA_VERSION)),
        );
        let mut bytes = serde_json::to_vec(&value)?;
        bytes.push(b'\n');
        atomic_write_new(&path, &bytes)
    }

    pub(super) fn missing_candidate_ids(
        &self,
        repetition: u32,
        file_index: usize,
        experiment: &ValidatedExperiment,
    ) -> Result<Vec<String>> {
        self.validate_experiment(experiment)?;
        ensure!(
            repetition < experiment.performance().repetitions(),
            "repetition {repetition} is outside the manifest protocol"
        );
        ensure!(
            file_index < self.manifest.files.len(),
            "file index {file_index} is outside the manifest"
        );
        Ok(experiment
            .post_inference()
            .iter()
            .filter(|candidate| !self.record_exists(repetition, file_index, &candidate.id))
            .map(|candidate| candidate.id.clone())
            .collect())
    }

    pub(super) fn write_record<T: Serialize>(
        &self,
        repetition: u32,
        file_index: usize,
        candidate_id: &str,
        payload: &T,
    ) -> Result<()> {
        ensure!(
            repetition < self.manifest.spec.performance.repetitions,
            "repetition {repetition} is outside the manifest protocol"
        );
        let file = self
            .manifest
            .files
            .get(file_index)
            .ok_or_else(|| eyre!("file index {file_index} is outside the manifest"))?;
        ensure!(
            self.manifest
                .spec
                .post_inference
                .iter()
                .any(|candidate| candidate.id == candidate_id),
            "candidate '{candidate_id}' is not part of the manifest"
        );
        ensure!(
            is_safe_component(candidate_id),
            "unsafe candidate id '{candidate_id}'"
        );
        let path = self.record_path(repetition, file_index, candidate_id);
        if path.exists() {
            bail!("record already exists: {}", path.display());
        }
        let mut value = serde_json::to_value(payload)?;
        if !value.is_object() {
            let payload = std::mem::replace(&mut value, Value::Null);
            let mut object = Map::new();
            object.insert("payload".to_owned(), payload);
            value = Value::Object(object);
        }
        let object = value
            .as_object_mut()
            .ok_or_else(|| eyre!("record payload must be a JSON object"))?;
        object.insert(
            "candidate_id".to_owned(),
            Value::String(candidate_id.to_owned()),
        );
        object.insert("file_id".to_owned(), Value::String(file.id.clone()));
        object.insert(
            "file_index".to_owned(),
            Value::Number(serde_json::Number::from(file_index)),
        );
        object.insert(
            "repetition".to_owned(),
            Value::Number(serde_json::Number::from(repetition)),
        );
        object.insert(
            "schema_version".to_owned(),
            Value::Number(serde_json::Number::from(RECORD_SCHEMA_VERSION)),
        );
        let mut bytes = serde_json::to_vec(&value)?;
        bytes.push(b'\n');
        atomic_write_new(&path, &bytes)
    }

    fn require_identity(&self) -> Result<&ManifestIdentity> {
        self.manifest
            .identity
            .as_ref()
            .ok_or_else(|| eyre!("run manifest has no execution identity"))
    }

    fn validate_experiment(&self, experiment: &ValidatedExperiment) -> Result<()> {
        ensure!(
            experiment.id() == self.manifest.spec.experiment_id,
            "experiment identity '{}' does not match manifest experiment_id '{}'",
            experiment.id(),
            self.manifest.spec.experiment_id
        );
        let expected = serde_json::to_value(&self.manifest.spec)?;
        let actual = serde_json::to_value(experiment.spec())?;
        ensure!(
            expected == actual,
            "experiment specification does not match immutable run manifest"
        );
        Ok(())
    }

    fn read_records(&self, experiment: &ValidatedExperiment) -> Result<Vec<StoredRecord>> {
        let records_dir = self.run_dir.join(RECORDS_DIR);
        if !records_dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths = Vec::new();
        collect_json_paths(&records_dir, &mut paths)?;
        paths.sort();
        let candidate_order: HashMap<_, _> = experiment
            .post_inference()
            .iter()
            .enumerate()
            .map(|(index, candidate)| (candidate.id.as_str(), index))
            .collect();
        let file_order: HashMap<_, _> = self
            .manifest
            .files
            .iter()
            .enumerate()
            .map(|(index, file)| (file.id.as_str(), index))
            .collect();
        let mut seen_targets = HashSet::new();
        let mut records = Vec::with_capacity(paths.len());
        for path in paths {
            let value: Value = serde_json::from_reader(BufReader::new(
                File::open(&path)
                    .wrap_err_with(|| format!("failed to read record {}", path.display()))?,
            ))
            .wrap_err_with(|| format!("invalid record {}", path.display()))?;
            let object = value
                .as_object()
                .ok_or_else(|| eyre!("record {} must contain a JSON object", path.display()))?;
            let schema_version = json_u32(object, "schema_version", &path)?;
            ensure!(
                schema_version == RECORD_SCHEMA_VERSION,
                "record {} schema_version {} is not supported; expected {RECORD_SCHEMA_VERSION}",
                path.display(),
                schema_version
            );
            let repetition = json_u32(object, "repetition", &path)?;
            let file_index = json_usize(object, "file_index", &path)?;
            let file_id = json_string(object, "file_id", &path)?;
            let candidate_id = json_string(object, "candidate_id", &path)?;
            ensure!(
                repetition < experiment.performance().repetitions(),
                "record {} repetition {} is outside the manifest",
                path.display(),
                repetition
            );
            let expected_file = self.manifest.files.get(file_index).ok_or_else(|| {
                eyre!(
                    "record {} file index {} is outside the manifest",
                    path.display(),
                    file_index
                )
            })?;
            ensure!(
                expected_file.id == file_id,
                "record {} file identity '{}' does not match manifest file '{}'",
                path.display(),
                file_id,
                expected_file.id
            );
            let candidate_index = *candidate_order.get(candidate_id.as_str()).ok_or_else(|| {
                eyre!(
                    "record {} candidate '{}' is not part of the manifest",
                    path.display(),
                    candidate_id
                )
            })?;
            let path_key = parse_record_path(&path, &self.run_dir)?;
            ensure!(
                path_key.repetition == repetition
                    && path_key.file_component == expected_file.record_component()
                    && path_key.candidate_id == candidate_id,
                "record path {} does not match its identity fields",
                path.display()
            );
            ensure!(
                file_order.get(file_id.as_str()) == Some(&file_index),
                "record {} has an unknown file identity",
                path.display()
            );
            let key = (repetition, file_index, candidate_index);
            ensure!(
                seen_targets.insert(key),
                "duplicate record target in {}",
                path.display()
            );
            records.push(StoredRecord {
                value,
                repetition,
                file_index,
                candidate_index,
            });
        }
        records
            .sort_by_key(|record| (record.repetition, record.file_index, record.candidate_index));
        Ok(records)
    }
}

#[derive(Clone, Debug)]
struct StoredRecord {
    value: Value,
    repetition: u32,
    file_index: usize,
    candidate_index: usize,
}

#[derive(Debug, Serialize)]
struct ProjectionSummary {
    schema_version: u32,
    experiment_id: String,
    status: &'static str,
    completed_records: usize,
    expected_records: usize,
    repetitions: u32,
    dataset_noise_margin_der: f64,
    files: Vec<FileProjection>,
    candidates: Vec<CandidateProjection>,
}

#[derive(Debug, Serialize)]
struct FileProjection {
    index: usize,
    id: String,
    records: usize,
}

#[derive(Debug, Serialize)]
struct CandidateProjection {
    index: usize,
    id: String,
    records: usize,
    successful_records: usize,
    failed_records: usize,
    duration_weighted_der: Option<f64>,
    missed_percent: Option<f64>,
    false_alarm_percent: Option<f64>,
    confusion_percent: Option<f64>,
    usable_training_embeddings: Option<CountRangeProjection>,
    median_workload_seconds: Option<f64>,
    workload_mad_seconds: Option<f64>,
    median_post_inference_seconds: Option<f64>,
    post_inference_mad_seconds: Option<f64>,
    comparison: Option<ComparisonProjection>,
}

#[derive(Debug, Serialize)]
struct CountRangeProjection {
    minimum: usize,
    median: f64,
    maximum: usize,
}

#[derive(Debug, Serialize)]
struct ComparisonProjection {
    baseline_id: String,
    seed: u64,
    bootstrap_samples: usize,
    baseline_der: f64,
    candidate_der: f64,
    der_improvement: f64,
    baseline_der_interval: IntervalProjection,
    candidate_der_interval: IntervalProjection,
    der_improvement_interval: IntervalProjection,
    clear_der_improvement: bool,
    non_inferior: bool,
    speed_improvement_percent: Option<f64>,
    speed_improvement_mad_percent: Option<f64>,
    material_speed_improvement: bool,
    per_file_der_guard_passed: bool,
    per_file_der_changes: Vec<FileDerChangeProjection>,
    accepted: bool,
}

#[derive(Debug, Serialize)]
struct FileDerChangeProjection {
    file_id: String,
    baseline_der: f64,
    candidate_der: f64,
    change: f64,
    reference_speaker_count: usize,
    baseline_predicted_speaker_count: usize,
    candidate_predicted_speaker_count: usize,
    baseline_speaker_count_error: isize,
    candidate_speaker_count_error: isize,
    documented_cause: Option<String>,
}

#[derive(Debug, Serialize)]
struct IntervalProjection {
    lower: f64,
    upper: f64,
}

struct CandidateMetrics {
    observations: Vec<FileDerObservation>,
    successful_records: usize,
    failed_records: usize,
    duration_weighted_der: Option<f64>,
    missed_percent: Option<f64>,
    false_alarm_percent: Option<f64>,
    confusion_percent: Option<f64>,
    usable_training_embeddings: Option<CountRangeProjection>,
    repetition_wall_seconds: BTreeMap<u32, f64>,
    median_workload_seconds: Option<f64>,
    workload_mad_seconds: Option<f64>,
    median_post_inference_seconds: Option<f64>,
    post_inference_mad_seconds: Option<f64>,
    speaker_counts: BTreeMap<String, SpeakerCountMeasurement>,
}

#[derive(Deserialize)]
struct CompleteRecordMeasurement {
    stage_timings: RecordStageTimings,
    #[serde(default)]
    clustering: Option<RecordClusteringDiagnostics>,
    der: DerMeasurement,
}

#[derive(Deserialize)]
struct RecordClusteringDiagnostics {
    usable_training_embeddings: usize,
}

#[derive(Deserialize)]
struct RecordStageTimings {
    post_inference_seconds: f64,
}

#[derive(Deserialize)]
struct DerMeasurement {
    reference_speaker_time: f64,
    missed: f64,
    false_alarm: f64,
    confusion: f64,
    reference_speaker_count: usize,
    predicted_speaker_count: usize,
}

#[derive(Clone, Copy)]
struct SpeakerCountMeasurement {
    reference: usize,
    predicted: usize,
}

impl ProjectionSummary {
    fn from_records(store: &RunStore, records: &[StoredRecord]) -> Result<Self> {
        let manifest = &store.manifest;
        let repetition_wall_seconds = read_repetition_wall_seconds(&store.run_dir)?;
        let mut file_counts = vec![0; manifest.files.len()];
        let mut candidate_counts = vec![0; manifest.spec.post_inference.len()];
        for record in records {
            file_counts[record.file_index] += 1;
            candidate_counts[record.candidate_index] += 1;
        }
        let expected_records = manifest.spec.performance.repetitions as usize
            * manifest.files.len()
            * manifest.spec.post_inference.len();
        let status = if records.len() == expected_records {
            "complete"
        } else {
            "incomplete"
        };

        let metrics = (0..manifest.spec.post_inference.len())
            .map(|candidate_index| {
                CandidateMetrics::from_records(
                    manifest,
                    records,
                    candidate_index,
                    &repetition_wall_seconds,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let external_baseline = load_external_baseline(store)?;
        let dataset_noise_margin_der = comparison_noise_margin(
            external_baseline
                .as_ref()
                .map(|baseline| baseline.noise_margin),
        );
        let baseline_id = external_baseline
            .as_ref()
            .map(|baseline| baseline.id.clone())
            .or_else(|| {
                manifest
                    .spec
                    .post_inference
                    .first()
                    .map(|candidate| candidate.id.clone())
            })
            .unwrap_or_default();

        let mut candidates = Vec::with_capacity(metrics.len());
        for (index, candidate) in manifest.spec.post_inference.iter().enumerate() {
            let comparison = if let Some(baseline) = &external_baseline {
                build_comparison(
                    manifest,
                    &baseline_id,
                    &baseline.metrics,
                    &metrics[index],
                    dataset_noise_margin_der,
                    &candidate.documented_der_outliers,
                )?
            } else if index == 0 {
                None
            } else {
                build_comparison(
                    manifest,
                    &baseline_id,
                    &metrics[0],
                    &metrics[index],
                    dataset_noise_margin_der,
                    &candidate.documented_der_outliers,
                )?
            };
            let candidate_metrics = &metrics[index];
            candidates.push(CandidateProjection {
                index,
                id: candidate.id.clone(),
                records: candidate_counts[index],
                successful_records: candidate_metrics.successful_records,
                failed_records: candidate_metrics.failed_records,
                duration_weighted_der: candidate_metrics.duration_weighted_der,
                missed_percent: candidate_metrics.missed_percent,
                false_alarm_percent: candidate_metrics.false_alarm_percent,
                confusion_percent: candidate_metrics.confusion_percent,
                usable_training_embeddings: candidate_metrics
                    .usable_training_embeddings
                    .as_ref()
                    .map(|range| CountRangeProjection {
                        minimum: range.minimum,
                        median: range.median,
                        maximum: range.maximum,
                    }),
                median_workload_seconds: candidate_metrics.median_workload_seconds,
                workload_mad_seconds: candidate_metrics.workload_mad_seconds,
                median_post_inference_seconds: candidate_metrics.median_post_inference_seconds,
                post_inference_mad_seconds: candidate_metrics.post_inference_mad_seconds,
                comparison,
            });
        }

        Ok(Self {
            schema_version: MANIFEST_SCHEMA_VERSION,
            experiment_id: manifest.spec.experiment_id.clone(),
            status,
            completed_records: records.len(),
            expected_records,
            repetitions: manifest.spec.performance.repetitions,
            dataset_noise_margin_der,
            files: manifest
                .files
                .iter()
                .enumerate()
                .map(|(index, file)| FileProjection {
                    index,
                    id: file.id.clone(),
                    records: file_counts[index],
                })
                .collect(),
            candidates,
        })
    }

    fn report(&self, manifest: &RunManifest) -> String {
        let mut report = String::new();
        report.push_str(&format!("# macOS experiment {}\n\n", self.experiment_id));
        report.push_str(&format!(
            "- Status: {}\n- Records: {}/{}\n- Repetitions: {}\n\n",
            self.status, self.completed_records, self.expected_records, self.repetitions
        ));
        report.push_str("## Files\n\n");
        report
            .push_str("| Index | File | Duration (s) | Records |\n| ---: | --- | ---: | ---: |\n");
        for (file, projection) in manifest.files.iter().zip(&self.files) {
            report.push_str(&format!(
                "| {} | {} | {:.3} | {} |\n",
                projection.index, file.id, file.duration_seconds, projection.records
            ));
        }
        report.push_str("\n## Candidates\n\n");
        report.push_str(
            "| Index | Candidate | Records | Success | DER | Training embeddings min/median/max | Median workload (s) | Median post (s) | Decision |\n\
             | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |\n",
        );
        for candidate in &self.candidates {
            let der = candidate
                .duration_weighted_der
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_owned());
            let wall = candidate
                .median_workload_seconds
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_owned());
            let post = candidate
                .median_post_inference_seconds
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "n/a".to_owned());
            let training_embeddings = candidate
                .usable_training_embeddings
                .as_ref()
                .map(|range| format!("{}/{:.1}/{}", range.minimum, range.median, range.maximum))
                .unwrap_or_else(|| "n/a".to_owned());
            let decision = candidate
                .comparison
                .as_ref()
                .map(|comparison| if comparison.accepted { "pass" } else { "stop" })
                .unwrap_or("baseline");
            report.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                candidate.index,
                candidate.id,
                candidate.records,
                candidate.successful_records,
                der,
                training_embeddings,
                wall,
                post,
                decision
            ));
        }
        for candidate in &self.candidates {
            let Some(comparison) = &candidate.comparison else {
                continue;
            };
            report.push_str(&format!("\n## Per-file DER changes: {}\n\n", candidate.id));
            report.push_str(
                "| File | Baseline DER | Candidate DER | Change | Reference speakers | Baseline predicted | Candidate predicted | Baseline count error | Candidate count error | Documented cause |\n\
                 | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n",
            );
            for change in &comparison.per_file_der_changes {
                report.push_str(&format!(
                    "| {} | {:.3} | {:.3} | {:+.3} | {} | {} | {} | {:+} | {:+} | {} |\n",
                    change.file_id,
                    change.baseline_der,
                    change.candidate_der,
                    change.change,
                    change.reference_speaker_count,
                    change.baseline_predicted_speaker_count,
                    change.candidate_predicted_speaker_count,
                    change.baseline_speaker_count_error,
                    change.candidate_speaker_count_error,
                    change.documented_cause.as_deref().unwrap_or("")
                ));
            }
            report.push_str(&format!(
                "\nPer-file 2-DER guard: {}\n",
                if comparison.per_file_der_guard_passed {
                    "pass"
                } else {
                    "fail"
                }
            ));
        }
        report
    }
}

fn comparison_noise_margin(external_baseline_noise_margin: Option<f64>) -> f64 {
    // Post-inference candidates share one set of inference artifacts, so inference noise does not
    // apply to comparisons within the run
    external_baseline_noise_margin.unwrap_or(0.0)
}

impl CandidateMetrics {
    fn from_records(
        manifest: &RunManifest,
        records: &[StoredRecord],
        candidate_index: usize,
        process_wall_seconds: &BTreeMap<u32, f64>,
    ) -> Result<Self> {
        let candidate_records: Vec<_> = records
            .iter()
            .filter(|record| record.candidate_index == candidate_index)
            .collect();
        let failed_records = candidate_records
            .iter()
            .filter(|record| record.value["status"] == "failed")
            .count();
        let mut complete = Vec::new();
        for record in &candidate_records {
            if record.value["status"] != "complete" {
                continue;
            }
            let measurement: CompleteRecordMeasurement = serde_json::from_value(
                record.value.clone(),
            )
            .wrap_err_with(|| {
                format!(
                    "invalid complete record for candidate index {candidate_index}, file index {}",
                    record.file_index
                )
            })?;
            complete.push((record.repetition, record.file_index, measurement));
        }

        let mut by_file: BTreeMap<usize, Vec<&CompleteRecordMeasurement>> = BTreeMap::new();
        let mut repetition_post_inference_seconds = BTreeMap::new();
        for (repetition, file_index, measurement) in &complete {
            by_file.entry(*file_index).or_default().push(measurement);
            *repetition_post_inference_seconds
                .entry(*repetition)
                .or_insert(0.0) += measurement.stage_timings.post_inference_seconds;
        }

        let mut observations = Vec::with_capacity(by_file.len());
        let mut total_reference = 0.0;
        let mut total_error = 0.0;
        let mut total_missed = 0.0;
        let mut total_false_alarm = 0.0;
        let mut total_confusion = 0.0;
        let mut speaker_counts = BTreeMap::new();
        let mut usable_training_embeddings = Vec::new();
        for (file_index, measurements) in by_file {
            let reference = measurements[0].der.reference_speaker_time;
            ensure!(
                reference.is_finite() && reference > 0.0,
                "candidate index {candidate_index}, file index {file_index} has invalid reference duration"
            );
            ensure!(
                measurements.iter().all(|measurement| {
                    (measurement.der.reference_speaker_time - reference).abs() <= 1e-9
                }),
                "candidate index {candidate_index}, file index {file_index} changed its reference duration"
            );
            let missed = median(
                &measurements
                    .iter()
                    .map(|measurement| measurement.der.missed)
                    .collect::<Vec<_>>(),
            )?;
            let false_alarm = median(
                &measurements
                    .iter()
                    .map(|measurement| measurement.der.false_alarm)
                    .collect::<Vec<_>>(),
            )?;
            let confusion = median(
                &measurements
                    .iter()
                    .map(|measurement| measurement.der.confusion)
                    .collect::<Vec<_>>(),
            )?;
            let error = missed + false_alarm + confusion;
            let reference_speaker_count = measurements[0].der.reference_speaker_count;
            let predicted_speaker_count = measurements[0].der.predicted_speaker_count;
            ensure!(
                measurements.iter().all(|measurement| {
                    measurement.der.reference_speaker_count == reference_speaker_count
                        && measurement.der.predicted_speaker_count == predicted_speaker_count
                }),
                "candidate index {candidate_index}, file index {file_index} changed its speaker counts"
            );
            observations.push(FileDerObservation::new(
                manifest.files[file_index].id.clone(),
                reference,
                error,
            )?);
            speaker_counts.insert(
                manifest.files[file_index].id.clone(),
                SpeakerCountMeasurement {
                    reference: reference_speaker_count,
                    predicted: predicted_speaker_count,
                },
            );
            let counts = measurements
                .iter()
                .filter_map(|measurement| {
                    measurement
                        .clustering
                        .as_ref()
                        .map(|clustering| clustering.usable_training_embeddings)
                })
                .collect::<Vec<_>>();
            ensure!(
                counts.is_empty() || counts.len() == measurements.len(),
                "candidate index {candidate_index}, file index {file_index} has incomplete training-embedding diagnostics"
            );
            if let Some(first) = counts.first() {
                ensure!(
                    counts.iter().all(|count| count == first),
                    "candidate index {candidate_index}, file index {file_index} changed its usable training-embedding count"
                );
                usable_training_embeddings.push(*first);
            }
            total_reference += reference;
            total_error += error;
            total_missed += missed;
            total_false_alarm += false_alarm;
            total_confusion += confusion;
        }

        let repetition_wall_seconds = process_wall_seconds.clone();
        let workload_values: Vec<_> = repetition_wall_seconds.values().copied().collect();
        let median_workload_seconds = (!workload_values.is_empty())
            .then(|| median(&workload_values))
            .transpose()?;
        let workload_mad_seconds = (!workload_values.is_empty())
            .then(|| median_absolute_deviation(&workload_values))
            .transpose()?;
        let post_inference_values: Vec<_> = repetition_post_inference_seconds
            .values()
            .copied()
            .collect();
        let median_post_inference_seconds = (!post_inference_values.is_empty())
            .then(|| median(&post_inference_values))
            .transpose()?;
        let post_inference_mad_seconds = (!post_inference_values.is_empty())
            .then(|| median_absolute_deviation(&post_inference_values))
            .transpose()?;
        let percentage =
            |numerator: f64| (total_reference > 0.0).then_some(numerator / total_reference * 100.0);
        let usable_training_embeddings = count_range(&usable_training_embeddings);

        Ok(Self {
            observations,
            successful_records: complete.len(),
            failed_records,
            duration_weighted_der: percentage(total_error),
            missed_percent: percentage(total_missed),
            false_alarm_percent: percentage(total_false_alarm),
            confusion_percent: percentage(total_confusion),
            usable_training_embeddings,
            repetition_wall_seconds,
            median_workload_seconds,
            workload_mad_seconds,
            median_post_inference_seconds,
            post_inference_mad_seconds,
            speaker_counts,
        })
    }
}

fn count_range(values: &[usize]) -> Option<CountRangeProjection> {
    if values.is_empty() {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let median = if sorted.len().is_multiple_of(2) {
        let upper = sorted.len() / 2;
        (sorted[upper - 1] as f64 + sorted[upper] as f64) / 2.0
    } else {
        sorted[sorted.len() / 2] as f64
    };

    Some(CountRangeProjection {
        minimum: sorted[0],
        median,
        maximum: sorted[sorted.len() - 1],
    })
}

struct ExternalBaseline {
    id: String,
    metrics: CandidateMetrics,
    noise_margin: f64,
}

fn load_external_baseline(candidate_store: &RunStore) -> Result<Option<ExternalBaseline>> {
    let manifest = &candidate_store.manifest;
    let Some(path) = &manifest.spec.baseline_run else {
        return Ok(None);
    };
    let path = if path.is_absolute() {
        path.clone()
    } else {
        project_root().join(path)
    };
    let (source_store, _source_experiment) = RunStore::open(&path)?;
    ensure!(
        source_store.manifest.spec.dataset.id == manifest.spec.dataset.id,
        "baseline dataset '{}' does not match candidate dataset '{}'",
        source_store.manifest.spec.dataset.id,
        manifest.spec.dataset.id
    );
    let comparison_dir = candidate_store.run_dir.join("comparison-baseline");
    let store = if comparison_dir.is_dir() {
        let nested_manifest: RunManifest = serde_json::from_reader(BufReader::new(File::open(
            comparison_dir.join("manifest.json"),
        )?))?;
        RunStore {
            run_dir: comparison_dir,
            manifest: nested_manifest,
        }
    } else {
        source_store
    };
    let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone())?;
    let records = store.read_records(&experiment)?;
    ensure!(
        !records.is_empty(),
        "baseline run contains no durable records: {}",
        path.display()
    );
    let candidate = store
        .manifest
        .spec
        .post_inference
        .first()
        .ok_or_else(|| eyre!("baseline run contains no candidate"))?;
    let process_wall_seconds = read_repetition_wall_seconds(&store.run_dir)?;
    Ok(Some(ExternalBaseline {
        id: format!("{}/{}", store.manifest.spec.experiment_id, candidate.id),
        metrics: CandidateMetrics::from_records(
            &store.manifest,
            &records,
            0,
            &process_wall_seconds,
        )?,
        noise_margin: baseline_noise_margin(&records)?,
    }))
}

fn build_comparison(
    manifest: &RunManifest,
    baseline_id: &str,
    baseline: &CandidateMetrics,
    candidate: &CandidateMetrics,
    noise_margin: f64,
    documented_outliers: &[super::domain::DocumentedDerOutlier],
) -> Result<Option<ComparisonProjection>> {
    if baseline.observations.is_empty()
        || baseline.observations.len() != candidate.observations.len()
    {
        return Ok(None);
    }
    let paired = pair_file_observations(&baseline.observations, &candidate.observations)?;
    let bootstrap = paired_file_bootstrap(&paired, manifest.spec.performance.seed)?;
    let improvement_interval = bootstrap.improvement_interval();
    let baseline_interval = bootstrap.baseline_interval();
    let candidate_interval = bootstrap.candidate_interval();
    let non_inferior = bootstrap.non_inferior(noise_margin)?;
    let clear_der_improvement = bootstrap.clear_improvement();
    for documented in documented_outliers {
        ensure!(
            paired
                .iter()
                .any(|observation| observation.file_id() == documented.file_id),
            "documented DER outlier '{}' is not in the run manifest",
            documented.file_id
        );
    }
    let mut per_file_der_changes = paired
        .iter()
        .map(|observation| -> Result<FileDerChangeProjection> {
            let baseline_der =
                observation.baseline_der_numerator() / observation.reference_duration() * 100.0;
            let candidate_der =
                observation.candidate_der_numerator() / observation.reference_duration() * 100.0;
            let baseline_counts = baseline
                .speaker_counts
                .get(observation.file_id())
                .ok_or_else(|| {
                    eyre!(
                        "baseline is missing speaker counts for '{}'",
                        observation.file_id()
                    )
                })?;
            let candidate_counts = candidate
                .speaker_counts
                .get(observation.file_id())
                .ok_or_else(|| {
                    eyre!(
                        "candidate is missing speaker counts for '{}'",
                        observation.file_id()
                    )
                })?;
            let documented_cause = documented_outliers
                .iter()
                .find(|outlier| outlier.file_id == observation.file_id())
                .map(|outlier| outlier.cause.clone());
            Ok(FileDerChangeProjection {
                file_id: observation.file_id().to_owned(),
                baseline_der,
                candidate_der,
                change: candidate_der - baseline_der,
                reference_speaker_count: baseline_counts.reference,
                baseline_predicted_speaker_count: baseline_counts.predicted,
                candidate_predicted_speaker_count: candidate_counts.predicted,
                baseline_speaker_count_error: baseline_counts.predicted as isize
                    - baseline_counts.reference as isize,
                candidate_speaker_count_error: candidate_counts.predicted as isize
                    - candidate_counts.reference as isize,
                documented_cause,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    per_file_der_changes.sort_by(|left, right| {
        right
            .change
            .abs()
            .total_cmp(&left.change.abs())
            .then_with(|| left.file_id.cmp(&right.file_id))
    });
    let per_file_der_guard_passed = per_file_der_guard_passes(&per_file_der_changes);

    let speed_improvements = baseline
        .repetition_wall_seconds
        .iter()
        .filter_map(|(repetition, baseline_seconds)| {
            candidate
                .repetition_wall_seconds
                .get(repetition)
                .map(|candidate_seconds| {
                    speed_improvement_percent(*baseline_seconds, *candidate_seconds)
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let speed_improvement = match (
        baseline.median_workload_seconds,
        candidate.median_workload_seconds,
    ) {
        (Some(baseline_seconds), Some(candidate_seconds)) => Some(speed_improvement_percent(
            baseline_seconds,
            candidate_seconds,
        )?),
        _ => None,
    };
    let speed_mad = (!speed_improvements.is_empty())
        .then(|| median_absolute_deviation(&speed_improvements))
        .transpose()?;
    let material_speed_improvement = match (speed_improvement, speed_mad) {
        (Some(improvement), Some(mad)) => material_speed_rule(improvement, mad)?,
        _ => false,
    };
    let accepted = acceptance_passes(
        manifest.spec.acceptance,
        clear_der_improvement,
        bootstrap.improvement_percent(),
        non_inferior,
        material_speed_improvement,
        per_file_der_guard_passed,
    );

    Ok(Some(ComparisonProjection {
        baseline_id: baseline_id.to_owned(),
        seed: bootstrap.seed(),
        bootstrap_samples: bootstrap.sample_count(),
        baseline_der: bootstrap.baseline_der_percent(),
        candidate_der: bootstrap.candidate_der_percent(),
        der_improvement: bootstrap.improvement_percent(),
        baseline_der_interval: IntervalProjection {
            lower: baseline_interval.lower(),
            upper: baseline_interval.upper(),
        },
        candidate_der_interval: IntervalProjection {
            lower: candidate_interval.lower(),
            upper: candidate_interval.upper(),
        },
        der_improvement_interval: IntervalProjection {
            lower: improvement_interval.lower(),
            upper: improvement_interval.upper(),
        },
        clear_der_improvement,
        non_inferior,
        speed_improvement_percent: speed_improvement,
        speed_improvement_mad_percent: speed_mad,
        material_speed_improvement,
        per_file_der_guard_passed,
        per_file_der_changes,
        accepted,
    }))
}

fn acceptance_passes(
    policy: super::domain::AcceptancePolicy,
    clear_der_improvement: bool,
    der_improvement: f64,
    non_inferior: bool,
    material_speed_improvement: bool,
    per_file_der_guard_passed: bool,
) -> bool {
    match policy {
        super::domain::AcceptancePolicy::StandardDer => {
            clear_der_improvement && der_improvement >= 0.2 && per_file_der_guard_passed
        }
        super::domain::AcceptancePolicy::StandardPerformance => {
            material_speed_improvement && non_inferior && per_file_der_guard_passed
        }
        super::domain::AcceptancePolicy::FastPareto => {
            non_inferior
                && (material_speed_improvement || clear_der_improvement)
                && per_file_der_guard_passed
        }
    }
}

fn per_file_der_guard_passes(changes: &[FileDerChangeProjection]) -> bool {
    changes
        .iter()
        .all(|change| change.change.abs() <= 2.0 || change.documented_cause.is_some())
}

fn baseline_noise_margin(records: &[StoredRecord]) -> Result<f64> {
    let mut by_repetition: BTreeMap<u32, (f64, f64)> = BTreeMap::new();
    for record in records.iter().filter(|record| record.candidate_index == 0) {
        if record.value["status"] != "complete" {
            continue;
        }
        let measurement: CompleteRecordMeasurement = serde_json::from_value(record.value.clone())?;
        let entry = by_repetition.entry(record.repetition).or_default();
        entry.0 += measurement.der.missed + measurement.der.false_alarm + measurement.der.confusion;
        entry.1 += measurement.der.reference_speaker_time;
    }
    let repeat_der: Vec<_> = by_repetition
        .values()
        .filter_map(|(numerator, denominator)| {
            (*denominator > 0.0).then_some(numerator / denominator * 100.0)
        })
        .collect();
    let mut differences = Vec::new();
    for left in 0..repeat_der.len() {
        for right in (left + 1)..repeat_der.len() {
            differences.push(repeat_der[left] - repeat_der[right]);
        }
    }
    dataset_noise_margin(&differences)
}

fn read_repetition_wall_seconds(run_dir: &Path) -> Result<BTreeMap<u32, f64>> {
    let directory = run_dir.join("repetitions");
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut values = BTreeMap::new();
    let mut entries = fs::read_dir(&directory)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        if !path.is_file() || path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let value: Value = serde_json::from_reader(BufReader::new(File::open(&path)?))?;
        if value["resumed"].as_bool().unwrap_or(false) {
            continue;
        }
        let repetition = value["repetition"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| eyre!("invalid repetition record {}", path.display()))?;
        let seconds = value["whole_process_seconds"]
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| eyre!("invalid whole-process time in {}", path.display()))?;
        ensure!(
            values.insert(repetition, seconds).is_none(),
            "duplicate repetition record {repetition}"
        );
    }
    Ok(values)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RecordPathKey {
    repetition: u32,
    file_component: String,
    candidate_id: String,
}

fn resolve_dataset_files(experiment: &ValidatedExperiment) -> Result<Vec<ManifestFile>> {
    let dataset = crate::datasets::find_dataset(&experiment.spec().dataset.id)
        .ok_or_else(|| eyre!("unknown dataset '{}'", experiment.spec().dataset.id))?;
    let dataset_dir = dataset.dataset_dir(experiment.datasets_dir());
    let requested = &experiment.spec().dataset.files;
    let pairs = if requested.is_empty() {
        discover_files(
            &dataset_dir,
            experiment.spec().dataset.max_files,
            experiment.spec().dataset.max_minutes as f64,
        )?
    } else {
        // Explicit file IDs pin order. Discovery still owns pairing, WAV validation, and local
        // file filtering; limits are checked after the requested order is applied.
        discover_files(&dataset_dir, u32::MAX, f64::MAX)?
    };
    ensure!(
        !pairs.is_empty(),
        "no paired wav+rttm files found in {}",
        dataset_dir.display()
    );

    let selected = if requested.is_empty() {
        pairs
    } else {
        let mut by_name = HashMap::new();
        for (wav, rttm) in pairs {
            let stem = file_stem_string(&wav)?;
            let name = wav
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or_else(|| eyre!("dataset WAV has no valid file name: {}", wav.display()))?;
            by_name.insert(stem, (wav.clone(), rttm.clone()));
            by_name.insert(name.to_owned(), (wav, rttm));
        }
        let mut selected = Vec::with_capacity(requested.len());
        let mut seen = HashSet::new();
        for requested_name in requested {
            let pair = by_name.get(requested_name).ok_or_else(|| {
                eyre!(
                    "dataset file '{}' was not found in {}",
                    requested_name,
                    dataset_dir.display()
                )
            })?;
            let id = file_stem_string(&pair.0)?;
            ensure!(
                seen.insert(id.clone()),
                "dataset files select duplicate '{id}'"
            );
            selected.push(pair.clone());
        }
        ensure!(
            selected.len() <= experiment.spec().dataset.max_files as usize,
            "dataset.files contains {} files but max_files is {}",
            selected.len(),
            experiment.spec().dataset.max_files
        );
        let total_minutes = selected
            .iter()
            .map(|(wav, _)| crate::cmd::wav_duration_seconds(wav).unwrap_or(0.0) / 60.0)
            .sum::<f64>();
        ensure!(
            total_minutes <= experiment.spec().dataset.max_minutes as f64,
            "dataset.files duration {total_minutes:.3} minutes exceeds max_minutes {}",
            experiment.spec().dataset.max_minutes
        );
        selected
    };

    let mut ids = BTreeSet::new();
    let mut components = BTreeSet::new();
    let mut files = Vec::with_capacity(selected.len());
    for (wav, rttm) in selected {
        let id = file_stem_string(&wav)?;
        ensure!(
            ids.insert(id.clone()),
            "dataset discovery returned duplicate file '{id}'"
        );
        let component = path_component(&id);
        ensure!(
            components.insert(component),
            "dataset file IDs collide after record path encoding: '{id}'"
        );
        let duration_seconds = crate::cmd::wav_duration_seconds(&wav)
            .wrap_err_with(|| format!("failed to read WAV duration {}", wav.display()))?;
        files.push(ManifestFile {
            id,
            wav,
            rttm,
            duration_seconds,
        });
    }
    ensure!(!files.is_empty(), "dataset selection is empty");
    Ok(files)
}

fn model_digest(experiment: &ValidatedExperiment, root: &Path) -> Result<String> {
    digest_paths_cached(
        root,
        &[experiment.models_dir().to_path_buf()],
        &root.join("_benchmarks/macos/digest-cache"),
    )
}

fn dataset_digest(files: &[ManifestFile], root: &Path) -> Result<String> {
    let dataset_paths = files
        .iter()
        .flat_map(|file| [file.wav.clone(), file.rttm.clone()])
        .collect::<Vec<_>>();
    digest_paths(root, &dataset_paths)
}

fn build_identity(
    experiment: &ValidatedExperiment,
    files: &[ManifestFile],
    worker: &Path,
    root: &Path,
) -> Result<ManifestIdentity> {
    let model_sha256 = model_digest(experiment, root)?;
    let worker_sha256 = digest_paths(root, &[worker.to_path_buf()])?;
    let dataset_sha256 = dataset_digest(files, root)?;
    let host = collect_host_identity(root)?;
    Ok(ManifestIdentity {
        host,
        model_sha256,
        worker_sha256,
        dataset_sha256,
        created_at: chrono::Utc::now().to_rfc3339(),
        seed: experiment.performance().seed,
        cache_state: "operating-system file cache unchanged; no broad cache clear".to_owned(),
        run_order: RunOrder {
            repetitions: (0..experiment.performance().repetitions()).collect(),
            files: files.iter().map(|file| file.id.clone()).collect(),
            candidates: experiment
                .post_inference()
                .iter()
                .map(|candidate| candidate.id.clone())
                .collect(),
            comparison_processes: comparison_process_order(experiment),
        },
        fallback_events: Vec::new(),
    })
}

fn comparison_process_order(experiment: &ValidatedExperiment) -> Vec<String> {
    if experiment.spec().baseline_run.is_none() {
        return (0..experiment.performance().repetitions())
            .map(|repetition| format!("candidate:r{repetition:03}"))
            .collect();
    }
    let mut order = Vec::new();
    for first in (0..experiment.performance().repetitions()).step_by(2) {
        let second = first + 1;
        order.extend([
            format!("baseline:r{first:03}"),
            format!("candidate:r{first:03}"),
            format!("candidate:r{second:03}"),
            format!("baseline:r{second:03}"),
        ]);
    }
    order
}

fn collect_host_identity(root: &Path) -> Result<Option<HostIdentity>> {
    #[cfg(target_os = "macos")]
    {
        HostIdentity::collect(root).map(Some)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = root;
        Ok(None)
    }
}

fn create_run_dir(run_dir: &Path) -> Result<()> {
    match fs::create_dir(run_dir) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            bail!("experiment run already exists: {}", run_dir.display())
        }
        Err(error) => Err(error)
            .wrap_err_with(|| format!("failed to create experiment run {}", run_dir.display())),
    }
}

fn write_manifest(run_dir: &Path, manifest: &RunManifest) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(manifest)?;
    bytes.push(b'\n');
    atomic_write_new(&run_dir.join("manifest.json"), &bytes)
}

fn validate_manifest_files(manifest: &RunManifest, experiment: &ValidatedExperiment) -> Result<()> {
    ensure!(
        !manifest.files.is_empty(),
        "run manifest contains no dataset files"
    );
    ensure!(
        manifest.files.len() <= experiment.spec().dataset.max_files as usize,
        "run manifest contains {} files but max_files is {}",
        manifest.files.len(),
        experiment.spec().dataset.max_files
    );
    let mut ids = BTreeSet::new();
    let mut components = BTreeSet::new();
    let mut total_minutes = 0.0;
    for file in &manifest.files {
        ensure!(
            !file.id.is_empty(),
            "run manifest contains an empty file id"
        );
        ensure!(
            ids.insert(file.id.clone()),
            "run manifest contains duplicate file '{}'",
            file.id
        );
        ensure!(
            components.insert(file.record_component()),
            "run manifest file IDs collide after record path encoding"
        );
        ensure!(
            file.wav.is_file(),
            "manifest WAV file does not exist: {}",
            file.wav.display()
        );
        ensure!(
            file.rttm.is_file(),
            "manifest RTTM file does not exist: {}",
            file.rttm.display()
        );
        ensure!(
            file_stem_string(&file.wav)? == file.id,
            "manifest file id '{}' does not match WAV {}",
            file.id,
            file.wav.display()
        );
        ensure!(
            file.duration_seconds.is_finite() && file.duration_seconds > 0.0,
            "manifest file '{}' has invalid duration {}",
            file.id,
            file.duration_seconds
        );
        total_minutes += file.duration_seconds / 60.0;
    }
    ensure!(
        total_minutes <= experiment.spec().dataset.max_minutes as f64,
        "run manifest duration {total_minutes:.3} minutes exceeds max_minutes {}",
        experiment.spec().dataset.max_minutes
    );
    Ok(())
}

fn validate_payload_ids<T>(
    missing: &[String],
    payloads: &[(String, T)],
    repetition: u32,
    file_index: usize,
) -> Result<()> {
    let expected: BTreeSet<_> = missing.iter().map(String::as_str).collect();
    let actual: BTreeSet<_> = payloads.iter().map(|(id, _)| id.as_str()).collect();
    ensure!(
        payloads.len() == actual.len(),
        "executor returned duplicate candidate IDs for repetition {repetition}, file {file_index}"
    );
    ensure!(
        expected == actual,
        "executor candidate IDs for repetition {repetition}, file {file_index} do not match missing targets"
    );
    Ok(())
}

fn collect_json_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries = fs::read_dir(dir)
        .wrap_err_with(|| format!("failed to read records directory {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_json_paths(&path, paths)?;
        } else if file_type.is_file() && path.extension().is_some_and(|ext| ext == "json") {
            paths.push(path);
        }
    }
    Ok(())
}

fn parse_record_path(path: &Path, run_dir: &Path) -> Result<RecordPathKey> {
    let relative = path
        .strip_prefix(run_dir)
        .map_err(|_| eyre!("record path is outside run directory: {}", path.display()))?;
    let mut components = relative.components();
    ensure!(
        components.next() == Some(Component::Normal(OsStr::new(RECORDS_DIR))),
        "record {} is not under records/",
        path.display()
    );
    let repetition_component = components.next().ok_or_else(|| {
        eyre!(
            "record path has no repetition component: {}",
            path.display()
        )
    })?;
    let file_component = components
        .next()
        .ok_or_else(|| eyre!("record path has no file component: {}", path.display()))?;
    let candidate_component = components
        .next()
        .ok_or_else(|| eyre!("record path has no candidate component: {}", path.display()))?;
    ensure!(
        components.next().is_none(),
        "record path has too many components: {}",
        path.display()
    );

    let repetition_name = repetition_component.as_os_str().to_str().ok_or_else(|| {
        eyre!(
            "record repetition component is not UTF-8: {}",
            path.display()
        )
    })?;
    let repetition = repetition_name
        .strip_prefix('r')
        .ok_or_else(|| eyre!("record repetition component is invalid: {}", path.display()))?
        .parse::<u32>()?;
    let candidate_name = candidate_component.as_os_str().to_str().ok_or_else(|| {
        eyre!(
            "record candidate component is not UTF-8: {}",
            path.display()
        )
    })?;
    let candidate_id = candidate_name
        .strip_suffix(".json")
        .ok_or_else(|| eyre!("record candidate component is invalid: {}", path.display()))?;
    Ok(RecordPathKey {
        repetition,
        file_component: file_component.as_os_str().to_string_lossy().into_owned(),
        candidate_id: candidate_id.to_owned(),
    })
}

fn json_u32(object: &Map<String, Value>, key: &str, path: &Path) -> Result<u32> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| eyre!("record {} field '{key}' must be a uint32", path.display()))
}

fn json_usize(object: &Map<String, Value>, key: &str, path: &Path) -> Result<usize> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| eyre!("record {} field '{key}' must be a usize", path.display()))
}

fn json_string(object: &Map<String, Value>, key: &str, path: &Path) -> Result<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| eyre!("record {} field '{key}' must be a string", path.display()))
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {}", parent.display()))?;
    }
    let temp_path = temporary_path(path);
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&temp_path)
        .wrap_err_with(|| format!("failed to open temporary file {}", temp_path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp_path, path).wrap_err_with(|| {
        format!(
            "failed to atomically replace {} with {}",
            path.display(),
            temp_path.display()
        )
    })?;
    sync_parent(path)
}

fn atomic_write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        !path.exists(),
        "refusing to overwrite immutable file {}",
        path.display()
    );
    atomic_write(path, bytes)
}

fn temporary_path(path: &Path) -> PathBuf {
    let file_name = path.file_name().unwrap_or_else(|| OsStr::new("output"));
    path.with_file_name(format!("{}.tmp", file_name.to_string_lossy()))
}

fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            File::open(parent)?.sync_all()?;
        }
    }
    Ok(())
}

fn path_component(value: &str) -> String {
    if is_safe_component(value) {
        return value.to_owned();
    }
    let mut encoded = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    if encoded.is_empty() {
        "%00".to_owned()
    } else {
        encoded
    }
}

fn is_safe_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

pub(super) fn profile(experiment: ValidatedExperiment, worker: &Path) -> Result<()> {
    ensure!(
        cfg!(target_os = "macos"),
        "macOS profiling requires a macOS host"
    );
    let root = project_root();
    let run_dir = root.join("_benchmarks").join("macos").join(experiment.id());
    let store = if run_dir.is_dir() {
        let (store, stored_experiment) = RunStore::open(&run_dir)?;
        store.validate_experiment(&experiment)?;
        ensure!(
            stored_experiment.id() == experiment.id(),
            "profile experiment does not match the existing run"
        );
        store
    } else {
        RunStore::create(&experiment, worker)?
    };
    super::execute::run_managed_with_worker(worker, &store, &experiment)?;

    let profile = experiment.profile();
    ensure!(
        profile.file_index < store.manifest.files.len(),
        "profile file index {} is outside the {}-file manifest",
        profile.file_index,
        store.manifest.files.len()
    );
    store.assert_worker(worker)?;
    let traces_dir = store.run_dir.join("traces");
    fs::create_dir_all(&traces_dir)?;
    let stamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let artifact_stem = format!("{stamp}-{}", profile.template.file_stem());
    let trace_path = traces_dir.join(format!("{artifact_stem}.trace"));
    let stage_log_path = traces_dir.join(format!("{artifact_stem}.log"));
    let stage_output_path = traces_dir.join(format!("{artifact_stem}-stages.json"));
    let output = Command::new("xcrun")
        .args([
            "xctrace",
            "record",
            "--template",
            profile.template.xctrace_name(),
            "--output",
        ])
        .arg(&trace_path)
        .arg("--time-limit")
        .arg(format!("{}s", profile.time_limit_seconds))
        .args(["--no-prompt", "--launch", "--"])
        .arg(worker)
        .arg("mac-experiment")
        .arg("profile-worker")
        .arg(&store.run_dir)
        .arg("--file-index")
        .arg(profile.file_index.to_string())
        .arg("--stage-output")
        .arg(&stage_output_path)
        .current_dir(&root)
        .output()
        .wrap_err("failed to launch xctrace")?;
    let mut diagnostics = output.stdout;
    diagnostics.extend_from_slice(&output.stderr);
    atomic_write(&stage_log_path, &diagnostics)?;
    let mut stderr_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(store.run_dir.join("stderr.log"))?;
    stderr_log.write_all(&diagnostics)?;
    stderr_log.sync_all()?;
    ensure!(
        output.status.success(),
        "xctrace failed with {}; see {}",
        output.status,
        stage_log_path.display()
    );
    ensure!(
        trace_path.exists(),
        "xctrace did not create {}",
        trace_path.display()
    );
    ensure!(
        stage_output_path.is_file(),
        "profile worker did not create {}",
        stage_output_path.display()
    );
    let mut exported_tables = Vec::new();
    let mut metal_tables: Option<(Option<PathBuf>, Option<PathBuf>)> = None;
    let toc_path = traces_dir.join(format!("{artifact_stem}-toc.xml"));
    export_trace(&trace_path, "--toc", None, &toc_path)?;
    exported_tables.push(toc_path);
    for schema in profile.template.export_schemas() {
        let output_path = traces_dir.join(format!("{artifact_stem}-{schema}.xml"));
        let xpath = format!("/trace-toc/run[@number=\"1\"]/data/table[@schema=\"{schema}\"]");
        export_trace(&trace_path, "--xpath", Some(&xpath), &output_path)?;
        if *schema == "metal-gpu-intervals" {
            metal_tables.get_or_insert_with(Default::default).0 = Some(output_path.clone());
        } else if *schema == "metal-application-command-buffer-submissions" {
            metal_tables.get_or_insert_with(Default::default).1 = Some(output_path.clone());
        }
        exported_tables.push(output_path);
    }
    let metal_summary = if let Some((Some(gpu_intervals), Some(command_submissions))) = metal_tables
    {
        let path = traces_dir.join(format!("{artifact_stem}-metal-summary.json"));
        super::trace_analysis::write_metal_summary(&gpu_intervals, &command_submissions, &path)?;
        Some(path)
    } else {
        None
    };

    #[cfg(target_os = "macos")]
    let compute_plan = {
        let path = traces_dir.join(format!("{artifact_stem}-compute-plan.json"));
        super::compute_plan::write_report(&experiment, &path)?;
        Some(path)
    };
    #[cfg(not(target_os = "macos"))]
    let compute_plan = None;

    let metadata = TraceMetadata {
        schema_version: 1,
        experiment_id: experiment.id().to_owned(),
        template: profile.template.xctrace_name().to_owned(),
        file_index: profile.file_index,
        file_id: store.manifest.files[profile.file_index].id.clone(),
        trace: trace_path.clone(),
        stage_log: stage_log_path.clone(),
        stage_output: stage_output_path,
        manifest: store.run_dir.join("manifest.json"),
        uninstrumented_summary: store.run_dir.join("summary.json"),
        compute_plan,
        metal_summary,
        exported_tables,
        instrumented_time_is_acceptance_evidence: false,
    };
    let mut metadata_bytes = serde_json::to_vec_pretty(&metadata)?;
    metadata_bytes.push(b'\n');
    atomic_write(
        &traces_dir.join(format!("{artifact_stem}.json")),
        &metadata_bytes,
    )?;
    println!("profile trace: {}", trace_path.display());
    Ok(())
}

fn export_trace(
    trace_path: &Path,
    mode: &str,
    query: Option<&str>,
    output_path: &Path,
) -> Result<()> {
    let mut command = Command::new("xcrun");
    command
        .args(["xctrace", "export", "--input"])
        .arg(trace_path)
        .arg(mode);
    if let Some(query) = query {
        command.arg(query);
    }
    let output = command.output()?;
    ensure!(
        output.status.success(),
        "xctrace export failed for {} with {}: {}",
        trace_path.display(),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    atomic_write(output_path, &output.stdout)
}

#[derive(Serialize)]
struct TraceMetadata {
    schema_version: u32,
    experiment_id: String,
    template: String,
    file_index: usize,
    file_id: String,
    trace: PathBuf,
    stage_log: PathBuf,
    stage_output: PathBuf,
    manifest: PathBuf,
    uninstrumented_summary: PathBuf,
    compute_plan: Option<PathBuf>,
    metal_summary: Option<PathBuf>,
    exported_tables: Vec<PathBuf>,
    instrumented_time_is_acceptance_evidence: bool,
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::commands::mac_experiment::domain::{
        AcceptancePolicy, CoreMlMode, DatasetSlice, EmbeddingComputeUnits, FbankNormalizationScope,
        FbankPreparationWorkers, InferenceLayout, InferenceVariant, MacExperimentSpec,
        PerformanceProtocol, PostInferenceVariant, SegmentationWorkers, ShapeLadder,
    };

    fn valid_spec() -> MacExperimentSpec {
        MacExperimentSpec {
            schema_version: 1,
            experiment_id: "mac-smoke".to_owned(),
            dataset: DatasetSlice {
                id: "ami-ihm".to_owned(),
                max_files: 2,
                max_minutes: 30,
                files: Vec::new(),
            },
            datasets_dir: None,
            models_dir: None,
            inference: InferenceVariant {
                mode: CoreMlMode::CoreMl,
                layout: InferenceLayout::OneSecondPhased,
                shape_ladder: ShapeLadder::Full,
                segmentation_workers: SegmentationWorkers::Automatic,
                filterbank_preparation_workers: FbankPreparationWorkers::Two,
                filterbank_normalization_scope: FbankNormalizationScope::Chunk,
                embedding_compute_units: EmbeddingComputeUnits::All,
            },
            post_inference: vec![PostInferenceVariant {
                id: "default".to_owned(),
                vbx_max_iters: None,
                vbx_fb: None,
                clean_frame_seconds: None,
                ahc_stopping: None,
                clustering_backend: None,
                documented_der_outliers: Vec::new(),
            }],
            performance: PerformanceProtocol {
                warmups: 0,
                repetitions: 1,
                sleep_seconds: 0,
                seed: 42,
            },
            acceptance: AcceptancePolicy::StandardPerformance,
            profile: None,
            baseline_run: None,
        }
    }

    fn test_manifest() -> RunManifest {
        RunManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            spec: valid_spec(),
            files: vec![
                ManifestFile {
                    id: "second".to_owned(),
                    wav: PathBuf::from("/tmp/second.wav"),
                    rttm: PathBuf::from("/tmp/second.rttm"),
                    duration_seconds: 2.0,
                },
                ManifestFile {
                    id: "first".to_owned(),
                    wav: PathBuf::from("/tmp/first.wav"),
                    rttm: PathBuf::from("/tmp/first.rttm"),
                    duration_seconds: 1.0,
                },
            ],
            identity: None,
        }
    }

    fn test_store(temp: &Path) -> RunStore {
        RunStore {
            run_dir: temp.to_owned(),
            manifest: test_manifest(),
        }
    }

    fn test_identity(seed: u64) -> ManifestIdentity {
        ManifestIdentity {
            host: None,
            model_sha256: "model".to_owned(),
            worker_sha256: "worker".to_owned(),
            dataset_sha256: "dataset".to_owned(),
            created_at: "2026-01-01T00:00:00Z".to_owned(),
            seed,
            cache_state: "test".to_owned(),
            run_order: RunOrder {
                repetitions: vec![0],
                files: vec!["second".to_owned(), "first".to_owned()],
                candidates: vec!["default".to_owned()],
                comparison_processes: vec!["candidate:r000".to_owned()],
            },
            fallback_events: Vec::new(),
        }
    }

    #[test]
    fn record_path_contains_repetition_file_and_candidate() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        assert_eq!(
            store.record_path(2, 1, "default"),
            temp.path().join("records/r002/first/default.json")
        );
    }

    #[test]
    fn temporary_records_are_ignored() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        let path = store.record_path(0, 0, "default");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path.with_file_name("default.json.tmp"), b"not complete").unwrap();
        let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone()).unwrap();
        let records = store.read_records(&experiment).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn projections_are_stable_in_manifest_order() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone()).unwrap();
        store
            .write_record(0, 1, "default", &json!({"value": "first"}))
            .unwrap();
        store
            .write_record(0, 0, "default", &json!({"value": "second"}))
            .unwrap();
        store.rebuild_projections(&experiment).unwrap();
        let first = fs::read(temp.path().join("files.jsonl")).unwrap();
        let first_summary = fs::read(temp.path().join("summary.json")).unwrap();
        store.rebuild_projections(&experiment).unwrap();
        let second = fs::read(temp.path().join("files.jsonl")).unwrap();
        let second_summary = fs::read(temp.path().join("summary.json")).unwrap();
        assert_eq!(first, second);
        assert_eq!(first_summary, second_summary);
        let lines: Vec<_> = String::from_utf8(first)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect();
        assert_eq!(lines[0]["file_id"], "second");
        assert_eq!(lines[1]["file_id"], "first");
    }

    #[test]
    fn executor_receives_only_missing_candidates() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone()).unwrap();
        store
            .write_record(0, 0, "default", &json!({"value": "existing"}))
            .unwrap();
        let mut calls = 0;
        store
            .execute(experiment, |_repetition, file, candidates| {
                calls += 1;
                assert_eq!(file.id, "first");
                assert_eq!(candidates, &["default".to_owned()]);
                Ok(vec![("default".to_owned(), json!({"value": "new"}))])
            })
            .unwrap();
        assert_eq!(calls, 1);
        assert!(store.record_exists(0, 1, "default"));
    }

    #[test]
    fn repetition_started_distinguishes_partial_runs() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone()).unwrap();

        assert!(!store.repetition_started(0, &experiment).unwrap());
        store
            .write_record(0, 0, "default", &json!({"value": "existing"}))
            .unwrap();

        assert!(store.repetition_started(0, &experiment).unwrap());
        assert!(!store.repetition_complete(0, &experiment).unwrap());
    }

    #[test]
    fn resumed_repetition_time_is_not_performance_evidence() {
        let temp = tempdir().unwrap();
        let store = test_store(temp.path());
        store
            .write_repetition_record(
                0,
                &json!({
                    "status": "complete",
                    "resumed": true,
                    "whole_process_seconds": 3.0,
                    "peak_rss_bytes": null,
                    "completed_at": "2026-09-02T00:00:00Z"
                }),
            )
            .unwrap();

        assert!(
            read_repetition_wall_seconds(temp.path())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn per_file_der_guard_checks_improvements_and_regressions() {
        let change = |change| FileDerChangeProjection {
            file_id: "file".to_owned(),
            baseline_der: 10.0,
            candidate_der: 10.0 + change,
            change,
            reference_speaker_count: 2,
            baseline_predicted_speaker_count: 2,
            candidate_predicted_speaker_count: 2,
            baseline_speaker_count_error: 0,
            candidate_speaker_count_error: 0,
            documented_cause: None,
        };

        assert!(per_file_der_guard_passes(&[change(-2.0), change(2.0)]));
        assert!(!per_file_der_guard_passes(&[change(-2.1)]));
        assert!(!per_file_der_guard_passes(&[change(2.1)]));
        let mut documented = change(-2.1);
        documented.documented_cause = Some("speaker count matches reference".to_owned());
        assert!(per_file_der_guard_passes(&[documented]));
    }

    #[test]
    fn fast_acceptance_requires_the_per_file_guard() {
        assert!(!acceptance_passes(
            AcceptancePolicy::FastPareto,
            true,
            1.0,
            true,
            true,
            false,
        ));
    }

    #[test]
    fn missing_process_record_keeps_speed_unavailable() {
        let manifest = test_manifest();
        let complete = |file_index| StoredRecord {
            value: json!({
                "status": "complete",
                "wall_seconds": 1.0,
                "stage_timings": { "post_inference_seconds": 0.25 },
                "der": {
                    "reference_speaker_time": 10.0,
                    "missed": 1.0,
                    "false_alarm": 0.0,
                    "confusion": 0.0,
                    "reference_speaker_count": 2,
                    "predicted_speaker_count": 2
                }
            }),
            repetition: 0,
            file_index,
            candidate_index: 0,
        };
        let metrics = CandidateMetrics::from_records(
            &manifest,
            &[complete(0), complete(1)],
            0,
            &BTreeMap::new(),
        )
        .unwrap();

        assert!(metrics.repetition_wall_seconds.is_empty());
        assert_eq!(metrics.median_workload_seconds, None);
    }

    #[test]
    fn count_range_reports_even_median() {
        let range = count_range(&[47, 138, 206, 136]).unwrap();

        assert_eq!(range.minimum, 47);
        assert_eq!(range.median, 137.0);
        assert_eq!(range.maximum, 206);
    }

    #[test]
    fn one_file_execution_receives_all_missing_post_candidates_once() {
        let temp = tempdir().unwrap();
        let mut store = test_store(temp.path());
        store
            .manifest
            .spec
            .post_inference
            .push(PostInferenceVariant {
                id: "short-vbx".to_owned(),
                vbx_max_iters: Some(5),
                vbx_fb: None,
                clean_frame_seconds: None,
                ahc_stopping: None,
                clustering_backend: None,
                documented_der_outliers: Vec::new(),
            });
        store.manifest.files.truncate(1);
        let experiment = ValidatedExperiment::from_spec(store.manifest.spec.clone()).unwrap();
        let mut calls = 0;
        store
            .execute(experiment, |_repetition, _file, candidates| {
                calls += 1;
                assert_eq!(candidates, &["default".to_owned(), "short-vbx".to_owned()]);
                Ok(candidates
                    .iter()
                    .map(|candidate| (candidate.clone(), json!({ "value": candidate })))
                    .collect())
            })
            .unwrap();
        assert_eq!(calls, 1);
    }

    #[test]
    fn comparison_baseline_uses_candidate_file_slice() {
        let temp = tempdir().unwrap();
        let source = test_store(&temp.path().join("source"));
        let mut candidate = test_store(&temp.path().join("candidate"));
        candidate.manifest.files = vec![source.manifest.files[1].clone()];
        candidate.manifest.spec.dataset.max_files = 1;

        let (comparison, _experiment) = candidate.comparison_baseline_store(&source).unwrap();

        assert_eq!(comparison.manifest.files.len(), 1);
        assert_eq!(comparison.manifest.files[0].id, "first");
        assert_eq!(comparison.manifest.spec.dataset.files, ["first"]);
        assert_eq!(comparison.manifest.spec.dataset.max_files, 1);
    }

    #[test]
    fn comparison_baseline_preserves_source_asset_digests() {
        let temp = tempdir().unwrap();
        let mut source = test_store(&temp.path().join("source"));
        source.manifest.identity = Some(test_identity(7));
        let mut candidate = test_store(&temp.path().join("candidate"));
        candidate.manifest.files = vec![source.manifest.files[1].clone()];
        candidate.manifest.spec.dataset.max_files = 1;
        candidate.manifest.spec.performance.repetitions = 4;
        candidate.manifest.spec.performance.seed = 99;
        candidate.manifest.identity = Some(test_identity(99));

        let (comparison, _experiment) = candidate.comparison_baseline_store(&source).unwrap();
        let identity = comparison.manifest.identity.as_ref().unwrap();

        assert_eq!(identity.model_sha256, "model");
        assert_eq!(identity.dataset_sha256, "dataset");
        assert_eq!(identity.worker_sha256, "worker");
        assert_eq!(identity.seed, 99);
        assert_eq!(identity.run_order.repetitions, [0, 1, 2, 3]);
        assert_eq!(identity.run_order.files, ["first"]);
        assert_eq!(identity.run_order.candidates, ["default"]);
        assert!(identity.run_order.comparison_processes.is_empty());
        assert_ne!(identity.created_at, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn open_rejects_changed_model_and_dataset_digests() {
        let temp = tempdir().unwrap();
        let models = temp.path().join("models");
        fs::create_dir(&models).unwrap();
        fs::write(models.join("weights.bin"), b"model-v1").unwrap();
        let wav = temp.path().join("clip.wav");
        let rttm = temp.path().join("clip.rttm");
        fs::write(&wav, b"wav-v1").unwrap();
        fs::write(&rttm, b"rttm-v1").unwrap();

        let mut spec = valid_spec();
        spec.models_dir = Some(models.clone());
        let experiment = ValidatedExperiment::from_spec(spec.clone()).unwrap();
        let files = vec![ManifestFile {
            id: "clip".to_owned(),
            wav: wav.clone(),
            rttm: rttm.clone(),
            duration_seconds: 1.0,
        }];
        let root = project_root();
        let run_dir = temp.path().join(experiment.id());
        fs::create_dir(&run_dir).unwrap();
        write_manifest(
            &run_dir,
            &RunManifest {
                schema_version: MANIFEST_SCHEMA_VERSION,
                spec,
                files: files.clone(),
                identity: Some(ManifestIdentity {
                    host: None,
                    model_sha256: model_digest(&experiment, &root).unwrap(),
                    worker_sha256: "worker".to_owned(),
                    dataset_sha256: dataset_digest(&files, &root).unwrap(),
                    created_at: "2026-01-01T00:00:00Z".to_owned(),
                    seed: 42,
                    cache_state: "test".to_owned(),
                    run_order: RunOrder::default(),
                    fallback_events: Vec::new(),
                }),
            },
        )
        .unwrap();

        RunStore::open(&run_dir).map(|_| ()).unwrap();

        fs::write(models.join("weights.bin"), b"model-v2").unwrap();
        let error = RunStore::open(&run_dir).map(|_| ()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("experiment models changed after manifest creation")
        );

        fs::write(models.join("weights.bin"), b"model-v1").unwrap();
        fs::write(&wav, b"wav-v2").unwrap();
        let error = RunStore::open(&run_dir).map(|_| ()).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("experiment dataset changed after manifest creation")
        );
    }

    #[test]
    fn post_inference_comparison_has_no_inference_noise_margin() {
        assert_eq!(comparison_noise_margin(None), 0.0);
        assert_eq!(comparison_noise_margin(Some(0.23)), 0.23);
    }
}
