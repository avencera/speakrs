use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Local};
use color_eyre::eyre::{Result, bail, ensure, eyre};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::catalog::{ImplementationCatalog, ImplementationId};
use crate::datasets::{DatasetCatalog, DatasetId};

/// Current typed benchmark record schema
pub const SCHEMA_VERSION: u32 = 3;
const LEGACY_SCHEMA_VERSION: u32 = 2;
pub const RUN_MANIFEST_FILE: &str = "run.json";
pub const SCORE_FILE: &str = "score.json";

/// Stable identity allocated before a benchmark directory is created
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        ensure!(!value.is_empty(), "run ID cannot be empty");
        ensure!(
            value.trim() == value,
            "run ID cannot contain surrounding whitespace"
        );
        ensure!(
            !value.contains('/') && !value.contains('\\'),
            "run ID cannot contain a path separator: {value}"
        );
        ensure!(
            value != "." && value != "..",
            "run ID cannot be a path traversal component"
        );
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Dataset identity recorded in a benchmark run
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatasetIdentity {
    Catalog { id: DatasetId },
    SingleFile { file_id: String },
}

impl DatasetIdentity {
    pub fn catalog(id: DatasetId) -> Self {
        Self::Catalog { id }
    }

    pub fn single_file(file_id: impl Into<String>) -> Result<Self> {
        let file_id = file_id.into();
        ensure!(
            !file_id.is_empty(),
            "single-file dataset ID cannot be empty"
        );
        ensure!(
            file_id.trim() == file_id
                && !file_id.contains('/')
                && !file_id.contains('\\')
                && file_id != "."
                && file_id != "..",
            "invalid single-file dataset ID {file_id}"
        );
        Ok(Self::SingleFile { file_id })
    }

    pub fn id_string(&self) -> String {
        match self {
            Self::Catalog { id } => id.as_str().to_owned(),
            Self::SingleFile { file_id } => file_id.clone(),
        }
    }

    pub fn display_name(&self) -> String {
        match self {
            Self::Catalog { id } => DatasetCatalog::all()
                .iter()
                .find(|spec| spec.id == *id)
                .map(|spec| spec.display_name.to_owned())
                .unwrap_or_else(|| id.as_str().to_owned()),
            Self::SingleFile { file_id } => file_id.clone(),
        }
    }
}

/// Identity shared by every dataset record in a benchmark suite
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub run_id: RunId,
    pub started_at: String,
    pub host: String,
    pub description: Option<String>,
    pub implementations: Vec<ImplementationId>,
    pub datasets: Vec<DatasetIdentity>,
}

impl RunIdentity {
    pub fn validate(&self) -> Result<()> {
        RunId::new(self.run_id.as_str().to_owned())?;
        ensure!(
            !self.implementations.is_empty(),
            "benchmark implementation manifest is empty"
        );
        let mut implementations = BTreeSet::new();
        for implementation in &self.implementations {
            ensure!(
                implementations.insert(*implementation),
                "benchmark run contains duplicate implementation {}",
                implementation.as_str()
            );
        }
        ensure!(
            !self.datasets.is_empty(),
            "benchmark dataset manifest is empty"
        );
        let mut datasets = BTreeSet::new();
        for dataset in &self.datasets {
            dataset.validate()?;
            ensure!(
                datasets.insert(dataset.identity_key()),
                "benchmark run contains duplicate dataset {}",
                dataset.id_string()
            );
        }
        Ok(())
    }
}

/// One collision-safe benchmark suite created before dataset work
#[derive(Clone, Debug)]
pub struct BenchmarkRun {
    pub identity: RunIdentity,
    pub root: PathBuf,
}

impl BenchmarkRun {
    pub fn create(
        benchmarks_root: &Path,
        implementations: Vec<ImplementationId>,
        datasets: Vec<DatasetId>,
        description: Option<String>,
        now: DateTime<Local>,
        host: String,
    ) -> Result<Self> {
        let dataset_identities = datasets
            .iter()
            .copied()
            .map(DatasetIdentity::catalog)
            .collect();
        Self::create_with_dataset_identities(
            benchmarks_root,
            implementations,
            dataset_identities,
            description,
            now,
            host,
        )
    }

    pub fn create_with_dataset_identities(
        benchmarks_root: &Path,
        implementations: Vec<ImplementationId>,
        datasets: Vec<DatasetIdentity>,
        description: Option<String>,
        now: DateTime<Local>,
        host: String,
    ) -> Result<Self> {
        ensure!(
            !implementations.is_empty(),
            "benchmark run has no implementations"
        );
        ensure!(!datasets.is_empty(), "benchmark run has no datasets");
        let mut implementation_ids = implementations.clone();
        implementation_ids.sort_by_key(|id| id.as_str());
        implementation_ids.dedup();
        ensure!(
            implementation_ids.len() == implementations.len(),
            "benchmark run contains duplicate implementations"
        );
        let mut dataset_keys: Vec<String> =
            datasets.iter().map(DatasetIdentity::identity_key).collect();
        dataset_keys.sort();
        dataset_keys.dedup();
        ensure!(
            dataset_keys.len() == datasets.len(),
            "benchmark run contains duplicate datasets"
        );

        fs::create_dir_all(benchmarks_root)?;
        let stamp = now.format("%Y%m%d-%H%M%S").to_string();
        let mut suffix = 0u32;
        let (run_id, root) = loop {
            let value = if suffix == 0 {
                stamp.clone()
            } else {
                format!("{stamp}-{suffix}")
            };
            let run_id = RunId::new(value.clone())?;
            let candidate = benchmarks_root.join(&value);
            match fs::create_dir(&candidate) {
                Ok(()) => break (run_id, candidate),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    suffix += 1;
                    if suffix > 1000 {
                        bail!("could not allocate a unique run directory under {stamp}");
                    }
                }
                Err(error) => return Err(error.into()),
            }
        };

        let identity = RunIdentity {
            run_id,
            started_at: now.to_rfc3339(),
            host,
            description,
            implementations: implementation_ids,
            datasets: datasets.clone(),
        };
        identity.validate()?;
        let manifest = BenchmarkSuiteManifest {
            schema_version: SCHEMA_VERSION,
            run: identity.clone(),
        };
        write_new_file(
            &root.join(RUN_MANIFEST_FILE),
            &(serde_json::to_vec_pretty(&manifest)?),
        )?;
        Ok(Self { identity, root })
    }

    pub fn dataset_dir(&self, dataset: DatasetId) -> PathBuf {
        if self.identity.datasets.len() > 1 {
            self.root.join(dataset.as_str())
        } else {
            self.root.clone()
        }
    }
}

impl DatasetIdentity {
    fn identity_key(&self) -> String {
        match self {
            Self::Catalog { id } => format!("catalog:{}", id.as_str()),
            Self::SingleFile { file_id } => format!("single_file:{file_id}"),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Catalog { .. } => Ok(()),
            Self::SingleFile { file_id } => Self::single_file(file_id.clone()).map(|_| ()),
        }
    }
}

/// Manifest written when a benchmark suite is allocated
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSuiteManifest {
    pub schema_version: u32,
    pub run: RunIdentity,
}

/// A path or an explicit statement that historical data did not record it
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ArtifactLocation {
    Available { path: PathBuf },
    Unavailable { reason: String },
}

impl ArtifactLocation {
    pub fn available(path: impl Into<PathBuf>) -> Self {
        Self::Available { path: path.into() }
    }

    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }

    pub(crate) fn resolve(&self, record_dir: &Path) -> Result<PathBuf> {
        match self {
            Self::Available { path } => {
                if path.is_absolute() {
                    Ok(path.clone())
                } else {
                    Ok(record_dir.join(path))
                }
            }
            Self::Unavailable { reason } => bail!("scoring input is unavailable: {reason}"),
        }
    }

    fn validate(&self, field: &str) -> Result<()> {
        match self {
            Self::Available { path } => {
                ensure!(
                    !path.as_os_str().is_empty(),
                    "{field} has an empty artifact path"
                );
                if path.is_relative() {
                    ensure!(
                        !path.components().any(|component| {
                            matches!(component, std::path::Component::ParentDir)
                        }),
                        "{field} contains a relative path traversal"
                    );
                }
            }
            Self::Unavailable { reason } => ensure!(
                !reason.trim().is_empty(),
                "{field} has an empty unavailable reason"
            ),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputFile {
    pub file_id: String,
    pub reference_rttm: ArtifactLocation,
    pub audio: Option<ArtifactLocation>,
    pub duration_seconds: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionOptions {
    pub policy: String,
    pub max_files: u32,
    pub max_minutes: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputManifest {
    pub files: Vec<InputFile>,
    pub selection: SelectionOptions,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoringOptions {
    pub collar_seconds: f64,
    pub ignore_overlap: bool,
}

impl Default for ScoringOptions {
    fn default() -> Self {
        Self {
            collar_seconds: 0.0,
            ignore_overlap: false,
        }
    }
}

impl ScoringOptions {
    pub fn validate_supported(&self) -> Result<()> {
        ensure!(
            self.collar_seconds == 0.0,
            "stored-run scoring does not support collar_seconds={}; benchmark runs must use collar_seconds=0",
            self.collar_seconds
        );
        ensure!(
            !self.ignore_overlap,
            "stored-run scoring does not support ignore_overlap=true"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMetadata {
    pub git_sha: String,
    pub gpu: String,
    pub cpu: String,
    pub region: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredStatus {
    Completed,
    Skipped,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredPerFileMeasurement {
    pub file_id: String,
    pub reference_speaker_time: f64,
    pub missed: f64,
    pub false_alarm: f64,
    pub confusion: f64,
    pub der: Option<f64>,
    pub missed_percent: Option<f64>,
    pub false_alarm_percent: Option<f64>,
    pub confusion_percent: Option<f64>,
    pub reference_speaker_count: usize,
    pub predicted_speaker_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredMeasurement {
    pub der: Option<f64>,
    pub missed: Option<f64>,
    pub false_alarm: Option<f64>,
    pub confusion: Option<f64>,
    pub reference_speaker_time: Option<f64>,
    pub time_seconds: Option<f64>,
    pub files: usize,
    pub per_file: Vec<StoredPerFileMeasurement>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredImplementationResult {
    pub status: StoredStatus,
    pub reason: Option<String>,
    pub measurement: Option<StoredMeasurement>,
    pub hypotheses: BTreeMap<String, ArtifactLocation>,
    #[serde(default)]
    pub source_metadata: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub schema_version: u32,
    pub run: RunIdentity,
    pub dataset: DatasetIdentity,
    pub inputs: InputManifest,
    pub scoring: ScoringOptions,
    pub metadata: StoredMetadata,
    pub total_audio_minutes: f64,
    pub implementations: BTreeMap<ImplementationId, StoredImplementationResult>,
    #[serde(default)]
    pub source_metadata: BTreeMap<String, Value>,
}

/// Authoritative score calculated from the hypotheses in one or more records
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreReport {
    pub schema_version: u32,
    pub run_id: RunId,
    pub scoring: ScoringOptions,
    pub datasets: Vec<DatasetScore>,
    pub implementations: BTreeMap<ImplementationId, ScoreResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetScore {
    pub dataset: DatasetIdentity,
    pub implementations: BTreeMap<ImplementationId, ScoreResult>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoreResult {
    pub status: StoredStatus,
    pub reason: Option<String>,
    pub measurement: Option<StoredMeasurement>,
}

impl BenchmarkRecord {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "unsupported benchmark schema version {}; expected {SCHEMA_VERSION}",
            self.schema_version
        );
        self.run.validate()?;
        self.dataset.validate()?;
        ensure!(
            self.run
                .datasets
                .iter()
                .any(|dataset| dataset == &self.dataset),
            "dataset {} is not present in the run manifest",
            self.dataset.id_string()
        );
        ensure!(
            !self.inputs.files.is_empty(),
            "benchmark input manifest is empty"
        );
        let mut file_ids = BTreeSet::<String>::new();
        for file in &self.inputs.files {
            ensure!(!file.file_id.is_empty(), "benchmark input file ID is empty");
            ensure!(
                file_ids.insert(file.file_id.clone()),
                "duplicate input file ID {}",
                file.file_id
            );
            file.reference_rttm
                .validate(&format!("reference RTTM for {}", file.file_id))?;
            if let Some(audio) = &file.audio {
                audio.validate(&format!("audio for {}", file.file_id))?;
            }
            if let Some(duration) = file.duration_seconds {
                ensure!(
                    duration.is_finite() && duration >= 0.0,
                    "invalid duration for {}",
                    file.file_id
                );
            }
        }
        for id in &self.run.implementations {
            ensure!(
                self.implementations.contains_key(id),
                "implementation {} is missing a result",
                id.as_str()
            );
        }
        for (id, result) in &self.implementations {
            ensure!(
                self.run.implementations.contains(id),
                "implementation {} is not present in the run manifest",
                id.as_str()
            );
            match result.status {
                StoredStatus::Completed => ensure!(
                    result.measurement.is_some(),
                    "completed implementation {} has no measurement",
                    id.as_str()
                ),
                StoredStatus::Skipped | StoredStatus::Failed => ensure!(
                    result.measurement.is_none(),
                    "{} implementation {} has a fabricated measurement",
                    match result.status {
                        StoredStatus::Skipped => "skipped",
                        StoredStatus::Failed => "failed",
                        StoredStatus::Completed => unreachable!(),
                    },
                    id.as_str()
                ),
            }
            for file_id in result.hypotheses.keys() {
                ensure!(
                    file_ids.contains(file_id),
                    "hypothesis has unknown file ID {file_id}"
                );
                result
                    .hypotheses
                    .get(file_id)
                    .expect("hypothesis key was just checked")
                    .validate(&format!("hypothesis {} {file_id}", id.as_str()))?;
            }
            if result.status == StoredStatus::Completed {
                for file_id in &file_ids {
                    ensure!(
                        result.hypotheses.contains_key(file_id),
                        "completed implementation {} is missing hypothesis for file {}",
                        id.as_str(),
                        file_id
                    );
                }
                if let Some(measurement) = &result.measurement {
                    validate_measurement(measurement, &file_ids, id.as_str())?;
                }
            }
        }
        ensure!(
            self.scoring.collar_seconds.is_finite() && self.scoring.collar_seconds >= 0.0,
            "invalid scoring collar"
        );
        ensure!(
            self.total_audio_minutes.is_finite() && self.total_audio_minutes >= 0.0,
            "invalid total audio duration"
        );
        Ok(())
    }

    pub fn read(path: &Path) -> Result<Self> {
        let raw: Value = serde_json::from_reader(File::open(path)?)?;
        let schema_version = raw
            .get("schema_version")
            .and_then(Value::as_u64)
            .ok_or_else(|| eyre!("benchmark record {} has no schema version", path.display()))?;
        let schema_version = u32::try_from(schema_version).map_err(|_| {
            eyre!(
                "benchmark record {} has an invalid schema version",
                path.display()
            )
        })?;
        let record = match schema_version {
            SCHEMA_VERSION => serde_json::from_value(raw)?,
            LEGACY_SCHEMA_VERSION => {
                let source = path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .canonicalize()
                    .map_err(|error| {
                        eyre!(
                            "cannot read legacy benchmark source {}: {error}",
                            path.display()
                        )
                    })?;
                convert_legacy_record(&raw, &source)?
            }
            version => {
                bail!("unsupported benchmark schema version {version}; expected {SCHEMA_VERSION}")
            }
        };
        record
            .validate()
            .map_err(|error| eyre!("invalid benchmark record {}: {error}", path.display()))?;
        Ok(record)
    }

    pub fn write_new(&self, path: &Path) -> Result<()> {
        self.validate()?;
        write_new_file(path, &serde_json::to_vec_pretty(self)?)
    }
}

fn validate_measurement(
    measurement: &StoredMeasurement,
    file_ids: &BTreeSet<String>,
    implementation: &str,
) -> Result<()> {
    ensure!(
        measurement.files == file_ids.len(),
        "measurement for {implementation} records {} files but the input manifest has {}",
        measurement.files,
        file_ids.len()
    );
    let mut measured_file_ids = BTreeSet::new();
    for file in &measurement.per_file {
        ensure!(
            file_ids.contains(&file.file_id),
            "measurement for {implementation} has unknown file ID {}",
            file.file_id
        );
        ensure!(
            measured_file_ids.insert(&file.file_id),
            "measurement for {implementation} has duplicate file ID {}",
            file.file_id
        );
        for (name, value) in [
            ("reference speaker time", file.reference_speaker_time),
            ("missed", file.missed),
            ("false alarm", file.false_alarm),
            ("confusion", file.confusion),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "measurement for {implementation} has invalid {name} for {}",
                file.file_id
            );
        }
        for (name, value) in [
            ("DER", file.der),
            ("missed percentage", file.missed_percent),
            ("false-alarm percentage", file.false_alarm_percent),
            ("confusion percentage", file.confusion_percent),
        ] {
            if let Some(value) = value {
                ensure!(
                    value.is_finite() && value >= 0.0,
                    "measurement for {implementation} has invalid {name} for {}",
                    file.file_id
                );
            }
        }
    }
    if !measurement.per_file.is_empty() {
        ensure!(
            measured_file_ids.len() == file_ids.len(),
            "measurement for {implementation} is missing per-file measurements"
        );
    }
    for (name, value) in [
        ("DER", measurement.der),
        ("missed", measurement.missed),
        ("false alarm", measurement.false_alarm),
        ("confusion", measurement.confusion),
        ("reference speaker time", measurement.reference_speaker_time),
        ("time", measurement.time_seconds),
    ] {
        if let Some(value) = value {
            ensure!(
                value.is_finite() && value >= 0.0,
                "measurement for {implementation} has invalid {name}"
            );
        }
    }
    Ok(())
}

pub fn write_new_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| eyre!("destination has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    ensure!(
        !path.exists(),
        "destination already exists: {}",
        path.display()
    );
    let temp = create_temp_path(path)?;
    let result = write_temp_and_publish(&temp, path, bytes, false);
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

/// Atomically replace an existing file after all new bytes are durable
pub fn atomic_replace_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| eyre!("destination has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)?;
    let temp = create_temp_path(path)?;
    let result = write_temp_and_publish(&temp, path, bytes, true);
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn create_temp_path(path: &Path) -> Result<PathBuf> {
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .ok_or_else(|| eyre!("destination has no parent: {}", path.display()))?;
    let filename = path
        .file_name()
        .ok_or_else(|| eyre!("destination has no file name: {}", path.display()))?
        .to_string_lossy();
    for _ in 0..100 {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{filename}.tmp-{}-{counter}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                drop(file);
                return Ok(candidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(eyre!(
                    "could not create temporary destination {}: {error}",
                    candidate.display()
                ));
            }
        }
    }
    bail!(
        "could not allocate a temporary destination beside {}",
        path.display()
    )
}

fn write_temp_and_publish(
    temp: &Path,
    destination: &Path,
    bytes: &[u8],
    replace: bool,
) -> Result<()> {
    let mut file = OpenOptions::new().write(true).truncate(true).open(temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    if !replace {
        ensure!(
            !destination.exists(),
            "destination already exists: {}",
            destination.display()
        );
        match fs::hard_link(temp, destination) {
            Ok(()) => {
                fs::remove_file(temp)?;
                sync_parent(destination);
                return Ok(());
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                bail!("destination already exists: {}", destination.display())
            }
            Err(error) => {
                return Err(eyre!(
                    "could not publish new destination {} without replacement: {error}",
                    destination.display()
                ));
            }
        }
    }
    fs::rename(temp, destination)?;
    sync_parent(destination);
    Ok(())
}

fn sync_parent(path: &Path) {
    if let Some(parent) = path.parent()
        && let Ok(directory) = File::open(parent)
    {
        let _ = directory.sync_all();
    }
}

/// Convert an old flat or nested result directory beside the unchanged source
pub fn convert_legacy_results(source: &Path, destination: &Path) -> Result<BenchmarkRecord> {
    let source = source.canonicalize().map_err(|error| {
        eyre!(
            "cannot read legacy benchmark source {}: {error}",
            source.display()
        )
    })?;
    ensure!(
        source.is_dir(),
        "legacy benchmark source must be a directory: {}",
        source.display()
    );
    reject_destination_alias(&source, destination)?;
    ensure!(
        !destination.exists(),
        "conversion destination already exists: {}",
        destination.display()
    );

    let results_path = source.join("results.json");
    ensure!(
        results_path.is_file(),
        "legacy benchmark source has no results.json: {}",
        source.display()
    );
    let raw: Value = serde_json::from_reader(File::open(&results_path)?)?;
    let record = convert_legacy_record(&raw, &source)?;
    record.write_new(destination)?;
    Ok(record)
}

fn convert_legacy_record(raw: &Value, source: &Path) -> Result<BenchmarkRecord> {
    let object = raw
        .as_object()
        .ok_or_else(|| eyre!("legacy results.json must contain an object"))?;
    ensure!(
        object.contains_key("results"),
        "unsupported benchmark archive: results field is missing"
    );
    let results = object["results"]
        .as_object()
        .ok_or_else(|| eyre!("legacy results field must contain an object"))?;

    let run_id = legacy_run_id(object, source)?;
    let dataset =
        legacy_dataset_identity(object, source.file_name().and_then(|name| name.to_str()))?;
    let mut implementation_ids = Vec::new();
    let mut implementation_results = BTreeMap::new();
    for (display_name, value) in results {
        let id = map_legacy_implementation(display_name)?;
        ensure!(
            !implementation_results.contains_key(&id),
            "ambiguous legacy implementation mapping for {display_name}"
        );
        implementation_ids.push(id);
        implementation_results.insert(id, legacy_result(display_name, value, source)?);
    }
    ensure!(
        !implementation_ids.is_empty(),
        "legacy benchmark results are empty"
    );

    let files = legacy_input_files(object, source)?;
    for result in implementation_results.values_mut() {
        if let Some(measurement) = result.measurement.as_mut()
            && measurement.files == 0
        {
            measurement.files = files.len();
        }
        if result.status == StoredStatus::Completed {
            for file in &files {
                result
                    .hypotheses
                    .entry(file.file_id.clone())
                    .or_insert_with(|| {
                        ArtifactLocation::unavailable(
                            "legacy benchmark did not record a hypothesis location",
                        )
                    });
            }
        }
    }
    let started_at = object
        .get("timestamp")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let run = RunIdentity {
        run_id,
        started_at,
        host: "legacy archive".to_owned(),
        description,
        implementations: implementation_ids,
        datasets: vec![dataset.clone()],
    };
    let metadata = StoredMetadata {
        git_sha: value_string(object, "git_sha"),
        gpu: value_string(object, "gpu"),
        cpu: value_string(object, "cpu"),
        region: value_string(object, "region"),
    };
    let total_audio_minutes = object
        .get("total_audio_minutes")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let selection = legacy_selection(object);
    let mut source_metadata = BTreeMap::new();
    source_metadata.insert("legacy_results".to_owned(), raw.clone());
    let record = BenchmarkRecord {
        schema_version: SCHEMA_VERSION,
        run,
        dataset,
        inputs: InputManifest { files, selection },
        scoring: ScoringOptions {
            collar_seconds: object.get("collar").and_then(Value::as_f64).unwrap_or(0.0),
            ignore_overlap: false,
        },
        metadata,
        total_audio_minutes,
        implementations: implementation_results,
        source_metadata,
    };
    record.validate()?;
    Ok(record)
}

fn reject_destination_alias(source: &Path, destination: &Path) -> Result<()> {
    let source = source.canonicalize()?;
    let destination = absolute_path(destination)?;
    let existing = nearest_existing_ancestor(&destination)?;
    ensure!(
        existing != source && !existing.starts_with(&source),
        "conversion destination is inside source archive"
    );
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn nearest_existing_ancestor(path: &Path) -> Result<PathBuf> {
    let mut candidate = path.to_owned();
    while !candidate.exists() {
        candidate = candidate
            .parent()
            .ok_or_else(|| eyre!("destination has no existing ancestor: {}", path.display()))?
            .to_owned();
    }
    Ok(candidate.canonicalize()?)
}

fn legacy_run_id(object: &serde_json::Map<String, Value>, source: &Path) -> Result<RunId> {
    if let Some(value) = object.get("run_id") {
        let value = value
            .as_str()
            .ok_or_else(|| eyre!("legacy run_id field must be a string"))?;
        return RunId::new(value.to_owned());
    }
    let name = source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| eyre!("legacy result path has no UTF-8 name"))?;
    if looks_like_timestamp(name) {
        return RunId::new(name.to_owned());
    }
    let parent = source
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .ok_or_else(|| eyre!("nested legacy result missing parent run ID"))?;
    ensure!(
        looks_like_timestamp(parent),
        "cannot derive legacy run ID from {}",
        source.display()
    );
    RunId::new(parent.to_owned())
}

fn legacy_dataset_identity(
    object: &serde_json::Map<String, Value>,
    directory_name: Option<&str>,
) -> Result<DatasetIdentity> {
    let raw = object
        .get("dataset")
        .and_then(Value::as_str)
        .or(directory_name)
        .ok_or_else(|| eyre!("legacy result has no dataset identity"))?;
    if let Some(spec) = DatasetCatalog::parse_cli(raw).or_else(|| {
        DatasetCatalog::all()
            .iter()
            .find(|spec| spec.display_name.eq_ignore_ascii_case(raw))
    }) {
        return Ok(DatasetIdentity::catalog(spec.id));
    }
    let file_list = object.get("file_list").and_then(Value::as_array);
    if file_list.is_some_and(|files| files.len() == 1 && files[0].as_str() == Some(raw)) {
        return DatasetIdentity::single_file(raw.to_owned());
    }
    bail!("unknown or ambiguous legacy dataset identity {raw}")
}

fn legacy_input_files(
    object: &serde_json::Map<String, Value>,
    source: &Path,
) -> Result<Vec<InputFile>> {
    let ids = object
        .get("file_list")
        .and_then(Value::as_array)
        .ok_or_else(|| eyre!("legacy result has no file_list"))?;
    ensure!(!ids.is_empty(), "legacy result file_list is empty");
    ids.iter()
        .map(|id| {
            let (file_id, reference_rttm, audio, duration_seconds) =
                if let Some(file_id) = id.as_str() {
                    (file_id.to_owned(), None, None, None)
                } else if let Some(file) = id.as_object() {
                    let file_id = file
                        .get("file_id")
                        .or_else(|| file.get("id"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| eyre!("legacy file_list object has no file ID"))?
                        .to_owned();
                    (
                        file_id,
                        file.get("reference_rttm")
                            .or_else(|| file.get("rttm"))
                            .and_then(Value::as_str)
                            .map(|path| legacy_available_path(source, path)),
                        file.get("audio")
                            .or_else(|| file.get("wav"))
                            .and_then(Value::as_str)
                            .map(|path| legacy_available_path(source, path)),
                        file.get("duration_seconds")
                            .or_else(|| file.get("duration"))
                            .and_then(Value::as_f64),
                    )
                } else {
                    return Err(eyre!(
                        "legacy file_list contains neither a string nor an object"
                    ));
                };
            ensure!(
                !file_id.is_empty(),
                "legacy file_list contains an empty file ID"
            );
            let reference_rttm = match reference_rttm {
                Some(location) => location,
                None => legacy_location_for_file(object, "reference_rttm", &file_id, source)?
                    .unwrap_or_else(|| {
                        ArtifactLocation::unavailable(
                            "legacy benchmark did not record reference RTTM locations",
                        )
                    }),
            };
            let audio = match audio {
                Some(location) => Some(location),
                None => legacy_location_for_file(object, "audio", &file_id, source)?,
            };
            Ok(InputFile {
                file_id,
                reference_rttm,
                audio,
                duration_seconds,
            })
        })
        .collect()
}

fn legacy_location_for_file(
    object: &serde_json::Map<String, Value>,
    field: &str,
    file_id: &str,
    source: &Path,
) -> Result<Option<ArtifactLocation>> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let values = value
        .as_object()
        .ok_or_else(|| eyre!("legacy {field} field must be an object keyed by file ID"))?;
    let Some(value) = values.get(file_id) else {
        return Ok(None);
    };
    if let Some(path) = value.as_str() {
        return Ok(Some(legacy_available_path(source, path)));
    }
    let location: ArtifactLocation = serde_json::from_value(value.clone())
        .map_err(|error| eyre!("legacy {field} location for {file_id} is invalid: {error}"))?;
    Ok(Some(normalize_legacy_location(source, location)))
}

fn legacy_available_path(source: &Path, path: &str) -> ArtifactLocation {
    ArtifactLocation::available(normalize_legacy_path(source, path))
}

fn normalize_legacy_location(source: &Path, location: ArtifactLocation) -> ArtifactLocation {
    match location {
        ArtifactLocation::Available { path } if path.is_relative() => {
            ArtifactLocation::available(source.join(path))
        }
        location => location,
    }
}

fn normalize_legacy_path(source: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_owned()
    } else {
        source.join(path)
    }
}

fn legacy_selection(object: &serde_json::Map<String, Value>) -> SelectionOptions {
    let selection = object.get("selection_limits").and_then(Value::as_object);
    SelectionOptions {
        policy: object
            .get("selection_policy")
            .and_then(Value::as_str)
            .unwrap_or("legacy")
            .to_owned(),
        max_files: selection
            .and_then(|value| value.get("max_files"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(u32::MAX),
        max_minutes: selection
            .and_then(|value| value.get("max_minutes"))
            .and_then(Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(u32::MAX),
    }
}

fn legacy_result(
    display_name: &str,
    value: &Value,
    source: &Path,
) -> Result<StoredImplementationResult> {
    let object = value
        .as_object()
        .ok_or_else(|| eyre!("legacy result for {display_name} must contain an object"))?;
    let status = match object.get("status").and_then(Value::as_str) {
        Some("failed") => StoredStatus::Failed,
        Some("skipped") => StoredStatus::Skipped,
        Some("completed") | None => StoredStatus::Completed,
        Some(status) => bail!("unknown legacy status {status} for {display_name}"),
    };
    let measurement = match status {
        StoredStatus::Completed => Some(legacy_measurement(object, display_name)?),
        StoredStatus::Skipped | StoredStatus::Failed => None,
    };
    let hypotheses = legacy_hypotheses(object, display_name, source)?;
    let source_metadata = [("legacy_result".to_owned(), value.clone())]
        .into_iter()
        .collect();
    Ok(StoredImplementationResult {
        status,
        reason: object
            .get("reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        measurement,
        hypotheses,
        source_metadata,
    })
}

fn legacy_measurement(
    object: &serde_json::Map<String, Value>,
    display_name: &str,
) -> Result<StoredMeasurement> {
    let per_file = legacy_per_file_measurements(object, display_name)?;
    let files = object
        .get("files")
        .or_else(|| object.get("file_count"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .or_else(|| (!per_file.is_empty()).then_some(per_file.len()))
        .unwrap_or(0);
    Ok(StoredMeasurement {
        der: number_field(object, &["der", "der_percent"]),
        missed: number_field(object, &["missed", "missed_percent"]),
        false_alarm: number_field(object, &["false_alarm", "false_alarm_percent"]),
        confusion: number_field(object, &["confusion", "confusion_percent"]),
        reference_speaker_time: number_field(
            object,
            &[
                "reference_speaker_time",
                "ref_speaker_time",
                "reference_time",
                "reference_speech_time",
                "total_ref",
            ],
        ),
        time_seconds: number_field(object, &["time", "time_seconds", "total_audio_seconds"]),
        files,
        per_file,
    })
}

fn number_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_f64))
}

fn required_number_field(object: &serde_json::Map<String, Value>, names: &[&str]) -> Result<f64> {
    number_field(object, names).ok_or_else(|| {
        eyre!(
            "legacy per-file result is missing one of the numeric fields: {}",
            names.join(", ")
        )
    })
}

fn legacy_per_file_measurements(
    object: &serde_json::Map<String, Value>,
    display_name: &str,
) -> Result<Vec<StoredPerFileMeasurement>> {
    let Some(value) = object
        .get("per_file")
        .or_else(|| object.get("per_file_results"))
    else {
        return Ok(Vec::new());
    };
    let entries = if let Some(entries) = value.as_array() {
        entries
            .iter()
            .map(|entry| {
                entry.as_object().cloned().ok_or_else(|| {
                    eyre!("legacy per-file result for {display_name} is not an object")
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else if let Some(entries) = value.as_object() {
        entries
            .iter()
            .map(|(file_id, entry)| {
                let mut object = entry.as_object().cloned().ok_or_else(|| {
                    eyre!("legacy per-file result for {display_name} is not an object")
                })?;
                object
                    .entry("file_id".to_owned())
                    .or_insert_with(|| Value::String(file_id.clone()));
                Ok(object)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        bail!("legacy per-file result for {display_name} must be an array or object")
    };
    entries
        .into_iter()
        .map(|entry| {
            let file_id = entry
                .get("file_id")
                .or_else(|| entry.get("id"))
                .and_then(Value::as_str)
                .ok_or_else(|| eyre!("legacy per-file result for {display_name} has no file ID"))?
                .to_owned();
            Ok(StoredPerFileMeasurement {
                file_id,
                reference_speaker_time: required_number_field(
                    &entry,
                    &[
                        "reference_speaker_time",
                        "ref_speaker_time",
                        "reference_time",
                        "reference_speech_time",
                        "total_ref",
                    ],
                )?,
                missed: required_number_field(&entry, &["missed", "missed_seconds"])?,
                false_alarm: required_number_field(
                    &entry,
                    &["false_alarm", "false_alarm_seconds"],
                )?,
                confusion: required_number_field(&entry, &["confusion", "confusion_seconds"])?,
                der: number_field(&entry, &["der", "der_percent"]),
                missed_percent: number_field(&entry, &["missed_percent"]),
                false_alarm_percent: number_field(&entry, &["false_alarm_percent"]),
                confusion_percent: number_field(&entry, &["confusion_percent"]),
                reference_speaker_count: entry
                    .get("reference_speaker_count")
                    .or_else(|| entry.get("ref_speakers"))
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0),
                predicted_speaker_count: entry
                    .get("predicted_speaker_count")
                    .or_else(|| entry.get("hyp_speakers"))
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0),
            })
        })
        .collect()
}

fn legacy_hypotheses(
    object: &serde_json::Map<String, Value>,
    display_name: &str,
    source: &Path,
) -> Result<BTreeMap<String, ArtifactLocation>> {
    let Some(value) = object.get("hypotheses") else {
        return Ok(BTreeMap::new());
    };
    let entries = value
        .as_object()
        .ok_or_else(|| eyre!("legacy hypotheses for {display_name} must be an object"))?;
    entries
        .iter()
        .map(|(file_id, value)| {
            let location = if let Some(path) = value.as_str() {
                legacy_available_path(source, path)
            } else {
                serde_json::from_value(value.clone())
                    .map(|location| normalize_legacy_location(source, location))
                    .map_err(|error| {
                        eyre!("legacy hypothesis {display_name}/{file_id} is invalid: {error}")
                    })?
            };
            Ok((file_id.clone(), location))
        })
        .collect()
}

fn map_legacy_implementation(value: &str) -> Result<ImplementationId> {
    let normalized = value.trim().to_ascii_lowercase();
    if let Some(spec) = ImplementationCatalog::all().iter().find(|spec| {
        spec.display_name.eq_ignore_ascii_case(value)
            || spec.cli_name().eq_ignore_ascii_case(value)
            || spec
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(value))
    }) {
        return Ok(spec.id);
    }
    let historical = match normalized.as_str() {
        "speakrs" | "speakrs filter" | "speakrs smooth" | "speakrs cvbx" => {
            Some(ImplementationId::SpeakrsCoreMl)
        }
        "speakrs coreml fastlite" => Some(ImplementationId::SpeakrsCoreMlFast),
        "speakrs cuda hybrid" => Some(ImplementationId::SpeakrsCuda),
        _ => None,
    };
    historical.ok_or_else(|| eyre!("unknown or ambiguous legacy implementation identity {value}"))
}

fn value_string(object: &serde_json::Map<String, Value>, key: &str) -> String {
    object
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn looks_like_timestamp(name: &str) -> bool {
    name.len() >= 15
        && name.as_bytes().get(8) == Some(&b'-')
        && name.as_bytes().get(15).is_none_or(|byte| *byte == b'-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ImplementationId;
    use crate::datasets::DatasetId;
    use std::fs;

    const SCHEMA_V2_WRITER_FIXTURE: &str = r#"{
      "schema_version": 2,
      "dataset": "VoxConverse Dev",
      "run_id": "20240101-010203",
      "timestamp": "2024-01-01T01:02:03Z",
      "git_sha": "old-sha",
      "gpu": "none",
      "cpu": "old-cpu",
      "region": "local",
      "files": 1,
      "total_audio_minutes": 2.5,
      "collar": 0.0,
      "selection_policy": "shortest_first_by_duration",
      "selection_limits": {"max_files": 10, "max_minutes": 20},
      "pyannote_batch_sizes": {
        "note": "null means the script default (cuda=32, non-cuda=16)",
        "segmentation_override": null,
        "embedding_override": null
      },
      "file_list": ["clip"],
      "results": {
        "SpeakerKit": {
          "status": "completed",
          "reason": null,
          "der": 12.5,
          "missed": 3.0,
          "false_alarm": 4.0,
          "confusion": 5.5,
          "time": 4.0,
          "files": 1
        }
      }
    }"#;

    fn completed(id: ImplementationId) -> (ImplementationId, StoredImplementationResult) {
        (
            id,
            StoredImplementationResult {
                status: StoredStatus::Completed,
                reason: None,
                measurement: Some(StoredMeasurement {
                    der: Some(2.0),
                    missed: Some(1.0),
                    false_alarm: Some(0.5),
                    confusion: Some(0.5),
                    reference_speaker_time: Some(10.0),
                    time_seconds: Some(1.0),
                    files: 1,
                    per_file: Vec::new(),
                }),
                hypotheses: [("one".to_owned(), ArtifactLocation::unavailable("test"))]
                    .into_iter()
                    .collect(),
                source_metadata: BTreeMap::new(),
            },
        )
    }

    fn record(run: &BenchmarkRun) -> BenchmarkRecord {
        let id = ImplementationId::SpeakrsCpu;
        let (id, result) = completed(id);
        BenchmarkRecord {
            schema_version: SCHEMA_VERSION,
            run: run.identity.clone(),
            dataset: DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            inputs: InputManifest {
                files: vec![InputFile {
                    file_id: "one".to_owned(),
                    reference_rttm: ArtifactLocation::unavailable("test"),
                    audio: None,
                    duration_seconds: Some(1.0),
                }],
                selection: SelectionOptions {
                    policy: "test".to_owned(),
                    max_files: 1,
                    max_minutes: 1,
                },
            },
            scoring: ScoringOptions::default(),
            metadata: StoredMetadata {
                git_sha: "test".to_owned(),
                gpu: "none".to_owned(),
                cpu: "test".to_owned(),
                region: "local".to_owned(),
            },
            total_audio_minutes: 1.0 / 60.0,
            implementations: [(id, result)].into_iter().collect(),
            source_metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn same_time_runs_do_not_collide() {
        let dir = tempfile::tempdir().unwrap();
        let now = Local::now();
        let first = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            now,
            "host".to_owned(),
        )
        .unwrap();
        let second = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            now,
            "host".to_owned(),
        )
        .unwrap();
        assert_ne!(first.identity.run_id, second.identity.run_id);
        assert!(first.root.exists());
        assert!(second.root.exists());
    }

    #[test]
    fn renamed_directory_does_not_change_recorded_identity() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        let path = run.root.join("results.json");
        let original = record(&run);
        original.write_new(&path).unwrap();
        let renamed = dir.path().join("renamed");
        fs::rename(&run.root, &renamed).unwrap();
        let loaded = BenchmarkRecord::read(&renamed.join("results.json")).unwrap();
        assert_eq!(loaded.run.run_id, original.run.run_id);
    }

    #[test]
    fn read_dispatches_real_schema_v2_writer_fixture() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("renamed-archive");
        fs::create_dir_all(&archive).unwrap();
        let path = archive.join("results.json");
        fs::write(&path, SCHEMA_V2_WRITER_FIXTURE).unwrap();
        let source_bytes = fs::read(&path).unwrap();

        let record = BenchmarkRecord::read(&path).unwrap();

        assert_eq!(record.schema_version, SCHEMA_VERSION);
        assert_eq!(record.run.run_id.as_str(), "20240101-010203");
        assert_eq!(
            record.dataset,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev)
        );
        assert_eq!(record.inputs.files[0].file_id, "clip");
        assert_eq!(
            record.implementations[&ImplementationId::SpeakerKit]
                .measurement
                .as_ref()
                .unwrap()
                .der,
            Some(12.5)
        );
        assert_eq!(
            record.source_metadata["legacy_results"]["schema_version"].as_u64(),
            Some(2)
        );
        assert_eq!(source_bytes, fs::read(&path).unwrap());
    }

    #[test]
    fn multi_dataset_run_has_one_suite_and_dataset_children() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev, DatasetId::AmiIhm],
            None,
            Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        assert_eq!(
            run.dataset_dir(DatasetId::VoxconverseDev),
            run.root.join("voxconverse-dev")
        );
        assert_eq!(run.dataset_dir(DatasetId::AmiIhm), run.root.join("ami-ihm"));
        assert_eq!(run.identity.datasets.len(), 2);
    }

    #[test]
    fn legacy_conversion_preserves_measurements_and_source() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("20240101-010203");
        fs::create_dir_all(&source).unwrap();
        let raw = br#"{
          "dataset":"VoxConverse Dev",
          "timestamp":"2024-01-01T01:02:03Z",
          "file_list":["a","b"],
          "files":2,
          "total_audio_minutes":2.5,
          "collar":0.0,
          "results":{
            "SpeakRs":{"der":10.0,"missed":2.0,"false_alarm":3.0,"confusion":5.0,"time":4.0,"files":2},
            "SpeakerKit":{"status":"failed","reason":"timeout","der":null,"files":0}
          }
        }"#;
        fs::write(source.join("results.json"), raw).unwrap();
        let checksum = sha256(&source.join("results.json"));
        let destination = dir.path().join("converted.json");
        let converted = convert_legacy_results(&source, &destination).unwrap();
        assert_eq!(converted.schema_version, SCHEMA_VERSION);
        assert_eq!(converted.run.run_id.as_str(), "20240101-010203");
        assert_eq!(converted.inputs.files.len(), 2);
        assert_eq!(
            converted.implementations[&ImplementationId::SpeakrsCoreMl]
                .measurement
                .as_ref()
                .unwrap()
                .der,
            Some(10.0)
        );
        assert_eq!(
            converted.implementations[&ImplementationId::SpeakerKit].status,
            StoredStatus::Failed
        );
        assert_eq!(checksum, sha256(&source.join("results.json")));
    }

    #[test]
    fn legacy_conversion_prefers_recorded_run_id_after_archive_rename() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("20240101-010203");
        fs::create_dir_all(&original).unwrap();
        fs::write(
            original.join("results.json"),
            r#"{
              "schema_version": 2,
              "run_id": "recorded-run-id",
              "dataset": "VoxConverse Dev",
              "file_list": ["clip"],
              "results": {"SpeakerKit": {"status": "failed", "reason": "timeout", "files": 0}}
            }"#,
        )
        .unwrap();
        let renamed = dir.path().join("renamed-archive");
        fs::rename(&original, &renamed).unwrap();

        let destination = dir.path().join("converted.json");
        let converted = convert_legacy_results(&renamed, &destination).unwrap();

        assert_eq!(converted.run.run_id.as_str(), "recorded-run-id");
    }

    #[test]
    fn conversion_rejects_destination_conflict_and_alias() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("20240101-010203");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("results.json"),
            r#"{"dataset":"VoxConverse Dev","file_list":["a"],"results":{"SpeakerKit":{"der":1.0,"files":1}}}"#,
        )
        .unwrap();
        let conflict = dir.path().join("out.json");
        fs::write(&conflict, b"existing").unwrap();
        assert!(convert_legacy_results(&source, &conflict).is_err());
        assert!(convert_legacy_results(&source, &source.join("out.json")).is_err());
    }

    #[test]
    fn write_new_rejects_existing_files_and_atomic_replace_is_repeatable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("result.json");
        write_new_file(&path, b"first").unwrap();
        assert!(write_new_file(&path, b"second").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first");
        atomic_replace_file(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
    }

    #[test]
    fn flat_and_nested_layouts_use_explicit_readers() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("20240101-010203").join("ami-ihm");
        fs::create_dir_all(&source).unwrap();
        fs::write(
            source.join("results.json"),
            r#"{"dataset":"AMI IHM","file_list":["a"],"results":{"SpeakerKit":{"der":1.0,"files":1}}}"#,
        )
        .unwrap();
        let destination = dir.path().join("nested.json");
        let converted = convert_legacy_results(&source, &destination).unwrap();
        assert_eq!(converted.run.run_id.as_str(), "20240101-010203");
        assert_eq!(
            converted.dataset,
            DatasetIdentity::catalog(DatasetId::AmiIhm)
        );
    }

    #[test]
    fn legacy_conversion_preserves_per_file_data_and_relative_artifact_targets() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("20240101-010203");
        fs::create_dir_all(source.join("artifacts")).unwrap();
        fs::write(source.join("artifacts/ref.rttm"), b"reference").unwrap();
        fs::write(source.join("artifacts/hyp.rttm"), b"hypothesis").unwrap();
        fs::write(
            source.join("results.json"),
            r#"{
              "dataset":"VoxConverse Dev",
              "file_list":[{"file_id":"a","reference_rttm":"artifacts/ref.rttm","duration_seconds":2.0}],
              "results":{"SpeakerKit":{"files":1,"der":25.0,"per_file":[{"file_id":"a","reference_speaker_time":2.0,"missed":0.5,"false_alarm":0.0,"confusion":0.0,"der":25.0}],"hypotheses":{"a":"artifacts/hyp.rttm"}}}
            }"#,
        )
        .unwrap();
        let destination = dir.path().join("converted.json");
        let converted = convert_legacy_results(&source, &destination).unwrap();
        let source = source.canonicalize().unwrap();
        let file = &converted.inputs.files[0];
        assert_eq!(file.duration_seconds, Some(2.0));
        assert_eq!(
            file.reference_rttm,
            ArtifactLocation::available(source.join("artifacts/ref.rttm"))
        );
        let result = &converted.implementations[&ImplementationId::SpeakerKit];
        assert_eq!(result.measurement.as_ref().unwrap().per_file.len(), 1);
        assert_eq!(
            result.hypotheses["a"],
            ArtifactLocation::available(source.join("artifacts/hyp.rttm"))
        );
    }

    fn sha256(path: &Path) -> Vec<u8> {
        use sha2::{Digest, Sha256};
        let mut digest = Sha256::new();
        digest.update(fs::read(path).unwrap());
        digest.finalize().to_vec()
    }
}
