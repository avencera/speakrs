use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, ensure};

use super::run_store::{
    ArtifactLocation, BenchmarkRecord, DatasetIdentity, InputFile, InputManifest, RunIdentity,
    ScoringOptions, SelectionOptions, StoredImplementationResult, StoredMeasurement,
    StoredMetadata, StoredPerFileMeasurement, StoredStatus, write_new_file,
};
use super::*;
use crate::catalog::ImplementationId;
use crate::cmd::wav_duration_seconds;
use crate::path::file_stem_string;

pub struct DerResultsWriter<'a> {
    pub run_dir: &'a Path,
    pub run_identity: &'a RunIdentity,
    pub dataset: DatasetIdentity,
    pub implementations: &'a [(ImplementationId, ImplType)],
    pub results: &'a HashMap<ImplementationId, DerImplResult>,
    pub files: &'a [(PathBuf, PathBuf)],
    pub total_audio_minutes: f64,
    pub collar: f64,
    pub description: Option<&'a str>,
    pub max_files: u32,
    pub max_minutes: u32,
    pub metadata: &'a BenchmarkMetadata,
    pub pyannote_batch_sizes: PyannoteBatchSizes,
}

struct SummaryFormatter<'a>(&'a BenchmarkRecord);

struct PendingArtifact {
    relative_path: PathBuf,
    bytes: Vec<u8>,
}

struct PreparedRecord {
    record: BenchmarkRecord,
    artifacts: Vec<PendingArtifact>,
}

struct Publication {
    published_files: Vec<PathBuf>,
    created_directories: Vec<PathBuf>,
    committed: bool,
}

impl Publication {
    fn new() -> Self {
        Self {
            published_files: Vec::new(),
            created_directories: Vec::new(),
            committed: false,
        }
    }

    fn publish(&mut self, staged: &Path, destination: &Path) -> Result<()> {
        let parent = destination.parent().ok_or_else(|| {
            color_eyre::eyre::eyre!("destination has no parent: {}", destination.display())
        })?;
        self.create_parent_directories(parent)?;
        ensure!(
            !destination.exists(),
            "destination already exists: {}",
            destination.display()
        );
        fs::hard_link(staged, destination)?;
        self.published_files.push(destination.to_owned());
        Ok(())
    }

    fn create_parent_directories(&mut self, directory: &Path) -> Result<()> {
        if directory.as_os_str().is_empty() {
            return Ok(());
        }
        let mut missing = Vec::new();
        let mut current = directory;
        while !current.exists() {
            missing.push(current.to_owned());
            current = current.parent().ok_or_else(|| {
                color_eyre::eyre::eyre!("destination has no parent: {}", directory.display())
            })?;
        }
        ensure!(
            current.is_dir(),
            "destination parent is not a directory: {}",
            current.display()
        );
        for path in missing.into_iter().rev() {
            match fs::create_dir(&path) {
                Ok(()) => self.created_directories.push(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    ensure!(
                        path.is_dir(),
                        "destination parent is not a directory: {}",
                        path.display()
                    );
                }
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn commit(mut self) {
        self.committed = true;
    }

    fn rollback(&mut self) -> Result<()> {
        let mut failures = Vec::new();
        for path in self.published_files.iter().rev() {
            if let Err(error) = fs::remove_file(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                failures.push(format!("remove {}: {error}", path.display()));
            }
        }
        for path in self.created_directories.iter().rev() {
            if let Err(error) = fs::remove_dir(path)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                failures.push(format!("remove {}: {error}", path.display()));
            }
        }
        if failures.is_empty() {
            self.committed = true;
            Ok(())
        } else {
            Err(color_eyre::eyre::eyre!(failures.join("; ")))
        }
    }
}

impl Drop for Publication {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        for path in self.published_files.iter().rev() {
            let _ = fs::remove_file(path);
        }
        for path in self.created_directories.iter().rev() {
            let _ = fs::remove_dir(path);
        }
    }
}

impl<'a> SummaryFormatter<'a> {
    fn build_lines(&self) -> Vec<String> {
        let record = self.0;
        let batch_sizes = record.source_metadata.get("pyannote_batch_sizes");
        let seg_batch_size = batch_sizes
            .and_then(|value| value.get("segmentation"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        let emb_batch_size = batch_sizes
            .and_then(|value| value.get("embedding"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("default");
        let file_list = record
            .inputs
            .files
            .iter()
            .map(|file| file.file_id.as_str())
            .collect::<Vec<_>>();
        let total_audio_seconds = record.total_audio_minutes * 60.0;
        let header = self.header_line();
        let metadata = &record.metadata;
        let mut lines = vec![
            format!(
                "{} DER ({} files, {:.1} min, collar={:.0}ms)",
                record.dataset.display_name(),
                record.inputs.files.len(),
                record.total_audio_minutes,
                record.scoring.collar_seconds * 1000.0,
            ),
            format!(
                "Commit: {}  GPU: {}  CPU: {}  Region: {}",
                metadata.git_sha, metadata.gpu, metadata.cpu, metadata.region
            ),
            format!(
                "Selection: greedy by duration, executed in input order, capped at max_files={}, max_minutes={}",
                record.inputs.selection.max_files, record.inputs.selection.max_minutes
            ),
            format!("pyannote batch sizes: seg={seg_batch_size}, emb={emb_batch_size}"),
            format!("Files: {}", file_list.join(", ")),
        ];
        if let Some(description) = &record.run.description {
            lines.push(format!("Description: {description}"));
        }
        lines.push(String::new());
        lines.push(header.clone());
        lines.push("─".repeat(header.len()));

        for implementation_id in &record.run.implementations {
            if let Some(result) = record.implementations.get(implementation_id) {
                let name = crate::catalog::ImplementationCatalog::display_name(*implementation_id);
                lines.push(self.result_line(name, result, total_audio_seconds));
            }
        }

        lines
    }

    fn header_line(&self) -> String {
        let name_width = 22;
        format!(
            "{:<name_width$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
            "Implementation",
            "DER%",
            "Missed%",
            "FalseAlarm%",
            "Confusion%",
            "Time",
            "RTFx",
            "Status"
        )
    }

    fn result_line(
        &self,
        implementation_name: &str,
        result: &StoredImplementationResult,
        total_audio_seconds: f64,
    ) -> String {
        let name_width = 22;
        let measurement = result.measurement.as_ref();
        let (der_str, missed_str, false_alarm_str, confusion_str) =
            match measurement.and_then(|measurement| measurement.der) {
                Some(der) => (
                    format!("{der:.1}%"),
                    format!(
                        "{:.1}%",
                        measurement.and_then(|value| value.missed).unwrap_or(0.0)
                    ),
                    format!(
                        "{:.1}%",
                        measurement
                            .and_then(|value| value.false_alarm)
                            .unwrap_or(0.0)
                    ),
                    format!(
                        "{:.1}%",
                        measurement.and_then(|value| value.confusion).unwrap_or(0.0)
                    ),
                ),
                None => (
                    "N/A".to_owned(),
                    "—".to_owned(),
                    "—".to_owned(),
                    "—".to_owned(),
                ),
            };
        let time = measurement.and_then(|measurement| measurement.time_seconds);
        let time_str = time
            .map(|time| format!("{time:.1}s"))
            .unwrap_or_else(|| "—".to_owned());
        let rtfx_str = time
            .filter(|time| *time > 0.0)
            .map(|time| format!("{:.1}x", total_audio_seconds / time))
            .unwrap_or_else(|| "—".to_owned());
        let status_str = match result.status {
            StoredStatus::Completed => "ok".to_owned(),
            StoredStatus::Skipped => format!(
                "skipped ({})",
                result.reason.as_deref().unwrap_or("no reason recorded")
            ),
            StoredStatus::Failed => {
                let reason = result.reason.as_deref().unwrap_or("no reason recorded");
                format!("failed ({reason})")
            }
        };
        format!(
            "{:<name_width$} {:>8} {:>10} {:>13} {:>12} {:>8} {:>7}  {}",
            implementation_name,
            der_str,
            missed_str,
            false_alarm_str,
            confusion_str,
            time_str,
            rtfx_str,
            status_str
        )
    }
}

impl<'a> DerResultsWriter<'a> {
    pub fn write(&self) -> Result<()> {
        let prepared = self.build_record()?;
        let summary_lines = SummaryFormatter(&prepared.record).build_lines();
        let record_bytes = serde_json::to_vec_pretty(&prepared.record)?;
        let summary_bytes = (summary_lines.join("\n") + "\n").into_bytes();
        let mut artifacts = prepared.artifacts;
        artifacts.push(PendingArtifact {
            relative_path: PathBuf::from("results.json"),
            bytes: record_bytes,
        });
        artifacts.push(PendingArtifact {
            relative_path: PathBuf::from("results.txt"),
            bytes: summary_bytes,
        });
        self.publish_artifacts(artifacts)?;

        println!("\nResults saved to {}/", self.run_dir.display());
        for line in &summary_lines {
            println!("{line}");
        }

        Ok(())
    }

    fn build_record(&self) -> Result<PreparedRecord> {
        ensure!(
            self.run_identity
                .datasets
                .iter()
                .any(|dataset| dataset == &self.dataset),
            "dataset {} is not present in the run identity",
            self.dataset.id_string()
        );
        let files = self
            .files
            .iter()
            .map(|(wav, rttm)| {
                let file_id = file_stem_string(wav)?;
                let duration = wav_duration_seconds(wav)?;
                Ok(InputFile {
                    file_id,
                    reference_rttm: ArtifactLocation::available(stable_input_path(rttm)?),
                    audio: Some(ArtifactLocation::available(stable_input_path(wav)?)),
                    duration_seconds: Some(duration),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut implementations = BTreeMap::new();
        let mut artifacts = Vec::new();
        for (implementation_id, _) in self.implementations {
            let result = self.results.get(implementation_id).ok_or_else(|| {
                color_eyre::eyre::eyre!("missing result for {}", implementation_id.as_str())
            })?;
            let (stored, mut implementation_artifacts) =
                self.store_implementation(*implementation_id, result, &files)?;
            implementations.insert(*implementation_id, stored);
            artifacts.append(&mut implementation_artifacts);
        }

        let record = BenchmarkRecord {
            schema_version: super::run_store::SCHEMA_VERSION,
            run: self.run_identity.clone(),
            dataset: self.dataset.clone(),
            inputs: InputManifest {
                files,
                selection: SelectionOptions {
                    policy: "shortest_first_by_duration".to_owned(),
                    max_files: self.max_files,
                    max_minutes: self.max_minutes,
                },
            },
            scoring: ScoringOptions {
                collar_seconds: self.collar,
                ignore_overlap: false,
            },
            metadata: StoredMetadata {
                git_sha: self.metadata.git_sha.clone(),
                gpu: self.metadata.gpu.clone(),
                cpu: self.metadata.cpu.clone(),
                region: self.metadata.region.clone(),
            },
            total_audio_minutes: self.total_audio_minutes,
            implementations,
            source_metadata: self.source_metadata(),
        };
        record.validate()?;
        Ok(PreparedRecord { record, artifacts })
    }

    fn publish_artifacts(&self, artifacts: Vec<PendingArtifact>) -> Result<()> {
        let run_parent = self.run_dir.parent().unwrap_or_else(|| Path::new("."));
        let staging = tempfile::Builder::new()
            .prefix(".benchmark-results-")
            .tempdir_in(run_parent)?;
        let mut staged_paths = Vec::with_capacity(artifacts.len());
        for artifact in &artifacts {
            validate_artifact_path(&artifact.relative_path)?;
            let destination = self.run_dir.join(&artifact.relative_path);
            ensure!(
                !destination.exists(),
                "destination already exists: {}",
                destination.display()
            );
            let staged = staging.path().join(&artifact.relative_path);
            write_new_file(&staged, &artifact.bytes)?;
            staged_paths.push((staged, destination));
        }

        let mut publication = Publication::new();
        for (staged, destination) in staged_paths {
            if let Err(error) = publication.publish(&staged, &destination) {
                return match publication.rollback() {
                    Ok(()) => Err(error),
                    Err(rollback_error) => Err(color_eyre::eyre::eyre!(
                        "artifact publication failed ({error}); rollback also failed ({rollback_error})"
                    )),
                };
            }
        }
        publication.commit();
        Ok(())
    }

    fn store_implementation(
        &self,
        implementation_id: ImplementationId,
        result: &DerImplResult,
        files: &[InputFile],
    ) -> Result<(StoredImplementationResult, Vec<PendingArtifact>)> {
        let status = match result.status {
            DerImplStatus::Completed => StoredStatus::Completed,
            DerImplStatus::Skipped => StoredStatus::Skipped,
            DerImplStatus::Failed => StoredStatus::Failed,
        };
        let (measurement, hypotheses, artifacts) = if matches!(status, StoredStatus::Completed) {
            let display_name =
                crate::catalog::ImplementationCatalog::display_name(implementation_id);
            let input_mismatch = completed_input_mismatch_reason(
                display_name,
                result,
                files.iter().map(|file| file.file_id.as_str()),
            );
            if let Some(reason) = input_mismatch {
                return Ok((
                    StoredImplementationResult {
                        status: StoredStatus::Failed,
                        reason: Some(reason),
                        measurement: None,
                        hypotheses: BTreeMap::new(),
                        source_metadata: BTreeMap::new(),
                    },
                    Vec::new(),
                ));
            }
            let measurement = StoredMeasurement {
                der: result.der,
                missed: result.missed,
                false_alarm: result.false_alarm,
                confusion: result.confusion,
                reference_speaker_time: Some(
                    result
                        .per_file
                        .iter()
                        .map(|file| file.reference_speaker_time)
                        .sum(),
                ),
                time_seconds: result.time,
                files: result.files,
                per_file: result
                    .per_file
                    .iter()
                    .map(|file| StoredPerFileMeasurement {
                        file_id: file.file_id.clone(),
                        reference_speaker_time: file.reference_speaker_time,
                        missed: file.missed,
                        false_alarm: file.false_alarm,
                        confusion: file.confusion,
                        der: file.der,
                        missed_percent: file.missed_percent,
                        false_alarm_percent: file.false_alarm_percent,
                        confusion_percent: file.confusion_percent,
                        reference_speaker_count: file.reference_speaker_count,
                        predicted_speaker_count: file.predicted_speaker_count,
                    })
                    .collect(),
            };
            let mut hypotheses = BTreeMap::new();
            let mut artifacts = Vec::with_capacity(files.len());
            for file in files {
                let relative = hypothesis_path(implementation_id, &file.file_id)?;
                let body = result.hypotheses.get(&file.file_id).ok_or_else(|| {
                    color_eyre::eyre::eyre!(
                        "completed {} result is missing hypothesis for {}",
                        implementation_id.as_str(),
                        file.file_id
                    )
                })?;
                artifacts.push(PendingArtifact {
                    relative_path: relative.clone(),
                    bytes: body.as_bytes().to_vec(),
                });
                hypotheses.insert(file.file_id.clone(), ArtifactLocation::available(relative));
            }
            (Some(measurement), hypotheses, artifacts)
        } else {
            (None, BTreeMap::new(), Vec::new())
        };
        Ok((
            StoredImplementationResult {
                status,
                reason: result.reason.clone(),
                measurement,
                hypotheses,
                source_metadata: BTreeMap::new(),
            },
            artifacts,
        ))
    }

    fn source_metadata(&self) -> BTreeMap<String, serde_json::Value> {
        let (segmentation, embedding) = self.pyannote_batch_sizes.summary_values();
        [(
            "pyannote_batch_sizes".to_owned(),
            serde_json::json!({
                "segmentation": segmentation,
                "embedding": embedding,
            }),
        )]
        .into_iter()
        .collect()
    }
}

fn completed_input_mismatch_reason<'a>(
    display_name: &str,
    result: &DerImplResult,
    expected_file_ids: impl IntoIterator<Item = &'a str>,
) -> Option<String> {
    if !matches!(result.status, DerImplStatus::Completed) {
        return None;
    }

    let expected: BTreeSet<&str> = expected_file_ids.into_iter().collect();
    let actual: BTreeSet<&str> = result.hypotheses.keys().map(String::as_str).collect();
    let mut mismatches = Vec::new();
    if result.files != expected.len() {
        mismatches.push(format!(
            "records {} files but the input manifest has {}",
            result.files,
            expected.len()
        ));
    }
    if actual.len() != expected.len() {
        mismatches.push(format!(
            "records {} hypotheses but the input manifest has {} files",
            actual.len(),
            expected.len()
        ));
    }
    if actual != expected {
        let missing = expected.difference(&actual).copied().collect::<Vec<_>>();
        let unexpected = actual.difference(&expected).copied().collect::<Vec<_>>();
        mismatches.push(format!(
            "hypothesis IDs do not match the input manifest (missing: {}; unexpected: {})",
            missing.join(", "),
            unexpected.join(", ")
        ));
    }

    (!mismatches.is_empty())
        .then(|| format!("completed {display_name} result {}", mismatches.join("; ")))
}

fn hypothesis_path(implementation_id: ImplementationId, file_id: &str) -> Result<PathBuf> {
    ensure!(
        !file_id.is_empty()
            && !file_id.contains('/')
            && !file_id.contains('\\')
            && file_id != "."
            && file_id != "..",
        "invalid benchmark file ID {file_id}"
    );
    Ok(PathBuf::from("hypotheses")
        .join(implementation_id.as_str())
        .join(format!("{file_id}.rttm")))
}

fn stable_input_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn validate_artifact_path(path: &Path) -> Result<()> {
    ensure!(
        !path.as_os_str().is_empty() && path.is_relative(),
        "benchmark artifact path must be relative: {}",
        path.display()
    );
    ensure!(
        !path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir)),
        "benchmark artifact path contains a relative path traversal: {}",
        path.display()
    );
    Ok(())
}

pub fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{seconds:.0}s")
    } else {
        let mins = (seconds / 60.0).floor() as u64;
        let secs = (seconds % 60.0).round() as u64;
        format!("{mins}m {secs:02}s")
    }
}

pub fn now_stamp() -> String {
    chrono::Local::now().format("%H:%M:%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ImplementationId;
    use crate::commands::benchmark::run_store::RunId;
    use crate::datasets::DatasetId;
    use hound::{SampleFormat, WavSpec, WavWriter};
    use std::fs;

    fn write_wav(path: &Path) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 1_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        writer.write_sample(0_i16).unwrap();
        writer.finalize().unwrap();
    }

    fn run_identity() -> RunIdentity {
        RunIdentity {
            run_id: RunId::new("test-run").unwrap(),
            started_at: "2026-01-01T00:00:00Z".to_owned(),
            host: "test-host".to_owned(),
            description: None,
            implementations: vec![ImplementationId::SpeakrsCpu, ImplementationId::SpeakerKit],
            datasets: vec![DatasetIdentity::catalog(DatasetId::VoxconverseDev)],
        }
    }

    fn metadata() -> BenchmarkMetadata {
        BenchmarkMetadata {
            git_sha: "test".to_owned(),
            gpu: "none".to_owned(),
            cpu: "test".to_owned(),
            region: "local".to_owned(),
        }
    }

    #[test]
    fn completed_result_with_wrong_hypothesis_ids_is_failed_before_storage() {
        let mut result =
            DerImplResult::completed(Some(0.0), Some(0.0), Some(0.0), Some(0.0), 1.0, 1);
        result
            .hypotheses
            .insert("unexpected".to_owned(), String::new());

        let reason =
            completed_input_mismatch_reason("SpeakerKit", &result, std::iter::once("expected"))
                .unwrap();

        assert!(reason.contains("missing: expected"));
        assert!(reason.contains("unexpected: unexpected"));
    }

    #[test]
    fn completed_result_with_missing_hypotheses_is_recorded_as_failed() {
        let temp_dir = tempfile::tempdir().unwrap();
        let run_dir = temp_dir.path().join("run");
        fs::create_dir(&run_dir).unwrap();
        let wav = temp_dir.path().join("sample.wav");
        let rttm = temp_dir.path().join("sample.rttm");
        write_wav(&wav);
        fs::write(&rttm, "SPEAKER sample 1 0 1 <NA> <NA> speaker <NA> <NA>\n").unwrap();
        let source_wav = fs::read(&wav).unwrap();
        let source_rttm = fs::read(&rttm).unwrap();
        let run_identity = run_identity();
        let files = vec![(wav.clone(), rttm.clone())];
        let implementations = vec![
            (ImplementationId::SpeakrsCpu, ImplType::Speakrs("cpu")),
            (ImplementationId::SpeakerKit, ImplType::SpeakerKitBench),
        ];
        let mut results = HashMap::from([(
            ImplementationId::SpeakrsCpu,
            DerImplResult {
                status: DerImplStatus::Completed,
                reason: None,
                der: Some(0.0),
                missed: Some(0.0),
                false_alarm: Some(0.0),
                confusion: Some(0.0),
                time: Some(1.0),
                files: 1,
                per_file: Vec::new(),
                hypotheses: HashMap::from([(
                    "sample".to_owned(),
                    "SPEAKER sample 1 0 1 <NA> <NA> speaker <NA> <NA>\n".to_owned(),
                )]),
            },
        )]);
        results.insert(
            ImplementationId::SpeakerKit,
            DerImplResult::completed(Some(0.0), Some(0.0), Some(0.0), Some(0.0), 1.0, 1),
        );
        let metadata = metadata();
        let writer = DerResultsWriter {
            run_dir: &run_dir,
            run_identity: &run_identity,
            dataset: DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            implementations: &implementations,
            results: &results,
            files: &files,
            total_audio_minutes: 1.0 / 60.0,
            collar: 0.0,
            description: None,
            max_files: 1,
            max_minutes: 1,
            metadata: &metadata,
            pyannote_batch_sizes: PyannoteBatchSizes::default(),
        };
        writer.write().unwrap();

        assert!(run_dir.join("results.json").is_file());
        assert!(run_dir.join("results.txt").is_file());
        assert!(run_dir.join("hypotheses/cpu/sample.rttm").is_file());
        assert!(!run_dir.join("hypotheses/speakerkit").exists());
        let record = BenchmarkRecord::read(&run_dir.join("results.json")).unwrap();
        let speakerkit = &record.implementations[&ImplementationId::SpeakerKit];
        assert_eq!(speakerkit.status, StoredStatus::Failed);
        assert!(speakerkit.reason.as_deref().unwrap().contains("hypotheses"));
        assert_eq!(fs::read(&wav).unwrap(), source_wav);
        assert_eq!(fs::read(&rttm).unwrap(), source_rttm);
    }

    #[test]
    fn publication_rolls_back_after_a_partial_publish() {
        let temp_dir = tempfile::tempdir().unwrap();
        let run_dir = temp_dir.path().join("run");
        fs::create_dir(&run_dir).unwrap();
        fs::create_dir(run_dir.join("hypotheses")).unwrap();
        fs::write(run_dir.join("hypotheses/speakerkit"), "blocker").unwrap();

        let staging = temp_dir.path().join("staging");
        fs::create_dir(&staging).unwrap();
        let cpu_staged = staging.join("cpu.rttm");
        let speakerkit_staged = staging.join("speakerkit.rttm");
        fs::write(&cpu_staged, "cpu").unwrap();
        fs::write(&speakerkit_staged, "speakerkit").unwrap();

        let cpu_destination = run_dir.join("hypotheses/cpu/sample.rttm");
        let speakerkit_destination = run_dir.join("hypotheses/speakerkit/sample.rttm");
        let mut publication = Publication::new();
        publication.publish(&cpu_staged, &cpu_destination).unwrap();
        assert!(
            publication
                .publish(&speakerkit_staged, &speakerkit_destination)
                .is_err()
        );
        drop(publication);

        assert!(!cpu_destination.exists());
        assert!(run_dir.join("hypotheses/speakerkit").is_file());
    }
}
