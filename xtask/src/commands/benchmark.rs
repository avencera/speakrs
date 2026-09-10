use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, ensure, eyre};

mod der;
mod jobs;
mod report;
mod run_store;
mod runner;
mod selection;
mod types;

pub use der::{DerArgs, der};
pub use run_store::{BenchmarkRun, convert_legacy_results};

pub fn score(run_dir: &Path) -> Result<()> {
    let records = load_score_records(run_dir)?;
    let mut dataset_scores = Vec::with_capacity(records.len());
    let mut aggregate_inputs: BTreeMap<
        crate::catalog::ImplementationId,
        Vec<run_store::ScoreResult>,
    > = BTreeMap::new();
    let mut run_id = None;
    let mut scoring = None;
    for loaded in records {
        let record = &loaded.record;
        record.scoring.validate_supported()?;
        if let Some(expected) = &run_id {
            ensure!(
                expected == &record.run.run_id,
                "stored benchmark records have different run IDs"
            );
        } else {
            run_id = Some(record.run.run_id.clone());
        }
        if let Some(expected) = &scoring {
            ensure!(
                expected == &record.scoring,
                "stored benchmark records have different scoring options"
            );
        } else {
            scoring = Some(record.scoring.clone());
        }
        let scores = score_record(&loaded.directory, record)?;
        for (implementation, result) in &scores {
            aggregate_inputs
                .entry(*implementation)
                .or_default()
                .push(result.clone());
        }
        dataset_scores.push(run_store::DatasetScore {
            dataset: record.dataset.clone(),
            implementations: scores,
        });
    }
    let report = run_store::ScoreReport {
        schema_version: run_store::SCHEMA_VERSION,
        run_id: run_id.ok_or_else(|| eyre!("stored benchmark run has no records"))?,
        scoring: scoring.ok_or_else(|| eyre!("stored benchmark run has no scoring options"))?,
        datasets: dataset_scores,
        implementations: aggregate_score_results(aggregate_inputs),
    };
    let mut payload = serde_json::to_vec_pretty(&report)?;
    payload.push(b'\n');
    run_store::atomic_replace_file(&run_dir.join(run_store::SCORE_FILE), &payload)?;
    let summary = format_score_summary(&report);
    println!(
        "Scored {} dataset record(s); updated {}/{} and printed:\n{}",
        report.datasets.len(),
        run_dir.display(),
        run_store::SCORE_FILE,
        summary
    );
    Ok(())
}

struct LoadedRecord {
    directory: PathBuf,
    record: run_store::BenchmarkRecord,
}

fn load_score_records(run_dir: &Path) -> Result<Vec<LoadedRecord>> {
    ensure!(
        run_dir.is_dir(),
        "benchmark run directory does not exist: {}",
        run_dir.display()
    );
    let root_results = run_dir.join("results.json");
    let mut paths = Vec::new();
    if root_results.is_file() {
        let root_record = run_store::BenchmarkRecord::read(&root_results)?;
        if root_record.run.datasets.len() == 1 {
            paths.push((run_dir.to_owned(), root_results));
        } else {
            let child_paths = child_result_paths(run_dir)?;
            ensure!(
                !child_paths.is_empty(),
                "multi-dataset benchmark run has no dataset result records"
            );
            paths.extend(child_paths);
        }
    } else {
        paths.extend(child_result_paths(run_dir)?);
    }
    ensure!(
        !paths.is_empty(),
        "schema version {} results.json not found in {} or its dataset directories",
        run_store::SCHEMA_VERSION,
        run_dir.display()
    );

    let suite_manifest = match fs::File::open(run_dir.join(run_store::RUN_MANIFEST_FILE)) {
        Ok(manifest_file) => {
            let manifest: run_store::BenchmarkSuiteManifest =
                serde_json::from_reader(manifest_file)?;
            ensure!(
                manifest.schema_version == run_store::SCHEMA_VERSION,
                "unsupported benchmark suite manifest schema version {}; expected {}",
                manifest.schema_version,
                run_store::SCHEMA_VERSION
            );
            manifest.run.validate()?;
            Some(manifest.run)
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };

    let mut records = Vec::with_capacity(paths.len());
    for (directory, path) in paths {
        records.push(LoadedRecord {
            directory,
            record: run_store::BenchmarkRecord::read(&path)?,
        });
    }
    let first = &records[0].record;
    if let Some(manifest) = suite_manifest {
        ensure!(
            manifest == first.run,
            "stored benchmark records do not match the suite manifest"
        );
    }
    ensure!(
        records.iter().all(|loaded| loaded.record.run == first.run),
        "stored benchmark records do not share one suite identity"
    );
    ensure!(
        records.iter().all(|loaded| {
            first
                .run
                .datasets
                .iter()
                .any(|dataset| dataset == &loaded.record.dataset)
        }),
        "stored benchmark record is not present in its suite manifest"
    );
    ensure!(
        first.run.datasets.len() == records.len(),
        "stored benchmark suite expects {} dataset records but found {}",
        first.run.datasets.len(),
        records.len()
    );
    for (index, record) in records.iter().enumerate() {
        ensure!(
            records[..index]
                .iter()
                .all(|previous| previous.record.dataset != record.record.dataset),
            "stored benchmark suite contains duplicate dataset {}",
            record.record.dataset.id_string()
        );
    }
    Ok(records)
}

fn child_result_paths(run_dir: &Path) -> Result<Vec<(PathBuf, PathBuf)>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(run_dir)? {
        let entry = entry?;
        let directory = entry.path();
        if directory.is_dir() {
            let results = directory.join("results.json");
            if results.is_file() {
                paths.push((directory, results));
            }
        }
    }
    paths.sort_by(|left, right| left.1.cmp(&right.1));
    Ok(paths)
}

fn score_record(
    record_dir: &Path,
    record: &run_store::BenchmarkRecord,
) -> Result<BTreeMap<crate::catalog::ImplementationId, run_store::ScoreResult>> {
    let mut scores = BTreeMap::new();
    for (implementation, result) in &record.implementations {
        let score = match result.status {
            run_store::StoredStatus::Skipped | run_store::StoredStatus::Failed => {
                run_store::ScoreResult {
                    status: result.status,
                    reason: result.reason.clone(),
                    measurement: None,
                }
            }
            run_store::StoredStatus::Completed => {
                let stored_measurement = result.measurement.as_ref().ok_or_else(|| {
                    eyre!(
                        "completed implementation {} has no measurement",
                        implementation.as_str()
                    )
                })?;
                let mut files = Vec::with_capacity(record.inputs.files.len());
                let mut hypotheses = HashMap::with_capacity(record.inputs.files.len());
                for input in &record.inputs.files {
                    let reference = input.reference_rttm.resolve(record_dir)?;
                    ensure!(
                        reference.is_file(),
                        "missing reference RTTM for {}: {}",
                        input.file_id,
                        reference.display()
                    );
                    validate_rttm(&reference, "reference", &input.file_id)?;
                    let hypothesis = result.hypotheses.get(&input.file_id).ok_or_else(|| {
                        eyre!(
                            "completed implementation {} is missing hypothesis for file {}",
                            implementation.as_str(),
                            input.file_id
                        )
                    })?;
                    let hypothesis_path = hypothesis.resolve(record_dir)?;
                    ensure!(
                        hypothesis_path.is_file(),
                        "missing hypothesis for {} ({}): {}",
                        implementation.as_str(),
                        input.file_id,
                        hypothesis_path.display()
                    );
                    let hypothesis_text = fs::read_to_string(&hypothesis_path)?;
                    validate_rttm_text(
                        &hypothesis_text,
                        &hypothesis_path,
                        "hypothesis",
                        &input.file_id,
                    )?;
                    files.push((PathBuf::from(format!("{}.wav", input.file_id)), reference));
                    hypotheses.insert(input.file_id.clone(), hypothesis_text);
                }
                let accumulation =
                    DerAccumulation::compute_with_options(&files, &hypotheses, &record.scoring)?;
                let (der, missed, false_alarm, confusion) = accumulation.der_percentages();
                let measurement = run_store::StoredMeasurement {
                    der,
                    missed,
                    false_alarm,
                    confusion,
                    reference_speaker_time: Some(accumulation.total_ref),
                    time_seconds: stored_measurement.time_seconds,
                    files: accumulation.file_count,
                    per_file: accumulation
                        .per_file()
                        .iter()
                        .map(|file| run_store::StoredPerFileMeasurement {
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
                run_store::ScoreResult {
                    status: run_store::StoredStatus::Completed,
                    reason: None,
                    measurement: Some(measurement),
                }
            }
        };
        scores.insert(*implementation, score);
    }
    Ok(scores)
}

fn aggregate_score_results(
    inputs: BTreeMap<crate::catalog::ImplementationId, Vec<run_store::ScoreResult>>,
) -> BTreeMap<crate::catalog::ImplementationId, run_store::ScoreResult> {
    inputs
        .into_iter()
        .map(|(implementation, results)| {
            let first_non_completed = results
                .iter()
                .find(|result| result.status != run_store::StoredStatus::Completed);
            if let Some(result) = first_non_completed {
                let status = if results
                    .iter()
                    .any(|result| result.status == run_store::StoredStatus::Failed)
                {
                    run_store::StoredStatus::Failed
                } else {
                    run_store::StoredStatus::Skipped
                };
                let reason = results
                    .iter()
                    .filter_map(|result| result.reason.as_deref())
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                return (
                    implementation,
                    run_store::ScoreResult {
                        status,
                        reason: if reason.is_empty() {
                            result.reason.clone()
                        } else {
                            Some(reason.join("; "))
                        },
                        measurement: None,
                    },
                );
            }
            let measurements: Vec<_> = results
                .iter()
                .filter_map(|result| result.measurement.as_ref())
                .collect();
            let measurement = aggregate_measurements(&measurements);
            (
                implementation,
                run_store::ScoreResult {
                    status: run_store::StoredStatus::Completed,
                    reason: None,
                    measurement: Some(measurement),
                },
            )
        })
        .collect()
}

fn aggregate_measurements(
    measurements: &[&run_store::StoredMeasurement],
) -> run_store::StoredMeasurement {
    let (missed, false_alarm, confusion) = measurements.iter().fold(
        (0.0, 0.0, 0.0),
        |(missed, false_alarm, confusion), measurement| {
            let (file_missed, file_false_alarm, file_confusion) =
                measurement_components(measurement);
            (
                missed + file_missed,
                false_alarm + file_false_alarm,
                confusion + file_confusion,
            )
        },
    );
    let reference_speaker_time: f64 = measurements
        .iter()
        .filter_map(|measurement| measurement.reference_speaker_time)
        .sum();
    let der = percentage(missed + false_alarm + confusion, reference_speaker_time);
    let per_file = measurements
        .iter()
        .flat_map(|measurement| measurement.per_file.iter().cloned())
        .collect::<Vec<_>>();
    run_store::StoredMeasurement {
        der,
        missed: percentage(missed, reference_speaker_time),
        false_alarm: percentage(false_alarm, reference_speaker_time),
        confusion: percentage(confusion, reference_speaker_time),
        reference_speaker_time: Some(reference_speaker_time),
        time_seconds: {
            let times = measurements
                .iter()
                .filter_map(|measurement| measurement.time_seconds)
                .collect::<Vec<_>>();
            (!times.is_empty()).then_some(times.into_iter().sum())
        },
        files: measurements
            .iter()
            .map(|measurement| measurement.files)
            .sum(),
        per_file,
    }
}

fn measurement_components(measurement: &run_store::StoredMeasurement) -> (f64, f64, f64) {
    if !measurement.per_file.is_empty() {
        return measurement.per_file.iter().fold(
            (0.0, 0.0, 0.0),
            |(missed, false_alarm, confusion), file| {
                (
                    missed + file.missed,
                    false_alarm + file.false_alarm,
                    confusion + file.confusion,
                )
            },
        );
    }
    let denominator = measurement.reference_speaker_time.unwrap_or(0.0);
    (
        measurement.missed.unwrap_or(0.0) * denominator / 100.0,
        measurement.false_alarm.unwrap_or(0.0) * denominator / 100.0,
        measurement.confusion.unwrap_or(0.0) * denominator / 100.0,
    )
}

fn percentage(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0).then_some(numerator / denominator * 100.0)
}

fn format_score_summary(report: &run_store::ScoreReport) -> String {
    let mut lines = vec![format!("Benchmark score {}", report.run_id.as_str())];
    for dataset in &report.datasets {
        lines.push(format!("{}:", dataset.dataset.display_name()));
        for (implementation, result) in &dataset.implementations {
            lines.push(format!(
                "  {}: {}",
                implementation.as_str(),
                format_score_result(result)
            ));
        }
    }
    lines.push("Aggregate:".to_owned());
    for (implementation, result) in &report.implementations {
        lines.push(format!(
            "  {}: {}",
            implementation.as_str(),
            format_score_result(result)
        ));
    }
    lines.join("\n")
}

fn validate_rttm(path: &Path, kind: &str, expected_file_id: &str) -> Result<()> {
    let text = fs::read_to_string(path)?;
    validate_rttm_text(&text, path, kind, expected_file_id)
}

fn validate_rttm_text(text: &str, path: &Path, kind: &str, expected_file_id: &str) -> Result<()> {
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        ensure!(
            fields.first() == Some(&"SPEAKER") && fields.len() >= 8,
            "invalid {kind} RTTM line {} in {}",
            line_number + 1,
            path.display()
        );
        ensure!(
            fields[1] == expected_file_id,
            "{kind} RTTM recording ID on line {} in {} is {}; expected {}",
            line_number + 1,
            path.display(),
            fields[1],
            expected_file_id
        );
        let start = fields[3].parse::<f64>().map_err(|error| {
            eyre!(
                "invalid {kind} RTTM start on line {} in {}: {error}",
                line_number + 1,
                path.display()
            )
        })?;
        let duration = fields[4].parse::<f64>().map_err(|error| {
            eyre!(
                "invalid {kind} RTTM duration on line {} in {}: {error}",
                line_number + 1,
                path.display()
            )
        })?;
        ensure!(
            start.is_finite() && start >= 0.0 && duration.is_finite() && duration >= 0.0,
            "invalid {kind} RTTM interval on line {} in {}",
            line_number + 1,
            path.display()
        );
    }
    Ok(())
}

fn format_score_result(result: &run_store::ScoreResult) -> String {
    match (&result.status, &result.measurement) {
        (run_store::StoredStatus::Completed, Some(measurement)) => measurement
            .der
            .map(|der| format!("completed ({der:.3}%)"))
            .unwrap_or_else(|| "completed (no reference speaker time)".to_owned()),
        (run_store::StoredStatus::Skipped, _) => format!(
            "skipped ({})",
            result.reason.as_deref().unwrap_or("no reason recorded")
        ),
        (run_store::StoredStatus::Failed, _) => format!(
            "failed ({})",
            result.reason.as_deref().unwrap_or("no reason recorded")
        ),
        (run_store::StoredStatus::Completed, None) => "completed (missing measurement)".to_owned(),
    }
}

pub use jobs::{
    BenchmarkJobConfig, BenchmarkJobResult, GpuBenchmarkSuiteConfig, ProgressUpdate, gpu_impls,
    run_benchmark_job, run_gpu_benchmark_suite, run_speakrs_gpu, validate_gpu_impls,
};
pub use report::{DerResultsWriter, format_eta, now_stamp};
pub(crate) use selection::discover_files;
pub(crate) use types::{BatchCommandRunner, PREFLIGHT_TIMEOUT, PyannoteRsFileRunner};
pub use types::{
    BenchmarkMetadata, DerAccumulation, DerImplResult, DerImplStatus, ImplType, PerFileDerResult,
    PyannoteBatchSizes,
};

#[cfg(test)]
mod score_tests {
    use super::score;
    use crate::catalog::ImplementationId;
    use crate::commands::benchmark::run_store::{
        ArtifactLocation, BenchmarkRecord, BenchmarkRun, DatasetIdentity, InputFile, InputManifest,
        ScoringOptions, SelectionOptions, StoredImplementationResult, StoredMeasurement,
        StoredMetadata, StoredStatus,
    };
    use crate::datasets::DatasetId;
    use std::collections::BTreeMap;
    use std::fs;

    fn record(
        run: &BenchmarkRun,
        dataset: DatasetIdentity,
        reference: &str,
        hypothesis: &str,
    ) -> BenchmarkRecord {
        let implementation = ImplementationId::SpeakrsCpu;
        BenchmarkRecord {
            schema_version: super::run_store::SCHEMA_VERSION,
            run: run.identity.clone(),
            dataset,
            inputs: InputManifest {
                files: vec![InputFile {
                    file_id: "sample".to_owned(),
                    reference_rttm: ArtifactLocation::available(reference),
                    audio: None,
                    duration_seconds: Some(10.0),
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
            total_audio_minutes: 10.0 / 60.0,
            implementations: [(
                implementation,
                StoredImplementationResult {
                    status: StoredStatus::Completed,
                    reason: None,
                    measurement: Some(StoredMeasurement {
                        der: Some(0.0),
                        missed: Some(0.0),
                        false_alarm: Some(0.0),
                        confusion: Some(0.0),
                        reference_speaker_time: Some(10.0),
                        time_seconds: Some(1.0),
                        files: 1,
                        per_file: Vec::new(),
                    }),
                    hypotheses: [("sample".to_owned(), ArtifactLocation::available(hypothesis))]
                        .into_iter()
                        .collect(),
                    source_metadata: BTreeMap::new(),
                },
            )]
            .into_iter()
            .collect(),
            source_metadata: BTreeMap::new(),
        }
    }

    fn record_with_files(
        run: &BenchmarkRun,
        dataset: DatasetIdentity,
        files: &[(&str, &str, &str)],
    ) -> BenchmarkRecord {
        let mut record = record(run, dataset, files[0].1, files[0].2);
        record.inputs.files = files
            .iter()
            .map(|(file_id, reference, _)| InputFile {
                file_id: (*file_id).to_owned(),
                reference_rttm: ArtifactLocation::available(*reference),
                audio: None,
                duration_seconds: Some(10.0),
            })
            .collect();
        let implementation = record
            .implementations
            .get_mut(&ImplementationId::SpeakrsCpu)
            .unwrap();
        implementation.measurement.as_mut().unwrap().files = files.len();
        implementation.hypotheses = files
            .iter()
            .map(|(file_id, _, hypothesis)| {
                (
                    (*file_id).to_owned(),
                    ArtifactLocation::available(*hypothesis),
                )
            })
            .collect();
        record
    }

    #[test]
    fn score_rejects_incomplete_schema_version_2_records() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("results.json"), r#"{"schema_version":1}"#).unwrap();
        assert!(score(dir.path()).is_err());
        std::fs::write(dir.path().join("results.json"), r#"{"schema_version":2}"#).unwrap();
        assert!(score(dir.path()).is_err());
    }

    #[test]
    fn score_recalculates_changed_hypothesis_and_replaces_output() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        fs::write(
            run.root.join("reference.rttm"),
            "SPEAKER sample 1 0 10 <NA> <NA> A <NA> <NA>\n",
        )
        .unwrap();
        fs::write(
            run.root.join("hypothesis.rttm"),
            "SPEAKER sample 1 0 5 <NA> <NA> X <NA> <NA>\n",
        )
        .unwrap();
        let initial = record(
            &run,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            "reference.rttm",
            "hypothesis.rttm",
        );
        initial.write_new(&run.root.join("results.json")).unwrap();
        score(&run.root).unwrap();
        let first: serde_json::Value =
            serde_json::from_slice(&fs::read(run.root.join("score.json")).unwrap()).unwrap();
        assert_eq!(
            first["implementations"]["cpu"]["measurement"]["der"],
            serde_json::json!(50.0)
        );
        fs::write(run.root.join("hypothesis.rttm"), "").unwrap();
        score(&run.root).unwrap();
        let second: serde_json::Value =
            serde_json::from_slice(&fs::read(run.root.join("score.json")).unwrap()).unwrap();
        assert_eq!(
            second["implementations"]["cpu"]["measurement"]["der"],
            serde_json::json!(100.0)
        );
        assert!(run.root.join("results.json").is_file());
    }

    #[test]
    fn score_rejects_swapped_reference_recording_ids() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        for (name, recording_id) in [("first", "second"), ("second", "first")] {
            fs::write(
                run.root.join(format!("{name}-reference.rttm")),
                format!("SPEAKER {recording_id} 1 0 10 <NA> <NA> A <NA> <NA>\n"),
            )
            .unwrap();
            fs::write(
                run.root.join(format!("{name}-hypothesis.rttm")),
                format!("SPEAKER {name} 1 0 10 <NA> <NA> A <NA> <NA>\n"),
            )
            .unwrap();
        }
        let stored = record_with_files(
            &run,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            &[
                ("first", "first-reference.rttm", "first-hypothesis.rttm"),
                ("second", "second-reference.rttm", "second-hypothesis.rttm"),
            ],
        );
        stored.write_new(&run.root.join("results.json")).unwrap();

        let error = score(&run.root).unwrap_err().to_string();
        assert!(error.contains("reference RTTM recording ID"), "{error}");
    }

    #[test]
    fn score_rejects_wrong_hypothesis_recording_id() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        fs::write(
            run.root.join("reference.rttm"),
            "SPEAKER sample 1 0 10 <NA> <NA> A <NA> <NA>\n",
        )
        .unwrap();
        fs::write(
            run.root.join("hypothesis.rttm"),
            "SPEAKER other 1 0 10 <NA> <NA> A <NA> <NA>\n",
        )
        .unwrap();
        let stored = record(
            &run,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            "reference.rttm",
            "hypothesis.rttm",
        );
        stored.write_new(&run.root.join("results.json")).unwrap();

        let error = score(&run.root).unwrap_err().to_string();
        assert!(error.contains("hypothesis RTTM recording ID"), "{error}");
    }

    #[test]
    fn score_accepts_empty_rttm_for_zero_reference_denominator() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        fs::write(run.root.join("reference.rttm"), "# no speech\n").unwrap();
        fs::write(run.root.join("hypothesis.rttm"), "\n").unwrap();
        let stored = record(
            &run,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            "reference.rttm",
            "hypothesis.rttm",
        );
        stored.write_new(&run.root.join("results.json")).unwrap();

        score(&run.root).unwrap();
    }

    #[test]
    fn score_aggregates_multi_dataset_suite_and_reports_failed_status() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu, ImplementationId::SpeakerKit],
            vec![DatasetId::VoxconverseDev, DatasetId::AmiIhm],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        for (dataset, directory, duration) in [
            (DatasetId::VoxconverseDev, "voxconverse-dev", 10.0),
            (DatasetId::AmiIhm, "ami-ihm", 5.0),
        ] {
            let child = run.root.join(directory);
            fs::create_dir_all(&child).unwrap();
            fs::write(
                child.join("reference.rttm"),
                format!("SPEAKER sample 1 0 {duration} <NA> <NA> A <NA> <NA>\n"),
            )
            .unwrap();
            fs::write(
                child.join("hypothesis.rttm"),
                format!("SPEAKER sample 1 0 {duration} <NA> <NA> X <NA> <NA>\n"),
            )
            .unwrap();
            let mut child_record = record(
                &run,
                DatasetIdentity::catalog(dataset),
                "reference.rttm",
                "hypothesis.rttm",
            );
            child_record.inputs.files[0].duration_seconds = Some(duration);
            child_record.total_audio_minutes = duration / 60.0;
            child_record.implementations.insert(
                ImplementationId::SpeakerKit,
                StoredImplementationResult {
                    status: StoredStatus::Failed,
                    reason: Some("timeout".to_owned()),
                    measurement: None,
                    hypotheses: BTreeMap::new(),
                    source_metadata: BTreeMap::new(),
                },
            );
            child_record.write_new(&child.join("results.json")).unwrap();
        }
        score(&run.root).unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(run.root.join("score.json")).unwrap()).unwrap();
        assert_eq!(report["datasets"].as_array().unwrap().len(), 2);
        assert_eq!(
            report["implementations"]["cpu"]["measurement"]["der"],
            serde_json::json!(0.0)
        );
        assert_eq!(report["implementations"]["speakerkit"]["status"], "failed");
    }

    #[test]
    fn score_rejects_unsupported_scoring_options() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            chrono::Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        let mut record = record(
            &run,
            DatasetIdentity::catalog(DatasetId::VoxconverseDev),
            "reference.rttm",
            "hypothesis.rttm",
        );
        record.scoring.collar_seconds = 0.1;
        record.write_new(&run.root.join("results.json")).unwrap();
        assert!(score(&run.root).is_err());
    }
}
