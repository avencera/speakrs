use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{Context, Result, ensure};

use super::domain::{
    BridgeSpec, JoinedRecord, JoinedSystemRecord, REPORT_SCHEMA_VERSION, ReportDocument,
    ReportMissing, ReportStatus, SYSTEM_SCHEMA_VERSION, ScoreDocumentState, SystemKind,
    SystemManifest, digest_bytes, ensure_regular_file, make_tree_read_only, validate_peak_memory,
    validate_runtime,
};

pub struct ReportOptions {
    pub spec_path: PathBuf,
    pub unchanged_speakrs: PathBuf,
    pub frozen_python: PathBuf,
    pub hybrid: PathBuf,
    pub output_dir: PathBuf,
}

pub fn run(options: ReportOptions) -> Result<()> {
    let spec = BridgeSpec::load(&options.spec_path)?;
    ensure!(
        !options.output_dir.exists(),
        "report output directory already exists: {}",
        options.output_dir.display()
    );
    let systems = vec![
        read_system(&options.unchanged_speakrs)?,
        read_system(&options.frozen_python)?,
        read_system(&options.hybrid)?,
    ];
    validate_systems(&spec, &systems)?;
    let records = join_records(&spec, &systems)?;
    let missing = systems
        .iter()
        .filter_map(|system| match system.score_document {
            ScoreDocumentState::Available { .. } => None,
            ScoreDocumentState::Unavailable { ref reason } => Some(ReportMissing {
                system: system.system,
                reason: reason.clone(),
            }),
        })
        .collect::<Vec<_>>();
    let status = if missing.is_empty() {
        ReportStatus::Complete
    } else {
        ReportStatus::Incomplete { missing }
    };
    let report = ReportDocument {
        schema_version: REPORT_SCHEMA_VERSION,
        experiment_id: spec.experiment_id,
        join: spec.join,
        systems,
        records,
        status,
    };
    let staging = create_output_staging(&options.output_dir)?;
    write_new(
        staging.path().join("report.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    write_new(
        staging.path().join("report.md"),
        render_markdown(&report).as_bytes(),
    )?;
    publish_output(staging, &options.output_dir)?;
    println!("wrote wavlm-bridge report {}", options.output_dir.display());
    Ok(())
}

fn read_system(path: &Path) -> Result<SystemManifest> {
    ensure_regular_file(path, "system manifest")?;
    let bytes = fs::read(path)
        .wrap_err_with(|| format!("failed to read system manifest {}", path.display()))?;
    let system: SystemManifest = serde_json::from_slice(&bytes)
        .wrap_err_with(|| format!("invalid system manifest {}", path.display()))?;
    ensure!(
        system.schema_version == SYSTEM_SCHEMA_VERSION,
        "unsupported system manifest schema {}",
        system.schema_version
    );
    Ok(system)
}

fn validate_systems(spec: &BridgeSpec, systems: &[SystemManifest]) -> Result<()> {
    ensure!(
        systems.len() == 3,
        "B3 report requires exactly three systems"
    );
    let expected = [
        SystemKind::UnchangedSpeakrs,
        SystemKind::FrozenPythonWavlm,
        SystemKind::HybridWavlmSpeakrs,
    ];
    let mut seen = BTreeSet::new();
    for system in systems {
        ensure!(
            system.experiment_id == spec.experiment_id,
            "system experiment identity mismatch"
        );
        ensure!(system.join == spec.join, "system join identity mismatch");
        ensure!(
            seen.insert(system.system),
            "duplicate system kind in report inputs"
        );
        ensure!(
            expected.contains(&system.system),
            "unsupported system kind in report input"
        );
        let score_available =
            matches!(&system.score_document, ScoreDocumentState::Available { .. });
        validate_system_records(spec, system, score_available)?;
        if let ScoreDocumentState::Available { document } = &system.score_document {
            ensure!(
                document.system == system.system,
                "score document system identity mismatch"
            );
            ensure!(
                document.join == system.join,
                "score document join identity mismatch"
            );
            document.validate_for(spec)?;
        }
    }
    for expected_kind in expected {
        ensure!(
            seen.contains(&expected_kind),
            "report is missing system {expected_kind:?}"
        );
    }
    Ok(())
}

fn validate_system_records(
    spec: &BridgeSpec,
    system: &SystemManifest,
    require_measurements: bool,
) -> Result<()> {
    let expected = spec
        .recordings
        .iter()
        .map(|recording| recording.id.as_str())
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    ensure!(
        system.records.len() == expected.len(),
        "system record count does not match membership"
    );
    for record in &system.records {
        ensure!(
            seen.insert(record.recording_id.as_str()),
            "duplicate system record {}",
            record.recording_id
        );
        ensure!(
            expected.contains(record.recording_id.as_str()),
            "system record is outside membership: {}",
            record.recording_id
        );
        let expected_record = spec
            .recordings
            .iter()
            .find(|recording| recording.id == record.recording_id)
            .expect("membership set was checked");
        ensure!(
            record.audio_sha256 == expected_record.audio.sha256,
            "audio identity mismatch for {}",
            record.recording_id
        );
        ensure!(
            record.reference_sha256 == expected_record.reference.sha256,
            "reference identity mismatch for {}",
            record.recording_id
        );
        ensure!(
            record.uem_sha256 == expected_record.uem.sha256,
            "UEM identity mismatch for {}",
            record.recording_id
        );
        ensure!(
            record.source == expected_record.source,
            "source identity mismatch for {}",
            record.recording_id
        );
        ensure!(
            record.domain == expected_record.domain,
            "domain identity mismatch for {}",
            record.recording_id
        );
        ensure!(
            record.parent_group == expected_record.parent_group,
            "parent identity mismatch for {}",
            record.recording_id
        );
        if require_measurements {
            validate_runtime(&record.runtime_seconds, &record.recording_id)?;
            validate_peak_memory(&record.peak_memory_bytes, &record.recording_id)?;
        }
    }
    ensure!(seen == expected, "system membership is incomplete");
    Ok(())
}

fn join_records(spec: &BridgeSpec, systems: &[SystemManifest]) -> Result<Vec<JoinedRecord>> {
    let mut by_system = BTreeMap::<SystemKind, &SystemManifest>::new();
    for system in systems {
        by_system.insert(system.system, system);
    }
    spec.recordings
        .iter()
        .map(|recording| {
            let mut joined = Vec::with_capacity(3);
            for kind in [
                SystemKind::UnchangedSpeakrs,
                SystemKind::FrozenPythonWavlm,
                SystemKind::HybridWavlmSpeakrs,
            ] {
                let system = by_system.get(&kind).expect("system set was validated");
                let record = system
                    .records
                    .iter()
                    .find(|record| record.recording_id == recording.id)
                    .expect("record set was validated");
                joined.push(JoinedSystemRecord {
                    system: kind,
                    hypothesis: record.hypothesis.clone(),
                    speaker_tracks: record.speaker_tracks.clone(),
                    runtime_seconds: record.runtime_seconds.clone(),
                    peak_memory_bytes: record.peak_memory_bytes.clone(),
                });
            }
            Ok(JoinedRecord {
                recording_id: recording.id.clone(),
                source: recording.source.clone(),
                domain: recording.domain.clone(),
                parent_group: recording.parent_group.clone(),
                systems: joined,
            })
        })
        .collect()
}

fn render_markdown(report: &ReportDocument) -> String {
    let status = match &report.status {
        ReportStatus::Complete => "complete",
        ReportStatus::Incomplete { .. } => "incomplete: score rows are unavailable",
    };
    let mut text = format!(
        "# WavLM bridge report\n\n- Experiment: `{}`\n- Status: **{}**\n- Recordings: {}\n\n",
        report.experiment_id,
        status,
        report.records.len()
    );
    text.push_str("This report joins the three systems by the fixed membership, reference, UEM, and scorer identities. It does not calculate scores.\n");
    if let ReportStatus::Incomplete { missing } = &report.status {
        text.push_str("\nUnavailable score documents:\n");
        for item in missing {
            text.push_str(&format!("- `{:?}`: {:?}\n", item.system, item.reason));
        }
    }
    text
}

fn create_output_staging(output: &Path) -> Result<tempfile::TempDir> {
    ensure!(
        !output.exists(),
        "report output directory already exists: {}",
        output.display()
    );
    let parent = output
        .parent()
        .ok_or_else(|| color_eyre::eyre::eyre!("report output has no parent"))?;
    fs::create_dir_all(parent)?;
    Ok(tempfile::tempdir_in(parent)?)
}

fn publish_output(staging: tempfile::TempDir, output: &Path) -> Result<()> {
    fs::create_dir(output).wrap_err_with(|| {
        format!(
            "failed to reserve report output {} without replacement",
            output.display()
        )
    })?;
    publish_staged_tree(staging.path(), output)?;
    let report_bytes = fs::read(output.join("report.json"))?;
    write_new(
        output.join(".complete"),
        format!("{}\n", digest_bytes(&report_bytes)).as_bytes(),
    )?;
    make_tree_read_only(output)?;
    Ok(())
}

fn publish_staged_tree(staging: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(staging)? {
        let entry = entry?;
        let source = entry.path();
        let target = destination.join(entry.file_name());
        let metadata = fs::symlink_metadata(&source)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "report staging tree contains a symlink: {}",
            source.display()
        );
        if metadata.is_dir() {
            fs::create_dir(&target).wrap_err_with(|| {
                format!(
                    "failed to create report output directory {} without replacement",
                    target.display()
                )
            })?;
            publish_staged_tree(&source, &target)?;
        } else if metadata.is_file() {
            write_new(&target, &fs::read(&source)?)?;
        } else {
            ensure!(
                false,
                "unsupported report staging member: {}",
                source.display()
            );
        }
    }
    Ok(())
}

fn write_new(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_manifest_rejects_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("system.json");
        fs::write(&path, br#"{"schema_version":1,"unexpected":true}"#).unwrap();
        assert!(read_system(&path).is_err());
    }

    #[test]
    fn report_publication_never_replaces_existing_directory() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("report");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("sentinel"), b"keep").unwrap();
        let staging = tempfile::tempdir_in(root.path()).unwrap();
        fs::write(staging.path().join("report.json"), b"new").unwrap();

        assert!(publish_output(staging, &output).is_err());
        assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
        assert!(!output.join(".complete").exists());
    }

    #[test]
    fn interrupted_report_publication_has_no_completion_marker() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("report");
        let staging = tempfile::tempdir_in(root.path()).unwrap();
        fs::write(staging.path().join("partial"), b"partial").unwrap();

        assert!(publish_output(staging, &output).is_err());
        assert!(output.exists());
        assert!(!output.join(".complete").exists());
    }
}
