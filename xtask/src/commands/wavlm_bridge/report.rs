use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use color_eyre::eyre::{Context, Result, bail, ensure};

use super::domain::{
    ArtifactRef, ArtifactState, BridgeSpec, JoinedRecord, JoinedSystemRecord,
    REPORT_SCHEMA_VERSION, ReportDocument, ReportMissing, ReportStatus, SYSTEM_SCHEMA_VERSION,
    ScoreDocument, ScoreDocumentState, SpeakerTracks, SystemKind, SystemManifest, digest_bytes,
    ensure_directory, ensure_regular_file, make_tree_read_only, validate_peak_memory,
    validate_runtime,
};

pub struct ReportOptions {
    pub spec_path: PathBuf,
    pub unchanged_speakrs: PathBuf,
    pub frozen_python: PathBuf,
    pub hybrid: PathBuf,
    pub output_dir: PathBuf,
}

struct LoadedSystem {
    manifest: SystemManifest,
    manifest_path: PathBuf,
}

pub fn run(options: ReportOptions) -> Result<()> {
    let spec = BridgeSpec::load(&options.spec_path)?;
    ensure!(
        !options.output_dir.exists(),
        "report output directory already exists: {}",
        options.output_dir.display()
    );
    let loaded_systems = vec![
        read_system(&options.unchanged_speakrs)?,
        read_system(&options.frozen_python)?,
        read_system(&options.hybrid)?,
    ];
    validate_systems(&spec, &loaded_systems)?;
    let records = join_records(&spec, &loaded_systems)?;
    let missing = loaded_systems
        .iter()
        .filter_map(|loaded| match &loaded.manifest.score_document {
            ScoreDocumentState::Available { .. } => None,
            ScoreDocumentState::Unavailable { reason } => Some(ReportMissing {
                system: loaded.manifest.system,
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
        systems: loaded_systems
            .into_iter()
            .map(|loaded| loaded.manifest)
            .collect(),
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

fn read_system(path: &Path) -> Result<LoadedSystem> {
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
    Ok(LoadedSystem {
        manifest: system,
        manifest_path: path.to_owned(),
    })
}

fn validate_systems(spec: &BridgeSpec, systems: &[LoadedSystem]) -> Result<()> {
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
    for loaded in systems {
        let system = &loaded.manifest;
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
        spec.recipe(&system.recipe_id)?;
        let score_available =
            matches!(&system.score_document, ScoreDocumentState::Available { .. });
        let artifact_root = score_available
            .then(|| artifact_root_from_manifest(&loaded.manifest_path, system))
            .transpose()?;
        validate_system_records(spec, system, artifact_root.as_deref(), score_available)?;
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
            validate_score_bindings(system, document)?;
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
    artifact_root: Option<&Path>,
    require_score_evidence: bool,
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
        if require_score_evidence {
            let artifact_root = artifact_root
                .ok_or_else(|| color_eyre::eyre::eyre!("score artifacts have no run root"))?;
            validate_hypothesis_artifact(artifact_root, &record.hypothesis, &record.recording_id)?;
            validate_speaker_tracks_artifact(
                artifact_root,
                &record.speaker_tracks,
                &record.recording_id,
            )?;
            validate_runtime(&record.runtime_seconds, &record.recording_id)?;
            validate_peak_memory(&record.peak_memory_bytes, &record.recording_id)?;
        }
    }
    ensure!(seen == expected, "system membership is incomplete");
    Ok(())
}

fn validate_hypothesis_artifact(
    artifact_root: &Path,
    state: &ArtifactState,
    recording_id: &str,
) -> Result<()> {
    let bytes = validate_record_artifact(artifact_root, state, "hypothesis", recording_id)?;
    let text = std::str::from_utf8(&bytes)
        .wrap_err_with(|| format!("hypothesis artifact is not UTF-8 for {recording_id}"))?;
    for (line_number, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let fields = line.split_whitespace().collect::<Vec<_>>();
        ensure!(
            fields.len() >= 8 && fields[0] == "SPEAKER",
            "invalid hypothesis RTTM line {} for {recording_id}",
            line_number + 1
        );
        ensure!(
            fields[1] == recording_id,
            "hypothesis recording identity mismatch on line {} for {recording_id}",
            line_number + 1
        );
        let start = fields[3].parse::<f64>().wrap_err_with(|| {
            format!(
                "invalid hypothesis start on line {} for {recording_id}",
                line_number + 1
            )
        })?;
        let duration = fields[4].parse::<f64>().wrap_err_with(|| {
            format!(
                "invalid hypothesis duration on line {} for {recording_id}",
                line_number + 1
            )
        })?;
        ensure!(
            start.is_finite() && start >= 0.0 && duration.is_finite() && duration >= 0.0,
            "invalid hypothesis interval on line {} for {recording_id}",
            line_number + 1
        );
        ensure!(!fields[7].is_empty(), "hypothesis speaker id is empty");
    }
    Ok(())
}

fn validate_score_bindings(system: &SystemManifest, document: &ScoreDocument) -> Result<()> {
    let records = system
        .records
        .iter()
        .map(|record| (record.recording_id.as_str(), record))
        .collect::<BTreeMap<_, _>>();
    for score in &document.per_record {
        ensure!(
            score.recipe_id.as_str() == system.recipe_id.as_str(),
            "score recipe identity mismatch for {}",
            score.recording_id
        );
        let record = records.get(score.recording_id.as_str()).ok_or_else(|| {
            color_eyre::eyre::eyre!(
                "score row has no matching system record {}",
                score.recording_id
            )
        })?;
        let hypothesis = match &record.hypothesis {
            ArtifactState::Available { artifact } => artifact,
            ArtifactState::Unavailable { .. } => {
                bail!("hypothesis is unavailable for {}", score.recording_id)
            }
        };
        ensure!(
            score.hypothesis_sha256 == hypothesis.sha256,
            "score hypothesis digest mismatch for {}",
            score.recording_id
        );
    }
    Ok(())
}

fn validate_record_artifact(
    artifact_root: &Path,
    state: &ArtifactState,
    label: &str,
    recording_id: &str,
) -> Result<Vec<u8>> {
    let artifact = match state {
        ArtifactState::Available { artifact } => artifact,
        ArtifactState::Unavailable { .. } => {
            bail!("{label} is unavailable for {recording_id}")
        }
    };
    read_verified_artifact(artifact_root, artifact, label, recording_id)
}

fn validate_speaker_tracks_artifact(
    artifact_root: &Path,
    state: &ArtifactState,
    recording_id: &str,
) -> Result<()> {
    let bytes = validate_record_artifact(artifact_root, state, "speaker tracks", recording_id)?;
    let tracks: SpeakerTracks = serde_json::from_slice(&bytes).wrap_err_with(|| {
        format!("invalid speaker tracks artifact for recording {recording_id}")
    })?;
    ensure!(
        tracks.schema_version == SYSTEM_SCHEMA_VERSION,
        "unsupported speaker tracks schema for {recording_id}"
    );
    ensure!(
        tracks.recording_id == recording_id,
        "speaker tracks recording identity mismatch for {recording_id}"
    );
    for track in tracks.tracks {
        ensure!(!track.speaker_id.is_empty(), "speaker track id is empty");
        ensure!(
            track.total_seconds.is_finite() && track.total_seconds >= 0.0,
            "speaker track has invalid total duration for {recording_id}"
        );
        for segment in track.segments {
            ensure!(
                segment.start_seconds.is_finite()
                    && segment.start_seconds >= 0.0
                    && segment.duration_seconds.is_finite()
                    && segment.duration_seconds > 0.0,
                "speaker track has invalid segment for {recording_id}"
            );
        }
    }
    Ok(())
}

fn read_verified_artifact(
    artifact_root: &Path,
    artifact: &ArtifactRef,
    label: &str,
    recording_id: &str,
) -> Result<Vec<u8>> {
    let path = safe_artifact_path(artifact_root, &artifact.relative_path, label)?;
    let bytes = fs::read(&path).wrap_err_with(|| {
        format!(
            "failed to read {label} artifact for recording {recording_id}: {}",
            path.display()
        )
    })?;
    ensure!(
        bytes.len() as u64 == artifact.bytes,
        "{label} artifact size mismatch for {recording_id}"
    );
    ensure!(
        digest_bytes(&bytes) == artifact.sha256,
        "{label} artifact digest mismatch for {recording_id}"
    );
    Ok(bytes)
}

fn safe_artifact_path(root: &Path, relative: &Path, label: &str) -> Result<PathBuf> {
    ensure_directory(root, "run artifact root")?;
    ensure!(
        !relative.as_os_str().is_empty() && !relative.is_absolute(),
        "{label} artifact path must be non-empty and relative"
    );
    let mut components = Vec::new();
    for component in relative.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(name) => components.push(name),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                bail!("{label} artifact path escapes its run root")
            }
        }
    }
    ensure!(!components.is_empty(), "{label} artifact path is empty");

    let mut current = root.to_owned();
    for (index, component) in components.iter().enumerate() {
        current.push(component);
        let metadata = fs::symlink_metadata(&current).wrap_err_with(|| {
            format!(
                "failed to inspect {label} artifact member {}",
                current.display()
            )
        })?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "{label} artifact contains a symlink: {}",
            current.display()
        );
        if index + 1 < components.len() {
            ensure!(
                metadata.is_dir(),
                "{label} artifact path contains a non-directory: {}",
                current.display()
            );
        }
    }
    ensure_regular_file(&current, label)?;
    Ok(current)
}

fn artifact_root_from_manifest(path: &Path, system: &SystemManifest) -> Result<PathBuf> {
    ensure_regular_file(path, "system manifest")?;
    ensure_output_artifacts_available(system)?;
    let mut matches = manifest_ancestors(path)
        .into_iter()
        .filter(|candidate| {
            let relative_manifest = if candidate == Path::new(".") {
                path
            } else {
                let Some(relative_manifest) = path.strip_prefix(candidate).ok() else {
                    return false;
                };
                relative_manifest
            };
            safe_artifact_path(candidate, relative_manifest, "system manifest").is_ok()
                && verify_system_artifacts(candidate, system).is_ok()
        })
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "expected exactly one run artifact root for system manifest {}, found {}",
        path.display(),
        matches.len()
    );
    Ok(matches.remove(0))
}

fn ensure_output_artifacts_available(system: &SystemManifest) -> Result<()> {
    for record in &system.records {
        ensure!(
            matches!(&record.hypothesis, ArtifactState::Available { .. }),
            "hypothesis is unavailable for {}",
            record.recording_id
        );
        ensure!(
            matches!(&record.speaker_tracks, ArtifactState::Available { .. }),
            "speaker tracks are unavailable for {}",
            record.recording_id
        );
    }
    Ok(())
}

fn verify_system_artifacts(root: &Path, system: &SystemManifest) -> Result<()> {
    ensure_directory(root, "run artifact root")?;
    for record in &system.records {
        validate_hypothesis_artifact(root, &record.hypothesis, &record.recording_id)?;
        validate_speaker_tracks_artifact(root, &record.speaker_tracks, &record.recording_id)?;
    }
    Ok(())
}

fn manifest_ancestors(path: &Path) -> Vec<PathBuf> {
    let mut ancestors = Vec::new();
    let mut current = path.parent().map(normalize_ancestor);
    while let Some(directory) = current {
        ancestors.push(directory.clone());
        let parent = directory.parent().map(normalize_ancestor);
        if parent.as_ref() == Some(&directory) {
            break;
        }
        current = parent;
    }
    ancestors
}

fn normalize_ancestor(path: &Path) -> PathBuf {
    if path.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        path.to_owned()
    }
}

fn join_records(spec: &BridgeSpec, systems: &[LoadedSystem]) -> Result<Vec<JoinedRecord>> {
    let mut by_system = BTreeMap::<SystemKind, &SystemManifest>::new();
    for loaded in systems {
        by_system.insert(loaded.manifest.system, &loaded.manifest);
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
    use super::super::domain::{
        ActivitySpec, AggregationIdentity, ArtifactInput, ArtifactState, AvailabilityReason,
        BridgeMode, BundleInput, ClusteringSpec, EmbeddingModelIdentity, JoinIdentity,
        MeasurementState, MembershipIdentity, MembershipRole, ModelIdentity, OverlapPolicy,
        Precision, QualificationIdentity, RecipeIdentity, RecipeSpec, ReconstructionMethod,
        ReconstructionSpec, RecordScore, RecordingSpec, RuntimeIdentity, SCORE_SCHEMA_VERSION,
        SPEC_SCHEMA_VERSION, ScorerIdentity, SpeakerCountPolicy, SystemRecord, VbxInitialization,
    };
    use super::*;

    fn distinct_recipe_spec() -> BridgeSpec {
        let membership_sha256 = digest_bytes(b"membership");
        let join = JoinIdentity {
            membership_sha256: membership_sha256.clone(),
            audio_manifest_sha256: digest_bytes(b"audio"),
            reference_manifest_sha256: digest_bytes(b"reference"),
            uem_manifest_sha256: digest_bytes(b"uem"),
            scorer: ScorerIdentity {
                implementation: "pyannote.metrics".into(),
                version: "4.0.0".into(),
                collar_seconds: 0.0,
                overlap: OverlapPolicy::Included,
                speaker_count: SpeakerCountPolicy::Automatic,
                config_sha256: digest_bytes(b"scorer"),
            },
            aggregation: AggregationIdentity {
                id: "aggregation".into(),
                revision: "v1".into(),
                equal_domain_average: true,
                pooled_secondary: true,
            },
            qualification: QualificationIdentity {
                id: "qualification".into(),
                revision: "v1".into(),
                sentinel_membership_sha256: None,
                monitor_membership_sha256: None,
                diagnostic_membership_sha256: None,
            },
        };
        BridgeSpec {
            schema_version: SPEC_SCHEMA_VERSION,
            experiment_id: "experiment".into(),
            membership: MembershipIdentity {
                id: "membership".into(),
                sha256: membership_sha256,
                role: MembershipRole::FixedProbe,
                recording_ids: vec!["recording".into()],
            },
            join,
            runtime: RuntimeIdentity {
                mode: BridgeMode::Cpu,
                models_sha256: digest_bytes(b"models"),
                embedding_model: EmbeddingModelIdentity {
                    id: "embedding".into(),
                    revision: "v1".into(),
                    path: "embedding.onnx".into(),
                    sha256: digest_bytes(b"embedding"),
                    sidecar_sha256: digest_bytes(b"sidecar"),
                },
                plda: ModelIdentity {
                    id: "plda".into(),
                    revision: "v1".into(),
                    path: "plda".into(),
                    sha256: digest_bytes(b"plda"),
                },
                precision: Precision::Float32,
            },
            recordings: vec![RecordingSpec {
                id: "recording".into(),
                source: "source".into(),
                domain: "domain".into(),
                parent_group: "parent".into(),
                audio: ArtifactInput {
                    path: "recording.wav".into(),
                    sha256: digest_bytes(b"recording-audio"),
                },
                bundle: BundleInput {
                    path: "recording-bundle".into(),
                    bundle_id: digest_bytes(b"recording-bundle"),
                    manifest_sha256: digest_bytes(b"recording-manifest"),
                },
                reference: ArtifactInput {
                    path: "recording.rttm".into(),
                    sha256: digest_bytes(b"recording-reference"),
                },
                uem: ArtifactInput {
                    path: "recording.uem".into(),
                    sha256: digest_bytes(b"recording-uem"),
                },
            }],
            recipes: ["control", "python", "hybrid"]
                .into_iter()
                .map(|id| RecipeSpec {
                    id: id.into(),
                    revision: "v1".into(),
                    seed: 1,
                    decoder: RecipeIdentity {
                        id: "decoder".into(),
                        revision: "v1".into(),
                    },
                    embedding: RecipeIdentity {
                        id: "embedding".into(),
                        revision: "v1".into(),
                    },
                    clustering: ClusteringSpec {
                        ahc_threshold: 0.5,
                        speaker_keep_threshold: 0.5,
                        vbx_fa: 1.0,
                        vbx_fb: 1.0,
                        vbx_max_iters: 1,
                        vbx_epsilon: 0.001,
                        vbx_initialization: VbxInitialization::Hard,
                        clean_frame_duration_seconds: 0.1,
                    },
                    reconstruction: ReconstructionSpec {
                        id: "reconstruction".into(),
                        revision: "v1".into(),
                        activity: ActivitySpec {
                            min_active_frames: 1,
                            max_inactive_gap_frames: 1,
                            pad_before_frames: 0,
                            pad_after_frames: 0,
                        },
                        merge_gap_seconds: 0.0,
                        method: ReconstructionMethod::Standard,
                    },
                })
                .collect(),
        }
    }

    fn unavailable_system(spec: &BridgeSpec, kind: SystemKind, recipe_id: &str) -> LoadedSystem {
        let recording = &spec.recordings[0];
        let unavailable_artifact = ArtifactState::Unavailable {
            reason: AvailabilityReason::NotCalculated,
        };
        let unavailable_runtime = MeasurementState::Unavailable {
            reason: AvailabilityReason::NotMeasured,
        };
        let unavailable_peak_memory = MeasurementState::Unavailable {
            reason: AvailabilityReason::NotMeasured,
        };
        LoadedSystem {
            manifest: SystemManifest {
                schema_version: SYSTEM_SCHEMA_VERSION,
                experiment_id: spec.experiment_id.clone(),
                system: kind,
                join: spec.join.clone(),
                recipe_id: recipe_id.into(),
                records: vec![SystemRecord {
                    recording_id: recording.id.clone(),
                    source: recording.source.clone(),
                    domain: recording.domain.clone(),
                    parent_group: recording.parent_group.clone(),
                    audio_sha256: recording.audio.sha256.clone(),
                    reference_sha256: recording.reference.sha256.clone(),
                    uem_sha256: recording.uem.sha256.clone(),
                    hypothesis: unavailable_artifact.clone(),
                    speaker_tracks: unavailable_artifact,
                    runtime_seconds: unavailable_runtime,
                    peak_memory_bytes: unavailable_peak_memory,
                }],
                score_document: ScoreDocumentState::Unavailable {
                    reason: AvailabilityReason::NotCalculated,
                },
            },
            manifest_path: PathBuf::from("system.json"),
        }
    }

    fn score_for(recipe_id: &str, system: SystemKind, spec: &BridgeSpec) -> ScoreDocument {
        ScoreDocument {
            schema_version: SCORE_SCHEMA_VERSION,
            system,
            join: spec.join.clone(),
            per_record: vec![RecordScore {
                recipe_id: recipe_id.into(),
                recording_id: "recording".into(),
                hypothesis_sha256: digest_bytes(b"hypothesis"),
                miss_seconds: 0.0,
                false_alarm_seconds: 0.0,
                confusion_seconds: 0.0,
                reference_speaker_seconds: 1.0,
                der: Some(0.0),
                jer: Some(0.0),
                reference_speaker_count: 1,
                predicted_speaker_count: 1,
                fragmentation: Some(0.0),
                short_speaker_retention: Some(1.0),
                mixed_mask_fallback: Some(0.0),
                runtime_seconds: MeasurementState::Unavailable {
                    reason: AvailabilityReason::NotMeasured,
                },
                peak_memory_bytes: MeasurementState::Unavailable {
                    reason: AvailabilityReason::NotMeasured,
                },
            }],
            source_rows: Vec::new(),
            domain_rows: Vec::new(),
            parent_rows: Vec::new(),
            hierarchical_equal_domain: None,
            pooled: None,
        }
    }

    #[test]
    fn system_manifest_rejects_unknown_fields() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("system.json");
        fs::write(&path, br#"{"schema_version":1,"unexpected":true}"#).unwrap();
        assert!(read_system(&path).is_err());
    }

    #[test]
    fn report_accepts_distinct_recipe_for_each_system() {
        let spec = distinct_recipe_spec();
        spec.validate().unwrap();
        let systems = vec![
            unavailable_system(&spec, SystemKind::UnchangedSpeakrs, "control"),
            unavailable_system(&spec, SystemKind::FrozenPythonWavlm, "python"),
            unavailable_system(&spec, SystemKind::HybridWavlmSpeakrs, "hybrid"),
        ];

        validate_systems(&spec, &systems).unwrap();
    }

    #[test]
    fn score_bindings_use_the_system_recipe() {
        let spec = distinct_recipe_spec();
        let systems = [
            (SystemKind::UnchangedSpeakrs, "control"),
            (SystemKind::FrozenPythonWavlm, "python"),
            (SystemKind::HybridWavlmSpeakrs, "hybrid"),
        ];

        for (kind, recipe_id) in systems {
            let mut loaded = unavailable_system(&spec, kind, recipe_id);
            loaded.manifest.records[0].hypothesis = ArtifactState::Available {
                artifact: ArtifactRef {
                    relative_path: PathBuf::from("hypothesis.rttm"),
                    sha256: digest_bytes(b"hypothesis"),
                    bytes: 10,
                },
            };
            let document = score_for(recipe_id, kind, &spec);

            validate_score_bindings(&loaded.manifest, &document).unwrap();
        }
    }

    #[test]
    fn artifact_root_candidates_follow_manifest_ancestors() {
        let path = Path::new("/tmp/run/systems/hybrid_wavlm_speakrs/reference.json");
        let ancestors = manifest_ancestors(path);

        assert_eq!(
            ancestors[0],
            PathBuf::from("/tmp/run/systems/hybrid_wavlm_speakrs")
        );
        assert!(ancestors.contains(&PathBuf::from("/tmp/run")));
    }

    #[test]
    fn artifact_verification_rejects_stale_bytes_and_path_escape() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("run");
        fs::create_dir_all(root.join("artifacts")).unwrap();
        fs::write(root.join("artifacts/output.rttm"), b"actual").unwrap();
        let artifact = ArtifactRef {
            relative_path: PathBuf::from("artifacts/output.rttm"),
            sha256: digest_bytes(b"expected"),
            bytes: 8,
        };
        assert!(read_verified_artifact(&root, &artifact, "hypothesis", "r1").is_err());

        let escaped = ArtifactRef {
            relative_path: PathBuf::from("../outside"),
            sha256: digest_bytes(b"outside"),
            bytes: 7,
        };
        assert!(read_verified_artifact(&root, &escaped, "hypothesis", "r1").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn artifact_verification_rejects_symlink_members() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("run");
        fs::create_dir_all(root.join("artifacts")).unwrap();
        fs::write(directory.path().join("outside"), b"outside").unwrap();
        symlink(
            directory.path().join("outside"),
            root.join("artifacts/output.rttm"),
        )
        .unwrap();
        let artifact = ArtifactRef {
            relative_path: PathBuf::from("artifacts/output.rttm"),
            sha256: digest_bytes(b"outside"),
            bytes: 7,
        };

        assert!(read_verified_artifact(&root, &artifact, "hypothesis", "r1").is_err());
    }

    #[test]
    fn artifact_reader_rejects_unavailable_state() {
        let root = tempfile::tempdir().unwrap();
        let state = ArtifactState::Unavailable {
            reason: AvailabilityReason::NotCalculated,
        };

        assert!(validate_record_artifact(root.path(), &state, "hypothesis", "r1").is_err());
    }

    #[test]
    fn score_schema_version_is_bumped() {
        assert_eq!(SCORE_SCHEMA_VERSION, 2);
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
