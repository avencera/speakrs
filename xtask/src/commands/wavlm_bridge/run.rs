use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use color_eyre::eyre::{Context, Result, ensure};
use serde::Serialize;

use speakrs::imported_segmentation::{SegmentationBundle, load_imported_segmentation_bundle};
use speakrs::inference::EmbeddingArtifactMetadata;
use speakrs::pipeline::{EmbeddingAvailability, PipelineGeometry};

use super::cache::{self, CacheHit};
use super::domain::{
    ArtifactRef, ArtifactState, AvailabilityCounts, AvailabilityReason, BridgeMode, BridgeSpec,
    GeometryReceipt, RecipeSpec, RunDocument, RunRecording, RuntimeIdentity, Sha256Digest,
    SpeakerTrack, SpeakerTracks, StageKind, SystemKind, SystemManifest, SystemRecord, TimedSegment,
    ValidationDocument, canonical_json_digest, digest_bytes, digest_file, digest_tree,
    ensure_regular_file, make_tree_read_only,
};
use super::embedding_execution::{load_embedding_model, run_embedding_stage};
use crate::wav::load_wav_samples;

use super::stage::{
    ExecutionIdentity, availability_counts_from_snapshot, clustering_dependencies,
    embedding_snapshot_bytes, embedding_stage_key, embedding_stage_receipt_files, make_receipt,
    receipt_ref, receipt_with_output, reconstruction_dependencies, run_from_embedding_cache,
    stage_name, stage_receipts,
};

pub struct ValidatedRecording {
    pub spec: super::domain::RecordingSpec,
    pub samples: Vec<f32>,
    pub bundle: SegmentationBundle,
}

pub fn validate_bundle_audio(bundle_path: &Path, audio_path: &Path) -> Result<ValidationDocument> {
    ensure_regular_file(audio_path, "audio")?;
    let (samples, sample_rate) = load_wav_samples(&audio_path.to_string_lossy())?;
    ensure!(
        sample_rate == 16_000,
        "expected 16 kHz WAV, got {sample_rate} Hz"
    );
    let bundle = load_imported_segmentation_bundle(bundle_path)
        .wrap_err_with(|| format!("failed to validate bundle {}", bundle_path.display()))?;
    validate_waveform_and_bundle(&bundle, &samples, sample_rate, None)?;
    let manifest = bundle.manifest();
    let geometry = geometry_receipt(&PipelineGeometry::from_imported(
        &manifest.audio,
        &manifest.geometry,
    )?);
    Ok(ValidationDocument {
        schema_version: super::domain::VALIDATION_SCHEMA_VERSION,
        bundle_path: bundle_path.to_owned(),
        bundle_id: manifest.identity.bundle_id.clone(),
        manifest_sha256: manifest.identity.manifest_digest.clone(),
        audio_path: audio_path.to_owned(),
        sample_rate,
        sample_count: samples.len(),
        waveform_sha256: speakrs::canonical_waveform_digest(&samples),
        geometry,
    })
}

pub fn validate_recording(recording: &super::domain::RecordingSpec) -> Result<ValidatedRecording> {
    ensure_regular_file(&recording.audio.path, "audio")?;
    ensure_regular_file(&recording.reference.path, "reference")?;
    ensure_regular_file(&recording.uem.path, "UEM")?;
    let (samples, sample_rate) = load_wav_samples(&recording.audio.path.to_string_lossy())?;
    ensure!(
        sample_rate == 16_000,
        "recording {} is not 16 kHz",
        recording.id
    );
    let bundle = load_imported_segmentation_bundle(&recording.bundle.path)
        .wrap_err_with(|| format!("failed to validate bundle for recording {}", recording.id))?;
    validate_waveform_and_bundle(&bundle, &samples, sample_rate, Some(recording))?;
    ensure!(
        digest_file(&recording.reference.path)? == recording.reference.sha256,
        "reference digest mismatch for recording {}",
        recording.id
    );
    ensure!(
        digest_file(&recording.uem.path)? == recording.uem.sha256,
        "UEM digest mismatch for recording {}",
        recording.id
    );
    Ok(ValidatedRecording {
        spec: recording.clone(),
        samples,
        bundle,
    })
}

fn validate_waveform_and_bundle(
    bundle: &SegmentationBundle,
    samples: &[f32],
    sample_rate: u32,
    recording: Option<&super::domain::RecordingSpec>,
) -> Result<()> {
    let manifest = bundle.manifest();
    ensure!(
        manifest.audio.sample_rate == sample_rate,
        "bundle sample rate {} does not match audio sample rate {sample_rate}",
        manifest.audio.sample_rate
    );
    ensure!(
        manifest.audio.channels == 1,
        "bundle audio contract is not mono"
    );
    ensure!(
        manifest.audio.sample_count == samples.len() as u64,
        "bundle sample count {} does not match audio sample count {}",
        manifest.audio.sample_count,
        samples.len()
    );
    let waveform = speakrs::canonical_waveform_digest(samples);
    ensure!(
        waveform == manifest.audio.waveform_sha256,
        "bundle waveform digest does not match audio"
    );
    if let Some(recording) = recording {
        ensure!(
            waveform == recording.audio.sha256,
            "audio digest mismatch for recording {}",
            recording.id
        );
        ensure!(
            manifest.identity.bundle_id == recording.bundle.bundle_id,
            "bundle id mismatch for recording {}",
            recording.id
        );
        ensure!(
            manifest.identity.manifest_digest == recording.bundle.manifest_sha256,
            "bundle manifest digest mismatch for recording {}",
            recording.id
        );
    }
    Ok(())
}

pub struct RunOptions {
    pub spec_path: PathBuf,
    pub models_dir: PathBuf,
    pub mode: BridgeMode,
    pub output_dir: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub recipe_ids: Vec<String>,
}

pub fn run(options: RunOptions) -> Result<()> {
    let spec_bytes = fs::read(&options.spec_path)
        .wrap_err_with(|| format!("failed to read bridge spec {}", options.spec_path.display()))?;
    let spec: BridgeSpec = serde_json::from_slice(&spec_bytes)
        .wrap_err_with(|| format!("invalid bridge spec {}", options.spec_path.display()))?;
    spec.validate()?;
    let spec = resolve_spec_paths(spec, options.spec_path.parent());
    ensure!(
        spec.runtime.mode == options.mode,
        "requested mode {:?} does not match spec runtime mode {:?}",
        options.mode,
        spec.runtime.mode
    );
    let execution_mode = options.mode.to_execution_mode()?;
    ensure!(
        !options.output_dir.exists(),
        "output directory already exists: {}",
        options.output_dir.display()
    );
    if let Some(cache_dir) = &options.cache_dir
        && cache_dir.exists()
    {
        super::domain::ensure_directory(cache_dir, "cache directory")?;
    }
    let model_digest = digest_tree(&options.models_dir)?;
    ensure!(
        model_digest == spec.runtime.models_sha256,
        "models directory digest does not match spec runtime identity"
    );
    let (embedding_path, plda_dir) = validate_model_assets(&options.models_dir, &spec.runtime)?;

    let recipes = select_recipes(&spec, &options.recipe_ids)?;
    let mut embedding_model = load_embedding_model(embedding_path, execution_mode)?;
    let execution_identity = ExecutionIdentity::capture()?;
    let mut run_context = RunContext {
        embedding_model: &mut embedding_model,
        plda_dir: &plda_dir,
        mode: execution_mode,
        runtime: &spec.runtime,
        execution: &execution_identity,
    };
    let staging = create_output_staging(&options.output_dir)?;
    let recipe_ids = recipes
        .iter()
        .map(|recipe| recipe.id.clone())
        .collect::<Vec<_>>();
    let mut run_records_by_recipe = recipes
        .iter()
        .map(|_| Vec::with_capacity(spec.recordings.len()))
        .collect::<Vec<Vec<_>>>();
    let mut system_records_by_recipe = recipes
        .iter()
        .map(|_| Vec::with_capacity(spec.recordings.len()))
        .collect::<Vec<Vec<_>>>();
    for recording_spec in &spec.recordings {
        let recording = validate_recording(recording_spec)?;
        for (recipe_index, recipe) in recipes.iter().enumerate() {
            let cache_key = cache_key(
                &spec,
                &recording.spec,
                recipe,
                &model_digest,
                &execution_identity,
            )?;
            let cache_hit = options
                .cache_dir
                .as_deref()
                .map(|root| cache::lookup(root, &cache_key, &recording.spec.id, &recipe.id))
                .transpose()?
                .flatten();
            let (outputs, runtime_seconds, embedding_cache_reused) = match cache_hit {
                Some(hit) => (
                    materialize_cache_hit(staging.path(), recipe, &recording, hit)?,
                    None,
                    false,
                ),
                None => {
                    let geometry = geometry_receipt(&PipelineGeometry::from_imported(
                        &recording.bundle.manifest().audio,
                        &recording.bundle.manifest().geometry,
                    )?);
                    let embedding_key =
                        embedding_stage_key(recipe, &recording, &run_context, &geometry)?;
                    let embedding_hit = options
                        .cache_dir
                        .as_deref()
                        .map(|root| {
                            cache::lookup_embedding(root, &embedding_key, &recording.spec.id)
                        })
                        .transpose()?
                        .flatten();
                    let started = Instant::now();
                    let (outputs, embedding_reused) = match embedding_hit {
                        Some(hit) => (
                            run_from_embedding_cache(
                                recipe,
                                &recording,
                                &mut run_context,
                                &cache_key,
                                &geometry,
                                hit,
                            )?,
                            true,
                        ),
                        None => (
                            run_one(recipe, &recording, &mut run_context, &cache_key)?,
                            false,
                        ),
                    };
                    let runtime = started.elapsed().as_secs_f64();
                    if let Some(cache_dir) = &options.cache_dir {
                        let stage_files = outputs
                            .receipt_files
                            .iter()
                            .map(|(name, bytes)| (name.clone(), bytes.clone()))
                            .collect::<Vec<_>>();
                        let publication = cache::CachePublication {
                            stage_receipts: &stage_files,
                            embedding_snapshot: &outputs.embedding_snapshot_bytes,
                            speaker_tracks: &outputs.speaker_tracks_bytes,
                            hypothesis: &outputs.hypothesis_bytes,
                        };
                        cache::publish(
                            cache_dir,
                            &cache_key,
                            &recording.spec.id,
                            &recipe.id,
                            &publication,
                        )?;
                        if !embedding_reused {
                            cache::publish_embedding(
                                cache_dir,
                                &embedding_key,
                                &recording.spec.id,
                                &outputs.embedding_stage_receipt_files,
                                &outputs.embedding_snapshot_bytes,
                            )?;
                        }
                    }
                    (
                        materialize_outputs(staging.path(), recipe, &recording, &outputs)?,
                        Some(runtime),
                        embedding_reused,
                    )
                }
            };
            let hypothesis = outputs.hypothesis.clone();
            let speaker_tracks = outputs.speaker_tracks.clone();
            let stage_receipts = outputs.stage_receipts.clone();
            let cache_reused = runtime_seconds.is_none();
            run_records_by_recipe[recipe_index].push(RunRecording {
                recording_id: recording.spec.id.clone(),
                recipe_id: recipe.id.clone(),
                cache_key: cache_key.clone(),
                cache_reused,
                embedding_cache_reused,
                stage_receipts,
                hypothesis: hypothesis.clone(),
                speaker_tracks: speaker_tracks.clone(),
            });
            system_records_by_recipe[recipe_index].push(SystemRecord {
                recording_id: recording.spec.id.clone(),
                source: recording.spec.source.clone(),
                domain: recording.spec.domain.clone(),
                parent_group: recording.spec.parent_group.clone(),
                audio_sha256: recording.spec.audio.sha256.clone(),
                reference_sha256: recording.spec.reference.sha256.clone(),
                uem_sha256: recording.spec.uem.sha256.clone(),
                hypothesis: ArtifactState::Available {
                    artifact: hypothesis,
                },
                speaker_tracks: ArtifactState::Available {
                    artifact: speaker_tracks,
                },
                runtime_seconds: runtime_seconds
                    .map(|value| super::domain::MeasurementState::Available { value })
                    .unwrap_or(super::domain::MeasurementState::Unavailable {
                        reason: AvailabilityReason::NotMeasured,
                    }),
                peak_memory_bytes: super::domain::MeasurementState::Unavailable {
                    reason: AvailabilityReason::NotMeasured,
                },
            });
        }
    }
    for (recipe_index, recipe) in recipes.iter().enumerate() {
        let system = SystemManifest {
            schema_version: super::domain::SYSTEM_SCHEMA_VERSION,
            experiment_id: spec.experiment_id.clone(),
            system: SystemKind::HybridWavlmSpeakrs,
            join: spec.join.clone(),
            recipe_id: recipe.id.clone(),
            records: std::mem::take(&mut system_records_by_recipe[recipe_index]),
            score_document: super::domain::ScoreDocumentState::Unavailable {
                reason: AvailabilityReason::NotCalculated,
            },
        };
        let system_relative = PathBuf::from("systems")
            .join("hybrid_wavlm_speakrs")
            .join(format!("{}.json", recipe.id));
        write_new(
            staging.path().join(&system_relative),
            &serde_json::to_vec_pretty(&system)?,
        )?;
    }
    let run_records = run_records_by_recipe.into_iter().flatten().collect();
    let run_document = RunDocument {
        schema_version: super::domain::RUN_SCHEMA_VERSION,
        experiment_id: spec.experiment_id.clone(),
        spec_sha256: digest_bytes(&spec_bytes),
        runtime: spec.runtime.clone(),
        recipes: recipe_ids,
        records: run_records,
    };
    write_new(
        staging.path().join("run.json"),
        &serde_json::to_vec_pretty(&run_document)?,
    )?;
    publish_output(staging, &options.output_dir)?;
    println!("wrote wavlm-bridge run {}", options.output_dir.display());
    Ok(())
}

fn select_recipes<'a>(spec: &'a BridgeSpec, requested: &[String]) -> Result<Vec<&'a RecipeSpec>> {
    let mut ids = BTreeSet::new();
    if requested.is_empty() {
        return Ok(spec.recipes.iter().collect());
    }
    let mut selected = Vec::with_capacity(requested.len());
    for id in requested {
        ensure!(ids.insert(id), "duplicate --recipe {id}");
        selected.push(spec.recipe(id)?);
    }
    Ok(selected)
}

fn resolve_spec_paths(mut spec: BridgeSpec, base: Option<&Path>) -> BridgeSpec {
    let Some(base) = base else {
        return spec;
    };
    for recording in &mut spec.recordings {
        recording.audio.path = resolve_path(base, &recording.audio.path);
        recording.bundle.path = resolve_path(base, &recording.bundle.path);
        recording.reference.path = resolve_path(base, &recording.reference.path);
        recording.uem.path = resolve_path(base, &recording.uem.path);
    }
    spec
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

fn validate_model_assets(
    models_dir: &Path,
    runtime: &RuntimeIdentity,
) -> Result<(PathBuf, PathBuf)> {
    super::domain::ensure_directory(models_dir, "models directory")?;
    let embedding_path = models_dir.join(&runtime.embedding_model.path);
    ensure_regular_file(&embedding_path, "embedding model")?;
    ensure!(
        digest_file(&embedding_path)? == runtime.embedding_model.sha256,
        "embedding model digest does not match runtime identity"
    );
    let sidecar_path = EmbeddingArtifactMetadata::path_for(&embedding_path);
    ensure_regular_file(&sidecar_path, "embedding model sidecar")?;
    ensure!(
        digest_file(&sidecar_path)? == runtime.embedding_model.sidecar_sha256,
        "embedding model sidecar digest does not match runtime identity"
    );
    EmbeddingArtifactMetadata::read_for(&embedding_path)
        .wrap_err_with(|| format!("invalid embedding sidecar {}", sidecar_path.display()))?;

    let plda_dir = models_dir.join(&runtime.plda.path);
    super::domain::ensure_directory(&plda_dir, "PLDA directory")?;
    ensure!(
        digest_tree(&plda_dir)? == runtime.plda.sha256,
        "PLDA directory digest does not match runtime identity"
    );
    Ok((embedding_path, plda_dir))
}

pub(crate) struct PreparedOutputs {
    pub(crate) cache_key: Sha256Digest,
    pub(crate) receipt_files: Vec<(String, Vec<u8>)>,
    pub(crate) embedding_stage_receipt_files: Vec<(String, Vec<u8>)>,
    pub(crate) embedding_snapshot_bytes: Vec<u8>,
    pub(crate) speaker_tracks_bytes: Vec<u8>,
    pub(crate) hypothesis_bytes: Vec<u8>,
    pub(crate) stage_receipts: Vec<ArtifactRef>,
    pub(crate) speaker_tracks: ArtifactRef,
    pub(crate) hypothesis: ArtifactRef,
}

pub(crate) struct RunContext<'a> {
    pub(crate) embedding_model: &'a mut speakrs::inference::EmbeddingModel,
    pub(crate) plda_dir: &'a Path,
    pub(crate) mode: speakrs::ExecutionMode,
    pub(crate) runtime: &'a RuntimeIdentity,
    pub(crate) execution: &'a ExecutionIdentity,
}

fn run_one(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    context: &mut RunContext<'_>,
    expected_cache_key: &Sha256Digest,
) -> Result<PreparedOutputs> {
    let geometry = geometry_receipt(&PipelineGeometry::from_imported(
        &recording.bundle.manifest().audio,
        &recording.bundle.manifest().geometry,
    )?);
    let runtime = context.runtime.clone();
    let (pipeline, snapshot) =
        run_embedding_stage(recipe, recording, context.embedding_model, context.plda_dir)?;
    let config = recipe.pipeline_config(context.mode)?;
    let availability = availability_counts_from_snapshot(&snapshot);
    let mut stage_receipts = stage_receipts(
        recipe,
        recording,
        &runtime,
        context.execution,
        geometry.clone(),
        availability,
    )?;
    let snapshot_bytes = embedding_snapshot_bytes(
        recording,
        &stage_receipts.embedding.receipt.cache_key,
        &snapshot,
        &geometry,
    )?;
    let snapshot_relative_path = recording_relative(recipe, recording)
        .join("stages")
        .join("embedding_snapshot.json");
    stage_receipts.embedding = receipt_with_output(
        &stage_receipts.embedding,
        artifact_for_path(&snapshot_relative_path, &snapshot_bytes),
    )?;
    let result = pipeline.finish_embedding_stage(snapshot, &config)?;
    let tracks = speaker_tracks(recording, &result, geometry.clone());
    let tracks_bytes = serde_json::to_vec_pretty(&tracks)?;
    let hypothesis_bytes = result.rttm(&recording.spec.id).into_bytes();
    let output_relative = recording_relative(recipe, recording);
    let reconstruction_outputs = vec![
        artifact_for_path(&output_relative.join("speaker_tracks.json"), &tracks_bytes),
        artifact_for_path(&output_relative.join("output.rttm"), &hypothesis_bytes),
    ];
    let clustering_dependencies = clustering_dependencies(recipe, context.execution)?;
    let reconstruction_dependencies =
        reconstruction_dependencies(recipe, &geometry, context.execution)?;
    let clustering_receipt = make_receipt(
        StageKind::Clustering,
        clustering_dependencies,
        vec![receipt_ref(&stage_receipts.embedding)],
        geometry.clone(),
        availability,
    );
    let mut reconstruction_receipt = make_receipt(
        StageKind::Reconstruction,
        reconstruction_dependencies,
        vec![receipt_ref(&clustering_receipt)],
        geometry,
        availability,
    );
    reconstruction_receipt.receipt.outputs = reconstruction_outputs;
    reconstruction_receipt.receipt.cache_key = expected_cache_key.clone();
    reconstruction_receipt.receipt_sha256 = canonical_json_digest(&reconstruction_receipt.receipt)?;
    let receipt_documents = [
        stage_receipts.bundle.clone(),
        stage_receipts.decode.clone(),
        stage_receipts.embedding.clone(),
        clustering_receipt,
        reconstruction_receipt,
    ];
    let receipt_files = receipt_documents
        .iter()
        .map(|document| {
            Ok((
                format!("{}.receipt.json", stage_name(document.receipt.stage)),
                serde_json::to_vec_pretty(document)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let embedding_stage_receipt_files =
        embedding_stage_receipt_files(&stage_receipts, &snapshot_bytes)?;
    Ok(PreparedOutputs {
        cache_key: expected_cache_key.clone(),
        receipt_files,
        embedding_stage_receipt_files,
        embedding_snapshot_bytes: snapshot_bytes,
        speaker_tracks_bytes: tracks_bytes,
        hypothesis_bytes,
        stage_receipts: Vec::new(),
        speaker_tracks: ArtifactRef {
            relative_path: PathBuf::new(),
            sha256: digest_bytes(b""),
            bytes: 0,
        },
        hypothesis: ArtifactRef {
            relative_path: PathBuf::new(),
            sha256: digest_bytes(b""),
            bytes: 0,
        },
    })
}

fn materialize_outputs(
    run_root: &Path,
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    outputs: &PreparedOutputs,
) -> Result<PreparedOutputs> {
    let relative = recording_relative(recipe, recording);
    let root = run_root.join(&relative);
    fs::create_dir_all(root.join("stages"))?;
    let mut stage_receipts = Vec::new();
    for (name, bytes) in &outputs.receipt_files {
        let path = root.join("stages").join(name);
        write_new(&path, bytes)?;
        stage_receipts.push(artifact_for_run(
            run_root,
            &relative.join("stages").join(name),
        )?);
    }
    write_new(
        root.join("stages").join("embedding_snapshot.json"),
        &outputs.embedding_snapshot_bytes,
    )?;
    let tracks_path = root.join("speaker_tracks.json");
    write_new(&tracks_path, &outputs.speaker_tracks_bytes)?;
    let hypothesis_path = root.join("output.rttm");
    write_new(&hypothesis_path, &outputs.hypothesis_bytes)?;
    Ok(PreparedOutputs {
        cache_key: outputs.cache_key.clone(),
        receipt_files: outputs.receipt_files.clone(),
        embedding_stage_receipt_files: outputs.embedding_stage_receipt_files.clone(),
        embedding_snapshot_bytes: outputs.embedding_snapshot_bytes.clone(),
        speaker_tracks_bytes: outputs.speaker_tracks_bytes.clone(),
        hypothesis_bytes: outputs.hypothesis_bytes.clone(),
        stage_receipts,
        speaker_tracks: artifact_for_run(run_root, &relative.join("speaker_tracks.json"))?,
        hypothesis: artifact_for_run(run_root, &relative.join("output.rttm"))?,
    })
}

fn materialize_cache_hit(
    run_root: &Path,
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    hit: CacheHit,
) -> Result<PreparedOutputs> {
    let relative = recording_relative(recipe, recording);
    let root = run_root.join(&relative);
    fs::create_dir_all(root.join("stages"))?;
    let mut stage_receipts = Vec::new();
    for (cache_relative, bytes, _) in &hit.stage_receipts {
        let name = cache_relative
            .file_name()
            .ok_or_else(|| color_eyre::eyre::eyre!("cache receipt has no file name"))?;
        let relative_path = relative.join("stages").join(name);
        write_new(run_root.join(&relative_path), bytes)?;
        stage_receipts.push(artifact_for_run(run_root, &relative_path)?);
    }
    write_new(
        root.join("stages").join("embedding_snapshot.json"),
        &hit.embedding_snapshot.0,
    )?;
    write_new(
        run_root.join(relative.join("speaker_tracks.json")),
        &hit.speaker_tracks.0,
    )?;
    write_new(
        run_root.join(relative.join("output.rttm")),
        &hit.hypothesis.0,
    )?;
    Ok(PreparedOutputs {
        cache_key: hit.key,
        receipt_files: Vec::new(),
        embedding_stage_receipt_files: Vec::new(),
        embedding_snapshot_bytes: hit.embedding_snapshot.0,
        speaker_tracks_bytes: hit.speaker_tracks.0,
        hypothesis_bytes: hit.hypothesis.0,
        stage_receipts,
        speaker_tracks: artifact_for_run(run_root, &relative.join("speaker_tracks.json"))?,
        hypothesis: artifact_for_run(run_root, &relative.join("output.rttm"))?,
    })
}

pub(crate) fn recording_relative(recipe: &RecipeSpec, recording: &ValidatedRecording) -> PathBuf {
    PathBuf::from("recipes")
        .join(&recipe.id)
        .join("recordings")
        .join(&recording.spec.id)
}

fn cache_key(
    spec: &BridgeSpec,
    recording: &super::domain::RecordingSpec,
    recipe: &RecipeSpec,
    model_digest: &Sha256Digest,
    execution: &ExecutionIdentity,
) -> Result<Sha256Digest> {
    // cache stores unscored inference artifacts
    // reference, UEM, and scorer identities belong to report admission
    #[derive(Serialize)]
    struct Key<'a> {
        recording: &'a str,
        audio: &'a Sha256Digest,
        bundle: &'a Sha256Digest,
        bundle_manifest: &'a Sha256Digest,
        recipe: &'a RecipeSpec,
        model: &'a Sha256Digest,
        runtime: &'a RuntimeIdentity,
        execution: &'a ExecutionIdentity,
    }
    canonical_json_digest(&Key {
        recording: &recording.id,
        audio: &recording.audio.sha256,
        bundle: &recording.bundle.bundle_id,
        bundle_manifest: &recording.bundle.manifest_sha256,
        recipe,
        model: model_digest,
        runtime: &spec.runtime,
        execution,
    })
}

pub(crate) fn availability_counts(result: &speakrs::DiarizationResult) -> AvailabilityCounts {
    let mut counts = AvailabilityCounts {
        chunks: result.embedding_availability.chunks(),
        local_slots: result.embedding_availability.speakers(),
        ..AvailabilityCounts::default()
    };
    for state in result.embedding_availability.iter() {
        match state {
            EmbeddingAvailability::Available => counts.available += 1,
            EmbeddingAvailability::Inactive { .. } => counts.inactive += 1,
            EmbeddingAvailability::InferenceFailed { .. } => counts.inference_failed += 1,
        }
    }
    let receipt = result.embedding_receipt;
    counts.clean_mask = receipt.clean_mask_count;
    counts.full_mask_fallback = receipt.full_mask_fallback_count;
    counts
}

pub(crate) fn speaker_tracks(
    recording: &ValidatedRecording,
    result: &speakrs::DiarizationResult,
    geometry: GeometryReceipt,
) -> SpeakerTracks {
    let mut tracks = Vec::new();
    let speakers = result
        .segments
        .iter()
        .map(|segment| segment.speaker.clone())
        .collect::<BTreeSet<_>>();
    for speaker_id in speakers {
        let segments = result
            .segments
            .iter()
            .filter(|segment| segment.speaker == speaker_id)
            .map(|segment| TimedSegment {
                start_seconds: segment.start,
                duration_seconds: segment.duration(),
            })
            .collect::<Vec<_>>();
        let total_seconds = segments
            .iter()
            .map(|segment| segment.duration_seconds)
            .sum();
        tracks.push(SpeakerTrack {
            speaker_id,
            segments,
            total_seconds,
        });
    }
    SpeakerTracks {
        schema_version: super::domain::SYSTEM_SCHEMA_VERSION,
        recording_id: recording.spec.id.clone(),
        geometry,
        tracks,
    }
}

fn geometry_receipt(geometry: &PipelineGeometry) -> GeometryReceipt {
    let frame_grid =
        |grid: &speakrs::imported_segmentation::FrameGrid| super::domain::FrameGridReceipt {
            frame_count: grid.frame_count,
            origin: super::domain::RationalReceipt {
                numerator: grid.origin.numerator,
                denominator: grid.origin.denominator,
            },
            step: super::domain::RationalReceipt {
                numerator: grid.step.numerator,
                denominator: grid.step.denominator,
            },
            support: super::domain::RationalReceipt {
                numerator: grid.support.numerator,
                denominator: grid.support.denominator,
            },
        };
    GeometryReceipt {
        sample_rate: geometry.sample_rate(),
        sample_count: geometry.sample_count(),
        window_samples: geometry.window_samples(),
        step_samples: geometry.step_samples(),
        chunks: geometry
            .chunks()
            .iter()
            .map(|chunk| super::domain::ChunkExtentReceipt {
                index: chunk.index(),
                start_samples: chunk.start_samples(),
                valid_samples: chunk.valid_samples(),
                padding_samples: chunk.padding_samples(),
            })
            .collect(),
        frame_grid: frame_grid(geometry.frame_grid()),
        aggregate_grid: frame_grid(geometry.aggregate_grid()),
        start_frames: geometry.start_frames().to_vec(),
        output_frames: geometry.output_frames(),
        output_extent_start_samples: geometry.output_extent().start_samples,
        output_extent_end_samples: geometry.output_extent().end_samples,
        output_extent_policy: geometry.output_extent_policy(),
    }
}

fn artifact_for_run(root: &Path, relative: &Path) -> Result<ArtifactRef> {
    let bytes = fs::read(root.join(relative))?;
    Ok(ArtifactRef {
        relative_path: relative.to_owned(),
        sha256: digest_bytes(&bytes),
        bytes: bytes.len() as u64,
    })
}

pub(crate) fn artifact_for_bytes(relative_path: &str, bytes: &[u8]) -> ArtifactRef {
    artifact_for_path(Path::new(relative_path), bytes)
}

pub(crate) fn artifact_for_path(relative_path: &Path, bytes: &[u8]) -> ArtifactRef {
    ArtifactRef {
        relative_path: relative_path.to_owned(),
        sha256: digest_bytes(bytes),
        bytes: bytes.len() as u64,
    }
}

fn create_output_staging(output: &Path) -> Result<tempfile::TempDir> {
    ensure!(
        !output.exists(),
        "output directory already exists: {}",
        output.display()
    );
    let parent = output
        .parent()
        .ok_or_else(|| color_eyre::eyre::eyre!("output directory has no parent"))?;
    fs::create_dir_all(parent)?;
    tempfile::tempdir_in(parent).wrap_err_with(|| {
        format!(
            "failed to create output staging directory in {}",
            parent.display()
        )
    })
}

fn publish_output(staging: tempfile::TempDir, output: &Path) -> Result<()> {
    fs::create_dir(output).wrap_err_with(|| {
        format!(
            "failed to reserve output directory {} without replacement",
            output.display()
        )
    })?;
    publish_staged_tree(staging.path(), output)?;
    let run_bytes = fs::read(output.join("run.json"))?;
    write_new(
        output.join(".complete"),
        format!("{}\n", digest_bytes(&run_bytes)).as_bytes(),
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
            "output staging tree contains a symlink: {}",
            source.display()
        );
        if metadata.is_dir() {
            fs::create_dir(&target).wrap_err_with(|| {
                format!(
                    "failed to create output directory {} without replacement",
                    target.display()
                )
            })?;
            publish_staged_tree(&source, &target)?;
        } else if metadata.is_file() {
            write_new(&target, &fs::read(&source)?)?;
        } else {
            ensure!(
                false,
                "unsupported output staging member: {}",
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
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .wrap_err_with(|| format!("failed to create immutable file {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::wavlm_bridge::domain::StageDependencyName;
    use crate::commands::wavlm_bridge::domain::{EmbeddingModelIdentity, ModelIdentity, Precision};
    use crate::commands::wavlm_bridge::stage::dependency;

    #[test]
    fn validates_explicit_fixed_embedding_asset_and_sidecar() {
        let models = tempfile::tempdir().unwrap();
        let embedding_path = models.path().join("wespeaker-voxceleb-resnet34-fixed.onnx");
        let embedding_bytes = b"fixed embedding placeholder";
        fs::write(&embedding_path, embedding_bytes).unwrap();
        let geometry =
            speakrs::inference::EmbeddingInputGeometry::new(16_000, 128_000, 399, 256).unwrap();
        let metadata =
            EmbeddingArtifactMetadata::new(geometry, digest_bytes(embedding_bytes), 100, 4_000)
                .unwrap();
        let sidecar_path = EmbeddingArtifactMetadata::path_for(&embedding_path);
        fs::write(&sidecar_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        let plda_dir = models.path().join("plda");
        fs::create_dir(&plda_dir).unwrap();
        let runtime = RuntimeIdentity {
            mode: BridgeMode::Cpu,
            models_sha256: digest_tree(models.path()).unwrap(),
            embedding_model: EmbeddingModelIdentity {
                id: "wespeaker-voxceleb-resnet34-fixed".into(),
                revision: "b2-fixed".into(),
                path: PathBuf::from("wespeaker-voxceleb-resnet34-fixed.onnx"),
                sha256: digest_bytes(embedding_bytes),
                sidecar_sha256: digest_file(&sidecar_path).unwrap(),
            },
            plda: ModelIdentity {
                id: "plda".into(),
                revision: "b2-fixed".into(),
                path: PathBuf::from("plda"),
                sha256: digest_tree(&plda_dir).unwrap(),
            },
            precision: Precision::Float32,
        };
        let (actual_embedding, actual_plda) = validate_model_assets(models.path(), &runtime)
            .expect("typed fixed assets should validate without model loading");
        assert_eq!(actual_embedding, embedding_path);
        assert_eq!(actual_plda, plda_dir);
    }

    #[test]
    fn output_publication_never_replaces_existing_directory() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("run");
        fs::create_dir(&output).unwrap();
        fs::write(output.join("sentinel"), b"keep").unwrap();
        let staging = tempfile::tempdir_in(root.path()).unwrap();
        fs::write(staging.path().join("run.json"), b"new").unwrap();

        assert!(publish_output(staging, &output).is_err());
        assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"keep");
        assert!(!output.join(".complete").exists());
    }

    #[test]
    fn interrupted_publication_has_no_completion_marker() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("run");
        let staging = tempfile::tempdir_in(root.path()).unwrap();
        fs::write(staging.path().join("partial"), b"partial").unwrap();

        assert!(publish_output(staging, &output).is_err());
        assert!(output.exists());
        assert!(!output.join(".complete").exists());
    }

    #[test]
    fn embedding_receipt_key_ignores_post_embedding_availability() {
        let geometry = GeometryReceipt {
            sample_rate: 16_000,
            sample_count: 0,
            window_samples: 1,
            step_samples: 1,
            chunks: Vec::new(),
            frame_grid: super::super::domain::FrameGridReceipt {
                frame_count: 0,
                origin: super::super::domain::RationalReceipt {
                    numerator: 0,
                    denominator: 1,
                },
                step: super::super::domain::RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
                support: super::super::domain::RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
            },
            aggregate_grid: super::super::domain::FrameGridReceipt {
                frame_count: 0,
                origin: super::super::domain::RationalReceipt {
                    numerator: 0,
                    denominator: 1,
                },
                step: super::super::domain::RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
                support: super::super::domain::RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
            },
            start_frames: Vec::new(),
            output_frames: 0,
            output_extent_start_samples: 0,
            output_extent_end_samples: 0,
            output_extent_policy: speakrs::imported_segmentation::OutputExtentPolicy::AggregateGrid,
        };
        let dependencies = vec![
            dependency(StageDependencyName::Embedding, "embedding@v1"),
            dependency(
                StageDependencyName::Implementation,
                "speakrs-imported-embedding-v1",
            ),
            dependency(StageDependencyName::Runtime, "\"cpu\""),
        ];
        let first = make_receipt(
            StageKind::Embedding,
            dependencies.clone(),
            Vec::new(),
            geometry.clone(),
            AvailabilityCounts::default(),
        );
        let second = make_receipt(
            StageKind::Embedding,
            dependencies,
            Vec::new(),
            geometry.clone(),
            AvailabilityCounts {
                chunks: 4,
                local_slots: 3,
                available: 8,
                inactive: 2,
                inference_failed: 2,
                clean_mask: 7,
                full_mask_fallback: 1,
            },
        );
        assert_eq!(first.receipt.cache_key, second.receipt.cache_key);
        assert_ne!(first.receipt_sha256, second.receipt_sha256);

        let first_values = receipt_with_output(
            &first,
            artifact_for_bytes("embedding_snapshot.json", b"first vectors"),
        )
        .unwrap();
        let second_values = receipt_with_output(
            &first,
            artifact_for_bytes("embedding_snapshot.json", b"second vectors"),
        )
        .unwrap();
        assert_eq!(
            first_values.receipt.cache_key,
            second_values.receipt.cache_key
        );
        assert_ne!(first_values.receipt_sha256, second_values.receipt_sha256);

        let cpu = make_receipt(
            StageKind::Embedding,
            vec![dependency(StageDependencyName::Runtime, "\"cpu\"")],
            Vec::new(),
            geometry.clone(),
            AvailabilityCounts::default(),
        );
        let cuda = make_receipt(
            StageKind::Embedding,
            vec![dependency(StageDependencyName::Runtime, "\"cuda\"")],
            Vec::new(),
            geometry,
            AvailabilityCounts::default(),
        );
        assert_ne!(cpu.receipt.cache_key, cuda.receipt.cache_key);
    }
}
