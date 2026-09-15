use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;

use serde::Serialize;

use speakrs::pipeline::{
    EmbeddingAvailability, EmbeddingFailureReason, EmbeddingReceipt,
    EmbeddingStageEntry as LibraryEmbeddingStageEntry, EmbeddingStageSnapshot,
    ImportedDiarizationPipeline, InactiveEmbeddingReason, PipelineGeometry,
};

use super::cache;
use super::domain::{
    ArtifactRef, AvailabilityCounts, EmbeddingFailureReason as DocumentEmbeddingFailureReason,
    EmbeddingInactiveReason, EmbeddingSlotAvailability, EmbeddingStageDocument,
    EmbeddingStageEntry as DocumentEmbeddingStageEntry, GeometryReceipt, ReceiptDocument,
    ReceiptRef, RecipeSpec, RuntimeIdentity, Sha256Digest, StageDependency, StageDependencyName,
    StageKind, StageReceipt, canonical_json_digest, digest_bytes, digest_file,
};
use super::embedding_execution::ensure_recipe_matches_bundle;
use super::run::{RunContext, ValidatedRecording};

#[derive(Clone)]
pub(crate) struct InferenceStageReceipts {
    pub(crate) bundle: ReceiptDocument,
    pub(crate) decode: ReceiptDocument,
    pub(crate) embedding: ReceiptDocument,
}

pub(crate) const EMBEDDING_IMPLEMENTATION_IDENTITY: &str =
    env!("WAVLM_EMBEDDING_IMPLEMENTATION_SHA256");

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ExecutionIdentity {
    executable_sha256: Sha256Digest,
    onnx_runtime_build: String,
    onnx_runtime_api: u32,
}

impl ExecutionIdentity {
    pub(crate) fn capture() -> color_eyre::eyre::Result<Self> {
        let executable = env::current_exe()?;
        Ok(Self {
            executable_sha256: digest_file(&executable)?,
            onnx_runtime_build: ort::info().to_owned(),
            onnx_runtime_api: ort::MINOR_VERSION,
        })
    }

    pub(crate) fn executable_sha256(&self) -> &Sha256Digest {
        &self.executable_sha256
    }
}

pub(crate) fn stage_receipts(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    runtime: &RuntimeIdentity,
    execution: &ExecutionIdentity,
    geometry: GeometryReceipt,
    availability: AvailabilityCounts,
) -> color_eyre::eyre::Result<InferenceStageReceipts> {
    let (bundle_dependencies, decode_dependencies, embedding_dependencies) =
        stage_dependencies(recipe, recording, runtime, execution, &geometry)?;
    let bundle = make_receipt(
        StageKind::Bundle,
        bundle_dependencies,
        Vec::new(),
        geometry.clone(),
        availability,
    );
    let decode = make_receipt(
        StageKind::Decode,
        decode_dependencies,
        vec![receipt_ref(&bundle)],
        geometry.clone(),
        availability,
    );
    let embedding = make_receipt(
        StageKind::Embedding,
        embedding_dependencies,
        vec![receipt_ref(&decode)],
        geometry,
        availability,
    );
    Ok(InferenceStageReceipts {
        bundle,
        decode,
        embedding,
    })
}

pub(crate) fn stage_dependencies(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    runtime: &RuntimeIdentity,
    execution: &ExecutionIdentity,
    geometry: &GeometryReceipt,
) -> color_eyre::eyre::Result<(
    Vec<StageDependency>,
    Vec<StageDependency>,
    Vec<StageDependency>,
)> {
    let geometry_digest = canonical_json_digest(geometry)?;
    Ok((
        vec![
            dependency(StageDependencyName::Recording, recording.spec.id.as_str()),
            dependency(
                StageDependencyName::Audio,
                recording.spec.audio.sha256.as_str(),
            ),
            dependency(
                StageDependencyName::Bundle,
                recording.spec.bundle.bundle_id.as_str(),
            ),
            dependency(StageDependencyName::Geometry, geometry_digest.as_ref()),
            dependency(
                StageDependencyName::Implementation,
                EMBEDDING_IMPLEMENTATION_IDENTITY,
            ),
        ],
        vec![
            dependency(
                StageDependencyName::Bundle,
                recording.spec.bundle.manifest_sha256.as_str(),
            ),
            dependency(
                StageDependencyName::Decoder,
                &recipe_identity(&recipe.decoder),
            ),
            dependency(
                StageDependencyName::Implementation,
                EMBEDDING_IMPLEMENTATION_IDENTITY,
            ),
        ],
        vec![
            dependency(
                StageDependencyName::Embedding,
                &recipe_identity(&recipe.embedding),
            ),
            dependency(
                StageDependencyName::Implementation,
                EMBEDDING_IMPLEMENTATION_IDENTITY,
            ),
            dependency(
                StageDependencyName::Model,
                &format!(
                    "{}:{}",
                    runtime.embedding_model.sha256, runtime.embedding_model.sidecar_sha256
                ),
            ),
            dependency(
                StageDependencyName::Runtime,
                &embedding_runtime_identity(runtime, execution),
            ),
            dependency(StageDependencyName::Geometry, geometry_digest.as_ref()),
        ],
    ))
}

fn embedding_runtime_identity(runtime: &RuntimeIdentity, execution: &ExecutionIdentity) -> String {
    #[derive(Serialize)]
    struct Identity<'a> {
        mode: super::domain::BridgeMode,
        onnx_runtime_build: &'a str,
        onnx_runtime_api: u32,
    }

    serde_json::to_string(&Identity {
        mode: runtime.mode,
        onnx_runtime_build: &execution.onnx_runtime_build,
        onnx_runtime_api: execution.onnx_runtime_api,
    })
    .expect("bridge runtime identity serialization cannot fail")
}

pub(crate) fn embedding_stage_key(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    context: &RunContext<'_>,
    geometry: &GeometryReceipt,
) -> color_eyre::eyre::Result<Sha256Digest> {
    let receipts = stage_receipts(
        recipe,
        recording,
        context.runtime,
        context.execution,
        geometry.clone(),
        AvailabilityCounts::default(),
    )?;
    Ok(receipts.embedding.receipt.cache_key)
}

pub(crate) fn embedding_snapshot_bytes(
    recording: &ValidatedRecording,
    stage_key: &Sha256Digest,
    snapshot: &EmbeddingStageSnapshot,
    geometry: &GeometryReceipt,
) -> color_eyre::eyre::Result<Vec<u8>> {
    let document = EmbeddingStageDocument {
        schema_version: super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
        recording_id: recording.spec.id.clone(),
        stage_key: stage_key.clone(),
        geometry: geometry.clone(),
        segmentation_shape: snapshot.segmentation_shape(),
        segmentation_values: snapshot.segmentation_values(),
        entries: snapshot
            .entries()
            .iter()
            .map(|entry| DocumentEmbeddingStageEntry {
                availability: document_availability(entry.availability()),
                values: entry.values().map(ToOwned::to_owned),
            })
            .collect(),
        embedding_receipt: availability_counts_from_snapshot(snapshot),
    };
    document.validate()?;
    let bytes = serde_json::to_vec_pretty(&document)?;
    color_eyre::eyre::ensure!(
        bytes.len() <= super::domain::MAX_EMBEDDING_STAGE_BYTES,
        "embedding snapshot exceeds size bound"
    );
    Ok(bytes)
}

pub(crate) fn embedding_stage_receipt_files(
    receipts: &InferenceStageReceipts,
    snapshot_bytes: &[u8],
) -> color_eyre::eyre::Result<Vec<(String, Vec<u8>)>> {
    let snapshot_artifact =
        super::run::artifact_for_bytes("embedding_snapshot.json", snapshot_bytes);
    let mut embedding = receipts.embedding.clone();
    embedding.receipt.outputs = vec![snapshot_artifact];
    embedding.receipt_sha256 = canonical_json_digest(&embedding.receipt)?;
    [receipts.bundle.clone(), receipts.decode.clone(), embedding]
        .iter()
        .map(|document| {
            Ok((
                format!("{}.receipt.json", stage_name(document.receipt.stage)),
                serde_json::to_vec_pretty(document)?,
            ))
        })
        .collect()
}

pub(crate) fn receipt_with_output(
    document: &ReceiptDocument,
    output: ArtifactRef,
) -> color_eyre::eyre::Result<ReceiptDocument> {
    let mut document = document.clone();
    document.receipt.outputs = vec![output];
    document.receipt_sha256 = canonical_json_digest(&document.receipt)?;
    Ok(document)
}

pub(crate) fn document_availability(
    availability: &EmbeddingAvailability,
) -> super::domain::EmbeddingSlotAvailability {
    match availability {
        EmbeddingAvailability::Available => EmbeddingSlotAvailability::Available,
        EmbeddingAvailability::Inactive { reason } => EmbeddingSlotAvailability::Inactive {
            reason: match reason {
                InactiveEmbeddingReason::NoActivity => EmbeddingInactiveReason::NoActivity,
            },
        },
        EmbeddingAvailability::InferenceFailed { reason } => {
            EmbeddingSlotAvailability::InferenceFailed {
                reason: match reason {
                    EmbeddingFailureReason::ModelExecution => {
                        DocumentEmbeddingFailureReason::ModelExecution
                    }
                    EmbeddingFailureReason::InvalidOutput => {
                        DocumentEmbeddingFailureReason::InvalidOutput
                    }
                    EmbeddingFailureReason::LegacyUnavailable => {
                        DocumentEmbeddingFailureReason::LegacyUnavailable
                    }
                },
            }
        }
    }
}

pub(crate) fn library_availability(
    availability: &EmbeddingSlotAvailability,
) -> color_eyre::eyre::Result<EmbeddingAvailability> {
    Ok(match availability {
        EmbeddingSlotAvailability::Available => EmbeddingAvailability::Available,
        EmbeddingSlotAvailability::Inactive { reason } => EmbeddingAvailability::Inactive {
            reason: match reason {
                EmbeddingInactiveReason::NoActivity => InactiveEmbeddingReason::NoActivity,
            },
        },
        EmbeddingSlotAvailability::InferenceFailed { reason } => {
            EmbeddingAvailability::InferenceFailed {
                reason: match reason {
                    DocumentEmbeddingFailureReason::ModelExecution => {
                        EmbeddingFailureReason::ModelExecution
                    }
                    DocumentEmbeddingFailureReason::InvalidOutput => {
                        EmbeddingFailureReason::InvalidOutput
                    }
                    DocumentEmbeddingFailureReason::LegacyUnavailable => {
                        EmbeddingFailureReason::LegacyUnavailable
                    }
                },
            }
        }
    })
}

pub(crate) fn availability_counts_from_snapshot(
    snapshot: &EmbeddingStageSnapshot,
) -> AvailabilityCounts {
    let shape = snapshot.segmentation_shape();
    let mut counts = AvailabilityCounts {
        chunks: shape[0],
        local_slots: shape[2],
        ..AvailabilityCounts::default()
    };
    for entry in snapshot.entries() {
        match entry.availability() {
            EmbeddingAvailability::Available => counts.available += 1,
            EmbeddingAvailability::Inactive { .. } => counts.inactive += 1,
            EmbeddingAvailability::InferenceFailed { .. } => counts.inference_failed += 1,
        }
    }
    let receipt = snapshot.embedding_receipt();
    counts.clean_mask = receipt.clean_mask_count;
    counts.full_mask_fallback = receipt.full_mask_fallback_count;
    counts
}

pub(crate) fn dependency(name: StageDependencyName, value: &str) -> StageDependency {
    StageDependency {
        name,
        value: value.to_owned(),
    }
}

pub(crate) fn clustering_dependencies(
    recipe: &RecipeSpec,
    execution: &ExecutionIdentity,
) -> color_eyre::eyre::Result<Vec<StageDependency>> {
    Ok(vec![
        dependency(
            StageDependencyName::Clustering,
            canonical_json_digest(&recipe.clustering)?.as_ref(),
        ),
        dependency(StageDependencyName::Seed, &recipe.seed.to_string()),
        dependency(
            StageDependencyName::Implementation,
            execution.executable_sha256().as_ref(),
        ),
    ])
}

pub(crate) fn reconstruction_dependencies(
    recipe: &RecipeSpec,
    geometry: &GeometryReceipt,
    execution: &ExecutionIdentity,
) -> color_eyre::eyre::Result<Vec<StageDependency>> {
    Ok(vec![
        dependency(
            StageDependencyName::Reconstruction,
            canonical_json_digest(&recipe.reconstruction)?.as_ref(),
        ),
        dependency(
            StageDependencyName::Geometry,
            canonical_json_digest(geometry)?.as_ref(),
        ),
        dependency(
            StageDependencyName::Implementation,
            execution.executable_sha256().as_ref(),
        ),
    ])
}

pub(crate) fn recipe_identity(identity: &super::domain::RecipeIdentity) -> String {
    format!("{}@{}", identity.id, identity.revision)
}

pub(crate) fn make_receipt(
    stage: StageKind,
    mut dependencies: Vec<StageDependency>,
    parents: Vec<ReceiptRef>,
    geometry: GeometryReceipt,
    availability: AvailabilityCounts,
) -> ReceiptDocument {
    dependencies.sort_by_key(|dependency| dependency.name);
    let cache_key = stage_dependency_key(&stage, &parents, &dependencies, &geometry);
    let receipt = StageReceipt {
        schema_version: super::domain::RECEIPT_SCHEMA_VERSION,
        stage,
        cache_key,
        parents,
        dependencies,
        geometry,
        availability,
        outputs: Vec::new(),
    };
    let receipt_sha256 =
        canonical_json_digest(&receipt).expect("receipt serialization cannot fail");
    ReceiptDocument {
        receipt,
        receipt_sha256,
    }
}

fn stage_dependency_key(
    stage: &StageKind,
    parents: &[ReceiptRef],
    dependencies: &[StageDependency],
    geometry: &GeometryReceipt,
) -> Sha256Digest {
    let parent_identity = parents
        .iter()
        .map(|parent| (parent.stage, parent.cache_key.clone()))
        .collect::<Vec<_>>();
    let key_material = (stage, &parent_identity, dependencies, geometry);
    canonical_json_digest(&key_material).expect("receipt key serialization cannot fail")
}

pub(crate) fn receipt_ref(document: &ReceiptDocument) -> ReceiptRef {
    ReceiptRef {
        stage: document.receipt.stage,
        cache_key: document.receipt.cache_key.clone(),
        receipt_sha256: document.receipt_sha256.clone(),
    }
}

pub(crate) fn stage_name(stage: StageKind) -> &'static str {
    match stage {
        StageKind::Bundle => "bundle",
        StageKind::Decode => "decode",
        StageKind::Embedding => "embedding",
        StageKind::Clustering => "clustering",
        StageKind::Reconstruction => "reconstruction",
    }
}

pub(crate) fn ensure_stage_dependency_identity(
    actual: &InferenceStageReceipts,
    expected: &InferenceStageReceipts,
) -> color_eyre::eyre::Result<()> {
    for (actual, expected) in [
        (&actual.bundle, &expected.bundle),
        (&actual.decode, &expected.decode),
        (&actual.embedding, &expected.embedding),
    ] {
        color_eyre::eyre::ensure!(
            actual.receipt.stage == expected.receipt.stage
                && actual.receipt.cache_key == expected.receipt.cache_key
                && actual
                    .receipt
                    .parents
                    .iter()
                    .map(|parent| (parent.stage, &parent.cache_key))
                    .eq(expected
                        .receipt
                        .parents
                        .iter()
                        .map(|parent| (parent.stage, &parent.cache_key)))
                && actual.receipt.dependencies == expected.receipt.dependencies
                && actual.receipt.geometry == expected.receipt.geometry
                && actual.receipt.availability == expected.receipt.availability,
            "embedding cache stage dependency identity does not match the requested recipe"
        );
    }
    Ok(())
}

pub(crate) fn run_from_embedding_cache(
    recipe: &RecipeSpec,
    recording: &ValidatedRecording,
    context: &mut RunContext<'_>,
    expected_cache_key: &Sha256Digest,
    geometry: &GeometryReceipt,
    hit: cache::EmbeddingCacheHit,
) -> color_eyre::eyre::Result<super::run::PreparedOutputs> {
    ensure_recipe_matches_bundle(recipe, recording)?;
    let pipeline_geometry = PipelineGeometry::from_imported(
        &recording.bundle.manifest().audio,
        &recording.bundle.manifest().geometry,
    )?;
    let snapshot_document: EmbeddingStageDocument = serde_json::from_slice(&hit.snapshot.0)?;
    snapshot_document.validate()?;
    color_eyre::eyre::ensure!(
        snapshot_document.geometry == *geometry,
        "cached embedding snapshot geometry does not match the recording"
    );
    color_eyre::eyre::ensure!(
        snapshot_document.stage_key == hit.key,
        "cached embedding snapshot key does not match its receipts"
    );
    let entries = snapshot_document
        .entries
        .iter()
        .map(|entry| {
            Ok(LibraryEmbeddingStageEntry::new(
                library_availability(&entry.availability)?,
                entry.values.clone(),
            ))
        })
        .collect::<color_eyre::eyre::Result<Vec<_>>>()?;
    let snapshot = EmbeddingStageSnapshot::from_flat_parts(
        pipeline_geometry,
        snapshot_document.segmentation_shape,
        snapshot_document.segmentation_values,
        entries,
        EmbeddingReceipt {
            clean_mask_count: snapshot_document.embedding_receipt.clean_mask,
            full_mask_fallback_count: snapshot_document.embedding_receipt.full_mask_fallback,
            inactive_count: snapshot_document.embedding_receipt.inactive,
            inference_failure_count: snapshot_document.embedding_receipt.inference_failed,
        },
    )?;
    let mut documents = BTreeSet::new();
    let mut stage_documents = Vec::new();
    for (_, bytes, _) in &hit.stage_receipts {
        let document: ReceiptDocument = serde_json::from_slice(bytes)?;
        documents.insert(document.receipt.stage);
        stage_documents.push(document);
    }
    color_eyre::eyre::ensure!(
        documents.len() == 3,
        "embedding cache receipts are incomplete"
    );
    stage_documents.sort_by_key(|document| document.receipt.stage);
    let cached_stage_receipts = InferenceStageReceipts {
        bundle: stage_documents
            .iter()
            .find(|document| document.receipt.stage == StageKind::Bundle)
            .cloned()
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding cache has no bundle receipt"))?,
        decode: stage_documents
            .iter()
            .find(|document| document.receipt.stage == StageKind::Decode)
            .cloned()
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding cache has no decode receipt"))?,
        embedding: stage_documents
            .iter()
            .find(|document| document.receipt.stage == StageKind::Embedding)
            .cloned()
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding cache has no embedding receipt"))?,
    };
    let expected_stage_receipts = stage_receipts(
        recipe,
        recording,
        context.runtime,
        context.execution,
        geometry.clone(),
        snapshot_document.embedding_receipt,
    )?;
    ensure_stage_dependency_identity(&cached_stage_receipts, &expected_stage_receipts)?;
    let pipeline = ImportedDiarizationPipeline::new(
        recording.bundle.clone(),
        context.embedding_model,
        context.plda_dir,
    )?;
    let config = recipe.pipeline_config(context.mode)?;
    let result = pipeline.finish_embedding_stage(snapshot, &config)?;
    let availability = super::run::availability_counts(&result);
    let tracks = super::run::speaker_tracks(recording, &result, geometry.clone());
    let tracks_bytes = serde_json::to_vec_pretty(&tracks)?;
    let hypothesis_bytes = result.rttm(&recording.spec.id).into_bytes();
    let output_relative = super::run::recording_relative(recipe, recording);
    let reconstruction_outputs = vec![
        super::run::artifact_for_path(&output_relative.join("speaker_tracks.json"), &tracks_bytes),
        super::run::artifact_for_path(&output_relative.join("output.rttm"), &hypothesis_bytes),
    ];
    let clustering_dependencies = clustering_dependencies(recipe, context.execution)?;
    let reconstruction_dependencies =
        reconstruction_dependencies(recipe, geometry, context.execution)?;
    let snapshot_relative_path = super::run::recording_relative(recipe, recording)
        .join("stages")
        .join("embedding_snapshot.json");
    let embedding_receipt = receipt_with_output(
        &cached_stage_receipts.embedding,
        super::run::artifact_for_path(&snapshot_relative_path, &hit.snapshot.0),
    )?;
    let clustering_receipt = make_receipt(
        StageKind::Clustering,
        clustering_dependencies,
        vec![receipt_ref(&embedding_receipt)],
        geometry.clone(),
        availability,
    );
    let mut reconstruction_receipt = make_receipt(
        StageKind::Reconstruction,
        reconstruction_dependencies,
        vec![receipt_ref(&clustering_receipt)],
        geometry.clone(),
        availability,
    );
    reconstruction_receipt.receipt.outputs = reconstruction_outputs;
    reconstruction_receipt.receipt.cache_key = expected_cache_key.clone();
    reconstruction_receipt.receipt_sha256 = canonical_json_digest(&reconstruction_receipt.receipt)?;
    let receipt_documents = [
        cached_stage_receipts.bundle.clone(),
        cached_stage_receipts.decode.clone(),
        embedding_receipt,
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
        .collect::<color_eyre::eyre::Result<Vec<_>>>()?;
    Ok(super::run::PreparedOutputs {
        cache_key: expected_cache_key.clone(),
        receipt_files,
        embedding_stage_receipt_files: Vec::new(),
        embedding_snapshot_bytes: hit.snapshot.0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::wavlm_bridge::domain::{
        BridgeMode, EmbeddingModelIdentity, ModelIdentity, Precision, digest_bytes,
    };

    fn runtime() -> RuntimeIdentity {
        RuntimeIdentity {
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
        }
    }

    fn execution(executable: &[u8], runtime_build: &str) -> ExecutionIdentity {
        ExecutionIdentity {
            executable_sha256: digest_bytes(executable),
            onnx_runtime_build: runtime_build.into(),
            onnx_runtime_api: 17,
        }
    }

    #[test]
    fn embedding_implementation_identity_is_a_source_digest() {
        assert_eq!(EMBEDDING_IMPLEMENTATION_IDENTITY.len(), 64);
        assert!(
            EMBEDDING_IMPLEMENTATION_IDENTITY
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
    }

    #[test]
    fn embedding_runtime_identity_tracks_onnx_runtime_but_not_downstream_executable() {
        let runtime = runtime();
        let first = execution(b"first executable", "runtime-a");
        let rebuilt = execution(b"rebuilt executable", "runtime-a");
        let upgraded_runtime = execution(b"first executable", "runtime-b");

        assert_eq!(
            embedding_runtime_identity(&runtime, &first),
            embedding_runtime_identity(&runtime, &rebuilt)
        );
        assert_ne!(
            embedding_runtime_identity(&runtime, &first),
            embedding_runtime_identity(&runtime, &upgraded_runtime)
        );
        assert_ne!(
            canonical_json_digest(&first).unwrap(),
            canonical_json_digest(&rebuilt).unwrap()
        );
    }
}
