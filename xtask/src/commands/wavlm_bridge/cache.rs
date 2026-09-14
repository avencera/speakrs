use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use color_eyre::eyre::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use super::domain::{
    ArtifactRef, CACHE_SCHEMA_VERSION, EMBEDDING_CACHE_SCHEMA_VERSION, EmbeddingStageDocument,
    ReceiptDocument, ReceiptRef, Sha256Digest, SpeakerTracks, StageKind, canonical_json_digest,
    digest_bytes, ensure_directory, ensure_regular_file, make_tree_read_only,
};

/// One immutable cache entry for a recording and named recipe
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheEntry {
    pub schema_version: u32,
    pub cache_key: Sha256Digest,
    pub recording_id: String,
    pub recipe_id: String,
    pub stage_receipts: Vec<ArtifactRef>,
    pub speaker_tracks: ArtifactRef,
    pub hypothesis: ArtifactRef,
}

/// Files loaded from a verified cache entry
#[derive(Debug)]
pub struct CacheHit {
    pub key: Sha256Digest,
    pub stage_receipts: Vec<(PathBuf, Vec<u8>, ArtifactRef)>,
    pub speaker_tracks: (Vec<u8>, ArtifactRef),
    pub hypothesis: (Vec<u8>, ArtifactRef),
}

/// Files loaded from a verified embedding-stage cache entry
#[derive(Debug)]
pub struct EmbeddingCacheHit {
    pub key: Sha256Digest,
    pub stage_receipts: Vec<(PathBuf, Vec<u8>, ArtifactRef)>,
    pub snapshot: (Vec<u8>, ArtifactRef),
}

pub fn lookup(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
    recipe_id: &str,
) -> Result<Option<CacheHit>> {
    if !cache_root.exists() {
        return Ok(None);
    }
    ensure_directory(cache_root, "cache root")?;
    let entry_root = cache_root.join(key.as_str());
    if !entry_root.exists() {
        return Ok(None);
    }
    ensure_directory(&entry_root, "cache entry")?;
    let entry_path = entry_root.join("entry.json");
    let marker_path = entry_root.join(".complete");
    ensure_regular_file(&entry_path, "cache entry manifest")?;
    ensure_regular_file(&marker_path, "cache completion marker")?;
    let entry_bytes = fs::read(&entry_path)?;
    let expected_marker = digest_bytes(&entry_bytes);
    let marker = fs::read_to_string(&marker_path)?;
    ensure!(
        marker.trim() == expected_marker.as_str(),
        "cache entry {} has a stale completion marker",
        entry_root.display()
    );
    let entry: CacheEntry = serde_json::from_slice(&entry_bytes)
        .wrap_err_with(|| format!("invalid cache entry {}", entry_path.display()))?;
    ensure!(
        entry.schema_version == CACHE_SCHEMA_VERSION,
        "unsupported cache schema {}",
        entry.schema_version
    );
    ensure!(entry.cache_key == *key, "cache entry key mismatch");
    ensure!(
        entry.recording_id == recording_id,
        "cache recording identity mismatch"
    );
    ensure!(
        entry.recipe_id == recipe_id,
        "cache recipe identity mismatch"
    );
    ensure!(
        !entry.stage_receipts.is_empty(),
        "cache entry has no stage receipts"
    );

    let mut receipts = Vec::with_capacity(entry.stage_receipts.len());
    let mut documents = BTreeMap::new();
    for artifact in &entry.stage_receipts {
        let path = member_path(&entry_root, &artifact.relative_path)?;
        let bytes = read_hashed_member(&path, artifact)?;
        let document: ReceiptDocument = serde_json::from_slice(&bytes)
            .wrap_err_with(|| format!("invalid cached receipt {}", path.display()))?;
        let digest = canonical_json_digest(&document.receipt)?;
        ensure!(
            digest == document.receipt_sha256,
            "cached receipt {} has an invalid receipt hash",
            path.display()
        );
        ensure!(
            document.receipt.cache_key == entry.cache_key
                || document.receipt.stage != StageKind::Reconstruction,
            "cached reconstruction receipt does not use the entry key"
        );
        ensure!(
            documents.insert(document.receipt.stage, document).is_none(),
            "cache entry contains duplicate stage receipts"
        );
        receipts.push((artifact.relative_path.clone(), bytes, artifact.clone()));
    }
    validate_receipt_chain(&documents)?;
    let reconstruction = documents
        .get(&StageKind::Reconstruction)
        .expect("checked stage order");
    ensure!(
        reconstruction
            .receipt
            .outputs
            .contains(&entry.speaker_tracks),
        "cached reconstruction receipt does not name speaker tracks"
    );
    ensure!(
        reconstruction.receipt.outputs.contains(&entry.hypothesis),
        "cached reconstruction receipt does not name RTTM"
    );
    let tracks_path = member_path(&entry_root, &entry.speaker_tracks.relative_path)?;
    let tracks_bytes = read_hashed_member(&tracks_path, &entry.speaker_tracks)?;
    let tracks: SpeakerTracks = serde_json::from_slice(&tracks_bytes)
        .wrap_err_with(|| format!("invalid cached speaker tracks {}", tracks_path.display()))?;
    ensure!(
        tracks.recording_id == entry.recording_id,
        "cached speaker tracks recording identity mismatch"
    );
    let hypothesis_path = member_path(&entry_root, &entry.hypothesis.relative_path)?;
    let hypothesis_bytes = read_hashed_member(&hypothesis_path, &entry.hypothesis)?;
    Ok(Some(CacheHit {
        key: entry.cache_key,
        stage_receipts: receipts,
        speaker_tracks: (tracks_bytes, entry.speaker_tracks),
        hypothesis: (hypothesis_bytes, entry.hypothesis),
    }))
}

pub fn lookup_embedding(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
) -> Result<Option<EmbeddingCacheHit>> {
    if !cache_root.exists() {
        return Ok(None);
    }
    ensure_directory(cache_root, "cache root")?;
    let embedding_root = cache_root.join("embedding");
    if !embedding_root.exists() {
        return Ok(None);
    }
    ensure_directory(&embedding_root, "embedding cache root")?;
    let entry_root = embedding_root.join(key.as_str());
    if !entry_root.exists() {
        return Ok(None);
    }
    ensure_directory(&entry_root, "embedding cache entry")?;
    let entry_path = entry_root.join("entry.json");
    let marker_path = entry_root.join(".complete");
    ensure_regular_file(&entry_path, "embedding cache entry manifest")?;
    ensure_regular_file(&marker_path, "embedding cache completion marker")?;
    ensure!(
        fs::metadata(&entry_path)?.len() <= super::domain::MAX_EMBEDDING_STAGE_BYTES as u64,
        "embedding cache entry manifest exceeds size bound"
    );
    let entry_bytes = fs::read(&entry_path)?;
    ensure!(
        entry_bytes.len() <= super::domain::MAX_EMBEDDING_STAGE_BYTES,
        "embedding cache entry manifest exceeds size bound"
    );
    let marker = fs::read_to_string(&marker_path)?;
    ensure!(
        marker.trim() == digest_bytes(&entry_bytes).as_str(),
        "embedding cache entry has a stale completion marker"
    );
    let entry: EmbeddingCacheEntry = serde_json::from_slice(&entry_bytes)
        .wrap_err_with(|| format!("invalid embedding cache entry {}", entry_path.display()))?;
    ensure!(
        entry.schema_version == EMBEDDING_CACHE_SCHEMA_VERSION,
        "unsupported embedding cache schema {}",
        entry.schema_version
    );
    ensure!(
        entry.cache_key == *key,
        "embedding cache entry key mismatch"
    );
    ensure!(
        entry.recording_id == recording_id,
        "embedding cache recording identity mismatch"
    );
    ensure!(
        entry.stage_receipts.len() == 3,
        "embedding cache requires three stage receipts"
    );
    let mut receipts = Vec::with_capacity(3);
    let mut documents = BTreeMap::new();
    for artifact in &entry.stage_receipts {
        let path = member_path(&entry_root, &artifact.relative_path)?;
        let bytes = read_bounded_member(&path, artifact, super::domain::MAX_EMBEDDING_STAGE_BYTES)?;
        let document: ReceiptDocument = serde_json::from_slice(&bytes)
            .wrap_err_with(|| format!("invalid embedding receipt {}", path.display()))?;
        validate_receipt_document(&document, &path)?;
        ensure!(
            document.receipt.schema_version == super::domain::RECEIPT_SCHEMA_VERSION,
            "embedding cache receipt has unsupported schema"
        );
        ensure!(
            matches!(
                document.receipt.stage,
                StageKind::Bundle | StageKind::Decode | StageKind::Embedding
            ),
            "embedding cache contains a non-embedding stage"
        );
        ensure!(
            documents.insert(document.receipt.stage, document).is_none(),
            "embedding cache contains duplicate stage receipts"
        );
        receipts.push((artifact.relative_path.clone(), bytes, artifact.clone()));
    }
    validate_embedding_receipt_chain(&documents, key)?;
    let snapshot_path = member_path(&entry_root, &entry.snapshot.relative_path)?;
    let snapshot = read_bounded_member(
        &snapshot_path,
        &entry.snapshot,
        super::domain::MAX_EMBEDDING_STAGE_BYTES,
    )?;
    let document: EmbeddingStageDocument = serde_json::from_slice(&snapshot)
        .wrap_err_with(|| format!("invalid embedding snapshot {}", snapshot_path.display()))?;
    document.validate()?;
    ensure!(
        document.recording_id == recording_id && document.stage_key == *key,
        "embedding snapshot identity does not match cache entry"
    );
    let embedding = documents
        .get(&StageKind::Embedding)
        .expect("embedding receipt was checked");
    ensure!(
        embedding.receipt.outputs.contains(&entry.snapshot),
        "embedding receipt does not name the snapshot"
    );
    Ok(Some(EmbeddingCacheHit {
        key: entry.cache_key,
        stage_receipts: receipts,
        snapshot: (snapshot, entry.snapshot),
    }))
}

pub fn publish(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
    recipe_id: &str,
    stage_receipts: &[(String, Vec<u8>)],
    speaker_tracks: &[u8],
    hypothesis: &[u8],
) -> Result<CacheEntry> {
    fs::create_dir_all(cache_root)
        .wrap_err_with(|| format!("failed to create cache root {}", cache_root.display()))?;
    ensure_directory(cache_root, "cache root")?;
    let entry_root = cache_root.join(key.as_str());
    ensure!(
        !entry_root.exists(),
        "cache entry already exists: {}",
        entry_root.display()
    );
    let staging = tempfile::tempdir_in(cache_root).wrap_err_with(|| {
        format!(
            "failed to create cache staging directory in {}",
            cache_root.display()
        )
    })?;
    let staging_root = staging.path();
    fs::create_dir(staging_root.join("stages"))?;
    let mut receipt_artifacts = Vec::with_capacity(stage_receipts.len());
    for (name, bytes) in stage_receipts {
        let relative = PathBuf::from("stages").join(name);
        let path = staging_root.join(&relative);
        write_new(&path, bytes)?;
        receipt_artifacts.push(artifact_for(staging_root, &relative)?);
    }
    let tracks_relative = PathBuf::from("speaker_tracks.json");
    write_new(&staging_root.join(&tracks_relative), speaker_tracks)?;
    let tracks_artifact = artifact_for(staging_root, &tracks_relative)?;
    let hypothesis_relative = PathBuf::from("output.rttm");
    write_new(&staging_root.join(&hypothesis_relative), hypothesis)?;
    let hypothesis_artifact = artifact_for(staging_root, &hypothesis_relative)?;
    let entry = CacheEntry {
        schema_version: CACHE_SCHEMA_VERSION,
        cache_key: key.clone(),
        recording_id: recording_id.to_owned(),
        recipe_id: recipe_id.to_owned(),
        stage_receipts: receipt_artifacts,
        speaker_tracks: tracks_artifact,
        hypothesis: hypothesis_artifact,
    };
    let entry_bytes = serde_json::to_vec_pretty(&entry)?;
    write_new(&staging_root.join("entry.json"), &entry_bytes)?;
    fs::create_dir(&entry_root).wrap_err_with(|| {
        format!(
            "failed to reserve cache entry {} without replacement",
            entry_root.display()
        )
    })?;
    publish_staged_tree(staging_root, &entry_root)?;
    let marker = format!("{}\n", digest_bytes(&entry_bytes));
    write_new(&entry_root.join(".complete"), marker.as_bytes())?;
    make_tree_read_only(&entry_root)?;
    Ok(entry)
}

pub fn publish_embedding(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
    stage_receipts: &[(String, Vec<u8>)],
    snapshot: &[u8],
) -> Result<EmbeddingCacheEntry> {
    ensure!(
        stage_receipts.len() == 3,
        "embedding cache requires three stage receipts"
    );
    ensure!(
        snapshot.len() <= super::domain::MAX_EMBEDDING_STAGE_BYTES,
        "embedding snapshot exceeds size bound"
    );
    fs::create_dir_all(cache_root)
        .wrap_err_with(|| format!("failed to create cache root {}", cache_root.display()))?;
    ensure_directory(cache_root, "cache root")?;
    let embedding_root = cache_root.join("embedding");
    if !embedding_root.exists() {
        fs::create_dir(&embedding_root)?;
    }
    ensure_directory(&embedding_root, "embedding cache root")?;
    let entry_root = embedding_root.join(key.as_str());
    let staging = tempfile::tempdir_in(&embedding_root).wrap_err_with(|| {
        format!(
            "failed to create embedding cache staging directory in {}",
            embedding_root.display()
        )
    })?;
    let staging_root = staging.path();
    fs::create_dir(staging_root.join("stages"))?;
    let mut receipt_artifacts = Vec::with_capacity(stage_receipts.len());
    for (name, bytes) in stage_receipts {
        let relative = PathBuf::from("stages").join(name);
        write_new(&staging_root.join(&relative), bytes)?;
        receipt_artifacts.push(artifact_for(staging_root, &relative)?);
    }
    let snapshot_relative = PathBuf::from("embedding_snapshot.json");
    write_new(&staging_root.join(&snapshot_relative), snapshot)?;
    let snapshot_artifact = artifact_for(staging_root, &snapshot_relative)?;
    let entry = EmbeddingCacheEntry {
        schema_version: EMBEDDING_CACHE_SCHEMA_VERSION,
        cache_key: key.clone(),
        recording_id: recording_id.to_owned(),
        stage_receipts: receipt_artifacts,
        snapshot: snapshot_artifact,
    };
    let entry_bytes = serde_json::to_vec_pretty(&entry)?;
    write_new(&staging_root.join("entry.json"), &entry_bytes)?;
    fs::create_dir(&entry_root).wrap_err_with(|| {
        format!(
            "failed to reserve embedding cache entry {} without replacement",
            entry_root.display()
        )
    })?;
    publish_staged_tree(staging_root, &entry_root)?;
    write_new(
        &entry_root.join(".complete"),
        format!("{}\n", digest_bytes(&entry_bytes)).as_bytes(),
    )?;
    make_tree_read_only(&entry_root)?;
    Ok(entry)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingCacheEntry {
    pub schema_version: u32,
    pub cache_key: Sha256Digest,
    pub recording_id: String,
    pub stage_receipts: Vec<ArtifactRef>,
    pub snapshot: ArtifactRef,
}

fn publish_staged_tree(staging: &Path, destination: &Path) -> Result<()> {
    for entry in fs::read_dir(staging)? {
        let entry = entry?;
        let source = entry.path();
        let target = destination.join(entry.file_name());
        let metadata = fs::symlink_metadata(&source)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "cache staging tree contains a symlink: {}",
            source.display()
        );
        if metadata.is_dir() {
            fs::create_dir(&target).wrap_err_with(|| {
                format!(
                    "failed to create cache directory {} without replacement",
                    target.display()
                )
            })?;
            publish_staged_tree(&source, &target)?;
        } else if metadata.is_file() {
            write_new(&target, &fs::read(&source)?)?;
        } else {
            ensure!(
                false,
                "unsupported cache staging member: {}",
                source.display()
            );
        }
    }
    Ok(())
}

fn validate_receipt_chain(documents: &BTreeMap<StageKind, ReceiptDocument>) -> Result<()> {
    let order = [
        StageKind::Bundle,
        StageKind::Decode,
        StageKind::Embedding,
        StageKind::Clustering,
        StageKind::Reconstruction,
    ];
    ensure!(
        documents.len() == order.len(),
        "cache entry does not contain all five stages"
    );
    for (index, stage) in order.iter().enumerate() {
        let document = documents
            .get(stage)
            .ok_or_else(|| color_eyre::eyre::eyre!("cache entry is missing {stage:?}"))?;
        ensure!(
            document.receipt.schema_version == super::domain::RECEIPT_SCHEMA_VERSION,
            "cache receipt has unsupported schema"
        );
        if index > 0 {
            let parent = documents
                .get(&order[index - 1])
                .expect("checked stage order");
            let reference = ReceiptRef::from(&parent.receipt, &parent.receipt_sha256);
            ensure!(
                document.receipt.parents.contains(&reference),
                "cache receipt {stage:?} does not reference the previous stage"
            );
        }
    }
    Ok(())
}

fn validate_embedding_receipt_chain(
    documents: &BTreeMap<StageKind, ReceiptDocument>,
    key: &Sha256Digest,
) -> Result<()> {
    let order = [StageKind::Bundle, StageKind::Decode, StageKind::Embedding];
    ensure!(
        documents.len() == order.len(),
        "embedding cache does not contain exactly three stages"
    );
    for (index, stage) in order.iter().enumerate() {
        let document = documents
            .get(stage)
            .ok_or_else(|| color_eyre::eyre::eyre!("embedding cache is missing {stage:?}"))?;
        if index > 0 {
            let parent = documents
                .get(&order[index - 1])
                .expect("checked embedding stage order");
            let reference = ReceiptRef::from(&parent.receipt, &parent.receipt_sha256);
            ensure!(
                document.receipt.parents.contains(&reference),
                "embedding cache receipt {stage:?} does not reference the previous stage"
            );
        }
    }
    let embedding = documents
        .get(&StageKind::Embedding)
        .expect("checked embedding stage order");
    ensure!(
        embedding.receipt.cache_key == *key,
        "embedding receipt key does not match cache key"
    );
    Ok(())
}

fn validate_receipt_document(document: &ReceiptDocument, path: &Path) -> Result<()> {
    let digest = canonical_json_digest(&document.receipt)?;
    ensure!(
        digest == document.receipt_sha256,
        "cached receipt {} has an invalid receipt hash",
        path.display()
    );
    Ok(())
}

fn read_hashed_member(path: &Path, artifact: &ArtifactRef) -> Result<Vec<u8>> {
    read_bounded_member(path, artifact, usize::MAX)
}

fn read_bounded_member(path: &Path, artifact: &ArtifactRef, limit: usize) -> Result<Vec<u8>> {
    ensure_regular_file(path, "cache artifact")?;
    ensure!(
        artifact.bytes <= limit as u64,
        "cache artifact exceeds size bound: {}",
        path.display()
    );
    let bytes = fs::read(path)?;
    ensure!(
        bytes.len() as u64 == artifact.bytes,
        "cache artifact size mismatch: {}",
        path.display()
    );
    ensure!(
        digest_bytes(&bytes) == artifact.sha256,
        "cache artifact digest mismatch: {}",
        path.display()
    );
    Ok(bytes)
}

fn artifact_for(root: &Path, relative: &Path) -> Result<ArtifactRef> {
    let path = member_path(root, relative)?;
    let bytes = fs::read(&path)?;
    Ok(ArtifactRef {
        relative_path: relative.to_owned(),
        sha256: digest_bytes(&bytes),
        bytes: bytes.len() as u64,
    })
}

fn validate_member_path(path: &Path) -> Result<()> {
    ensure!(
        path.components().next().is_some(),
        "cache member path cannot be empty"
    );
    ensure!(
        !path.is_absolute(),
        "cache member path must be relative: {}",
        path.display()
    );
    ensure!(
        !path.components().any(|component| matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )),
        "cache member path escapes entry: {}",
        path.display()
    );
    Ok(())
}

fn member_path(root: &Path, relative: &Path) -> Result<PathBuf> {
    validate_member_path(relative)?;
    ensure_directory(root, "cache entry")?;
    let root_canonical = fs::canonicalize(root)?;
    let mut current = root.to_owned();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            continue;
        };
        current.push(name);
        let metadata = fs::symlink_metadata(&current)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "cache member path contains a symlink: {}",
            current.display()
        );
    }
    let canonical = fs::canonicalize(&current)?;
    ensure!(
        canonical.starts_with(&root_canonical),
        "cache member path escapes entry: {}",
        relative.display()
    );
    Ok(current)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
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

impl ReceiptRef {
    fn from(receipt: &super::domain::StageReceipt, hash: &Sha256Digest) -> Self {
        Self {
            stage: receipt.stage,
            cache_key: receipt.cache_key.clone(),
            receipt_sha256: hash.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::wavlm_bridge::domain::{
        AvailabilityCounts, EmbeddingStageDocument, GeometryReceipt, RationalReceipt,
        StageDependency, StageReceipt,
    };

    fn geometry() -> GeometryReceipt {
        GeometryReceipt {
            sample_rate: 16_000,
            sample_count: 0,
            window_samples: 1,
            step_samples: 1,
            chunks: Vec::new(),
            frame_grid: super::super::domain::FrameGridReceipt {
                frame_count: 0,
                origin: RationalReceipt {
                    numerator: 0,
                    denominator: 1,
                },
                step: RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
                support: RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
            },
            aggregate_grid: super::super::domain::FrameGridReceipt {
                frame_count: 0,
                origin: RationalReceipt {
                    numerator: 0,
                    denominator: 1,
                },
                step: RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
                support: RationalReceipt {
                    numerator: 1,
                    denominator: 1,
                },
            },
            start_frames: Vec::new(),
            output_frames: 0,
            output_extent_start_samples: 0,
            output_extent_end_samples: 0,
            output_extent_policy: speakrs::imported_segmentation::OutputExtentPolicy::AggregateGrid,
        }
    }

    fn stage_documents(
        key: &Sha256Digest,
        tracks: &ArtifactRef,
        hypothesis: &ArtifactRef,
    ) -> Vec<(String, Vec<u8>)> {
        let mut documents = Vec::new();
        let stages = [
            StageKind::Bundle,
            StageKind::Decode,
            StageKind::Embedding,
            StageKind::Clustering,
            StageKind::Reconstruction,
        ];
        let mut parents = Vec::new();
        for stage in stages {
            let outputs = if stage == StageKind::Reconstruction {
                vec![tracks.clone(), hypothesis.clone()]
            } else {
                Vec::new()
            };
            let receipt = StageReceipt {
                schema_version: super::super::domain::RECEIPT_SCHEMA_VERSION,
                stage,
                cache_key: if stage == StageKind::Reconstruction {
                    key.clone()
                } else {
                    digest_bytes(format!("{stage:?}").as_bytes())
                },
                parents: parents.clone(),
                dependencies: Vec::<StageDependency>::new(),
                geometry: geometry(),
                availability: AvailabilityCounts::default(),
                outputs,
            };
            let hash = canonical_json_digest(&receipt).unwrap();
            let document = ReceiptDocument {
                receipt,
                receipt_sha256: hash.clone(),
            };
            parents = vec![ReceiptRef {
                stage,
                cache_key: document.receipt.cache_key.clone(),
                receipt_sha256: hash,
            }];
            documents.push((
                format!("{}.receipt.json", stage_name(stage)),
                serde_json::to_vec(&document).unwrap(),
            ));
        }
        documents
    }

    fn write_self_consistent_entry(
        cache: &Path,
        key: &Sha256Digest,
        tracks_relative: &Path,
        tracks: &[u8],
        hypothesis_relative: &Path,
        hypothesis: &[u8],
    ) {
        let entry_root = cache.join(key.as_str());
        fs::create_dir(&entry_root).unwrap();
        let tracks_artifact = ArtifactRef {
            relative_path: tracks_relative.to_owned(),
            sha256: digest_bytes(tracks),
            bytes: tracks.len() as u64,
        };
        let hypothesis_artifact = ArtifactRef {
            relative_path: hypothesis_relative.to_owned(),
            sha256: digest_bytes(hypothesis),
            bytes: hypothesis.len() as u64,
        };
        for (name, bytes) in stage_documents(key, &tracks_artifact, &hypothesis_artifact) {
            let path = entry_root.join("stages").join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        for (relative, bytes) in [(tracks_relative, tracks), (hypothesis_relative, hypothesis)] {
            let path = entry_root.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        let stage_receipts = fs::read_dir(entry_root.join("stages"))
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                let bytes = fs::read(&path).unwrap();
                ArtifactRef {
                    relative_path: PathBuf::from("stages").join(path.file_name().unwrap()),
                    sha256: digest_bytes(&bytes),
                    bytes: bytes.len() as u64,
                }
            })
            .collect();
        let entry = CacheEntry {
            schema_version: CACHE_SCHEMA_VERSION,
            cache_key: key.clone(),
            recording_id: "recording".into(),
            recipe_id: "recipe".into(),
            stage_receipts,
            speaker_tracks: tracks_artifact,
            hypothesis: hypothesis_artifact,
        };
        let bytes = serde_json::to_vec(&entry).unwrap();
        fs::write(entry_root.join("entry.json"), &bytes).unwrap();
        fs::write(
            entry_root.join(".complete"),
            format!("{}\n", digest_bytes(&bytes)),
        )
        .unwrap();
    }

    fn embedding_stage_documents(
        key: &Sha256Digest,
        snapshot: &ArtifactRef,
    ) -> Vec<(String, Vec<u8>)> {
        let stages = [StageKind::Bundle, StageKind::Decode, StageKind::Embedding];
        let mut parents = Vec::new();
        let mut documents = Vec::new();
        for stage in stages {
            let receipt = StageReceipt {
                schema_version: super::super::domain::RECEIPT_SCHEMA_VERSION,
                stage,
                cache_key: if stage == StageKind::Embedding {
                    key.clone()
                } else {
                    digest_bytes(format!("{stage:?}").as_bytes())
                },
                parents: parents.clone(),
                dependencies: Vec::new(),
                geometry: geometry(),
                availability: AvailabilityCounts::default(),
                outputs: if stage == StageKind::Embedding {
                    vec![snapshot.clone()]
                } else {
                    Vec::new()
                },
            };
            let hash = canonical_json_digest(&receipt).unwrap();
            let document = ReceiptDocument {
                receipt,
                receipt_sha256: hash.clone(),
            };
            parents = vec![ReceiptRef {
                stage,
                cache_key: document.receipt.cache_key.clone(),
                receipt_sha256: hash,
            }];
            documents.push((
                format!("{}.receipt.json", stage_name(stage)),
                serde_json::to_vec(&document).unwrap(),
            ));
        }
        documents
    }

    #[test]
    fn embedding_cache_round_trip_is_immutable() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"embedding-stage");
        let snapshot_document = EmbeddingStageDocument {
            schema_version: super::super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: key.clone(),
            geometry: geometry(),
            segmentation_shape: [0, 0, 0],
            segmentation_values: Vec::new(),
            entries: Vec::new(),
            embedding_receipt: AvailabilityCounts::default(),
        };
        let snapshot = serde_json::to_vec(&snapshot_document).unwrap();
        let snapshot_artifact = ArtifactRef {
            relative_path: "embedding_snapshot.json".into(),
            sha256: digest_bytes(&snapshot),
            bytes: snapshot.len() as u64,
        };
        let stage_receipts = embedding_stage_documents(&key, &snapshot_artifact);
        publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot).unwrap();
        let hit = lookup_embedding(cache.path(), &key, "recording")
            .unwrap()
            .expect("published embedding cache should be reusable");
        assert_eq!(hit.key, key);
        assert_eq!(hit.snapshot.0, snapshot);
        assert!(
            publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot).is_err()
        );
    }

    #[test]
    fn embedding_cache_rejects_interrupted_entry() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"interrupted-embedding-stage");
        let entry_root = cache.path().join("embedding").join(key.as_str());
        fs::create_dir_all(&entry_root).unwrap();
        fs::write(entry_root.join("entry.json"), b"{}").unwrap();
        assert!(lookup_embedding(cache.path(), &key, "recording").is_err());
    }

    #[test]
    fn partial_entry_fails_closed() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"partial");
        let entry_root = cache.path().join(key.as_str());
        fs::create_dir(&entry_root).unwrap();
        let entry = CacheEntry {
            schema_version: CACHE_SCHEMA_VERSION,
            cache_key: key.clone(),
            recording_id: "recording".into(),
            recipe_id: "recipe".into(),
            stage_receipts: Vec::new(),
            speaker_tracks: ArtifactRef {
                relative_path: "speaker_tracks.json".into(),
                sha256: digest_bytes(b""),
                bytes: 0,
            },
            hypothesis: ArtifactRef {
                relative_path: "output.rttm".into(),
                sha256: digest_bytes(b""),
                bytes: 0,
            },
        };
        let bytes = serde_json::to_vec(&entry).unwrap();
        fs::write(entry_root.join("entry.json"), &bytes).unwrap();
        fs::write(
            entry_root.join(".complete"),
            format!("{}\n", digest_bytes(&bytes)),
        )
        .unwrap();
        let error = match lookup(cache.path(), &key, "recording", "recipe") {
            Ok(_) => panic!("partial cache entry was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("no stage receipts"));
    }

    #[test]
    fn published_entry_is_reused_only_after_receipts_validate() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"complete");
        let tracks = serde_json::to_vec(&SpeakerTracks {
            schema_version: super::super::domain::SYSTEM_SCHEMA_VERSION,
            recording_id: "recording".into(),
            geometry: geometry(),
            tracks: Vec::new(),
        })
        .unwrap();
        let mut documents = Vec::new();
        let stages = [
            StageKind::Bundle,
            StageKind::Decode,
            StageKind::Embedding,
            StageKind::Clustering,
            StageKind::Reconstruction,
        ];
        let mut parents = Vec::new();
        for stage in stages {
            let outputs = if stage == StageKind::Reconstruction {
                vec![
                    ArtifactRef {
                        relative_path: "speaker_tracks.json".into(),
                        sha256: digest_bytes(&tracks),
                        bytes: tracks.len() as u64,
                    },
                    ArtifactRef {
                        relative_path: "output.rttm".into(),
                        sha256: digest_bytes(b""),
                        bytes: 0,
                    },
                ]
            } else {
                Vec::new()
            };
            let receipt = StageReceipt {
                schema_version: super::super::domain::RECEIPT_SCHEMA_VERSION,
                stage,
                cache_key: if stage == StageKind::Reconstruction {
                    key.clone()
                } else {
                    digest_bytes(format!("{stage:?}").as_bytes())
                },
                parents: parents.clone(),
                dependencies: Vec::<StageDependency>::new(),
                geometry: geometry(),
                availability: AvailabilityCounts::default(),
                outputs,
            };
            let hash = canonical_json_digest(&receipt).unwrap();
            let document = ReceiptDocument {
                receipt,
                receipt_sha256: hash.clone(),
            };
            parents = vec![ReceiptRef {
                stage,
                cache_key: document.receipt.cache_key.clone(),
                receipt_sha256: hash,
            }];
            documents.push((
                format!("{}.receipt.json", stage_name(stage)),
                serde_json::to_vec(&document).unwrap(),
            ));
        }
        publish(
            cache.path(),
            &key,
            "recording",
            "recipe",
            &documents,
            &tracks,
            b"",
        )
        .unwrap();
        let hit = lookup(cache.path(), &key, "recording", "recipe").unwrap();
        assert!(hit.is_some());
        assert!(
            publish(
                cache.path(),
                &key,
                "recording",
                "recipe",
                &documents,
                &tracks,
                b"",
            )
            .is_err(),
            "cache publication replaced an existing entry"
        );
    }

    #[test]
    fn lookup_rejects_speaker_track_and_hypothesis_traversal() {
        let tracks = serde_json::to_vec(&SpeakerTracks {
            schema_version: super::super::domain::SYSTEM_SCHEMA_VERSION,
            recording_id: "recording".into(),
            geometry: geometry(),
            tracks: Vec::new(),
        })
        .unwrap();
        for (tracks_relative, hypothesis_relative) in [
            (
                PathBuf::from("../outside-tracks.json"),
                PathBuf::from("output.rttm"),
            ),
            (
                PathBuf::from("speaker_tracks.json"),
                PathBuf::from("../outside.rttm"),
            ),
        ] {
            let cache = tempfile::tempdir().unwrap();
            let key = digest_bytes(tracks_relative.to_string_lossy().as_bytes());
            write_self_consistent_entry(
                cache.path(),
                &key,
                &tracks_relative,
                &tracks,
                &hypothesis_relative,
                b"",
            );
            let error = lookup(cache.path(), &key, "recording", "recipe")
                .expect_err("cache traversal was accepted");
            assert!(error.to_string().contains("escapes entry"));
        }
    }

    #[cfg(unix)]
    #[test]
    fn lookup_rejects_speaker_track_symlink() {
        use std::os::unix::fs::symlink;

        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"symlink");
        let tracks = serde_json::to_vec(&SpeakerTracks {
            schema_version: super::super::domain::SYSTEM_SCHEMA_VERSION,
            recording_id: "recording".into(),
            geometry: geometry(),
            tracks: Vec::new(),
        })
        .unwrap();
        let tracks_relative = PathBuf::from("speaker_tracks-link.json");
        write_self_consistent_entry(
            cache.path(),
            &key,
            &tracks_relative,
            &tracks,
            Path::new("output.rttm"),
            b"",
        );
        let entry_root = cache.path().join(key.as_str());
        let outside = cache.path().join("outside-tracks.json");
        fs::write(&outside, &tracks).unwrap();
        fs::remove_file(entry_root.join(&tracks_relative)).unwrap();
        symlink(&outside, entry_root.join(&tracks_relative)).unwrap();
        let error = lookup(cache.path(), &key, "recording", "recipe")
            .expect_err("cache symlink was accepted");
        assert!(error.to_string().contains("symlink"));
    }

    #[cfg(unix)]
    #[test]
    fn lookup_rejects_hypothesis_symlink() {
        use std::os::unix::fs::symlink;

        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"hypothesis-symlink");
        let tracks = serde_json::to_vec(&SpeakerTracks {
            schema_version: super::super::domain::SYSTEM_SCHEMA_VERSION,
            recording_id: "recording".into(),
            geometry: geometry(),
            tracks: Vec::new(),
        })
        .unwrap();
        let hypothesis_relative = PathBuf::from("output-link.rttm");
        write_self_consistent_entry(
            cache.path(),
            &key,
            Path::new("speaker_tracks.json"),
            &tracks,
            &hypothesis_relative,
            b"",
        );
        let entry_root = cache.path().join(key.as_str());
        let outside = cache.path().join("outside-output.rttm");
        fs::write(&outside, b"").unwrap();
        fs::remove_file(entry_root.join(&hypothesis_relative)).unwrap();
        symlink(&outside, entry_root.join(&hypothesis_relative)).unwrap();
        let error = lookup(cache.path(), &key, "recording", "recipe")
            .expect_err("cache symlink was accepted");
        assert!(error.to_string().contains("symlink"));
    }

    fn stage_name(stage: StageKind) -> &'static str {
        match stage {
            StageKind::Bundle => "bundle",
            StageKind::Decode => "decode",
            StageKind::Embedding => "embedding",
            StageKind::Clustering => "clustering",
            StageKind::Reconstruction => "reconstruction",
        }
    }
}
