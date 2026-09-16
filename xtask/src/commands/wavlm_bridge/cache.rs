use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

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
    pub embedding_snapshot: ArtifactRef,
    pub speaker_tracks: ArtifactRef,
    pub hypothesis: ArtifactRef,
}

/// Files loaded from a verified cache entry
#[derive(Debug)]
pub struct CacheHit {
    pub key: Sha256Digest,
    pub stage_receipts: Vec<(PathBuf, Vec<u8>, ArtifactRef)>,
    pub embedding_snapshot: (Vec<u8>, ArtifactRef),
    pub speaker_tracks: (Vec<u8>, ArtifactRef),
    pub hypothesis: (Vec<u8>, ArtifactRef),
}

pub(crate) struct CachePublication<'a> {
    pub(crate) stage_receipts: &'a [(String, Vec<u8>)],
    pub(crate) embedding_snapshot: &'a [u8],
    pub(crate) speaker_tracks: &'a [u8],
    pub(crate) hypothesis: &'a [u8],
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
    if !completion_marker_present(&marker_path, "cache completion marker")? {
        return Ok(None);
    }
    ensure_regular_file(&entry_path, "cache entry manifest")?;
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
    let embedding = documents
        .get(&StageKind::Embedding)
        .expect("checked stage order");
    let snapshot_path = member_path(&entry_root, &entry.embedding_snapshot.relative_path)?;
    let snapshot_bytes = read_bounded_member(
        &snapshot_path,
        &entry.embedding_snapshot,
        super::domain::MAX_EMBEDDING_STAGE_BYTES,
    )?;
    let snapshot_document: EmbeddingStageDocument = serde_json::from_slice(&snapshot_bytes)
        .wrap_err_with(|| {
            format!(
                "invalid cached embedding snapshot {}",
                snapshot_path.display()
            )
        })?;
    snapshot_document.validate()?;
    ensure!(
        snapshot_document.recording_id == entry.recording_id
            && snapshot_document.stage_key == embedding.receipt.cache_key,
        "cached embedding snapshot identity does not match cache entry"
    );
    ensure!(
        snapshot_document.geometry == embedding.receipt.geometry,
        "cached embedding snapshot geometry does not match embedding receipt"
    );
    ensure!(
        embedding
            .receipt
            .outputs
            .contains(&entry.embedding_snapshot),
        "cached embedding receipt does not name the snapshot"
    );
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
        embedding_snapshot: (snapshot_bytes, entry.embedding_snapshot),
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
    if !completion_marker_present(&marker_path, "embedding cache completion marker")? {
        return Ok(None);
    }
    ensure_regular_file(&entry_path, "embedding cache entry manifest")?;
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
    publication: &CachePublication<'_>,
) -> Result<CacheEntry> {
    fs::create_dir_all(cache_root)
        .wrap_err_with(|| format!("failed to create cache root {}", cache_root.display()))?;
    ensure_directory(cache_root, "cache root")?;
    let entry_root = cache_root.join(key.as_str());
    let staging = tempfile::tempdir_in(cache_root).wrap_err_with(|| {
        format!(
            "failed to create cache staging directory in {}",
            cache_root.display()
        )
    })?;
    let staging_root = staging.path();
    fs::create_dir(staging_root.join("stages"))?;
    let mut receipt_artifacts = Vec::with_capacity(publication.stage_receipts.len());
    for (name, bytes) in publication.stage_receipts {
        let relative = PathBuf::from("stages").join(name);
        let path = staging_root.join(&relative);
        write_new(&path, bytes)?;
        receipt_artifacts.push(artifact_for(staging_root, &relative)?);
    }
    let recording_relative = run_relative(recipe_id, recording_id);
    let snapshot_relative = recording_relative.join("stages/embedding_snapshot.json");
    write_new(
        &staging_root.join(&snapshot_relative),
        publication.embedding_snapshot,
    )?;
    let snapshot_artifact = artifact_for(staging_root, &snapshot_relative)?;
    let tracks_relative = recording_relative.join("speaker_tracks.json");
    write_new(
        &staging_root.join(&tracks_relative),
        publication.speaker_tracks,
    )?;
    let tracks_artifact = artifact_for(staging_root, &tracks_relative)?;
    let hypothesis_relative = recording_relative.join("output.rttm");
    write_new(
        &staging_root.join(&hypothesis_relative),
        publication.hypothesis,
    )?;
    let hypothesis_artifact = artifact_for(staging_root, &hypothesis_relative)?;
    let entry = CacheEntry {
        schema_version: CACHE_SCHEMA_VERSION,
        cache_key: key.clone(),
        recording_id: recording_id.to_owned(),
        recipe_id: recipe_id.to_owned(),
        stage_receipts: receipt_artifacts,
        embedding_snapshot: snapshot_artifact,
        speaker_tracks: tracks_artifact,
        hypothesis: hypothesis_artifact,
    };
    let entry_bytes = serde_json::to_vec_pretty(&entry)?;
    write_new(&staging_root.join("entry.json"), &entry_bytes)?;
    write_new(
        &staging_root.join(".complete"),
        format!("{}\n", digest_bytes(&entry_bytes)).as_bytes(),
    )?;
    make_tree_read_only(staging_root)?;
    publish_or_reuse(
        staging,
        &entry_root,
        "cache entry",
        || existing_entry(cache_root, key, recording_id, recipe_id),
        entry,
    )
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
        "embedding snapshot is {} bytes, maximum is {} bytes",
        snapshot.len(),
        super::domain::MAX_EMBEDDING_STAGE_BYTES
    );
    fs::create_dir_all(cache_root)
        .wrap_err_with(|| format!("failed to create cache root {}", cache_root.display()))?;
    ensure_directory(cache_root, "cache root")?;
    let embedding_root = cache_root.join("embedding");
    fs::create_dir_all(&embedding_root)?;
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
    write_new(
        &staging_root.join(".complete"),
        format!("{}\n", digest_bytes(&entry_bytes)).as_bytes(),
    )?;
    make_tree_read_only(staging_root)?;
    publish_or_reuse(
        staging,
        &entry_root,
        "embedding cache entry",
        || existing_embedding_entry(cache_root, key, recording_id),
        entry,
    )
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

fn run_relative(recipe_id: &str, recording_id: &str) -> PathBuf {
    PathBuf::from("recipes")
        .join(recipe_id)
        .join("recordings")
        .join(recording_id)
}

fn publish_or_reuse<T: PartialEq>(
    staging: tempfile::TempDir,
    destination: &Path,
    label: &str,
    existing: impl Fn() -> Result<Option<T>>,
    entry: T,
) -> Result<T> {
    let staging_path = staging.path().to_owned();
    let result = (|| -> Result<T> {
        match atomic_publish_no_replace(&staging_path, destination) {
            Ok(()) => Ok(entry),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                if let Some(existing) = existing()? {
                    ensure!(
                        existing == entry,
                        "{label} {} conflicts with the requested contents",
                        destination.display()
                    );
                    return Ok(existing);
                }
                if let Err(error) = remove_partial(destination) {
                    if destination.join(".complete").exists()
                        && let Some(existing) = existing()?
                    {
                        ensure!(
                            existing == entry,
                            "{label} {} conflicts with the requested contents",
                            destination.display()
                        );
                        return Ok(existing);
                    }
                    return Err(error);
                }
                match atomic_publish_no_replace(&staging_path, destination) {
                    Ok(()) => Ok(entry),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        let existing = existing()?.ok_or_else(|| {
                            color_eyre::eyre::eyre!(
                                "{label} {} was concurrently published without a complete marker",
                                destination.display()
                            )
                        })?;
                        ensure!(
                            existing == entry,
                            "{label} {} conflicts with the requested contents",
                            destination.display()
                        );
                        Ok(existing)
                    }
                    Err(error) => Err(error).wrap_err_with(|| {
                        format!(
                            "failed to publish {label} {} without replacement",
                            destination.display()
                        )
                    }),
                }
            }
            Err(error) => Err(error).wrap_err_with(|| {
                format!(
                    "failed to publish {label} {} without replacement",
                    destination.display()
                )
            }),
        }
    })();
    let cleanup = cleanup_staging(staging);
    match cleanup {
        Ok(()) => result,
        Err(cleanup_error) => match result {
            Ok(_) => Err(cleanup_error),
            Err(error) => Err(error).wrap_err_with(|| {
                format!(
                    "failed to clean up cache staging directory {}: {cleanup_error}",
                    staging_path.display()
                )
            }),
        },
    }
}

fn existing_entry(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
    recipe_id: &str,
) -> Result<Option<CacheEntry>> {
    let entry_root = cache_root.join(key.as_str());
    if !entry_root.exists() {
        return Ok(None);
    }
    if !entry_root.join(".complete").exists() {
        return Ok(None);
    }
    ensure!(
        lookup(cache_root, key, recording_id, recipe_id)?.is_some(),
        "cache entry disappeared during validation"
    );
    let bytes = fs::read(entry_root.join("entry.json"))?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

fn existing_embedding_entry(
    cache_root: &Path,
    key: &Sha256Digest,
    recording_id: &str,
) -> Result<Option<EmbeddingCacheEntry>> {
    let entry_root = cache_root.join("embedding").join(key.as_str());
    if !entry_root.exists() {
        return Ok(None);
    }
    if !entry_root.join(".complete").exists() {
        return Ok(None);
    }
    ensure!(
        lookup_embedding(cache_root, key, recording_id)?.is_some(),
        "embedding cache entry disappeared during validation"
    );
    let bytes = fs::read(entry_root.join("entry.json"))?;
    Ok(Some(serde_json::from_slice(&bytes)?))
}

fn completion_marker_present(path: &Path, label: &str) -> Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(_) => {
            ensure_regular_file(path, label)?;
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).wrap_err_with(|| format!("failed to inspect {label} {}", path.display()))
        }
    }
}

fn cleanup_staging(staging: tempfile::TempDir) -> Result<()> {
    let staging_path = staging.path().to_owned();
    match fs::symlink_metadata(&staging_path) {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(error).wrap_err_with(|| {
                format!(
                    "failed to inspect cache staging directory {}",
                    staging_path.display()
                )
            });
        }
    }
    make_tree_writable(&staging_path)?;
    staging.close().wrap_err_with(|| {
        format!(
            "failed to remove cache staging directory {}",
            staging_path.display()
        )
    })
}

fn make_tree_writable(root: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(root)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "cache staging tree contains a symlink: {}",
        root.display()
    );
    let mut permissions = metadata.permissions();
    #[cfg(unix)]
    permissions.set_mode(permissions.mode() | 0o200);
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    fs::set_permissions(root, permissions)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(root)? {
            make_tree_writable(&entry?.path())?;
        }
    }
    Ok(())
}

fn remove_partial(destination: &Path) -> Result<()> {
    if !destination.exists() {
        return Ok(());
    }
    let metadata = match fs::symlink_metadata(destination) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    ensure!(
        !metadata.file_type().is_symlink(),
        "refusing to remove symlink cache entry {}",
        destination.display()
    );
    ensure!(
        !destination.join(".complete").exists(),
        "cache entry {} has a completion marker but failed validation",
        destination.display()
    );
    if let Err(error) = fs::remove_dir_all(destination)
        && error.kind() != io::ErrorKind::NotFound
    {
        return Err(error).wrap_err_with(|| {
            format!(
                "failed to remove partial cache entry {}",
                destination.display()
            )
        });
    }
    Ok(())
}

fn atomic_publish_no_replace(source: &Path, destination: &Path) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        let source = CString::new(source.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
        let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
        })?;
        let result =
            // SAFETY: both pointers reference NUL-terminated paths owned by these CStrings
            unsafe { libc::renamex_np(source.as_ptr(), destination.as_ptr(), libc::RENAME_EXCL) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    #[cfg(target_os = "linux")]
    {
        let source = CString::new(source.as_os_str().as_bytes())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "source path contains NUL"))?;
        let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "destination path contains NUL")
        })?;
        // SAFETY: both pointers reference NUL-terminated paths owned by these CStrings
        let result = unsafe {
            libc::renameat2(
                libc::AT_FDCWD,
                source.as_ptr(),
                libc::AT_FDCWD,
                destination.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        if destination.exists() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, destination));
        }
        fs::rename(source, destination)
    }
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
        AvailabilityCounts, EmbeddingStageDocument, GeometryReceipt, PackedSegmentationMask,
        RationalReceipt, StageDependency, StageReceipt,
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
        snapshot: &ArtifactRef,
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
            let outputs = match stage {
                StageKind::Embedding => vec![snapshot.clone()],
                StageKind::Reconstruction => vec![tracks.clone(), hypothesis.clone()],
                _ => Vec::new(),
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
        let snapshot_bytes = serde_json::to_vec(&EmbeddingStageDocument {
            schema_version: super::super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: digest_bytes(b"Embedding"),
            geometry: geometry(),
            segmentation_shape: [0, 0, 0],
            segmentation_values: PackedSegmentationMask::empty(),
            entries: Vec::new(),
            embedding_receipt: AvailabilityCounts::default(),
        })
        .unwrap();
        let snapshot_relative =
            PathBuf::from("recipes/recipe/recordings/recording/stages/embedding_snapshot.json");
        let snapshot_artifact = ArtifactRef {
            relative_path: snapshot_relative.clone(),
            sha256: digest_bytes(&snapshot_bytes),
            bytes: snapshot_bytes.len() as u64,
        };
        for (name, bytes) in stage_documents(
            key,
            &snapshot_artifact,
            &tracks_artifact,
            &hypothesis_artifact,
        ) {
            let path = entry_root.join("stages").join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, bytes).unwrap();
        }
        let snapshot_path = entry_root.join(&snapshot_relative);
        fs::create_dir_all(snapshot_path.parent().unwrap()).unwrap();
        fs::write(snapshot_path, &snapshot_bytes).unwrap();
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
            embedding_snapshot: snapshot_artifact,
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
            segmentation_values: PackedSegmentationMask::empty(),
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
        let partial_root = cache.path().join("embedding").join(key.as_str());
        fs::create_dir_all(&partial_root).unwrap();
        fs::write(partial_root.join("partial"), b"interrupted").unwrap();
        publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot).unwrap();
        let hit = lookup_embedding(cache.path(), &key, "recording")
            .unwrap()
            .expect("published embedding cache should be reusable");
        assert_eq!(hit.key, key);
        assert_eq!(hit.snapshot.0, snapshot);
        assert!(
            fs::metadata(cache.path().join("embedding").join(key.as_str()))
                .unwrap()
                .permissions()
                .readonly()
        );
        assert!(
            publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot).is_ok()
        );
        let conflicting_snapshot = [snapshot.as_slice(), b"conflict"].concat();
        assert!(
            publish_embedding(
                cache.path(),
                &key,
                "recording",
                &stage_receipts,
                &conflicting_snapshot
            )
            .is_err()
        );
    }

    #[test]
    fn concurrent_embedding_publications_reuse_one_complete_tree() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"concurrent-embedding-stage");
        let snapshot_document = EmbeddingStageDocument {
            schema_version: super::super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: key.clone(),
            geometry: geometry(),
            segmentation_shape: [0, 0, 0],
            segmentation_values: PackedSegmentationMask::empty(),
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

        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot)
                    .is_ok()
            });
            let second = scope.spawn(|| {
                publish_embedding(cache.path(), &key, "recording", &stage_receipts, &snapshot)
                    .is_ok()
            });
            assert!(first.join().unwrap());
            assert!(second.join().unwrap());
        });

        assert!(
            lookup_embedding(cache.path(), &key, "recording")
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn losing_publication_removes_read_only_staging_tree() {
        let cache = tempfile::tempdir().unwrap();
        let destination = cache.path().join("destination");
        fs::create_dir(&destination).unwrap();
        fs::write(destination.join("winner"), b"winner").unwrap();

        let staging = tempfile::tempdir_in(cache.path()).unwrap();
        let staging_path = staging.path().to_owned();
        fs::write(staging.path().join("staged"), b"staged").unwrap();
        make_tree_read_only(staging.path()).unwrap();

        let result = publish_or_reuse(
            staging,
            &destination,
            "test cache",
            || Ok(Some("winner")),
            "winner",
        )
        .unwrap();

        assert_eq!(result, "winner");
        assert!(!staging_path.exists());
        assert_eq!(fs::read(destination.join("winner")).unwrap(), b"winner");
    }

    #[test]
    fn publication_error_removes_read_only_staging_tree() {
        let cache = tempfile::tempdir().unwrap();
        let destination = cache.path().join("missing").join("destination");
        let staging = tempfile::tempdir_in(cache.path()).unwrap();
        let staging_path = staging.path().to_owned();
        fs::write(staging.path().join("staged"), b"staged").unwrap();
        make_tree_read_only(staging.path()).unwrap();

        let error = publish_or_reuse(staging, &destination, "test cache", || Ok(None), ())
            .expect_err("publication into a missing parent should fail");

        assert!(error.to_string().contains("without replacement"));
        assert!(!staging_path.exists());
        assert!(!destination.exists());
    }

    #[test]
    fn embedding_cache_treats_interrupted_entry_as_miss() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"interrupted-embedding-stage");
        let entry_root = cache.path().join("embedding").join(key.as_str());
        fs::create_dir_all(&entry_root).unwrap();
        fs::write(entry_root.join("entry.json"), b"{}").unwrap();
        assert!(
            lookup_embedding(cache.path(), &key, "recording")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn embedding_cache_rejects_previous_cache_schema() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"previous-embedding-cache-schema");
        let snapshot = serde_json::to_vec(&EmbeddingStageDocument {
            schema_version: super::super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: key.clone(),
            geometry: geometry(),
            segmentation_shape: [0, 0, 0],
            segmentation_values: PackedSegmentationMask::empty(),
            entries: Vec::new(),
            embedding_receipt: AvailabilityCounts::default(),
        })
        .unwrap();
        let snapshot_artifact = ArtifactRef {
            relative_path: "embedding_snapshot.json".into(),
            sha256: digest_bytes(&snapshot),
            bytes: snapshot.len() as u64,
        };
        let stage_receipts = embedding_stage_documents(&key, &snapshot_artifact);
        let entry_root = cache.path().join("embedding").join(key.as_str());
        fs::create_dir_all(entry_root.join("stages")).unwrap();
        let stage_artifacts = stage_receipts
            .iter()
            .map(|(name, bytes)| {
                let relative_path = PathBuf::from("stages").join(name);
                fs::write(entry_root.join(&relative_path), bytes).unwrap();
                ArtifactRef {
                    relative_path,
                    sha256: digest_bytes(bytes),
                    bytes: bytes.len() as u64,
                }
            })
            .collect::<Vec<_>>();
        fs::write(entry_root.join("embedding_snapshot.json"), &snapshot).unwrap();
        let snapshot_artifact = ArtifactRef {
            relative_path: "embedding_snapshot.json".into(),
            sha256: digest_bytes(&snapshot),
            bytes: snapshot.len() as u64,
        };
        let entry = EmbeddingCacheEntry {
            schema_version: EMBEDDING_CACHE_SCHEMA_VERSION - 1,
            cache_key: key.clone(),
            recording_id: "recording".into(),
            stage_receipts: stage_artifacts,
            snapshot: snapshot_artifact,
        };
        let entry_bytes = serde_json::to_vec(&entry).unwrap();
        fs::write(entry_root.join("entry.json"), &entry_bytes).unwrap();
        fs::write(
            entry_root.join(".complete"),
            format!("{}\n", digest_bytes(&entry_bytes)),
        )
        .unwrap();

        let error = lookup_embedding(cache.path(), &key, "recording")
            .expect_err("previous embedding cache schema was reused");
        assert!(
            error
                .to_string()
                .contains("unsupported embedding cache schema")
        );
    }

    #[test]
    fn completed_embedding_entry_with_invalid_manifest_fails_closed() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"completed-invalid-embedding");
        let entry_root = cache.path().join("embedding").join(key.as_str());
        fs::create_dir_all(&entry_root).unwrap();
        let bytes = b"{}";
        fs::write(entry_root.join("entry.json"), bytes).unwrap();
        fs::write(
            entry_root.join(".complete"),
            format!("{}\n", digest_bytes(bytes)),
        )
        .unwrap();

        let error = lookup_embedding(cache.path(), &key, "recording")
            .expect_err("completed invalid embedding entry was accepted");
        assert!(error.to_string().contains("invalid embedding cache entry"));
    }

    #[test]
    fn completed_entry_with_invalid_manifest_fails_closed() {
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
            embedding_snapshot: ArtifactRef {
                relative_path: "recipes/recipe/recordings/recording/stages/embedding_snapshot.json"
                    .into(),
                sha256: digest_bytes(b""),
                bytes: 0,
            },
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
    fn lookup_treats_interrupted_entry_as_miss() {
        let cache = tempfile::tempdir().unwrap();
        let key = digest_bytes(b"interrupted-cache-entry");
        let entry_root = cache.path().join(key.as_str());
        fs::create_dir_all(&entry_root).unwrap();
        fs::write(entry_root.join("entry.json"), b"{}").unwrap();

        assert!(
            lookup(cache.path(), &key, "recording", "recipe")
                .unwrap()
                .is_none()
        );
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
        let snapshot = serde_json::to_vec(&EmbeddingStageDocument {
            schema_version: super::super::domain::EMBEDDING_STAGE_SCHEMA_VERSION,
            recording_id: "recording".into(),
            stage_key: digest_bytes(b"Embedding"),
            geometry: geometry(),
            segmentation_shape: [0, 0, 0],
            segmentation_values: PackedSegmentationMask::empty(),
            entries: Vec::new(),
            embedding_receipt: AvailabilityCounts::default(),
        })
        .unwrap();
        let snapshot_artifact = ArtifactRef {
            relative_path: "recipes/recipe/recordings/recording/stages/embedding_snapshot.json"
                .into(),
            sha256: digest_bytes(&snapshot),
            bytes: snapshot.len() as u64,
        };
        let tracks_relative =
            PathBuf::from("recipes/recipe/recordings/recording/speaker_tracks.json");
        let hypothesis_relative = PathBuf::from("recipes/recipe/recordings/recording/output.rttm");
        let tracks_artifact = ArtifactRef {
            relative_path: tracks_relative,
            sha256: digest_bytes(&tracks),
            bytes: tracks.len() as u64,
        };
        let hypothesis_artifact = ArtifactRef {
            relative_path: hypothesis_relative,
            sha256: digest_bytes(b""),
            bytes: 0,
        };
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
            let outputs = match stage {
                StageKind::Embedding => vec![snapshot_artifact.clone()],
                StageKind::Reconstruction => {
                    vec![tracks_artifact.clone(), hypothesis_artifact.clone()]
                }
                _ => Vec::new(),
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
        let partial_root = cache.path().join(key.as_str());
        fs::create_dir_all(&partial_root).unwrap();
        fs::write(partial_root.join("partial"), b"interrupted").unwrap();
        let publication = CachePublication {
            stage_receipts: &documents,
            embedding_snapshot: &snapshot,
            speaker_tracks: &tracks,
            hypothesis: b"",
        };
        std::thread::scope(|scope| {
            let first =
                scope.spawn(|| publish(cache.path(), &key, "recording", "recipe", &publication));
            let second =
                scope.spawn(|| publish(cache.path(), &key, "recording", "recipe", &publication));
            assert!(first.join().unwrap().is_ok());
            assert!(second.join().unwrap().is_ok());
        });
        assert!(
            fs::metadata(cache.path().join(key.as_str()))
                .unwrap()
                .permissions()
                .readonly()
        );
        let hit = lookup(cache.path(), &key, "recording", "recipe")
            .unwrap()
            .expect("published cache should be reusable");
        assert_eq!(hit.embedding_snapshot.0, snapshot);
        assert!(
            publish(cache.path(), &key, "recording", "recipe", &publication,).is_ok(),
            "valid cache publication should be reusable"
        );
        let conflicting_snapshot = [snapshot.as_slice(), b"conflict"].concat();
        let conflicting_publication = CachePublication {
            embedding_snapshot: &conflicting_snapshot,
            ..publication
        };
        assert!(
            publish(
                cache.path(),
                &key,
                "recording",
                "recipe",
                &conflicting_publication,
            )
            .is_err()
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
