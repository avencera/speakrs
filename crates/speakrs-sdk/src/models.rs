//! Model downloads, the model cache, and manifests that verify model files

use std::collections::HashSet;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use speakrs::ExecutionMode;
use thiserror::Error;

use crate::cancel::CancelToken;

pub use speakrs::models::{
    MODEL_REPOSITORY as DEFAULT_MODEL_REPOSITORY, MODEL_REVISION as DEFAULT_MODEL_REVISION,
    required_model_files,
};

/// Environment variable that overrides the model cache directory
pub const CACHE_DIR_ENV: &str = "SPEAKRS_CACHE_DIR";

/// Where to find or download the models for one execution mode
#[derive(Debug, Clone)]
pub struct PrepareModelsOptions {
    /// Mode whose files must be present
    pub mode: ExecutionMode,
    /// Use this directory instead of the download cache
    pub model_dir: Option<PathBuf>,
    /// Verify every listed file's size and SHA-256 before returning
    pub manifest: Option<ModelManifest>,
    /// Checked before each download, so cancelling stops before the next file
    pub cancel_token: Option<CancelToken>,
}

impl PrepareModelsOptions {
    /// Download into, or reuse, the cache for `mode`
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            model_dir: None,
            manifest: None,
            cancel_token: None,
        }
    }
}

/// A model directory that holds every file one mode loads
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedModels {
    /// Directory passed to the pipeline builder
    pub model_dir: PathBuf,
    /// Mode the directory was prepared for
    pub mode: ExecutionMode,
    /// Model repository revision, when known
    ///
    /// Set for cache downloads and for manifests; `None` for an unverified local directory
    pub revision: Option<String>,
    /// Manifest the files were verified against, if one was given
    pub manifest: Option<ModelManifest>,
}

/// One cached revision of the model repository
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    /// Hugging Face repository id
    pub repository: String,
    /// Revision (commit hash) of the snapshot
    pub revision: String,
    /// Snapshot directory
    pub path: PathBuf,
}

/// Model downloads and cache housekeeping
///
/// The cache uses the Hugging Face hub layout, so it is shared with
/// `speakrs::ModelManager` and other hub clients that use the same directory
#[derive(Debug, Clone)]
pub struct ModelStore {
    cache_dir: PathBuf,
}

impl Default for ModelStore {
    fn default() -> Self {
        Self::new(Self::default_cache_dir())
    }
}

impl ModelStore {
    /// Use `cache_dir` as the Hugging Face hub cache
    pub fn new(cache_dir: impl Into<PathBuf>) -> Self {
        Self {
            cache_dir: cache_dir.into(),
        }
    }

    /// `$SPEAKRS_CACHE_DIR`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`
    pub fn default_cache_dir() -> PathBuf {
        if let Some(path) = std::env::var_os(CACHE_DIR_ENV) {
            return PathBuf::from(path);
        }

        if let Some(path) = std::env::var_os("HF_HOME") {
            return PathBuf::from(path).join("hub");
        }

        // matches hf_hub::Cache::default, which speakrs::ModelManager uses
        std::env::home_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join(".cache")
            .join("huggingface")
            .join("hub")
    }

    /// The hub cache directory this store reads and writes
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Return a directory that holds every file `options.mode` loads
    ///
    /// Without a `model_dir`, missing files are downloaded into the cache when the
    /// `online` feature is enabled. `on_download` receives each file name before it
    /// is fetched
    pub fn prepare(
        &self,
        options: PrepareModelsOptions,
        on_download: &dyn Fn(&str),
    ) -> Result<PreparedModels, ModelPrepareError> {
        let mode = options.mode;
        let (model_dir, revision) = match options.model_dir {
            Some(model_dir) => {
                let revision = options
                    .manifest
                    .as_ref()
                    .map(|manifest| manifest.revision.clone());
                (model_dir, revision)
            }
            None => (
                self.ensure_snapshot(mode, options.cancel_token.as_ref(), on_download)?,
                Some(DEFAULT_MODEL_REVISION.to_string()),
            ),
        };

        match &options.manifest {
            Some(manifest) => {
                manifest.check_covers(mode)?;
                manifest.verify(&model_dir)?;
            }
            None => {
                for file in required_model_files(mode) {
                    if !model_dir.join(&file).is_file() {
                        return Err(ModelManifestError::MissingFile { path: file }.into());
                    }
                }
            }
        }

        Ok(PreparedModels {
            model_dir,
            mode,
            revision,
            manifest: options.manifest,
        })
    }

    /// List the cached revisions of the model repository
    pub fn list_cache(&self) -> Result<Vec<CacheEntry>, ModelPrepareError> {
        let snapshots = self.repo_dir().join("snapshots");
        let entries = match std::fs::read_dir(&snapshots) {
            Ok(entries) => entries,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(err) => return Err(ModelPrepareError::cache(&snapshots, err)),
        };

        let mut cache_entries = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|err| ModelPrepareError::cache(&snapshots, err))?;
            if !entry.path().is_dir() {
                continue;
            }

            cache_entries.push(CacheEntry {
                repository: DEFAULT_MODEL_REPOSITORY.to_string(),
                revision: entry.file_name().to_string_lossy().into_owned(),
                path: entry.path(),
            });
        }

        cache_entries.sort_by(|lhs, rhs| lhs.revision.cmp(&rhs.revision));
        Ok(cache_entries)
    }

    /// Delete one cached revision and the blobs no other revision uses
    ///
    /// Returns `false` when the revision was not cached
    pub fn cleanup_revision(&self, revision: &str) -> Result<bool, ModelPrepareError> {
        if !is_safe_cache_revision(revision) {
            return Err(ModelPrepareError::InvalidRevision(revision.to_string()));
        }

        let repo_dir = self.repo_dir();
        let snapshot = repo_dir.join("snapshots").join(revision);
        if !snapshot.is_dir() {
            return Ok(false);
        }

        // refs go first so a failure never leaves a ref to a deleted snapshot
        remove_refs_to(&repo_dir.join("refs"), revision)
            .map_err(|err| ModelPrepareError::cache(&repo_dir, err))?;
        match std::fs::remove_dir_all(&snapshot) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(err) => return Err(ModelPrepareError::cache(&snapshot, err)),
        }

        remove_unreferenced_blobs(&repo_dir)
            .map_err(|err| ModelPrepareError::cache(&repo_dir, err))?;
        Ok(true)
    }

    fn repo_dir(&self) -> PathBuf {
        self.cache_dir.join(format!(
            "models--{}",
            DEFAULT_MODEL_REPOSITORY.replace('/', "--")
        ))
    }

    #[cfg(feature = "online")]
    fn ensure_snapshot(
        &self,
        mode: ExecutionMode,
        cancel_token: Option<&CancelToken>,
        on_download: &dyn Fn(&str),
    ) -> Result<PathBuf, ModelPrepareError> {
        use hf_hub::api::sync::ApiBuilder;
        use hf_hub::{Cache, Repo, RepoType};

        let cache = Cache::new(self.cache_dir.clone());
        let mut builder = ApiBuilder::from_cache(cache.clone()).with_progress(false);
        if let Ok(endpoint) = std::env::var("HF_ENDPOINT") {
            builder = builder.with_endpoint(endpoint);
        }

        let api = builder.build().map_err(ModelPrepareError::download)?;
        let repo = Repo::with_revision(
            DEFAULT_MODEL_REPOSITORY.to_string(),
            RepoType::Model,
            DEFAULT_MODEL_REVISION.to_string(),
        );
        let cached = cache.repo(repo.clone());
        let remote = api.repo(repo);
        for file in required_model_files(mode) {
            if cached.get(&file).is_none() {
                on_download(&file);
                if cancel_token.is_some_and(CancelToken::is_cancelled) {
                    return Err(ModelPrepareError::Cancelled);
                }
            }
            remote.get(&file).map_err(ModelPrepareError::download)?;
        }

        // the pinned revision is a commit hash, so it names its own snapshot directory
        Ok(self.snapshot_dir())
    }

    #[cfg(not(feature = "online"))]
    fn ensure_snapshot(
        &self,
        _mode: ExecutionMode,
        _cancel_token: Option<&CancelToken>,
        _on_download: &dyn Fn(&str),
    ) -> Result<PathBuf, ModelPrepareError> {
        let model_dir = self.snapshot_dir();
        if model_dir.is_dir() {
            Ok(model_dir)
        } else {
            Err(ModelPrepareError::MissingCachedSnapshot(model_dir))
        }
    }

    fn snapshot_dir(&self) -> PathBuf {
        self.repo_dir()
            .join("snapshots")
            .join(DEFAULT_MODEL_REVISION)
    }
}

/// Failure to download, locate, or clean up model files
#[derive(Debug, Error)]
pub enum ModelPrepareError {
    /// A model file is missing or does not match the manifest
    #[error(transparent)]
    Manifest(#[from] ModelManifestError),

    /// The Hugging Face download failed
    #[error("failed to download models: {0}")]
    Download(String),

    /// The build has no download support and the cache holds no snapshot
    #[error("model cache `{}` is empty and this build cannot download models", .0.display())]
    MissingCachedSnapshot(PathBuf),

    /// A cache directory could not be read or changed
    #[error("model cache `{}`: {message}", path.display())]
    Cache {
        /// The cache directory or file
        path: PathBuf,
        /// Why it failed
        message: String,
    },

    /// The revision is not a single path component
    #[error("invalid revision `{0}`: expected a single directory name")]
    InvalidRevision(String),

    /// The cancel token was set before a download
    #[error("model download was cancelled")]
    Cancelled,
}

impl ModelPrepareError {
    #[cfg(feature = "online")]
    fn download(error: impl std::fmt::Display) -> Self {
        Self::Download(error.to_string())
    }

    fn cache(path: &Path, error: std::io::Error) -> Self {
        Self::Cache {
            path: path.to_path_buf(),
            message: error.to_string(),
        }
    }
}

/// Sizes and SHA-256 digests of a model directory's files
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Repository the files came from
    pub repository: String,
    /// Repository revision the files came from
    pub revision: String,
    /// One entry per file, sorted by path
    pub files: Vec<ModelManifestEntry>,
}

/// Size and digest of one model file
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifestEntry {
    /// Path relative to the model directory
    pub path: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// Lowercase hex SHA-256 digest
    pub sha256: String,
}

impl ModelManifest {
    /// Hash every file `mode` loads from `model_dir`
    pub fn generate(
        model_dir: impl AsRef<Path>,
        mode: ExecutionMode,
    ) -> Result<Self, ModelManifestError> {
        let model_dir = model_dir.as_ref();
        let mut files = required_model_files(mode)
            .into_iter()
            .map(|path| {
                let absolute = model_dir.join(&path);
                ModelManifestEntry::from_file(path, &absolute)
            })
            .collect::<Result<Vec<_>, _>>()?;
        files.sort_by(|lhs, rhs| lhs.path.cmp(&rhs.path));

        Ok(Self {
            repository: DEFAULT_MODEL_REPOSITORY.to_string(),
            revision: DEFAULT_MODEL_REVISION.to_string(),
            files,
        })
    }

    /// Check that the manifest lists every file `mode` loads
    pub fn check_covers(&self, mode: ExecutionMode) -> Result<(), ModelManifestError> {
        for path in required_model_files(mode) {
            if !self.files.iter().any(|entry| entry.path == path) {
                return Err(ModelManifestError::NotListed { path });
            }
        }

        Ok(())
    }

    /// Check that every listed file exists with the recorded size and digest
    ///
    /// Paths must stay inside `model_dir`, so a manifest cannot point at other files
    pub fn verify(&self, model_dir: impl AsRef<Path>) -> Result<(), ModelManifestError> {
        let model_dir = model_dir.as_ref();
        for expected in &self.files {
            if !is_relative_inside(&expected.path) {
                return Err(ModelManifestError::UnsafePath {
                    path: expected.path.clone(),
                });
            }

            let actual = ModelManifestEntry::from_file(
                expected.path.clone(),
                &model_dir.join(&expected.path),
            )?;
            if actual.size_bytes != expected.size_bytes {
                return Err(ModelManifestError::SizeMismatch {
                    path: expected.path.clone(),
                    expected: expected.size_bytes,
                    actual: actual.size_bytes,
                });
            }
            if actual.sha256 != expected.sha256 {
                return Err(ModelManifestError::ChecksumMismatch {
                    path: expected.path.clone(),
                    expected: expected.sha256.clone(),
                    actual: actual.sha256,
                });
            }
        }

        Ok(())
    }

    /// Serialize as JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest fields always serialize")
    }

    /// Parse a manifest written by [`Self::to_json`]
    pub fn from_json(json: &str) -> Result<Self, ModelManifestError> {
        serde_json::from_str(json).map_err(|err| ModelManifestError::Parse(err.to_string()))
    }
}

impl ModelManifestEntry {
    fn from_file(path: String, absolute_path: &Path) -> Result<Self, ModelManifestError> {
        let read_error = |err: std::io::Error| ModelManifestError::ReadFile {
            path: path.clone(),
            message: err.to_string(),
        };
        let mut file = std::fs::File::open(absolute_path).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                ModelManifestError::MissingFile { path: path.clone() }
            } else {
                read_error(err)
            }
        })?;
        let size_bytes = file.metadata().map_err(read_error)?.len();

        let mut hasher = Sha256::new();
        let mut buffer = vec![0; 256 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(read_error)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        let sha256 = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(Self {
            path,
            size_bytes,
            sha256,
        })
    }
}

/// A model file is missing, unreadable, or differs from its manifest entry
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModelManifestError {
    /// A required file does not exist
    #[error("model file `{path}` is missing")]
    MissingFile {
        /// Path relative to the model directory
        path: String,
    },

    /// A file exists but could not be read
    #[error("failed to read model file `{path}`: {message}")]
    ReadFile {
        /// Path relative to the model directory
        path: String,
        /// Why it failed
        message: String,
    },

    /// A file's size differs from the manifest
    #[error("model file `{path}` has size {actual}, expected {expected}")]
    SizeMismatch {
        /// Path relative to the model directory
        path: String,
        /// Size the manifest lists, in bytes
        expected: u64,
        /// Size on disk, in bytes
        actual: u64,
    },

    /// A file's digest differs from the manifest
    #[error("model file `{path}` has SHA-256 {actual}, expected {expected}")]
    ChecksumMismatch {
        /// Path relative to the model directory
        path: String,
        /// Digest the manifest lists, as lowercase hex
        expected: String,
        /// Digest of the file on disk, as lowercase hex
        actual: String,
    },

    /// Manifest JSON could not be parsed
    #[error("invalid model manifest JSON: {0}")]
    Parse(String),

    /// The manifest omits a file the execution mode loads
    #[error("model manifest does not list `{path}`, which this mode needs")]
    NotListed {
        /// Path relative to the model directory
        path: String,
    },

    /// A manifest path is absolute or leaves the model directory
    #[error("model manifest path `{path}` must be relative and stay inside the model directory")]
    UnsafePath {
        /// The path as the manifest writes it
        path: String,
    },
}

fn is_relative_inside(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && Path::new(path)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn is_safe_cache_revision(revision: &str) -> bool {
    if revision.contains('\\') {
        return false;
    }

    let mut components = Path::new(revision).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

/// Remove refs, including nested ones such as `refs/pr/1`, that point at a deleted snapshot
fn remove_refs_to(refs_dir: &Path, revision: &str) -> std::io::Result<()> {
    let Ok(entries) = std::fs::read_dir(refs_dir) else {
        return Ok(());
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            remove_refs_to(&path, revision)?;
        } else if std::fs::read(&path)?.trim_ascii() == revision.as_bytes() {
            std::fs::remove_file(path)?;
        }
    }

    Ok(())
}

// a download renames its finished blob into place just before linking it from its
// snapshot, so a blob this new may belong to a download still in progress
const BLOB_GRACE_PERIOD: Duration = Duration::from_secs(10 * 60);

/// Snapshots link into `blobs/`, so a blob no remaining snapshot links to is garbage
fn remove_unreferenced_blobs(repo_dir: &Path) -> std::io::Result<()> {
    let mut referenced = HashSet::new();
    // any read error aborts, since a partial scan would make kept blobs look unused
    collect_link_targets(&repo_dir.join("snapshots"), &mut referenced)?;

    let blobs = match std::fs::read_dir(repo_dir.join("blobs")) {
        Ok(blobs) => blobs,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };

    let now = SystemTime::now();
    for blob in blobs {
        let blob = blob?;
        let name = blob.file_name();
        // a download in progress writes `<blob>.part` next to `<blob>.lock`, and
        // finished blobs are bare hashes, so anything with an extension is left alone
        let in_progress = name.to_string_lossy().contains('.');
        if in_progress || referenced.contains(&name) {
            continue;
        }

        let modified = blob.metadata()?.modified()?;
        let recent = now
            .duration_since(modified)
            .is_ok_and(|age| age < BLOB_GRACE_PERIOD);
        if !recent {
            std::fs::remove_file(blob.path())?;
        }
    }

    Ok(())
}

fn collect_link_targets(
    dir: &Path,
    targets: &mut HashSet<std::ffi::OsString>,
) -> std::io::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };

    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            if let Some(name) = std::fs::read_link(entry.path())?.file_name() {
                targets.insert(name.to_owned());
            }
        } else if file_type.is_dir() {
            collect_link_targets(&entry.path(), targets)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_round_trips_and_rejects_changed_files() {
        let dir = temp_dir("manifest");
        for file in required_model_files(ExecutionMode::Cpu) {
            write_file(&dir.join(&file), file.as_bytes());
        }

        let manifest = ModelManifest::generate(&dir, ExecutionMode::Cpu).unwrap();
        let parsed = ModelManifest::from_json(&manifest.to_json()).unwrap();
        assert_eq!(parsed, manifest);
        parsed.check_covers(ExecutionMode::Cpu).unwrap();
        parsed.verify(&dir).unwrap();

        let mut partial = manifest.clone();
        partial.files.pop();
        assert!(matches!(
            partial.check_covers(ExecutionMode::Cpu),
            Err(ModelManifestError::NotListed { .. })
        ));

        for path in ["../outside", "/etc/hosts", "a/../../b", ""] {
            let mut escaping = manifest.clone();
            escaping.files[0].path = path.to_string();
            assert!(
                matches!(
                    escaping.verify(&dir),
                    Err(ModelManifestError::UnsafePath { .. })
                ),
                "{path}"
            );
        }

        let first = &manifest.files[0].path;
        write_file(&dir.join(first), b"changed contents of a different length");
        assert!(matches!(
            manifest.verify(&dir),
            Err(ModelManifestError::SizeMismatch { .. })
        ));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn prepare_reports_the_first_missing_file_of_a_local_directory() {
        let dir = temp_dir("prepare-missing");
        let store = ModelStore::new(dir.join("cache"));
        let mut options = PrepareModelsOptions::new(ExecutionMode::Cpu);
        options.model_dir = Some(dir.clone());

        let err = store.prepare(options, &|_| {}).unwrap_err();

        assert!(matches!(
            err,
            ModelPrepareError::Manifest(ModelManifestError::MissingFile { .. })
        ));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_removes_snapshot_refs_and_only_unshared_blobs() {
        let cache = temp_dir("cleanup");
        let store = ModelStore::new(&cache);
        let repo = store.repo_dir();
        write_file(&repo.join("blobs/shared"), b"shared");
        write_file(&repo.join("blobs/old-only"), b"old");
        write_file(&repo.join("blobs/downloading.part"), b"partial");
        write_file(&repo.join("blobs/just-published"), b"new");
        // the unreferenced blob must predate the grace period to be collected
        std::fs::File::options()
            .write(true)
            .open(repo.join("blobs/old-only"))
            .unwrap()
            .set_modified(SystemTime::now() - 2 * BLOB_GRACE_PERIOD)
            .unwrap();
        write_file(&repo.join("refs/main"), b"old-rev");
        write_file(&repo.join("refs/pr/1"), b"old-rev\n");
        write_file(&repo.join("refs/other"), b"new-rev");
        for (revision, blobs) in [
            ("old-rev", ["shared", "old-only"]),
            ("new-rev", ["shared", "shared"]),
        ] {
            let snapshot = repo.join("snapshots").join(revision);
            std::fs::create_dir_all(snapshot.join("nested")).unwrap();
            std::os::unix::fs::symlink(format!("../../blobs/{}", blobs[0]), snapshot.join("a"))
                .unwrap();
            std::os::unix::fs::symlink(
                format!("../../../blobs/{}", blobs[1]),
                snapshot.join("nested/b"),
            )
            .unwrap();
        }

        let revisions: Vec<_> = store
            .list_cache()
            .unwrap()
            .into_iter()
            .map(|e| e.revision)
            .collect();
        assert_eq!(revisions, ["new-rev", "old-rev"]);

        assert!(store.cleanup_revision("old-rev").unwrap());
        assert!(!store.cleanup_revision("old-rev").unwrap());
        assert!(repo.join("blobs/shared").exists());
        assert!(!repo.join("blobs/old-only").exists());
        assert!(repo.join("blobs/downloading.part").exists());
        assert!(repo.join("blobs/just-published").exists());
        assert!(!repo.join("refs/main").exists());
        assert!(!repo.join("refs/pr/1").exists());
        assert!(repo.join("refs/other").exists());
        assert!(matches!(
            store.cleanup_revision("../escape"),
            Err(ModelPrepareError::InvalidRevision(_))
        ));

        let _ = std::fs::remove_dir_all(cache);
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("speakrs-sdk-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }
}
