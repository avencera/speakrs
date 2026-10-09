//! A tune file supplies timing choices, never new execution or accuracy permissions

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::driver_version::DriverRelease;
use super::{ApprovedChoice, Catalogue, LibraryVersions, Tuple};
use crate::inference::cuda::CudaMath;
use crate::inference::cuda::device::DeviceAttributes;

const FORMAT_VERSION: u32 = 3;
const MAX_FILE_BYTES: u64 = 4 << 20;

/// Every identity component must match before any row is used
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceKey {
    pub(super) device_name: String,
    pub(super) capability: [u32; 2],
    pub(super) sm_count: u32,
    pub(super) driver_version: DriverRelease,
    pub(super) libraries: LibraryVersions,
    pub(super) speakrs_version: String,
    pub(super) artifact_version: String,
    pub(super) accuracy_policy: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MathKey {
    Fp32,
    Tf32,
}

impl From<CudaMath> for MathKey {
    fn from(math: CudaMath) -> Self {
        match math {
            CudaMath::Fp32 => Self::Fp32,
            CudaMath::Tf32 => Self::Tf32,
        }
    }
}

impl From<MathKey> for CudaMath {
    fn from(math: MathKey) -> Self {
        match math {
            MathKey::Fp32 => Self::Fp32,
            MathKey::Tf32 => Self::Tf32,
        }
    }
}

/// Text identities are looked up in a trusted catalogue, not parsed into pins
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "implementation", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ChoiceKey {
    Library,
    Kernel { module: String, config_pin: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Entry {
    pub(super) boundary: String,
    pub(super) batch: usize,
    pub(super) math: MathKey,
    pub(super) choice: ChoiceKey,
    pub(super) median_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TuneFile {
    format_version: u32,
    pub(super) key: DeviceKey,
    pub(super) entries: Vec<Entry>,
}

impl TuneFile {
    pub(super) fn new(key: DeviceKey, entries: Vec<Entry>) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            key,
            entries,
        }
    }

    /// Reject the complete file on a mismatch; partial reuse hides stale tuning
    pub(super) fn validate(
        self,
        expected: &DeviceKey,
        catalogue: &Catalogue,
    ) -> Result<ValidatedFile, FileError> {
        if self.format_version != FORMAT_VERSION {
            return Err(FileError::Invalid("unsupported tune-file format".into()));
        }
        if self.key != *expected {
            return Err(FileError::KeyMismatch);
        }
        let mut entries = BTreeMap::new();
        for entry in self.entries {
            let tuple = Tuple::parse(&entry.boundary, entry.batch, entry.math.into())?;
            if !entry.median_ms.is_finite() || entry.median_ms <= 0.0 {
                return Err(FileError::Invalid(
                    "a median must be finite and positive".into(),
                ));
            }
            let choice = catalogue
                .choices(tuple)
                .iter()
                .find(|candidate| candidate.key() == entry.choice)
                .cloned()
                .ok_or_else(|| {
                    FileError::Invalid(format!(
                        "unapproved configuration for {} b{} {:?}",
                        entry.boundary, entry.batch, entry.math
                    ))
                })?;
            if entries.insert(tuple, choice).is_some() {
                return Err(FileError::Invalid(
                    "duplicate boundary, batch and math".into(),
                ));
            }
        }
        Ok(ValidatedFile(entries))
    }

    /// Rename only a complete, flushed JSON document into place
    pub(super) fn write(&self, path: &Path) -> Result<(), FileError> {
        self.write_with_nonce(path, || {
            let mut bytes = [0; 16];
            getrandom::fill(&mut bytes).map_err(std::io::Error::other)?;
            Ok(u128::from_ne_bytes(bytes))
        })
    }

    fn write_with_nonce(
        &self,
        path: &Path,
        mut nonce: impl FnMut() -> Result<u128, std::io::Error>,
    ) -> Result<(), FileError> {
        let parent = path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent)?;
        let name = path
            .file_name()
            .ok_or_else(|| FileError::Invalid("the tune path must name a file".into()))?;
        let (temp, mut output) = loop {
            let temp = parent.join(format!(".{}.{:032x}.tmp", name.to_string_lossy(), nonce()?));
            match Self::open_temp(&temp) {
                Ok(output) => break (temp, output),
                // an existing file can belong to another writer; never remove it
                Err(FileError::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    continue;
                }
                Err(error) => return Err(error),
            }
        };

        let result = (|| -> Result<(), FileError> {
            serde_json::to_writer_pretty(&mut output, self)?;
            output.write_all(b"\n")?;
            output.sync_all()?;
            std::fs::rename(&temp, path)?;
            Ok(())
        })();
        drop(output);
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result
    }

    fn open_temp(path: &Path) -> Result<std::fs::File, FileError> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        Ok(options.open(path)?)
    }
}

/// Only this type may reach runtime selection
#[derive(Debug, Default)]
pub(super) struct ValidatedFile(BTreeMap<Tuple, ApprovedChoice>);

impl ValidatedFile {
    pub(super) fn choice(&self, tuple: Tuple) -> Option<ApprovedChoice> {
        self.0.get(&tuple).cloned()
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum FileError {
    #[error(
        "tune-file device, driver, numerical-library, speakrs, artifact or accuracy-policy key does not match"
    )]
    KeyMismatch,
    #[error("invalid CUDA tune file: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub(super) fn read(path: &Path) -> Result<Option<TuneFile>, FileError> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(FileError::Invalid("file exceeds 4 MiB".into()));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if value
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        != Some(u64::from(FORMAT_VERSION))
    {
        return Err(FileError::Invalid("unsupported tune-file format".into()));
    }
    Ok(Some(serde_json::from_value(value)?))
}

/// The path is stable across upgrades, while the key invalidates old measurements
pub(super) fn path(
    device: &DeviceAttributes,
    override_path: Option<&Path>,
) -> Result<PathBuf, FileError> {
    if let Some(path) = override_path {
        return Ok(path.to_owned());
    }
    if let Some(path) = std::env::var_os("SPEAKRS_CUDA_TUNE_FILE") {
        return Ok(path.into());
    }
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .ok_or_else(|| {
            FileError::Invalid("set SPEAKRS_CUDA_TUNE_FILE or XDG_CONFIG_HOME".into())
        })?;
    let cc = device.capability();
    let identity = format!(
        "{}:{:?}:{}",
        device.name(),
        [cc.major, cc.minor],
        device.multiprocessors().get()
    );
    let hash = super::super::kernels::ArtifactHash::of(identity.as_bytes());
    Ok(root.join("speakrs").join(format!("cuda-tune-{hash}.json")))
}

#[cfg(test)]
mod tests;
