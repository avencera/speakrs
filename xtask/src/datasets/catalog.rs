use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::eyre::{Result, bail, eyre};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};

/// Name of the completion manifest stored beside each published dataset
pub const DATASET_MANIFEST_FILE: &str = ".speakrs-dataset.json";
const DATASET_MANIFEST_VERSION: u32 = 1;

/// Canonical dataset identity
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DatasetId {
    VoxconverseDev,
    VoxconverseTest,
    AmiIhm,
    AmiSdm,
    Aishell4,
    Earnings21,
    Alimeeting,
    AvaAvd,
    Icsi,
}

impl DatasetId {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::VoxconverseDev => "voxconverse-dev",
            Self::VoxconverseTest => "voxconverse-test",
            Self::AmiIhm => "ami-ihm",
            Self::AmiSdm => "ami-sdm",
            Self::Aishell4 => "aishell4",
            Self::Earnings21 => "earnings21",
            Self::Alimeeting => "alimeeting",
            Self::AvaAvd => "ava-avd",
            Self::Icsi => "icsi",
        }
    }

    pub fn parse_cli(name: &str) -> Option<Self> {
        DatasetCatalog::parse_cli(name).map(|spec| spec.id)
    }

    pub fn parse(value: &str) -> Option<Self> {
        DATASETS
            .iter()
            .find(|spec| spec.id.as_str() == value)
            .map(|spec| spec.id)
    }
}

impl Serialize for DatasetId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DatasetId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown dataset id {value}")))
    }
}

/// One catalog dataset
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DatasetSpec {
    pub id: DatasetId,
    pub aliases: &'static [&'static str],
    pub display_name: &'static str,
}

const DATASETS: &[DatasetSpec] = &[
    DatasetSpec {
        id: DatasetId::VoxconverseDev,
        aliases: &["vd", "vox-dev"],
        display_name: "VoxConverse Dev",
    },
    DatasetSpec {
        id: DatasetId::VoxconverseTest,
        aliases: &["vt", "vox-test"],
        display_name: "VoxConverse Test",
    },
    DatasetSpec {
        id: DatasetId::AmiIhm,
        aliases: &["ai", "ami-i"],
        display_name: "AMI IHM",
    },
    DatasetSpec {
        id: DatasetId::AmiSdm,
        aliases: &["as", "ami-s"],
        display_name: "AMI SDM",
    },
    DatasetSpec {
        id: DatasetId::Aishell4,
        aliases: &["a4", "aishell"],
        display_name: "AISHELL-4",
    },
    DatasetSpec {
        id: DatasetId::Earnings21,
        aliases: &["e21", "earnings"],
        display_name: "Earnings-21",
    },
    DatasetSpec {
        id: DatasetId::Alimeeting,
        aliases: &["ali", "alimeet"],
        display_name: "AliMeeting",
    },
    DatasetSpec {
        id: DatasetId::AvaAvd,
        aliases: &["ava"],
        display_name: "AVA-AVD",
    },
    DatasetSpec {
        id: DatasetId::Icsi,
        aliases: &[],
        display_name: "ICSI",
    },
];

/// Static dataset catalog
pub struct DatasetCatalog;

impl DatasetCatalog {
    pub fn all() -> &'static [DatasetSpec] {
        DATASETS
    }

    pub fn parse_cli(name: &str) -> Option<&'static DatasetSpec> {
        DATASETS
            .iter()
            .find(|spec| spec.id.as_str() == name || spec.aliases.contains(&name))
    }

    pub fn resolve(name: &str) -> Result<&'static DatasetSpec> {
        Self::parse_cli(name).ok_or_else(|| {
            eyre!("unknown dataset: {name}. Use --dataset list to see available datasets")
        })
    }
}

/// One validated WAV/RTTM pair in a published snapshot
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetFile {
    file_id: String,
    wav: PathBuf,
    rttm: PathBuf,
    duration_seconds_millis: u64,
}

impl DatasetFile {
    /// Return the stable file identity used by the dataset and RTTM rows
    pub fn file_id(&self) -> &str {
        &self.file_id
    }

    /// Return the validated WAV path
    pub fn wav(&self) -> &Path {
        &self.wav
    }

    /// Return the validated RTTM path
    pub fn rttm(&self) -> &Path {
        &self.rttm
    }

    /// Return the WAV duration recorded in the completion manifest
    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds_millis as f64 / 1000.0
    }
}

/// Non-empty validated dataset snapshot consumed by benchmark selection
#[derive(Clone, Debug)]
pub struct DatasetSnapshot {
    dataset: DatasetId,
    files: Box<[DatasetFile]>,
    source_provenance: String,
}

impl DatasetSnapshot {
    /// Return the canonical dataset identity
    pub fn dataset(&self) -> DatasetId {
        self.dataset
    }

    /// Return the immutable inventory owned by this checked snapshot
    pub fn files(&self) -> &[DatasetFile] {
        &self.files
    }

    /// Return the source evidence recorded when this snapshot was published
    pub fn source_provenance(&self) -> &str {
        &self.source_provenance
    }

    /// Validate an acquired directory before its completion manifest is written
    pub(crate) fn validate_staged(
        dataset: DatasetId,
        dir: &Path,
        source_provenance: &str,
    ) -> Result<Self> {
        let files = scan_paired_directory(dataset, dir)?;
        let source_provenance = non_empty_provenance(source_provenance)?;
        Ok(Self {
            dataset,
            files: files.into_boxed_slice(),
            source_provenance,
        })
    }

    /// Write the completion manifest after all staged files pass validation
    pub(crate) fn write_manifest(&self, dir: &Path) -> Result<()> {
        let manifest = DatasetManifest::from_snapshot(self, dir)?;
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        let temp_path = unique_sibling_path(dir, DATASET_MANIFEST_FILE, "tmp");
        let write_result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            let manifest_path = dir.join(DATASET_MANIFEST_FILE);
            if manifest_path.exists() {
                fs::remove_file(&manifest_path)?;
            }
            fs::rename(&temp_path, manifest_path)?;
            Ok(())
        })();
        if write_result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        write_result
    }

    /// Read and validate a published directory and its completion manifest
    pub fn from_paired_directory(dataset: DatasetId, dir: &Path) -> Result<Self> {
        Self::from_paired_directory_with_hashes(dataset, dir, false)
    }

    /// Read and verify a published directory and its completion manifest
    pub fn verify_paired_directory(dataset: DatasetId, dir: &Path) -> Result<Self> {
        Self::from_paired_directory_with_hashes(dataset, dir, true)
    }

    fn from_paired_directory_with_hashes(
        dataset: DatasetId,
        dir: &Path,
        verify_hashes: bool,
    ) -> Result<Self> {
        let manifest_path = dir.join(DATASET_MANIFEST_FILE);
        let manifest: DatasetManifest = serde_json::from_reader(File::open(&manifest_path)?)
            .map_err(|error| {
                eyre!(
                    "dataset {} completion manifest is invalid at {}: {error}",
                    dataset.as_str(),
                    manifest_path.display()
                )
            })?;
        let files = scan_paired_directory(dataset, dir)?;
        manifest.validate(dataset, dir, &files, verify_hashes)?;
        Ok(Self {
            dataset,
            files: files.into_boxed_slice(),
            source_provenance: manifest.source_provenance,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DatasetManifest {
    schema_version: u32,
    dataset: DatasetId,
    source_provenance: String,
    files: Vec<DatasetManifestFile>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct DatasetManifestFile {
    file_id: String,
    wav: String,
    rttm: String,
    duration_seconds_millis: u64,
    wav_bytes: u64,
    #[serde(default)]
    wav_modified_nanos: Option<u128>,
    wav_sha256: String,
    rttm_bytes: u64,
    #[serde(default)]
    rttm_modified_nanos: Option<u128>,
    rttm_sha256: String,
}

impl DatasetManifest {
    fn from_snapshot(snapshot: &DatasetSnapshot, dir: &Path) -> Result<Self> {
        let mut files = Vec::with_capacity(snapshot.files.len());
        for file in snapshot.files() {
            files.push(DatasetManifestFile::from_file(file, dir)?);
        }
        Ok(Self {
            schema_version: DATASET_MANIFEST_VERSION,
            dataset: snapshot.dataset,
            source_provenance: snapshot.source_provenance.clone(),
            files,
        })
    }

    fn validate(
        &self,
        dataset: DatasetId,
        dir: &Path,
        files: &[DatasetFile],
        verify_hashes: bool,
    ) -> Result<()> {
        if self.schema_version != DATASET_MANIFEST_VERSION {
            bail!(
                "dataset {} has unsupported completion manifest version {}",
                dataset.as_str(),
                self.schema_version
            );
        }
        if self.dataset != dataset {
            bail!(
                "dataset completion manifest identifies {}, expected {}",
                self.dataset.as_str(),
                dataset.as_str()
            );
        }
        non_empty_provenance(&self.source_provenance)?;
        if self.files.len() != files.len() {
            bail!(
                "dataset {} completion inventory has {} files, installed directory has {}",
                dataset.as_str(),
                self.files.len(),
                files.len()
            );
        }

        let mut manifest_files = HashMap::with_capacity(self.files.len());
        for file in &self.files {
            if manifest_files.insert(file.file_id.as_str(), file).is_some() {
                bail!(
                    "dataset {} completion inventory repeats file {}",
                    dataset.as_str(),
                    file.file_id
                );
            }
        }

        for file in files {
            let manifest_file = manifest_files.get(file.file_id.as_str()).ok_or_else(|| {
                eyre!(
                    "dataset {} completion inventory is missing file {}",
                    dataset.as_str(),
                    file.file_id
                )
            })?;
            manifest_file.validate(file, dir, verify_hashes)?;
        }
        Ok(())
    }
}

impl DatasetManifestFile {
    fn from_file(file: &DatasetFile, dir: &Path) -> Result<Self> {
        let wav_metadata = file_metadata(&file.wav)?;
        let rttm_metadata = file_metadata(&file.rttm)?;
        Ok(Self {
            file_id: file.file_id.clone(),
            wav: relative_file_name(dir, "wav", &file.wav)?,
            rttm: relative_file_name(dir, "rttm", &file.rttm)?,
            duration_seconds_millis: file.duration_seconds_millis,
            wav_bytes: wav_metadata.bytes,
            wav_modified_nanos: wav_metadata.modified_nanos,
            wav_sha256: sha256_file(&file.wav)?,
            rttm_bytes: rttm_metadata.bytes,
            rttm_modified_nanos: rttm_metadata.modified_nanos,
            rttm_sha256: sha256_file(&file.rttm)?,
        })
    }

    fn validate(&self, file: &DatasetFile, dir: &Path, verify_hashes: bool) -> Result<()> {
        let expected_wav = relative_file_name(dir, "wav", &file.wav)?;
        let expected_rttm = relative_file_name(dir, "rttm", &file.rttm)?;
        if self.file_id != file.file_id
            || self.wav != expected_wav
            || self.rttm != expected_rttm
            || self.duration_seconds_millis != file.duration_seconds_millis
        {
            bail!(
                "dataset completion inventory does not match file {}",
                file.file_id
            );
        }
        let wav_metadata = file_metadata(&file.wav)?;
        let rttm_metadata = file_metadata(&file.rttm)?;
        let wav_metadata_matches = self.wav_modified_nanos.is_some()
            && self.wav_bytes == wav_metadata.bytes
            && self.wav_modified_nanos == wav_metadata.modified_nanos;
        let rttm_metadata_matches = self.rttm_modified_nanos.is_some()
            && self.rttm_bytes == rttm_metadata.bytes
            && self.rttm_modified_nanos == rttm_metadata.modified_nanos;

        if (verify_hashes || !wav_metadata_matches)
            && (self.wav_bytes != wav_metadata.bytes || self.wav_sha256 != sha256_file(&file.wav)?)
        {
            bail!(
                "dataset WAV changed after publication: {}",
                file.wav.display()
            );
        }
        if (verify_hashes || !rttm_metadata_matches)
            && (self.rttm_bytes != rttm_metadata.bytes
                || self.rttm_sha256 != sha256_file(&file.rttm)?)
        {
            bail!(
                "dataset RTTM changed after publication: {}",
                file.rttm.display()
            );
        }
        Ok(())
    }
}

fn scan_paired_directory(dataset: DatasetId, dir: &Path) -> Result<Vec<DatasetFile>> {
    if !dir.is_dir() {
        bail!(
            "dataset {} installation directory is missing: {}",
            dataset.as_str(),
            dir.display()
        );
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let file_type = entry.file_type()?;
        let expected_type = match name.to_str() {
            Some("wav") | Some("rttm") => ("directory", file_type.is_dir()),
            Some(DATASET_MANIFEST_FILE) => ("file", file_type.is_file()),
            _ => {
                bail!(
                    "dataset {} contains an unexpected installation entry: {}",
                    dataset.as_str(),
                    entry.path().display()
                );
            }
        };
        if !expected_type.1 {
            bail!(
                "dataset {} has a non-regular {} installation entry: {}",
                dataset.as_str(),
                expected_type.0,
                entry.path().display()
            );
        }
    }

    let wav_files = collect_audio_files(dataset, &dir.join("wav"), "wav")?;
    let rttm_files = collect_audio_files(dataset, &dir.join("rttm"), "rttm")?;
    if wav_files.is_empty() && rttm_files.is_empty() {
        bail!(
            "dataset {} installation has no WAV/RTTM pairs under {}",
            dataset.as_str(),
            dir.display()
        );
    }

    let missing_rttm: Vec<_> = wav_files
        .keys()
        .filter(|file_id| !rttm_files.contains_key(*file_id))
        .cloned()
        .collect();
    let missing_wav: Vec<_> = rttm_files
        .keys()
        .filter(|file_id| !wav_files.contains_key(*file_id))
        .cloned()
        .collect();
    if !missing_rttm.is_empty() || !missing_wav.is_empty() {
        bail!(
            "dataset {} has unmatched files (missing RTTM: {}; missing WAV: {})",
            dataset.as_str(),
            if missing_rttm.is_empty() {
                "none".to_string()
            } else {
                missing_rttm.join(", ")
            },
            if missing_wav.is_empty() {
                "none".to_string()
            } else {
                missing_wav.join(", ")
            }
        );
    }

    let mut files = Vec::with_capacity(wav_files.len());
    for (file_id, wav) in wav_files {
        let rttm = rttm_files
            .get(&file_id)
            .expect("matching RTTM was checked above")
            .clone();
        let duration = crate::cmd::wav_duration_seconds(&wav).map_err(|error| {
            eyre!(
                "dataset {} has unreadable WAV {}: {error}",
                dataset.as_str(),
                wav.display()
            )
        })?;
        if !duration.is_finite() || duration <= 0.0 {
            bail!(
                "dataset {} has non-positive WAV duration: {}",
                dataset.as_str(),
                wav.display()
            );
        }
        let duration_millis = duration * 1000.0;
        if !duration_millis.is_finite() || duration_millis > u64::MAX as f64 {
            bail!(
                "dataset {} has an invalid WAV duration: {}",
                dataset.as_str(),
                wav.display()
            );
        }
        validate_rttm(&rttm, &file_id, dataset)?;
        files.push(DatasetFile {
            file_id,
            wav,
            rttm,
            duration_seconds_millis: duration_millis.round().max(1.0) as u64,
        });
    }
    Ok(files)
}

fn collect_audio_files(
    dataset: DatasetId,
    dir: &Path,
    extension: &str,
) -> Result<BTreeMap<String, PathBuf>> {
    if !dir.is_dir() {
        bail!(
            "dataset {} is missing {} directory under {}",
            dataset.as_str(),
            dir.file_name().unwrap_or_default().to_string_lossy(),
            dir.parent().unwrap_or(dir).display()
        );
    }

    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if matches!(entry.file_name().to_str(), Some(".DS_Store" | "__MACOSX")) {
            continue;
        }
        let path = entry.path();
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            bail!(
                "dataset {} contains a non-regular file under {}: {}",
                dataset.as_str(),
                dir.display(),
                path.display()
            );
        }
        if !path
            .extension()
            .is_some_and(|value| value.eq_ignore_ascii_case(extension))
        {
            bail!(
                "dataset {} contains an unexpected file under {}: {}",
                dataset.as_str(),
                dir.display(),
                path.display()
            );
        }
        let file_id = crate::path::file_stem_string(&path)?;
        if file_id.is_empty() {
            bail!(
                "dataset {} contains an empty file identity",
                dataset.as_str()
            );
        }
        if files.insert(file_id.clone(), path).is_some() {
            bail!(
                "dataset {} contains duplicate file identity {} under {}",
                dataset.as_str(),
                file_id,
                dir.display()
            );
        }
    }
    Ok(files)
}

fn validate_rttm(path: &Path, file_id: &str, dataset: DatasetId) -> Result<()> {
    let content = fs::read_to_string(path).map_err(|error| {
        eyre!(
            "dataset {} has unreadable RTTM {}: {error}",
            dataset.as_str(),
            path.display()
        )
    })?;
    for (line_number, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(";;") {
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 10 || fields[0] != "SPEAKER" || fields[1] != file_id {
            bail!(
                "dataset {} has invalid RTTM {} at line {}",
                dataset.as_str(),
                path.display(),
                line_number + 1
            );
        }
        let start = fields[3].parse::<f64>().map_err(|_| {
            eyre!(
                "dataset {} has invalid RTTM start time at {}:{}",
                dataset.as_str(),
                path.display(),
                line_number + 1
            )
        })?;
        let duration = fields[4].parse::<f64>().map_err(|_| {
            eyre!(
                "dataset {} has invalid RTTM duration at {}:{}",
                dataset.as_str(),
                path.display(),
                line_number + 1
            )
        })?;
        if !start.is_finite()
            || start < 0.0
            || !duration.is_finite()
            || duration <= 0.0
            || !(start + duration).is_finite()
        {
            bail!(
                "dataset {} has invalid RTTM interval at {}:{}",
                dataset.as_str(),
                path.display(),
                line_number + 1
            );
        }
    }
    Ok(())
}

fn relative_file_name(dir: &Path, subdir: &str, path: &Path) -> Result<String> {
    let expected_dir = dir.join(subdir);
    if path.parent() != Some(expected_dir.as_path()) {
        bail!(
            "dataset file is outside its expected directory: {}",
            path.display()
        );
    }
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        bail!("dataset file has an invalid name: {}", path.display());
    };
    Ok(format!("{subdir}/{name}"))
}

#[derive(Clone, Copy)]
struct FileMetadata {
    bytes: u64,
    modified_nanos: Option<u128>,
}

fn file_metadata(path: &Path) -> Result<FileMetadata> {
    let metadata = fs::metadata(path)?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos());
    Ok(FileMetadata {
        bytes: metadata.len(),
        modified_nanos,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn non_empty_provenance(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty() {
        bail!("dataset source provenance is empty");
    }
    Ok(value.to_owned())
}

fn unique_sibling_path(dir: &Path, name: &str, suffix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    dir.join(format!(
        ".{name}.{suffix}-{}-{timestamp}",
        std::process::id()
    ))
}

/// Select the lexicographically first far-field recording for an AliMeeting annotation
pub fn select_alimeeting_far_field<'a>(
    recordings: &'a [PathBuf],
    annotation_stem: &str,
) -> Option<&'a PathBuf> {
    let mut matches: Vec<&PathBuf> = recordings
        .iter()
        .filter(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| {
                    stem == annotation_stem || stem.starts_with(&format!("{annotation_stem}_"))
                })
        })
        .collect();
    matches.sort_by_key(|path| path.file_name().map(|name| name.to_os_string()));
    matches.first().copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn write_test_wav(path: &Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..160 {
            writer.write_sample(0_i16).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn dataset_ids_and_aliases_are_unique() {
        let mut names = HashSet::new();
        for spec in DatasetCatalog::all() {
            assert!(names.insert(spec.id.as_str()));
            for alias in spec.aliases {
                assert!(names.insert(alias), "duplicate alias {alias}");
            }
        }
    }

    #[test]
    fn aliases_serialize_to_canonical_ids() {
        assert_eq!(
            DatasetCatalog::parse_cli("vd").unwrap().id.as_str(),
            "voxconverse-dev"
        );
        assert_eq!(
            DatasetCatalog::parse_cli("ali").unwrap().id,
            DatasetId::Alimeeting
        );
    }

    #[test]
    fn empty_snapshot_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("wav")).unwrap();
        std::fs::create_dir_all(dir.path().join("rttm")).unwrap();
        let error =
            DatasetSnapshot::validate_staged(DatasetId::Alimeeting, dir.path(), "test source")
                .unwrap_err()
                .to_string();
        assert!(error.contains("no WAV/RTTM pairs"), "{error}");
    }

    #[test]
    fn alimeeting_selects_lexicographically_first_far_field() {
        let recordings = vec![
            PathBuf::from("R8001_M8004_MS802.wav"),
            PathBuf::from("R8001_M8004_MS801.wav"),
            PathBuf::from("R8002_M8005_MS801.wav"),
        ];
        let selected = select_alimeeting_far_field(&recordings, "R8001_M8004").unwrap();
        assert_eq!(selected.file_name().unwrap(), "R8001_M8004_MS801.wav");
    }

    #[test]
    fn snapshot_rejects_unmatched_stems() {
        let dir = tempfile::tempdir().unwrap();
        let wav_dir = dir.path().join("wav");
        let rttm_dir = dir.path().join("rttm");
        std::fs::create_dir_all(&wav_dir).unwrap();
        std::fs::create_dir_all(&rttm_dir).unwrap();
        std::fs::write(wav_dir.join("only.wav"), b"not-a-wav").unwrap();
        let error =
            DatasetSnapshot::validate_staged(DatasetId::Aishell4, dir.path(), "test source")
                .unwrap_err()
                .to_string();
        assert!(error.contains("only"), "{error}");
    }

    #[test]
    fn empty_rttm_is_valid_for_a_zero_reference_file() {
        let dir = tempfile::tempdir().unwrap();
        let wav_dir = dir.path().join("wav");
        let rttm_dir = dir.path().join("rttm");
        std::fs::create_dir_all(&wav_dir).unwrap();
        std::fs::create_dir_all(&rttm_dir).unwrap();
        write_test_wav(&wav_dir.join("silent.wav"));
        std::fs::write(rttm_dir.join("silent.rttm"), b"").unwrap();

        let snapshot =
            DatasetSnapshot::validate_staged(DatasetId::Aishell4, dir.path(), "test source")
                .unwrap();
        assert_eq!(snapshot.files().len(), 1);
    }

    #[test]
    fn paired_wav_and_rttm_pass_validation() {
        let dir = tempfile::tempdir().unwrap();
        let wav_dir = dir.path().join("wav");
        let rttm_dir = dir.path().join("rttm");
        std::fs::create_dir_all(&wav_dir).unwrap();
        std::fs::create_dir_all(&rttm_dir).unwrap();
        write_test_wav(&wav_dir.join("recording.wav"));
        std::fs::write(
            rttm_dir.join("recording.rttm"),
            b"SPEAKER recording 1 0.000 0.010 <NA> <NA> speaker <NA> <NA>\n",
        )
        .unwrap();

        let snapshot =
            DatasetSnapshot::validate_staged(DatasetId::Aishell4, dir.path(), "test source")
                .unwrap();
        assert_eq!(snapshot.files().len(), 1);
        assert!(
            snapshot
                .files()
                .iter()
                .all(|file| file.duration_seconds() > 0.0)
        );
    }

    #[test]
    fn archive_noise_is_ignored_but_dataset_files_are_validated() {
        let dir = tempfile::tempdir().unwrap();
        let wav_dir = dir.path().join("wav");
        let rttm_dir = dir.path().join("rttm");
        std::fs::create_dir_all(wav_dir.join("__MACOSX")).unwrap();
        std::fs::create_dir_all(&rttm_dir).unwrap();
        write_test_wav(&wav_dir.join("recording.wav"));
        std::fs::write(rttm_dir.join("recording.rttm"), b"").unwrap();
        std::fs::write(wav_dir.join(".DS_Store"), b"archive metadata").unwrap();
        std::fs::write(rttm_dir.join(".DS_Store"), b"archive metadata").unwrap();
        std::fs::write(wav_dir.join("__MACOSX/.DS_Store"), b"archive metadata").unwrap();

        let snapshot =
            DatasetSnapshot::validate_staged(DatasetId::Aishell4, dir.path(), "test source")
                .unwrap();
        assert_eq!(snapshot.files().len(), 1);
    }
}
