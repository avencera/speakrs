use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, eyre};

use super::Dataset;

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
    pub file_id: String,
    pub wav: PathBuf,
    pub rttm: PathBuf,
    pub duration_seconds_millis: u64,
}

impl DatasetFile {
    pub fn duration_seconds(&self) -> f64 {
        self.duration_seconds_millis as f64 / 1000.0
    }
}

/// Non-empty validated dataset snapshot consumed by benchmark selection
#[derive(Clone, Debug)]
pub struct DatasetSnapshot {
    pub dataset: DatasetId,
    pub files: Vec<DatasetFile>,
    pub source_provenance: Option<String>,
}

impl DatasetSnapshot {
    pub fn new(
        dataset: DatasetId,
        files: Vec<DatasetFile>,
        source_provenance: Option<String>,
    ) -> Result<Self> {
        if files.is_empty() {
            bail!(
                "dataset {} snapshot is empty; installation did not publish usable files",
                dataset.as_str()
            );
        }
        Ok(Self {
            dataset,
            files,
            source_provenance,
        })
    }

    pub fn from_paired_directory(dataset: DatasetId, dir: &Path) -> Result<Self> {
        let wav_dir = dir.join("wav");
        let rttm_dir = dir.join("rttm");
        let mut files = Vec::new();
        let mut unmatched = Vec::new();
        let mut unreadable = Vec::new();

        if !wav_dir.is_dir() || !rttm_dir.is_dir() {
            bail!(
                "dataset {} is missing wav/ or rttm/ under {}",
                dataset.as_str(),
                dir.display()
            );
        }

        let mut entries: Vec<_> = std::fs::read_dir(&wav_dir)?
            .filter_map(|entry| entry.ok())
            .collect();
        entries.sort_by_key(|entry| entry.file_name());

        for entry in entries {
            let wav = entry.path();
            if !wav
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
            {
                continue;
            }
            let stem = crate::path::file_stem_string(&wav)?;
            let rttm = rttm_dir.join(format!("{stem}.rttm"));
            if !rttm.exists() {
                unmatched.push(stem);
                continue;
            }
            match crate::cmd::wav_duration_seconds(&wav) {
                Ok(duration) if duration > 0.0 => files.push(DatasetFile {
                    file_id: stem,
                    wav,
                    rttm,
                    duration_seconds_millis: (duration * 1000.0).round() as u64,
                }),
                Ok(_) => unreadable.push(format!("{} (non-positive duration)", wav.display())),
                Err(error) => unreadable.push(format!("{} ({error})", wav.display())),
            }
        }

        if !unmatched.is_empty() {
            bail!(
                "dataset {} has WAV files without matching RTTM: {}",
                dataset.as_str(),
                unmatched.join(", ")
            );
        }
        if !unreadable.is_empty() {
            bail!(
                "dataset {} has unreadable WAV files: {}",
                dataset.as_str(),
                unreadable.join(", ")
            );
        }
        Self::new(dataset, files, None)
    }
}

impl Dataset {
    pub fn catalog_id(&self) -> Result<DatasetId> {
        DatasetId::parse_cli(&self.id)
            .ok_or_else(|| eyre!("dataset {} is not in the catalog", self.id))
    }

    pub fn snapshot(&self, base_dir: &Path) -> Result<DatasetSnapshot> {
        DatasetSnapshot::from_paired_directory(self.catalog_id()?, &self.dataset_dir(base_dir))
    }
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
        assert!(DatasetSnapshot::new(DatasetId::Alimeeting, Vec::new(), None).is_err());
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
        let error = DatasetSnapshot::from_paired_directory(DatasetId::Aishell4, dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("only"), "{error}");
    }
}
