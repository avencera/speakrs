use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};
use color_eyre::eyre::{Result, bail, eyre};
use serde::{Deserialize, Serialize};

use crate::catalog::ImplementationId;
use crate::datasets::DatasetId;

pub const SCHEMA_VERSION: u32 = 2;

/// Collision-safe suite identity
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunId(String);

impl RunId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Identity passed to reports; never inferred from a path
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunIdentity {
    pub run_id: RunId,
    pub schema_version: u32,
    pub started_at: String,
    pub host: String,
    pub description: Option<String>,
}

/// One collision-safe benchmark suite created before dataset work
#[derive(Clone, Debug)]
pub struct BenchmarkRun {
    pub identity: RunIdentity,
    pub root: PathBuf,
    pub implementations: Vec<ImplementationId>,
    pub datasets: Vec<DatasetId>,
}

impl BenchmarkRun {
    pub fn create(
        benchmarks_root: &Path,
        implementations: Vec<ImplementationId>,
        datasets: Vec<DatasetId>,
        description: Option<String>,
        now: DateTime<Local>,
        host: String,
    ) -> Result<Self> {
        fs::create_dir_all(benchmarks_root)?;
        let stamp = now.format("%Y%m%d-%H%M%S").to_string();
        let mut suffix = 0u32;
        let root = loop {
            let name = if suffix == 0 {
                stamp.clone()
            } else {
                format!("{stamp}-{suffix}")
            };
            let candidate = benchmarks_root.join(&name);
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    suffix += 1;
                    if suffix > 1000 {
                        bail!("could not allocate a unique run directory under {stamp}");
                    }
                }
                Err(error) => return Err(error.into()),
            }
        };
        let run_id = RunId(
            root.file_name()
                .ok_or_else(|| eyre!("run directory missing name"))?
                .to_string_lossy()
                .into_owned(),
        );
        let identity = RunIdentity {
            run_id,
            schema_version: SCHEMA_VERSION,
            started_at: now.to_rfc3339(),
            host,
            description,
        };
        Ok(Self {
            identity,
            root,
            implementations,
            datasets,
        })
    }

    pub fn dataset_dir(&self, dataset: DatasetId) -> PathBuf {
        if self.datasets.len() > 1 {
            self.root.join(dataset.as_str())
        } else {
            self.root.clone()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResultV2 {
    pub schema_version: u32,
    pub run: RunIdentity,
    pub dataset: String,
    pub implementations: serde_json::Value,
}

/// Convert an old flat or nested result directory beside the unchanged source
pub fn convert_legacy_results(source: &Path, destination: &Path) -> Result<BenchmarkResultV2> {
    let (run_id, dataset) = detect_legacy_layout(source)?;
    let identity = RunIdentity {
        run_id,
        schema_version: SCHEMA_VERSION,
        started_at: String::new(),
        host: "converted".to_owned(),
        description: Some("converted from schema 1".to_owned()),
    };
    let converted = BenchmarkResultV2 {
        schema_version: SCHEMA_VERSION,
        run: identity,
        dataset,
        implementations: serde_json::json!({}),
    };
    let payload = serde_json::to_string_pretty(&converted)? + "\n";
    fs::write(destination, payload)?;
    Ok(converted)
}

fn detect_legacy_layout(source: &Path) -> Result<(RunId, String)> {
    let name = source
        .file_name()
        .ok_or_else(|| eyre!("legacy result path missing name"))?
        .to_string_lossy()
        .into_owned();
    if looks_like_timestamp(&name) {
        return Ok((RunId(name), "unknown".to_owned()));
    }
    let parent = source
        .parent()
        .and_then(Path::file_name)
        .map(|component| component.to_string_lossy().into_owned())
        .ok_or_else(|| eyre!("nested legacy result missing parent timestamp"))?;
    if !looks_like_timestamp(&parent) {
        bail!("cannot derive nested timestamp from {}", source.display());
    }
    Ok((RunId(parent), name))
}

fn looks_like_timestamp(name: &str) -> bool {
    name.len() >= 15 && name.as_bytes().get(8) == Some(&b'-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::ImplementationId;
    use crate::datasets::DatasetId;

    #[test]
    fn same_time_runs_do_not_collide() {
        let dir = tempfile::tempdir().unwrap();
        let now = Local::now();
        let first = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            now,
            "host".to_owned(),
        )
        .unwrap();
        let second = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev],
            None,
            now,
            "host".to_owned(),
        )
        .unwrap();
        assert_ne!(first.identity.run_id, second.identity.run_id);
        assert!(first.root.exists());
        assert!(second.root.exists());
    }

    #[test]
    fn multi_dataset_run_has_one_suite_and_dataset_children() {
        let dir = tempfile::tempdir().unwrap();
        let run = BenchmarkRun::create(
            dir.path(),
            vec![ImplementationId::SpeakrsCpu],
            vec![DatasetId::VoxconverseDev, DatasetId::AmiIhm],
            None,
            Local::now(),
            "host".to_owned(),
        )
        .unwrap();
        assert_eq!(
            run.dataset_dir(DatasetId::VoxconverseDev),
            run.root.join("voxconverse-dev")
        );
        assert_eq!(run.dataset_dir(DatasetId::AmiIhm), run.root.join("ami-ihm"));
    }

    #[test]
    fn nested_legacy_layout_takes_timestamp_from_parent() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("20240101-010203").join("voxconverse-dev");
        fs::create_dir_all(&nested).unwrap();
        let dest = dir.path().join("converted.json");
        let converted = convert_legacy_results(&nested, &dest).unwrap();
        assert_eq!(converted.run.run_id.as_str(), "20240101-010203");
        assert_eq!(converted.dataset, "voxconverse-dev");
        assert_eq!(converted.schema_version, 2);
        let original = fs::read_dir(nested.parent().unwrap())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .count();
        assert!(original >= 1);
        assert!(dest.exists());
    }

    #[test]
    fn flat_legacy_layout_uses_directory_name() {
        let dir = tempfile::tempdir().unwrap();
        let flat = dir.path().join("20240101-010203");
        fs::create_dir_all(&flat).unwrap();
        let dest = dir.path().join("flat-v2.json");
        let converted = convert_legacy_results(&flat, &dest).unwrap();
        assert_eq!(converted.run.run_id.as_str(), "20240101-010203");
        assert_eq!(converted.dataset, "unknown");
    }
}
