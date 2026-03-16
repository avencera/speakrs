mod aishell4;
mod alimeeting;
mod ami;
mod earnings21;
mod voxconverse;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;

#[derive(Clone)]
pub struct Dataset {
    pub id: String,
    pub display_name: String,
    source: Source,
}

#[derive(Clone)]
enum Source {
    VoxConverseDev,
    VoxConverseTest,
    AmiIhm,
    AmiSdm,
    Aishell4,
    Earnings21,
    AliMeeting,
    Hf { repo: String },
}

impl Dataset {
    fn new(id: &str, display_name: &str, source: Source) -> Self {
        Self {
            id: id.to_string(),
            display_name: display_name.to_string(),
            source,
        }
    }

    pub fn dataset_dir(&self, base_dir: &Path) -> PathBuf {
        base_dir.join(&self.id)
    }

    pub fn ensure(&self, base_dir: &Path) -> Result<()> {
        let dir = self.dataset_dir(base_dir);

        // try Tigris first (voxconverse-dev is baked into the Docker image)
        if !matches!(self.source, Source::VoxConverseDev) && S5cmd::try_download(&self.id, &dir)? {
            return Ok(());
        }

        match &self.source {
            Source::VoxConverseDev => voxconverse::ensure_dev(&dir, base_dir),
            Source::VoxConverseTest => voxconverse::ensure_test(&dir),
            Source::AmiIhm => ami::ensure_ihm(&dir),
            Source::AmiSdm => ami::ensure_sdm(&dir, base_dir),
            Source::Aishell4 => aishell4::ensure(&dir),
            Source::Earnings21 => earnings21::ensure(&dir),
            Source::AliMeeting => alimeeting::ensure(&dir),
            Source::Hf { repo } => ensure_hf(&self.display_name, repo, &dir),
        }
    }
}

pub fn all_datasets() -> Vec<Dataset> {
    vec![
        Dataset::new("voxconverse-dev", "VoxConverse Dev", Source::VoxConverseDev),
        Dataset::new(
            "voxconverse-test",
            "VoxConverse Test",
            Source::VoxConverseTest,
        ),
        Dataset::new("ami-ihm", "AMI IHM", Source::AmiIhm),
        Dataset::new("ami-sdm", "AMI SDM", Source::AmiSdm),
        Dataset::new("aishell4", "AISHELL-4", Source::Aishell4),
        Dataset::new("earnings21", "Earnings-21", Source::Earnings21),
        Dataset::new("alimeeting", "AliMeeting", Source::AliMeeting),
        Dataset::new(
            "ava-avd",
            "AVA-AVD",
            Source::Hf {
                repo: "argmaxinc/ava-avd".into(),
            },
        ),
        Dataset::new(
            "callhome",
            "CALLHOME",
            Source::Hf {
                repo: "argmaxinc/callhome".into(),
            },
        ),
        Dataset::new(
            "icsi",
            "ICSI",
            Source::Hf {
                repo: "argmaxinc/icsi-meetings".into(),
            },
        ),
        Dataset::new(
            "msdwild",
            "MSDWILD",
            Source::Hf {
                repo: "argmaxinc/msdwild".into(),
            },
        ),
    ]
}

pub fn find_dataset(id: &str) -> Option<Dataset> {
    all_datasets().into_iter().find(|d| d.id == id)
}

pub fn list_dataset_ids() -> Vec<String> {
    all_datasets().into_iter().map(|d| d.id).collect()
}

// ---------------------------------------------------------------------------
// S5cmd -- parallel S3 downloads from Tigris
// ---------------------------------------------------------------------------

const S3_BUCKET: &str = "s3://speakrs/datasets";

pub struct S5cmd;

impl S5cmd {
    /// Check if s5cmd binary is installed and S3 credentials are configured
    pub fn available() -> bool {
        Command::new("s5cmd")
            .arg("version")
            .output()
            .is_ok_and(|o| o.status.success())
            && std::env::var("AWS_ACCESS_KEY_ID").is_ok()
    }

    /// Copy a dataset from Tigris S3 to a local directory
    pub fn sync(dataset_id: &str, local_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BUCKET}/{dataset_id}/*");
        fs::create_dir_all(local_dir)?;
        run_cmd(
            Command::new("s5cmd")
                .args(["cp", "--concurrency", "20", "--part-size", "25"])
                .arg(&s3_path)
                .arg(local_dir),
        )
    }

    /// Try s5cmd download, return Ok(true) if it worked, Ok(false) to fall back
    fn try_download(dataset_id: &str, dataset_dir: &Path) -> Result<bool> {
        if !Self::available() {
            return Ok(false);
        }

        println!("=== Downloading {dataset_id} via s5cmd from Tigris ===");
        match Self::sync(dataset_id, dataset_dir) {
            Ok(()) => {
                if dataset_dir.join("wav").is_dir() && dataset_dir.join("rttm").is_dir() {
                    println!("{dataset_id}: s5cmd download complete");
                    Ok(true)
                } else {
                    println!("{dataset_id}: s5cmd download incomplete, falling back");
                    Ok(false)
                }
            }
            Err(e) => {
                println!("{dataset_id}: s5cmd failed ({e}), falling back to direct download");
                Ok(false)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HF download fallback for Tigris-sourced datasets
// ---------------------------------------------------------------------------

fn hf_download(repo: &str, local_dir: &Path) -> Result<()> {
    fs::create_dir_all(local_dir)?;
    run_cmd(
        Command::new("uv")
            .args(["tool", "run", "--from", "huggingface-hub", "hf", "download"])
            .arg(repo)
            .args(["--repo-type", "dataset", "--local-dir"])
            .arg(local_dir),
    )
}

fn ensure_hf(display_name: &str, repo: &str, dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");
    if wav_dir.is_dir() && rttm_dir.is_dir() {
        return Ok(());
    }

    println!("=== Downloading {display_name} from HuggingFace ===");
    let tmp_name = repo.replace('/', "-");
    let tmp_dir = std::env::temp_dir().join(format!("{tmp_name}-hf"));
    let _ = fs::remove_dir_all(&tmp_dir);
    hf_download(repo, &tmp_dir)?;

    fs::create_dir_all(&wav_dir)?;
    fs::create_dir_all(&rttm_dir)?;

    for path in walk_files_with_ext(&tmp_dir, "rttm") {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        fs::copy(&path, rttm_dir.join(&name))?;
    }

    for path in walk_files_with_ext(&tmp_dir, "wav") {
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        convert_to_16k_mono(&path, &wav_dir.join(format!("{stem}.wav")))?;
    }

    let _ = fs::remove_dir_all(&tmp_dir);
    println!("{display_name} setup complete");
    Ok(())
}

fn walk_files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();
    walk_recursive(dir, ext, &mut results);
    results.sort();
    results
}

fn walk_recursive(dir: &Path, ext: &str, results: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_recursive(&path, ext, results);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
        {
            results.push(path);
        }
    }
}
