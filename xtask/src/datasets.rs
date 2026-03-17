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
            "icsi",
            "ICSI",
            Source::Hf {
                repo: "argmaxinc/icsi-meetings".into(),
            },
        ),
    ]
}

pub fn find_dataset(id: &str) -> Option<Dataset> {
    let id = resolve_alias(id);
    all_datasets().into_iter().find(|d| d.id == id)
}

fn resolve_alias(id: &str) -> &str {
    match id {
        "vd" | "vox-dev" => "voxconverse-dev",
        "vt" | "vox-test" => "voxconverse-test",
        "ai" | "ami-i" => "ami-ihm",
        "as" | "ami-s" => "ami-sdm",
        "a4" | "aishell" => "aishell4",
        "e21" | "earnings" => "earnings21",
        "ali" | "alimeet" => "alimeeting",
        "ava" => "ava-avd",
        other => other,
    }
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

    /// Upload a local dataset directory to Tigris S3
    pub fn upload(dataset_id: &str, local_dir: &Path) -> Result<()> {
        let s3_path = format!("{S3_BUCKET}/{dataset_id}/");
        run_cmd(
            Command::new("s5cmd")
                .args(["sync", "--concurrency", "20", "--part-size", "25"])
                .arg(format!("{}/*", local_dir.display()))
                .arg(&s3_path),
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
        let has_wavs = fs::read_dir(&wav_dir).is_ok_and(|mut d| d.next().is_some());
        let has_rttms = fs::read_dir(&rttm_dir).is_ok_and(|mut d| d.next().is_some());
        if has_wavs && has_rttms {
            return Ok(());
        }
    }

    println!("=== Downloading {display_name} from HuggingFace ===");
    let tmp_name = repo.replace('/', "-");
    let tmp_dir = std::env::temp_dir().join(format!("{tmp_name}-hf"));
    let _ = fs::remove_dir_all(&tmp_dir);
    hf_download(repo, &tmp_dir)?;

    // HF datasets use parquet format with embedded audio
    let parquet_dir = tmp_dir.join("data");
    let extract_script = crate::cmd::project_root().join("scripts/extract_hf_dataset.py");

    println!("Extracting parquet to wav + rttm...");
    run_cmd(
        Command::new("uv")
            .args(["run", "--script"])
            .arg(&extract_script)
            .arg(&parquet_dir)
            .arg(&wav_dir)
            .arg(&rttm_dir)
            .args(["--split", "test"]),
    )?;

    let _ = fs::remove_dir_all(&tmp_dir);
    println!("{display_name} setup complete");
    Ok(())
}
