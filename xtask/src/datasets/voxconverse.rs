use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::datasets::Dataset;

pub struct VoxConverseDev;
pub struct VoxConverseTest;

impl Dataset for VoxConverseDev {
    fn id(&self) -> &'static str {
        "voxconverse-dev"
    }

    fn display_name(&self) -> &'static str {
        "VoxConverse Dev"
    }

    fn ensure(&self, base_dir: &Path) -> Result<()> {
        let dir = self.dataset_dir(base_dir);
        // migrate from old fixtures/voxconverse/ location
        let old_dir = base_dir.join("voxconverse");
        if old_dir.is_dir() && !dir.exists() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(&old_dir, &dir)?;
            #[cfg(not(unix))]
            fs::rename(&old_dir, &dir)?;
        }
        ensure_split(&dir, "voxconverse_dev_wav.zip", "dev")
    }
}

impl Dataset for VoxConverseTest {
    fn id(&self) -> &'static str {
        "voxconverse-test"
    }

    fn display_name(&self) -> &'static str {
        "VoxConverse Test"
    }

    fn ensure(&self, base_dir: &Path) -> Result<()> {
        let dir = self.dataset_dir(base_dir);
        ensure_split(&dir, "voxconverse_test_wav.zip", "test")
    }
}

fn ensure_split(dir: &Path, zip_name: &str, rttm_subdir: &str) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if !wav_dir.is_dir() {
        println!("=== Downloading VoxConverse WAVs ({zip_name}) ===");
        fs::create_dir_all(dir)?;
        let zip_path = dir.join(zip_name);
        let url = format!("https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/{zip_name}");
        run_cmd(
            Command::new("curl")
                .args(["--fail", "-L", "-o"])
                .arg(&zip_path)
                .arg(&url),
        )?;
        run_cmd(
            Command::new("unzip")
                .args(["-q"])
                .arg(&zip_path)
                .arg("-d")
                .arg(dir),
        )?;
        // the zip extracts to audio/ — rename to wav/
        let audio_dir = dir.join("audio");
        if audio_dir.is_dir() && !wav_dir.is_dir() {
            fs::rename(&audio_dir, &wav_dir)?;
        }
        let _ = fs::remove_file(&zip_path);
    }

    if !rttm_dir.is_dir() {
        println!("=== Downloading VoxConverse ground truth RTTMs ===");
        let tmp_clone = std::env::temp_dir().join("voxconverse-clone");
        let _ = fs::remove_dir_all(&tmp_clone);
        run_cmd(
            Command::new("git")
                .args([
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/joonson/voxconverse",
                ])
                .arg(&tmp_clone),
        )?;
        fs::create_dir_all(&rttm_dir)?;
        for entry in fs::read_dir(tmp_clone.join(rttm_subdir))? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "rttm") {
                fs::copy(&path, rttm_dir.join(entry.file_name()))?;
            }
        }
        let _ = fs::remove_dir_all(&tmp_clone);
    }

    Ok(())
}
