use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, ensure};

use crate::cmd::run_cmd;
use crate::datasets::{DatasetId, DatasetSnapshot};

pub fn ensure_dev(dir: &Path, base_dir: &Path) -> Result<()> {
    let old_dir = base_dir.join("voxconverse");
    if !dir.join("wav").exists() && !dir.join("rttm").exists() {
        migrate_verified_legacy_directory(&old_dir, dir)?;
    }
    ensure_split(dir, "voxconverse_dev_wav.zip", "dev")
}

pub(super) fn migrate_verified_legacy_directory(source: &Path, destination: &Path) -> Result<bool> {
    if DatasetSnapshot::from_paired_directory(DatasetId::VoxconverseDev, source).is_err() {
        return Ok(false);
    }

    copy_directory(source, destination)?;
    Ok(true)
}

pub fn ensure_test(dir: &Path) -> Result<()> {
    ensure_split(dir, "voxconverse_test_wav.zip", "test")
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
        let audio_dir = dir.join("audio");
        let split_audio_dir = dir.join(zip_name.trim_end_matches(".zip"));
        let extracted_audio_dir = [audio_dir, split_audio_dir]
            .into_iter()
            .find(|path| path.is_dir());
        if let Some(extracted_audio_dir) = extracted_audio_dir
            && !wav_dir.is_dir()
        {
            fs::rename(extracted_audio_dir, &wav_dir)?;
        }
        let _ = fs::remove_dir_all(dir.join("__MACOSX"));
        let _ = fs::remove_file(&zip_path);
    }

    if !rttm_dir.is_dir() {
        println!("=== Downloading VoxConverse ground truth RTTMs ===");
        let tmp_clone = dir.join(".voxconverse-clone");
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

    ensure!(
        wav_dir.is_dir() && rttm_dir.is_dir(),
        "VoxConverse {rttm_subdir} extraction did not create wav and rttm directories"
    );

    Ok(())
}

fn copy_directory(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if source_path.is_dir() {
            copy_directory(&source_path, &destination_path)?;
        } else {
            fs::copy(source_path, destination_path)?;
        }
    }
    Ok(())
}
