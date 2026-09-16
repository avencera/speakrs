use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, ensure, eyre};

use crate::cmd::run_cmd;
use crate::datasets::{DatasetId, DatasetSnapshot};

pub(super) const VOXCONVERSE_DEV_FILE_COUNT: usize = 216;

pub fn ensure_dev(dir: &Path, base_dir: &Path, cache: &Path) -> Result<()> {
    let old_dir = base_dir.join("voxconverse");
    if !dir.join("wav").exists() && !dir.join("rttm").exists() {
        migrate_verified_legacy_directory(&old_dir, dir)?;
    }
    ensure_split(dir, cache, "voxconverse_dev_wav.zip", "dev")
}

pub(super) fn migrate_verified_legacy_directory(source: &Path, destination: &Path) -> Result<bool> {
    let Ok(snapshot) = DatasetSnapshot::validate_staged(
        DatasetId::VoxconverseDev,
        source,
        "legacy VoxConverse installation",
    ) else {
        return Ok(false);
    };
    if snapshot.files().len() != VOXCONVERSE_DEV_FILE_COUNT {
        return Ok(false);
    }

    copy_directory(source, destination)?;
    Ok(true)
}

pub fn ensure_test(dir: &Path, cache: &Path) -> Result<()> {
    ensure_split(dir, cache, "voxconverse_test_wav.zip", "test")
}

fn ensure_split(dir: &Path, cache: &Path, zip_name: &str, rttm_subdir: &str) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if !wav_dir.is_dir() {
        println!("=== Downloading VoxConverse WAVs ({zip_name}) ===");
        fs::create_dir_all(dir)?;
        let archive_dir = cache.join(zip_name.trim_end_matches(".zip"));
        let zip_path = cache.join(zip_name);
        let audio_dir = archive_dir.join("audio");
        let split_audio_dir = archive_dir.join(zip_name.trim_end_matches(".zip"));
        let completion_marker = archive_dir.join(".complete");
        if !completion_marker.is_file() {
            if archive_dir.exists() {
                fs::remove_dir_all(&archive_dir)?;
            }
            if !zip_path.exists() {
                let url =
                    format!("https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/{zip_name}");
                let partial_path = cache.join(format!("{zip_name}.part"));
                if partial_path.exists() {
                    fs::remove_file(&partial_path)?;
                }
                let download_result = run_cmd(
                    Command::new("curl")
                        .args(["--fail", "-L", "-o"])
                        .arg(&partial_path)
                        .arg(&url),
                );
                if let Err(error) = download_result {
                    if partial_path.exists()
                        && let Err(cleanup_error) = fs::remove_file(&partial_path)
                    {
                        return Err(color_eyre::eyre::eyre!(
                            "VoxConverse {rttm_subdir} archive download failed ({error}); partial cleanup failed ({cleanup_error})"
                        ));
                    }
                    return Err(error);
                }
                fs::rename(partial_path, &zip_path)?;
            }

            let extracting_dir =
                cache.join(format!(".{}.extracting", zip_name.trim_end_matches(".zip")));
            if extracting_dir.exists() {
                fs::remove_dir_all(&extracting_dir)?;
            }
            fs::create_dir_all(&extracting_dir)?;
            let extraction_result = run_cmd(
                Command::new("unzip")
                    .args(["-q"])
                    .arg(&zip_path)
                    .arg("-d")
                    .arg(&extracting_dir),
            );
            if let Err(error) = extraction_result {
                if let Err(cleanup_error) = fs::remove_dir_all(&extracting_dir) {
                    return Err(eyre!(
                        "VoxConverse {rttm_subdir} extraction failed ({error}); extraction cleanup failed ({cleanup_error})"
                    ));
                }
                return Err(error);
            }
            let macosx_dir = extracting_dir.join("__MACOSX");
            if macosx_dir.exists() {
                fs::remove_dir_all(macosx_dir)?;
            }
            fs::write(extracting_dir.join(".complete"), b"")?;
            fs::rename(extracting_dir, &archive_dir)?;
        }
        let extracted_audio_dir = [audio_dir, split_audio_dir]
            .into_iter()
            .find(|path| path.is_dir());
        if let Some(extracted_audio_dir) = extracted_audio_dir {
            copy_directory(&extracted_audio_dir, &wav_dir)?;
        } else {
            ensure!(
                false,
                "VoxConverse {rttm_subdir} archive did not contain an audio directory"
            );
        }
    }

    if !rttm_dir.is_dir() {
        println!("=== Downloading VoxConverse ground truth RTTMs ===");
        let tmp_clone = cache.join("voxconverse-clone");
        let source_rttm_dir = tmp_clone.join(rttm_subdir);
        if !source_rttm_dir.is_dir() {
            if tmp_clone.exists() {
                fs::remove_dir_all(&tmp_clone)?;
            }
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
        }
        fs::create_dir_all(&rttm_dir)?;
        for entry in fs::read_dir(source_rttm_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "rttm") {
                fs::copy(&path, rttm_dir.join(entry.file_name()))?;
            }
        }
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
        if matches!(entry.file_name().to_str(), Some(".DS_Store" | "__MACOSX")) {
            continue;
        }
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
