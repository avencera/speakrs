use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;
use crate::path::file_stem_string;

/// Earnings-21 -- corporate earnings call recordings
/// Audio + RTTM from revdotcom/speech-datasets
pub fn ensure(dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if wav_dir.is_dir() && rttm_dir.is_dir() {
        return Ok(());
    }

    println!("=== Downloading Earnings-21 ===");
    let tmp_clone = std::env::temp_dir().join("earnings21-clone");
    let _ = fs::remove_dir_all(&tmp_clone);

    run_cmd(
        Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/revdotcom/speech-datasets",
            ])
            .arg(&tmp_clone),
    )?;
    run_cmd(
        Command::new("git")
            .args(["sparse-checkout", "set", "earnings21"])
            .current_dir(&tmp_clone),
    )?;
    let _ = run_cmd(
        Command::new("git")
            .args(["lfs", "pull", "--include", "earnings21/media/*"])
            .current_dir(&tmp_clone),
    );

    let src_rttm_dir = tmp_clone.join("earnings21/rttms");
    if src_rttm_dir.is_dir() {
        fs::create_dir_all(&rttm_dir)?;
        for entry in fs::read_dir(&src_rttm_dir)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|e| e == "rttm") {
                fs::copy(entry.path(), rttm_dir.join(entry.file_name()))?;
            }
        }
    }

    let src_media_dir = tmp_clone.join("earnings21/media");
    if src_media_dir.is_dir() {
        fs::create_dir_all(&wav_dir)?;
        let mut entries: Vec<_> = fs::read_dir(&src_media_dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            if path.is_file() {
                let stem = file_stem_string(&path)?;
                let wav_path = wav_dir.join(format!("{stem}.wav"));
                convert_to_16k_mono(&path, &wav_path)?;
            }
        }
    }

    let _ = fs::remove_dir_all(&tmp_clone);
    println!("Earnings-21 setup complete");

    Ok(())
}
