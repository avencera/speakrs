use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;

/// AISHELL-4 test set -- Mandarin conference meetings
/// Audio (FLAC) + RTTM from OpenSLR
pub fn ensure(dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if wav_dir.is_dir() && rttm_dir.is_dir() {
        return Ok(());
    }

    println!("=== Downloading AISHELL-4 test set (5.2 GB) ===");
    let raw_dir = std::env::temp_dir().join("aishell4-raw");
    let tar_path = raw_dir.join("test.tar.gz");

    fs::create_dir_all(&raw_dir)?;
    if !tar_path.exists() {
        run_cmd(
            Command::new("curl")
                .args(["--fail", "-L", "-o"])
                .arg(&tar_path)
                .arg("https://openslr.trmal.net/resources/111/test.tar.gz"),
        )?;
    }

    println!("Extracting...");
    run_cmd(
        Command::new("tar")
            .args(["xzf"])
            .arg(&tar_path)
            .arg("-C")
            .arg(&raw_dir),
    )?;

    fs::create_dir_all(&wav_dir)?;
    fs::create_dir_all(&rttm_dir)?;

    // structure: test/wav/*.flac + test/TextGrid/*.rttm (+ *.TextGrid)
    let src_wav_dir = raw_dir.join("test/wav");
    let src_tg_dir = raw_dir.join("test/TextGrid");

    // copy pre-existing RTTM files
    if src_tg_dir.is_dir() {
        for entry in fs::read_dir(&src_tg_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "rttm") {
                fs::copy(&path, rttm_dir.join(entry.file_name()))?;
            }
        }
    }

    // convert FLAC to 16kHz mono WAV
    if src_wav_dir.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(&src_wav_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            let stem = path.file_stem().unwrap().to_string_lossy().to_string();
            convert_to_16k_mono(&path, &wav_dir.join(format!("{stem}.wav")))?;
        }
    }

    let _ = fs::remove_dir_all(&raw_dir);
    println!("AISHELL-4 setup complete");

    Ok(())
}
