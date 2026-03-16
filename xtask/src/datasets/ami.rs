use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, bail};

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;

/// AMI IHM (Individual Headset Mix)
/// RTTMs from BUTSpeechFIT, audio from Edinburgh DataShare
pub fn ensure_ihm(dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if !rttm_dir.is_dir() {
        println!("=== Downloading AMI IHM RTTMs (BUTSpeechFIT) ===");
        let tmp_clone = std::env::temp_dir().join("ami-diarization-setup");
        let _ = fs::remove_dir_all(&tmp_clone);
        run_cmd(
            Command::new("git")
                .args([
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "https://github.com/BUTSpeechFIT/AMI-diarization-setup",
                ])
                .arg(&tmp_clone),
        )?;
        run_cmd(
            Command::new("git")
                .args(["sparse-checkout", "set", "only_words/rttms"])
                .current_dir(&tmp_clone),
        )?;

        fs::create_dir_all(&rttm_dir)?;
        for split in &["dev", "test"] {
            let split_dir = tmp_clone.join("only_words/rttms").join(split);
            if split_dir.is_dir() {
                for entry in fs::read_dir(&split_dir)? {
                    let entry = entry?;
                    if entry.path().extension().is_some_and(|e| e == "rttm") {
                        fs::copy(entry.path(), rttm_dir.join(entry.file_name()))?;
                    }
                }
            }
        }
        let _ = fs::remove_dir_all(&tmp_clone);
    }

    if !wav_dir.is_dir() {
        fs::create_dir_all(&wav_dir)?;
    }

    download_ami_wavs(&rttm_dir, &wav_dir, "Mix-Headset", "ami-wav-download")?;
    Ok(())
}

/// AMI SDM (Single Distant Microphone, Array1-01)
/// Same RTTMs as IHM, different audio channel
pub fn ensure_sdm(dir: &Path, base_dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    // RTTMs are the same as IHM
    let ihm_rttm_dir = base_dir.join("ami-ihm").join("rttm");
    if !rttm_dir.is_dir() {
        if !ihm_rttm_dir.is_dir() {
            ensure_ihm(&base_dir.join("ami-ihm"))?;
        }
        fs::create_dir_all(&rttm_dir)?;
        for entry in fs::read_dir(&ihm_rttm_dir)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|e| e == "rttm") {
                fs::copy(entry.path(), rttm_dir.join(entry.file_name()))?;
            }
        }
    }

    if !wav_dir.is_dir() {
        fs::create_dir_all(&wav_dir)?;
    }

    download_ami_wavs(&rttm_dir, &wav_dir, "Array1-01", "ami-sdm-download")?;
    Ok(())
}

fn download_ami_wavs(
    rttm_dir: &Path,
    wav_dir: &Path,
    mic_name: &str,
    tmp_name: &str,
) -> Result<()> {
    let mut missing = Vec::new();
    for entry in fs::read_dir(rttm_dir)? {
        let entry = entry?;
        let stem = entry
            .path()
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .to_string();
        if !wav_dir.join(format!("{stem}.wav")).exists() {
            missing.push(stem);
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    println!(
        "=== Downloading {} AMI {mic_name} WAV files ===",
        missing.len()
    );
    let base_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus";
    let tmp_dir = std::env::temp_dir().join(tmp_name);
    fs::create_dir_all(&tmp_dir)?;

    let mut failed = Vec::new();
    for stem in &missing {
        let remote_name = format!("{stem}.{mic_name}.wav");
        let url = format!("{base_url}/{stem}/audio/{remote_name}");
        let tmp_path = tmp_dir.join(&remote_name);

        print!("  {stem}...");
        let result = run_cmd(
            Command::new("curl")
                .args(["--fail", "-L", "-s", "-o"])
                .arg(&tmp_path)
                .arg(&url),
        );

        if result.is_ok() && tmp_path.exists() {
            convert_to_16k_mono(&tmp_path, &wav_dir.join(format!("{stem}.wav")))?;
            let _ = fs::remove_file(&tmp_path);
            println!(" ok");
        } else {
            failed.push(stem.clone());
            println!(" failed");
        }
    }

    let _ = fs::remove_dir_all(&tmp_dir);

    if !failed.is_empty() {
        bail!(
            "{} AMI {mic_name} sessions failed to download. Missing: {}",
            failed.len(),
            failed.join(", ")
        );
    }

    Ok(())
}
