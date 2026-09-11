use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, bail, eyre};

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;
use crate::path::file_stem_string;

/// AMI IHM (Individual Headset Mix)
/// RTTMs from BUTSpeechFIT, audio from Edinburgh DataShare
pub fn ensure_ihm(dir: &Path, cache: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if !rttm_dir.is_dir() {
        println!("=== Downloading AMI IHM RTTMs (BUTSpeechFIT) ===");
        let tmp_clone = cache.join("ami-diarization-setup");
        let src_rttm_root = tmp_clone.join("only_words/rttms");
        if !src_rttm_root.is_dir() {
            if tmp_clone.exists() {
                fs::remove_dir_all(&tmp_clone)?;
            }
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
        }

        fs::create_dir_all(&rttm_dir)?;
        for split in &["dev", "test"] {
            let split_dir = src_rttm_root.join(split);
            if split_dir.is_dir() {
                for entry in fs::read_dir(&split_dir)? {
                    let entry = entry?;
                    if entry.path().extension().is_some_and(|e| e == "rttm") {
                        fs::copy(entry.path(), rttm_dir.join(entry.file_name()))?;
                    }
                }
            }
        }
    }

    if !wav_dir.is_dir() {
        fs::create_dir_all(&wav_dir)?;
    }

    download_ami_wavs(&rttm_dir, &wav_dir, cache, "Mix-Headset")?;
    Ok(())
}

/// AMI SDM (Single Distant Microphone, Array1-01)
/// Same RTTMs as IHM, different audio channel
pub fn ensure_sdm(dir: &Path, base_dir: &Path, cache: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    // rttms are the same as IHM
    let ihm_rttm_dir = base_dir.join("ami-ihm").join("rttm");
    if !rttm_dir.is_dir() {
        if !ihm_rttm_dir.is_dir() {
            bail!("AMI IHM RTTMs are required before staging AMI SDM; install ami-ihm first");
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

    download_ami_wavs(&rttm_dir, &wav_dir, cache, "Array1-01")?;
    Ok(())
}

fn download_ami_wavs(rttm_dir: &Path, wav_dir: &Path, cache: &Path, mic_name: &str) -> Result<()> {
    let mut missing = Vec::new();
    for entry in fs::read_dir(rttm_dir)? {
        let entry = entry?;
        let stem = file_stem_string(&entry.path())?;
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
    let tmp_dir = cache.join(format!("{mic_name}/downloads"));
    fs::create_dir_all(&tmp_dir)?;

    let mut failed = Vec::new();
    for stem in &missing {
        let remote_name = format!("{stem}.{mic_name}.wav");
        let url = format!("{base_url}/{stem}/audio/{remote_name}");

        print!("  {stem}...");
        let cached_path = tmp_dir.join(&remote_name);
        if !cached_path.exists() {
            let partial_path = tmp_dir.join(format!("{remote_name}.part"));
            if partial_path.exists() {
                fs::remove_file(&partial_path)?;
            }
            let download_result = run_cmd(
                Command::new("curl")
                    .args(["--fail", "-L", "-s", "-o"])
                    .arg(&partial_path)
                    .arg(&url),
            );
            if download_result.is_ok() && partial_path.exists() {
                fs::rename(&partial_path, &cached_path)?;
            } else if partial_path.exists()
                && let Err(cleanup_error) = fs::remove_file(&partial_path)
            {
                return Err(eyre!(
                    "AMI {mic_name} download failed for {stem}; partial cleanup failed ({cleanup_error})"
                ));
            }
            if let Err(_error) = download_result {
                failed.push(stem.clone());
                println!(" failed");
                continue;
            }
        }

        convert_to_16k_mono(&cached_path, &wav_dir.join(format!("{stem}.wav")))?;
        println!(" ok");
    }

    if !failed.is_empty() {
        bail!(
            "{} AMI {mic_name} sessions failed to download. Missing: {}",
            failed.len(),
            failed.join(", ")
        );
    }

    Ok(())
}
