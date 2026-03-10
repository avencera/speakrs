use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Result, bail};

use crate::cmd::run_cmd;
use crate::convert::convert_to_16k_mono;
use crate::datasets::Dataset;

/// AMI corpus, Individual Headset Mix (mix-headset)
/// RTTMs from BUTSpeechFIT, audio from Edinburgh DataShare
pub struct AmiIhm;

impl Dataset for AmiIhm {
    fn id(&self) -> &'static str {
        "ami-ihm"
    }

    fn display_name(&self) -> &'static str {
        "AMI IHM"
    }

    fn ensure(&self, base_dir: &Path) -> Result<()> {
        let dir = self.dataset_dir(base_dir);
        let wav_dir = dir.join("wav");
        let rttm_dir = dir.join("rttm");

        // step 1: download RTTMs from BUTSpeechFIT
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
            // copy dev + test RTTMs
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

        // step 2: download audio from Edinburgh DataShare
        if !wav_dir.is_dir() {
            fs::create_dir_all(&wav_dir)?;
        }

        let mut missing = Vec::new();
        for entry in fs::read_dir(&rttm_dir)? {
            let entry = entry?;
            let stem = entry
                .path()
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let wav_path = wav_dir.join(format!("{stem}.wav"));
            if !wav_path.exists() {
                missing.push(stem);
            }
        }

        if !missing.is_empty() {
            println!(
                "=== Downloading {} AMI IHM Mix-Headset WAV files ===",
                missing.len()
            );
            let base_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus";

            let tmp_dir = std::env::temp_dir().join("ami-wav-download");
            fs::create_dir_all(&tmp_dir)?;

            let mut failed = Vec::new();
            for stem in &missing {
                let remote_name = format!("{stem}.Mix-Headset.wav");
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
                    let wav_path = wav_dir.join(format!("{stem}.wav"));
                    convert_to_16k_mono(&tmp_path, &wav_path)?;
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
                    "{} AMI sessions failed to download. You can manually place \
                     Mix-Headset WAV files in {}. Missing: {}",
                    failed.len(),
                    wav_dir.display(),
                    failed.join(", ")
                );
            }
        }

        Ok(())
    }
}
