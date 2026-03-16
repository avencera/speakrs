use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::{convert_to_16k_mono, textgrid_to_rttm};

/// AliMeeting eval set -- Mandarin meetings with high overlap
/// Audio + TextGrid from OpenSLR
pub fn ensure(dir: &Path) -> Result<()> {
    let wav_dir = dir.join("wav");
    let rttm_dir = dir.join("rttm");

    if wav_dir.is_dir() && rttm_dir.is_dir() {
        return Ok(());
    }

    println!("=== Downloading AliMeeting eval set (3.4 GB) ===");
    let raw_dir = std::env::temp_dir().join("alimeeting-raw");
    let tar_path = raw_dir.join("Eval_Ali.tar.gz");

    fs::create_dir_all(&raw_dir)?;
    if !tar_path.exists() {
        run_cmd(
            Command::new("curl")
                .args(["--fail", "-L", "-o"])
                .arg(&tar_path)
                .arg("https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/AliMeeting/openlr/Eval_Ali.tar.gz"),
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

    let near_dir = raw_dir.join("Eval_Ali_near/audio_and_target");
    if near_dir.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(&near_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let session_dir = entry.path();
            let session_id = entry.file_name().to_string_lossy().to_string();

            let textgrid_path = session_dir.join(format!("{session_id}.TextGrid"));
            if textgrid_path.exists() {
                textgrid_to_rttm(
                    &textgrid_path,
                    &rttm_dir.join(format!("{session_id}.rttm")),
                    &session_id,
                )?;
            }

            let audio_path = session_dir.join(format!("{session_id}_near.wav"));
            let audio_path = if audio_path.exists() {
                audio_path
            } else {
                find_wav(&session_dir).unwrap_or(audio_path)
            };

            if audio_path.exists() {
                convert_to_16k_mono(&audio_path, &wav_dir.join(format!("{session_id}.wav")))?;
            }
        }
    }

    let _ = fs::remove_dir_all(&raw_dir);
    println!("AliMeeting setup complete");

    Ok(())
}

fn find_wav(dir: &Path) -> Option<std::path::PathBuf> {
    fs::read_dir(dir).ok()?.flatten().find_map(|e| {
        let p = e.path();
        if p.extension().is_some_and(|e| e == "wav") {
            Some(p)
        } else {
            None
        }
    })
}
