use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::{convert_to_16k_mono, textgrid_to_rttm};
use crate::datasets::Dataset;

/// AISHELL-4 test set — Mandarin conference meetings
/// Audio + TextGrid from OpenSLR
pub struct Aishell4;

impl Dataset for Aishell4 {
    fn id(&self) -> &'static str {
        "aishell4"
    }

    fn display_name(&self) -> &'static str {
        "AISHELL-4"
    }

    fn ensure(&self, base_dir: &Path) -> Result<()> {
        let dir = self.dataset_dir(base_dir);
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

        // AISHELL-4 structure: test/{session_id}/{session_id}.TextGrid + content.wav
        let test_dir = raw_dir.join("test");
        if test_dir.is_dir() {
            convert_sessions(&test_dir, &wav_dir, &rttm_dir)?;
        }

        let _ = fs::remove_dir_all(&raw_dir);
        println!("AISHELL-4 setup complete");

        Ok(())
    }
}

fn convert_sessions(test_dir: &Path, wav_dir: &Path, rttm_dir: &Path) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(test_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in &entries {
        let session_dir = entry.path();
        let session_id = entry.file_name().to_string_lossy().to_string();

        // find TextGrid file
        let textgrid = find_file_with_ext(&session_dir, "TextGrid");
        // find WAV file
        let source_wav = find_file_with_ext(&session_dir, "wav")
            .or_else(|| find_file_with_ext(&session_dir, "flac"));

        if let Some(tg) = textgrid {
            textgrid_to_rttm(
                &tg,
                &rttm_dir.join(format!("{session_id}.rttm")),
                &session_id,
            )?;
        }
        if let Some(audio) = source_wav {
            convert_to_16k_mono(&audio, &wav_dir.join(format!("{session_id}.wav")))?;
        }
    }

    Ok(())
}

fn find_file_with_ext(dir: &Path, ext: &str) -> Option<std::path::PathBuf> {
    // search dir and one level of subdirectories
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case(ext))
            {
                return Some(path);
            }
            if path.is_dir()
                && let Ok(sub_entries) = fs::read_dir(&path)
            {
                for sub in sub_entries.flatten() {
                    let sub_path = sub.path();
                    if sub_path.is_file()
                        && sub_path
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
                    {
                        return Some(sub_path);
                    }
                }
            }
        }
    }
    None
}
