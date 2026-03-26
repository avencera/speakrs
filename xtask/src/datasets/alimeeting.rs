use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::run_cmd;
use crate::convert::{convert_to_16k_mono, textgrid_to_rttm};
use crate::path::file_stem_string;

/// AliMeeting eval set -- Mandarin meetings with high overlap
/// Far-field audio + TextGrid from OpenSLR
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

    // structure: Eval_Ali/Eval_Ali_far/audio_dir/*.wav
    //            Eval_Ali/Eval_Ali_far/textgrid_dir/*.TextGrid
    let far_audio = raw_dir.join("Eval_Ali/Eval_Ali_far/audio_dir");
    let far_tg = raw_dir.join("Eval_Ali/Eval_Ali_far/textgrid_dir");

    if far_tg.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(&far_tg)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("textgrid"))
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            let session_id = file_stem_string(&path)?;
            textgrid_to_rttm(
                &path,
                &rttm_dir.join(format!("{session_id}.rttm")),
                &session_id,
            )?;
        }
    }

    if far_audio.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(&far_audio)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in &entries {
            let path = entry.path();
            let stem = file_stem_string(&path)?;
            // wav filenames have a mic suffix (e.g. R8001_M8004_MS801.wav)
            // textgrid filenames don't (R8001_M8004.TextGrid)
            // use the wav stem as-is for the output filename
            convert_to_16k_mono(&path, &wav_dir.join(format!("{stem}.wav")))?;
        }
    }

    let _ = fs::remove_dir_all(&raw_dir);
    println!("AliMeeting setup complete");

    Ok(())
}
