use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::Result;

use crate::cmd::wav_duration_seconds;
use crate::path::file_stem_string;

pub fn discover_files(
    dataset_dir: &Path,
    max_files: u32,
    max_minutes: f64,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let wav_dir = dataset_dir.join("wav");
    let rttm_dir = dataset_dir.join("rttm");

    let (wav_dir, rttm_dir) = if wav_dir.exists() && rttm_dir.exists() {
        (wav_dir, rttm_dir)
    } else {
        (dataset_dir.to_path_buf(), dataset_dir.to_path_buf())
    };

    let mut pairs: Vec<(PathBuf, PathBuf, f64)> = Vec::new();
    let mut entries: Vec<_> = fs::read_dir(&wav_dir)?
        .filter_map(|entry| entry.ok())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());

    for entry in entries {
        let wav_path = entry.path();
        if wav_path.extension().is_some_and(|ext| ext == "wav") {
            let stem = file_stem_string(&wav_path)?;
            let rttm_path = rttm_dir.join(format!("{stem}.rttm"));
            if rttm_path.exists()
                && let Ok(duration) = wav_duration_seconds(&wav_path)
            {
                pairs.push((wav_path, rttm_path, duration));
            }
        }
    }

    Ok(select_pairs_for_benchmark(pairs, max_files, max_minutes))
}

pub fn select_pairs_for_benchmark(
    pairs: Vec<(PathBuf, PathBuf, f64)>,
    max_files: u32,
    max_minutes: f64,
) -> Vec<(PathBuf, PathBuf)> {
    let mut indexed: Vec<_> = pairs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| a.1.2.total_cmp(&b.1.2));

    let mut selected: Vec<(usize, PathBuf, PathBuf)> = Vec::new();
    let mut total_minutes = 0.0;

    for (orig_idx, (wav, rttm, duration)) in indexed {
        if selected.len() >= max_files as usize {
            break;
        }
        if total_minutes + duration / 60.0 > max_minutes && !selected.is_empty() {
            break;
        }
        selected.push((orig_idx, wav, rttm));
        total_minutes += duration / 60.0;
    }

    selected.sort_by_key(|(idx, _, _)| *idx);
    selected
        .into_iter()
        .map(|(_, wav, rttm)| (wav, rttm))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_pairs_for_benchmark_greedy_selection_in_input_order() {
        let pairs = vec![
            (
                PathBuf::from("medium.wav"),
                PathBuf::from("medium.rttm"),
                60.0,
            ),
            (PathBuf::from("long.wav"), PathBuf::from("long.rttm"), 180.0),
            (
                PathBuf::from("short.wav"),
                PathBuf::from("short.rttm"),
                30.0,
            ),
        ];

        let selected = select_pairs_for_benchmark(pairs, 2, 2.0);
        let names: Vec<_> = selected
            .iter()
            .map(|(wav, _)| file_stem_string(wav).unwrap())
            .collect();

        assert_eq!(names, vec!["medium", "short"]);
    }

    #[test]
    fn select_pairs_for_benchmark_preserves_input_order() {
        let pairs = vec![
            (PathBuf::from("c.wav"), PathBuf::from("c.rttm"), 120.0),
            (PathBuf::from("a.wav"), PathBuf::from("a.rttm"), 30.0),
            (PathBuf::from("b.wav"), PathBuf::from("b.rttm"), 60.0),
        ];

        let selected = select_pairs_for_benchmark(pairs, u32::MAX, f64::MAX);
        let names: Vec<_> = selected
            .iter()
            .map(|(wav, _)| file_stem_string(wav).unwrap())
            .collect();

        assert_eq!(names, vec!["c", "a", "b"]);
    }

    #[test]
    fn split_rttm_by_file_id_groups_lines() {
        let grouped = super::super::types::split_rttm_by_file_id(concat!(
            "SPEAKER a 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>\n",
            "SPEAKER b 1 0.5 1.0 <NA> <NA> spk2 <NA> <NA>\n",
            "SPEAKER a 1 1.0 0.5 <NA> <NA> spk1 <NA> <NA>\n",
        ));

        assert_eq!(
            grouped.get("a").unwrap(),
            "SPEAKER a 1 0.0 1.0 <NA> <NA> spk1 <NA> <NA>\n\
SPEAKER a 1 1.0 0.5 <NA> <NA> spk1 <NA> <NA>\n"
        );
        assert_eq!(
            grouped.get("b").unwrap(),
            "SPEAKER b 1 0.5 1.0 <NA> <NA> spk2 <NA> <NA>\n"
        );
    }
}
