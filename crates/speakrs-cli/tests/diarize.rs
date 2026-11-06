//! Runs the `speakrs diarize` binary against the CPU fixture models

#![cfg(feature = "cpu")]

use std::path::PathBuf;
use std::process::{Command, Output};

fn repo_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(name)
}

// same lookup as the speakrs crate's tests, so a shared model directory works here too
fn model_dir() -> PathBuf {
    std::env::var_os("SPEAKRS_MODEL_FIXTURE_DIR")
        .filter(|path| !path.is_empty())
        .map_or_else(|| repo_path("fixtures/models"), PathBuf::from)
}

fn speakrs(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_speakrs"))
        .args(args)
        .output()
        .unwrap()
}

fn diarize(extra: &[&str]) -> String {
    let model_dir = model_dir();
    let mut args = vec!["diarize", "--mode", "cpu", "--model-dir"];
    args.push(model_dir.to_str().unwrap());
    args.extend_from_slice(extra);
    let output = speakrs(&args);
    assert!(
        output.status.success(),
        "speakrs {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// (file id, start, duration, speaker) for each RTTM line
fn turns(rttm: &str) -> Vec<(String, f64, f64, String)> {
    rttm.lines()
        .map(|line| {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 10, "not an RTTM line: {line}");
            assert_eq!(fields[0], "SPEAKER");
            (
                fields[1].to_string(),
                fields[3].parse().unwrap(),
                fields[4].parse().unwrap(),
                fields[7].to_string(),
            )
        })
        .collect()
}

#[test]
fn batch_prints_each_file_in_input_order_like_single_runs() {
    let long = repo_path("fixtures/test.wav");
    let short = repo_path("fixtures/test_short.wav");
    let long = long.to_str().unwrap();
    let short = short.to_str().unwrap();

    let single_long = diarize(&[long]);
    let single_short = diarize(&[short]);
    assert!(turns(&single_long).len() > 1);
    assert!(
        turns(&single_long)
            .iter()
            .all(|(file_id, ..)| file_id == "test")
    );
    assert!(
        turns(&single_short)
            .iter()
            .all(|(file_id, ..)| file_id == "test_short")
    );

    let batch = diarize(&[short, long]);
    assert_eq!(batch, format!("{single_short}{single_long}"));
}

#[test]
fn file_id_and_exclusive_shape_the_rttm() {
    let audio = repo_path("fixtures/test.wav");
    let rttm = diarize(&[
        "--file-id",
        "meeting",
        "--exclusive",
        audio.to_str().unwrap(),
    ]);
    let turns = turns(&rttm);

    assert!(turns.iter().all(|(file_id, ..)| file_id == "meeting"));
    let speakers: std::collections::BTreeSet<_> =
        turns.iter().map(|(.., speaker)| speaker).collect();
    assert!(speakers.len() > 1, "expected several speakers in {rttm}");
    // exclusive turns come out sorted by start and never overlap
    for pair in turns.windows(2) {
        let (_, start, duration, _) = &pair[0];
        let (_, next_start, ..) = &pair[1];
        assert!(start + duration <= next_start + 1e-6, "overlap in {rttm}");
    }
}

#[test]
fn rejects_file_id_for_several_files() {
    let output = speakrs(&["diarize", "--file-id", "x", "a.wav", "b.wav"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--file-id needs exactly one"));
}

#[test]
fn rejects_files_with_the_same_stem_before_downloading() {
    let cache = tempfile::tempdir().unwrap();
    let output = speakrs(&[
        "diarize",
        "--cache-dir",
        cache.path().to_str().unwrap(),
        "a/meeting.wav",
        "b/meeting.flac",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("b/meeting.flac has the same name, without its extension"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
}

// a mode the build lacks must fail before preparing models, so it downloads nothing
#[cfg(not(feature = "_cuda"))]
#[test]
fn unsupported_mode_fails_before_downloading() {
    let cache = tempfile::tempdir().unwrap();
    let output = speakrs(&[
        "diarize",
        "--mode",
        "cuda",
        "--cache-dir",
        cache.path().to_str().unwrap(),
        "missing.wav",
    ]);

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("does not support mode `cuda`"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(std::fs::read_dir(cache.path()).unwrap().count(), 0);
}
