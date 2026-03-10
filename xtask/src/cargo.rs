use std::path::Path;
use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::{project_root, run_cmd, tee_cmd};

/// Map a diarization mode to the cargo feature flags needed
pub fn features_for_mode(mode: &str) -> Vec<String> {
    match mode {
        "coreml" | "coreml-fast" => vec!["coreml".to_string()],
        "cuda" => vec!["cuda".to_string()],
        _ => vec![],
    }
}

fn cargo_features_args(features: &[String]) -> Vec<String> {
    if features.is_empty() {
        return vec![];
    }
    vec!["--features".to_string(), features.join(",")]
}

/// Build a binary with the given features in release mode
///
/// Binaries live in the xtask package, so we pass `-p xtask`
pub fn cargo_build(bin: &str, features: &[String]) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(project_root())
        .args(["build", "--release", "-p", "xtask", "--bin", bin]);
    for arg in cargo_features_args(features) {
        cmd.arg(arg);
    }
    run_cmd(&mut cmd)
}

/// Run a binary with the given features and args
///
/// If `tee_to` is Some, stdout is written to both the terminal and the file
pub fn cargo_run(
    bin: &str,
    features: &[String],
    args: &[&str],
    tee_to: Option<&Path>,
) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(project_root())
        .args(["run", "--release", "-p", "xtask", "--bin", bin]);
    for arg in cargo_features_args(features) {
        cmd.arg(arg);
    }
    cmd.arg("--");
    for arg in args {
        cmd.arg(arg);
    }

    match tee_to {
        Some(path) => tee_cmd(&mut cmd, path),
        None => run_cmd(&mut cmd),
    }
}
