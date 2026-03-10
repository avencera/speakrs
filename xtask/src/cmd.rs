use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use color_eyre::eyre::{Result, bail};

/// Run a command, inheriting stdio, and bail on non-zero exit
pub fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd.status()?;
    if !status.success() {
        bail!(
            "{} failed with {}",
            cmd.get_program().to_string_lossy(),
            status
        );
    }
    Ok(())
}

/// Run a command, printing stdout to the terminal and also writing it to a file
pub fn tee_cmd(cmd: &mut Command, path: &Path) -> Result<()> {
    cmd.stdout(Stdio::piped());
    let mut child = cmd.spawn()?;
    let stdout = child.stdout.take().expect("stdout was piped");
    let mut file = File::create(path)?;
    let reader = BufReader::new(stdout);

    for line in reader.lines() {
        let line = line?;
        println!("{line}");
        writeln!(file, "{line}")?;
    }

    let status = child.wait()?;
    if !status.success() {
        bail!(
            "{} failed with {}",
            cmd.get_program().to_string_lossy(),
            status
        );
    }
    Ok(())
}

/// Run a command, capture stdout, return (elapsed, stdout_text)
///
/// Returns empty stdout on non-zero exit or timeout
pub fn capture_cmd(cmd: &mut Command) -> Result<(std::time::Duration, String)> {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let start = std::time::Instant::now();
    let child = cmd.spawn()?;

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(_) => {
            return Ok((start.elapsed(), String::new()));
        }
    };
    let elapsed = start.elapsed();

    if !output.status.success() {
        return Ok((elapsed, String::new()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    Ok((elapsed, stdout))
}

/// Resolve the project root (parent of xtask/)
pub fn project_root() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir)
        .parent()
        .expect("xtask should be in a subdirectory of the project root")
        .to_path_buf()
}

/// Locate the FluidAudio checkout, checking FLUIDAUDIO_PATH env var then the default cache location
pub fn find_fluidaudio() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FLUIDAUDIO_PATH") {
        let p = PathBuf::from(path);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let default = PathBuf::from(home).join(".cache/cmd/repos/github.com/FluidInference/FluidAudio");
    if default.is_dir() {
        Some(default)
    } else {
        None
    }
}

/// Get WAV file duration in seconds using hound
pub fn wav_duration_seconds(path: &Path) -> Result<f64> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let num_samples = reader.duration(); // total samples per channel
    Ok(num_samples as f64 / spec.sample_rate as f64)
}
