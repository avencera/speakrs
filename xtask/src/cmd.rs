use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use color_eyre::eyre::{Result, bail, eyre};

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
    let stdout = child.stdout.take().ok_or_else(|| {
        eyre!(
            "stdout was not piped for {}",
            cmd.get_program().to_string_lossy()
        )
    })?;
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

/// Resolve the project root (parent of xtask/)
pub fn project_root() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root
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
