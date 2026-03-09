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

/// Resolve the project root (parent of xtask/)
pub fn project_root() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir)
        .parent()
        .expect("xtask should be in a subdirectory of the project root")
        .to_path_buf()
}
