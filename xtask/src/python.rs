use std::process::Command;

use color_eyre::eyre::Result;

use crate::cmd::{project_root, run_cmd};

/// Run `uv run <args>` from the project root
pub fn uv_run(args: &[&str]) -> Result<()> {
    run_cmd(
        Command::new("uv")
            .arg("run")
            .args(args)
            .current_dir(project_root()),
    )
}

/// Run `uv run --project <project> python <script> <args>` from the project root
pub fn uv_run_project(project: &str, script: &str, args: &[&str]) -> Result<()> {
    run_cmd(
        Command::new("uv")
            .args(["run", "--project", project, "python", script])
            .args(args)
            .current_dir(project_root()),
    )
}
