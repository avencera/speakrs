use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Read as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use color_eyre::eyre::Result;
use wait_timeout::ChildExt;

pub struct SingleRunOutput {
    pub elapsed_seconds: f64,
    pub rttm: String,
}

pub struct BatchRunOutput {
    pub total_seconds: f64,
    pub per_file_rttm: HashMap<String, String>,
}

#[derive(Clone)]
pub struct CommandSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub current_dir: Option<PathBuf>,
    pub envs: Vec<(OsString, OsString)>,
}

impl CommandSpec {
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            current_dir: None,
            envs: Vec::new(),
        }
    }

    pub fn from_argv(argv: &[String]) -> Self {
        debug_assert!(!argv.is_empty());
        let mut command_spec = Self::new(argv[0].clone());
        for arg in &argv[1..] {
            command_spec = command_spec.arg(arg.clone());
        }
        command_spec
    }

    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn current_dir(mut self, current_dir: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(current_dir.into());
        self
    }

    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }

    pub fn build_command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        if let Some(current_dir) = &self.current_dir {
            command.current_dir(current_dir);
        }
        for (key, value) in &self.envs {
            command.env(key, value);
        }
        command
    }
}

/// Possible failure modes for `capture_benchmark_cmd`
pub enum BenchmarkError {
    /// Process exceeded its time budget
    Timeout { program: String, timeout: Duration },
    /// Process exited with non-zero / signal
    ProcessFailed {
        program: String,
        status: std::process::ExitStatus,
    },
    /// Other I/O error
    Other(color_eyre::eyre::Report),
}

impl BenchmarkError {
    pub fn is_timeout(&self) -> bool {
        matches!(self, Self::Timeout { .. })
    }
}

impl std::fmt::Display for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout { program, timeout } => {
                write!(f, "{program} timed out after {}s", timeout.as_secs())
            }
            Self::ProcessFailed { program, status } => {
                write!(f, "{program} failed with {status}")
            }
            Self::Other(error) => write!(f, "{error}"),
        }
    }
}

impl std::fmt::Debug for BenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for BenchmarkError {}

pub fn capture_benchmark_cmd(cmd: &mut Command, timeout: Duration) -> Result<SingleRunOutput> {
    let program = cmd.get_program().to_string_lossy().to_string();

    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| BenchmarkError::Other(error.into()))?;

    // drain stdout in a background thread to prevent pipe buffer deadlock
    let stdout_handle = child.stdout.take().ok_or_else(|| {
        BenchmarkError::Other(color_eyre::eyre::eyre!(
            "missing stdout pipe for {}",
            cmd.get_program().to_string_lossy()
        ))
    })?;
    let reader = std::thread::spawn(move || {
        let mut buffer = Vec::new();
        let mut reader = stdout_handle;
        reader.read_to_end(&mut buffer).ok();
        buffer
    });

    let start = Instant::now();
    match child
        .wait_timeout(timeout)
        .map_err(|error| BenchmarkError::Other(error.into()))?
    {
        Some(status) => {
            let elapsed = start.elapsed().as_secs_f64();
            if !status.success() {
                return Err(BenchmarkError::ProcessFailed { program, status }.into());
            }
            let stdout_bytes = reader.join().map_err(|_| {
                BenchmarkError::Other(color_eyre::eyre::eyre!(
                    "stdout reader thread panicked for {program}"
                ))
            })?;
            Ok(SingleRunOutput {
                elapsed_seconds: elapsed,
                rttm: String::from_utf8_lossy(&stdout_bytes).to_string(),
            })
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            drop(reader);
            Err(BenchmarkError::Timeout { program, timeout }.into())
        }
    }
}
