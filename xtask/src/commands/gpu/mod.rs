mod runpod;
mod vastai;

use std::fmt;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::str::FromStr;

use color_eyre::eyre::{Result, bail};

use crate::cmd::{project_root, run_cmd};

pub const IMAGE: &str = "ghcr.io/avencera/speakrs-gpu:latest";
pub const GITHUB_REPO: &str = "https://github.com/avencera/speakrs.git";

pub const MODELS_SCRIPT: &str = r#"
export PATH="$HOME/.local/bin:$PATH"
MODELS_DIR=/workspace/speakrs/fixtures/models
if [ -f "$MODELS_DIR/segmentation-3.0.onnx" ] && [ -f "$MODELS_DIR/wespeaker-voxceleb-resnet34.onnx" ]; then
    echo "=== Models already present ==="
else
    echo "=== Downloading models ==="
    mkdir -p "$MODELS_DIR"
    uv tool run --from huggingface-hub hf download avencera/speakrs-models --local-dir "$MODELS_DIR" \
        --include "segmentation-3.0.onnx" \
        --include "wespeaker-voxceleb-resnet34.onnx" \
        --include "wespeaker-voxceleb-resnet34.onnx.data" \
        --include "plda_*.npy" \
        --include "wespeaker-voxceleb-resnet34.min_num_samples.txt"
fi
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    RunPod,
    VastAi,
}

impl FromStr for Backend {
    type Err = color_eyre::eyre::Report;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "runpod" => Ok(Self::RunPod),
            "vastai" | "vast" => Ok(Self::VastAi),
            other => bail!("unknown backend '{other}', expected 'runpod' or 'vastai'"),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RunPod => write!(f, "runpod"),
            Self::VastAi => write!(f, "vastai"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub backend: Backend,
    pub instance_id: String,
    pub ssh_host: String,
    pub ssh_port: String,
}

fn instances_dir() -> std::path::PathBuf {
    project_root().join(".gpu-instances")
}

fn read_instance(name: &str) -> Result<InstanceInfo> {
    let path = instances_dir().join(name);
    let content = fs::read_to_string(&path)
        .map_err(|_| color_eyre::eyre::eyre!("No instance '{name}' found"))?;

    let lines: Vec<&str> = content.trim().lines().collect();
    if lines.len() < 4 {
        bail!("Instance file '{name}' is malformed (expected 4 lines: backend, id, host, port)");
    }

    Ok(InstanceInfo {
        backend: lines[0].parse()?,
        instance_id: lines[1].to_string(),
        ssh_host: lines[2].to_string(),
        ssh_port: lines[3].to_string(),
    })
}

fn save_instance(name: &str, info: &InstanceInfo) -> Result<()> {
    let dir = instances_dir();
    fs::create_dir_all(&dir)?;
    let content = format!(
        "{}\n{}\n{}\n{}\n",
        info.backend, info.instance_id, info.ssh_host, info.ssh_port
    );
    fs::write(dir.join(name), content)?;
    Ok(())
}

fn list_instances() -> Result<Vec<String>> {
    let dir = instances_dir();
    if !dir.is_dir() {
        return Ok(vec![]);
    }

    let mut names: Vec<String> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .filter_map(|e| e.file_name().into_string().ok())
        .collect();
    names.sort();
    Ok(names)
}

fn resolve_instance(name: Option<&str>) -> Result<(String, InstanceInfo)> {
    if let Some(n) = name {
        let info = read_instance(n)?;
        return Ok((n.to_string(), info));
    }

    let names = list_instances()?;
    match names.len() {
        0 => bail!("No instances found. Run `cargo xtask gpu create <name>` first"),
        1 => {
            let n = &names[0];
            let info = read_instance(n)?;
            println!("Auto-selected instance '{n}'");
            Ok((n.clone(), info))
        }
        _ => {
            let input = names.join("\n");
            let output = Command::new("fzf")
                .arg("--header=Pick an instance")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .and_then(|mut child| {
                    if let Some(ref mut stdin) = child.stdin {
                        stdin.write_all(input.as_bytes()).ok();
                    }
                    child.wait_with_output()
                })?;

            if !output.status.success() {
                bail!("No instance selected");
            }

            let n = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let info = read_instance(&n)?;
            Ok((n, info))
        }
    }
}

pub fn get_ssh_args(info: &InstanceInfo) -> Vec<String> {
    vec![
        "-p".to_string(),
        info.ssh_port.clone(),
        format!("root@{}", info.ssh_host),
    ]
}

pub fn ssh_cmd(info: &InstanceInfo) -> Command {
    let args = get_ssh_args(info);
    let mut cmd = Command::new("ssh");
    cmd.args([
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]);
    cmd.args(&args);
    cmd
}

fn models_script_with_token() -> String {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        format!("export HF_TOKEN={token}\n{MODELS_SCRIPT}")
    } else {
        MODELS_SCRIPT.to_string()
    }
}

pub fn run_remote_script(info: &InstanceInfo, script: &str) -> Result<()> {
    let mut cmd = ssh_cmd(info);
    cmd.arg("bash -s").stdin(std::process::Stdio::piped());

    let mut child = cmd.spawn()?;
    if let Some(ref mut stdin) = child.stdin {
        stdin.write_all(script.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("Remote command failed");
    }
    Ok(())
}

// --- public API ---

pub fn create(name: &str, backend: Backend, gpu_type: &str, min_tflops: f64) -> Result<()> {
    if instances_dir().join(name).exists() {
        bail!("Instance '{name}' already exists. Destroy it first or pick a different name");
    }

    let info = match backend {
        Backend::RunPod => runpod::provision(gpu_type)?,
        Backend::VastAi => vastai::provision(min_tflops)?,
    };

    save_instance(name, &info)?;
    println!(
        "Instance '{name}' saved ({backend}, id={})",
        info.instance_id
    );
    Ok(())
}

pub fn setup(name: Option<&str>, branch: &str) -> Result<()> {
    let (n, info) = resolve_instance(name)?;

    match info.backend {
        Backend::RunPod => runpod::setup(&info, branch)?,
        Backend::VastAi => vastai::setup(&info)?,
    }

    println!("Instance '{n}' setup complete");
    Ok(())
}

pub fn benchmark(
    name: Option<&str>,
    args: &[String],
    detach: bool,
    force: bool,
    branch: &str,
) -> Result<()> {
    let (_n, info) = resolve_instance(name)?;

    let remote_cmd = match info.backend {
        Backend::RunPod => runpod::prepare_benchmark(&info, args, branch)?,
        Backend::VastAi => vastai::prepare_benchmark(&info, args)?,
    };

    if detach {
        let mut check = ssh_cmd(&info);
        check.arg("tmux has-session -t benchmark 2>/dev/null");
        if check.output()?.status.success() && !force {
            bail!(
                "A benchmark is already running. Use --force to replace it, or check with:\n  \
                 cargo xtask gpu status <name>"
            );
        }

        println!("Launching benchmark in detached tmux session...");
        let escaped_cmd = remote_cmd.replace('\'', "'\\''");
        let tmux_cmd = format!(
            "tmux kill-session -t benchmark 2>/dev/null; \
             mkdir -p /workspace/speakrs/_benchmarks && \
             tmux new-session -d -s benchmark \
             '{escaped_cmd} 2>&1 | tee /workspace/speakrs/_benchmarks/latest.log; \
             echo \"=== BENCHMARK COMPLETE ===\"'",
        );
        let mut cmd = ssh_cmd(&info);
        cmd.arg(&tmux_cmd);
        run_cmd(&mut cmd)?;

        println!("Benchmark running in background");
        println!("  cargo xtask gpu status <name>        — check progress");
        println!("  cargo xtask gpu attach <name>        — reconnect to live session");
        println!("  cargo xtask gpu pull-results <name>  — download results when done");
    } else {
        println!("Running benchmark on remote GPU...");
        let mut cmd = ssh_cmd(&info);
        cmd.arg(&remote_cmd);
        run_cmd(&mut cmd)?;
    }

    Ok(())
}

pub fn status(name: Option<&str>) -> Result<()> {
    let (_n, info) = resolve_instance(name)?;

    let script = concat!(
        "if tmux has-session -t benchmark 2>/dev/null; then ",
        "echo 'STATUS: RUNNING'; echo '---'; ",
        "tmux capture-pane -t benchmark -p | tail -30; ",
        "else ",
        "echo 'STATUS: NOT RUNNING'; ",
        "if [ -f /workspace/speakrs/_benchmarks/latest.log ]; then ",
        "echo '--- last output ---'; ",
        "tail -20 /workspace/speakrs/_benchmarks/latest.log; fi; fi",
    );

    let mut cmd = ssh_cmd(&info);
    cmd.arg(script);
    run_cmd(&mut cmd)?;
    Ok(())
}

pub fn attach(name: Option<&str>) -> Result<()> {
    let (_n, info) = resolve_instance(name)?;

    let mut check = ssh_cmd(&info);
    check.arg("tmux has-session -t benchmark 2>/dev/null");
    if !check.output()?.status.success() {
        bail!("No benchmark session running. Use `cargo xtask gpu status` to check");
    }

    let args = get_ssh_args(&info);
    let mut cmd = Command::new("ssh");
    cmd.args([
        "-t",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]);
    cmd.args(&args);
    cmd.arg("tmux attach -t benchmark");

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        bail!("Failed to exec ssh: {err}");
    }

    #[cfg(not(unix))]
    {
        run_cmd(&mut cmd)?;
        Ok(())
    }
}

pub fn pull_results(name: Option<&str>) -> Result<()> {
    let (_n, info) = resolve_instance(name)?;

    let root = project_root();
    let local_dir = root.join("_benchmarks");
    fs::create_dir_all(&local_dir)?;

    let ssh_opts = format!(
        "ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {}",
        info.ssh_port
    );
    let src = format!("root@{}:/workspace/speakrs/_benchmarks/", info.ssh_host);
    let dest = format!("{}/", local_dir.display());

    println!("Pulling benchmark results from remote...");
    run_cmd(
        Command::new("rsync")
            .args(["-az", "--info=progress2", "-e"])
            .arg(&ssh_opts)
            .args([&src, &dest]),
    )?;

    println!("Results saved to _benchmarks/");
    Ok(())
}

pub fn ssh(name: Option<&str>) -> Result<()> {
    let (_n, info) = resolve_instance(name)?;
    let args = get_ssh_args(&info);

    let mut cmd = Command::new("ssh");
    cmd.args([
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]);
    cmd.args(&args);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        bail!("Failed to exec ssh: {err}");
    }

    #[cfg(not(unix))]
    {
        run_cmd(&mut cmd)?;
        Ok(())
    }
}

pub fn destroy(name: Option<&str>, all: bool) -> Result<()> {
    if all {
        return destroy_all();
    }

    let (n, info) = resolve_instance(name)?;

    match info.backend {
        Backend::RunPod => runpod::destroy(&info)?,
        Backend::VastAi => vastai::destroy(&info)?,
    }

    fs::remove_file(instances_dir().join(&n))?;
    println!("Instance '{n}' destroyed and removed");
    Ok(())
}

fn destroy_all() -> Result<()> {
    let names = list_instances()?;
    if names.is_empty() {
        println!("No instances to destroy");
        return Ok(());
    }

    for name in &names {
        let info = read_instance(name)?;
        println!(
            "Destroying '{name}' ({}, id={})...",
            info.backend, info.instance_id
        );
        match info.backend {
            Backend::RunPod => runpod::destroy(&info)?,
            Backend::VastAi => vastai::destroy(&info)?,
        }
        fs::remove_file(instances_dir().join(name))?;
    }

    println!("All {} instances destroyed", names.len());
    Ok(())
}

pub fn list() -> Result<()> {
    let names = list_instances()?;
    if names.is_empty() {
        println!("No instances");
        return Ok(());
    }

    for name in &names {
        match read_instance(name) {
            Ok(info) => println!(
                "{name:20} {backend:8} {id:20} {host}:{port}",
                backend = info.backend,
                id = info.instance_id,
                host = info.ssh_host,
                port = info.ssh_port,
            ),
            Err(e) => println!("{name:20} ERROR: {e}"),
        }
    }
    Ok(())
}
