use std::fs;
use std::process::Command;

use color_eyre::eyre::{Result, bail};
use serde_json::Value;

use crate::cmd::{project_root, run_cmd};

const INSTANCE_FILE: &str = ".vastai-instance";

const SETUP_SCRIPT: &str = r#"
set -euo pipefail
apt-get update && apt-get install -y build-essential pkg-config libssl-dev git cmake curl python3-pip
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
if [ -d /workspace/speakrs ]; then
    cd /workspace/speakrs && git pull
else
    git clone https://github.com/avencera/speakrs.git /workspace/speakrs
    cd /workspace/speakrs
fi
cargo build --release --features cuda
cargo build --release -p xtask

# download ONNX models from HuggingFace
pip install huggingface-hub
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi
mkdir -p fixtures/models
huggingface-cli download avencera/speakrs-models --local-dir fixtures/models \
    --include "segmentation-3.0.onnx" "wespeaker-voxceleb-resnet34.onnx" \
    "wespeaker-voxceleb-resnet34.onnx.data" "plda_*.npy" \
    "wespeaker-voxceleb-resnet34.min_num_samples.txt"

echo "=== Setup complete ==="
"#;

fn instance_file_path() -> std::path::PathBuf {
    project_root().join(INSTANCE_FILE)
}

fn read_instance_id() -> Result<String> {
    let path = instance_file_path();
    match fs::read_to_string(&path) {
        Ok(id) => {
            let id = id.trim().to_string();
            if id.is_empty() {
                bail!("Instance file is empty. Run `cargo xtask gpu setup` first");
            }
            Ok(id)
        }
        Err(_) => bail!("No instance file found. Run `cargo xtask gpu setup` first"),
    }
}

fn save_instance_id(id: &str) -> Result<()> {
    fs::write(instance_file_path(), id)?;
    Ok(())
}

/// Parse SSH connection details from `vastai ssh-url`
///
/// Handles both formats:
///   old: `ssh -p PORT root@HOST`
///   new: `ssh://root@HOST:PORT`
fn get_ssh_args(instance_id: &str) -> Result<Vec<String>> {
    let output = Command::new("vastai")
        .args(["ssh-url", instance_id])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to get SSH URL: {stderr}");
    }

    let ssh_url = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if let Some(rest) = ssh_url.strip_prefix("ssh://") {
        // ssh://root@HOST:PORT → ["-p", "PORT", "root@HOST"]
        let (user_host, port) = rest
            .rsplit_once(':')
            .ok_or_else(|| color_eyre::eyre::eyre!("No port in ssh URL: {ssh_url}"))?;
        Ok(vec![
            "-p".to_string(),
            port.to_string(),
            user_host.to_string(),
        ])
    } else {
        // ssh -p PORT root@HOST
        let parts: Vec<&str> = ssh_url.split_whitespace().collect();
        if parts.len() < 4 {
            bail!("Unexpected ssh-url format: {ssh_url}");
        }
        Ok(parts[1..].iter().map(|s| s.to_string()).collect())
    }
}

fn ssh_cmd(instance_id: &str) -> Result<Command> {
    let args = get_ssh_args(instance_id)?;
    let mut cmd = Command::new("ssh");
    cmd.args([
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]);
    cmd.args(&args);
    Ok(cmd)
}

/// Extract port from SSH args (expects ["-p", "PORT", "user@host"])
fn ssh_port(ssh_args: &[String]) -> &str {
    &ssh_args[1]
}

/// Extract user@host from SSH args
fn ssh_user_host(ssh_args: &[String]) -> &str {
    ssh_args.last().expect("ssh args should have user@host")
}

pub fn setup(new: bool) -> Result<()> {
    let instance_id = if new {
        provision_instance()?
    } else {
        pick_instance()?
    };

    wait_for_instance(&instance_id)?;
    run_setup_script(&instance_id)?;

    println!("GPU instance ready! Use `cargo xtask gpu ssh` to connect");
    Ok(())
}

fn pick_instance() -> Result<String> {
    let output = Command::new("bash")
        .args([
            "-c",
            r#"vastai show instances --raw | jq -r '.[] | "\(.id)\t\(.actual_status)\t\(.gpu_name)\t$\(.dph_total)/hr"' | fzf --header='Pick an instance'"#,
        ])
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit())
        .output()?;

    if !output.status.success() {
        bail!("No instance selected");
    }

    let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let instance_id = selected
        .split_whitespace()
        .next()
        .ok_or_else(|| color_eyre::eyre::eyre!("Could not parse instance ID from selection"))?
        .to_string();

    save_instance_id(&instance_id)?;
    println!("Selected instance {instance_id}");

    Ok(instance_id)
}

fn provision_instance() -> Result<String> {
    println!("Searching for GPU offers...");
    let output = Command::new("vastai")
        .args([
            "search",
            "offers",
            "num_gpus=1 cuda_vers>=12.0 reliability>0.95 dph_total<0.50",
            "-o",
            "dph_total",
            "--raw",
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to search offers: {stderr}");
    }

    let offers: Value = serde_json::from_slice(&output.stdout)?;
    let empty = vec![];
    let offers = offers.as_array().unwrap_or(&empty);
    if offers.is_empty() {
        bail!("No suitable GPU offers found");
    }

    let offer = &offers[0];
    let offer_id = offer["id"].as_u64().unwrap_or(0);
    let gpu_name = offer["gpu_name"].as_str().unwrap_or("unknown");
    let dph = offer["dph_total"].as_f64().unwrap_or(0.0);
    println!("Selected: {gpu_name} @ ${dph:.3}/hr (offer {offer_id})");

    println!("Creating instance...");
    let output = Command::new("vastai")
        .args([
            "create",
            "instance",
            &offer_id.to_string(),
            "--image",
            "nvidia/cuda:12.2.0-devel-ubuntu22.04",
            "--disk",
            "40",
            "--ssh",
            "--direct",
            "--raw",
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to create instance: {stderr}");
    }

    let result: Value = serde_json::from_slice(&output.stdout)?;
    let instance_id = result["new_contract"]
        .as_u64()
        .map(|id| id.to_string())
        .unwrap_or_else(|| result.as_str().unwrap_or("").to_string());

    if instance_id.is_empty() {
        bail!("Could not parse instance ID from: {result}");
    }

    save_instance_id(&instance_id)?;
    println!("Instance {instance_id} created");

    Ok(instance_id)
}

fn wait_for_instance(instance_id: &str) -> Result<()> {
    println!("Waiting for instance {instance_id} to be running...");

    for i in 0..60 {
        std::thread::sleep(std::time::Duration::from_secs(5));

        let output = Command::new("vastai")
            .args(["show", "instance", instance_id, "--raw"])
            .output()?;

        if output.status.success() {
            let info: Value = serde_json::from_slice(&output.stdout)?;
            let status = info["actual_status"].as_str().unwrap_or("");
            if status == "running" {
                println!("Instance is running! (took ~{}s)", (i + 1) * 5);
                return Ok(());
            }
            if i % 6 == 0 {
                println!("  status: {status}, waiting...");
            }
        }
    }

    bail!("Instance did not start within 5 minutes");
}

fn run_setup_script(instance_id: &str) -> Result<()> {
    println!("Running setup script on remote...");
    let mut cmd = ssh_cmd(instance_id)?;

    // forward HF_TOKEN so the remote can download gated models
    let setup_script = if let Ok(token) = std::env::var("HF_TOKEN") {
        format!("export HF_TOKEN={token}\n{SETUP_SCRIPT}")
    } else {
        SETUP_SCRIPT.to_string()
    };

    cmd.arg("bash -s").stdin(std::process::Stdio::piped());

    let mut child = cmd.spawn()?;
    if let Some(ref mut stdin) = child.stdin {
        use std::io::Write;
        stdin.write_all(setup_script.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("Remote setup failed");
    }

    Ok(())
}

pub fn benchmark(args: &[String]) -> Result<()> {
    let instance_id = read_instance_id()?;

    println!("Syncing local changes to remote...");
    let ssh_args = get_ssh_args(&instance_id)?;

    let port = ssh_port(&ssh_args);
    let user_host = ssh_user_host(&ssh_args);

    let root = project_root();
    let src = format!("{}/", root.display());
    let dest = format!("{user_host}:/workspace/speakrs/");

    run_cmd(
        Command::new("rsync")
            .args([
                "-az",
                "--delete",
                "--exclude",
                "target",
                "--exclude",
                "fixtures/models",
                "--exclude",
                "fixtures/datasets",
                "--exclude",
                ".git",
                "-e",
            ])
            .arg(format!(
                "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p {port}"
            ))
            .args([&src, &dest]),
    )?;

    let remote_cmd = if args.is_empty() {
        "cd /workspace/speakrs && source $HOME/.cargo/env && cargo xtask benchmark run fixtures/sample.wav --rust-mode cuda".to_string()
    } else {
        format!(
            "cd /workspace/speakrs && source $HOME/.cargo/env && cargo xtask benchmark {}",
            args.join(" ")
        )
    };

    println!("Running benchmark on remote GPU...");
    let mut cmd = ssh_cmd(&instance_id)?;
    cmd.arg(&remote_cmd);
    run_cmd(&mut cmd)?;

    Ok(())
}

pub fn ssh() -> Result<()> {
    let instance_id = read_instance_id()?;
    let args = get_ssh_args(&instance_id)?;

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

pub fn destroy() -> Result<()> {
    let instance_id = read_instance_id()?;

    println!("Destroying instance {instance_id}...");
    run_cmd(Command::new("vastai").args(["destroy", "instance", &instance_id]))?;

    fs::remove_file(instance_file_path())?;
    println!("Instance destroyed and state file removed");
    Ok(())
}
