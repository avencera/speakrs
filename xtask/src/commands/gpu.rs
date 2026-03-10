use std::fs;
use std::process::Command;

use color_eyre::eyre::{bail, Result};
use serde_json::Value;

use crate::cmd::{project_root, run_cmd};

const INSTANCE_FILE: &str = ".vastai-instance";

const IMAGE_PREBUILT: &str = "ghcr.io/avencera/speakrs-gpu:latest";
const IMAGE_BARE: &str = "nvidia/cuda:12.9.0-devel-ubuntu24.04";

const MODELS_SCRIPT: &str = r#"
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

const SETUP_SCRIPT: &str = r#"
set -euo pipefail

if [ -n "${HF_TOKEN:-}" ]; then
    uv tool run --from huggingface-hub hf auth login --token "$HF_TOKEN"
fi

MODELS_DIR=/workspace/speakrs/fixtures/models
if [ -f "$MODELS_DIR/segmentation-3.0.onnx" ] && [ -f "$MODELS_DIR/wespeaker-voxceleb-resnet34.onnx" ]; then
    echo "=== Models already downloaded, skipping ==="
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

echo "=== Setup complete ==="
"#;

const BARE_SETUP_SCRIPT: &str = r#"
set -euo pipefail

if ! command -v cargo &>/dev/null; then
    echo "=== Installing build dependencies ==="
    apt-get update && apt-get install -y build-essential pkg-config libssl-dev git cmake curl libopenblas-dev unzip libclang-dev ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswresample-dev libpython3.12-dev libcudnn9-cuda-12

    echo "=== Installing Rust ==="
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
elif ! command -v unzip &>/dev/null || ! dpkg -s libclang-dev &>/dev/null 2>&1 || ! dpkg -s libcudnn9-cuda-12 &>/dev/null 2>&1; then
    echo "=== Installing missing dependencies ==="
    apt-get update && apt-get install -y unzip libclang-dev ffmpeg libavutil-dev libavcodec-dev libavformat-dev libswresample-dev libpython3.12-dev libcudnn9-cuda-12
fi
source $HOME/.cargo/env

if ! command -v uv &>/dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if ! grep -q '.local/bin' "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi

if [ -n "${HF_TOKEN:-}" ]; then
    uv tool run --from huggingface-hub hf auth login --token "$HF_TOKEN"
fi

MODELS_DIR=/workspace/speakrs/fixtures/models
if [ -f "$MODELS_DIR/segmentation-3.0.onnx" ] && [ -f "$MODELS_DIR/wespeaker-voxceleb-resnet34.onnx" ]; then
    echo "=== Models already downloaded, skipping ==="
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

echo "=== Building (initial, may be slow) ==="
cd /workspace/speakrs
cargo build --release -p xtask --features cuda

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

pub fn setup(new: bool, bare: bool, min_tflops: f64) -> Result<()> {
    let instance_id = if new {
        provision_instance(bare, min_tflops)?
    } else {
        pick_instance()?
    };

    wait_for_instance(&instance_id)?;
    rsync_source(&instance_id)?;
    run_setup_script(&instance_id, bare)?;

    println!("GPU instance ready! Use `cargo xtask gpu ssh` to connect");
    Ok(())
}

/// If only one running instance exists, use it. Otherwise prompt with fzf
fn pick_or_read_instance() -> Result<String> {
    let output = Command::new("vastai")
        .args(["show", "instances", "--raw"])
        .output()?;

    if !output.status.success() {
        return read_instance_id();
    }

    let instances: Value = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
    let empty = vec![];
    let running: Vec<&Value> = instances
        .as_array()
        .unwrap_or(&empty)
        .iter()
        .filter(|i| i["actual_status"].as_str().is_some_and(|s| s == "running"))
        .collect();

    match running.len() {
        0 => bail!("No running instances found. Run `cargo xtask gpu setup` first"),
        1 => {
            let id = running[0]["id"].as_u64().unwrap_or(0).to_string();
            save_instance_id(&id)?;
            Ok(id)
        }
        _ => pick_instance(),
    }
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

fn provision_instance(bare: bool, min_tflops: f64) -> Result<String> {
    println!("Searching for GPU offers (min {min_tflops} TFLOPS)...");

    let query = format!(
        "num_gpus=1 compute_cap>=700 cuda_vers>=12.4 reliability>0.95 dph_total<1.00 total_flops>={min_tflops} geolocation in [US,CA] inet_down>=4000"
    );

    let output = Command::new("vastai")
        .args(["search", "offers", &query, "-o", "dph_total", "--raw"])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to search offers: {stderr}");
    }

    let offers: Value = serde_json::from_slice(&output.stdout)?;
    let empty = vec![];
    let offers = offers.as_array().unwrap_or(&empty);

    // ORT prebuilt binaries require AVX-512 — filter to CPUs that have it
    // (Xeon Scalable Gold/Platinum/Silver, EPYC 9xxx/8xxx Zen4+)
    let avx512_prefixes = [
        "Gold", "Platinum", "Silver", "Xeon W-", "Xeon w9-", "EPYC 9", "EPYC 8",
    ];
    let offer = offers.iter().find(|o| {
        let cpu = o["cpu_name"].as_str().unwrap_or("");
        avx512_prefixes.iter().any(|p| cpu.contains(p))
    });
    let Some(offer) = offer else {
        bail!(
            "No suitable GPU offers found (need AVX-512 CPU: Xeon Gold/Platinum or EPYC 9xxx/8xxx)"
        );
    };
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
            if bare { IMAGE_BARE } else { IMAGE_PREBUILT },
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

fn run_setup_script(instance_id: &str, bare: bool) -> Result<()> {
    println!("Running setup script on remote...");
    let mut cmd = ssh_cmd(instance_id)?;

    let script = if bare {
        BARE_SETUP_SCRIPT
    } else {
        SETUP_SCRIPT
    };

    // forward HF_TOKEN so the remote can download gated models
    let setup_script = if let Ok(token) = std::env::var("HF_TOKEN") {
        format!("export HF_TOKEN={token}\n{script}")
    } else {
        script.to_string()
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

fn ensure_remote_models(instance_id: &str) -> Result<()> {
    println!("Ensuring models are present on remote...");
    let mut cmd = ssh_cmd(instance_id)?;
    cmd.arg("bash -s").stdin(std::process::Stdio::piped());

    let script = if let Ok(token) = std::env::var("HF_TOKEN") {
        format!("set -euo pipefail\nexport HF_TOKEN={token}\n{MODELS_SCRIPT}")
    } else {
        format!("set -euo pipefail\n{MODELS_SCRIPT}")
    };

    let mut child = cmd.spawn()?;
    if let Some(ref mut stdin) = child.stdin {
        use std::io::Write;
        stdin.write_all(script.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("Failed to download models on remote");
    }

    Ok(())
}

fn sync_datasets(instance_id: &str, args: &[String]) -> Result<()> {
    // extract --dataset values from args
    let datasets: Vec<&str> = args
        .windows(2)
        .filter_map(|w| {
            if w[0] == "--dataset" {
                Some(w[1].as_str())
            } else {
                None
            }
        })
        .collect();

    if datasets.is_empty() {
        return Ok(());
    }

    let root = project_root();
    let ssh_args = get_ssh_args(instance_id)?;
    let port = ssh_port(&ssh_args);
    let user_host = ssh_user_host(&ssh_args);
    let ssh_opts = format!(
        "ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}"
    );

    for dataset_id in datasets {
        let local_dir = root.join("fixtures/datasets").join(dataset_id);
        if !local_dir.is_dir() {
            continue;
        }
        println!("Syncing dataset {dataset_id} to remote...");
        let src = format!("{}/", local_dir.display());
        let dest = format!("{user_host}:/workspace/speakrs/fixtures/datasets/{dataset_id}/");
        run_cmd(
            Command::new("rsync")
                .args([
                    "-az",
                    "--info=progress2",
                    "--compress-level=1",
                    "--rsync-path",
                    "mkdir -p /workspace/speakrs/fixtures/datasets && rsync",
                    "-e",
                ])
                .arg(&ssh_opts)
                .args([&src, &dest]),
        )?;
    }

    Ok(())
}

fn rsync_source(instance_id: &str) -> Result<()> {
    println!("Syncing source to remote...");
    let ssh_args = get_ssh_args(instance_id)?;
    let port = ssh_port(&ssh_args);
    let user_host = ssh_user_host(&ssh_args);

    let root = project_root();
    let src = format!("{}/", root.display());
    let dest = format!("{user_host}:/workspace/speakrs/");

    // -T disables PTY so the remote login banner doesn't corrupt rsync's data stream
    let ssh_opts = format!(
        "ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}"
    );

    run_cmd(
        Command::new("rsync")
            .args([
                "-az",
                "--info=progress2",
                "--compress-level=1",
                "--delete",
                "--exclude",
                "target",
                "--exclude",
                "fixtures/models",
                "--exclude",
                "fixtures/datasets",
                "--exclude",
                ".git",
                "--exclude",
                ".venv",
                "--exclude",
                ".claude",
                "--exclude",
                ".ruff_cache",
                "--exclude",
                "scripts/native_coreml",
                "--exclude",
                "scripts/fluidaudio-bench",
                "--exclude",
                "scripts/pyannote_rs_bench",
                "--exclude",
                "_benchmarks",
                "--exclude",
                "adr",
                "--exclude",
                "notes",
                "--exclude",
                "benchmarks",
                "--exclude",
                "examples",
                "--rsync-path",
                "mkdir -p /workspace/speakrs && rsync",
                "-e",
            ])
            .arg(&ssh_opts)
            .args([&src, &dest]),
    )?;

    Ok(())
}

pub fn benchmark(args: &[String]) -> Result<()> {
    let instance_id = pick_or_read_instance()?;

    rsync_source(&instance_id)?;

    // sync local datasets to remote if present
    sync_datasets(&instance_id, args)?;

    // ensure models are present
    ensure_remote_models(&instance_id)?;

    // incremental build on remote — only recompiles changed crates
    println!("Building on remote (incremental)...");
    let mut build_cmd = ssh_cmd(&instance_id)?;
    build_cmd.arg("source $HOME/.cargo/env && cd /workspace/speakrs && cargo build --release -p xtask --features cuda");
    run_cmd(&mut build_cmd)?;

    let remote_cmd = if args.is_empty() {
        "cd /workspace/speakrs && source $HOME/.cargo/env && ./target/release/xtask benchmark run fixtures/sample.wav --rust-mode cuda".to_string()
    } else {
        format!(
            "cd /workspace/speakrs && source $HOME/.cargo/env && ./target/release/xtask benchmark {}",
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
