use std::process::Command;

use color_eyre::eyre::{Result, bail};
use serde_json::Value;

use super::{Backend, InstanceInfo, models_script_with_token, run_remote_script, ssh_cmd};
use crate::cmd::{project_root, run_cmd};

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

/// Parse SSH connection details from `vastai ssh-url`
///
/// Handles both formats:
///   old: `ssh -p PORT root@HOST`
///   new: `ssh://root@HOST:PORT`
fn get_vastai_ssh(instance_id: &str) -> Result<(String, String)> {
    let output = Command::new("vastai")
        .args(["ssh-url", instance_id])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Failed to get SSH URL: {stderr}");
    }

    let ssh_url = String::from_utf8_lossy(&output.stdout).trim().to_string();

    if let Some(rest) = ssh_url.strip_prefix("ssh://") {
        let (user_host, port) = rest
            .rsplit_once(':')
            .ok_or_else(|| color_eyre::eyre::eyre!("No port in ssh URL: {ssh_url}"))?;
        // user_host is "root@HOST"
        let host = user_host.strip_prefix("root@").unwrap_or(user_host);
        Ok((host.to_string(), port.to_string()))
    } else {
        // ssh -p PORT root@HOST
        let parts: Vec<&str> = ssh_url.split_whitespace().collect();
        if parts.len() < 4 {
            bail!("Unexpected ssh-url format: {ssh_url}");
        }
        let port = parts[2].to_string();
        let host = parts[3]
            .strip_prefix("root@")
            .unwrap_or(parts[3])
            .to_string();
        Ok((host, port))
    }
}

pub fn provision(min_tflops: f64) -> Result<InstanceInfo> {
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

    // ORT prebuilt binaries require AVX-512
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
            super::IMAGE,
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

    println!("Instance {instance_id} created, waiting for it to start...");
    wait_for_running(&instance_id)?;

    let (host, port) = get_vastai_ssh(&instance_id)?;

    Ok(InstanceInfo {
        backend: Backend::VastAi,
        instance_id,
        ssh_host: host,
        ssh_port: port,
        ssh_user: "root".to_string(),
    })
}

fn wait_for_running(instance_id: &str) -> Result<()> {
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

pub fn destroy(info: &InstanceInfo) -> Result<()> {
    println!("Destroying vast.ai instance {}...", info.instance_id);
    run_cmd(Command::new("vastai").args(["destroy", "instance", &info.instance_id]))?;
    println!("Instance destroyed");
    Ok(())
}

pub fn setup(info: &InstanceInfo) -> Result<()> {
    println!("Syncing source to remote (vast.ai)...");
    rsync_source(info)?;

    let script = if let Ok(token) = std::env::var("HF_TOKEN") {
        format!("export HF_TOKEN={token}\n{SETUP_SCRIPT}")
    } else {
        SETUP_SCRIPT.to_string()
    };

    run_remote_script(info, &script)
}

fn rsync_source(info: &InstanceInfo) -> Result<()> {
    let root = project_root();
    let src = format!("{}/", root.display());
    let dest = format!("root@{}:/workspace/speakrs/", info.ssh_host);

    let ssh_opts = format!(
        "ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {}",
        info.ssh_port
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
    )
}

fn sync_datasets(info: &InstanceInfo, args: &[String]) -> Result<()> {
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
    let ssh_opts = format!(
        "ssh -T -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {}",
        info.ssh_port
    );

    for dataset_id in datasets {
        let local_dir = root.join("fixtures/datasets").join(dataset_id);
        if !local_dir.is_dir() {
            continue;
        }
        println!("Syncing dataset {dataset_id} to remote...");
        let src = format!("{}/", local_dir.display());
        let dest = format!(
            "root@{}:/workspace/speakrs/fixtures/datasets/{dataset_id}/",
            info.ssh_host
        );
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

pub fn prepare_benchmark(info: &InstanceInfo, args: &[String]) -> Result<String> {
    rsync_source(info)?;
    sync_datasets(info, args)?;

    // ensure models
    println!("Ensuring models are present on remote...");
    let models_script = format!("set -euo pipefail\n{}", models_script_with_token());
    run_remote_script(info, &models_script)?;

    // incremental build
    println!("Building on remote (incremental)...");
    let mut build_cmd = ssh_cmd(info);
    build_cmd.arg("source $HOME/.cargo/env && cd /workspace/speakrs && cargo build --release -p xtask --features cuda");
    run_cmd(&mut build_cmd)?;

    let remote_cmd = if args.is_empty() {
        "cd /workspace/speakrs && source $HOME/.cargo/env && ./target/release/xtask benchmark run fixtures/sample.wav --rust-mode cuda".to_string()
    } else {
        let quoted_args: Vec<String> = args
            .iter()
            .map(|a| format!("'{}'", a.replace('\'', "'\\''")))
            .collect();
        format!(
            "cd /workspace/speakrs && source $HOME/.cargo/env && ./target/release/xtask benchmark {}",
            quoted_args.join(" ")
        )
    };

    Ok(remote_cmd)
}
