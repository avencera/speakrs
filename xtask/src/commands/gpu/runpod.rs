use std::process::Command;

use color_eyre::eyre::{Result, bail};
use serde_json::Value;

use super::{
    Backend, GITHUB_REPO, IMAGE, InstanceInfo, models_script_with_token, run_remote_script,
};
use crate::cmd::project_root;

const RUNPOD_API_URL: &str = "https://api.runpod.io/graphql";

fn runpod_api_key() -> Result<String> {
    std::env::var("RUNPOD_API_KEY")
        .map_err(|_| color_eyre::eyre::eyre!("RUNPOD_API_KEY not set. Add it to .envrc"))
}

/// Send a GraphQL request, returning the raw JSON (caller checks errors)
fn graphql_request(query: &str, variables: &Value) -> Result<Value> {
    let api_key = runpod_api_key()?;
    let body = serde_json::json!({
        "query": query,
        "variables": variables,
    });

    let response: Value = ureq::post(RUNPOD_API_URL)
        .header("Authorization", &format!("Bearer {api_key}"))
        .send_json(&body)?
        .body_mut()
        .read_json()?;

    Ok(response)
}

fn graphql_query(query: &str, variables: &Value) -> Result<Value> {
    let response = graphql_request(query, variables)?;

    if let Some(errors) = response.get("errors") {
        bail!("GraphQL errors: {errors}");
    }

    Ok(response)
}

pub fn provision(name: &str, gpu_types: &[&str]) -> Result<InstanceInfo> {
    let query = r#"
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
            }
        }
    "#;

    for (i, gpu_type) in gpu_types.iter().enumerate() {
        println!("Trying {gpu_type}...");

        let variables = serde_json::json!({
            "input": {
                "name": format!("speakrs-bench-{name}"),
                "imageName": IMAGE,
                "gpuTypeId": gpu_type,
                "cloudType": "SECURE",
                "gpuCount": 1,
                "volumeInGb": 40,
                "containerDiskInGb": 20,
                "volumeMountPath": "/workspace",
                "ports": "22/tcp",
                "startSsh": true,
            }
        });

        let response = graphql_request(query, &variables)?;

        // check for supply constraint error — try next GPU
        if let Some(errors) = response.get("errors") {
            let err_str = errors.to_string();
            if err_str.contains("SUPPLY_CONSTRAINT") || err_str.contains("no available") {
                println!(
                    "  {gpu_type} unavailable, {}",
                    if i + 1 < gpu_types.len() {
                        "trying next..."
                    } else {
                        "no more fallbacks"
                    }
                );
                continue;
            }
            bail!("GraphQL errors: {errors}");
        }

        let pod_id = response["data"]["podFindAndDeployOnDemand"]["id"]
            .as_str()
            .ok_or_else(|| {
                color_eyre::eyre::eyre!("Could not parse pod ID from response: {response}")
            })?
            .to_string();

        println!("Pod {pod_id} created with {gpu_type}, waiting for SSH...");
        return wait_for_ssh(&pod_id);
    }

    bail!("All GPU types exhausted — none available on RunPod")
}

fn wait_for_ssh(pod_id: &str) -> Result<InstanceInfo> {
    for i in 0..90 {
        std::thread::sleep(std::time::Duration::from_secs(5));

        let query = r#"
            query Pod($podId: String!) {
                pod(input: { podId: $podId }) {
                    desiredStatus
                    runtime {
                        ports {
                            ip
                            isIpPublic
                            privatePort
                            publicPort
                        }
                    }
                }
            }
        "#;

        let response = graphql_query(query, &serde_json::json!({ "podId": pod_id }))?;
        let pod = &response["data"]["pod"];
        let status = pod["desiredStatus"].as_str().unwrap_or("");

        if status != "RUNNING" {
            if i % 6 == 0 {
                println!("  status: {status}, waiting...");
            }
            continue;
        }

        let ports = pod["runtime"]["ports"].as_array();
        let ssh_port = ports.and_then(|ports| {
            ports.iter().find(|p| {
                p["privatePort"].as_u64() == Some(22) && p["isIpPublic"].as_bool() == Some(true)
            })
        });

        if let Some(port_info) = ssh_port {
            let host = port_info["ip"].as_str().unwrap_or("").to_string();
            let port = port_info["publicPort"].as_u64().unwrap_or(0).to_string();

            if !host.is_empty() && port != "0" {
                println!(
                    "Pod is running! SSH at {host}:{port} (took ~{}s)",
                    (i + 1) * 5
                );
                return Ok(InstanceInfo {
                    backend: Backend::RunPod,
                    instance_id: pod_id.to_string(),
                    ssh_host: host,
                    ssh_port: port,
                });
            }
        }

        if i % 6 == 0 {
            println!("  pod running, waiting for SSH port...");
        }
    }

    bail!("Pod did not become ready within 7.5 minutes");
}

pub fn destroy(info: &InstanceInfo) -> Result<()> {
    println!("Terminating RunPod pod {}...", info.instance_id);

    let query = r#"
        mutation TerminatePod($podId: String!) {
            podTerminate(input: { podId: $podId })
        }
    "#;

    graphql_query(query, &serde_json::json!({ "podId": info.instance_id }))?;
    println!("Pod terminated");
    Ok(())
}

pub fn setup(info: &InstanceInfo, branch: &str) -> Result<()> {
    println!("Running setup on remote (RunPod)...");

    let clone_or_pull = format!(
        "if [ -d /workspace/speakrs/.git ]; then \
         cd /workspace/speakrs && git fetch origin && git reset --hard origin/{branch}; \
         else \
         git clone {repo} /workspace/speakrs && cd /workspace/speakrs && git checkout {branch}; \
         fi",
        branch = branch,
        repo = GITHUB_REPO,
    );

    let script = format!(
        "set -euo pipefail\n\
         {clone_or_pull}\n\
         {models}\n\
         echo '=== Building ==='\n\
         cd /workspace/speakrs\n\
         source $HOME/.cargo/env 2>/dev/null || true\n\
         cargo build --release -p xtask --features cuda\n\
         echo '=== Setup complete ==='\n",
        clone_or_pull = clone_or_pull,
        models = models_script_with_token(),
    );

    run_remote_script(info, &script)
}

pub fn prepare_benchmark(info: &InstanceInfo, args: &[String], branch: &str) -> Result<String> {
    let local_head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(project_root())
        .output()?;
    let remote_head = Command::new("git")
        .args(["rev-parse", &format!("origin/{branch}")])
        .current_dir(project_root())
        .output()?;

    if local_head.status.success() && remote_head.status.success() {
        let local = String::from_utf8_lossy(&local_head.stdout)
            .trim()
            .to_string();
        let remote = String::from_utf8_lossy(&remote_head.stdout)
            .trim()
            .to_string();
        if local != remote {
            println!(
                "WARNING: local HEAD ({}) differs from origin/{branch} ({})",
                &local[..8],
                &remote[..8]
            );
            println!("  Push your changes before benchmarking: git push");
        }
    }

    let clone_or_pull =
        format!("cd /workspace/speakrs && git fetch origin && git reset --hard origin/{branch}",);

    let prep_script = format!(
        "set -euo pipefail\n\
         source $HOME/.cargo/env 2>/dev/null || true\n\
         {clone_or_pull}\n\
         {models}\n\
         echo '=== Building ==='\n\
         cd /workspace/speakrs\n\
         cargo build --release -p xtask --features cuda\n",
        clone_or_pull = clone_or_pull,
        models = models_script_with_token(),
    );

    println!("Preparing remote (git pull + build)...");
    run_remote_script(info, &prep_script)?;

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
