use std::process::Command;

use color_eyre::eyre::{Result, bail};
use serde_json::Value;

use super::{Backend, GITHUB_REPO, IMAGE, InstanceInfo, run_remote_script};
use std::io::Write;

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
                "minCudaVersion": "12.8",
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
                    machine {
                        podHostId
                    }
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

        if let Some((host, port, user)) = extract_ssh_info(pod) {
            println!(
                "Pod is running! SSH at {user}@{host}:{port} (took ~{}s)",
                (i + 1) * 5
            );
            return Ok(InstanceInfo {
                backend: Backend::RunPod,
                instance_id: pod_id.to_string(),
                ssh_host: host,
                ssh_port: port,
                ssh_user: user,
            });
        }

        if i % 6 == 0 {
            println!("  pod running, waiting for SSH port...");
        }
    }

    bail!("Pod did not become ready within 7.5 minutes");
}

/// Extract SSH connection info from a RunPod pod object
///
/// Two modes:
/// 1. Public IP: `runtime.ports` has an entry with `privatePort: 22`
///    → (ip, publicPort, "root")
/// 2. SSH proxy: ports is null (SECURE cloud type) — use `machine.podHostId`
///    → ("ssh.runpod.io", "22", podHostId)
fn extract_ssh_info(pod: &Value) -> Option<(String, String, String)> {
    // try public IP ports first
    if let Some(ports) = pod["runtime"]["ports"].as_array() {
        let ssh_port = ports
            .iter()
            .find(|p| {
                p["privatePort"].as_u64() == Some(22) && p["isIpPublic"].as_bool() == Some(true)
            })
            .or_else(|| ports.iter().find(|p| p["privatePort"].as_u64() == Some(22)));

        if let Some(p) = ssh_port {
            let host = p["ip"].as_str().unwrap_or("").to_string();
            let port = p["publicPort"].as_u64().unwrap_or(0).to_string();
            if !host.is_empty() && port != "0" {
                return Some((host, port, "root".to_string()));
            }
        }
    }

    // fallback: SSH proxy via podHostId
    let pod_host_id = pod["machine"]["podHostId"].as_str()?;
    Some((
        "ssh.runpod.io".to_string(),
        "22".to_string(),
        pod_host_id.to_string(),
    ))
}

pub fn list_pods() -> Result<Value> {
    let query = r#"
        query Pods {
            myself {
                pods {
                    id
                    name
                    desiredStatus
                    machine {
                        podHostId
                    }
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
        }
    "#;

    let response = graphql_query(query, &serde_json::json!({}))?;
    Ok(response)
}

pub fn import() -> Result<InstanceInfo> {
    let response = list_pods()?;
    let pods = response["data"]["myself"]["pods"]
        .as_array()
        .ok_or_else(|| color_eyre::eyre::eyre!("Could not parse pods from response"))?;

    let running: Vec<&Value> = pods
        .iter()
        .filter(|p| p["desiredStatus"].as_str() == Some("RUNNING"))
        .collect();

    if running.is_empty() {
        bail!("No running pods found on RunPod");
    }

    let pod = if running.len() == 1 {
        let p = running[0];
        let name = p["name"].as_str().unwrap_or("unnamed");
        let id = p["id"].as_str().unwrap_or("?");
        println!("Found one running pod: {id} | {name}");
        p
    } else {
        let descriptions: Vec<String> = running
            .iter()
            .map(|p| {
                let id = p["id"].as_str().unwrap_or("?");
                let name = p["name"].as_str().unwrap_or("unnamed");
                format!("{id} | {name}")
            })
            .collect();

        let input = descriptions.join("\n");
        let output = Command::new("fzf")
            .arg("--header=Pick a pod to import")
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
            bail!("No pod selected");
        }

        let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let selected_id = selected.split('|').next().map(|s| s.trim()).unwrap_or("");

        running
            .iter()
            .find(|p| p["id"].as_str() == Some(selected_id))
            .ok_or_else(|| color_eyre::eyre::eyre!("Selected pod not found"))?
    };

    let pod_id = pod["id"]
        .as_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("Pod missing id"))?
        .to_string();

    let (host, port, user) = extract_ssh_info(pod)
        .ok_or_else(|| color_eyre::eyre::eyre!("Pod {pod_id} has no SSH port available"))?;

    println!("SSH at {user}@{host}:{port}");

    Ok(InstanceInfo {
        backend: Backend::RunPod,
        instance_id: pod_id,
        ssh_host: host,
        ssh_port: port,
        ssh_user: user,
    })
}

pub fn resume(info: &InstanceInfo) -> Result<InstanceInfo> {
    println!("Resuming RunPod pod {}...", info.instance_id);

    let query = r#"
        mutation ResumePod($input: PodResumeInput!) {
            podResume(input: $input) {
                id
                desiredStatus
            }
        }
    "#;

    let variables = serde_json::json!({
        "input": {
            "podId": info.instance_id,
            "gpuCount": 1,
        }
    });

    graphql_query(query, &variables)?;
    println!("Resume requested, waiting for SSH...");
    wait_for_ssh(&info.instance_id)
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
    println!("Binary and models are baked into the image.");
    println!("Setup only clones the repo for the pyannote script (if not present).");

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
         echo '=== Verifying baked binary ==='\n\
         which speakrs-bm && speakrs-bm --help | head -1 || echo 'warning: speakrs-bm not found in PATH'\n\
         echo '=== Verifying baked models ==='\n\
         ls /workspace/models/segmentation-3.0.onnx /workspace/models/wespeaker-voxceleb-resnet34.onnx 2>/dev/null \
           && echo 'Models present' || echo 'warning: models not found at /workspace/models/'\n\
         echo '=== Cloning repo (for pyannote script) ==='\n\
         {clone_or_pull}\n\
         echo '=== Setup complete ==='\n",
        clone_or_pull = clone_or_pull,
    );

    run_remote_script(info, &script)
}
