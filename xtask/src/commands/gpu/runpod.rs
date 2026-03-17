use std::collections::HashSet;
use std::process::{Command, Stdio};

use color_eyre::eyre::{Result, bail};
use serde::Deserialize;

use super::{Backend, GITHUB_REPO, InstanceInfo, image};
use std::io::Write;

const RUNPOD_API_URL: &str = "https://api.runpod.io/graphql";

// ---------------------------------------------------------------------------
// GraphQL response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GqlResponse<T> {
    data: Option<T>,
    errors: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Port {
    ip: Option<String>,
    is_ip_public: Option<bool>,
    private_port: Option<u16>,
    public_port: Option<u16>,
}

#[derive(Deserialize)]
struct Runtime {
    ports: Option<Vec<Port>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Machine {
    pod_host_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Pod {
    id: Option<String>,
    name: Option<String>,
    desired_status: Option<String>,
    machine: Option<Machine>,
    runtime: Option<Runtime>,
}

// query wrappers, one per distinct shape

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreatePodData {
    pod_find_and_deploy_on_demand: CreatedPod,
}

#[derive(Deserialize)]
struct CreatedPod {
    id: String,
}

#[derive(Deserialize)]
struct PodQueryData {
    pod: Pod,
}

#[derive(Deserialize)]
struct PodsQueryData {
    myself: Myself,
}

#[derive(Deserialize)]
struct Myself {
    pods: Vec<Pod>,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

fn runpod_api_key() -> Result<String> {
    std::env::var("RUNPOD_API_KEY")
        .map_err(|_| color_eyre::eyre::eyre!("RUNPOD_API_KEY not set. Add it to .envrc"))
}

/// Send a GraphQL request, returning the raw JSON (caller checks errors)
fn graphql_request(
    query: &str,
    variables: &serde_json::Value,
) -> Result<GqlResponse<serde_json::Value>> {
    let api_key = runpod_api_key()?;
    let body = serde_json::json!({
        "query": query,
        "variables": variables,
    });

    let response: GqlResponse<serde_json::Value> = ureq::post(RUNPOD_API_URL)
        .header("Authorization", &format!("Bearer {api_key}"))
        .send_json(&body)?
        .body_mut()
        .read_json()?;

    Ok(response)
}

fn graphql_query<T: serde::de::DeserializeOwned>(
    query: &str,
    variables: &serde_json::Value,
) -> Result<T> {
    let response = graphql_request(query, variables)?;

    if let Some(errors) = response.errors {
        bail!("GraphQL errors: {errors}");
    }

    let data = response
        .data
        .ok_or_else(|| color_eyre::eyre::eyre!("No data in GraphQL response"))?;

    Ok(serde_json::from_value(data)?)
}

// ---------------------------------------------------------------------------
// SSH info extraction
// ---------------------------------------------------------------------------

impl Pod {
    /// Extract SSH connection from a pod's port list
    ///
    /// Two modes:
    /// 1. Public IP: `runtime.ports` has an entry with `privatePort: 22`
    ///    → (ip, publicPort, "root")
    /// 2. SSH proxy: ports is null (SECURE cloud type), use `machine.podHostId`
    ///    → ("ssh.runpod.io", "22", podHostId)
    fn ssh_info(&self) -> Option<(String, String, String)> {
        if let Some(runtime) = &self.runtime
            && let Some(ports) = &runtime.ports
        {
            let ssh_port = ports
                .iter()
                .find(|p| p.private_port == Some(22) && p.is_ip_public == Some(true))
                .or_else(|| ports.iter().find(|p| p.private_port == Some(22)));

            if let Some(entry) = ssh_port {
                let host = entry.ip.clone().unwrap_or_default();
                let port = entry.public_port.unwrap_or(0).to_string();
                if !host.is_empty() && port != "0" {
                    return Some((host, port, "root".to_string()));
                }
            }
        }

        let pod_host_id = self.machine.as_ref()?.pod_host_id.as_ref()?;
        Some((
            "ssh.runpod.io".to_string(),
            "22".to_string(),
            pod_host_id.clone(),
        ))
    }

    fn direct_ssh(&self) -> Option<(String, String)> {
        let ports = self.runtime.as_ref()?.ports.as_ref()?;
        let ssh_port = ports
            .iter()
            .find(|p| p.private_port == Some(22) && p.is_ip_public == Some(true))
            .or_else(|| ports.iter().find(|p| p.private_port == Some(22)));

        let entry = ssh_port?;
        let host = entry.ip.clone().unwrap_or_default();
        let port = entry.public_port.unwrap_or(0).to_string();
        if !host.is_empty() && port != "0" {
            Some((host, port))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Provision / wait
// ---------------------------------------------------------------------------

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
                "imageName": image(),
                "gpuTypeId": gpu_type,
                "cloudType": "SECURE",
                "gpuCount": 1,
                "volumeInGb": 50,
                "containerDiskInGb": 100,
                "volumeMountPath": "/workspace",
                "ports": "22/tcp",
                "startSsh": true,
                "minCudaVersion": "12.8",
            }
        });

        let response = graphql_request(query, &variables)?;

        // check for supply constraint error, try next GPU
        if let Some(errors) = response.errors {
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

        let data: CreatePodData = serde_json::from_value(
            response
                .data
                .ok_or_else(|| color_eyre::eyre::eyre!("No data in create response"))?,
        )?;

        let pod_id = data.pod_find_and_deploy_on_demand.id;
        println!("Pod {pod_id} created with {gpu_type}, waiting for SSH...");
        let info = wait_for_ssh(&pod_id)?;
        // TODO: re-enable wait_for_container_ready after image rebuild with new entrypoint
        return Ok(info);
    }

    bail!("All GPU types exhausted, none available on RunPod")
}

fn wait_for_ssh(pod_id: &str) -> Result<InstanceInfo> {
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

    for i in 0..90 {
        std::thread::sleep(std::time::Duration::from_secs(5));

        let data: PodQueryData = graphql_query(query, &serde_json::json!({ "podId": pod_id }))?;
        let status = data.pod.desired_status.as_deref().unwrap_or("");

        if status != "RUNNING" {
            if i % 6 == 0 {
                println!("  status: {status}, waiting...");
            }
            continue;
        }

        if let Some((host, port, user)) = data.pod.ssh_info() {
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

/// Wait until the entrypoint has finished (marker file exists)
///
/// RunPod's SSH proxy returns exit code 0 even when the container isn't
/// running ("container not found"), so we check stdout instead
#[allow(dead_code)]
pub fn wait_for_container_ready(info: &super::InstanceInfo) -> Result<()> {
    println!("Waiting for container to be ready...");
    for i in 0..60 {
        let mut cmd = info.ssh_cmd();
        cmd.args(["cat", "/tmp/.container-ready"]);
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        if let Ok(output) = cmd.output() {
            let combined = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );

            if !combined.contains("container not found")
                && !combined.contains("No such file")
                && output.status.success()
            {
                println!("Container ready (took ~{}s)", (i + 1) * 5);
                return Ok(());
            }
        }

        if i % 6 == 0 {
            println!("  entrypoint still running, waiting...");
        }
        std::thread::sleep(std::time::Duration::from_secs(5));
    }

    bail!("Container did not become ready within 5 minutes");
}

// ---------------------------------------------------------------------------
// Pod queries
// ---------------------------------------------------------------------------

fn query_pods() -> Result<Vec<Pod>> {
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

    let data: PodsQueryData = graphql_query(query, &serde_json::json!({}))?;
    Ok(data.myself.pods)
}

pub fn get_pod_ids() -> Result<HashSet<String>> {
    let pods = query_pods()?;
    Ok(pods.iter().filter_map(|p| p.id.clone()).collect())
}

/// Get the direct (non-proxy) SSH connection for a running pod
pub fn direct_ssh_info(instance_id: &str) -> Result<(String, String)> {
    let query = r#"
        query Pod($podId: String!) {
            pod(input: { podId: $podId }) {
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

    let data: PodQueryData = graphql_query(query, &serde_json::json!({ "podId": instance_id }))?;

    data.pod
        .direct_ssh()
        .ok_or_else(|| color_eyre::eyre::eyre!("No direct SSH port found for pod {instance_id}"))
}

pub fn import() -> Result<InstanceInfo> {
    let pods = query_pods()?;

    let running: Vec<&Pod> = pods
        .iter()
        .filter(|p| p.desired_status.as_deref() == Some("RUNNING"))
        .collect();

    if running.is_empty() {
        bail!("No running pods found on RunPod");
    }

    let pod = if running.len() == 1 {
        let only = running[0];
        let name = only.name.as_deref().unwrap_or("unnamed");
        let id = only.id.as_deref().unwrap_or("?");
        println!("Found one running pod: {id} | {name}");
        only
    } else {
        let descriptions: Vec<String> = running
            .iter()
            .map(|p| {
                let id = p.id.as_deref().unwrap_or("?");
                let name = p.name.as_deref().unwrap_or("unnamed");
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
            .find(|p| p.id.as_deref() == Some(selected_id))
            .ok_or_else(|| color_eyre::eyre::eyre!("Selected pod not found"))?
    };

    let pod_id = pod
        .id
        .as_ref()
        .ok_or_else(|| color_eyre::eyre::eyre!("Pod missing id"))?
        .clone();

    let (host, port, user) = pod
        .ssh_info()
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

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

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

    graphql_query::<serde_json::Value>(query, &variables)?;
    println!("Resume requested, waiting for SSH...");
    let new_info = wait_for_ssh(&info.instance_id)?;
    // TODO: re-enable wait_for_container_ready after image rebuild with new entrypoint
    Ok(new_info)
}

pub fn destroy(info: &InstanceInfo) -> Result<()> {
    println!("Terminating RunPod pod {}...", info.instance_id);

    let query = r#"
        mutation TerminatePod($podId: String!) {
            podTerminate(input: { podId: $podId })
        }
    "#;

    graphql_query::<serde_json::Value>(query, &serde_json::json!({ "podId": info.instance_id }))?;
    println!("Pod terminated");
    Ok(())
}

fn provision_ssh_key(info: &InstanceInfo) -> Result<()> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let pub_key_path = format!("{home}/.ssh/id_ed25519.pub");

    let pub_key = match std::fs::read_to_string(&pub_key_path) {
        Ok(key) => key.trim().to_string(),
        Err(_) => {
            println!("No SSH public key found at {pub_key_path}, skipping");
            return Ok(());
        }
    };

    let script = format!(
        "mkdir -p /root/.ssh && \
         chmod 700 /root/.ssh && \
         grep -qF '{pub_key}' /root/.ssh/authorized_keys 2>/dev/null || \
         echo '{pub_key}' >> /root/.ssh/authorized_keys && \
         chmod 600 /root/.ssh/authorized_keys",
    );

    info.run_remote_script(&script)?;
    println!("SSH key provisioned for direct access");
    Ok(())
}

pub fn setup(info: &InstanceInfo, branch: &str) -> Result<()> {
    println!("Running setup on remote (RunPod)...");
    println!("Binary and models are baked into the image.");
    println!("Setup provisions S3 creds, SSH key, and clones the repo.");

    info.provision_env_vars()?;
    info.provision_terminfo()?;
    provision_ssh_key(info)?;

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
    );

    info.run_remote_script(&script)
}
