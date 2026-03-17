mod runpod;
mod vastai;

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::str::FromStr;

use color_eyre::eyre::{Result, bail};

use crate::cmd::project_root;
#[cfg(not(unix))]
use crate::cmd::run_cmd;

pub const IMAGE: &str = "ghcr.io/avencera/speakrs-gpu:latest";
pub const GITHUB_REPO: &str = "https://github.com/avencera/speakrs.git";

pub const RUNPOD_GPU_FALLBACKS: &[&str] = &[
    // ≥70 TFLOPS, sorted by price
    "NVIDIA GeForce RTX 4090",        // 83 TFLOPS, $0.59/hr
    "NVIDIA RTX 6000 Ada Generation", // 91 TFLOPS, $0.77/hr
    "NVIDIA L40S",                    // 91 TFLOPS, $0.86/hr
    "NVIDIA GeForce RTX 5090",        // 105 TFLOPS, $0.89/hr
    "NVIDIA L40",                     // 90 TFLOPS, $0.99/hr
    // <70 TFLOPS fallbacks
    "NVIDIA RTX A6000",        // 39 TFLOPS, $0.49/hr
    "NVIDIA A40",              // 37 TFLOPS, $0.40/hr
    "NVIDIA GeForce RTX 3090", // 36 TFLOPS, $0.46/hr
];

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
    pub ssh_user: String,
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
        bail!(
            "Instance file '{name}' is malformed (expected 4+ lines: backend, id, host, port, [user])"
        );
    }

    Ok(InstanceInfo {
        backend: lines[0].parse()?,
        instance_id: lines[1].to_string(),
        ssh_host: lines[2].to_string(),
        ssh_port: lines[3].to_string(),
        ssh_user: lines.get(4).unwrap_or(&"root").to_string(),
    })
}

fn save_instance(name: &str, info: &InstanceInfo) -> Result<()> {
    let dir = instances_dir();
    fs::create_dir_all(&dir)?;
    let content = format!(
        "{}\n{}\n{}\n{}\n{}\n",
        info.backend, info.instance_id, info.ssh_host, info.ssh_port, info.ssh_user
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
        .filter(|n| !n.ends_with(".setup-done"))
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
        format!("{}@{}", info.ssh_user, info.ssh_host),
    ]
}

fn is_proxy(info: &InstanceInfo) -> bool {
    info.ssh_host == "ssh.runpod.io"
}

/// SSH command — adds `-tt` (PTY) only for RunPod proxy connections
pub fn ssh_cmd(info: &InstanceInfo) -> Command {
    let mut cmd = Command::new("ssh");
    if is_proxy(info) {
        cmd.arg("-tt");
    }
    cmd.args([
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
    ]);
    cmd.args(get_ssh_args(info));
    cmd
}

pub fn run_remote_script(info: &InstanceInfo, script: &str) -> Result<()> {
    if is_proxy(info) {
        let mut cmd = ssh_cmd(info);
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(script.as_bytes())?;
            stdin.write_all(b"\nexit\n")?;
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("Remote script failed");
        }
    } else {
        let mut cmd = ssh_cmd(info);
        cmd.args(["bash", "-s"]);
        cmd.stdin(Stdio::piped());
        let mut child = cmd.spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(script.as_bytes())?;
        }
        let status = child.wait()?;
        if !status.success() {
            bail!("Remote script failed");
        }
    }
    Ok(())
}

// --- setup marker ---

fn setup_done_path(name: &str) -> PathBuf {
    instances_dir().join(format!("{name}.setup-done"))
}

fn mark_setup_done(name: &str) {
    let _ = fs::write(setup_done_path(name), "");
}

fn is_setup_done(name: &str) -> bool {
    setup_done_path(name).exists()
}

fn current_branch() -> Result<String> {
    let head = fs::read_to_string(project_root().join(".git/HEAD"))?;
    Ok(head
        .strip_prefix("ref: refs/heads/")
        .unwrap_or(head.trim())
        .trim()
        .to_string())
}

const ENV_VARS: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ENDPOINT_URL",
    "AWS_REGION",
    "HF_TOKEN",
];

/// Write local env vars to `/root/.env` on the remote and ensure `.bashrc` sources it
pub fn provision_env_vars(info: &InstanceInfo) -> Result<()> {
    let mut lines = Vec::new();
    for var in ENV_VARS {
        if let Ok(val) = std::env::var(var) {
            lines.push(format!("export {var}=\"{val}\""));
        }
    }

    if lines.is_empty() {
        println!("No S3/HF env vars found locally, skipping");
        return Ok(());
    }

    let count = lines.len();
    let env_content = lines.join("\n");

    // content built in Rust, heredoc is just the SSH transport
    let script = format!(
        "cat > /root/.env << 'SPEAKRS_EOF'\n{env_content}\nSPEAKRS_EOF\n\
         chmod 600 /root/.env\n\
         grep -q 'source /root/.env' /root/.bashrc 2>/dev/null || \
         echo '[ -f /root/.env ] && source /root/.env' >> /root/.bashrc",
    );
    run_remote_script(info, &script)?;

    println!("Wrote /root/.env with {count} var(s)");
    Ok(())
}

// --- public API ---

pub fn create(name: &str, backend: Backend, gpu_types: &[&str], min_tflops: f64) -> Result<()> {
    if instances_dir().join(name).exists() {
        bail!("Instance '{name}' already exists. Destroy it first or pick a different name");
    }

    let info = match backend {
        Backend::RunPod => runpod::provision(name, gpu_types)?,
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

    mark_setup_done(&n);
    println!("Instance '{n}' setup complete");
    Ok(())
}

pub fn ssh(name: Option<&str>) -> Result<()> {
    let (n, info) = resolve_instance(name)?;

    if is_setup_done(&n) {
        provision_env_vars(&info)?;
    } else {
        println!("First SSH, running setup...");
        let branch = current_branch()?;
        match info.backend {
            Backend::RunPod => runpod::setup(&info, &branch)?,
            Backend::VastAi => vastai::setup(&info)?,
        }
        mark_setup_done(&n);
    }

    let mut cmd = ssh_cmd(&info);

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
    let _ = fs::remove_file(setup_done_path(&n));
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
        let _ = fs::remove_file(setup_done_path(name));
    }

    println!("All {} instances destroyed", names.len());
    Ok(())
}

pub fn start(name: Option<&str>) -> Result<()> {
    let (n, info) = resolve_instance(name)?;

    match info.backend {
        Backend::RunPod => {
            let updated = runpod::resume(&info)?;
            save_instance(&n, &updated)?;
            println!(
                "Instance '{n}' started (ssh {}@{}:{})",
                updated.ssh_user, updated.ssh_host, updated.ssh_port
            );
        }
        Backend::VastAi => bail!("Start not supported for VastAI yet"),
    }

    Ok(())
}

pub fn import(name: &str, backend: Backend) -> Result<()> {
    if instances_dir().join(name).exists() {
        bail!("Instance '{name}' already exists");
    }

    let info = match backend {
        Backend::RunPod => runpod::import()?,
        Backend::VastAi => bail!("Import not supported for VastAI yet"),
    };

    save_instance(name, &info)?;
    println!(
        "Instance '{name}' imported ({backend}, id={})",
        info.instance_id
    );
    Ok(())
}

pub fn sync() -> Result<()> {
    let names = list_instances()?;
    if names.is_empty() {
        println!("No instances");
        return Ok(());
    }

    let mut has_runpod = false;
    let mut has_vastai = false;
    let mut instances: Vec<(String, InstanceInfo)> = Vec::new();

    for name in &names {
        let info = read_instance(name)?;
        match info.backend {
            Backend::RunPod => has_runpod = true,
            Backend::VastAi => has_vastai = true,
        }
        instances.push((name.clone(), info));
    }

    let runpod_ids = if has_runpod {
        match runpod::get_pod_ids() {
            Ok(ids) => Some(ids),
            Err(e) => {
                eprintln!("Warning: failed to query RunPod, skipping RunPod instances: {e}");
                None
            }
        }
    } else {
        None
    };

    let vastai_ids = if has_vastai {
        match vastai::get_instance_ids() {
            Ok(ids) => Some(ids),
            Err(e) => {
                eprintln!("Warning: failed to query vast.ai, skipping vast.ai instances: {e}");
                None
            }
        }
    } else {
        None
    };

    let mut removed = 0;
    for (name, info) in &instances {
        let active_ids = match info.backend {
            Backend::RunPod => &runpod_ids,
            Backend::VastAi => &vastai_ids,
        };

        let Some(ids) = active_ids else {
            continue;
        };

        if !ids.contains(&info.instance_id) {
            fs::remove_file(instances_dir().join(name))?;
            let _ = fs::remove_file(setup_done_path(name));
            println!(
                "Removed '{name}' ({}, id={})",
                info.backend, info.instance_id
            );
            removed += 1;
        }
    }

    if removed == 0 {
        println!("All instances are still active");
    } else {
        println!("Removed {removed} stale instance(s)");
    }

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
