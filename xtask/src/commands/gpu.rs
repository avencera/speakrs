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

const IMAGE_BASE: &str = "ghcr.io/avencera/speakrs-gpu";
const LOCAL_DIR: &str = "_local";
const IMAGE_TAG_FILE: &str = "gpu-image-tag";
const INSTANCES_DIR: &str = "gpu-instances";

pub fn image() -> String {
    let tag_file = project_root().join(LOCAL_DIR).join(IMAGE_TAG_FILE);
    let tag = fs::read_to_string(tag_file)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "latest".to_string());
    format!("{IMAGE_BASE}:{tag}")
}
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
    project_root().join(LOCAL_DIR).join(INSTANCES_DIR)
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
    if let Some(instance_name) = name {
        let info = read_instance(instance_name)?;
        return Ok((instance_name.to_string(), info));
    }

    let names = list_instances()?;
    match names.len() {
        0 => bail!("No instances found. Run `cargo xtask gpu create <name>` first"),
        1 => {
            let instance_name = &names[0];
            let info = read_instance(instance_name)?;
            println!("Auto-selected instance '{instance_name}'");
            Ok((instance_name.clone(), info))
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

            let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let info = read_instance(&selected)?;
            Ok((selected, info))
        }
    }
}

impl InstanceInfo {
    pub fn ssh_args(&self) -> Vec<String> {
        vec![
            "-p".to_string(),
            self.ssh_port.clone(),
            format!("{}@{}", self.ssh_user, self.ssh_host),
        ]
    }

    fn is_proxy(&self) -> bool {
        self.ssh_host == "ssh.runpod.io"
    }

    /// SSH command, adds `-tt` (PTY) only for RunPod proxy connections
    pub fn ssh_cmd(&self) -> Command {
        let mut cmd = Command::new("ssh");
        if self.is_proxy() {
            cmd.arg("-tt");
        }
        cmd.args([
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]);
        cmd.args(self.ssh_args());
        cmd
    }

    pub fn run_remote_script(&self, script: &str) -> Result<()> {
        if self.is_proxy() {
            let mut cmd = self.ssh_cmd();
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
            let mut cmd = self.ssh_cmd();
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

    /// Copy local terminal info to the remote so tmux works with non-standard terminals (e.g. Ghostty)
    pub fn provision_terminfo(&self) -> Result<()> {
        let term = std::env::var("TERM").unwrap_or_default();
        if term.is_empty() || term == "xterm-256color" || term == "xterm" {
            return Ok(());
        }

        let output = Command::new("infocmp")
            .args(["-x", &term])
            .output()
            .ok()
            .filter(|o| o.status.success());

        let Some(output) = output else {
            return Ok(());
        };

        let terminfo = String::from_utf8_lossy(&output.stdout);
        if terminfo.is_empty() {
            return Ok(());
        }

        let script = format!("cat << 'TERMINFO_EOF' | tic -x -\n{terminfo}\nTERMINFO_EOF");
        self.run_remote_script(&script)?;
        println!("Installed {term} terminfo on remote");

        Ok(())
    }

    /// Write local env vars to `/root/.env` on the remote and ensure `.bashrc` sources it
    pub fn provision_env_vars(&self) -> Result<()> {
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
        self.run_remote_script(&script)?;

        println!("Wrote /root/.env with {count} var(s)");
        Ok(())
    }
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
    let (instance_name, info) = resolve_instance(name)?;
    // TODO: re-enable wait_for_container_ready after image rebuild with new entrypoint

    match info.backend {
        Backend::RunPod => runpod::setup(&info, branch)?,
        Backend::VastAi => vastai::setup(&info)?,
    }

    mark_setup_done(&instance_name);
    println!("Instance '{instance_name}' setup complete");
    Ok(())
}

pub fn ssh(name: Option<&str>) -> Result<()> {
    let (instance_name, info) = resolve_instance(name)?;
    // TODO: re-enable wait_for_container_ready after image rebuild with new entrypoint

    if is_setup_done(&instance_name) {
        info.provision_env_vars()?;
        info.provision_terminfo()?;
    } else {
        println!("First SSH, running setup...");
        let branch = current_branch()?;
        match info.backend {
            Backend::RunPod => runpod::setup(&info, &branch)?,
            Backend::VastAi => vastai::setup(&info)?,
        }
        mark_setup_done(&instance_name);
    }

    let mut cmd = info.ssh_cmd();

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

    let (instance_name, info) = resolve_instance(name)?;

    match info.backend {
        Backend::RunPod => runpod::destroy(&info)?,
        Backend::VastAi => vastai::destroy(&info)?,
    }

    fs::remove_file(instances_dir().join(&instance_name))?;
    let _ = fs::remove_file(setup_done_path(&instance_name));
    println!("Instance '{instance_name}' destroyed and removed");
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
    let (instance_name, info) = resolve_instance(name)?;

    match info.backend {
        Backend::RunPod => {
            let updated = runpod::resume(&info)?;
            save_instance(&instance_name, &updated)?;
            println!(
                "Instance '{instance_name}' started (ssh {}@{}:{})",
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

pub fn download_benchmarks(name: Option<&str>) -> Result<()> {
    let (instance_name, info) = resolve_instance(name)?;

    let local_dir = project_root()
        .join("_benchmarks")
        .join(format!("{instance_name}-{}", info.instance_id));
    fs::create_dir_all(&local_dir)?;

    let (host, port) = runpod::direct_ssh_info(&info.instance_id)?;

    println!(
        "Downloading benchmarks from '{instance_name}' ({host}:{port}) to {}",
        local_dir.display()
    );

    let remote = format!("root@{host}:/workspace/_benchmarks/.");
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let key_path = format!("{home}/.ssh/id_ed25519");

    let status = Command::new("scp")
        .args([
            "-r",
            "-i",
            &key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            &port,
            &remote,
        ])
        .arg(&local_dir)
        .status()?;

    if !status.success() {
        bail!("scp failed");
    }

    println!("Done");
    Ok(())
}
