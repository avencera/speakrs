use std::fs;
use std::io::{BufReader, Read};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
#[cfg(target_os = "macos")]
use std::process::Command;

use color_eyre::eyre::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct HostIdentity {
    pub git_sha: String,
    pub dirty_diff_sha256: String,
    pub cpu: String,
    pub macos: String,
    pub xcode: String,
    pub metal_compiler: String,
    pub rustc: String,
    pub cargo: String,
    pub xctrace: String,
    pub power_source: String,
    pub thermal_state: String,
}

#[cfg(target_os = "macos")]
impl HostIdentity {
    pub(super) fn collect(root: &Path) -> Result<Self> {
        Ok(Self {
            git_sha: command_text(root, "git", &["rev-parse", "HEAD"])?,
            dirty_diff_sha256: dirty_diff_digest(root)?,
            cpu: command_text(root, "sysctl", &["-n", "machdep.cpu.brand_string"])?,
            macos: command_text(root, "sw_vers", &["-productVersion"])?,
            xcode: command_text(root, "xcodebuild", &["-version"])?.replace('\n', "; "),
            metal_compiler: command_text(root, "xcrun", &["metal", "--version"])?
                .lines()
                .next()
                .unwrap_or("unknown")
                .to_owned(),
            rustc: command_text(root, "rustc", &["--version", "--verbose"])?,
            cargo: command_text(root, "cargo", &["--version", "--verbose"])?,
            xctrace: optional_command_text(root, "xcrun", &["xctrace", "version"]),
            power_source: optional_command_text(root, "pmset", &["-g", "batt"]).replace('\n', "; "),
            thermal_state: optional_command_text(root, "pmset", &["-g", "therm"])
                .replace('\n', "; "),
        })
    }
}

pub(super) fn digest_paths(root: &Path, paths: &[PathBuf]) -> Result<String> {
    let files = files_for_paths(paths)?;
    digest_files(root, &files)
}

pub(super) fn digest_paths_cached(
    root: &Path,
    paths: &[PathBuf],
    cache_dir: &Path,
) -> Result<String> {
    let files = files_for_paths(paths)?;
    let mut metadata_digest = Sha256::new();
    for path in &files {
        let metadata = fs::metadata(path)?;
        let relative = path.strip_prefix(root).unwrap_or(path);
        update_length_prefixed(&mut metadata_digest, relative.to_string_lossy().as_bytes());
        metadata_digest.update(metadata.len().to_le_bytes());
        metadata_digest.update(metadata.dev().to_le_bytes());
        metadata_digest.update(metadata.ino().to_le_bytes());
        metadata_digest.update(metadata.ctime().to_le_bytes());
        metadata_digest.update(metadata.ctime_nsec().to_le_bytes());
        let modified = metadata
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        metadata_digest.update(modified.as_secs().to_le_bytes());
        metadata_digest.update(modified.subsec_nanos().to_le_bytes());
    }
    let key = format!("{:x}", metadata_digest.finalize());
    let cache_path = cache_dir.join(format!("{key}.sha256"));
    if let Ok(cached) = fs::read_to_string(&cache_path) {
        let cached = cached.trim();
        if cached.len() == 64 && cached.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Ok(cached.to_owned());
        }
    }

    let digest = digest_files(root, &files)?;
    fs::create_dir_all(cache_dir)?;
    let temporary_path = cache_path.with_extension("sha256.tmp");
    fs::write(&temporary_path, format!("{digest}\n"))?;
    fs::rename(temporary_path, cache_path)?;
    Ok(digest)
}

fn files_for_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for path in paths {
        collect_files(path, &mut files)?;
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn digest_files(root: &Path, files: &[PathBuf]) -> Result<String> {
    let mut digest = Sha256::new();
    for path in files {
        let relative = path.strip_prefix(root).unwrap_or(path.as_path());
        update_length_prefixed(&mut digest, relative.to_string_lossy().as_bytes());
        let file =
            fs::File::open(path).wrap_err_with(|| format!("failed to hash {}", path.display()))?;
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 128 * 1024];
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            digest.update(&buffer[..read]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "macos")]
fn dirty_diff_digest(root: &Path) -> Result<String> {
    let diff = Command::new("git")
        .args(["diff", "--binary", "HEAD"])
        .current_dir(root)
        .output()?;
    ensure!(
        diff.status.success(),
        "git diff failed with {}",
        diff.status
    );
    let status = Command::new("git")
        .args(["status", "--porcelain=v1", "-z"])
        .current_dir(root)
        .output()?;
    ensure!(
        status.status.success(),
        "git status failed with {}",
        status.status
    );

    let mut digest = Sha256::new();
    update_length_prefixed(&mut digest, &diff.stdout);
    update_length_prefixed(&mut digest, &status.stdout);
    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z"])
        .current_dir(root)
        .output()?;
    ensure!(
        untracked.status.success(),
        "git ls-files failed with {}",
        untracked.status
    );
    for relative in untracked
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        update_length_prefixed(&mut digest, relative);
        let path = root.join(String::from_utf8_lossy(relative).as_ref());
        if path.is_file() {
            let contents = fs::read(&path)
                .wrap_err_with(|| format!("failed to hash untracked file {}", path.display()))?;
            update_length_prefixed(&mut digest, &contents);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(target_os = "macos")]
fn command_text(root: &Path, program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()?;
    ensure!(
        output.status.success(),
        "{program} {} failed with {}",
        args.join(" "),
        output.status
    );
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(target_os = "macos")]
fn optional_command_text(root: &Path, program: &str, args: &[&str]) -> String {
    command_text(root, program, args).unwrap_or_else(|_| "unknown".to_owned())
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    ensure!(
        path.is_dir(),
        "digest path does not exist: {}",
        path.display()
    );
    let mut entries: Vec<_> = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        collect_files(&entry.path(), files)?;
    }
    Ok(())
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn digest_paths_is_stable_across_input_order() {
        let temp = tempfile::tempdir().unwrap();
        let a = temp.path().join("a");
        let b = temp.path().join("b");
        fs::File::create(&a).unwrap().write_all(b"one").unwrap();
        fs::File::create(&b).unwrap().write_all(b"two").unwrap();

        let left = digest_paths(temp.path(), &[a.clone(), b.clone()]).unwrap();
        let right = digest_paths(temp.path(), &[b, a]).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn cached_digest_changes_after_file_change() {
        let temp = tempfile::tempdir().unwrap();
        let cache = temp.path().join("cache");
        let model = temp.path().join("model");
        fs::File::create(&model)
            .unwrap()
            .write_all(b"first")
            .unwrap();
        let first = digest_paths_cached(temp.path(), std::slice::from_ref(&model), &cache).unwrap();

        fs::File::create(&model)
            .unwrap()
            .write_all(b"second")
            .unwrap();
        let second = digest_paths_cached(temp.path(), &[model], &cache).unwrap();

        assert_ne!(first, second);
        assert_eq!(fs::read_dir(cache).unwrap().count(), 2);
    }
}
