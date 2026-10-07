//! Entry point for the locked CUDA qualification harness

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;

use color_eyre::eyre::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::cmd::project_root;

const LOCK_PATH: &str = "scripts/cuda/qualify/LOCK";
/// The locked inventory definition, shared with `scripts/cuda/qualify/lock.py`
const SCOPE_PATH: &str = "scripts/cuda/qualify/SCOPE.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema: u32,
    files: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Scope {
    schema: u32,
    directories: Vec<ScopeDirectory>,
    #[serde(default)]
    external_assets: BTreeMap<String, String>,
    files: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopeDirectory {
    path: String,
    exclude: Vec<String>,
}

/// Checks the snapshot before starting Python or using the GPU box
///
/// Set `SPEAKRS_QUALIFY_OWNER_DIGEST` to the SHA256 digest kept outside the tree
/// by the owner. A missing digest, changed harness, failed check or blocked check
/// is a command failure. A child process exit is not qualification evidence
pub fn run(target: &str, implementation: &str, collection: Option<&str>) -> Result<()> {
    if (target == "segdense") != collection.is_some() {
        bail!("segdense requires --collection; other targets must omit --collection");
    }

    let root = project_root();
    let digest = verify_lock(&root)?;
    let expected = std::env::var("SPEAKRS_QUALIFY_OWNER_DIGEST")
        .wrap_err("set SPEAKRS_QUALIFY_OWNER_DIGEST to the owner's outside-tree digest")?;
    if digest != expected {
        bail!("harness LOCK differs from the owner's outside-tree digest");
    }

    // a new cache directory prevents stale untracked Python bytecode from running
    let cache = tempfile::tempdir().wrap_err("create a fresh qualification Python cache")?;
    let mut command = Command::new("python3");
    command
        .arg(root.join("scripts/cuda/qualify/qualify.py"))
        .arg(target)
        .arg(implementation)
        .env("PYTHONPYCACHEPREFIX", cache.path())
        .current_dir(&root);
    if let Some(collection) = collection {
        command.arg("--collection").arg(collection);
    }

    let status = command
        .status()
        .wrap_err("start CUDA qualification harness")?;
    if !status.success() {
        bail!("CUDA qualification did not pass ({status}); see the result JSON and summary");
    }

    Ok(())
}

fn verify_lock(root: &Path) -> Result<String> {
    let lock = root.join(LOCK_PATH);
    if fs::symlink_metadata(&lock)?.file_type().is_symlink() {
        bail!("harness LOCK must not be a symlink");
    }

    let bytes = fs::read(&lock).wrap_err("read CUDA harness LOCK")?;
    let snapshot: Snapshot = serde_json::from_slice(&bytes).wrap_err("parse CUDA harness LOCK")?;
    if snapshot.schema != 1 {
        bail!("unsupported CUDA harness LOCK schema");
    }

    let scope_path = root.join(SCOPE_PATH);
    if fs::symlink_metadata(&scope_path)?.file_type().is_symlink() {
        bail!("harness SCOPE.json must not be a symlink");
    }

    let scope: Scope = serde_json::from_slice(&fs::read(&scope_path)?)
        .wrap_err("parse CUDA harness SCOPE.json")?;
    if scope.schema != 1 {
        bail!("unsupported CUDA harness scope schema");
    }

    let mut files = BTreeMap::new();
    for directory in &scope.directories {
        let base = root.join(&directory.path);
        if !fs::symlink_metadata(&base)?.file_type().is_dir() {
            bail!("missing regular harness directory: {}", base.display());
        }

        inventory(root, &base, &base, &directory.exclude, &mut files)?;
    }

    for relative in &scope.files {
        let path = root.join(relative);
        let mut ancestor = path.as_path();
        while ancestor != root {
            if fs::symlink_metadata(ancestor)?.file_type().is_symlink() {
                bail!("symlink in harness path: {}", ancestor.display());
            }

            ancestor = ancestor
                .parent()
                .ok_or_else(|| color_eyre::eyre::eyre!("invalid harness path"))?;
        }

        insert_file(root, &path, &mut files)?;
    }

    for (name, digest) in scope.external_assets {
        if root.join(&name).exists() || files.contains_key(&name) {
            bail!("external asset must stay outside the tree: {name}");
        }

        if digest.len() != 64
            || !digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            bail!("invalid external asset digest: {name}");
        }

        files.insert(name, digest);
    }

    if files != snapshot.files {
        bail!("harness files differ from LOCK; qualification refused before Python starts");
    }

    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn inventory(
    root: &Path,
    base: &Path,
    path: &Path,
    exclude: &[String],
    files: &mut BTreeMap<String, String>,
) -> Result<()> {
    if path != base && excluded(base, path, exclude) {
        return Ok(());
    }

    let kind = fs::symlink_metadata(path)?.file_type();
    if kind.is_symlink() {
        bail!("symlink in CUDA harness: {}", path.display());
    }

    if kind.is_dir() {
        for entry in fs::read_dir(path)? {
            inventory(root, base, &entry?.path(), exclude, files)?;
        }

        return Ok(());
    }

    if path == root.join(LOCK_PATH) {
        return Ok(());
    }

    // cache bytes are never loaded by the qualification entry point
    if path.extension().is_some_and(|extension| extension == "pyc")
        && path
            .parent()
            .and_then(Path::file_name)
            .is_some_and(|name| name == "__pycache__")
    {
        return Ok(());
    }

    insert_file(root, path, files)
}

/// A path is excluded when it is a listed entry, relative to its directory, or lies
/// under one; a rule ending in `*` excludes every path that starts with the rest
fn excluded(base: &Path, path: &Path, exclude: &[String]) -> bool {
    let Ok(relative) = path.strip_prefix(base) else {
        return false;
    };

    let relative = relative.to_string_lossy().replace('\\', "/");
    exclude.iter().any(|rule| match rule.strip_suffix('*') {
        Some(prefix) => relative.starts_with(prefix),
        None => {
            relative == *rule
                || relative
                    .strip_prefix(rule.as_str())
                    .is_some_and(|rest| rest.starts_with('/'))
        }
    })
}

fn insert_file(root: &Path, path: &Path, files: &mut BTreeMap<String, String>) -> Result<()> {
    if !path.is_file() {
        bail!("harness path is not a regular file: {}", path.display());
    }

    let relative = path
        .strip_prefix(root)?
        .to_str()
        .ok_or_else(|| color_eyre::eyre::eyre!("non-UTF8 harness path"))?
        .replace('\\', "/");
    files.insert(relative, format!("{:x}", Sha256::digest(fs::read(path)?)));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use sha2::{Digest, Sha256};

    use super::{LOCK_PATH, SCOPE_PATH, verify_lock};

    const SCOPE: &str = r#"{"schema":1,"directories":[{"path":"scripts/cuda/qualify","exclude":[]},{"path":"src/inference/cuda","exclude":["candidate","ptx/lstm.*"]}],"files":["Cargo.toml"]}"#;

    #[test]
    fn rejects_added_changed_and_deleted_files_before_python() {
        let temporary = tempfile::tempdir().expect("temporary harness");
        let root = temporary.path();
        let mut files = BTreeMap::new();
        for relative in [SCOPE_PATH, "Cargo.toml", "src/inference/cuda/dispatch.rs"] {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().expect("parent")).expect("parent directories");
            let contents = if relative == SCOPE_PATH {
                SCOPE
            } else {
                "initial\n"
            };
            fs::write(path, contents).expect("harness file");
            files.insert(
                relative,
                format!("{:x}", Sha256::digest(contents.as_bytes())),
            );
        }

        // candidate code and candidate PTX are outside the lock
        let candidate = root.join("src/inference/cuda/candidate/conv.rs");
        fs::create_dir_all(candidate.parent().expect("parent")).expect("candidate directory");
        fs::write(&candidate, "candidate").expect("candidate file");
        let candidate_ptx = root.join("src/inference/cuda/ptx/lstm.sm75.ptx");
        fs::create_dir_all(candidate_ptx.parent().expect("parent")).expect("ptx directory");
        fs::write(&candidate_ptx, "ptx").expect("candidate ptx");

        let snapshot = serde_json::json!({"schema": 1, "files": files});
        fs::write(
            root.join(LOCK_PATH),
            serde_json::to_vec(&snapshot).expect("JSON"),
        )
        .expect("LOCK");
        assert!(verify_lock(root).is_ok());
        fs::write(&candidate, "changed candidate").expect("candidate edit");
        fs::write(&candidate_ptx, "changed ptx").expect("candidate ptx edit");
        assert!(verify_lock(root).is_ok());
        let library_ptx = root.join("src/inference/cuda/ptx/segmentation.sm75.ptx");
        fs::write(&library_ptx, "library ptx").expect("library ptx");
        assert!(verify_lock(root).is_err());
        fs::remove_file(&library_ptx).expect("remove library ptx");
        let added = root.join("scripts/cuda/qualify/extra.py");
        fs::write(&added, "extra").expect("added file");
        assert!(verify_lock(root).is_err());
        fs::remove_file(&added).expect("remove added file");
        let changed = root.join("src/inference/cuda/dispatch.rs");
        fs::write(&changed, "changed").expect("changed file");
        assert!(verify_lock(root).is_err());
        fs::remove_file(changed).expect("remove changed file");
        assert!(verify_lock(root).is_err());
    }
}
