use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_lock::{Dependency, Lockfile, Package};

mod build_support;

use build_support::{digest_inputs, is_post_embedding_source};

const BASE_EMBEDDING_DEPENDENCIES: &[&str] = &[
    "ndarray",
    "ndarray-npy",
    "ort",
    "serde",
    "serde_json",
    "sha2",
];
const COREML_EMBEDDING_DEPENDENCIES: &[&str] =
    &["block2", "objc2", "objc2-core-ml", "objc2-foundation"];
const BUILD_IDENTITY_ENV: &[&str] = &[
    "CARGO_CFG_TARGET_ARCH",
    "CARGO_CFG_TARGET_ENDIAN",
    "CARGO_CFG_TARGET_ENV",
    "CARGO_CFG_TARGET_FEATURE",
    "CARGO_CFG_TARGET_FAMILY",
    "CARGO_CFG_TARGET_OS",
    "CARGO_CFG_TARGET_POINTER_WIDTH",
    "CARGO_CFG_TARGET_VENDOR",
    "CARGO_ENCODED_RUSTFLAGS",
    "DEBUG",
    "OPT_LEVEL",
    "PROFILE",
    "TARGET",
];
const EMBEDDING_FEATURE_ENV: &[&str] = &[
    "CARGO_FEATURE_COREML",
    "CARGO_FEATURE_CUDA",
    "CARGO_FEATURE_LOAD_DYNAMIC",
];

fn main() {
    let sha = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo::rerun-if-changed=../.git/HEAD");
    println!("cargo::rerun-if-changed=../.git/refs");
    println!("cargo:rustc-env=GIT_SHA={sha}");

    let implementation = embedding_implementation_digest();
    println!("cargo:rustc-env=WAVLM_EMBEDDING_IMPLEMENTATION_SHA256={implementation}");
}

fn embedding_implementation_digest() -> String {
    let mut files = Vec::new();
    collect_rust_sources(Path::new("../src"), &mut files);
    files.retain(|path| !is_post_embedding_source(path));
    files.extend([
        PathBuf::from("build.rs"),
        PathBuf::from("build_support.rs"),
        PathBuf::from("src/commands/wavlm_bridge/embedding_execution.rs"),
    ]);
    files.sort();
    files.dedup();

    let mut inputs = Vec::new();
    for path in files {
        println!("cargo::rerun-if-changed={}", path.display());
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        inputs.push((format!("source:{}", path.display()), bytes));
    }

    inputs.extend(embedding_dependency_inputs());
    inputs.extend(embedding_manifest_inputs());
    inputs.extend(embedding_build_inputs());

    digest_inputs(&inputs)
}

fn embedding_dependency_inputs() -> Vec<(String, Vec<u8>)> {
    let lock_path = Path::new("../Cargo.lock");
    println!("cargo::rerun-if-changed={}", lock_path.display());
    let lockfile = Lockfile::load(lock_path)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", lock_path.display()));
    let root_names = embedding_dependency_roots();
    let mut pending = workspace_dependencies(&lockfile, &root_names);
    let mut admitted = BTreeSet::new();
    let mut packages = Vec::new();

    while let Some(dependency) = pending.pop() {
        if !admitted.insert(dependency.clone()) {
            continue;
        }

        let package = resolve_package(&lockfile, &dependency);
        pending.extend(package.dependencies.iter().cloned());
        packages.push(package);
    }

    packages.sort();
    packages
        .into_iter()
        .map(|package| {
            let label = format!("dependency:{}:{}", package.name, package.version);
            let mut dependencies = package
                .dependencies
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            dependencies.sort();
            let value = format!(
                "source={:?}\nchecksum={:?}\ndependencies={}",
                package.source,
                package.checksum,
                dependencies.join(",")
            )
            .into_bytes();
            (label, value)
        })
        .collect()
}

fn embedding_dependency_roots() -> BTreeSet<&'static str> {
    let mut roots: BTreeSet<_> = BASE_EMBEDDING_DEPENDENCIES.iter().copied().collect();
    if env::var_os("CARGO_FEATURE_COREML").is_some() {
        roots.extend(COREML_EMBEDDING_DEPENDENCIES.iter().copied());
    }
    if env::var_os("CARGO_FEATURE_LOAD_DYNAMIC").is_some() {
        roots.insert("libloading");
    }

    roots
}

fn workspace_dependencies(lockfile: &Lockfile, names: &BTreeSet<&str>) -> Vec<Dependency> {
    lockfile
        .packages
        .iter()
        .filter(|package| {
            package.source.is_none() && matches!(package.name.as_str(), "speakrs" | "xtask")
        })
        .flat_map(|package| package.dependencies.iter())
        .filter(|dependency| names.contains(dependency.name.as_str()))
        .cloned()
        .collect()
}

fn resolve_package<'a>(lockfile: &'a Lockfile, dependency: &Dependency) -> &'a Package {
    let matches = lockfile
        .packages
        .iter()
        .filter(|package| dependency.matches(package))
        .filter(|package| {
            dependency.source.is_none() || dependency.source.as_ref() == package.source.as_ref()
        })
        .collect::<Vec<_>>();

    match matches.as_slice() {
        [package] => package,
        [] => panic!("embedding dependency {dependency} is missing from Cargo.lock"),
        _ => panic!("embedding dependency {dependency} is ambiguous in Cargo.lock"),
    }
}

fn embedding_manifest_inputs() -> Vec<(String, Vec<u8>)> {
    let dependency_names = embedding_dependency_roots();
    [Path::new("../Cargo.toml"), Path::new("Cargo.toml")]
        .into_iter()
        .flat_map(|path| {
            println!("cargo::rerun-if-changed={}", path.display());
            let source = fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let manifest = toml::from_str::<toml::Value>(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            let mut inputs = Vec::new();
            collect_manifest_dependencies(
                &manifest,
                path.to_string_lossy().as_ref(),
                &dependency_names,
                &mut inputs,
            );
            collect_manifest_features(&manifest, path.to_string_lossy().as_ref(), &mut inputs);
            inputs
        })
        .collect()
}

fn collect_manifest_dependencies(
    value: &toml::Value,
    path: &str,
    dependency_names: &BTreeSet<&str>,
    inputs: &mut Vec<(String, Vec<u8>)>,
) {
    let Some(table) = value.as_table() else {
        return;
    };

    for (key, value) in table {
        let child_path = format!("{path}.{key}");
        if key == "dependencies" {
            let dependencies = value
                .as_table()
                .unwrap_or_else(|| panic!("{child_path} must be a table"));
            for (name, specification) in dependencies {
                if dependency_names.contains(name.as_str()) {
                    inputs.push((
                        format!("manifest:{child_path}.{name}"),
                        specification.to_string().into_bytes(),
                    ));
                }
            }
        } else {
            collect_manifest_dependencies(value, &child_path, dependency_names, inputs);
        }
    }
}

fn collect_manifest_features(
    manifest: &toml::Value,
    path: &str,
    inputs: &mut Vec<(String, Vec<u8>)>,
) {
    let Some(features) = manifest.get("features").and_then(toml::Value::as_table) else {
        return;
    };

    for feature in ["coreml", "cuda", "load-dynamic"] {
        if let Some(specification) = features.get(feature) {
            inputs.push((
                format!("manifest:{path}.features.{feature}"),
                specification.to_string().into_bytes(),
            ));
        }
    }
}

fn embedding_build_inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs = Vec::new();
    for name in BUILD_IDENTITY_ENV.iter().chain(EMBEDDING_FEATURE_ENV) {
        println!("cargo::rerun-if-env-changed={name}");
        let value = env::var(name).unwrap_or_default();
        inputs.push((format!("build:{name}"), value.into_bytes()));
    }

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(&rustc)
        .arg("--version")
        .arg("--verbose")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| output.stdout)
        .unwrap_or_else(|| b"unknown".to_vec());
    inputs.push(("build:rustc-version".to_owned(), version));
    inputs.sort_by(|left, right| left.0.cmp(&right.0));

    inputs
}

fn collect_rust_sources(root: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", root.display()))
    {
        let path = entry
            .expect("source directory entry must be readable")
            .path();
        if path.is_dir() {
            collect_rust_sources(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}
