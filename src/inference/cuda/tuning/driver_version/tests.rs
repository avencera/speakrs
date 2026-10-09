//! Host-only checks of driver-source selection and identity

use super::{CudaTuneError, DriverIdentity, DriverRelease};

fn missing() -> std::io::Result<String> {
    Err(std::io::ErrorKind::NotFound.into())
}

#[test]
fn nvml_release_wins_without_reading_files_or_cuda() {
    let version = DriverIdentity::select(
        || Some("580.95.05".into()),
        |_| panic!("NVML must avoid file reads"),
        || panic!("NVML must avoid the CUDA fallback"),
    )
    .unwrap();
    assert_eq!(
        version.clone().require_release().unwrap(),
        DriverRelease::Nvml("580.95.05".into())
    );
    assert_eq!(
        serde_json::to_value(version.require_release().unwrap()).unwrap(),
        serde_json::json!({"source": "nvml", "version": "580.95.05"})
    );
}

#[test]
fn missing_nvml_uses_sysfs_before_procfs() {
    let mut paths = Vec::new();
    let version = DriverIdentity::select(
        || None,
        |path| {
            paths.push(path.to_owned());
            Ok(" 580.95.05 \n".into())
        },
        || panic!("a module release must avoid the CUDA fallback"),
    )
    .unwrap();
    assert_eq!(
        version.require_release().unwrap(),
        DriverRelease::Sysfs("580.95.05".into())
    );
    assert_eq!(paths, ["/sys/module/nvidia/version"]);
}

#[test]
fn unavailable_sysfs_uses_the_procfs_first_line() {
    let first_line = "NVRM version: NVIDIA UNIX x86_64 Kernel Module 580.95.05";
    for sysfs in [None, Some(" \n")] {
        let mut paths = Vec::new();
        let version = DriverIdentity::select(
            || None,
            |path| {
                paths.push(path.to_owned());
                match path {
                    "/sys/module/nvidia/version" => {
                        sysfs.map(str::to_owned).map_or_else(missing, Ok)
                    }
                    "/proc/driver/nvidia/version" => {
                        Ok(format!("{first_line}\nGCC version: ignored\n"))
                    }
                    _ => panic!("unexpected driver path {path}"),
                }
            },
            || panic!("procfs must avoid the CUDA fallback"),
        )
        .unwrap();
        assert_eq!(
            version.require_release().unwrap(),
            DriverRelease::Procfs(first_line.into())
        );
        assert_eq!(
            paths,
            ["/sys/module/nvidia/version", "/proc/driver/nvidia/version"]
        );
    }
}

#[test]
fn missing_or_empty_sources_use_the_cuda_api_level() {
    for nvml in [None, Some(" \n")] {
        for file in [None, Some("\n")] {
            let version = DriverIdentity::select(
                || nvml.map(str::to_owned),
                |_| file.map(str::to_owned).map_or_else(missing, Ok),
                || Ok(13000),
            )
            .unwrap();
            assert_eq!(version, DriverIdentity::ApiOnly(13000));
            assert!(matches!(
                version.require_release(),
                Err(CudaTuneError::DriverReleaseUnreadable { cuda_api: 13000 })
            ));
        }
    }
}

#[test]
fn equal_text_from_different_sources_never_matches() {
    let versions = [
        DriverRelease::Nvml("13000".into()),
        DriverRelease::Sysfs("13000".into()),
        DriverRelease::Procfs("13000".into()),
    ];
    for (index, version) in versions.iter().enumerate() {
        let json = serde_json::to_vec(version).unwrap();
        assert_eq!(
            serde_json::from_slice::<DriverRelease>(&json).unwrap(),
            *version
        );
        for other in &versions[index + 1..] {
            assert_ne!(version, other);
        }
    }
}

#[test]
fn an_unavailable_or_invalid_cuda_api_version_returns_the_error() {
    let result = DriverIdentity::select(
        || None,
        |_| missing(),
        || Err(CudaTuneError::Invalid("driver query failed".into())),
    );
    assert!(
        matches!(result, Err(CudaTuneError::Invalid(reason)) if reason == "driver query failed")
    );
    for version in [0, -1] {
        let result = DriverIdentity::select(|| None, |_| missing(), || Ok(version));
        assert!(
            matches!(result, Err(CudaTuneError::Invalid(reason)) if reason == "CUDA driver API version is not positive")
        );
    }
}

#[test]
fn nvml_symbols_return_the_release_and_balance_initialization() {
    use std::process::Command;

    use libloading::Library;

    let directory = std::env::temp_dir().join(format!("speakrs-nvml-stub-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    for (index, init_status, query_status, terminate, expected) in [
        (0, 0, 0, true, Some("580.95.05")),
        (1, 1, 0, true, None),
        (2, 0, 1, true, None),
        (3, 0, 0, false, None),
    ] {
        let source = directory.join(format!("stub-{index}.c"));
        let path = directory.join(format!("stub-{index}.{}", std::env::consts::DLL_EXTENSION));
        let copy = if terminate {
            "memcpy(version, \"580.95.05\", 10);"
        } else {
            "memset(version, 'x', length);"
        };
        std::fs::write(
            &source,
            format!(
                r#"
#include <string.h>
static unsigned int shutdown_count;
unsigned int nvmlInit_v2(void) {{ return {init_status}; }}
unsigned int nvmlSystemGetDriverVersion(char *version, unsigned int length) {{
    if (length != 80) return 2;
    {copy}
    return {query_status};
}}
unsigned int nvmlShutdown(void) {{ ++shutdown_count; return 0; }}
unsigned int stubShutdownCount(void) {{ return shutdown_count; }}
"#
            ),
        )
        .unwrap();
        let output = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // safety: the host stub has no library initializer or caller preconditions
        let library = unsafe { Library::new(&path) }.unwrap();
        assert_eq!(
            super::nvml_version_from_library(&library).as_deref(),
            expected
        );
        // safety: the test stub exports this exact signature, and the library is live
        let shutdown_count = unsafe {
            library
                .get::<unsafe extern "C" fn() -> u32>(b"stubShutdownCount\0")
                .unwrap()()
        };
        assert_eq!(shutdown_count, u32::from(init_status == 0));
    }
    std::fs::remove_dir_all(directory).unwrap();
}
