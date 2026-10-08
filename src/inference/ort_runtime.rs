//! ONNX Runtime process setup, execution providers, and shared sessions

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::ffi::CStr;
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::path::Path;
use std::path::PathBuf;
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::sync::OnceLock;
use std::sync::{Arc, Mutex, MutexGuard};

use ort::ep;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

use super::{ExecutionMode, InferenceBackend, InferenceError, ModelLoadError};

/// Shared ownership of one ONNX Runtime session
#[derive(Clone)]
pub(crate) struct SharedSession(Arc<Mutex<Session>>);

impl SharedSession {
    pub(crate) fn new(session: Session) -> Self {
        Self(Arc::new(Mutex::new(session)))
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, Session>, InferenceError> {
        self.0.lock().map_err(|_| InferenceError::LockPoisoned {
            resource: "shared ONNX Runtime session",
        })
    }
}

/// ONNX Runtime execution provider for one session
///
/// Providers exist only for features compiled into this build, so a provider value is always
/// usable. Filterbank sessions use [`OrtProvider::Cpu`] in every ORT mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrtProvider {
    Cpu,
    #[cfg(feature = "migraphx")]
    MiGraphX,
}

impl OrtProvider {
    pub(crate) fn apply(self, builder: SessionBuilder) -> Result<SessionBuilder, ort::Error> {
        match self {
            Self::Cpu => Ok(builder.with_execution_providers([ep::CPU::default()
                .with_arena_allocator(false)
                .build()])?),
            #[cfg(feature = "migraphx")]
            Self::MiGraphX => Ok(builder.with_execution_providers([ep::MIGraphX::default()
                .with_device_id(0)
                .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                .build()
                .error_on_failure()])?),
        }
    }
}

/// Map an execution mode to ORT execution providers
///
/// speakrs never builds ONNX Runtime sessions for CoreML or CUDA modes. When the `coreml` or
/// `cuda` feature is enabled, this helper maps those modes to the ORT CPU provider for callers
/// that build their own sessions
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "cpu", feature = "migraphx", feature = "load-dynamic")))
)]
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ModelLoadError> {
    Ok(session_provider(mode.backend()?).apply(builder)?)
}

/// Provider for a caller-built session, with native-backend modes mapped to the CPU provider
fn session_provider(backend: InferenceBackend) -> OrtProvider {
    match backend {
        InferenceBackend::Ort(provider) => provider,
        #[cfg(feature = "coreml")]
        InferenceBackend::CoreMl => OrtProvider::Cpu,
        #[cfg(feature = "cuda")]
        InferenceBackend::Cuda => OrtProvider::Cpu,
    }
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
static ORT_RUNTIME_INIT: OnceLock<Result<(), OrtRuntimeError>> = OnceLock::new();

pub(crate) fn ensure_ort_ready() -> Result<(), ModelLoadError> {
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    {
        let init_result = ORT_RUNTIME_INIT.get_or_init(|| OrtRuntimeLoader::new().initialize());
        init_result.clone()?;
    }

    Ok(())
}

/// Errors that can occur while preparing the process-wide ONNX Runtime environment
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "cpu", feature = "migraphx", feature = "load-dynamic")))
)]
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum OrtRuntimeError {
    /// Dynamic runtime discovery or validation failed before `ort` could initialize
    #[error(transparent)]
    Dynamic(#[from] DynamicRuntimeError),
    /// `ort::init_from` failed after runtime validation succeeded
    #[error("failed to initialize ONNX Runtime: {message}")]
    Initialization {
        /// The initialization error returned by `ort`
        message: String,
    },
}

/// Errors from locating or validating the dynamic ONNX Runtime library
#[cfg_attr(
    docsrs,
    doc(cfg(any(feature = "cpu", feature = "migraphx", feature = "load-dynamic")))
)]
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum DynamicRuntimeError {
    /// No candidate runtime library was found
    #[error(
        "missing ONNX Runtime dynamic library `{library_name}`; set `ORT_DYLIB_PATH` or place it next to the test/binary\nsearched: {searched}"
    )]
    Missing {
        /// The platform-specific dynamic library filename
        library_name: &'static str,
        /// The candidate paths checked before giving up
        searched: String,
    },
    /// Loading the requested runtime library failed
    #[error("failed to load ONNX Runtime dynamic library at `{path}`: {message}")]
    Load {
        /// The path that failed to load
        path: PathBuf,
        /// The dynamic loader error
        message: String,
    },
    /// The requested runtime library does not export `OrtGetApiBase`
    #[error("ONNX Runtime dynamic library at `{path}` does not export `OrtGetApiBase`")]
    MissingApiBase {
        /// The path that was missing the required symbol
        path: PathBuf,
    },
    /// The requested runtime library returned a null API base pointer
    #[error("ONNX Runtime dynamic library at `{path}` returned a null `OrtApiBase`")]
    NullApiBase {
        /// The path that returned a null API pointer
        path: PathBuf,
    },
    /// The requested runtime library is older than the `ort` crate expects
    #[error(
        "ONNX Runtime dynamic library at `{path}` is too old; expected >= 1.{required_minor}.x, got `{found_version}`"
    )]
    IncompatibleVersion {
        /// The incompatible runtime library path
        path: PathBuf,
        /// The minimum ONNX Runtime minor version required by `ort`
        required_minor: u32,
        /// The version reported by the discovered runtime library
        found_version: String,
    },
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
struct OrtRuntimeLoader {
    library_name: &'static str,
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
impl OrtRuntimeLoader {
    fn new() -> Self {
        Self {
            library_name: Self::default_library_name(),
        }
    }

    fn initialize(&self) -> Result<(), OrtRuntimeError> {
        let path = self.resolve_library_path()?;
        self.validate_library(&path)?;

        ort::init_from(&path)
            .map(|builder| {
                builder.commit();
            })
            .map_err(|error| OrtRuntimeError::Initialization {
                message: error.to_string(),
            })
    }

    fn resolve_library_path(&self) -> Result<PathBuf, DynamicRuntimeError> {
        if let Ok(path) = std::env::var("ORT_DYLIB_PATH")
            && !path.is_empty()
        {
            let path = PathBuf::from(path);
            return path.exists().then_some(path.clone()).ok_or_else(|| {
                DynamicRuntimeError::Missing {
                    library_name: self.library_name,
                    searched: path.display().to_string(),
                }
            });
        }

        let candidates = self.candidate_paths();
        candidates
            .iter()
            .find(|path| path.exists())
            .cloned()
            .ok_or_else(|| DynamicRuntimeError::Missing {
                library_name: self.library_name,
                searched: Self::format_paths(&candidates),
            })
    }

    fn candidate_paths(&self) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        if let Ok(exe) = std::env::current_exe()
            && let Some(exe_dir) = exe.parent()
        {
            candidates.push(exe_dir.join(self.library_name));
            if let Some(parent) = exe_dir.parent() {
                candidates.push(parent.join(self.library_name));
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            candidates.push(cwd.join(self.library_name));
            candidates.push(cwd.join("target/debug").join(self.library_name));
            candidates.push(cwd.join("target/debug/deps").join(self.library_name));
            candidates.push(cwd.join("target/release").join(self.library_name));
            candidates.push(cwd.join("target/release/deps").join(self.library_name));
        }

        dedup_paths(candidates)
    }

    fn validate_library(&self, path: &Path) -> Result<(), DynamicRuntimeError> {
        // safety: we only open the candidate runtime long enough to validate its exported API
        let library = unsafe { libloading::Library::new(path) }.map_err(|error| {
            DynamicRuntimeError::Load {
                path: path.to_path_buf(),
                message: error.to_string(),
            }
        })?;

        // safety: the library handle stays alive while the retrieved symbol is used below
        let get_api_base: libloading::Symbol<
            unsafe extern "C" fn() -> *const ort::sys::OrtApiBase,
        > = unsafe { library.get(b"OrtGetApiBase") }.map_err(|_| {
            DynamicRuntimeError::MissingApiBase {
                path: path.to_path_buf(),
            }
        })?;

        // safety: `OrtGetApiBase` has the stable ONNX Runtime entrypoint signature
        let api_base = unsafe { get_api_base() };
        if api_base.is_null() {
            return Err(DynamicRuntimeError::NullApiBase {
                path: path.to_path_buf(),
            });
        }

        // safety: the validated runtime exposes a process-stable version string pointer
        let version_ptr = unsafe { ((*api_base).GetVersionString)() };
        // safety: ONNX Runtime documents the version string as a null-terminated C string
        let version = unsafe { CStr::from_ptr(version_ptr) }
            .to_string_lossy()
            .into_owned();
        let minor = version
            .split('.')
            .nth(1)
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(0);
        if minor < ort::MINOR_VERSION {
            return Err(DynamicRuntimeError::IncompatibleVersion {
                path: path.to_path_buf(),
                required_minor: ort::MINOR_VERSION,
                found_version: version,
            });
        }

        Ok(())
    }

    const fn default_library_name() -> &'static str {
        #[cfg(target_os = "windows")]
        {
            "onnxruntime.dll"
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            "libonnxruntime.so"
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            "libonnxruntime.dylib"
        }
    }

    fn format_paths(paths: &[PathBuf]) -> String {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
fn dedup_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique = Vec::with_capacity(paths.len());
    for path in paths {
        if !unique.contains(&path) {
            unique.push(path);
        }
    }
    unique
}

#[cfg(all(test, feature = "load-dynamic", not(target_arch = "wasm32")))]
mod tests {
    use super::{DynamicRuntimeError, OrtRuntimeError, ensure_ort_ready};
    use crate::inference::ModelLoadError;

    #[test]
    fn dynamic_runtime_preflight_fails_instead_of_hanging() {
        let original = std::env::var_os("ORT_DYLIB_PATH");
        let missing = std::env::temp_dir().join("missing-ort-runtime/libonnxruntime.dylib");
        // safety: this test mutates a process-global env var and restores it before returning
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", &missing);
        }

        let error = ensure_ort_ready().unwrap_err();
        assert!(matches!(
            error,
            ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
                DynamicRuntimeError::Missing { .. }
            ))
        ));

        // safety: this test restores the original process-global env var before returning
        unsafe {
            match original {
                Some(value) => std::env::set_var("ORT_DYLIB_PATH", value),
                None => std::env::remove_var("ORT_DYLIB_PATH"),
            }
        }
    }
}
