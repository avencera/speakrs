pub(crate) mod embedding;
pub(crate) mod segmentation;

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::ffi::CStr;
use std::fmt;
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::path::Path;
use std::path::PathBuf;
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
use std::sync::OnceLock;

pub use embedding::EmbeddingModel;
pub use segmentation::{SegmentationError, SegmentationModel};

#[cfg(feature = "coreml")]
pub(crate) mod coreml;

use ort::ep;
use ort::session::builder::SessionBuilder;

#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
static ORT_RUNTIME_INIT: OnceLock<Result<(), OrtRuntimeError>> = OnceLock::new();

/// CoreML compute unit selection for chunk embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CoreMlComputeUnits {
    /// Use all available compute units: CPU + GPU + Neural Engine (default)
    #[default]
    All,
    /// Use CPU + Neural Engine only (skip GPU)
    CpuAndNeuralEngine,
}

#[cfg(feature = "coreml")]
impl CoreMlComputeUnits {
    pub(crate) fn to_ml_compute_units(self) -> objc2_core_ml::MLComputeUnits {
        match self {
            Self::All => crate::inference::coreml::CoreMlModel::default_compute_units(),
            Self::CpuAndNeuralEngine => objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
        }
    }
}

/// Which backend and acceleration to use for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// CPU-only via ORT (portable, slowest)
    Cpu,
    /// Native CoreML with FP32 precision and ~1s step
    CoreMl,
    /// Native CoreML with W8A16 segmentation and ~2s step
    CoreMlFast,
    /// NVIDIA GPU with concurrent fused seg+emb via crossbeam
    Cuda,
    /// NVIDIA GPU with concurrent fused seg+emb and ~2s step
    CudaFast,
}

impl ExecutionMode {
    pub(crate) fn validate(self) -> Result<(), ExecutionModeError> {
        match self {
            Self::Cpu => Ok(()),
            Self::CoreMl | Self::CoreMlFast => {
                #[cfg(feature = "coreml")]
                {
                    Ok(())
                }

                #[cfg(not(feature = "coreml"))]
                {
                    Err(ExecutionModeError {
                        mode: self,
                        feature: "coreml",
                    })
                }
            }
            Self::Cuda | Self::CudaFast => {
                #[cfg(feature = "cuda")]
                {
                    Ok(())
                }

                #[cfg(not(feature = "cuda"))]
                {
                    Err(ExecutionModeError {
                        mode: self,
                        feature: "cuda",
                    })
                }
            }
        }
    }

    /// Lowercase identifier used in logs, docs, and user-facing errors
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::CoreMlFast => "coreml-fast",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
        }
    }
}

impl fmt::Display for ExecutionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors that can occur while loading a model or initializing ONNX Runtime
#[derive(Debug, thiserror::Error)]
pub enum ModelLoadError {
    /// Requested execution mode is not supported by this build
    #[error(transparent)]
    UnsupportedExecutionMode(#[from] ExecutionModeError),
    /// ONNX Runtime could not be prepared for this process
    #[error(transparent)]
    Runtime(#[from] OrtRuntimeError),
    /// ONNX Runtime returned an error after initialization completed
    #[error(transparent)]
    Ort(#[from] ort::Error),
}

/// Errors that can occur while preparing the process-wide ONNX Runtime environment
#[derive(Debug, Clone, thiserror::Error)]
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
#[derive(Debug, Clone, thiserror::Error)]
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

/// Errors from requesting an execution mode that is not supported in the current build
#[derive(Debug, Clone, thiserror::Error)]
#[error("{mode} requires the `{feature}` Cargo feature")]
pub struct ExecutionModeError {
    mode: ExecutionMode,
    feature: &'static str,
}

impl From<ExecutionModeError> for ort::Error {
    fn from(error: ExecutionModeError) -> Self {
        ort::Error::new(error.to_string())
    }
}

/// Map an execution mode to ORT execution providers
///
/// CoreML modes use ORT CPU for any sessions that still go through ORT such as FBANK,
/// While segmentation and embedding tail sessions use native CoreML directly
pub fn with_execution_mode(
    builder: SessionBuilder,
    mode: ExecutionMode,
) -> Result<SessionBuilder, ort::Error> {
    mode.validate()?;

    match mode {
        ExecutionMode::Cpu | ExecutionMode::CoreMl | ExecutionMode::CoreMlFast => Ok(builder
            .with_execution_providers([ep::CPU::default().with_arena_allocator(false).build()])?),
        ExecutionMode::Cuda | ExecutionMode::CudaFast => {
            #[cfg(feature = "cuda")]
            {
                Ok(builder.with_execution_providers([ep::CUDA::default()
                    .with_device_id(0)
                    .with_tf32(true)
                    .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Exhaustive)
                    .with_conv_max_workspace(true)
                    .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
                    .with_prefer_nhwc(true)
                    .build()
                    .error_on_failure()])?)
            }

            #[cfg(not(feature = "cuda"))]
            {
                unreachable!("mode validation rejects CUDA modes without the `cuda` feature")
            }
        }
    }
}

pub(crate) fn ensure_ort_ready() -> Result<(), ModelLoadError> {
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    {
        let init_result = ORT_RUNTIME_INIT.get_or_init(|| OrtRuntimeLoader::new().initialize());
        init_result.clone()?;
    }

    Ok(())
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

#[cfg(test)]
mod tests {
    #[cfg(any(not(feature = "coreml"), not(feature = "cuda")))]
    use super::ExecutionMode;
    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
    use super::{DynamicRuntimeError, OrtRuntimeError, ensure_ort_ready};

    #[cfg(not(feature = "coreml"))]
    #[test]
    fn coreml_modes_require_feature() {
        let error = ExecutionMode::CoreMl.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml requires the `coreml` Cargo feature"
        );

        let error = ExecutionMode::CoreMlFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "coreml-fast requires the `coreml` Cargo feature"
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_modes_require_feature() {
        let error = ExecutionMode::Cuda.validate().unwrap_err();
        assert_eq!(error.to_string(), "cuda requires the `cuda` Cargo feature");

        let error = ExecutionMode::CudaFast.validate().unwrap_err();
        assert_eq!(
            error.to_string(),
            "cuda-fast requires the `cuda` Cargo feature"
        );
    }

    #[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
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
            super::ModelLoadError::Runtime(OrtRuntimeError::Dynamic(
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
