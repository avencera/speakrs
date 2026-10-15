//! Execution modes and runtime options as plain strings
//!
//! Bindings cannot see which backend features the SDK was built with, so this
//! module owns every `cfg` that depends on them

use speakrs::ExecutionMode;
use speakrs::pipeline::RuntimeConfig;

use crate::error::SdkError;

/// Stable display name of a mode, as bindings expose it
pub fn mode_name(mode: ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Cpu => "CPU",
        ExecutionMode::CoreMl => "CoreML",
        ExecutionMode::CoreMlFast => "CoreMLFast",
        ExecutionMode::Cuda => "CUDA",
        ExecutionMode::CudaFast => "CUDAFast",
        ExecutionMode::MiGraphX => "MIGraphX",
        _ => "unknown",
    }
}

/// Parse a mode name; case, `-`, and `_` are ignored, so `cuda-fast` names `CUDAFast`
///
/// MIGraphX is not offered by the bindings
pub fn parse_mode(name: &str) -> Result<ExecutionMode, SdkError> {
    let normalized: String = name
        .chars()
        .filter(|c| !matches!(c, '-' | '_'))
        .flat_map(char::to_lowercase)
        .collect();
    match normalized.as_str() {
        "cpu" => Ok(ExecutionMode::Cpu),
        "coreml" => Ok(ExecutionMode::CoreMl),
        "coremlfast" => Ok(ExecutionMode::CoreMlFast),
        "cuda" => Ok(ExecutionMode::Cuda),
        "cudafast" => Ok(ExecutionMode::CudaFast),
        _ => Err(SdkError::configuration(format!(
            "unknown execution mode `{name}`; expected one of CPU, CoreML, CoreMLFast, CUDA, CUDAFast"
        ))),
    }
}

/// Modes compiled into this build
///
/// A CUDA mode can still fail at model load when no NVIDIA driver or supported GPU
/// is present
pub fn available_modes() -> Vec<ExecutionMode> {
    let mut modes = Vec::new();
    if cfg!(feature = "cpu") {
        modes.push(ExecutionMode::Cpu);
    }
    if cfg!(feature = "coreml") {
        modes.extend([ExecutionMode::CoreMl, ExecutionMode::CoreMlFast]);
    }
    if cfg!(feature = "cuda") {
        modes.extend([ExecutionMode::Cuda, ExecutionMode::CudaFast]);
    }
    modes
}

/// Runtime settings that only some backends read
///
/// `None` keeps the backend default. Setting an option whose backend is not compiled
/// in is a configuration error
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeOptions {
    /// CoreML compute units: `all`, `cpu_and_neural_engine`, or `cpu_only`
    pub coreml_compute_units: Option<String>,
    /// CUDA segmentation precision: `fp32` or `tf32`
    pub cuda_segmentation_math: Option<String>,
    /// CUDA embedding precision: `fp32` or `tf32`
    pub cuda_embedding_math: Option<String>,
    /// Whether CUDA modes capture and replay CUDA graphs
    pub cuda_graphs: Option<bool>,
}

impl RuntimeOptions {
    /// Build the core runtime config
    pub fn to_runtime_config(&self) -> Result<RuntimeConfig, SdkError> {
        #[cfg_attr(not(any(feature = "coreml", feature = "cuda")), allow(unused_mut))]
        let mut config = RuntimeConfig::default();

        if let Some(units) = &self.coreml_compute_units {
            #[cfg(feature = "coreml")]
            {
                config.chunk_emb_compute_units = parse_compute_units(units)?;
            }
            #[cfg(not(feature = "coreml"))]
            return Err(missing_backend("coreml_compute_units", "CoreML", units));
        }

        let cuda_options = [
            (
                "cuda_segmentation_math",
                self.cuda_segmentation_math.as_deref(),
            ),
            ("cuda_embedding_math", self.cuda_embedding_math.as_deref()),
        ];
        for (name, value) in cuda_options {
            let Some(value) = value else { continue };
            #[cfg(feature = "cuda")]
            {
                let math = parse_cuda_math(name, value)?;
                if name == "cuda_segmentation_math" {
                    config.cuda_segmentation_math = math;
                } else {
                    config.cuda_embedding_math = math;
                }
            }
            #[cfg(not(feature = "cuda"))]
            return Err(missing_backend(name, "CUDA", value));
        }

        if let Some(enabled) = self.cuda_graphs {
            #[cfg(feature = "cuda")]
            {
                config.cuda_graphs = if enabled {
                    speakrs::inference::CudaGraphs::Enabled
                } else {
                    speakrs::inference::CudaGraphs::Disabled
                };
            }
            #[cfg(not(feature = "cuda"))]
            return Err(missing_backend("cuda_graphs", "CUDA", &enabled.to_string()));
        }

        Ok(config)
    }

    /// The effective value of every option this build supports
    pub fn resolved(&self) -> Result<Self, SdkError> {
        let config = self.to_runtime_config()?;
        #[cfg(not(any(feature = "coreml", feature = "cuda")))]
        let _ = config;

        Ok(Self {
            #[cfg(feature = "coreml")]
            coreml_compute_units: Some(compute_units_name(config.chunk_emb_compute_units).into()),
            #[cfg(not(feature = "coreml"))]
            coreml_compute_units: None,
            #[cfg(feature = "cuda")]
            cuda_segmentation_math: Some(cuda_math_name(config.cuda_segmentation_math).into()),
            #[cfg(feature = "cuda")]
            cuda_embedding_math: Some(cuda_math_name(config.cuda_embedding_math).into()),
            #[cfg(feature = "cuda")]
            cuda_graphs: Some(config.cuda_graphs == speakrs::inference::CudaGraphs::Enabled),
            #[cfg(not(feature = "cuda"))]
            cuda_segmentation_math: None,
            #[cfg(not(feature = "cuda"))]
            cuda_embedding_math: None,
            #[cfg(not(feature = "cuda"))]
            cuda_graphs: None,
        })
    }
}

#[cfg(not(all(feature = "coreml", feature = "cuda")))]
fn missing_backend(option: &str, backend: &str, value: &str) -> SdkError {
    SdkError::new(
        crate::error::SdkErrorCategory::UnsupportedRuntime,
        format!("`{option}={value}` needs {backend} support, which this build does not include"),
    )
}

#[cfg(feature = "coreml")]
fn parse_compute_units(value: &str) -> Result<speakrs::CoreMlComputeUnits, SdkError> {
    use speakrs::CoreMlComputeUnits;

    match value {
        "all" => Ok(CoreMlComputeUnits::All),
        "cpu_and_neural_engine" => Ok(CoreMlComputeUnits::CpuAndNeuralEngine),
        "cpu_only" => Ok(CoreMlComputeUnits::CpuOnly),
        _ => Err(SdkError::configuration(format!(
            "unknown coreml_compute_units `{value}`; expected all, cpu_and_neural_engine, or cpu_only"
        ))),
    }
}

#[cfg(feature = "coreml")]
fn compute_units_name(units: speakrs::CoreMlComputeUnits) -> &'static str {
    use speakrs::CoreMlComputeUnits;

    match units {
        CoreMlComputeUnits::All => "all",
        CoreMlComputeUnits::CpuAndNeuralEngine => "cpu_and_neural_engine",
        CoreMlComputeUnits::CpuOnly => "cpu_only",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

#[cfg(feature = "cuda")]
fn parse_cuda_math(option: &str, value: &str) -> Result<speakrs::inference::CudaMath, SdkError> {
    use speakrs::inference::CudaMath;

    match value {
        "fp32" => Ok(CudaMath::Fp32),
        "tf32" => Ok(CudaMath::Tf32),
        _ => Err(SdkError::configuration(format!(
            "unknown {option} `{value}`; expected fp32 or tf32"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn cuda_math_name(math: speakrs::inference::CudaMath) -> &'static str {
    use speakrs::inference::CudaMath;

    match math {
        CudaMath::Fp32 => "fp32",
        CudaMath::Tf32 => "tf32",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SdkErrorCategory;

    #[test]
    fn mode_names_round_trip_and_accept_cli_spellings() {
        for mode in [
            ExecutionMode::Cpu,
            ExecutionMode::CoreMl,
            ExecutionMode::CoreMlFast,
            ExecutionMode::Cuda,
            ExecutionMode::CudaFast,
        ] {
            assert_eq!(parse_mode(mode_name(mode)).unwrap(), mode);
        }
        assert_eq!(parse_mode("cuda-fast").unwrap(), ExecutionMode::CudaFast);
        assert_eq!(
            parse_mode("coreml_fast").unwrap(),
            ExecutionMode::CoreMlFast
        );
        assert_eq!(
            parse_mode("migraphx").unwrap_err().category,
            SdkErrorCategory::Configuration
        );
    }

    #[test]
    fn runtime_options_reject_settings_for_missing_backends() {
        let coreml = RuntimeOptions {
            coreml_compute_units: Some("cpu_only".into()),
            ..RuntimeOptions::default()
        };
        let cuda = RuntimeOptions {
            cuda_embedding_math: Some("fp32".into()),
            ..RuntimeOptions::default()
        };

        assert_eq!(coreml.to_runtime_config().is_ok(), cfg!(feature = "coreml"));
        assert_eq!(cuda.to_runtime_config().is_ok(), cfg!(feature = "cuda"));
    }
}
