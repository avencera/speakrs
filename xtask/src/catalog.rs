//! Typed implementation catalog for benchmark selection.

use color_eyre::eyre::{Result, bail};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Canonical implementation identity used inside the tool domain
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ImplementationId {
    PyannoteMps,
    PyannoteCpu,
    PyannoteCuda,
    SpeakrsCoreMl,
    SpeakrsCoreMlFast,
    SpeakrsCuda,
    SpeakrsCudaFast,
    SpeakrsCpu,
    FluidAudio,
    SpeakerKit,
    PyannoteRs,
}

impl ImplementationId {
    /// Canonical serialized ID used in schema version 2 results
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PyannoteMps => "pyannote",
            Self::PyannoteCpu => "pyannote-cpu",
            Self::PyannoteCuda => "pyannote-cuda",
            Self::SpeakrsCoreMl => "coreml",
            Self::SpeakrsCoreMlFast => "coreml-fast",
            Self::SpeakrsCuda => "cuda",
            Self::SpeakrsCudaFast => "cuda-fast",
            Self::SpeakrsCpu => "cpu",
            Self::FluidAudio => "fluidaudio",
            Self::SpeakerKit => "speakerkit",
            Self::PyannoteRs => "pyannote-rs",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        CATALOG
            .iter()
            .find(|spec| spec.id.as_str() == value)
            .map(|spec| spec.id)
    }
}

impl Serialize for ImplementationId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ImplementationId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(&value)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown implementation id {value}")))
    }
}

/// Speakrs execution mode stored as a typed value, not a CLI string
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpeakrsMode {
    Cpu,
    CoreMl,
    CoreMlFast,
    Cuda,
    CudaFast,
}

impl SpeakrsMode {
    /// CLI token for `diarize --mode`
    pub const fn as_cli(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::CoreMl => "coreml",
            Self::CoreMlFast => "coreml-fast",
            Self::Cuda => "cuda",
            Self::CudaFast => "cuda-fast",
        }
    }

    pub const fn is_coreml(self) -> bool {
        matches!(self, Self::CoreMl | Self::CoreMlFast)
    }

    pub const fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda | Self::CudaFast)
    }
}

/// Pyannote device stored as a typed value
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PyannoteDevice {
    Cpu,
    Mps,
    Cuda,
}

impl PyannoteDevice {
    /// CLI token for pyannote `--device`
    pub const fn as_cli(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Mps => "mps",
            Self::Cuda => "cuda",
        }
    }
}

/// How a catalog entry is executed
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunnerKind {
    Speakrs(SpeakrsMode),
    Pyannote(PyannoteDevice),
    PyannoteRs,
    FluidAudio,
    SpeakerKit,
}

/// Cargo feature required to build a speakrs runner
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoFeature {
    CoreMl,
    Cuda,
}

impl CargoFeature {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CoreMl => "coreml",
            Self::Cuda => "cuda",
        }
    }
}

/// Platforms that can run a catalog entry
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformSet {
    Macos,
    Linux,
    All,
}

impl PlatformSet {
    pub const fn contains_macos(self) -> bool {
        matches!(self, Self::Macos | Self::All)
    }

    pub const fn contains_linux(self) -> bool {
        matches!(self, Self::Linux | Self::All)
    }
}

/// Extra capabilities recorded for a catalog entry
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CapabilitySet {
    pub gpu: bool,
}

/// One catalog entry
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImplementationSpec {
    pub id: ImplementationId,
    pub aliases: &'static [&'static str],
    pub display_name: &'static str,
    pub runner: RunnerKind,
    pub platforms: PlatformSet,
    pub cargo_features: &'static [CargoFeature],
    pub capabilities: CapabilitySet,
}

impl ImplementationSpec {
    pub const fn cli_name(self) -> &'static str {
        self.id.as_str()
    }
}

const CATALOG: &[ImplementationSpec] = &[
    ImplementationSpec {
        id: ImplementationId::PyannoteMps,
        aliases: &["pmps"],
        display_name: "pyannote MPS",
        runner: RunnerKind::Pyannote(PyannoteDevice::Mps),
        platforms: PlatformSet::Macos,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::PyannoteCpu,
        aliases: &["pcpu"],
        display_name: "pyannote CPU",
        runner: RunnerKind::Pyannote(PyannoteDevice::Cpu),
        platforms: PlatformSet::All,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: false },
    },
    ImplementationSpec {
        id: ImplementationId::PyannoteCuda,
        aliases: &["pg"],
        display_name: "pyannote CUDA",
        runner: RunnerKind::Pyannote(PyannoteDevice::Cuda),
        platforms: PlatformSet::Linux,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakrsCoreMl,
        aliases: &["scm"],
        display_name: "speakrs CoreML",
        runner: RunnerKind::Speakrs(SpeakrsMode::CoreMl),
        platforms: PlatformSet::Macos,
        cargo_features: &[CargoFeature::CoreMl],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakrsCoreMlFast,
        aliases: &["scmf"],
        display_name: "speakrs CoreML Fast",
        runner: RunnerKind::Speakrs(SpeakrsMode::CoreMlFast),
        platforms: PlatformSet::Macos,
        cargo_features: &[CargoFeature::CoreMl],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakrsCuda,
        aliases: &["sg"],
        display_name: "speakrs CUDA",
        runner: RunnerKind::Speakrs(SpeakrsMode::Cuda),
        platforms: PlatformSet::Linux,
        cargo_features: &[CargoFeature::Cuda],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakrsCudaFast,
        aliases: &["sgf"],
        display_name: "speakrs CUDA Fast",
        runner: RunnerKind::Speakrs(SpeakrsMode::CudaFast),
        platforms: PlatformSet::Linux,
        cargo_features: &[CargoFeature::Cuda],
        capabilities: CapabilitySet { gpu: true },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakrsCpu,
        aliases: &["scpu"],
        display_name: "speakrs CPU",
        runner: RunnerKind::Speakrs(SpeakrsMode::Cpu),
        platforms: PlatformSet::All,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: false },
    },
    ImplementationSpec {
        id: ImplementationId::FluidAudio,
        aliases: &["fa"],
        display_name: "FluidAudio",
        runner: RunnerKind::FluidAudio,
        platforms: PlatformSet::Macos,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: false },
    },
    ImplementationSpec {
        id: ImplementationId::SpeakerKit,
        aliases: &["sk"],
        display_name: "SpeakerKit",
        runner: RunnerKind::SpeakerKit,
        platforms: PlatformSet::Macos,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: false },
    },
    ImplementationSpec {
        id: ImplementationId::PyannoteRs,
        aliases: &["prs"],
        display_name: "pyannote-rs",
        runner: RunnerKind::PyannoteRs,
        platforms: PlatformSet::All,
        cargo_features: &[],
        capabilities: CapabilitySet { gpu: false },
    },
];

/// Static catalog of benchmark implementations
pub struct ImplementationCatalog;

impl ImplementationCatalog {
    pub fn all() -> &'static [ImplementationSpec] {
        CATALOG
    }

    pub fn gpu() -> impl Iterator<Item = &'static ImplementationSpec> {
        CATALOG
            .iter()
            .filter(|spec| spec.capabilities.gpu)
            .filter(|spec| {
                matches!(
                    spec.runner,
                    RunnerKind::Speakrs(SpeakrsMode::Cuda | SpeakrsMode::CudaFast)
                        | RunnerKind::Pyannote(PyannoteDevice::Cuda)
                )
            })
    }

    pub fn parse_cli(name: &str) -> Option<&'static ImplementationSpec> {
        CATALOG
            .iter()
            .find(|spec| spec.cli_name() == name || spec.aliases.contains(&name))
    }

    pub fn resolve_many(names: &[String]) -> Result<Vec<&'static ImplementationSpec>> {
        if names.is_empty() {
            return Ok(CATALOG.iter().collect());
        }
        let mut selected = Vec::new();
        for name in names {
            if name == "list" {
                continue;
            }
            let Some(spec) = Self::parse_cli(name) else {
                let available: Vec<&str> = CATALOG.iter().map(|spec| spec.cli_name()).collect();
                bail!(
                    "unknown implementation: {name}. Available: {}",
                    available.join(", ")
                );
            };
            selected.push(spec);
        }
        Ok(selected)
    }

    pub fn cargo_features(selected: &[&ImplementationSpec]) -> Vec<String> {
        let mut features = Vec::new();
        #[cfg(target_os = "macos")]
        if selected
            .iter()
            .any(|spec| matches!(spec.runner, RunnerKind::Speakrs(mode) if mode.is_coreml()))
        {
            features.push(CargoFeature::CoreMl.as_str().to_owned());
        }
        if selected
            .iter()
            .any(|spec| matches!(spec.runner, RunnerKind::Speakrs(mode) if mode.is_cuda()))
        {
            features.push(CargoFeature::Cuda.as_str().to_owned());
        }
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_and_aliases_are_unique() {
        let mut ids = HashSet::new();
        let mut names = HashSet::new();
        for spec in ImplementationCatalog::all() {
            assert!(ids.insert(spec.id), "duplicate id {:?}", spec.id);
            assert!(
                names.insert(spec.cli_name()),
                "duplicate cli name {}",
                spec.cli_name()
            );
            for alias in spec.aliases {
                assert!(names.insert(alias), "duplicate alias {alias}");
            }
        }
    }

    #[test]
    fn catalog_records_runner_platform_and_features() {
        let coreml = ImplementationCatalog::parse_cli("scm").unwrap();
        assert_eq!(coreml.id, ImplementationId::SpeakrsCoreMl);
        assert_eq!(coreml.runner, RunnerKind::Speakrs(SpeakrsMode::CoreMl));
        assert!(coreml.platforms.contains_macos());
        assert_eq!(coreml.cargo_features, &[CargoFeature::CoreMl]);

        let cuda = ImplementationCatalog::parse_cli("sg").unwrap();
        assert_eq!(cuda.runner, RunnerKind::Speakrs(SpeakrsMode::Cuda));
        assert_eq!(cuda.cargo_features, &[CargoFeature::Cuda]);
        assert!(cuda.capabilities.gpu);

        let cpu = ImplementationCatalog::parse_cli("cpu").unwrap();
        assert_eq!(cpu.runner, RunnerKind::Speakrs(SpeakrsMode::Cpu));
        assert!(cpu.cargo_features.is_empty());
    }

    #[test]
    fn gpu_catalog_is_a_query_not_a_second_registry() {
        let gpu: Vec<_> = ImplementationCatalog::gpu().map(|spec| spec.id).collect();
        assert_eq!(
            gpu,
            vec![
                ImplementationId::PyannoteCuda,
                ImplementationId::SpeakrsCuda,
                ImplementationId::SpeakrsCudaFast,
            ]
        );
    }

    #[test]
    fn unknown_cli_name_is_rejected() {
        assert!(ImplementationCatalog::resolve_many(&["nope".to_owned()]).is_err());
    }
}
