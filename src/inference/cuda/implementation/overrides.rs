//! Accuracy-backed override requests, kept separate from speed-qualified production selection

use super::Target;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
use crate::inference::cuda::{CudaMath, KernelModule};

/// A validated execution boundary, independent of an implementation choice
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Boundary {
    area: KernelModule,
    name: String,
    batch: usize,
    math: CudaMath,
}

impl Boundary {
    /// Reject unsupported areas and malformed boundary tuples before evidence lookup
    pub(crate) fn new(
        area: KernelModule,
        name: &str,
        batch: usize,
        math: CudaMath,
    ) -> Result<Self, OverrideError> {
        if !matches!(
            area,
            KernelModule::Fbank
                | KernelModule::Embedding
                | KernelModule::Segmentation
                | KernelModule::Resnet
                | KernelModule::Lstm
                | KernelModule::Sincnet
        ) {
            return Err(OverrideError::InvalidArea);
        }
        if batch == 0
            || name.split('.').next() != Some(area.name())
            || name.split('.').any(str::is_empty)
            || !name.contains('.')
        {
            return Err(OverrideError::InvalidTuple);
        }
        Ok(Self {
            area,
            name: name.to_owned(),
            batch,
            math,
        })
    }
}

/// A pinned accuracy record and a pinned deterministic execution record
#[derive(Debug, Clone, Copy)]
pub(crate) struct Evidence {
    accuracy: ArtifactHash,
    deterministic: ArtifactHash,
}

impl Evidence {
    /// Require both proofs; a set-level record cannot replace either configuration proof
    ///
    /// The caller must supply trusted, reviewed record hashes for this configuration
    /// This constructor checks proof presence, not record contents or qualification results
    pub(crate) fn new(
        accuracy: Option<ArtifactHash>,
        deterministic: Option<ArtifactHash>,
    ) -> Result<Self, OverrideError> {
        Ok(Self {
            accuracy: accuracy.ok_or(OverrideError::MissingAccuracy)?,
            deterministic: deterministic.ok_or(OverrideError::MissingDeterminism)?,
        })
    }
}

/// One enumerated configuration with evidence bound to its exact execution tuple
#[derive(Debug)]
pub(crate) struct Configuration {
    id: String,
    boundary: Boundary,
    target: Target,
    evidence: Evidence,
}

impl Configuration {
    /// Bind this configuration's proofs to one target, including the loaded artifact
    pub(crate) fn new(
        id: &str,
        boundary: Boundary,
        target: Target,
        evidence: Evidence,
    ) -> Result<Self, OverrideError> {
        validate_target(target)?;
        if id.is_empty() || id.trim() != id {
            return Err(OverrideError::InvalidConfiguration);
        }
        Ok(Self {
            id: id.to_owned(),
            boundary,
            target,
            evidence,
        })
    }
}

// a cubin for another device cannot be the artifact loaded at this target
fn validate_target(target: Target) -> Result<(), OverrideError> {
    if target.device < target.tier.min_capability()
        || matches!(target.artifact, LoadedArtifact::Cubin { arch, .. } if arch != target.device)
    {
        return Err(OverrideError::InvalidTarget);
    }
    Ok(())
}

/// An explicit choice; Library policy is enforced by the runtime owner
#[derive(Debug)]
pub(crate) enum Choice {
    Library,
    Configuration(String),
}

/// One override request for one execution tuple
#[derive(Debug)]
pub(crate) struct Request {
    boundary: Boundary,
    target: Target,
    choice: Choice,
}

impl Request {
    /// Keep the boundary and target together so resolution cannot omit either
    pub(crate) fn new(boundary: Boundary, target: Target, choice: Choice) -> Self {
        Self {
            boundary,
            target,
            choice,
        }
    }
}

/// An enumerated set with unique configuration IDs and per-configuration evidence
#[derive(Debug)]
pub(crate) struct ConfigurationSet(Vec<Configuration>);

impl ConfigurationSet {
    /// Reject duplicate IDs even if their execution tuples differ
    pub(crate) fn new(configurations: Vec<Configuration>) -> Result<Self, OverrideError> {
        for (index, configuration) in configurations.iter().enumerate() {
            if configurations[..index]
                .iter()
                .any(|previous| previous.id == configuration.id)
            {
                return Err(OverrideError::DuplicateId(configuration.id.clone()));
            }
        }
        Ok(Self(configurations))
    }

    /// Resolve only exact matches; no production-speed evidence is implied
    pub(crate) fn resolve(&self, request: Request) -> Result<Resolved, OverrideError> {
        validate_target(request.target)?;
        let Choice::Configuration(id) = request.choice else {
            return Ok(Resolved::Library);
        };
        let configuration = self
            .0
            .iter()
            .find(|configuration| configuration.id == id)
            .ok_or(OverrideError::UnknownConfiguration(id))?;
        if configuration.boundary.area != request.boundary.area {
            return Err(OverrideError::AreaMismatch);
        }
        if configuration.boundary != request.boundary {
            return Err(OverrideError::TupleMismatch);
        }
        if configuration.target != request.target {
            return Err(OverrideError::TargetMismatch);
        }
        Ok(Resolved::Custom(Token {
            id: configuration.id.clone(),
            boundary: request.boundary,
            target: request.target,
            evidence: configuration.evidence,
        }))
    }
}

/// A custom selection that can only be created by exact evidence-backed resolution
#[derive(Debug)]
pub(crate) struct Token {
    id: String,
    boundary: Boundary,
    target: Target,
    evidence: Evidence,
}

impl Token {
    /// Return the selected configuration without granting access to token construction
    pub(crate) fn configuration_id(&self) -> &str {
        &self.id
    }

    /// Reject reuse of a token for a different configuration or execution tuple
    pub(crate) fn check(
        &self,
        id: &str,
        boundary: &Boundary,
        target: Target,
    ) -> Result<(), OverrideError> {
        validate_target(target)?;
        if self.id != id {
            return Err(OverrideError::InvalidConfiguration);
        }
        if self.boundary.area != boundary.area {
            return Err(OverrideError::AreaMismatch);
        }
        if self.boundary != *boundary {
            return Err(OverrideError::TupleMismatch);
        }
        if self.target != target {
            return Err(OverrideError::TargetMismatch);
        }
        Ok(())
    }
}

/// An override selection, never a production-speed qualification token
#[derive(Debug)]
pub(crate) enum Resolved {
    Library,
    Custom(Token),
}

impl Resolved {
    /// State explicitly that override selection does not establish speed qualification
    pub(crate) fn diagnostic(&self) -> &'static str {
        "CUDA override is unqualified for speed"
    }

    /// Log the limitation and, for custom selections, the exact pinned proofs
    pub(crate) fn log(&self) {
        match self {
            Self::Library => tracing::warn!("{}; selected Library", self.diagnostic()),
            Self::Custom(token) => tracing::warn!(
                configuration = token.id,
                accuracy = ?token.evidence.accuracy,
                deterministic = ?token.evidence.deterministic,
                "{}", self.diagnostic()
            ),
        }
    }
}

/// Typed failures that cannot produce a custom override token
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum OverrideError {
    #[error("unsupported override area")]
    InvalidArea,
    #[error("invalid override boundary or batch")]
    InvalidTuple,
    #[error("override target cannot execute its selected artifact or tier")]
    InvalidTarget,
    #[error("invalid override configuration")]
    InvalidConfiguration,
    #[error("duplicate override configuration ID: {0}")]
    DuplicateId(String),
    #[error("unknown override configuration ID: {0}")]
    UnknownConfiguration(String),
    #[error("override area does not match evidence")]
    AreaMismatch,
    #[error("override tuple does not match evidence")]
    TupleMismatch,
    #[error("override target does not match evidence")]
    TargetMismatch,
    #[error("configuration has no pinned accuracy proof")]
    MissingAccuracy,
    #[error("configuration has no pinned deterministic proof")]
    MissingDeterminism,
}

#[cfg(test)]
mod tests {
    use super::{
        Boundary, Choice, Configuration, ConfigurationSet, Evidence, OverrideError, Request,
        Resolved, Target,
    };
    use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};
    use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

    fn target() -> Target {
        Target {
            tier: PtxTier::Sm75,
            device: ComputeCapability::new(12, 0),
            artifact: LoadedArtifact::PtxJit {
                sha256: ArtifactHash::of(b"kernel"),
            },
        }
    }

    fn boundary(batch: usize) -> Boundary {
        Boundary::new(
            KernelModule::Resnet,
            "resnet.layer1.conv1",
            batch,
            CudaMath::Fp32,
        )
        .unwrap()
    }

    fn configuration(id: &str) -> Configuration {
        Configuration::new(
            id,
            boundary(1),
            target(),
            Evidence::new(
                Some(ArtifactHash::of(b"accuracy")),
                Some(ArtifactHash::of(b"deterministic")),
            )
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn exact_selection_and_token_reuse() {
        let set = ConfigurationSet::new(vec![configuration("tile-a")]).unwrap();
        let selection = set
            .resolve(Request::new(
                boundary(1),
                target(),
                Choice::Configuration("tile-a".to_owned()),
            ))
            .unwrap();
        let Resolved::Custom(token) = &selection else {
            panic!("expected custom selection")
        };
        assert_eq!(token.configuration_id(), "tile-a");
        assert_eq!(token.check("tile-a", &boundary(1), target()), Ok(()));
        assert_eq!(
            token.check("tile-b", &boundary(1), target()),
            Err(OverrideError::InvalidConfiguration)
        );
        assert_eq!(
            token.check("tile-a", &boundary(32), target()),
            Err(OverrideError::TupleMismatch)
        );
        assert!(selection.diagnostic().contains("unqualified for speed"));
        selection.log();
    }

    #[test]
    fn resolution_requires_exact_configuration_tuple_and_target() {
        let set = ConfigurationSet::new(vec![configuration("tile-a")]).unwrap();
        let resolve = |boundary, target, id: &str| {
            set.resolve(Request::new(
                boundary,
                target,
                Choice::Configuration(id.to_owned()),
            ))
            .unwrap_err()
        };
        assert_eq!(
            resolve(boundary(1), target(), "unknown"),
            OverrideError::UnknownConfiguration("unknown".to_owned())
        );
        assert_eq!(
            resolve(boundary(32), target(), "tile-a"),
            OverrideError::TupleMismatch
        );
        let other_area =
            Boundary::new(KernelModule::Lstm, "lstm.stack", 1, CudaMath::Fp32).unwrap();
        assert_eq!(
            resolve(other_area, target(), "tile-a"),
            OverrideError::AreaMismatch
        );
        let other_math = Boundary::new(
            KernelModule::Resnet,
            "resnet.layer1.conv1",
            1,
            CudaMath::Tf32,
        )
        .unwrap();
        assert_eq!(
            resolve(other_math, target(), "tile-a"),
            OverrideError::TupleMismatch
        );
        let targets = [
            Target {
                tier: PtxTier::Sm80,
                ..target()
            },
            Target {
                device: ComputeCapability::new(12, 1),
                ..target()
            },
            Target {
                artifact: LoadedArtifact::PtxJit {
                    sha256: ArtifactHash::of(b"other kernel"),
                },
                ..target()
            },
            Target {
                artifact: LoadedArtifact::Cubin {
                    arch: target().device,
                    sha256: ArtifactHash::of(b"kernel"),
                },
                ..target()
            },
        ];
        for other_target in targets {
            assert_eq!(
                resolve(boundary(1), other_target, "tile-a"),
                OverrideError::TargetMismatch
            );
        }
    }

    #[test]
    fn invalid_records_cannot_enter_the_set() {
        assert_eq!(
            ConfigurationSet::new(vec![configuration("same"), configuration("same")]).unwrap_err(),
            OverrideError::DuplicateId("same".to_owned())
        );
        assert_eq!(
            Evidence::new(None, Some(ArtifactHash::of(b"proof"))).unwrap_err(),
            OverrideError::MissingAccuracy
        );
        assert_eq!(
            Evidence::new(Some(ArtifactHash::of(b"proof")), None).unwrap_err(),
            OverrideError::MissingDeterminism
        );
        assert_eq!(
            Boundary::new(KernelModule::Probe, "probe.layer", 1, CudaMath::Fp32).unwrap_err(),
            OverrideError::InvalidArea
        );
        for (name, batch) in [
            ("resnet.layer", 0),
            ("lstm.stack", 1),
            ("resnet..layer", 1),
            ("resnet", 1),
        ] {
            assert_eq!(
                Boundary::new(KernelModule::Resnet, name, batch, CudaMath::Fp32).unwrap_err(),
                OverrideError::InvalidTuple
            );
        }
        assert_eq!(
            Configuration::new(" ", boundary(1), target(), configuration("valid").evidence)
                .unwrap_err(),
            OverrideError::InvalidConfiguration
        );
    }

    #[test]
    fn impossible_targets_are_rejected_before_selection() {
        let set = ConfigurationSet::new(vec![configuration("tile-a")]).unwrap();
        for invalid in [
            Target {
                device: ComputeCapability::new(7, 0),
                ..target()
            },
            Target {
                artifact: LoadedArtifact::Cubin {
                    arch: ComputeCapability::new(8, 0),
                    sha256: ArtifactHash::of(b"kernel"),
                },
                ..target()
            },
        ] {
            assert_eq!(
                set.resolve(Request::new(boundary(1), invalid, Choice::Library))
                    .unwrap_err(),
                OverrideError::InvalidTarget
            );
            assert_eq!(
                Configuration::new(
                    "invalid",
                    boundary(1),
                    invalid,
                    configuration("valid").evidence
                )
                .unwrap_err(),
                OverrideError::InvalidTarget
            );
        }
    }

    #[test]
    fn all_production_areas_support_library_and_custom_selection() {
        for area in [
            KernelModule::Fbank,
            KernelModule::Embedding,
            KernelModule::Segmentation,
            KernelModule::Resnet,
            KernelModule::Lstm,
            KernelModule::Sincnet,
        ] {
            let name = format!("{}.boundary", area.name());
            let boundary = Boundary::new(area, &name, 1, CudaMath::Fp32).unwrap();
            let configuration = Configuration::new(
                "area-config",
                boundary.clone(),
                target(),
                configuration("proof-owner").evidence,
            )
            .unwrap();
            let set = ConfigurationSet::new(vec![configuration]).unwrap();
            let library = set
                .resolve(Request::new(boundary.clone(), target(), Choice::Library))
                .unwrap();
            assert!(matches!(library, Resolved::Library));
            let selected = set
                .resolve(Request::new(
                    boundary.clone(),
                    target(),
                    Choice::Configuration("area-config".to_owned()),
                ))
                .unwrap();
            let Resolved::Custom(token) = selected else {
                panic!("expected custom selection")
            };
            assert_eq!(token.check("area-config", &boundary, target()), Ok(()));
        }
    }

    #[test]
    fn library_does_not_require_custom_evidence() {
        let selection = ConfigurationSet::new(vec![])
            .unwrap()
            .resolve(Request::new(boundary(32), target(), Choice::Library))
            .unwrap();
        assert!(matches!(selection, Resolved::Library));
        assert!(selection.diagnostic().contains("unqualified for speed"));
    }
}
