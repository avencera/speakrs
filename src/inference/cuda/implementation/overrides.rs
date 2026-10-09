//! Accuracy-backed override requests, kept separate from speed-qualified production selection

use super::{BoundaryId, Target};
use crate::inference::cuda::CudaMath;
use crate::inference::cuda::candidate::ConfigPin;
use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact};

/// A validated execution tuple, independent of an implementation choice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Boundary {
    id: BoundaryId,
    batch: usize,
    math: CudaMath,
}

impl Boundary {
    /// Reject batch zero before evidence lookup; the identifier is already typed
    pub(crate) fn new(id: BoundaryId, batch: usize, math: CudaMath) -> Result<Self, OverrideError> {
        if batch == 0 {
            return Err(OverrideError::InvalidTuple);
        }
        Ok(Self { id, batch, math })
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

/// One complete configuration with evidence bound to its exact execution tuple
#[derive(Debug)]
pub(crate) struct Configuration {
    pin: ConfigPin,
    boundary: Boundary,
    target: Target,
    evidence: Evidence,
}

impl Configuration {
    /// Bind this configuration's proofs to one target, including the loaded artifact
    pub(crate) fn new(
        pin: ConfigPin,
        boundary: Boundary,
        target: Target,
        evidence: Evidence,
    ) -> Result<Self, OverrideError> {
        validate_target(target)?;
        if pin.area() != target.module.area() {
            return Err(OverrideError::AreaMismatch);
        }
        Ok(Self {
            pin,
            boundary,
            target,
            evidence,
        })
    }

    fn same_execution(&self, other: &Self) -> bool {
        self.pin == other.pin && self.boundary == other.boundary && self.target == other.target
    }
}

// a cubin for another device cannot be the artifact loaded at this target
fn validate_target(target: Target) -> Result<(), OverrideError> {
    if target.device < target.module.tier().min_capability()
        || matches!(target.module.artifact(), LoadedArtifact::Cubin { arch, .. } if arch != target.device)
    {
        return Err(OverrideError::InvalidTarget);
    }
    Ok(())
}

/// An explicit choice; Library policy is enforced by the runtime owner
#[derive(Debug)]
pub(crate) enum Choice {
    Library,
    Configuration(ConfigPin),
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

/// An enumerated set with one proof per configuration and execution tuple
#[derive(Debug)]
pub(crate) struct ConfigurationSet(Vec<Configuration>);

impl ConfigurationSet {
    /// Reject a second proof for the same pin, tuple and target
    pub(crate) fn new(configurations: Vec<Configuration>) -> Result<Self, OverrideError> {
        for (index, configuration) in configurations.iter().enumerate() {
            if configurations[..index]
                .iter()
                .any(|previous| previous.same_execution(configuration))
            {
                return Err(OverrideError::DuplicateConfiguration(configuration.pin));
            }
        }
        Ok(Self(configurations))
    }

    /// Resolve only exact matches; no production-speed evidence is implied
    pub(crate) fn resolve(&self, request: Request) -> Result<Resolved, OverrideError> {
        validate_target(request.target)?;
        let Choice::Configuration(pin) = request.choice else {
            return Ok(Resolved::Library);
        };
        if pin.area() != request.target.module.area() {
            return Err(OverrideError::AreaMismatch);
        }
        let pinned: Vec<_> = self.0.iter().filter(|config| config.pin == pin).collect();
        if pinned.is_empty() {
            return Err(OverrideError::UnknownConfiguration(pin));
        }
        let tuple: Vec<_> = pinned
            .into_iter()
            .filter(|config| config.boundary == request.boundary)
            .collect();
        if tuple.is_empty() {
            return Err(OverrideError::TupleMismatch);
        }
        let configuration = tuple
            .into_iter()
            .find(|config| config.target == request.target)
            .ok_or(OverrideError::TargetMismatch)?;
        Ok(Resolved::Custom(Token {
            pin,
            boundary: request.boundary,
            target: request.target,
            evidence: configuration.evidence,
        }))
    }
}

/// A custom selection that can only be created by exact evidence-backed resolution
#[derive(Debug)]
pub(crate) struct Token {
    pin: ConfigPin,
    boundary: Boundary,
    target: Target,
    evidence: Evidence,
}

impl Token {
    /// Return the selected configuration without granting access to token construction
    pub(crate) fn pin(&self) -> ConfigPin {
        self.pin
    }

    /// Reject reuse of a token for a different configuration or execution tuple
    pub(crate) fn check(
        &self,
        pin: ConfigPin,
        boundary: &Boundary,
        target: Target,
    ) -> Result<(), OverrideError> {
        validate_target(target)?;
        if self.pin != pin {
            return Err(OverrideError::ConfigurationMismatch);
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
                configuration = ?token.pin,
                boundary = %token.boundary.id,
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
    #[error("invalid override batch")]
    InvalidTuple,
    #[error("override target cannot execute its selected artifact or tier")]
    InvalidTarget,
    #[error("override token names another configuration")]
    ConfigurationMismatch,
    #[error("duplicate override configuration: {0:?}")]
    DuplicateConfiguration(ConfigPin),
    #[error("unknown override configuration: {0:?}")]
    UnknownConfiguration(ConfigPin),
    #[error("override configuration runs on another module area")]
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
    use crate::inference::cuda::candidate::{
        ConfigPin, ConvKernel, ConvPin, ConvShape, LstmPin, SincPin,
    };
    use crate::inference::cuda::implementation::BoundaryId;
    use crate::inference::cuda::kernels::{ArtifactHash, LoadedArtifact, ModuleRequest};
    use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

    const TILE_A: ConfigPin = ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32));
    const TILE_B: ConfigPin = ConfigPin::Conv(ConvPin::LegacyWaves(ConvShape::C32));

    fn module(area: KernelModule, tier: PtxTier, artifact: LoadedArtifact) -> ModuleRequest {
        ModuleRequest::new(area, tier, artifact)
    }

    fn target_for(area: KernelModule) -> Target {
        Target {
            module: module(
                area,
                PtxTier::Sm75,
                LoadedArtifact::PtxJit {
                    sha256: ArtifactHash::of(b"kernel"),
                },
            ),
            device: ComputeCapability::new(12, 0),
        }
    }

    fn target() -> Target {
        target_for(KernelModule::Resnet)
    }

    fn boundary(batch: usize) -> Boundary {
        Boundary::new(
            BoundaryId::named("resnet.layer1.0.conv1"),
            batch,
            CudaMath::Fp32,
        )
        .unwrap()
    }

    fn evidence() -> Evidence {
        Evidence::new(
            Some(ArtifactHash::of(b"accuracy")),
            Some(ArtifactHash::of(b"deterministic")),
        )
        .unwrap()
    }

    fn configuration(pin: ConfigPin) -> Configuration {
        Configuration::new(pin, boundary(1), target(), evidence()).unwrap()
    }

    #[test]
    fn exact_selection_and_token_reuse() {
        let set = ConfigurationSet::new(vec![configuration(TILE_A)]).unwrap();
        let selection = set
            .resolve(Request::new(
                boundary(1),
                target(),
                Choice::Configuration(TILE_A),
            ))
            .unwrap();
        let Resolved::Custom(token) = &selection else {
            panic!("expected custom selection")
        };
        assert_eq!(token.pin(), TILE_A);
        assert_eq!(token.check(TILE_A, &boundary(1), target()), Ok(()));
        assert_eq!(
            token.check(TILE_B, &boundary(1), target()),
            Err(OverrideError::ConfigurationMismatch)
        );
        assert_eq!(
            token.check(TILE_A, &boundary(32), target()),
            Err(OverrideError::TupleMismatch)
        );
        assert!(selection.diagnostic().contains("unqualified for speed"));
        selection.log();
    }

    #[test]
    fn resolution_requires_exact_configuration_tuple_and_target() {
        let set = ConfigurationSet::new(vec![configuration(TILE_A)]).unwrap();
        let resolve = |boundary, target, pin| {
            set.resolve(Request::new(boundary, target, Choice::Configuration(pin)))
                .unwrap_err()
        };
        assert_eq!(
            resolve(boundary(1), target(), TILE_B),
            OverrideError::UnknownConfiguration(TILE_B)
        );
        assert_eq!(
            resolve(boundary(32), target(), TILE_A),
            OverrideError::TupleMismatch
        );
        // a pin for another module area cannot be resolved against this target
        assert_eq!(
            resolve(
                boundary(1),
                target(),
                ConfigPin::Lstm(LstmPin::LegacyCooperative)
            ),
            OverrideError::AreaMismatch
        );
        let other_math = Boundary::new(
            BoundaryId::named("resnet.layer1.0.conv1"),
            1,
            CudaMath::Tf32,
        )
        .unwrap();
        assert_eq!(
            resolve(other_math, target(), TILE_A),
            OverrideError::TupleMismatch
        );
        let other_layer = Boundary::new(
            BoundaryId::named("resnet.layer1.0.conv2"),
            1,
            CudaMath::Fp32,
        )
        .unwrap();
        assert_eq!(
            resolve(other_layer, target(), TILE_A),
            OverrideError::TupleMismatch
        );
        let artifact = target().module.artifact();
        let targets = [
            Target {
                module: module(KernelModule::Resnet, PtxTier::Sm80, artifact),
                ..target()
            },
            Target {
                device: ComputeCapability::new(12, 1),
                ..target()
            },
            Target {
                module: module(
                    KernelModule::Resnet,
                    PtxTier::Sm75,
                    LoadedArtifact::PtxJit {
                        sha256: ArtifactHash::of(b"other kernel"),
                    },
                ),
                ..target()
            },
            Target {
                module: module(
                    KernelModule::Resnet,
                    PtxTier::Sm75,
                    LoadedArtifact::Cubin {
                        arch: target().device,
                        sha256: ArtifactHash::of(b"kernel"),
                    },
                ),
                ..target()
            },
        ];
        for other_target in targets {
            assert_eq!(
                resolve(boundary(1), other_target, TILE_A),
                OverrideError::TargetMismatch
            );
        }
    }

    #[test]
    fn invalid_records_cannot_enter_the_set() {
        assert_eq!(
            ConfigurationSet::new(vec![configuration(TILE_A), configuration(TILE_A)]).unwrap_err(),
            OverrideError::DuplicateConfiguration(TILE_A)
        );
        // one pin may carry proofs for several tuples
        assert!(
            ConfigurationSet::new(vec![
                configuration(TILE_A),
                Configuration::new(TILE_A, boundary(32), target(), evidence()).unwrap(),
            ])
            .is_ok()
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
            Boundary::new(BoundaryId::named("lstm.stack"), 0, CudaMath::Fp32).unwrap_err(),
            OverrideError::InvalidTuple
        );
        assert_eq!(
            Configuration::new(
                ConfigPin::Sinc(SincPin::ConvAbsPool),
                boundary(1),
                target(),
                evidence()
            )
            .unwrap_err(),
            OverrideError::AreaMismatch
        );
    }

    #[test]
    fn impossible_targets_are_rejected_before_selection() {
        let set = ConfigurationSet::new(vec![configuration(TILE_A)]).unwrap();
        for invalid in [
            Target {
                device: ComputeCapability::new(7, 0),
                ..target()
            },
            Target {
                module: module(
                    KernelModule::Resnet,
                    PtxTier::Sm75,
                    LoadedArtifact::Cubin {
                        arch: ComputeCapability::new(8, 0),
                        sha256: ArtifactHash::of(b"kernel"),
                    },
                ),
                ..target()
            },
        ] {
            assert_eq!(
                set.resolve(Request::new(boundary(1), invalid, Choice::Library))
                    .unwrap_err(),
                OverrideError::InvalidTarget
            );
            assert_eq!(
                Configuration::new(TILE_A, boundary(1), invalid, evidence()).unwrap_err(),
                OverrideError::InvalidTarget
            );
        }
    }

    #[test]
    fn every_candidate_area_supports_library_and_custom_selection() {
        for (pin, name) in [
            (TILE_A, "resnet.layer2.1.conv1"),
            (ConfigPin::Lstm(LstmPin::LegacyCooperative), "lstm.stack"),
            (
                ConfigPin::Sinc(SincPin::ConvAbsPool),
                "sincnet.conv0.abs_pool",
            ),
        ] {
            let target = target_for(pin.area());
            let boundary = Boundary::new(BoundaryId::named(name), 1, CudaMath::Fp32).unwrap();
            let configuration = Configuration::new(pin, boundary, target, evidence()).unwrap();
            let set = ConfigurationSet::new(vec![configuration]).unwrap();
            let library = set
                .resolve(Request::new(boundary, target, Choice::Library))
                .unwrap();
            assert!(matches!(library, Resolved::Library));
            let selected = set
                .resolve(Request::new(boundary, target, Choice::Configuration(pin)))
                .unwrap();
            let Resolved::Custom(token) = selected else {
                panic!("expected custom selection")
            };
            assert_eq!(token.check(pin, &boundary, target), Ok(()));
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
