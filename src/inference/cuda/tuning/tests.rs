//! Host-only proof that timing cannot create an execution permission

use super::{ApprovedChoice, BenchmarkMeasurement, Catalogue, Tuple, select_winners};
use crate::inference::cuda::device::{DeviceAttributes, test_support::Builder};
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::{ComputeCapability, CudaMath, PtxTier};

pub(crate) fn approved_choice(
    device: &DeviceAttributes,
    boundary: BoundaryId,
    batch: usize,
    math: CudaMath,
    tier: PtxTier,
) -> ApprovedChoice {
    let catalogue = Catalogue::new(device, tier).unwrap();
    catalogue
        .choices(Tuple::new(boundary, batch, math).unwrap())
        .iter()
        .find(|choice| matches!(choice, ApprovedChoice::Kernel(_)))
        .unwrap()
        .clone()
}

#[test]
fn winner_is_fastest_approved_choice_for_the_exact_tuple() {
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .build();
    let kernel = approved_choice(&device, boundary, 32, CudaMath::Tf32, PtxTier::Sm80);
    let (rows, entries) = select_winners(vec![
        BenchmarkMeasurement {
            boundary,
            batch: 32,
            math: CudaMath::Tf32,
            choice: kernel.clone(),
            median_ms: 2.0,
        },
        BenchmarkMeasurement {
            boundary,
            batch: 32,
            math: CudaMath::Tf32,
            choice: ApprovedChoice::Library,
            median_ms: 3.0,
        },
    ])
    .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].selected, kernel.label());
    assert_eq!(rows[0].candidates.len(), 2);
    assert_eq!(entries[0].choice, kernel.key());
    assert_eq!(entries[0].median_ms, 2.0);
}

#[test]
fn invalid_times_cannot_be_written_as_measurements() {
    for median_ms in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let result = select_winners(vec![BenchmarkMeasurement {
            boundary: BoundaryId::named("fbank.dft"),
            batch: 1,
            math: CudaMath::Fp32,
            choice: ApprovedChoice::Library,
            median_ms,
        }]);
        assert!(result.is_err());
    }
    assert!(select_winners(Vec::new()).is_err());
}

#[test]
fn rejected_projection_and_sinc_precision_do_not_enter_the_catalogue() {
    use crate::inference::cuda::candidate::{ConfigPin, LstmPin, LstmProjection};
    for (capability, batch) in [
        (ComputeCapability::new(12, 0), 1),
        (ComputeCapability::new(12, 0), 32),
        (ComputeCapability::new(8, 9), 32),
    ] {
        if capability == ComputeCapability::new(8, 9) && !cfg!(feature = "cuda-sm80") {
            continue;
        }
        let (sms, name) = if capability == ComputeCapability::new(8, 9) {
            (34, "NVIDIA GeForce RTX 4060 Ti")
        } else {
            (36, "NVIDIA GeForce RTX 5060 Ti")
        };
        let device = Builder::new(capability)
            .multiprocessors(sms)
            .name(name)
            .build();
        let catalogue = Catalogue::new(&device, PtxTier::Sm120).unwrap();
        let lstm = Tuple::new(BoundaryId::named("lstm.stack"), batch, CudaMath::Tf32).unwrap();
        assert!(
            catalogue
                .choices(lstm)
                .iter()
                .any(|choice| matches!(choice, ApprovedChoice::Kernel(_)))
        );
        for choice in catalogue.choices(lstm) {
            if let ApprovedChoice::Kernel(config) = choice {
                assert!(matches!(
                    config.pin(),
                    ConfigPin::Lstm(LstmPin::Projected(
                        LstmProjection::Small | LstmProjection::Large
                    ))
                ));
            }
        }
        let sinc = Tuple::new(
            BoundaryId::named("sincnet.conv0.abs_pool"),
            batch,
            CudaMath::Tf32,
        )
        .unwrap();
        assert!(
            catalogue
                .choices(sinc)
                .iter()
                .all(|choice| matches!(choice, ApprovedChoice::Library))
        );
    }
}

#[test]
#[cfg(any(feature = "cuda-sm80", feature = "cuda-sm90", feature = "cuda-sm120"))]
fn catalogue_exposes_only_current_pins_and_approved_scalar_trunk_alternatives() {
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .build();
    let catalogue = Catalogue::new(&device, PtxTier::Sm80).unwrap();
    let tuple = Tuple::new(
        BoundaryId::named("resnet.layer2.0.conv1"),
        32,
        CudaMath::Tf32,
    )
    .unwrap();
    let kernels: Vec<_> = catalogue
        .choices(tuple)
        .iter()
        .filter_map(|choice| match choice {
            ApprovedChoice::Kernel(config) => Some(config),
            _ => None,
        })
        .collect();
    assert_eq!(kernels.len(), 2);
    assert_eq!(kernels[0].family, "default");
    assert_eq!(kernels[1].family, "fp32");
    assert_ne!(kernels[0].pin(), kernels[1].pin());
    assert!(kernels.iter().all(|config| config.matches(
        tuple.boundary,
        tuple.batch,
        tuple.math.into()
    )));
    assert!(
        kernels
            .iter()
            .all(|config| !config.matches(tuple.boundary, 1, tuple.math.into()))
    );
}

#[test]
fn catalogue_filters_the_pipeline_points_and_batch_classes() {
    use crate::inference::cuda::KernelModule;
    for (cc, sms, shared, name) in [
        (ComputeCapability::new(7, 5), 40, 65536, "Tesla T4"),
        (
            ComputeCapability::new(8, 0),
            108,
            163840,
            "NVIDIA A100-PCIE-40GB",
        ),
        (
            ComputeCapability::new(8, 0),
            108,
            163840,
            "NVIDIA A100-SXM4-40GB",
        ),
        (
            ComputeCapability::new(8, 9),
            34,
            101376,
            "NVIDIA GeForce RTX 4060 Ti",
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            101376,
            "NVIDIA GeForce RTX 5060 Ti",
        ),
    ] {
        if cc.major == 7 && !cfg!(feature = "cuda-sm75")
            || cc.major == 8 && !cfg!(feature = "cuda-sm80")
            || cc.major >= 12
                && !cfg!(any(
                    feature = "cuda-sm80",
                    feature = "cuda-sm90",
                    feature = "cuda-sm120"
                ))
        {
            continue;
        }
        let device = Builder::new(cc)
            .multiprocessors(sms)
            .shared_optin_bytes(shared)
            .name(name)
            .build();
        let catalogue = Catalogue::new(&device, PtxTier::Sm120).unwrap();
        for boundary in
            BoundaryId::all().filter(|id| *id != BoundaryId::named("lstm.stack.input_proj"))
        {
            let math = match boundary.area() {
                KernelModule::Resnet | KernelModule::Embedding => CudaMath::Tf32,
                _ => CudaMath::Fp32,
            };
            let batches = boundary.batches().iter();
            for batch in batches {
                let tuple = Tuple::new(boundary, batch, math).unwrap();
                // these implemented defaults lack portable algorithm evidence;
                // measured recipes remain outside the tuner's approval owner
                let t4_wide_winograd = cc.major == 7
                    && (boundary.name().starts_with("resnet.layer3.")
                        || boundary.name().starts_with("resnet.layer4."))
                    && !boundary.name().ends_with("shortcut.0")
                    && !matches!(
                        boundary.name(),
                        "resnet.layer3.0.conv1" | "resnet.layer4.0.conv1"
                    );
                let expects_kernel = !t4_wide_winograd;
                assert_eq!(
                    catalogue
                        .choices(tuple)
                        .iter()
                        .any(|choice| matches!(choice, ApprovedChoice::Kernel(_))),
                    expects_kernel,
                    "{name} {boundary} b{batch}",
                );
            }
        }
    }
}

/// An H100 rule implements two-product Winograd without portable accuracy approval
#[cfg(feature = "cuda-sm80")]
pub(crate) fn unapproved_configuration() -> (
    DeviceAttributes,
    Tuple,
    crate::inference::cuda::kernels::ModuleRequest,
    crate::inference::cuda::candidate::ConfigPin,
) {
    use crate::inference::cuda::candidate::{
        ConfigPin, WideconvAlgorithm, WideconvPin, WideconvProducts,
    };
    let device = Builder::new(ComputeCapability::new(9, 0))
        .multiprocessors(132)
        .shared_optin_bytes(227328)
        .name("NVIDIA H100")
        .build();
    let boundary = BoundaryId::named("resnet.layer3.1.conv1");
    let tuple = Tuple::new(boundary, 32, CudaMath::Tf32).unwrap();
    let (_, _, _, module, pin, _) =
        crate::inference::cuda::implementation::tuning_configurations(&device, PtxTier::Sm80)
            .unwrap()
            .into_iter()
            .find(|(id, batch, math, _, pin, _)| {
                *id == boundary
                    && *batch == 32
                    && *math == CudaMath::Tf32
                    && matches!(pin, ConfigPin::Wideconv(WideconvPin::Configured(config))
                if config.algorithm == WideconvAlgorithm::Winograd(WideconvProducts::Tf32x2))
            })
            .expect("the implemented H100 default uses two products");
    (device, tuple, module, pin)
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn implemented_two_product_winograd_is_not_accuracy_approved() {
    let (device, tuple, module, pin) = unapproved_configuration();
    let catalogue = Catalogue::new(&device, PtxTier::Sm80).unwrap();
    assert!(catalogue.choices(tuple).iter().all(|choice| {
        !matches!(choice, ApprovedChoice::Kernel(config) if config.module() == module && config.pin() == pin)
    }));
    // an unknown device still has portable approved kernels, without a device recipe
    let direct = Tuple::new(
        BoundaryId::named("resnet.layer1.0.conv1"),
        32,
        CudaMath::Tf32,
    )
    .unwrap();
    assert!(
        catalogue
            .choices(direct)
            .iter()
            .any(|choice| matches!(choice, ApprovedChoice::Kernel(_)))
    );
}

#[test]
fn math_modes_have_explicit_algorithm_limits() {
    use super::accuracy::Policy;
    use crate::inference::cuda::candidate::{
        ConfigPin, ConvKernel, ConvPin, LstmPin, LstmProjection,
    };
    let boundary = BoundaryId::named("resnet.layer1.0.conv1");
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        assert!(
            Policy::approve(
                boundary,
                math,
                ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32))
            )
            .is_some()
        );
        assert_eq!(
            Policy::approve(
                boundary,
                math,
                ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32Tensor))
            )
            .is_some(),
            math == CudaMath::Tf32
        );
        assert!(
            Policy::approve(
                BoundaryId::named("lstm.stack"),
                math,
                ConfigPin::Lstm(LstmPin::Projected(LstmProjection::Tensor))
            )
            .is_none()
        );
    }
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn winograd_approval_is_limited_to_the_reviewed_algorithms_and_mode() {
    use super::accuracy::Policy;
    use crate::inference::cuda::candidate::{
        ConfigPin, WideconvAlgorithm, WideconvPin, WideconvProducts,
    };
    let (_, tuple, _, pin) = unapproved_configuration();
    let ConfigPin::Wideconv(WideconvPin::Configured(mut config)) = pin else {
        panic!("fixed wideconv pin")
    };
    for products in [
        WideconvProducts::Tf32x1,
        WideconvProducts::Tf32x2,
        WideconvProducts::Tf32x3,
        WideconvProducts::Bf16x3,
        WideconvProducts::Fp32Sweep2,
    ] {
        config.algorithm = WideconvAlgorithm::Winograd(products);
        for math in [CudaMath::Fp32, CudaMath::Tf32] {
            assert!(
                Policy::approve(
                    tuple.boundary,
                    math,
                    ConfigPin::Wideconv(WideconvPin::Configured(config))
                )
                .is_none(),
                "{products:?} {math:?}"
            );
        }
    }
    config.algorithm = WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1Staged);
    assert!(
        Policy::approve(
            tuple.boundary,
            CudaMath::Tf32,
            ConfigPin::Wideconv(WideconvPin::Configured(config))
        )
        .is_some()
    );
    assert!(
        Policy::approve(
            tuple.boundary,
            CudaMath::Fp32,
            ConfigPin::Wideconv(WideconvPin::Configured(config))
        )
        .is_none()
    );
    config.algorithm = WideconvAlgorithm::Winograd(WideconvProducts::Fp32);
    let c64 = BoundaryId::named("resnet.layer2.1.conv1");
    assert!(
        Policy::approve(
            c64,
            CudaMath::Tf32,
            ConfigPin::Wideconv(WideconvPin::Configured(config))
        )
        .is_some()
    );
    assert!(
        Policy::approve(
            c64,
            CudaMath::Fp32,
            ConfigPin::Wideconv(WideconvPin::Configured(config))
        )
        .is_none()
    );
    assert!(
        Policy::approve(
            tuple.boundary,
            CudaMath::Tf32,
            ConfigPin::Wideconv(WideconvPin::Configured(config))
        )
        .is_none()
    );
    assert!(
        Policy::approve(
            BoundaryId::named("resnet.layer2.0.conv1"),
            CudaMath::Tf32,
            ConfigPin::Wideconv(WideconvPin::Configured(config)),
        )
        .is_none()
    );
    config.algorithm = WideconvAlgorithm::ImplicitGemm;
    assert!(
        Policy::approve(
            tuple.boundary,
            CudaMath::Tf32,
            ConfigPin::Wideconv(WideconvPin::Configured(config)),
        )
        .is_none()
    );
}

#[test]
fn benchmark_visits_every_exact_approved_choice() {
    use super::{BenchKind, TuneControl};
    for (cc, sms, name, tier) in [
        (ComputeCapability::new(7, 5), 40, "Tesla T4", PtxTier::Sm75),
        (
            ComputeCapability::new(8, 9),
            128,
            "NVIDIA GeForce RTX 4090",
            PtxTier::Sm80,
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
            PtxTier::Sm120,
        ),
    ] {
        if (cc.major == 7 && !cfg!(feature = "cuda-sm75"))
            || (cc.major == 8 && !cfg!(feature = "cuda-sm80"))
            || (cc.major == 12 && !cfg!(feature = "cuda-sm120"))
        {
            continue;
        }
        let device = Builder::new(cc).multiprocessors(sms).name(name).build();
        let catalogue = Catalogue::new(&device, tier).unwrap();
        let passes: Vec<_> = catalogue.benchmark_kinds().collect();
        let controls: Vec<_> = passes
            .iter()
            .map(|kind| TuneControl::benchmark(*kind, &device, tier).unwrap())
            .collect();
        for (tuple, approved) in &catalogue.0 {
            let visited: Vec<_> = controls
                .iter()
                .filter_map(|control| {
                    control
                        .choice(tuple.boundary, tuple.batch, tuple.math.into())
                        .map(|choice| choice.key())
                })
                .collect();
            assert_eq!(
                visited,
                approved.iter().map(ApprovedChoice::key).collect::<Vec<_>>(),
                "{name} {tuple:?}"
            );
        }
        for (tuple, approved) in catalogue.0.iter().take(1) {
            let control =
                TuneControl::benchmark(BenchKind::CatalogueSlot(approved.len()), &device, tier)
                    .unwrap();
            assert!(
                control
                    .choice(tuple.boundary, tuple.batch, tuple.math.into())
                    .is_none()
            );
            assert_eq!(
                control.plan_choice(tuple.boundary, tuple.batch, tuple.math.into()),
                approved.first().cloned()
            );
        }
        if cc.major == 7 {
            let tuple = Tuple::new(
                BoundaryId::named("resnet.layer2.1.conv1"),
                32,
                CudaMath::Tf32,
            )
            .unwrap();
            let kernels: Vec<_> = catalogue
                .choices(tuple)
                .iter()
                .filter_map(|choice| match choice {
                    ApprovedChoice::Kernel(config) => Some(config),
                    ApprovedChoice::Library => None,
                })
                .collect();
            assert_eq!(kernels.len(), 2);
            assert!(kernels.iter().all(|config| config.family == "default"));
            assert_ne!(kernels[0].module(), kernels[1].module());
            assert_ne!(
                catalogue.choices(tuple)[0].label(),
                catalogue.choices(tuple)[1].label()
            );
        }
        assert!(matches!(passes[0], BenchKind::CatalogueSlot(0)));
    }
}
