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

/// Select an approved FP16 choice without depending on catalogue order
#[cfg(feature = "_cuda-libraries")]
pub(crate) fn approved_fp16_choice(
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
        .find(|choice| choice.is_fp16())
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
    assert_eq!(kernels.len(), 3);
    let named = |family| {
        kernels
            .iter()
            .find(|config| config.family == family)
            .unwrap()
    };
    assert!(named("fp16").pin().is_fp16());
    assert_ne!(named("default").pin(), named("fp32").pin());
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
                let expects_kernel = true;
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

/// An implemented two-product configuration without portable accuracy approval
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
            .find(|(id, batch, math, _, _, _)| {
                *id == boundary && *batch == 32 && *math == CudaMath::Tf32
            })
            .expect("the H100 default is implemented");
    let ConfigPin::Wideconv(WideconvPin::Configured(mut config)) = pin else {
        panic!("configured wideconv pin")
    };
    config.algorithm = WideconvAlgorithm::Winograd(WideconvProducts::Tf32x2);
    let pin = ConfigPin::Wideconv(WideconvPin::Configured(config));
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
fn fp16_accuracy_approval_is_tf32_only_for_both_tile_sizes() {
    use super::accuracy::{Approval, Policy};
    use crate::inference::cuda::candidate::{ConfigPin, WideconvFp16Tiles, WideconvPin};
    let boundary = BoundaryId::named("resnet.layer3.1.conv1");
    let pin = WideconvPin::fp16_wide(boundary.name(), 32, CudaMath::Tf32).unwrap();
    let WideconvPin::Configured(mut config) = pin else {
        panic!("FP16 pin")
    };
    for tiles in [WideconvFp16Tiles::Wide, WideconvFp16Tiles::Narrow] {
        config.algorithm = crate::inference::cuda::candidate::WideconvAlgorithm::Fp16(tiles);
        let pin = ConfigPin::Wideconv(WideconvPin::Configured(config));
        assert_eq!(
            Policy::approve(boundary, CudaMath::Tf32, pin),
            Some(Approval::Fp16Trunk)
        );
        assert_eq!(Policy::approve(boundary, CudaMath::Fp32, pin), None);
    }
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
            .map(|kind| TuneControl::benchmark(*kind, true, &device, tier).unwrap())
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
            let control = TuneControl::benchmark(
                BenchKind::CatalogueSlot(approved.len()),
                true,
                &device,
                tier,
            )
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
            assert_eq!(kernels.len(), 3);
            assert_eq!(
                kernels
                    .iter()
                    .filter(|config| config.family == "default")
                    .count(),
                2
            );
            assert_eq!(
                kernels
                    .iter()
                    .filter(|config| config.family == "fp16")
                    .count(),
                1
            );
            assert_ne!(kernels[0].module(), kernels[2].module());
            assert_ne!(
                catalogue.choices(tuple)[0].label(),
                catalogue.choices(tuple)[1].label()
            );
        }
        assert!(matches!(passes[0], BenchKind::CatalogueSlot(0)));
    }
}

#[test]
fn duplicate_candidate_identities_cannot_form_a_report_row() {
    let boundary = BoundaryId::named("resnet.seg_1");
    let measurement = || BenchmarkMeasurement {
        boundary,
        batch: 1,
        math: CudaMath::Tf32,
        choice: ApprovedChoice::Library,
        median_ms: 0.1,
    };
    assert!(
        matches!(select_winners(vec![measurement(), measurement()]), Err(super::CudaTuneError::Invalid(reason)) if reason.contains("duplicate candidate identity"))
    );
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn fp16_tuning_retains_the_non_fp16_4060_ti_wide_choice() {
    use super::TuneControl;
    use crate::inference::cuda::candidate::{
        ConfigPin, DriverCandidate, Fp16Policy, WideconvOxide,
    };

    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let tier = PtxTier::Sm80;
    let catalogue = Catalogue::new(&device, tier).unwrap();
    for name in ["resnet.layer3.1.conv1", "resnet.layer4.1.conv1"] {
        let boundary = BoundaryId::named(name);
        for batch in [8, 16, 32] {
            let tuple = Tuple::new(boundary, batch, CudaMath::Tf32).unwrap();
            let choices = catalogue.choices(tuple);
            let non_fp16 = WideconvOxide::driver_pin(
                boundary,
                batch,
                CudaMath::Tf32,
                &device,
                tier,
                Fp16Policy::Excluded,
            )
            .unwrap();
            assert!(!non_fp16.is_fp16());
            assert!(choices.iter().any(|choice| matches!(choice,
                ApprovedChoice::Kernel(config) if config.pin() == non_fp16)));
            assert!(choices.iter().any(|choice| matches!(choice,
                ApprovedChoice::Kernel(config) if matches!(config.pin(), ConfigPin::Wideconv(_)) && config.pin().is_fp16())));
            let visited: Vec<_> = catalogue
                .benchmark_kinds()
                .filter_map(|kind| {
                    TuneControl::benchmark(kind, true, &device, tier)
                        .unwrap()
                        .choice(boundary, batch, CudaMath::Tf32)
                })
                .collect();
            assert_eq!(visited, choices);
        }
    }
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn fp16_tuning_is_discoverable_without_an_ada_recipe_and_never_in_fp32() {
    use super::TuneControl;
    use crate::inference::cuda::candidate::{ConfigPin, WideconvPin};

    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(46)
        .name("NVIDIA GeForce RTX 4070")
        .build();
    let tier = PtxTier::Sm80;
    let catalogue = Catalogue::new(&device, tier).unwrap();
    for name in [
        "resnet.layer1.1.conv1",
        "resnet.layer2.1.conv1",
        "resnet.layer3.1.conv1",
        "resnet.layer4.1.conv1",
    ] {
        let boundary = BoundaryId::named(name);
        for batch in [1, 4, 8, 16, 32] {
            let expected =
                ConfigPin::Wideconv(WideconvPin::fp16_wide(name, batch, CudaMath::Tf32).unwrap());
            let tuple = Tuple::new(boundary, batch, CudaMath::Tf32).unwrap();
            assert!(
                catalogue
                    .choices(tuple)
                    .iter()
                    .any(|choice| matches!(choice,
                ApprovedChoice::Kernel(config) if config.pin() == expected))
            );
            assert!(catalogue.benchmark_kinds().any(|kind| {
                TuneControl::benchmark(kind, true, &device, tier)
                    .unwrap()
                    .choice(boundary, batch, CudaMath::Tf32)
                    .is_some_and(|choice| choice.is_fp16())
            }));
        }
    }
    assert!(
        catalogue
            .0
            .iter()
            .filter(|(tuple, _)| tuple.math == CudaMath::Fp32.into())
            .all(|(_, choices)| choices.iter().all(|choice| !choice.is_fp16()))
    );
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn measured_fp16_dense_does_not_grant_portable_tuner_approval() {
    use super::accuracy::Policy;
    use crate::inference::cuda::candidate::{ConfigPin, SegdenseEntry};

    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .shared_optin_bytes(101376)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let (boundary, _, _, _, pin, _) =
        crate::inference::cuda::implementation::tuning_configurations(&device, PtxTier::Sm80)
            .unwrap()
            .into_iter()
            .find(|(_, _, _, _, pin, _)| matches!(pin, ConfigPin::Segdense(pin) if pin.entry() == SegdenseEntry::EmbedB32F16))
            .expect("the measured device implements the FP16 dense head");
    for math in [CudaMath::Fp32, CudaMath::Tf32] {
        assert!(Policy::approve(boundary, math, pin).is_none());
    }
}

#[test]
#[cfg(feature = "cuda-sm80")]
fn unmeasured_three_product_segmentation_is_approved_only_in_fp32() {
    use super::accuracy::Policy;
    use crate::inference::cuda::candidate::{ConfigPin, SegdenseEntry};

    let device = Builder::new(ComputeCapability::new(9, 0))
        .multiprocessors(132)
        .shared_optin_bytes(227328)
        .name("NVIDIA H100")
        .build();
    let configurations =
        crate::inference::cuda::implementation::tuning_configurations(&device, PtxTier::Sm80)
            .unwrap();
    for entry in [SegdenseEntry::Conv1B32X3, SegdenseEntry::Conv2B32X3] {
        let (boundary, _, math, _, pin, _) = configurations.iter()
            .find(|(_, _, _, _, pin, _)| matches!(pin, ConfigPin::Segdense(pin) if pin.entry() == entry))
            .expect("the unmeasured H100 FP32 default uses three products");
        assert_eq!(*math, CudaMath::Fp32);
        assert!(Policy::approve(*boundary, CudaMath::Fp32, *pin).is_some());
        assert!(Policy::approve(*boundary, CudaMath::Tf32, *pin).is_none());
    }
}

#[test]
fn tuning_requires_library_opt_in() {
    use super::CudaTuneOptions;
    let mut options = CudaTuneOptions::default();
    assert!(!options.include_library);
    assert!(options.validate_library().is_ok());
    options.include_library = true;
    assert_eq!(
        options.validate_library().is_ok(),
        cfg!(feature = "_cuda-libraries")
    );
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn kernel_only_tuning_excludes_library_from_timed_and_untimed_plans() {
    use super::TuneControl;
    let device = Builder::new(ComputeCapability::new(12, 0)).build();
    let boundary = BoundaryId::named("sincnet.conv0.abs_pool");
    let tier = PtxTier::Sm120;
    let catalogue = Catalogue::new(&device, tier).unwrap();
    let tuple = Tuple::new(boundary, 1, CudaMath::Tf32).unwrap();
    assert_eq!(catalogue.choices(tuple), [ApprovedChoice::Library]);
    for kind in catalogue.benchmark_kinds() {
        let control = TuneControl::benchmark(kind, false, &device, tier).unwrap();
        assert_eq!(control.choice(boundary, 1, CudaMath::Tf32), None);
        assert_eq!(control.plan_choice(boundary, 1, CudaMath::Tf32), None);
    }
    let control =
        TuneControl::benchmark(super::BenchKind::CatalogueSlot(0), true, &device, tier).unwrap();
    assert_eq!(
        control.choice(boundary, 1, CudaMath::Tf32),
        Some(ApprovedChoice::Library)
    );
    assert_eq!(
        control.plan_choice(boundary, 1, CudaMath::Tf32),
        Some(ApprovedChoice::Library)
    );
}

#[test]
fn measured_fp16_trunk_extensions_are_approved_only_in_tf32() {
    use super::accuracy::{Approval, Policy};
    use crate::inference::cuda::candidate::{ConfigPin, WideconvPin};

    for name in [
        "resnet.layer2.0.conv1",
        "resnet.layer3.0.conv1",
        "resnet.layer4.0.conv1",
    ] {
        let boundary = BoundaryId::named(name);
        for batch in [1, 4, 8, 16, 32] {
            let pin = WideconvPin::measured_t4_fp16(name, batch, CudaMath::Tf32)
                .expect("the T4 has a measured stride-2 FP16 pin");
            let pin = ConfigPin::Wideconv(pin);
            assert_eq!(
                Policy::approve(boundary, CudaMath::Tf32, pin),
                Some(Approval::Fp16Trunk)
            );
            assert_eq!(Policy::approve(boundary, CudaMath::Fp32, pin), None);
        }
    }

    for name in [
        "resnet.layer3.0.conv1",
        "resnet.layer3.1.conv1",
        "resnet.layer4.0.conv1",
        "resnet.layer4.1.conv1",
    ] {
        let boundary = BoundaryId::named(name);
        for batch in [4, 8, 16, 32] {
            if name == "resnet.layer4.0.conv1" && batch == 4 {
                assert_eq!(
                    WideconvPin::measured_a100_fp16(name, batch, CudaMath::Tf32),
                    None
                );
                continue;
            }

            let pin = WideconvPin::measured_a100_fp16(name, batch, CudaMath::Tf32)
                .expect("the A100 has a measured wide-trunk FP16 pin");
            let pin = ConfigPin::Wideconv(pin);
            assert_eq!(
                Policy::approve(boundary, CudaMath::Tf32, pin),
                Some(Approval::Fp16Trunk)
            );
            assert_eq!(Policy::approve(boundary, CudaMath::Fp32, pin), None);
        }
    }
}

/// Where the batch-32 `seg_1` default runs FP16 products, which the tuner does not
/// approve, the catalogue still offers our TF32 kernel against the library
#[cfg(feature = "cuda-sm80")]
#[test]
fn fp16_embedding_default_leaves_an_approved_tf32_alternative() {
    use crate::inference::cuda::candidate::{ConfigPin, SegdenseEntry};
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(128)
        .shared_optin_bytes(101376)
        .name("NVIDIA GeForce RTX 4090")
        .build();
    let catalogue = Catalogue::new(&device, PtxTier::Sm80).unwrap();
    let boundary = BoundaryId::named("resnet.seg_1");
    let entries = |math| {
        let tuple = Tuple::new(boundary, 32, math).unwrap();
        catalogue
            .choices(tuple)
            .iter()
            .filter_map(|choice| match choice {
                ApprovedChoice::Kernel(config) => match config.pin {
                    ConfigPin::Segdense(pin) => Some((config.family, pin.entry())),
                    _ => None,
                },
                ApprovedChoice::Library => None,
            })
            .collect::<Vec<_>>()
    };

    let tf32 = entries(CudaMath::Tf32);
    // the kernel bench times the first choice and the scalar bench the "fp32" one
    assert_eq!(
        tf32.first(),
        Some(&("tf32", SegdenseEntry::EmbedB32Tf32)),
        "{tf32:?}"
    );
    assert!(
        tf32.contains(&("fp32", SegdenseEntry::EmbedB32)),
        "{tf32:?}"
    );
    assert!(
        tf32.iter()
            .all(|(_, entry)| *entry != SegdenseEntry::EmbedB32F16),
        "{tf32:?}"
    );
    assert!(
        entries(CudaMath::Fp32)
            .iter()
            .all(|(family, _)| *family != "tf32")
    );
}

#[test]
#[cfg(any(feature = "cuda-sm75", feature = "cuda-sm80"))]
fn catalogue_contains_every_measured_stride2_fp16_startup_pin() {
    use crate::inference::cuda::candidate::{ConfigPin, WideconvPin};
    use crate::inference::cuda::implementation::policy::Recipe;
    let mut checked = 0;
    for (cc, sms, name, tier, recipe, minimum_batches) in [
        (
            ComputeCapability::new(7, 5),
            40,
            "Tesla T4",
            PtxTier::Sm75,
            Recipe::TeslaT4,
            [1, 1, 1],
        ),
        (
            ComputeCapability::new(8, 9),
            128,
            "NVIDIA GeForce RTX 4090",
            PtxTier::Sm80,
            Recipe::Rtx4090,
            [1, 4, 4],
        ),
        (
            ComputeCapability::new(8, 0),
            108,
            "NVIDIA A100-SXM4-40GB",
            PtxTier::Sm80,
            Recipe::A100Sxm4,
            [33, 4, 8],
        ),
        (
            ComputeCapability::new(8, 0),
            108,
            "NVIDIA A100-PCIE-40GB",
            PtxTier::Sm80,
            Recipe::A100Pcie,
            [33, 4, 8],
        ),
    ] {
        if (tier == PtxTier::Sm75 && !cfg!(feature = "cuda-sm75"))
            || (tier == PtxTier::Sm80 && !cfg!(feature = "cuda-sm80"))
        {
            continue;
        }
        let device = Builder::new(cc).multiprocessors(sms).name(name).build();
        let catalogue = Catalogue::new(&device, tier).unwrap();
        for (name, minimum) in [
            "resnet.layer2.0.conv1",
            "resnet.layer3.0.conv1",
            "resnet.layer4.0.conv1",
        ]
        .into_iter()
        .zip(minimum_batches)
        {
            let boundary = BoundaryId::named(name);
            for batch in boundary.batches().iter().filter(|batch| *batch >= minimum) {
                let pin = recipe.fp16_pin(boundary, batch, CudaMath::Tf32).unwrap();
                assert!(pin.is_fp16(), "{recipe:?} {name} b{batch}");
                let tuple = Tuple::new(boundary, batch, CudaMath::Tf32).unwrap();
                assert!(
                    catalogue.choices(tuple).iter().any(|choice| {
                        matches!(choice, ApprovedChoice::Kernel(config) if config.pin() == pin)
                    }),
                    "{recipe:?} {name} b{batch}: {pin:?}"
                );
                assert!(
                    catalogue
                        .choices(Tuple::new(boundary, batch, CudaMath::Fp32).unwrap())
                        .iter()
                        .all(|choice| !choice.is_fp16())
                );
                assert_eq!(WideconvPin::fp16_wide(name, batch, CudaMath::Tf32), None);
                assert!(matches!(pin, ConfigPin::Wideconv(_)));
                checked += 1;
            }
        }
    }
    assert!(checked > 0);
}
