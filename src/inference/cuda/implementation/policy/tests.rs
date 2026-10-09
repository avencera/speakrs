//! Scope checks do not need a GPU or optional libraries

use super::{DeviceDefault, Recipe, RecipeMode, Source};
use crate::inference::cuda::candidate::Fp16Policy;
use crate::inference::cuda::device::test_support::Builder;
use crate::inference::cuda::implementation::BoundaryId;
use crate::inference::cuda::{ComputeCapability, CudaMath, KernelModule, PtxTier};

#[test]
fn precedence_is_explicit() {
    assert!(Source::TuneFile < Source::Recipe);
    assert!(Source::Recipe < Source::Default);
    assert!(Source::Default < Source::Library);
}

#[test]
fn ada_recipe_requires_exact_point_and_measured_tuples() {
    let device = Builder::new(ComputeCapability::new(8, 9))
        .multiprocessors(34)
        .name("NVIDIA GeForce RTX 4060 Ti")
        .build();
    let sinc = BoundaryId::named("sincnet.conv0.abs_pool");
    let select = |boundary, batch, math, device: &_| {
        Recipe::select(boundary, batch, math, device, RecipeMode::Disabled)
    };
    for batch in [1, 32] {
        assert_eq!(
            select(sinc, batch, CudaMath::Fp32, &device),
            Some(Recipe::Rtx4060TiSinc)
        );
    }
    assert_eq!(select(sinc, 7, CudaMath::Fp32, &device), None);
    assert_eq!(select(sinc, 32, CudaMath::Tf32, &device), None);
    assert_eq!(
        select(
            BoundaryId::named("sincnet.conv1"),
            32,
            CudaMath::Fp32,
            &device
        ),
        None
    );
    for wrong in [
        Builder::new(ComputeCapability::new(8, 9))
            .multiprocessors(33)
            .name(device.name())
            .build(),
        Builder::new(ComputeCapability::new(8, 9))
            .multiprocessors(34)
            .name("NVIDIA GeForce RTX 4070")
            .build(),
        Builder::new(ComputeCapability::new(8, 6))
            .multiprocessors(34)
            .name(device.name())
            .build(),
    ] {
        assert_eq!(select(sinc, 32, CudaMath::Fp32, &wrong), None);
    }
}

#[test]
fn a100_recipe_is_whole_pipeline_not_a_capability_or_fp32_embedding_claim() {
    let mode = RecipeMode::new(CudaMath::Fp32, CudaMath::Tf32);
    for recipe in [Recipe::A100Pcie, Recipe::A100Sxm4] {
        let name = match recipe {
            Recipe::A100Pcie => "NVIDIA A100-PCIE-40GB",
            Recipe::A100Sxm4 => "NVIDIA A100-SXM4-40GB",
            _ => unreachable!(),
        };
        let device = Builder::new(ComputeCapability::new(8, 0))
            .multiprocessors(108)
            .name(name)
            .build();
        for (boundary, math) in [
            ("resnet.layer1.0.conv1", CudaMath::Tf32),
            ("resnet.layer3.1.conv1", CudaMath::Tf32),
            ("sincnet.conv0.abs_pool", CudaMath::Fp32),
            ("lstm.stack", CudaMath::Fp32),
        ] {
            let boundary = BoundaryId::named(boundary);
            for batch in [1, 32] {
                assert_eq!(
                    Recipe::select(boundary, batch, math, &device, mode),
                    Some(recipe)
                );
                assert_eq!(
                    Recipe::select(boundary, batch, math, &device, RecipeMode::Disabled),
                    None
                );
            }
        }
        assert_eq!(
            Recipe::select(
                BoundaryId::named("resnet.layer1.0.conv1"),
                32,
                CudaMath::Fp32,
                &device,
                mode
            ),
            None
        );
        let wrong = Builder::new(ComputeCapability::new(8, 0))
            .multiprocessors(107)
            .name(name)
            .build();
        assert_eq!(
            Recipe::select(
                BoundaryId::named("lstm.stack"),
                32,
                CudaMath::Fp32,
                &wrong,
                mode
            ),
            None
        );
    }
    assert_eq!(
        RecipeMode::new(CudaMath::Fp32, CudaMath::Fp32),
        RecipeMode::Disabled
    );
    assert_eq!(
        RecipeMode::new(CudaMath::Tf32, CudaMath::Tf32),
        RecipeMode::Disabled
    );
}

#[test]
fn class_default_only_covers_early_tf32_trunk_on_ampere_and_newer() {
    for capability in [
        ComputeCapability::new(8, 0),
        ComputeCapability::new(8, 6),
        ComputeCapability::new(8, 9),
        ComputeCapability::new(9, 0),
        ComputeCapability::new(12, 0),
    ] {
        let device = Builder::new(capability).build();
        for batch in [1, 32] {
            assert_eq!(
                DeviceDefault::select(
                    KernelModule::Resnet,
                    BoundaryId::named("resnet.layer1.0.conv1"),
                    batch,
                    CudaMath::Tf32,
                    &device,
                    PtxTier::Sm80,
                    Fp16Policy::Allowed
                ),
                Some(DeviceDefault::AmpereTf32EarlyTrunk)
            );
        }
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Wideconv,
                BoundaryId::named("resnet.layer1.0.conv1"),
                32,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm80,
                Fp16Policy::Allowed
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                BoundaryId::named("resnet.layer1.0.conv1"),
                32,
                CudaMath::Fp32,
                &device,
                PtxTier::Sm80,
                Fp16Policy::Allowed
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                BoundaryId::named("resnet.layer1.0.conv1"),
                7,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm80,
                Fp16Policy::Allowed
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                BoundaryId::named("resnet.layer1.0.conv1"),
                32,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm75,
                Fp16Policy::Allowed
            ),
            None
        );
    }
    let turing = Builder::new(ComputeCapability::new(7, 5)).build();
    assert_eq!(
        DeviceDefault::select(
            KernelModule::Resnet,
            BoundaryId::named("resnet.layer1.0.conv1"),
            32,
            CudaMath::Tf32,
            &turing,
            PtxTier::Sm80,
            Fp16Policy::Allowed
        ),
        None
    );
}

#[test]
fn rtx_whole_recipes_cover_only_measured_devices_batches_and_precision() {
    for (cc, sms, name, recipe) in [
        (
            ComputeCapability::new(8, 9),
            34,
            "NVIDIA GeForce RTX 4060 Ti",
            Recipe::Rtx4060Ti,
        ),
        (
            ComputeCapability::new(12, 0),
            36,
            "NVIDIA GeForce RTX 5060 Ti",
            Recipe::Rtx5060Ti,
        ),
    ] {
        let device = Builder::new(cc).multiprocessors(sms).name(name).build();
        let boundary = BoundaryId::named("resnet.layer3.1.conv1");
        for batch in [1, 4, 8, 16, 32] {
            assert_eq!(
                Recipe::select(
                    boundary,
                    batch,
                    CudaMath::Tf32,
                    &device,
                    RecipeMode::Fp32SegmentationTf32Embedding
                ),
                Some(recipe)
            );
            assert_eq!(
                Recipe::select(
                    boundary,
                    batch,
                    CudaMath::Fp32,
                    &device,
                    RecipeMode::Fp32SegmentationTf32Embedding
                ),
                None
            );
            assert_eq!(
                Recipe::select(
                    boundary,
                    batch,
                    CudaMath::Tf32,
                    &device,
                    RecipeMode::Disabled
                ),
                None
            );
        }
        let wrong = Builder::new(cc).multiprocessors(sms + 1).name(name).build();
        assert_eq!(
            Recipe::select(
                boundary,
                32,
                CudaMath::Tf32,
                &wrong,
                RecipeMode::Fp32SegmentationTf32Embedding
            ),
            None
        );
        assert_eq!(
            Recipe::select(
                boundary,
                7,
                CudaMath::Tf32,
                &device,
                RecipeMode::Fp32SegmentationTf32Embedding
            ),
            None
        );
    }
}

#[test]
fn fp16_recipes_exclude_fp32_strided_layers_and_unmeasured_devices() {
    use crate::inference::cuda::candidate::{
        ConfigPin, WideconvAlgorithm, WideconvFp16Tiles, WideconvPin,
    };
    for recipe in [Recipe::TeslaT4, Recipe::Rtx4060Ti] {
        for batch in [1, 4, 8, 16, 32] {
            for name in [
                "resnet.layer1.0.conv2",
                "resnet.layer2.0.conv2",
                "resnet.layer3.1.conv1",
                "resnet.layer4.2.conv2",
            ] {
                let boundary = BoundaryId::named(name);
                assert_eq!(
                    recipe.fixed_pin(boundary, batch, CudaMath::Fp32, Fp16Policy::Allowed),
                    None
                );
                let pin = recipe.fixed_pin(boundary, batch, CudaMath::Tf32, Fp16Policy::Allowed);
                let expected = recipe == Recipe::TeslaT4
                    || batch >= 8
                    || name.starts_with("resnet.layer1.")
                    || name.starts_with("resnet.layer2.");
                assert_eq!(pin.is_some(), expected, "{recipe:?} {name} b{batch}");
                if let Some(ConfigPin::Wideconv(WideconvPin::Configured(config))) = pin {
                    let narrow = recipe == Recipe::TeslaT4
                        && batch == 1
                        && (name.starts_with("resnet.layer3.")
                            || name.starts_with("resnet.layer4."));
                    assert_eq!(
                        config.algorithm,
                        WideconvAlgorithm::Fp16(if narrow {
                            WideconvFp16Tiles::Narrow
                        } else {
                            WideconvFp16Tiles::Wide
                        })
                    );
                }
                assert_eq!(
                    Recipe::Rtx5060Ti.fixed_pin(
                        boundary,
                        batch,
                        CudaMath::Tf32,
                        Fp16Policy::Allowed
                    ),
                    None
                );
            }
            for name in ["resnet.conv1", "resnet.layer4.0.shortcut.0"] {
                assert_eq!(
                    recipe.fp16_pin(BoundaryId::named(name), batch, CudaMath::Tf32),
                    None
                );
            }
            // the 4060 Ti recipe keeps the stride-2 layers off FP16 tiles
            for name in [
                "resnet.layer2.0.conv1",
                "resnet.layer3.0.conv1",
                "resnet.layer4.0.conv1",
            ] {
                let boundary = BoundaryId::named(name);
                assert_eq!(recipe.fp16_pin(boundary, batch, CudaMath::Fp32), None);
                let pin = recipe.fp16_pin(boundary, batch, CudaMath::Tf32);
                if recipe == Recipe::Rtx4060Ti {
                    assert_eq!(pin, None, "{name} b{batch}");
                    continue;
                }
                let Some(ConfigPin::Wideconv(WideconvPin::Configured(config))) = pin else {
                    panic!("{name} b{batch}: {pin:?}")
                };
                // narrow tiles where wide ones give fewer than two waves on 40 SMs
                let narrow = batch == 1 && name != "resnet.layer2.0.conv1";
                assert_eq!(
                    config.algorithm,
                    WideconvAlgorithm::Fp16(if narrow {
                        WideconvFp16Tiles::Narrow
                    } else {
                        WideconvFp16Tiles::Wide
                    }),
                    "{name} b{batch}"
                );
            }
        }
    }
    for name in ["NVIDIA GeForce RTX 4070", "unmeasured Ada GPU"] {
        let device = Builder::new(ComputeCapability::new(8, 9))
            .multiprocessors(34)
            .name(name)
            .build();
        assert_eq!(Recipe::fp16_device(&device, PtxTier::Sm80), None);
    }
}

#[test]
fn a100_fp16_covers_only_wide_tf32_trunk_from_batch_4() {
    let sxm4 = Builder::new(ComputeCapability::new(8, 0))
        .multiprocessors(108)
        .name("NVIDIA A100-SXM4-40GB")
        .build();
    let pcie = Builder::new(ComputeCapability::new(8, 0))
        .multiprocessors(108)
        .name("NVIDIA A100-PCIE-40GB")
        .build();
    assert_eq!(
        Recipe::fp16_device(&sxm4, PtxTier::Sm80),
        Some(Recipe::A100Sxm4)
    );
    assert_eq!(Recipe::fp16_device(&sxm4, PtxTier::Sm75), None);
    assert_eq!(
        Recipe::fp16_device(&pcie, PtxTier::Sm80),
        Some(Recipe::A100Pcie)
    );

    for recipe in [Recipe::A100Sxm4, Recipe::A100Pcie] {
        a100_fp16_pins(recipe);
    }
}

fn a100_fp16_pins(recipe: Recipe) {
    use crate::inference::cuda::candidate::{
        ConfigPin, WideconvAlgorithm, WideconvFp16Tiles, WideconvPin,
    };
    for name in [
        "resnet.layer3.0.conv1",
        "resnet.layer3.1.conv1",
        "resnet.layer3.5.conv2",
        "resnet.layer4.0.conv1",
        "resnet.layer4.2.conv2",
    ] {
        let boundary = BoundaryId::named(name);
        let pin = recipe.fixed_pin(boundary, 32, CudaMath::Tf32, Fp16Policy::Allowed);
        let Some(ConfigPin::Wideconv(WideconvPin::Configured(config))) = pin else {
            panic!("{name}: {pin:?}")
        };
        assert_eq!(
            config.algorithm,
            WideconvAlgorithm::Fp16(WideconvFp16Tiles::Wide),
            "{name}"
        );
        assert_eq!(recipe.fp16_pin(boundary, 1, CudaMath::Tf32), None);
        let stride2 = name.ends_with(".0.conv1");
        if name == "resnet.layer4.0.conv1" {
            assert_eq!(recipe.fp16_pin(boundary, 4, CudaMath::Tf32), None);
        }
        for batch in [4, 8, 16] {
            if name == "resnet.layer4.0.conv1" && batch == 4 {
                continue;
            }
            let mid = recipe.fp16_pin(boundary, batch, CudaMath::Tf32);
            let Some(ConfigPin::Wideconv(WideconvPin::Configured(config))) = mid else {
                panic!("{name} b{batch}: {mid:?}")
            };
            let tiles = if stride2 {
                WideconvFp16Tiles::Wide
            } else {
                WideconvFp16Tiles::Narrow
            };
            assert_eq!(
                config.algorithm,
                WideconvAlgorithm::Fp16(tiles),
                "{name} b{batch}"
            );
        }
        assert_eq!(recipe.fp16_pin(boundary, 32, CudaMath::Fp32), None);
        assert_ne!(
            recipe.fixed_pin(boundary, 32, CudaMath::Tf32, Fp16Policy::Excluded),
            pin
        );
    }
    for name in [
        "resnet.conv1",
        "resnet.layer1.0.conv1",
        "resnet.layer2.0.conv1",
        "resnet.layer2.1.conv2",
        "resnet.layer3.0.shortcut.0",
        "resnet.layer4.0.shortcut.0",
    ] {
        assert_eq!(
            recipe.fp16_pin(BoundaryId::named(name), 32, CudaMath::Tf32),
            None,
            "{name}"
        );
    }
}

#[test]
fn t4_recipe_requires_its_exact_point_precision_and_batch_classes() {
    let device = Builder::new(ComputeCapability::new(7, 5))
        .multiprocessors(40)
        .name("Tesla T4")
        .build();
    let boundary = BoundaryId::named("resnet.layer3.1.conv1");
    let mode = RecipeMode::Fp32SegmentationTf32Embedding;
    for batch in [1, 4, 8, 16, 32] {
        assert_eq!(
            Recipe::select(boundary, batch, CudaMath::Tf32, &device, mode),
            Some(Recipe::TeslaT4)
        );
        assert_eq!(
            Recipe::select(boundary, batch, CudaMath::Fp32, &device, mode),
            None
        );
    }
    for wrong in [
        Builder::new(ComputeCapability::new(7, 5))
            .multiprocessors(39)
            .name("Tesla T4")
            .build(),
        Builder::new(ComputeCapability::new(7, 5))
            .multiprocessors(40)
            .name("unmeasured Turing GPU")
            .build(),
        Builder::new(ComputeCapability::new(8, 0))
            .multiprocessors(40)
            .name("Tesla T4")
            .build(),
    ] {
        assert_eq!(
            Recipe::select(boundary, 32, CudaMath::Tf32, &wrong, mode),
            None
        );
    }
    assert_eq!(
        Recipe::select(boundary, 7, CudaMath::Tf32, &device, mode),
        None
    );
    assert_eq!(
        Recipe::select(boundary, 32, CudaMath::Tf32, &device, RecipeMode::Disabled),
        None
    );
    assert_eq!(
        Recipe::TeslaT4.allows_tier_limit(PtxTier::Sm75),
        cfg!(feature = "cuda-sm75")
    );
}

#[test]
fn turing_default_changes_only_tf32_3x3_trunk_layers_after_the_stem() {
    let device = Builder::new(ComputeCapability::new(7, 5))
        .name("unmeasured Turing GPU")
        .build();
    for name in [
        "resnet.layer1.0.conv1",
        "resnet.layer2.0.conv1",
        "resnet.layer2.0.conv2",
        "resnet.layer3.0.conv1",
        "resnet.layer3.1.conv1",
        "resnet.layer4.0.conv1",
        "resnet.layer4.2.conv2",
    ] {
        for batch in [1, 4, 8, 16, 32] {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                assert_eq!(
                    DeviceDefault::select(
                        KernelModule::Wideconv,
                        BoundaryId::named(name),
                        batch,
                        math,
                        &device,
                        PtxTier::Sm75,
                        Fp16Policy::Allowed
                    ),
                    (math == CudaMath::Tf32).then_some(DeviceDefault::TuringFp16Trunk)
                );
            }
        }
    }
    for name in [
        "resnet.conv1",
        "resnet.layer2.0.shortcut.0",
        "resnet.layer4.0.shortcut.0",
    ] {
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Wideconv,
                BoundaryId::named(name),
                32,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm75,
                Fp16Policy::Allowed
            ),
            None
        );
    }
}
