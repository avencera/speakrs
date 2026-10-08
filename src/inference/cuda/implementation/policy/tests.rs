//! Scope checks do not need a GPU or optional libraries

use super::{DeviceDefault, Recipe, RecipeMode, Source};
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
                    batch,
                    CudaMath::Tf32,
                    &device,
                    PtxTier::Sm80
                ),
                Some(DeviceDefault::AmpereTf32EarlyTrunk)
            );
        }
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Wideconv,
                32,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm80
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                32,
                CudaMath::Fp32,
                &device,
                PtxTier::Sm80
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                7,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm80
            ),
            None
        );
        assert_eq!(
            DeviceDefault::select(
                KernelModule::Resnet,
                32,
                CudaMath::Tf32,
                &device,
                PtxTier::Sm75
            ),
            None
        );
    }
    let turing = Builder::new(ComputeCapability::new(7, 5)).build();
    assert_eq!(
        DeviceDefault::select(
            KernelModule::Resnet,
            32,
            CudaMath::Tf32,
            &turing,
            PtxTier::Sm80
        ),
        None
    );
}

#[test]
#[cfg(feature = "_cuda-libraries")]
fn retained_a100_pins_do_not_follow_staged_winograd() {
    use crate::inference::cuda::candidate::{
        ConfigPin, WideconvAlgorithm, WideconvPartition, WideconvPin, WideconvProducts,
        WideconvSplitCells, WideconvTensorKernel,
    };
    for recipe in [Recipe::A100Pcie, Recipe::A100Sxm4] {
        for batch in [1, 32] {
            let Some(ConfigPin::Wideconv(WideconvPin::Configured(c128))) = recipe.fixed_pin(
                BoundaryId::named("resnet.layer3.1.conv1"),
                batch,
                CudaMath::Tf32,
            ) else {
                panic!("retained C128 pin")
            };
            assert_eq!(
                c128.algorithm,
                WideconvAlgorithm::Winograd(WideconvProducts::Tf32x1)
            );
            assert_eq!(c128.partition, WideconvPartition::Whole);
            assert_eq!(c128.split_cells, WideconvSplitCells::All);
        }
        let Some(ConfigPin::Wideconv(WideconvPin::Configured(c256))) = recipe.fixed_pin(
            BoundaryId::named("resnet.layer4.1.conv1"),
            1,
            CudaMath::Tf32,
        ) else {
            panic!("retained C256 b1 pin")
        };
        assert_eq!(
            c256.algorithm,
            WideconvAlgorithm::Winograd(WideconvProducts::Tf32x3)
        );
        assert_eq!(c256.partition, WideconvPartition::Two);
        let Some(ConfigPin::Wideconv(WideconvPin::Configured(c256))) = recipe.fixed_pin(
            BoundaryId::named("resnet.layer4.1.conv1"),
            32,
            CudaMath::Tf32,
        ) else {
            panic!("retained C256 b32 pin")
        };
        assert_eq!(
            c256.algorithm,
            WideconvAlgorithm::TensorCore(WideconvTensorKernel::Tf32)
        );
        assert_eq!(c256.partition, WideconvPartition::Whole);
    }
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
fn ada_scalar_exceptions_use_only_the_fourteen_measured_tuples() {
    use crate::inference::cuda::candidate::{ConfigPin, ConvKernel, ConvPin};
    let scalar = Some(ConfigPin::Conv(ConvPin::Kernel(ConvKernel::C32)));
    for batch in [1, 4, 8, 16, 32] {
        for name in ["resnet.layer1.0.conv2", "resnet.layer1.1.conv2"] {
            let boundary = BoundaryId::named(name);
            assert_eq!(
                Recipe::Rtx4060Ti.fixed_pin(boundary, batch, CudaMath::Tf32),
                scalar
            );
            assert_eq!(
                Recipe::Rtx5060Ti.fixed_pin(boundary, batch, CudaMath::Tf32),
                None
            );
        }
    }
    let boundary = BoundaryId::named("resnet.layer1.2.conv2");
    assert_eq!(
        Recipe::Rtx4060Ti.fixed_pin(boundary, 1, CudaMath::Tf32),
        None
    );
    for batch in [4, 8, 16, 32] {
        assert_eq!(
            Recipe::Rtx4060Ti.fixed_pin(boundary, batch, CudaMath::Tf32),
            scalar
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
