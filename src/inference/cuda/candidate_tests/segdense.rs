use super::super::PlanError;

use super::super::segdense::{
    Choice, Config, HALF_SHARED, Hardware, Kind, Reduction, SegdensePin, Site, f16_bits,
    half_scale, pack_half_columns, selected_rows,
};
use crate::inference::cuda::{ComputeCapability, CudaMath, PtxTier};

const fn device(major: u32, minor: u32, multiprocessors: u32, shared_kib: u32) -> Hardware {
    Hardware {
        capability: ComputeCapability::new(major, minor),
        multiprocessors,
        shared_optin: shared_kib * 1024,
    }
}

const A100: Hardware = device(8, 0, 108, 163);
const H100: Hardware = device(9, 0, 132, 227);
const RTX_4060_TI: Hardware = device(8, 9, 34, 99);
const RTX_5060_TI: Hardware = device(12, 0, 36, 99);
const RTX_3090: Hardware = device(8, 6, 82, 99);
const T4: Hardware = device(7, 5, 40, 64);

const SITES: [Site; 6] = [
    Site::Conv1,
    Site::Conv2,
    Site::Linear0,
    Site::Linear1,
    Site::Linear2,
    Site::Embedding,
];

fn kernel(site: Site, batch: usize, math: CudaMath, tier: PtxTier, hardware: Hardware) -> Config {
    config(site, batch, math, tier, hardware).expect("production batch")
}

fn config(
    site: Site,
    batch: usize,
    math: CudaMath,
    tier: PtxTier,
    hardware: Hardware,
) -> Option<Config> {
    site.choice(batch, math, tier, hardware).map(Choice::config)
}

#[test]
fn every_model_batch_has_a_kernel_and_no_other_batch_does() {
    for hardware in [A100, H100, RTX_4060_TI, RTX_5060_TI, RTX_3090, T4] {
        for site in SITES {
            for math in [CudaMath::Fp32, CudaMath::Tf32] {
                for tier in [PtxTier::Sm75, PtxTier::Sm80] {
                    for batch in [0, 2, 7, 31, 33, 64] {
                        assert_eq!(site.choice(batch, math, tier, hardware), None);
                    }
                    for batch in [1, 32] {
                        let choice = site.choice(batch, math, tier, hardware).unwrap();
                        assert_eq!((choice.entry.site(), choice.entry.batch()), (site, batch));
                        assert_eq!(choice.entry.is_split(), choice.splits > 0);
                        let config = choice.config();
                        assert!(config.kernel.contains(&format!("_b{batch}")));
                        assert!(config.shared <= hardware.shared_optin.max(48 * 1024));
                    }
                }
            }
        }
    }
}

#[test]
fn pins_refuse_other_boundaries_and_lower_modules() {
    let pin = SegdensePin::select(
        Site::Linear0,
        32,
        CudaMath::Tf32,
        PtxTier::Sm80,
        &crate::inference::cuda::device::test_support::Builder::new(ComputeCapability::new(12, 0))
            .multiprocessors(36)
            .build(),
    )
    .unwrap();
    assert!(
        pin.check(Site::Linear0, 32, CudaMath::Tf32, PtxTier::Sm80)
            .is_ok()
    );
    assert!(
        pin.check(Site::Linear0, 32, CudaMath::Tf32, PtxTier::Sm120)
            .is_ok()
    );
    for (site, batch, math) in [
        (Site::Linear1, 32, CudaMath::Tf32),
        (Site::Linear0, 1, CudaMath::Tf32),
        (Site::Linear0, 32, CudaMath::Fp32),
    ] {
        assert!(matches!(
            pin.check(site, batch, math, PtxTier::Sm80),
            Err(PlanError::Geometry(_))
        ));
    }
    assert!(matches!(
        pin.check(Site::Linear0, 32, CudaMath::Tf32, PtxTier::Sm75),
        Err(PlanError::DeviceUnsupported { .. })
    ));
}

#[test]
fn tensor_convolutions_only_on_tensor_heavy_parts() {
    let convs = [
        (
            Site::Conv1,
            "spk_segdense_conv1_b32_tc",
            "spk_segdense_conv1_b32_x3",
            "spk_segdense_conv1_b32",
        ),
        (
            Site::Conv2,
            "spk_segdense_conv2_b32_tc",
            "spk_segdense_conv2_b32_x3",
            "spk_segdense_conv2_b32",
        ),
    ];
    for (site, tf32, fp32, simt) in convs {
        for hardware in [A100, H100] {
            let tc = kernel(site, 32, CudaMath::Tf32, PtxTier::Sm80, hardware);
            let x3 = kernel(site, 32, CudaMath::Fp32, PtxTier::Sm80, hardware);
            assert_eq!((tc.kernel, x3.kernel), (tf32, fp32));
            assert!(x3.shared <= hardware.shared_optin);
        }

        for hardware in [RTX_4060_TI, RTX_5060_TI, RTX_3090] {
            let tf32_mode = kernel(site, 32, CudaMath::Tf32, PtxTier::Sm80, hardware);
            let fp32_mode = kernel(site, 32, CudaMath::Fp32, PtxTier::Sm80, hardware);
            assert_eq!((tf32_mode.kernel, fp32_mode.kernel), (simt, simt));
        }
    }

    // batch 1 keeps the SIMT convolution everywhere
    assert_eq!(
        kernel(Site::Conv1, 1, CudaMath::Tf32, PtxTier::Sm80, A100).kernel,
        "spk_segdense_conv1_b1"
    );
}

#[test]
fn sm75_tier_never_selects_tensor_kernels() {
    for hardware in [A100, RTX_4060_TI, T4] {
        for site in SITES {
            for batch in [1, 32] {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    let config = kernel(site, batch, math, PtxTier::Sm75, hardware);
                    // the one TF32 kernel picked on every tier is plain SIMT in sm75
                    let plain_tf32 = (site, batch, math) == (Site::Linear1, 1, CudaMath::Tf32);
                    assert!(
                        (plain_tf32 || !config.kernel.contains("_tf32"))
                            && !config.kernel.ends_with("_tc")
                            && !config.kernel.ends_with("_x3")
                            && !config.kernel.ends_with("_f16"),
                        "{site:?} {batch} {math:?}: {}",
                        config.kernel
                    );
                }
            }
        }
    }
}

#[test]
fn embedding_splits_fill_one_wave() {
    let splits = |math, hardware| match kernel(Site::Embedding, 32, math, PtxTier::Sm80, hardware) {
        Config {
            kernel,
            kind: Kind::Split { splits, .. },
            ..
        } => (kernel, splits),
        other => panic!("not split-K: {other:?}"),
    };

    // FP16 products: two 96 x 128 tiles, one resident block per SM
    assert_eq!(
        splits(CudaMath::Tf32, RTX_4060_TI),
        ("spk_segdense_embed_b32_f16", 17)
    );
    assert_eq!(
        splits(CudaMath::Tf32, RTX_5060_TI),
        ("spk_segdense_embed_b32_f16", 18)
    );
    // under 32 SMs a slice would exceed the FP16 kernel's shared slice, and
    // without the shared memory opt-in it cannot run: TF32 products with two
    // resident blocks per SM, or one on consumer Blackwell
    let small = |hardware: Hardware| Hardware {
        multiprocessors: 30,
        ..hardware
    };
    assert_eq!(
        splits(CudaMath::Tf32, small(RTX_4060_TI)),
        ("spk_segdense_embed_b32_tf32", 30)
    );
    assert_eq!(
        splits(CudaMath::Tf32, small(RTX_5060_TI)),
        ("spk_segdense_embed_b32_tf32_k2", 15)
    );
    let no_optin = Hardware {
        shared_optin: HALF_SHARED - 1,
        ..RTX_4060_TI
    };
    assert_eq!(
        splits(CudaMath::Tf32, no_optin),
        ("spk_segdense_embed_b32_tf32", 34)
    );
    // four 96 x 64 tiles on tensor-heavy parts
    assert_eq!(
        splits(CudaMath::Tf32, A100),
        ("spk_segdense_embed_b32_tf32_e64", 54)
    );
    // FP32: two resident SIMT blocks per SM, or 3xTF32 on tensor-heavy parts
    assert_eq!(
        splits(CudaMath::Fp32, RTX_4060_TI),
        ("spk_segdense_embed_b32", 34)
    );
    // four 96 x 64 tiles for 3xTF32
    assert_eq!(
        splits(CudaMath::Fp32, A100),
        ("spk_segdense_embed_b32_x3", 54)
    );
    // a one-SM partition still gets one slice, and a 132-SM part no more than
    // the reduction adds
    assert_eq!(splits(CudaMath::Fp32, device(8, 9, 1, 99)).1, 1);
    assert_eq!(splits(CudaMath::Fp32, device(8, 9, 132, 99)).1, 128);
    assert_eq!(splits(CudaMath::Tf32, device(8, 9, 132, 99)).1, 66);
}

#[test]
fn reductions_cover_every_output_once() {
    for splits in 1..=128 {
        let shape = Reduction::shape(splits);
        let kernel = match splits {
            18 | 34 | 36 => format!("spk_segdense_reduce_e{splits}"),
            ..=40 => "spk_segdense_reduce_embed_flat".to_owned(),
            _ => "spk_segdense_reduce_embed".to_owned(),
        };
        assert_eq!(shape.kernel, kernel);
        assert_eq!(shape.splits, splits);
        // the flat and fixed kernels have a thread per four outputs, the warp
        // kernel a block per 128
        let outputs = match shape.kernel {
            "spk_segdense_reduce_embed" => shape.grid * 128,
            _ => shape.grid * shape.block * 4,
        };
        assert_eq!(outputs, 96 * 256);
    }
}

#[test]
fn selected_rows_have_the_largest_weight_energy() {
    // rows of two columns with energies 1, 25, 0, 25 and 4
    let weight = [1.0, 0.0, 3.0, -4.0, 0.0, 0.0, 0.0, 5.0, 2.0, 0.0];
    assert_eq!(selected_rows(&weight, 2, 3), [1, 3, 4]);
    // the tie between rows 1 and 3 goes to the lower index
    assert_eq!(selected_rows(&weight, 2, 1), [1]);
    assert_eq!(selected_rows(&weight, 2, 5), [0, 1, 2, 3, 4]);
}

#[test]
fn f16_rounding_matches_binary16() {
    let cases: [(f32, u16); 16] = [
        (1.0, 0x3c00),
        (-2.0, 0xc000),
        (-0.0, 0x8000),
        (65504.0, 0x7bff),
        // the last value below the halfway point to 65536 stays finite
        (65519.996, 0x7bff),
        (65520.0, 0x7c00),
        // ties go to the even significand
        (1.0 + 2f32.powi(-11), 0x3c00),
        (1.0 + 3.0 * 2f32.powi(-11), 0x3c02),
        (2f32.powi(-14), 0x0400),
        // the largest subnormal rounds up into the normal range
        (2f32.powi(-14) * (1.0 - 2f32.powi(-11)), 0x0400),
        (2f32.powi(-24), 0x0001),
        (2f32.powi(-25), 0x0000),
        (1.5 * 2f32.powi(-25), 0x0001),
        (f32::MIN_POSITIVE / 2.0, 0x0000),
        (f32::INFINITY, 0x7c00),
        (f32::NAN, 0x7e00),
    ];
    for (value, expected) in cases {
        assert_eq!(f16_bits(value), expected, "{value:e}");
    }
}

#[test]
fn half_columns_scale_into_the_f16_range() {
    // two columns of four reduction terms: magnitudes up to 3 and up to 2^-20
    let tiny = 2f32.powi(-20);
    let weight = [1.0, -3.0, 0.5, 2.0, tiny, -tiny / 2.0, 0.0, tiny / 4.0];
    let packed = pack_half_columns(&weight, 2, 4);
    assert_eq!(packed.len(), 4 / 2 * 2 + 2);

    // column 0 scales by 2^13 (3 * 2^13 lies in [2^14, 2^15)), column 1 by 2^34
    assert_eq!(packed[4..], [2f32.powi(-13), 2f32.powi(-34)]);
    let words: Vec<u32> = packed[..4].iter().map(|word| word.to_bits()).collect();
    let pair = |low: f32, high: f32| u32::from(f16_bits(low)) | (u32::from(f16_bits(high)) << 16);
    assert_eq!(
        words,
        [
            pair(8192.0, -24576.0),
            pair(16384.0, -8192.0),
            pair(4096.0, 16384.0),
            pair(0.0, 4096.0),
        ]
    );

    // a zero column and an infinite one keep normal scales
    assert_eq!(half_scale(0.0), (2f32.powi(126), 2f32.powi(-126)));
    assert_eq!(half_scale(f32::INFINITY), (2f32.powi(-113), 2f32.powi(113)));
}
