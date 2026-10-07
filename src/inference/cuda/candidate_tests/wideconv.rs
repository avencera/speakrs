use super::super::wideconv::{
    Algorithm, Layout, Partition, SplitCells, TensorKernel, WinogradProducts,
};
use crate::inference::cuda::geometry::Conv2d;
use crate::inference::cuda::{CudaError, CudaMath};

fn stem(batch: usize, input: [usize; 2]) -> Conv2d {
    Conv2d {
        batch,
        input,
        in_channels: 1,
        out_channels: 32,
        kernel: [3, 3],
        padding: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        math: CudaMath::Fp32,
    }
}

#[test]
fn rejects_size_overflow_before_output_arithmetic() {
    let result = Layout::new(
        stem(1, [usize::MAX, 1]),
        Partition::Whole,
        SplitCells::All,
        Algorithm::Spatial,
    );
    assert!(matches!(result, Err(CudaError::DimensionOverflow { .. })));
}

#[test]
fn rejects_grid_overflow_before_allocation() {
    assert!(matches!(
        Layout::new(
            stem(65536, [1, 1]),
            Partition::Whole,
            SplitCells::All,
            Algorithm::Spatial
        ),
        Err(CudaError::Unsupported { .. })
    ));
    assert!(
        Layout::new(
            stem(65535, [1, 1]),
            Partition::Whole,
            SplitCells::All,
            Algorithm::Spatial
        )
        .is_ok()
    );
}

#[test]
fn rejects_inputs_other_than_compiled_sizes() {
    let conv = |in_channels, out_channels, input, kernel: usize, stride| Conv2d {
        batch: 32,
        input,
        in_channels,
        out_channels,
        kernel: [kernel; 2],
        padding: [kernel / 2; 2],
        stride: [stride; 2],
        dilation: [1, 1],
        math: CudaMath::Tf32,
    };
    let cases = [
        (
            conv(128, 128, [20, 250], 3, 1),
            Algorithm::TensorCore(TensorKernel::Tf32),
        ),
        (
            conv(64, 128, [40, 499], 3, 2),
            Algorithm::TensorCore(TensorKernel::Tf32),
        ),
        (
            conv(128, 256, [20, 250], 3, 2),
            Algorithm::TensorCore(TensorKernel::Tf32x3),
        ),
        (conv(128, 256, [20, 250], 1, 2), Algorithm::Spatial),
        (conv(32, 64, [80, 998], 1, 2), Algorithm::Spatial),
    ];
    for (mut conv, algorithm) in cases {
        assert!(Layout::new(conv, Partition::Whole, SplitCells::All, algorithm).is_ok());
        conv.input[1] -= 2;
        assert!(matches!(
            Layout::new(conv, Partition::Whole, SplitCells::All, algorithm),
            Err(CudaError::Unsupported { .. })
        ));
    }
}

#[test]
fn strided_tensor_launch_mirrors_device_tiles() {
    let conv = |batch, in_channels, input, math| Conv2d {
        batch,
        input,
        in_channels,
        out_channels: 2 * in_channels,
        kernel: [3, 3],
        padding: [1, 1],
        stride: [2, 2],
        dilation: [1, 1],
        math,
    };
    let (f, t) = (CudaMath::Fp32, CudaMath::Tf32);
    let (one, slim, three, wide) = (
        TensorKernel::Tf32,
        TensorKernel::Tf32Slim,
        TensorKernel::Tf32x3,
        TensorKernel::Tf32x3Wide,
    );
    // 64 -> 128: three 96-column narrow tiles per output row, or two of 128; 128 ->
    // 256: two 64-column narrow tiles per row of 125, or one of 128; 3xTF32 tiles
    // are always 32 or 48 columns wide
    for (conv, products, grid, shared) in [
        (conv(1, 64, [40, 499], t), one, (1, 60, 1), 37376),
        (conv(32, 64, [40, 499], t), one, (1, 1280, 1), 49664),
        (conv(1, 128, [20, 250], t), one, (2, 20, 1), 25088),
        (conv(32, 128, [20, 250], t), one, (2, 320, 1), 49664),
        (conv(32, 64, [40, 499], f), three, (1, 5120, 1), 12800),
        (conv(32, 128, [20, 250], f), three, (2, 1280, 1), 12800),
        (conv(1, 128, [20, 250], t), three, (2, 40, 1), 12800),
        (conv(32, 64, [40, 499], f), wide, (1, 3840, 1), 18944),
        (conv(32, 128, [20, 250], f), wide, (2, 960, 1), 18944),
        (conv(1, 64, [40, 499], t), slim, (1, 80, 1), 25088),
        (conv(1, 128, [20, 250], t), slim, (2, 40, 1), 12800),
    ] {
        let layout = Layout::new(
            conv,
            Partition::Whole,
            SplitCells::All,
            Algorithm::TensorCore(products),
        )
        .expect("compiled strided shape");
        assert_eq!(layout.config.grid_dim, grid);
        assert_eq!(layout.config.shared_mem_bytes, shared);
    }
    // one product rounds to TF32, and three products cover only stride 2
    let same = Conv2d {
        stride: [1, 1],
        out_channels: 128,
        ..conv(32, 128, [20, 250], t)
    };
    for (conv, products) in [
        (conv(32, 64, [40, 499], f), one),
        (conv(1, 64, [40, 499], f), slim),
        (same, three),
        (same, slim),
    ] {
        assert!(matches!(
            Layout::new(
                conv,
                Partition::Whole,
                SplitCells::All,
                Algorithm::TensorCore(products)
            ),
            Err(CudaError::Unsupported { .. })
        ));
    }
}

#[test]
fn winograd_launch_mirrors_device_tiles() {
    let conv = |batch, channels, input, stride| Conv2d {
        batch,
        input,
        in_channels: channels,
        out_channels: channels,
        kernel: [3, 3],
        padding: [1, 1],
        stride: [stride; 2],
        dilation: [1, 1],
        math: CudaMath::Fp32,
    };
    // 64 output channels by one row of 32 tiles: 125 tile columns take 4 segments
    // per tile row and 63 take 2; whole cells take one CTA row, split cells one per
    // partition
    for (batch, channels, input, partition, split_cells, grid) in [
        (
            32,
            128,
            [20, 250],
            Partition::Whole,
            SplitCells::All,
            (2, 1280, 1),
        ),
        (
            32,
            256,
            [10, 125],
            Partition::Whole,
            SplitCells::All,
            (4, 320, 1),
        ),
        (
            1,
            128,
            [20, 250],
            Partition::Two,
            SplitCells::All,
            (2, 80, 1),
        ),
        (
            1,
            256,
            [10, 125],
            Partition::Four,
            SplitCells::All,
            (4, 40, 1),
        ),
        (
            1,
            128,
            [20, 250],
            Partition::Two,
            SplitCells::From(34),
            (2, 46, 1),
        ),
        (
            1,
            256,
            [10, 125],
            Partition::Four,
            SplitCells::From(8),
            (4, 16, 1),
        ),
    ] {
        let layout = Layout::new(
            conv(batch, channels, input, 1),
            partition,
            split_cells,
            Algorithm::Winograd(WinogradProducts::Fp32),
        )
        .expect("compiled same-channel shape");
        assert_eq!(layout.config.grid_dim, grid);
        assert_eq!(layout.config.block_dim, (256, 1, 1));
        assert_eq!(layout.config.shared_mem_bytes, 62_464);
        let planes = partition.count() as usize;
        let output = batch * channels * input[0] * input[1];
        assert_eq!(
            layout.workspace_len,
            if planes == 1 { 0 } else { planes * output }
        );
    }
    // the split tail cannot start past the last cell, nor split other algorithms
    let c128 = conv(1, 128, [20, 250], 1);
    assert!(
        Layout::new(
            c128,
            Partition::Two,
            SplitCells::From(41),
            Algorithm::Winograd(WinogradProducts::Fp32)
        )
        .is_err()
    );
    assert!(
        Layout::new(
            c128,
            Partition::Two,
            SplitCells::From(4),
            Algorithm::Spatial
        )
        .is_err()
    );
    for conv in [conv(32, 128, [20, 248], 1), conv(32, 128, [20, 250], 2)] {
        assert!(matches!(
            Layout::new(
                conv,
                Partition::Whole,
                SplitCells::All,
                Algorithm::Winograd(WinogradProducts::Fp32)
            ),
            Err(CudaError::Unsupported { .. })
        ));
    }
}

#[test]
fn selection_follows_device_attributes() {
    use super::super::wideconv::{Config, Device};
    use crate::inference::cuda::{ComputeCapability, PtxTier};

    let device = |major, minor, sms, tier| Device {
        capability: ComputeCapability::new(major, minor),
        sms,
        tier,
    };
    let ada = device(8, 9, 34, PtxTier::Sm80);
    let ada_sm75 = device(8, 9, 34, PtxTier::Sm75);
    let blackwell = device(12, 0, 36, PtxTier::Sm80);
    let a100 = device(8, 0, 108, PtxTier::Sm80);
    let a100_sm75 = device(8, 0, 108, PtxTier::Sm75);
    let same = |batch, channels, input, math| Conv2d {
        batch,
        input,
        in_channels: channels,
        out_channels: channels,
        kernel: [3, 3],
        padding: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        math,
    };
    let c128 = |batch, math| same(batch, 128, [20, 250], math);
    let c256 = |batch, math| same(batch, 256, [10, 125], math);
    let pick = |device, conv| Config::select(device, conv).expect("known contract");
    let wino = |products, partition, split_cells| Config {
        algorithm: Algorithm::Winograd(products),
        partition,
        split_cells,
    };
    let whole = |algorithm| Config {
        algorithm,
        partition: Partition::Whole,
        split_cells: SplitCells::All,
    };
    let tensor = whole(Algorithm::TensorCore(TensorKernel::Tf32));
    let slim = whole(Algorithm::TensorCore(TensorKernel::Tf32Slim));
    let spatial = whole(Algorithm::Spatial);
    let spatial_four = Config {
        algorithm: Algorithm::Spatial,
        partition: Partition::Four,
        split_cells: SplitCells::All,
    };
    let tc3 = whole(Algorithm::TensorCore(TensorKernel::Tf32x3));
    let tc3w = whole(Algorithm::TensorCore(TensorKernel::Tf32x3Wide));
    let strided = |batch, in_channels, input, math| Conv2d {
        batch,
        input,
        in_channels,
        out_channels: 2 * in_channels,
        kernel: [3, 3],
        padding: [1, 1],
        stride: [2, 2],
        dilation: [1, 1],
        math,
    };
    let c64s2 = |batch, math| strided(batch, 64, [40, 499], math);
    let c128s2 = |batch, math| strided(batch, 128, [20, 250], math);
    use Partition::{Four, Two, Whole};
    use SplitCells::{All, From};
    use WinogradProducts::{Fp32, Fp32Sweep2, Tf32x2, Tf32x3};
    let (f, t) = (CudaMath::Fp32, CudaMath::Tf32);
    // batch 1 has 80 (128 channels) or 40 (256 channels) CTAs: 34 or 36 SMs run the
    // 128-channel layers' whole waves and split the cells of the partial wave to fill
    // one wave, and split every cell of the 256-channel layers' one wave and a bit;
    // 108 SMs split the tensor-core products of every cell up to one wave, and in FP32
    // mode in at least two; batch 32 has thousands of CTAs
    for (device, conv, expected) in [
        (ada, c128(1, f), wino(Fp32Sweep2, Two, From(34))),
        (ada, c256(1, f), wino(Fp32, Four, All)),
        (ada, c128(32, f), wino(Fp32Sweep2, Whole, All)),
        (ada, c256(32, f), wino(Fp32, Whole, All)),
        (blackwell, c128(32, f), wino(Fp32Sweep2, Whole, All)),
        (ada, c128(1, t), wino(Fp32, Two, From(34))),
        (ada, c128(32, t), tensor),
        (ada_sm75, c128(32, t), wino(Fp32, Whole, All)),
        (blackwell, c128(1, f), wino(Fp32Sweep2, Four, From(36))),
        (blackwell, c256(1, f), wino(Fp32, Four, All)),
        (a100, c128(1, f), wino(Tf32x3, Two, All)),
        (a100, c256(1, f), wino(Tf32x3, Two, All)),
        (a100, c128(1, t), wino(Tf32x2, Whole, All)),
        (a100, c128(32, f), wino(Tf32x3, Whole, All)),
        (a100, c128(32, t), wino(Tf32x2, Whole, All)),
        (a100, c256(1, t), wino(Tf32x3, Two, All)),
        (a100, c256(32, t), tensor),
        (a100_sm75, c128(32, f), wino(Fp32Sweep2, Whole, All)),
        // stride 2: TF32 mode on the sm80 tier runs one TF32 product everywhere, in
        // slim tiles where the SMs outnumber the narrow tiles; FP32 mode runs 3xTF32
        // tiles on TF32-rich parts, except the 128 -> 256 layer below batch 8, and
        // the spatial kernels elsewhere
        (ada, c64s2(32, t), tensor),
        (ada, c64s2(1, t), tensor),
        (ada, c128s2(1, t), tensor),
        (blackwell, c128s2(1, t), tensor),
        (a100, c64s2(1, t), slim),
        (a100, c128s2(1, t), slim),
        (a100, c64s2(2, t), tensor),
        (a100, c128s2(2, t), slim),
        (ada, c64s2(32, f), spatial),
        (blackwell, c128s2(32, f), spatial),
        (a100, c64s2(32, f), tc3w),
        (a100, c128s2(32, f), tc3w),
        (a100, c64s2(8, f), tc3w),
        (a100, c64s2(1, f), tc3),
        (a100, c64s2(7, f), tc3),
        (a100, c128s2(1, f), spatial_four),
        (a100, c128s2(8, f), tc3w),
        (a100, c128s2(32, t), tensor),
        (a100_sm75, c64s2(32, f), spatial),
    ] {
        assert_eq!(pick(device, conv), expected, "{device:?} {conv:?}");
    }
    // the stem's 39 wide CTAs per item fill eight waves from batch 7 on 34 SMs and
    // from batch 23 on 108
    assert_eq!(pick(ada, stem(1, [80, 998])).algorithm, Algorithm::Spatial);
    assert_eq!(pick(ada, stem(7, [80, 998])).algorithm, Algorithm::WideStem);
    assert_eq!(
        pick(a100, stem(22, [80, 998])).algorithm,
        Algorithm::Spatial
    );
    assert_eq!(
        pick(a100, stem(32, [80, 998])).algorithm,
        Algorithm::WideStem
    );
    let wide = Layout::new(
        stem(32, [80, 998]),
        Partition::Whole,
        SplitCells::All,
        Algorithm::WideStem,
    )
    .expect("stem contract");
    assert_eq!(wide.config.grid_dim, (39, 32, 1));
    assert_eq!(wide.config.block_dim, (256, 1, 1));
}

/// The routing finds a kernel for every trunk convolution in exactly one area: the
/// PR #36 shapes or these
#[test]
fn trunk_convolutions_have_exactly_one_kernel_area() {
    use crate::inference::cuda::candidate::{ConvCandidate, ConvOxide, WideconvOxide};
    use crate::inference::cuda::implementation::BoundaryId;
    use crate::inference::cuda::{KernelModule, PtxTier};

    let resnet: Vec<&str> = ConvOxide::COVERAGE
        .entries()
        .iter()
        .flat_map(|entry| entry.layers.iter().copied())
        .collect();
    let trunk: Vec<_> = BoundaryId::all()
        .filter(|boundary| boundary.area() == KernelModule::Resnet)
        .collect();
    assert_eq!(trunk.len(), 36);
    for boundary in trunk {
        let name = boundary.name();
        for tier in [PtxTier::Sm75, PtxTier::Sm80] {
            for batch in [1, 7, 32, 33, 64] {
                for math in [CudaMath::Fp32, CudaMath::Tf32] {
                    let wide = WideconvOxide::coverage(tier).covers(name, batch, math);
                    assert!(
                        wide != resnet.contains(&name),
                        "{name} b{batch} {math:?} {tier}"
                    );
                }
            }
        }
    }
}

#[test]
fn rejects_empty_and_partitioned_stems() {
    for conv in [stem(0, [80, 998]), stem(1, [0, 998])] {
        assert!(matches!(
            Layout::new(conv, Partition::Whole, SplitCells::All, Algorithm::Spatial),
            Err(CudaError::Unsupported { .. })
        ));
    }
    assert!(
        Layout::new(
            stem(1, [80, 998]),
            Partition::Two,
            SplitCells::All,
            Algorithm::Spatial
        )
        .is_err()
    );
}
