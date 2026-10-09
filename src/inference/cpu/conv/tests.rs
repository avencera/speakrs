use ndarray::{Array2, Array3, array, s};

use super::{Conv1d, Conv1dSpec, Conv2d, Conv2dSpec, ConvWorkspace, TILE};

#[test]
fn one_dimensional_cross_correlation_and_scratch_reuse() {
    let spec = Conv1dSpec::new(2, 1, 2, 1).unwrap();
    let conv = Conv1d::new(spec, vec![1.0, 2.0, -1.0, 3.0], Some(vec![0.5])).unwrap();
    let input = array![[1.0, 2.0, 4.0, 7.0], [-2.0, 3.0, 1.0, 5.0]];
    let mut output = Array2::zeros((1, 3));
    let mut workspace = ConvWorkspace::new();
    conv.forward(input.view(), output.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(output, array![[16.5, 10.5, 32.5]]);
    let pointer = workspace.columns.as_ptr();
    conv.forward((-&input).view(), output.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(output, array![[-15.5, -9.5, -31.5]]);
    assert_eq!(workspace.columns.as_ptr(), pointer);
    assert_eq!(workspace.columns.ncols(), TILE);
}

#[test]
fn one_dimensional_stride_and_noncontiguous_views() {
    let conv = Conv1d::new(Conv1dSpec::new(1, 1, 2, 2).unwrap(), vec![2.0, -3.0], None).unwrap();
    let input = array![[1.0, 99.0, 2.0, 99.0, 4.0, 99.0, 7.0, 99.0]];
    let mut output = Array2::from_elem((1, 4), -100.0);
    conv.forward(
        input.slice(s![.., ..;2]),
        output.slice_mut(s![.., ..;2]),
        &mut ConvWorkspace::new(),
    )
    .unwrap();
    assert_eq!(output, array![[-4.0, -100.0, -13.0, -100.0]]);
}

#[test]
fn two_dimensional_non_symmetric_kernel_stride_and_padding() {
    let input =
        Array3::from_shape_vec((1, 3, 3), (1..=9).map(|value| value as f32).collect()).unwrap();
    let mut workspace = ConvWorkspace::new();
    let conv = Conv2d::new(
        Conv2dSpec::new(1, 1, 2, 1, 0).unwrap(),
        vec![1.0, 2.0, 3.0, 5.0],
        vec![-2.0],
    )
    .unwrap();
    let mut output = Array3::zeros((1, 2, 2));
    conv.forward(input.view(), output.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice().unwrap(), &[40.0, 51.0, 73.0, 84.0]);
    let conv = Conv2d::new(
        Conv2dSpec::new(1, 1, 2, 2, 1).unwrap(),
        vec![1.0, 2.0, 3.0, 5.0],
        vec![-2.0],
    )
    .unwrap();
    conv.forward(input.view(), output.view_mut(), &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice().unwrap(), &[3.0, 19.0, 41.0, 84.0]);
}

#[test]
fn two_dimensional_strided_output_matches_contiguous_and_changed_input() {
    let conv = Conv2d::new(
        Conv2dSpec::new(1, 1, 3, 1, 1).unwrap(),
        vec![1.0, -1.0, 2.0, 3.0, 0.0, -2.0, -3.0, 4.0, 2.0],
        vec![0.25],
    )
    .unwrap();
    let input = Array3::from_shape_fn((1, 3, 5), |(_, y, x)| (y * 7 + x * 3) as f32);
    let mut expected = Array3::zeros(input.dim());
    let mut storage = Array3::from_elem((1, 6, 10), 99.0);
    let mut workspace = ConvWorkspace::new();
    conv.forward(input.view(), expected.view_mut(), &mut workspace)
        .unwrap();
    conv.forward(
        input.view(),
        storage.slice_mut(s![.., ..;2, ..;2]),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(storage.slice(s![.., ..;2, ..;2]), expected);
    let changed = -&input;
    conv.forward(changed.view(), expected.view_mut(), &mut workspace)
        .unwrap();
    conv.forward(
        changed.slice(s![.., ..;-1, ..;-1]),
        storage.slice_mut(s![.., ..;2, ..;2]),
        &mut workspace,
    )
    .unwrap();
    let mut reversed = Array3::zeros(input.dim());
    conv.forward(
        changed.slice(s![.., ..;-1, ..;-1]),
        reversed.view_mut(),
        &mut workspace,
    )
    .unwrap();
    assert_eq!(storage.slice(s![.., ..;2, ..;2]), reversed);
    assert_ne!(expected[[0, 1, 2]], 0.25);
}

#[test]
fn shape_weight_and_overflow_failures_leave_output_unchanged() {
    assert!(Conv1dSpec::new(0, 1, 3, 1).is_err());
    assert!(Conv2dSpec::new(1, 1, usize::MAX, 1, 1).is_err());
    assert!(Conv2dSpec::new(1, 1, 3, 1, usize::MAX).is_err());
    let spec = Conv1dSpec::new(1, 2, 3, 1).unwrap();
    assert!(spec.output_len(2).is_err());
    assert!(Conv1d::new(spec, vec![1.0; 5], None).is_err());
    assert!(Conv1d::new(spec, vec![1.0; 6], Some(vec![0.0])).is_err());
    let conv = Conv1d::new(spec, vec![1.0; 6], None).unwrap();
    let mut output = Array2::from_elem((2, 2), 17.0);
    assert!(
        conv.forward(
            Array2::zeros((2, 4)).view(),
            output.view_mut(),
            &mut ConvWorkspace::new()
        )
        .is_err()
    );
    assert_eq!(output, Array2::from_elem((2, 2), 17.0));
    let spec = Conv2dSpec::new(1, 1, 3, 1, 1).unwrap();
    assert!(Conv2d::new(spec, vec![1.0; 9], vec![]).is_err());
    let conv = Conv2d::new(spec, vec![1.0; 9], vec![0.0]).unwrap();
    assert!(
        conv.forward(
            Array3::zeros((1, 3, 3)).view(),
            Array3::zeros((1, 2, 3)).view_mut(),
            &mut ConvWorkspace::new()
        )
        .is_err()
    );
}

#[test]
fn tile_boundaries_and_multiple_channels_match_direct_cross_correlation() {
    let input = Array3::from_shape_fn((2, 3, TILE + 7), |(c, y, x)| {
        ((c * 13 + y * 5 + x * 7) % 31) as f32 - 15.0
    });
    let weights: Vec<f32> = (0..36).map(|index| (index % 7) as f32 - 3.0).collect();
    let conv = Conv2d::new(
        Conv2dSpec::new(2, 2, 3, 1, 1).unwrap(),
        weights.clone(),
        vec![1.0, -2.0],
    )
    .unwrap();
    let mut output = Array3::zeros((2, 3, TILE + 7));
    conv.forward(input.view(), output.view_mut(), &mut ConvWorkspace::new())
        .unwrap();
    for channel in 0..2 {
        for y in 0..3 {
            for x in [0, 1, TILE - 1, TILE, TILE + 6] {
                let mut expected = if channel == 0 { 1.0 } else { -2.0 };
                for input_channel in 0..2 {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let sy = y as isize + ky as isize - 1;
                            let sx = x as isize + kx as isize - 1;
                            if (0..3).contains(&sy) && (0..(TILE + 7) as isize).contains(&sx) {
                                expected += weights
                                    [(channel * 2 + input_channel) * 9 + ky * 3 + kx]
                                    * input[[input_channel, sy as usize, sx as usize]];
                            }
                        }
                    }
                }
                assert_eq!(
                    output[[channel, y, x]],
                    expected,
                    "channel={channel} y={y} x={x}"
                );
            }
        }
    }
}
