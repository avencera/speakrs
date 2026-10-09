use super::pool;
use ndarray::{Array1, Array2, array, s};

#[test]
fn fractional_centered_statistics_and_degenerate_nan_layout() {
    let features = array![[1.0, 3.0], [2.0, 6.0]];
    let mut mask = vec![0.0; 589];
    mask[0] = 0.5;
    mask[294] = 1.0;
    let mut output = Array1::zeros(4);
    pool(features.view(), &mask, output.view_mut()).unwrap();
    assert!((output[0] - 7.0 / 3.0).abs() < 1e-6);
    assert!((output[1] - 14.0 / 3.0).abs() < 1e-6);
    assert!((output[2] - 2.0_f32.sqrt()).abs() < 1e-6);
    assert!((output[3] - 8.0_f32.sqrt()).abs() < 1e-6);
    pool(features.view(), &[], output.view_mut()).unwrap();
    assert_eq!(output, array![0.0, 0.0, 1e-5, 1e-5]);
    mask[294] = 0.0;
    mask[0] = 1.0;
    pool(features.view(), &mask, output.view_mut()).unwrap();
    assert_eq!(output[0], 1.0);
    assert_eq!(output[1], 2.0);
    assert!(output[2].is_nan() && output[3].is_nan());
}

#[test]
fn resize_floor_padding_truncation_and_strided_views() {
    let features = Array2::from_shape_fn((2, 250), |(_, t)| (t / 2) as f32);
    let mut mask = vec![0.0; 700];
    mask[588] = 1.0;
    mask[0] = 1.0;
    let mut result = Array1::zeros(4);
    pool(features.slice(s![..,..;2]), &mask, result.view_mut()).unwrap();
    // floor(124*589/125) is 584, so mask index 588 is not sampled
    assert_eq!(result[0], 0.0);
    assert!(result[2].is_nan());
    mask[584] = 1.0;
    pool(features.slice(s![..,..;2]), &mask, result.view_mut()).unwrap();
    assert_eq!(result[0], 62.0);
    assert!((result[2] - (7688.0_f32).sqrt()).abs() < 1e-5);
    let mut padded = Array1::from_elem(8, 17.0);
    pool(
        features.slice(s![..,..;2]),
        &mask,
        padded.slice_mut(s![..;2]),
    )
    .unwrap();
    assert_eq!(padded.slice(s![..;2]), result);
    assert!(padded.slice(s![1..;2]).iter().all(|value| *value == 17.0));
    assert!(pool(features.slice(s![.., ..0]), &mask, result.view_mut()).is_err());
}
