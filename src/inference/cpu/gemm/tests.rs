use ndarray::{Array2, array, s};

use super::matmul;

#[test]
fn nonsymmetric_product_and_accumulation() {
    let a = array![[1.0, 2.0, 4.0], [-2.0, 3.0, 1.0]];
    let b = array![[2.0, -1.0], [3.0, 5.0], [-4.0, 2.0]];
    let expected = array![[-8.0, 17.0], [1.0, 19.0]];
    let mut output = Array2::from_elem((2, 2), f32::NAN);
    matmul(a.view(), b.view(), output.view_mut(), 0.0).unwrap();
    assert_eq!(output, expected);
    matmul(a.view(), b.view(), output.view_mut(), 2.0).unwrap();
    assert_eq!(output, expected * 3.0);
}

#[test]
fn transposed_operands_output_and_strided_fallback() {
    let a = array![[1.0, -2.0], [2.0, 3.0], [4.0, 1.0]];
    let b = array![[2.0, 3.0, -4.0], [-1.0, 5.0, 2.0]];
    let expected = array![[-8.0, 17.0], [1.0, 19.0]];
    let mut output = Array2::zeros((2, 2));
    matmul(a.t(), b.t(), output.view_mut().reversed_axes(), 0.0).unwrap();
    assert_eq!(output.t(), expected);
    let mut wide_a = Array2::zeros((2, 6));
    wide_a.slice_mut(s![.., ..;2]).assign(&a.t());
    let mut wide_output = Array2::from_elem((2, 4), -7.0);
    matmul(
        wide_a.slice(s![.., ..;2]),
        b.t(),
        wide_output.slice_mut(s![.., ..;2]),
        0.0,
    )
    .unwrap();
    assert_eq!(wide_output.slice(s![.., ..;2]), expected);
    assert!(
        wide_output
            .slice(s![.., 1..;2])
            .iter()
            .all(|value| *value == -7.0)
    );
    matmul(
        wide_a.slice(s![.., ..;2]).slice_move(s![.., ..;-1]),
        b.t(),
        output.view_mut(),
        0.0,
    )
    .unwrap();
    assert_eq!(output, array![[10.0, 8.0], [19.0, 10.0]]);
}

#[test]
fn rejects_dimensions_and_handles_empty_reduction() {
    let a = Array2::zeros((2, 3));
    let b = Array2::zeros((4, 2));
    let mut output = Array2::from_elem((2, 2), 9.0);
    assert!(matmul(a.view(), b.view(), output.view_mut(), 0.0).is_err());
    assert_eq!(output, Array2::from_elem((2, 2), 9.0));
    matmul(
        Array2::zeros((2, 0)).view(),
        Array2::zeros((0, 2)).view(),
        output.view_mut(),
        0.0,
    )
    .unwrap();
    assert_eq!(output, Array2::<f32>::zeros((2, 2)));
}

#[test]
fn rectangular_transposed_views_preserve_logical_dimensions() {
    let a = array![[1.0, 2.0], [-2.0, 3.0], [4.0, 1.0]];
    let b = array![[2.0, -1.0, 3.0, 4.0], [3.0, 5.0, -2.0, 1.0]];
    let stored_a = a.t().to_owned();
    let stored_b = b.t().to_owned();
    let mut output = Array2::from_elem((4, 3), f32::NAN);
    matmul(
        stored_a.t(),
        stored_b.t(),
        output.view_mut().reversed_axes(),
        0.0,
    )
    .unwrap();
    assert_eq!(
        output.t(),
        array![
            [8.0, 9.0, -1.0, 6.0],
            [5.0, 17.0, -12.0, -5.0],
            [11.0, 1.0, 10.0, 17.0]
        ]
    );
}
