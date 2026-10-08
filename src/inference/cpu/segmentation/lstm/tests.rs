use super::{HIDDEN, Lstm, LstmLayer, LstmScratch, step_iofc};
use ndarray::Array2;

#[test]
fn concrete_iofc_step_keeps_gate_order_and_cell_state() {
    let (hidden, cell) = step_iofc([0.0, 0.0, 0.0, 0.5_f32.atanh()], 0.4);
    assert!((cell - 0.45).abs() < 1e-7);
    assert!((hidden - 0.5 * 0.45_f32.tanh()).abs() < 1e-7);
}

#[test]
fn reverse_direction_writes_original_time_and_resets_reused_state() {
    let mut w = vec![0.0; 2 * 4 * HIDDEN];
    for direction in 0..2 {
        w[direction * 4 * HIDDEN + 3 * HIDDEN] = 1.0;
    }
    let model = Lstm::new(LstmLayer {
        input: 1,
        w,
        r: vec![0.0; 2 * 4 * HIDDEN * HIDDEN],
        b: vec![0.0; 2 * 8 * HIDDEN],
    })
    .unwrap();
    let input = Array2::from_shape_fn((589, 1), |(time, _)| {
        if time < 3 {
            [0.2, -0.3, 0.7][time]
        } else {
            0.0
        }
    });
    let mut output = Array2::zeros((589, 256));
    let mut scratch = LstmScratch::new();
    model
        .forward(input.view(), &mut output, &mut scratch)
        .unwrap();
    let mut forward_cell = 0.0;
    let mut reverse_cell = 0.0;
    for time in 0..589 {
        forward_cell = 0.5 * forward_cell + 0.5 * input[[time, 0]].tanh();
        assert!((output[[time, 0]] - 0.5 * forward_cell.tanh()).abs() < 1e-7);
        let reverse = 588 - time;
        reverse_cell = 0.5 * reverse_cell + 0.5 * input[[reverse, 0]].tanh();
        assert!((output[[reverse, HIDDEN]] - 0.5 * reverse_cell.tanh()).abs() < 1e-7);
    }
    model
        .forward(Array2::zeros((589, 1)).view(), &mut output, &mut scratch)
        .unwrap();
    assert!(output.iter().all(|value| *value == 0.0));
}

#[test]
fn invalid_recurrent_weights_and_views_return_errors_before_state_or_output_changes() {
    let invalid = LstmLayer {
        input: 1,
        w: vec![0.0; 1],
        r: vec![0.0; 1],
        b: vec![0.0; 1],
    };
    assert!(Lstm::new(invalid).is_err());
    let model = Lstm::new(LstmLayer {
        input: 1,
        w: vec![0.0; 2 * 4 * HIDDEN],
        r: vec![0.0; 2 * 4 * HIDDEN * HIDDEN],
        b: vec![0.0; 2 * 8 * HIDDEN],
    })
    .unwrap();
    let mut scratch = LstmScratch::new();
    let mut output = Array2::from_elem((589, 256), 17.0);
    assert!(
        model
            .forward(Array2::zeros((588, 1)).view(), &mut output, &mut scratch)
            .is_err()
    );
    assert!(output.iter().all(|value| *value == 17.0));
    let storage = Array2::zeros((589, 2));
    model
        .forward(
            storage.slice(ndarray::s![..,..;2]),
            &mut output,
            &mut scratch,
        )
        .unwrap();
    assert!(output.iter().all(|value| *value == 0.0));
    let mut bad = Array2::from_elem((589, 255), 17.0);
    assert!(
        model
            .forward(storage.slice(ndarray::s![..,..;2]), &mut bad, &mut scratch)
            .is_err()
    );
    assert!(bad.iter().all(|value| *value == 17.0));
}
