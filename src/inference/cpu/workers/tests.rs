use super::{InferenceError, test_support};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn ordered_results_lazy_growth_and_private_reuse() {
    let created = AtomicUsize::new(1);
    let mut workers = test_support::with_budget((0usize, 0usize), 4);
    for length in [0, 1, 2, 3, 4, 5, 8, 9] {
        let inputs: Vec<_> = (0..length).collect();
        let output = workers
            .map(
                &inputs,
                || (created.fetch_add(1, Ordering::SeqCst), 0),
                |(id, calls), input| {
                    *calls += 1;
                    Ok((*input * 7, *id, *calls))
                },
            )
            .unwrap();
        assert_eq!(
            output.iter().map(|row| row.0).collect::<Vec<_>>(),
            inputs.iter().map(|input| input * 7).collect::<Vec<_>>()
        );
        assert_eq!(test_support::count(&workers), length.clamp(1, 4));
        assert_eq!(created.load(Ordering::SeqCst), length.clamp(1, 4));
        if length == 4 {
            assert_eq!(
                output.iter().map(|row| row.1).collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
            assert_eq!(
                output.iter().map(|row| row.2).collect::<Vec<_>>(),
                vec![4, 3, 2, 1]
            );
        }
    }
    assert_eq!(workers.workspaces, vec![(0, 11), (1, 8), (2, 7), (3, 6)]);
}

#[test]
fn first_input_error_is_stable_and_failed_chunks_stop() {
    let finished = AtomicUsize::new(0);
    let mut workers = test_support::with_budget(0, 4);
    let result = workers.map(
        &[0, 1, 2, 3, 4, 5, 6, 7],
        || 0,
        |calls, input| {
            *calls += 1;
            finished.fetch_add(1, Ordering::SeqCst);
            if matches!(input, 1 | 4) {
                return Err(InferenceError::BatchTooLarge {
                    context: "test job",
                    rows: *input,
                    capacity: 0,
                });
            }
            Ok(*input)
        },
    );
    assert!(matches!(
        result,
        Err(InferenceError::BatchTooLarge { rows: 1, .. })
    ));
    assert_eq!(finished.load(Ordering::SeqCst), 7);
    assert_eq!(workers.workspaces, vec![2, 2, 1, 2]);
}

#[test]
fn panic_is_typed_joined_and_owner_can_be_reused() {
    let finished = AtomicUsize::new(0);
    let mut workers = test_support::with_budget(0, 4);
    let result = workers.map(
        &[0, 1, 2, 3, 4, 5, 6, 7],
        || 0,
        |calls, input| {
            *calls += 1;
            if *input == 0 {
                *calls = 999;
                panic!("test worker panic");
            }
            finished.fetch_add(1, Ordering::SeqCst);
            Ok(*input)
        },
    );
    assert!(
        matches!(result, Err(InferenceError::WorkerPanic { worker }) if worker == "native CPU")
    );
    assert_eq!(finished.load(Ordering::SeqCst), 6);
    assert_eq!(workers.workspaces, vec![0, 2, 2, 2]);
    assert_eq!(
        workers
            .map(
                &[5, 6],
                || 0,
                |calls, input| {
                    *calls += 1;
                    Ok(input + *calls)
                }
            )
            .unwrap(),
        vec![6, 9]
    );
    let result = workers.map(
        &[0],
        || 0,
        |calls, _| -> Result<usize, InferenceError> {
            *calls = 999;
            panic!("single panic")
        },
    );
    assert!(matches!(result, Err(InferenceError::WorkerPanic { .. })));
    assert_eq!(workers.workspaces[0], 0);
}

#[test]
fn earlier_model_error_wins_over_later_panic_and_all_panics_reset() {
    let mut workers = test_support::with_budget(0, 4);
    let result = workers.map(
        &[0, 1, 2, 3, 4, 5, 6, 7],
        || 0,
        |calls, input| {
            *calls += 1;
            if *input == 1 {
                return Err(InferenceError::BatchTooLarge {
                    context: "test job",
                    rows: 1,
                    capacity: 0,
                });
            }
            if matches!(input, 4 | 6) {
                *calls = 999;
                panic!("later panic");
            }
            Ok(*input)
        },
    );
    assert!(matches!(
        result,
        Err(InferenceError::BatchTooLarge { rows: 1, .. })
    ));
    assert_eq!(workers.workspaces, vec![2, 2, 0, 0]);
    assert_eq!(
        workers
            .map(
                &[10, 20, 30, 40],
                || 0,
                |calls, input| {
                    *calls += 1;
                    Ok(input + *calls)
                }
            )
            .unwrap(),
        vec![13, 23, 31, 41]
    );
}
