use super::CpuSegmentationBackend;
use crate::inference::cpu::workers::test_support;
use crate::test_support::model_fixture_dir;

#[test]
fn required_adapter_useful_lengths_order_and_fresh_clone() {
    let selector = model_fixture_dir().join("segmentation-3.0.onnx");
    let mut backend = CpuSegmentationBackend::load(&selector, 160000).unwrap();
    backend.workers = test_support::with_budget(backend.model.workspace(), 4);
    let mut clone = backend.clone();
    clone.workers = test_support::with_budget(clone.model.workspace(), 1);
    assert!(CpuSegmentationBackend::load(&selector, 0).is_err());
    let a = vec![0.0; 300];
    let b: Vec<f32> = (0..700).map(|i| (i as f32 * 0.3).sin() * 0.1).collect();
    let first = backend.run_window(&a).unwrap();
    let second = backend.run_window(&b).unwrap();
    assert!(first.iter().zip(&second).any(|(a, b)| (a - b).abs() > 0.01));
    assert!(backend.run_batch(&[]).unwrap().is_empty());
    for count in [1, 2, 3, 4, 5, 8, 9] {
        let windows: Vec<&[f32]> = (0..count)
            .map(|index| {
                if index % 2 == 0 {
                    a.as_slice()
                } else {
                    b.as_slice()
                }
            })
            .collect();
        let actual = backend.run_batch(&windows).unwrap();
        assert_eq!(actual.len(), count);
        for (index, output) in actual.iter().enumerate() {
            assert_eq!(output, if index % 2 == 0 { &first } else { &second });
        }
    }
    assert_eq!(test_support::count(&backend.workers), 4);
    assert_eq!(test_support::count(&clone.workers), 1);
    let mut fresh = backend.clone();
    assert_eq!(test_support::count(&fresh.workers), 1);
    assert!(!std::ptr::eq(
        backend.workers.first(),
        fresh.workers.first()
    ));
    assert_eq!(
        fresh.run_batch(&[&a, &b]).unwrap(),
        vec![first.clone(), second.clone()]
    );
    assert_eq!(clone.run_window(&a).unwrap(), first);
    assert_eq!(clone.run_window(&b).unwrap(), second);
}
