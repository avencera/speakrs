use super::{CpuEmbedding, EmbeddingAssets, EmbeddingMeta, SplitTailInput};
use crate::test_support::model_fixture_dir;
use ndarray::{Array2, s};

fn load() -> CpuEmbedding {
    let assets =
        EmbeddingAssets::resolve(&model_fixture_dir().join("wespeaker-voxceleb-resnet34.onnx"))
            .expect("required pinned model fixture");
    CpuEmbedding::load(&assets).unwrap()
}

#[test]
fn required_adapter_empty_partial_full_batches_and_validation() {
    let mut backend = load();
    let mut clone = backend.clone();
    let meta = EmbeddingMeta {
        sample_rate: 16000,
        window_samples: 160000,
        mask_frames: 589,
        min_num_samples: 400,
    };
    let fbank = Array2::from_shape_fn((14, 80), |(t, b)| ((t * 3 + b * 13) as f32 * 0.03).sin());
    let a = vec![1.0; 589];
    let b = vec![0.3; 589];
    let c = vec![0.0; 589];
    let masks = [a.as_slice(), b.as_slice(), c.as_slice()];
    let expected = backend
        .embed_multi_mask_batch(&meta, &[&fbank], &masks)
        .unwrap();
    assert!(
        backend
            .embed_multi_mask_batch(&meta, &[], &[])
            .unwrap()
            .is_empty()
    );
    for count in [1, 7, 8] {
        let fbanks = vec![&fbank; count];
        let repeated: Vec<&[f32]> = (0..count).flat_map(|_| masks).collect();
        let actual = backend
            .embed_multi_mask_batch(&meta, &fbanks, &repeated)
            .unwrap();
        assert_eq!(actual.nrows(), count * 3);
        for chunk in 0..count {
            assert_eq!(actual.slice(s![chunk * 3..chunk * 3 + 3, ..]), expected);
        }
    }
    assert!(
        backend
            .embed_multi_mask_batch(&meta, &[&fbank], &[])
            .is_err()
    );
    assert!(
        backend
            .embed_multi_mask_batch(&meta, &[&fbank; 9], &[a.as_slice(); 27])
            .is_err()
    );
    let bad = Array2::zeros((999, 80));
    assert!(backend.embed_tail_single(&meta, &bad, &a).is_err());
    let storage = Array2::from_shape_fn(
        (589, 6),
        |(t, s)| if s % 2 == 0 && t < 300 { 1.0 } else { 0.0 },
    );
    let strided = storage.slice(s![..,..;2]);
    let clean = strided.to_owned();
    let rows = backend
        .embed_tail_batch(&meta, &fbank, &strided, &clean, 160000)
        .unwrap();
    assert!(rows.iter().all(|v| v.is_finite()));
    assert!(
        backend
            .embed_tail_batch(&meta, &fbank, &strided, &Array2::zeros((589, 2)), 160000)
            .is_err()
    );
    let inputs = [
        SplitTailInput {
            fbank: &fbank,
            weights: &a,
        },
        SplitTailInput {
            fbank: &fbank,
            weights: &c,
        },
    ];
    let tail = backend.embed_tail_batch_inputs(&meta, &inputs).unwrap();
    assert_eq!(tail.row(0), expected.row(0));
    assert_eq!(tail.row(1), expected.row(2));
    assert_eq!(
        clone
            .embed_multi_mask_batch(&meta, &[&fbank], &masks)
            .unwrap(),
        expected
    );
    let fbanks = backend
        .compute_chunk_fbanks_batch(&meta, &[&[], &[0.0; 300]])
        .unwrap();
    assert_eq!(fbanks[0], fbanks[1]);
    assert!(
        backend
            .compute_chunk_fbanks_batch(&meta, &[])
            .unwrap()
            .is_empty()
    );
    let audio: Vec<f32> = (0..1000).map(|t| (t as f32 * 0.1).sin() * 0.1).collect();
    let first = backend.embed_single(&meta, &audio, &a).unwrap();
    assert_eq!(clone.embed_single(&meta, &audio, &a).unwrap(), first);
    let batch = backend
        .embed_multi_mask_audio_batch(&meta, &[&audio], &masks)
        .unwrap();
    assert_eq!(batch.row(0), first);
    assert!(
        backend
            .embed_multi_mask_audio_batch(&meta, &[], &[])
            .unwrap()
            .is_empty()
    );
}

fn assert_same_values(actual: &Array2<f32>, expected: &Array2<f32>) {
    assert_eq!(actual.dim(), expected.dim());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
    }
}

#[test]
fn parallel_chunks_split_tails_nan_layout_and_whole_request_validation() {
    use super::EmbeddingWorker;
    use crate::inference::cpu::workers::test_support;

    let mut parallel = load();
    let mut serial = parallel.clone();
    parallel.workers =
        test_support::with_budget(EmbeddingWorker::new(&parallel.model, &parallel.fbank), 4);
    serial.workers =
        test_support::with_budget(EmbeddingWorker::new(&serial.model, &serial.fbank), 1);
    let meta = EmbeddingMeta {
        sample_rate: 16000,
        window_samples: 160000,
        mask_frames: 589,
        min_num_samples: 400,
    };
    let a = Array2::from_shape_fn((14, 80), |(t, b)| ((t * 3 + b * 13) as f32 * 0.03).sin());
    let b = Array2::from_shape_fn((23, 80), |(t, b)| ((t * 5 + b * 7) as f32 * 0.02).cos());
    let full = vec![1.0; 589];
    let empty = vec![0.0; 589];
    let mut single = empty.clone();
    single[0] = 1.0;
    let masks = [full.as_slice(), empty.as_slice(), single.as_slice()];

    parallel.workers.first().features.fill(-123.0);
    let malformed = Array2::zeros((999, 80));
    assert!(
        parallel
            .embed_multi_mask_batch(&meta, &[&a, &malformed], &[full.as_slice(); 6])
            .is_err()
    );
    assert_eq!(test_support::count(&parallel.workers), 1);
    assert!(
        parallel
            .workers
            .first()
            .features
            .iter()
            .all(|value| *value == -123.0)
    );
    let bad_inputs = [
        SplitTailInput {
            fbank: &a,
            weights: &full,
        },
        SplitTailInput {
            fbank: &malformed,
            weights: &full,
        },
    ];
    assert!(
        parallel
            .embed_tail_batch_inputs(&meta, &bad_inputs)
            .is_err()
    );
    assert_eq!(test_support::count(&parallel.workers), 1);
    let bad_meta = EmbeddingMeta {
        window_samples: 1,
        ..meta
    };
    assert!(
        parallel
            .embed_multi_mask_audio_batch(&bad_meta, &[&[], &[]], &[full.as_slice(); 6])
            .is_err()
    );
    assert!(
        parallel
            .compute_chunk_fbanks_batch(&bad_meta, &[&[], &[]])
            .is_err()
    );
    assert!(
        parallel
            .embed_multi_mask_audio_batch(&meta, &[&[], &[]], &[full.as_slice(); 3])
            .is_err()
    );
    assert_eq!(test_support::count(&parallel.workers), 1);
    assert!(
        parallel
            .workers
            .first()
            .features
            .iter()
            .all(|value| *value == -123.0)
    );

    for count in [0, 1, 2, 3, 4, 5, 8] {
        let fbanks: Vec<_> = (0..count)
            .map(|index| if index % 2 == 0 { &a } else { &b })
            .collect();
        let repeated: Vec<_> = (0..count).flat_map(|_| masks).collect();
        let expected = serial
            .embed_multi_mask_batch(&meta, &fbanks, &repeated)
            .unwrap();
        let actual = parallel
            .embed_multi_mask_batch(&meta, &fbanks, &repeated)
            .unwrap();
        assert_same_values(&actual, &expected);
        assert_eq!(actual.dim(), (count * 3, 256));
        if count > 0 {
            assert!(actual.row(0).iter().all(|value| value.is_finite()));
            assert!(actual.row(2).iter().all(|value| value.is_nan()));
        }
    }
    let inputs: Vec<_> = (0..9)
        .map(|index| SplitTailInput {
            fbank: if index % 2 == 0 { &a } else { &b },
            weights: masks[index % 3],
        })
        .collect();
    assert_same_values(
        &parallel.embed_tail_batch_inputs(&meta, &inputs).unwrap(),
        &serial.embed_tail_batch_inputs(&meta, &inputs).unwrap(),
    );
    let audio_a: Vec<_> = (0..1000).map(|t| (t as f32 * 0.1).sin() * 0.1).collect();
    let audio_b: Vec<_> = (0..1700).map(|t| (t as f32 * 0.07).cos() * 0.2).collect();
    let audios = [audio_a.as_slice(), audio_b.as_slice()];
    let repeated: Vec<_> = (0..2).flat_map(|_| masks).collect();
    assert_same_values(
        &parallel
            .embed_multi_mask_audio_batch(&meta, &audios, &repeated)
            .unwrap(),
        &serial
            .embed_multi_mask_audio_batch(&meta, &audios, &repeated)
            .unwrap(),
    );
    assert_eq!(
        parallel.compute_chunk_fbanks_batch(&meta, &audios).unwrap(),
        serial.compute_chunk_fbanks_batch(&meta, &audios).unwrap()
    );
    assert_eq!(test_support::count(&parallel.workers), 4);
    assert_eq!(test_support::count(&serial.workers), 1);
    let mut fresh = parallel.clone();
    assert_eq!(test_support::count(&fresh.workers), 1);
    assert_ne!(
        fresh.workers.first().features.as_ptr(),
        parallel.workers.first().features.as_ptr()
    );
    parallel.workers.first().features.fill(77.0);
    assert!(
        fresh
            .workers
            .first()
            .features
            .iter()
            .all(|value| *value == 0.0)
    );
    assert_same_values(
        &fresh
            .embed_multi_mask_audio_batch(&meta, &audios, &repeated)
            .unwrap(),
        &serial
            .embed_multi_mask_audio_batch(&meta, &audios, &repeated)
            .unwrap(),
    );
}
