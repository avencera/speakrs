//! Canonical head timing cannot include repeated single-chunk passes

use super::EmbeddingHead;

#[test]
fn head_measurements_never_mix_single_and_multi_pass_work() {
    let measured: Vec<_> = [1, 4, 8, 16, 32]
        .into_iter()
        .filter_map(|chunks| EmbeddingHead::measurement_batch(chunks).map(|batch| (chunks, batch)))
        .collect();
    assert_eq!(measured, [(1, 1), (32, 32)]);
    for (chunks, batch) in measured {
        assert_eq!(chunks / EmbeddingHead::chunks_per_pass(chunks), 1);
        assert_eq!(chunks, batch);
    }
}
