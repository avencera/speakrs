//! Model-free checks that types public signatures take are nameable outside the crate

#![cfg(feature = "_backend")]

use speakrs::inference::MaskedEmbeddingInput;
use speakrs::pipeline::{PldaError, PldaTransform};

#[test]
fn embed_batch_inputs_and_plda_transform_are_constructible() {
    let audio = [0.0_f32; 4];
    let mask = [1.0_f32; 2];
    let input = MaskedEmbeddingInput {
        audio: &audio,
        mask: &mask,
        clean_mask: None,
    };
    assert_eq!(input.audio.len() + input.mask.len(), 6);

    let missing = tempfile::tempdir().unwrap();
    let error = PldaTransform::from_dir(missing.path()).unwrap_err();
    assert!(matches!(error, PldaError::Io(_)));
}
