#[path = "../build_support.rs"]
mod build_support;

use std::path::Path;

#[test]
fn dependency_change_invalidates_embedding_identity() {
    let first = build_support::digest_inputs(&[
        ("source:embedding.rs".to_owned(), b"source".to_vec()),
        ("dependency:ort".to_owned(), b"2.0.0-rc.12".to_vec()),
    ]);
    let second = build_support::digest_inputs(&[
        ("source:embedding.rs".to_owned(), b"source".to_vec()),
        ("dependency:ort".to_owned(), b"2.0.0-rc.13".to_vec()),
    ]);

    assert_ne!(first, second);
}

#[test]
fn reconstruction_sources_are_outside_embedding_identity() {
    assert!(build_support::is_post_embedding_source(Path::new(
        "../src/reconstruct.rs"
    )));
    assert!(build_support::is_post_embedding_source(Path::new(
        "../src/pipeline/types/discrete.rs"
    )));
    assert!(!build_support::is_post_embedding_source(Path::new(
        "../src/pipeline/imported.rs"
    )));
}
