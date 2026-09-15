use std::path::Path;

use sha2::{Digest, Sha256};

pub(crate) fn digest_inputs(inputs: &[(String, Vec<u8>)]) -> String {
    let mut digest = Sha256::new();
    for (label, value) in inputs {
        update_length_prefixed(&mut digest, label.as_bytes());
        update_length_prefixed(&mut digest, value);
    }

    format!("{:x}", digest.finalize())
}

pub(crate) fn is_post_embedding_source(path: &Path) -> bool {
    let relative = path.strip_prefix("../src").unwrap_or(path);
    relative.starts_with("clustering")
        || matches!(
            relative.to_str(),
            Some(
                "binarize.rs"
                    | "metrics.rs"
                    | "reconstruct.rs"
                    | "segment.rs"
                    | "pipeline/clustering.rs"
                    | "pipeline/post_inference.rs"
                    | "pipeline/types/discrete.rs"
            )
        )
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}
