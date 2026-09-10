use std::num::{NonZeroU32, NonZeroUsize};

use color_eyre::eyre::{Result, eyre};

pub fn nonzero_u32(name: &str, value: u32) -> Result<NonZeroU32> {
    NonZeroU32::new(value).ok_or_else(|| eyre!("{name} must be greater than zero, got {value}"))
}

pub fn nonzero_usize(name: &str, value: usize) -> Result<NonZeroUsize> {
    NonZeroUsize::new(value).ok_or_else(|| eyre!("{name} must be greater than zero, got {value}"))
}

const ORT_EMBEDDING_MODES: &[&str] = &[
    "borrow",
    "owned",
    "prealloc",
    "stream-borrow",
    "stream-owned",
    "stream-prealloc",
    "stream-batched",
];

const STAGE_MODES: &[&str] = &["seg-only", "embed-stream", "embed-store", "embed-repeat"];

pub fn parse_ort_embedding_mode(mode: &str) -> Result<&'static str> {
    parse_known_mode("profile-ort-embedding", mode, ORT_EMBEDDING_MODES)
}

pub fn parse_stage_mode(mode: &str) -> Result<&'static str> {
    parse_known_mode("profile-stages", mode, STAGE_MODES)
}

fn parse_known_mode<'a>(command: &str, mode: &str, known: &[&'a str]) -> Result<&'a str> {
    known
        .iter()
        .copied()
        .find(|item| *item == mode)
        .ok_or_else(|| eyre!("unknown {command} mode '{mode}'"))
}

#[cfg(test)]
mod tests {
    use super::{nonzero_u32, nonzero_usize, parse_ort_embedding_mode, parse_stage_mode};

    #[test]
    fn rejects_zero_run_and_batch_counts() {
        assert!(nonzero_u32("runs", 0).is_err());
        assert_eq!(nonzero_u32("runs", 1).unwrap().get(), 1);
        assert!(nonzero_usize("batch-size", 0).is_err());
        assert_eq!(nonzero_usize("batch-size", 16).unwrap().get(), 16);
    }

    #[test]
    fn rejects_unknown_profile_modes() {
        assert!(parse_ort_embedding_mode("not-a-mode").is_err());
        assert_eq!(
            parse_ort_embedding_mode("stream-batched").unwrap(),
            "stream-batched"
        );
        assert!(parse_stage_mode("mystery").is_err());
        assert_eq!(parse_stage_mode("seg-only").unwrap(), "seg-only");
    }
}
