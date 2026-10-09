use std::num::NonZeroU32;

use color_eyre::eyre::{Result, eyre};

/// Rejects zero counts for commands that require at least one run
pub fn nonzero_u32(name: &str, value: u32) -> Result<NonZeroU32> {
    NonZeroU32::new(value).ok_or_else(|| eyre!("{name} must be greater than zero, got {value}"))
}

const STAGE_MODES: &[&str] = &["seg-only", "embed-stream", "embed-store", "embed-repeat"];

/// Accepts only supported CPU stage profiling modes
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
    use super::{nonzero_u32, parse_stage_mode};

    #[test]
    fn rejects_zero_run_counts() {
        assert!(nonzero_u32("runs", 0).is_err());
        assert_eq!(nonzero_u32("runs", 1).unwrap().get(), 1);
    }

    #[test]
    fn rejects_unknown_profile_modes() {
        assert!(parse_stage_mode("mystery").is_err());
        assert_eq!(parse_stage_mode("seg-only").unwrap(), "seg-only");
    }
}
