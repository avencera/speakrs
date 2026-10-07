//! One-shot faults owned by a specific profile lifecycle phase

use super::{CALL_COUNT, CudaError, Mutant};

/// An enqueue or host replay position, independent of graph-node evidence
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Eager,
    Captured,
    Replay,
}

/// A fault cannot be consumed by an enqueue from another lifecycle phase
pub(crate) struct FirstUse {
    pending: Option<Stage>,
}

impl FirstUse {
    /// Resolve the compatible replay alias through the typed mutation owner
    pub(crate) fn stage(mutant: Mutant) -> Option<Stage> {
        match mutant {
            Mutant::FirstUseFallbackEager => Some(Stage::Eager),
            Mutant::FirstUseFallbackCaptured => Some(Stage::Captured),
            Mutant::FirstUseFallbackReplay => Some(Stage::Replay),
            _ => None,
        }
    }

    /// Prepare one fault for one operator lifecycle
    pub(crate) fn new(choice: &str) -> Self {
        Self {
            pending: Mutant::parse(choice).and_then(Self::stage),
        }
    }

    /// Issue exactly one real Library call at the selected first-use position
    pub(crate) fn before<T>(
        &mut self,
        stage: Stage,
        fallback: impl FnOnce() -> Result<T, CudaError>,
    ) -> Option<Result<T, CudaError>> {
        if self.pending != Some(stage) {
            return None;
        }
        self.pending = None;
        Some((|| {
            let before = CALL_COUNT.with(std::cell::Cell::get);
            let result = fallback()?;
            assert_eq!(
                CALL_COUNT.with(std::cell::Cell::get) - before,
                1,
                "first use must issue exactly one Library API call"
            );
            Ok(result)
        })())
    }
}

mod tests {
    use super::{FirstUse, Stage};
    use crate::inference::cuda::test_support::{CALL_COUNT, check_call};

    #[test]
    fn faults_run_once_at_their_owned_lifecycle_position() {
        for (choice, expected) in [
            ("FirstUseFallbackEager", Stage::Eager),
            ("FirstUseFallbackCaptured", Stage::Captured),
            ("FirstUseFallbackReplay", Stage::Replay),
            ("FirstUseFallback", Stage::Replay),
        ] {
            let saved = CALL_COUNT.with(std::cell::Cell::get);
            let mut hook = FirstUse::new(choice);
            let mut calls = Vec::new();
            // five warm-ups must not consume a capture-only or replay-only fault
            for (position, stage) in [Stage::Eager; 5]
                .into_iter()
                .chain([Stage::Eager, Stage::Captured, Stage::Replay, Stage::Replay])
                .enumerate()
            {
                if let Some(result) = hook.before(stage, || {
                    calls.push((position, stage));
                    check_call("cublas.single");
                    Ok(())
                }) {
                    result.unwrap();
                }
            }
            let position = match expected {
                Stage::Eager => 0,
                Stage::Captured => 6,
                Stage::Replay => 7,
            };
            assert_eq!(calls, [(position, expected)]);
            assert_eq!(CALL_COUNT.with(std::cell::Cell::get) - saved, 1);
            CALL_COUNT.with(|count| count.set(saved));
        }
        for stage in [Stage::Eager, Stage::Captured, Stage::Replay] {
            assert!(
                FirstUse::new("Library")
                    .before::<()>(stage, || panic!("control fault"))
                    .is_none()
            );
        }
    }

    #[test]
    fn a_fault_cannot_claim_zero_or_two_api_calls() {
        for calls in [0, 2] {
            let saved = CALL_COUNT.with(std::cell::Cell::get);
            assert!(
                std::panic::catch_unwind(|| {
                    FirstUse::new("FirstUseFallbackReplay").before(Stage::Replay, || {
                        for _ in 0..calls {
                            check_call("cublas.extra");
                        }
                        Ok(())
                    })
                })
                .is_err()
            );
            CALL_COUNT.with(|count| count.set(saved));
        }
    }
}
