//! Shared ownership of candidate planning and production-only fallback

use super::CudaError;
use super::candidate::PlanError;
use super::implementation::{Choice, Selection};

/// Turns a candidate plan into the plan dispatch will run
///
/// `None` selects the already planned Library boundary. Only a device constraint
/// on a production selection permits fallback; real errors always propagate
pub(crate) fn candidate_plan<T>(
    choice: Choice,
    boundary: &str,
    batch: usize,
    result: Result<T, PlanError>,
) -> Result<Option<T>, CudaError> {
    match result {
        Ok(plan) => Ok(Some(plan)),
        Err(PlanError::DeviceUnsupported { reason }) => {
            if choice == Choice::Oxide(Selection::Production) {
                tracing::warn!(
                    "CUDA candidate unavailable boundary={boundary} batch={batch} reason={reason}; using Library"
                );
                return Ok(None);
            }

            Err(CudaError::Unsupported {
                context: "explicit CUDA candidate plan",
                reason: format!("boundary={boundary} batch={batch}: {reason}"),
            })
        }
        Err(PlanError::Cuda(error)) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::{Choice, CudaError, PlanError, Selection, candidate_plan};

    #[test]
    fn genuine_errors_propagate_in_both_modes() {
        for selection in [Selection::Production, Selection::Explicit] {
            // an Unsupported error is real unless the plan explicitly reports
            // a device constraint
            let error = CudaError::Unsupported {
                context: "invalid model shape",
                reason: "not a device limit".to_owned(),
            };
            let result = candidate_plan::<()>(
                Choice::Oxide(selection),
                "lstm.stack",
                1,
                Err(PlanError::Cuda(error)),
            );
            assert!(matches!(
                result,
                Err(CudaError::Unsupported {
                    context: "invalid model shape",
                    ..
                })
            ));
        }
    }

    #[test]
    fn successful_plans_are_retained_in_both_modes() {
        for selection in [Selection::Production, Selection::Explicit] {
            assert_eq!(
                candidate_plan(Choice::Oxide(selection), "lstm.stack", 1, Ok(42)).unwrap(),
                Some(42)
            );
        }
    }
}
