use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A cancellation flag shared between the caller and a running operation
///
/// Operations check the flag between stages and between batch files; a stage
/// that is already running finishes first
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// Create a token that is not cancelled
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation; every clone observes it
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    pub(crate) fn check(token: Option<&Self>) -> Result<(), crate::SdkError> {
        match token {
            Some(token) if token.is_cancelled() => Err(crate::SdkError::cancelled()),
            _ => Ok(()),
        }
    }
}
