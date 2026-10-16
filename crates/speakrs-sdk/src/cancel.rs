//! Cooperative cancellation shared between a caller and a running operation

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// A cancellation flag shared between the caller and a running operation
///
/// Operations check the flag between stages and between batch files; a stage
/// that is already running finishes first
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    parent: Option<Box<CancelToken>>,
}

impl CancelToken {
    /// Create a token that is not cancelled
    pub fn new() -> Self {
        Self::default()
    }

    /// A token that is cancelled when this one is, but whose own cancellation
    /// leaves this one untouched
    ///
    /// A binding uses one per call, so a failure that cancels the call never
    /// cancels a token the caller shares with other calls
    pub fn child(&self) -> Self {
        Self {
            flag: Arc::default(),
            parent: Some(Box::new(self.clone())),
        }
    }

    /// Request cancellation; every clone observes it
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation was requested on this token or a parent
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
            || self
                .parent
                .as_deref()
                .is_some_and(CancelToken::is_cancelled)
    }

    pub(crate) fn check(token: Option<&Self>) -> Result<(), crate::SdkError> {
        match token {
            Some(token) if token.is_cancelled() => Err(crate::SdkError::cancelled()),
            _ => Ok(()),
        }
    }
}
