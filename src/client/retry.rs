//! Bounded exponential backoff retry policy for cheap, idempotent operations.
//!
//! By design, this policy is ONLY applied to metadata/health endpoints
//! (`/api/version`, `/api/tags`, `/api/ps`). Expensive CPU inference calls
//! (`/api/chat`, `/api/generate`) are never automatically retried.

use std::future::Future;
use std::time::Duration;

use crate::error::WorkerResult;

/// Hard upper bound on retries to prevent runaway retry loops.
pub const MAX_ALLOWED_RETRIES: usize = 5;

/// Bounded exponential backoff configuration for idempotent requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    max_retries: usize,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    /// Creates a bounded [`RetryPolicy`], clamping `max_retries` to [`MAX_ALLOWED_RETRIES`].
    #[must_use]
    pub fn new(max_retries: usize, initial_backoff: Duration, max_backoff: Duration) -> Self {
        let clamped_retries = max_retries.min(MAX_ALLOWED_RETRIES);
        let safe_max = max_backoff.max(initial_backoff);
        Self {
            max_retries: clamped_retries,
            initial_backoff,
            max_backoff: safe_max,
        }
    }

    /// Disables all retries (single attempt).
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::ZERO,
            max_backoff: Duration::ZERO,
        }
    }

    /// Returns the configured maximum number of retries after the initial attempt.
    #[must_use]
    pub const fn max_retries(&self) -> usize {
        self.max_retries
    }

    /// Computes exponential backoff duration for the zero-based `retry_index`.
    #[must_use]
    pub fn backoff_for_retry(&self, retry_index: usize) -> Duration {
        let shift = u32::try_from(retry_index).unwrap_or(u32::MAX).min(16);
        let multiplier = 1u32.checked_shl(shift).unwrap_or(u32::MAX);
        self.initial_backoff
            .saturating_mul(multiplier)
            .min(self.max_backoff)
    }

    /// Executes an idempotent async operation with bounded exponential backoff
    /// whenever the returned [`crate::WorkerError::is_retryable`] is `true`.
    pub async fn execute_idempotent<T, F, Fut>(
        &self,
        operation_name: &str,
        mut op: F,
    ) -> WorkerResult<T>
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = WorkerResult<T>>,
    {
        let mut retry_index = 0usize;
        loop {
            match op().await {
                Ok(value) => return Ok(value),
                Err(err) if err.is_retryable() && retry_index < self.max_retries => {
                    let delay = self.backoff_for_retry(retry_index);
                    tracing::warn!(
                        operation = operation_name,
                        retry = retry_index + 1,
                        max_retries = self.max_retries,
                        backoff_ms = delay.as_millis(),
                        error = %err,
                        "transient failure on idempotent operation; backing off before retry"
                    );
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    retry_index += 1;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::error::WorkerError;

    #[test]
    fn backoff_doubles_exponentially_and_caps_at_max() {
        let policy = RetryPolicy::new(
            10, // clamped to MAX_ALLOWED_RETRIES
            Duration::from_millis(100),
            Duration::from_millis(350),
        );
        assert_eq!(policy.max_retries(), MAX_ALLOWED_RETRIES);
        assert_eq!(policy.backoff_for_retry(0), Duration::from_millis(100));
        assert_eq!(policy.backoff_for_retry(1), Duration::from_millis(200));
        assert_eq!(policy.backoff_for_retry(2), Duration::from_millis(350));
        assert_eq!(policy.backoff_for_retry(5), Duration::from_millis(350));
    }

    #[tokio::test]
    async fn retries_transient_errors_and_stops_on_non_retryable() {
        let policy = RetryPolicy::new(2, Duration::from_millis(1), Duration::from_millis(5));
        let attempts = AtomicUsize::new(0);

        let value = policy
            .execute_idempotent("test_recover", || {
                let n = attempts.fetch_add(1, Ordering::SeqCst);
                async move {
                    if n < 2 {
                        Err(WorkerError::HttpStatus {
                            status: 503,
                            message: "busy".into(),
                        })
                    } else {
                        Ok("recovered")
                    }
                }
            })
            .await
            .expect("should recover on third attempt");
        assert_eq!(value, "recovered");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);

        let non_retryable_attempts = AtomicUsize::new(0);
        let err = policy
            .execute_idempotent("test_no_retry", || {
                non_retryable_attempts.fetch_add(1, Ordering::SeqCst);
                async {
                    Err::<(), _>(WorkerError::ModelNotFound {
                        model: "missing:3b".into(),
                    })
                }
            })
            .await
            .expect_err("non-retryable error must fail immediately");
        assert_eq!(err.exit_code(), 4);
        assert_eq!(non_retryable_attempts.load(Ordering::SeqCst), 1);
    }
}
