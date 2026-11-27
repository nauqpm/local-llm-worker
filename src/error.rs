//! Typed error model, machine-readable error categories, and process exit codes.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Convenience result alias for all `ollama_worker` library operations.
pub type WorkerResult<T> = Result<T, WorkerError>;

/// Standardized process exit codes aligned with `docs/CONTRACTS.md`.
pub mod exit_codes {
    pub const SUCCESS: u8 = 0;
    pub const INTERNAL_ERROR: u8 = 1;
    pub const INVALID_INPUT: u8 = 2;
    pub const OLLAMA_UNAVAILABLE: u8 = 3;
    pub const MODEL_NOT_FOUND: u8 = 4;
    pub const TIMEOUT: u8 = 5;
    pub const VALIDATION_FAILED: u8 = 6;
    pub const CANCELLED: u8 = 130;
}

/// Stable, machine-readable error categories serialized into structured task results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    InvalidInput,
    OllamaUnavailable,
    ProtocolError,
    ModelNotFound,
    Timeout,
    ValidationFailed,
    Cancelled,
    Internal,
}

impl ErrorCategory {
    /// Maps an error category to its documented CLI process exit code.
    #[must_use]
    pub const fn exit_code(self) -> u8 {
        match self {
            Self::InvalidInput => exit_codes::INVALID_INPUT,
            Self::OllamaUnavailable | Self::ProtocolError => exit_codes::OLLAMA_UNAVAILABLE,
            Self::ModelNotFound => exit_codes::MODEL_NOT_FOUND,
            Self::Timeout => exit_codes::TIMEOUT,
            Self::ValidationFailed => exit_codes::VALIDATION_FAILED,
            Self::Cancelled => exit_codes::CANCELLED,
            Self::Internal => exit_codes::INTERNAL_ERROR,
        }
    }
}

/// Serializable error payload embedded in structured JSON responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub category: ErrorCategory,
    pub message: String,
    pub retryable: bool,
}

/// Strongly typed error enum returned by the `ollama_worker` core library.
#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("invalid configuration: {message}")]
    InvalidConfig { message: String },

    #[error(
        "remote Ollama endpoint '{url}' is blocked by loopback-first policy; pass --allow-remote-ollama to opt in explicitly"
    )]
    RemoteEndpointBlocked { url: String },

    #[error("invalid task input: {message}")]
    InvalidInput { message: String },

    #[error(
        "prompt exceeds context budget ({estimated_tokens} estimated tokens > {budget_tokens} token budget); reduce context instead of relying on silent truncation"
    )]
    ContextBudgetExceeded {
        estimated_tokens: usize,
        budget_tokens: usize,
    },

    #[error("requested model '{model}' was not found on the Ollama server")]
    ModelNotFound { model: String },

    #[error("Ollama server unavailable: {message}")]
    OllamaUnavailable { message: String },

    #[error("Ollama returned unexpected HTTP status {status}: {message}")]
    HttpStatus { status: u16, message: String },

    #[error("malformed response from Ollama: {message}")]
    ProtocolError { message: String },

    #[error("request timed out after {timeout_secs}s")]
    Timeout { timeout_secs: u64 },

    #[error("output validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("operation cancelled by signal")]
    Cancelled,

    #[error("internal worker error: {message}")]
    Internal { message: String },
}

impl WorkerError {
    /// Returns the coarse-grained error category for automation callers.
    #[must_use]
    pub const fn category(&self) -> ErrorCategory {
        match self {
            Self::InvalidConfig { .. }
            | Self::RemoteEndpointBlocked { .. }
            | Self::InvalidInput { .. }
            | Self::ContextBudgetExceeded { .. } => ErrorCategory::InvalidInput,
            Self::ModelNotFound { .. } => ErrorCategory::ModelNotFound,
            Self::OllamaUnavailable { .. } | Self::HttpStatus { .. } => {
                ErrorCategory::OllamaUnavailable
            }
            Self::ProtocolError { .. } => ErrorCategory::ProtocolError,
            Self::Timeout { .. } => ErrorCategory::Timeout,
            Self::ValidationFailed { .. } => ErrorCategory::ValidationFailed,
            Self::Cancelled => ErrorCategory::Cancelled,
            Self::Internal { .. } => ErrorCategory::Internal,
        }
    }

    /// Returns the documented process exit code for this error.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        self.category().exit_code()
    }

    /// Indicates whether the underlying failure is transient.
    ///
    /// Callers must still respect the policy of never automatically retrying
    /// expensive in-flight CPU inference requests.
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::OllamaUnavailable { .. } => true,
            Self::HttpStatus { status, .. } => *status == 429 || *status >= 500,
            Self::InvalidConfig { .. }
            | Self::RemoteEndpointBlocked { .. }
            | Self::InvalidInput { .. }
            | Self::ContextBudgetExceeded { .. }
            | Self::ModelNotFound { .. }
            | Self::ProtocolError { .. }
            | Self::Timeout { .. }
            | Self::ValidationFailed { .. }
            | Self::Cancelled
            | Self::Internal { .. } => false,
        }
    }

    /// Converts this error into its structured JSON representation.
    #[must_use]
    pub fn to_detail(&self) -> ErrorDetail {
        ErrorDetail {
            category: self.category(),
            message: self.to_string(),
            retryable: self.is_retryable(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_match_contract() {
        assert_eq!(
            WorkerError::InvalidConfig {
                message: "bad".into()
            }
            .exit_code(),
            2
        );
        assert_eq!(
            WorkerError::RemoteEndpointBlocked {
                url: "http://10.0.0.5:11434".into()
            }
            .exit_code(),
            2
        );
        assert_eq!(
            WorkerError::ContextBudgetExceeded {
                estimated_tokens: 5000,
                budget_tokens: 3000,
            }
            .exit_code(),
            2
        );
        assert_eq!(
            WorkerError::OllamaUnavailable {
                message: "connection refused".into()
            }
            .exit_code(),
            3
        );
        assert_eq!(
            WorkerError::ProtocolError {
                message: "invalid json".into()
            }
            .exit_code(),
            3
        );
        assert_eq!(
            WorkerError::ModelNotFound {
                model: "missing:3b".into()
            }
            .exit_code(),
            4
        );
        assert_eq!(WorkerError::Timeout { timeout_secs: 120 }.exit_code(), 5);
        assert_eq!(
            WorkerError::ValidationFailed {
                message: "empty output".into()
            }
            .exit_code(),
            6
        );
        assert_eq!(WorkerError::Cancelled.exit_code(), 130);
        assert_eq!(
            WorkerError::Internal {
                message: "unexpected".into()
            }
            .exit_code(),
            1
        );
    }

    #[test]
    fn retryable_classification_is_conservative() {
        assert!(
            WorkerError::OllamaUnavailable {
                message: "refused".into()
            }
            .is_retryable()
        );
        assert!(
            WorkerError::HttpStatus {
                status: 503,
                message: "busy".into()
            }
            .is_retryable()
        );
        assert!(
            !WorkerError::HttpStatus {
                status: 400,
                message: "bad request".into()
            }
            .is_retryable()
        );
        assert!(!WorkerError::Timeout { timeout_secs: 60 }.is_retryable());
        assert!(
            !WorkerError::ModelNotFound {
                model: "qwen2.5-coder:3b".into()
            }
            .is_retryable()
        );
    }

    #[test]
    fn error_detail_serializes_with_snake_case_category() {
        let detail = WorkerError::ModelNotFound {
            model: "qwen2.5-coder:3b".into(),
        }
        .to_detail();

        let json = serde_json::to_value(&detail).expect("ErrorDetail should serialize to JSON");
        assert_eq!(json["category"], "model_not_found");
        assert_eq!(json["retryable"], false);
        assert!(
            json["message"]
                .as_str()
                .expect("message should be a string")
                .contains("qwen2.5-coder:3b")
        );
    }
}
