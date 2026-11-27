//! Core library for `ollama-worker`.
//!
//! Separates reusable Ollama client, task orchestration, prompt construction,
//! metrics calculation, and non-executing output validation logic from the CLI
//! entry point.

pub mod client;
pub mod config;
pub mod error;
pub mod metrics;
pub mod prompt;
pub mod task;
pub mod validation;

pub use client::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, GenerateRequest, GenerateResponse,
    HttpOllamaClient, InstalledModel, ModelDetails, OllamaClient, OllamaOptions, RunningModel,
    StreamedChatResponse, VersionResponse, model_matches,
};
pub use config::{InferenceOptions, WorkerConfig};
pub use error::{ErrorCategory, ErrorDetail, WorkerError, WorkerResult};
pub use metrics::InferenceMetrics;
pub use prompt::{BuiltPrompt, PromptBuilder, detect_potential_secret, estimate_tokens};
pub use task::{
    DEFAULT_SYSTEM_PROMPT, DryRunReport, HealthReport, OutputExpectation, OutputFormat,
    SCHEMA_VERSION, TaskExecutor, TaskRequest, TaskResult, TaskStatus, TaskType, ValidationCheck,
    ValidationReport,
};
pub use validation::{OutputValidator, ValidationOutcome};

/// Default loopback base URL for a local Ollama server.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

/// Default per-request timeout in seconds for CPU-bound local inference.
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Returns the semantic version of the `ollama_worker` crate.
#[must_use]
pub const fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_use_loopback_and_bounded_timeout() {
        assert_eq!(DEFAULT_OLLAMA_BASE_URL, "http://127.0.0.1:11434");
        const { assert!(DEFAULT_TIMEOUT_SECS > 0) };
    }

    #[test]
    fn crate_version_is_non_empty() {
        assert!(!crate_version().is_empty());
    }
}
