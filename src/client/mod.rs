//! Native Ollama HTTP API client module.
//!
//! Targets Ollama's native endpoints (`/api/version`, `/api/tags`, `/api/ps`,
//! `/api/chat`, `/api/generate`) to retain nanosecond inference metrics,
//! streaming TTFT measurements, and per-request inference controls.

pub mod http;
pub mod types;

use std::future::Future;

use crate::error::WorkerResult;

pub use http::HttpOllamaClient;
pub use types::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, GenerateRequest, GenerateResponse,
    InstalledModel, ModelDetails, OllamaOptions, RunningModel, StreamedChatResponse,
    VersionResponse,
};

/// Checks whether an installed model name from `/api/tags` matches a requested model.
///
/// Ollama normalizes untagged model references (e.g. `llama3.2`) to `llama3.2:latest`.
#[must_use]
pub fn model_matches(installed_name: &str, requested: &str) -> bool {
    if installed_name == requested {
        return true;
    }
    if !requested.contains(':') {
        return installed_name
            .strip_suffix(":latest")
            .is_some_and(|base| base == requested);
    }
    false
}

/// Abstract client interface for Ollama operations, enabling offline unit testing.
pub trait OllamaClient: Send + Sync {
    fn version(&self) -> impl Future<Output = WorkerResult<VersionResponse>> + Send;

    fn list_models(&self) -> impl Future<Output = WorkerResult<Vec<InstalledModel>>> + Send;

    fn list_running_models(&self) -> impl Future<Output = WorkerResult<Vec<RunningModel>>> + Send;

    fn chat(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = WorkerResult<ChatResponse>> + Send;

    fn chat_stream(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = WorkerResult<StreamedChatResponse>> + Send;

    fn generate(
        &self,
        request: &GenerateRequest,
    ) -> impl Future<Output = WorkerResult<GenerateResponse>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_matching_handles_explicit_and_latest_tags() {
        assert!(model_matches("qwen2.5-coder:3b", "qwen2.5-coder:3b"));
        assert!(!model_matches("qwen2.5-coder:7b", "qwen2.5-coder:3b"));
        assert!(model_matches("qwen2.5-coder:latest", "qwen2.5-coder"));
        assert!(!model_matches("qwen2.5-coder:3b", "qwen2.5-coder"));
    }
}
