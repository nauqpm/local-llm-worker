//! Wire models (DTOs) for the Native Ollama HTTP API.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::InferenceOptions;

/// Response from `GET /api/version`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionResponse {
    pub version: String,
}

/// Optional metadata describing a local model in `/api/tags`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub families: Option<Vec<String>>,
    #[serde(default)]
    pub parameter_size: Option<String>,
    #[serde(default)]
    pub quantization_level: Option<String>,
}

/// Single installed model entry from `GET /api/tags`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModel {
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub modified_at: Option<String>,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub details: Option<ModelDetails>,
}

/// Wire payload for `GET /api/tags`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct TagsResponse {
    #[serde(default)]
    pub models: Vec<InstalledModel>,
}

/// Single resident model entry from `GET /api/ps`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningModel {
    pub name: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
    #[serde(default)]
    pub context_length: Option<u32>,
}

/// Wire payload for `GET /api/ps`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct PsResponse {
    #[serde(default)]
    pub models: Vec<RunningModel>,
}

/// Role in an Ollama `/api/chat` message list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Single message in an Ollama `/api/chat` request or response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }
}

/// Per-request Ollama `options` object mapped from [`InferenceOptions`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

impl OllamaOptions {
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.num_ctx.is_none()
            && self.num_predict.is_none()
            && self.seed.is_none()
    }
}

impl From<&InferenceOptions> for OllamaOptions {
    fn from(opts: &InferenceOptions) -> Self {
        Self {
            temperature: opts.temperature,
            num_ctx: opts.num_ctx,
            num_predict: opts.max_tokens,
            seed: opts.seed,
        }
    }
}

/// Request body for stateless `POST /api/chat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl ChatRequest {
    /// Builds a stateless `/api/chat` request with optional system prompt.
    #[must_use]
    pub fn stateless(
        model: impl Into<String>,
        system_prompt: Option<&str>,
        user_prompt: impl Into<String>,
        inference_options: &InferenceOptions,
    ) -> Self {
        let mut messages = Vec::with_capacity(2);
        if let Some(system) = system_prompt.map(str::trim).filter(|s| !s.is_empty()) {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(user_prompt));

        let ollama_opts = OllamaOptions::from(inference_options);
        Self {
            model: model.into(),
            messages,
            stream: inference_options.stream,
            format: None,
            options: (!ollama_opts.is_empty()).then_some(ollama_opts),
            keep_alive: inference_options.keep_alive.clone(),
        }
    }

    /// Attaches a native Ollama output format constraint (`"json"` or a JSON Schema object).
    #[must_use]
    pub fn with_format(mut self, format: Value) -> Self {
        self.format = Some(format);
        self
    }

    /// Explicitly overrides the `stream` flag.
    #[must_use]
    pub const fn with_stream(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }
}

/// Response body from non-streaming or assembled streaming `POST /api/chat`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    #[serde(default)]
    pub created_at: Option<String>,
    pub message: ChatMessage,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

/// Assembled streaming chat result including client-measured Time to First Token (TTFT).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamedChatResponse {
    pub response: ChatResponse,
    pub ttft_ms: Option<u64>,
}

/// Request body for `POST /api/generate` (used for endpoint benchmarking and unload calls).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl GenerateRequest {
    /// Builds a non-streaming `/api/generate` request.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        system_prompt: Option<&str>,
        prompt: impl Into<String>,
        inference_options: &InferenceOptions,
    ) -> Self {
        let ollama_opts = OllamaOptions::from(inference_options);
        Self {
            model: model.into(),
            prompt: prompt.into(),
            system: system_prompt
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToOwned::to_owned),
            stream: false,
            format: None,
            options: (!ollama_opts.is_empty()).then_some(ollama_opts),
            keep_alive: inference_options.keep_alive.clone(),
        }
    }

    /// Builds an Ollama unload request (`prompt: ""`, `keep_alive: "0"`), used for cold-start benchmarks.
    #[must_use]
    pub fn unload(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: String::new(),
            system: None,
            stream: false,
            format: None,
            options: None,
            keep_alive: Some("0".to_owned()),
        }
    }
}

/// Response body from non-streaming `POST /api/generate`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    #[serde(default)]
    pub created_at: Option<String>,
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u64>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl From<GenerateResponse> for ChatResponse {
    fn from(generated: GenerateResponse) -> Self {
        Self {
            model: generated.model,
            created_at: generated.created_at,
            message: ChatMessage {
                role: ChatRole::Assistant,
                content: generated.response,
            },
            done: generated.done,
            done_reason: generated.done_reason,
            total_duration: generated.total_duration,
            load_duration: generated.load_duration,
            prompt_eval_count: generated.prompt_eval_count,
            prompt_eval_duration: generated.prompt_eval_duration,
            eval_count: generated.eval_count,
            eval_duration: generated.eval_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stateless_chat_request_omits_unset_fields_and_num_ctx_by_default() {
        let opts = InferenceOptions::default();
        let req = ChatRequest::stateless(
            "qwen2.5-coder:3b",
            Some("Return only Python code."),
            "Write is_prime(n).",
            &opts,
        );

        let value = serde_json::to_value(&req).expect("ChatRequest should serialize");
        assert_eq!(value["model"], "qwen2.5-coder:3b");
        assert_eq!(value["stream"], false);
        assert_eq!(value["messages"].as_array().map(Vec::len), Some(2));
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][1]["role"], "user");
        assert_eq!(value["options"]["num_predict"], 512);
        assert!(value["options"].get("num_ctx").is_none());
        assert!(value.get("keep_alive").is_none());
        assert!(value.get("tools").is_none());
    }

    #[test]
    fn stateless_chat_request_skips_blank_system_prompt() {
        let opts = InferenceOptions {
            temperature: None,
            num_ctx: None,
            max_tokens: None,
            seed: None,
            keep_alive: None,
            stream: false,
        };
        let req = ChatRequest::stateless("qwen2.5-coder:3b", Some("   "), "Hello", &opts);
        let value = serde_json::to_value(&req).expect("ChatRequest should serialize");
        assert_eq!(value["messages"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["messages"][0]["role"], "user");
        assert!(value.get("options").is_none());
    }

    #[test]
    fn deserializes_native_ollama_chat_and_generate_responses() {
        let chat_payload = r#"{
            "model": "qwen2.5-coder:3b",
            "created_at": "2026-11-27T06:00:00Z",
            "message": {
                "role": "assistant",
                "content": "def is_prime(n: int) -> bool:\n    return n > 1"
            },
            "done": true,
            "done_reason": "stop",
            "total_duration": 1500000000,
            "load_duration": 200000000,
            "prompt_eval_count": 41,
            "prompt_eval_duration": 300000000,
            "eval_count": 25,
            "eval_duration": 1000000000
        }"#;

        let chat_resp: ChatResponse =
            serde_json::from_str(chat_payload).expect("valid ChatResponse should parse");
        assert_eq!(chat_resp.model, "qwen2.5-coder:3b");
        assert!(chat_resp.done);
        assert_eq!(chat_resp.prompt_eval_count, Some(41));

        let gen_payload = r#"{
            "model": "qwen2.5-coder:3b",
            "response": "return n > 1",
            "done": true,
            "done_reason": "stop",
            "total_duration": 900000000,
            "prompt_eval_count": 20,
            "eval_count": 10
        }"#;
        let gen_resp: GenerateResponse =
            serde_json::from_str(gen_payload).expect("valid GenerateResponse should parse");
        let converted: ChatResponse = gen_resp.into();
        assert_eq!(converted.message.content, "return n > 1");
        assert_eq!(converted.prompt_eval_count, Some(20));
    }
}
