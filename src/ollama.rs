//! Native Ollama HTTP API client and strongly typed wire models.
//!
//! Targets Ollama's native endpoints (`/api/version`, `/api/tags`, `/api/ps`,
//! `/api/chat`) to retain nanosecond inference metrics and per-request options.

use std::future::Future;

use reqwest::{Client, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::config::{InferenceOptions, WorkerConfig};
use crate::error::{WorkerError, WorkerResult};

const MAX_ERROR_BODY_LEN: usize = 256;

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
struct TagsResponse {
    #[serde(default)]
    models: Vec<InstalledModel>,
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
struct PsResponse {
    #[serde(default)]
    models: Vec<RunningModel>,
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

/// Per-request Ollama `options` object mapped from `InferenceOptions`.
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
    pub options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl ChatRequest {
    /// Builds a non-streaming, stateless `/api/chat` request with optional system prompt.
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
            stream: false,
            options: (!ollama_opts.is_empty()).then_some(ollama_opts),
            keep_alive: inference_options.keep_alive.clone(),
        }
    }
}

/// Response body from non-streaming `POST /api/chat`.
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

/// Error JSON format returned by Ollama on 4xx/5xx responses.
#[derive(Debug, Deserialize)]
struct OllamaErrorBody {
    error: Option<String>,
}

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
}

/// Production HTTP implementation of [`OllamaClient`] backed by `reqwest`.
#[derive(Debug, Clone)]
pub struct HttpOllamaClient {
    config: WorkerConfig,
    http: Client,
}

impl HttpOllamaClient {
    /// Constructs a new HTTP client using the validated [`WorkerConfig`].
    pub fn new(config: WorkerConfig) -> WorkerResult<Self> {
        let http = Client::builder()
            .timeout(config.timeout())
            .build()
            .map_err(|err| WorkerError::Internal {
                message: format!("failed to initialize HTTP client: {err}"),
            })?;

        Ok(Self { config, http })
    }

    /// Returns a reference to the underlying validated configuration.
    #[must_use]
    pub const fn config(&self) -> &WorkerConfig {
        &self.config
    }

    async fn send_json<Req, Resp>(
        &self,
        method: Method,
        api_path: &str,
        body: Option<&Req>,
        requested_model: Option<&str>,
    ) -> WorkerResult<Resp>
    where
        Req: Serialize + Sync,
        Resp: DeserializeOwned,
    {
        let url = self.config.endpoint_url(api_path)?;
        tracing::debug!(method = %method, url = %url, "sending request to Ollama");

        let mut builder = self.http.request(method, url);
        if let Some(payload) = body {
            builder = builder.json(payload);
        }

        let response = builder
            .send()
            .await
            .map_err(|err| self.map_transport_error(err))?;

        let status = response.status();
        let raw_bytes = response
            .bytes()
            .await
            .map_err(|err| self.map_transport_error(err))?;

        if !status.is_success() {
            return Err(map_http_error(status, &raw_bytes, requested_model));
        }

        serde_json::from_slice::<Resp>(&raw_bytes).map_err(|err| WorkerError::ProtocolError {
            message: format!("failed to parse JSON from {api_path}: {err}"),
        })
    }

    fn map_transport_error(&self, err: reqwest::Error) -> WorkerError {
        if err.is_timeout() {
            WorkerError::Timeout {
                timeout_secs: self.config.timeout().as_secs(),
            }
        } else {
            WorkerError::OllamaUnavailable {
                message: err.to_string(),
            }
        }
    }
}

impl OllamaClient for HttpOllamaClient {
    async fn version(&self) -> WorkerResult<VersionResponse> {
        self.send_json::<(), VersionResponse>(Method::GET, "/api/version", None, None)
            .await
    }

    async fn list_models(&self) -> WorkerResult<Vec<InstalledModel>> {
        let resp: TagsResponse = self
            .send_json::<(), TagsResponse>(Method::GET, "/api/tags", None, None)
            .await?;
        Ok(resp.models)
    }

    async fn list_running_models(&self) -> WorkerResult<Vec<RunningModel>> {
        let resp: PsResponse = self
            .send_json::<(), PsResponse>(Method::GET, "/api/ps", None, None)
            .await?;
        Ok(resp.models)
    }

    async fn chat(&self, request: &ChatRequest) -> WorkerResult<ChatResponse> {
        self.send_json(
            Method::POST,
            "/api/chat",
            Some(request),
            Some(&request.model),
        )
        .await
    }
}

fn map_http_error(
    status: StatusCode,
    raw_bytes: &[u8],
    requested_model: Option<&str>,
) -> WorkerError {
    let message = extract_error_message(raw_bytes);
    if status == StatusCode::NOT_FOUND
        && let Some(model) = requested_model
    {
        return WorkerError::ModelNotFound {
            model: model.to_owned(),
        };
    }

    WorkerError::HttpStatus {
        status: status.as_u16(),
        message,
    }
}

fn extract_error_message(raw_bytes: &[u8]) -> String {
    if let Ok(parsed) = serde_json::from_slice::<OllamaErrorBody>(raw_bytes)
        && let Some(msg) = parsed.error
        && !msg.trim().is_empty()
    {
        return msg;
    }

    let raw = String::from_utf8_lossy(raw_bytes);
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "empty error response body".into();
    }
    if trimmed.chars().count() > MAX_ERROR_BODY_LEN {
        let truncated: String = trimmed.chars().take(MAX_ERROR_BODY_LEN).collect();
        format!("{truncated}...")
    } else {
        trimmed.to_owned()
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
        };
        let req = ChatRequest::stateless("qwen2.5-coder:3b", Some("   "), "Hello", &opts);
        let value = serde_json::to_value(&req).expect("ChatRequest should serialize");
        assert_eq!(value["messages"].as_array().map(Vec::len), Some(1));
        assert_eq!(value["messages"][0]["role"], "user");
        assert!(value.get("options").is_none());
    }

    #[test]
    fn deserializes_native_ollama_chat_response_with_metrics() {
        let payload = r#"{
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

        let resp: ChatResponse =
            serde_json::from_str(payload).expect("valid ChatResponse should parse");
        assert_eq!(resp.model, "qwen2.5-coder:3b");
        assert!(resp.done);
        assert_eq!(resp.done_reason.as_deref(), Some("stop"));
        assert_eq!(resp.prompt_eval_count, Some(41));
        assert_eq!(resp.eval_count, Some(25));
    }

    #[test]
    fn model_matching_handles_explicit_and_latest_tags() {
        assert!(model_matches("qwen2.5-coder:3b", "qwen2.5-coder:3b"));
        assert!(!model_matches("qwen2.5-coder:7b", "qwen2.5-coder:3b"));
        assert!(model_matches("qwen2.5-coder:latest", "qwen2.5-coder"));
        assert!(!model_matches("qwen2.5-coder:3b", "qwen2.5-coder"));
    }

    #[test]
    fn maps_404_with_model_context_to_model_not_found() {
        let body = br#"{"error":"model 'missing:3b' not found"}"#;
        let err = map_http_error(StatusCode::NOT_FOUND, body, Some("missing:3b"));
        assert!(matches!(&err, WorkerError::ModelNotFound { model } if model == "missing:3b"));
        assert_eq!(err.exit_code(), 4);
    }

    #[test]
    fn extracts_json_error_or_falls_back_to_plain_text() {
        assert_eq!(
            extract_error_message(br#"{"error":"server overloaded"}"#),
            "server overloaded"
        );
        assert_eq!(extract_error_message(b"Bad Gateway"), "Bad Gateway");
        assert_eq!(extract_error_message(b"   "), "empty error response body");
    }
}
