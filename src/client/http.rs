//! Production `reqwest`-backed HTTP client for Native Ollama endpoints.

use std::time::Instant;

use reqwest::{Client, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::client::OllamaClient;
use crate::client::types::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, GenerateRequest, GenerateResponse,
    InstalledModel, PsResponse, RunningModel, StreamedChatResponse, TagsResponse, VersionResponse,
};
use crate::config::WorkerConfig;
use crate::error::{WorkerError, WorkerResult};

const MAX_ERROR_BODY_LEN: usize = 256;

/// Error JSON format returned by Ollama on 4xx/5xx responses.
#[derive(Debug, Deserialize)]
struct OllamaErrorBody {
    error: Option<String>,
}

/// Streaming NDJSON line emitted by `POST /api/chat` when `stream: true`.
#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    created_at: Option<String>,
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
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
        let non_streaming = request.clone().with_stream(false);
        self.send_json(
            Method::POST,
            "/api/chat",
            Some(&non_streaming),
            Some(&request.model),
        )
        .await
    }

    async fn chat_stream(&self, request: &ChatRequest) -> WorkerResult<StreamedChatResponse> {
        let url = self.config.endpoint_url("/api/chat")?;
        let streaming_request = request.clone().with_stream(true);
        tracing::debug!(url = %url, model = %streaming_request.model, "starting streaming chat request");

        let started = Instant::now();
        let mut response = self
            .http
            .post(url)
            .json(&streaming_request)
            .send()
            .await
            .map_err(|err| self.map_transport_error(err))?;

        let status = response.status();
        if !status.is_success() {
            let raw_bytes = response
                .bytes()
                .await
                .map_err(|err| self.map_transport_error(err))?;
            return Err(map_http_error(status, &raw_bytes, Some(&request.model)));
        }

        let mut assembler = StreamAssembler::new(&request.model, started);
        let mut line_buffer = Vec::<u8>::new();

        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|err| self.map_transport_error(err))?
        {
            line_buffer.extend_from_slice(&chunk);
            while let Some(newline_idx) = line_buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = line_buffer.drain(..=newline_idx).collect();
                assembler.ingest_line(&line)?;
            }
        }

        if !line_buffer.is_empty() {
            assembler.ingest_line(&line_buffer)?;
        }

        assembler.finish()
    }

    async fn generate(&self, request: &GenerateRequest) -> WorkerResult<GenerateResponse> {
        self.send_json(
            Method::POST,
            "/api/generate",
            Some(request),
            Some(&request.model),
        )
        .await
    }
}

/// Incremental NDJSON stream assembler tracking TTFT and verifying terminal `done: true`.
struct StreamAssembler {
    model: String,
    created_at: Option<String>,
    content: String,
    done: bool,
    done_reason: Option<String>,
    total_duration: Option<u64>,
    load_duration: Option<u64>,
    prompt_eval_count: Option<u64>,
    prompt_eval_duration: Option<u64>,
    eval_count: Option<u64>,
    eval_duration: Option<u64>,
    started: Instant,
    ttft_ms: Option<u64>,
}

impl StreamAssembler {
    fn new(model: &str, started: Instant) -> Self {
        Self {
            model: model.to_owned(),
            created_at: None,
            content: String::new(),
            done: false,
            done_reason: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
            started,
            ttft_ms: None,
        }
    }

    fn ingest_line(&mut self, raw_line: &[u8]) -> WorkerResult<()> {
        let line_str = std::str::from_utf8(raw_line).map_err(|err| WorkerError::ProtocolError {
            message: format!("streaming NDJSON chunk is not valid UTF-8: {err}"),
        })?;
        let trimmed = line_str.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let chunk: ChatStreamChunk =
            serde_json::from_str(trimmed).map_err(|err| WorkerError::ProtocolError {
                message: format!("failed to parse streaming NDJSON chunk from /api/chat: {err}"),
            })?;

        if let Some(model) = chunk.model {
            self.model = model;
        }
        if let Some(created_at) = chunk.created_at {
            self.created_at = Some(created_at);
        }
        if let Some(msg) = chunk.message
            && !msg.content.is_empty()
        {
            if self.ttft_ms.is_none() {
                self.ttft_ms =
                    Some(u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX));
            }
            self.content.push_str(&msg.content);
        }

        if chunk.done {
            self.done = true;
            self.done_reason = chunk.done_reason;
            self.total_duration = chunk.total_duration;
            self.load_duration = chunk.load_duration;
            self.prompt_eval_count = chunk.prompt_eval_count;
            self.prompt_eval_duration = chunk.prompt_eval_duration;
            self.eval_count = chunk.eval_count;
            self.eval_duration = chunk.eval_duration;
        }

        Ok(())
    }

    fn finish(self) -> WorkerResult<StreamedChatResponse> {
        if !self.done {
            return Err(WorkerError::ProtocolError {
                message: "Ollama streaming response ended prematurely before final done=true chunk"
                    .into(),
            });
        }

        Ok(StreamedChatResponse {
            response: ChatResponse {
                model: self.model,
                created_at: self.created_at,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: self.content,
                },
                done: true,
                done_reason: self.done_reason,
                total_duration: self.total_duration,
                load_duration: self.load_duration,
                prompt_eval_count: self.prompt_eval_count,
                prompt_eval_duration: self.prompt_eval_duration,
                eval_count: self.eval_count,
                eval_duration: self.eval_duration,
            },
            ttft_ms: self.ttft_ms,
        })
    }
}

pub(crate) fn map_http_error(
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

pub(crate) fn extract_error_message(raw_bytes: &[u8]) -> String {
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

    #[test]
    fn stream_assembler_combines_ndjson_chunks_and_records_ttft() {
        let mut assembler = StreamAssembler::new("qwen2.5-coder:3b", Instant::now());
        assembler
            .ingest_line(
                br#"{"model":"qwen2.5-coder:3b","message":{"role":"assistant","content":"def "},"done":false}"#,
            )
            .expect("chunk 1 should parse");
        assembler
            .ingest_line(
                br#"{"model":"qwen2.5-coder:3b","message":{"role":"assistant","content":"ok(): pass"},"done":false}"#,
            )
            .expect("chunk 2 should parse");
        assembler
            .ingest_line(
                br#"{"model":"qwen2.5-coder:3b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":500000000,"prompt_eval_count":20,"eval_count":6}"#,
            )
            .expect("final chunk should parse");

        let assembled = assembler.finish().expect("completed stream should succeed");
        assert_eq!(assembled.response.message.content, "def ok(): pass");
        assert_eq!(assembled.response.done_reason.as_deref(), Some("stop"));
        assert_eq!(assembled.response.eval_count, Some(6));
        assert!(assembled.ttft_ms.is_some());
    }

    #[test]
    fn stream_assembler_rejects_premature_eof_without_done_chunk() {
        let mut assembler = StreamAssembler::new("qwen2.5-coder:3b", Instant::now());
        assembler
            .ingest_line(
                br#"{"model":"qwen2.5-coder:3b","message":{"role":"assistant","content":"partial"},"done":false}"#,
            )
            .expect("chunk 1 should parse");

        let err = assembler
            .finish()
            .expect_err("truncated stream must return ProtocolError");
        assert!(matches!(err, WorkerError::ProtocolError { .. }));
        assert_eq!(err.exit_code(), 3);
    }
}
