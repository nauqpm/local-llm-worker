//! Task request/result contracts and the core [`TaskExecutor`] orchestrator.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::InferenceOptions;
use crate::error::{ErrorDetail, WorkerError, WorkerResult, exit_codes};
use crate::metrics::InferenceMetrics;
use crate::ollama::{ChatRequest, InstalledModel, OllamaClient, RunningModel, model_matches};

/// Current structured schema version emitted in task results.
pub const SCHEMA_VERSION: &str = "1";

/// Minimal, token-efficient system prompt used when a task does not specify one.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise, precise coding assistant. Follow instructions strictly and return only the requested output without unnecessary commentary.";

/// Terminal execution status of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Success,
    ValidationFailed,
    Timeout,
    Cancelled,
    Error,
}

/// Minimal strongly typed task request for single-shot local inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_id: String,
    pub model: String,
    pub instruction: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub options: InferenceOptions,
}

impl TaskRequest {
    /// Creates a new [`TaskRequest`] with default [`InferenceOptions`].
    #[must_use]
    pub fn new(
        task_id: impl Into<String>,
        model: impl Into<String>,
        instruction: impl Into<String>,
    ) -> Self {
        Self {
            task_id: task_id.into(),
            model: model.into(),
            instruction: instruction.into(),
            system_prompt: None,
            options: InferenceOptions::default(),
        }
    }

    /// Validates required fields and per-request inference options.
    pub fn validate(&self) -> WorkerResult<()> {
        if self.task_id.trim().is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "task_id must not be empty".into(),
            });
        }
        if self.model.trim().is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "model must not be empty".into(),
            });
        }
        if self.instruction.trim().is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "instruction must not be empty".into(),
            });
        }
        self.options.validate()
    }
}

/// Machine-readable result emitted on `stdout` for every `run` invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub schema_version: String,
    pub task_id: String,
    pub status: TaskStatus,
    pub model: String,
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    pub metrics: Option<InferenceMetrics>,
    pub error: Option<ErrorDetail>,
}

impl TaskResult {
    /// Builds a structured failure result from a [`WorkerError`].
    #[must_use]
    pub fn from_error(
        task_id: impl Into<String>,
        model: impl Into<String>,
        error: &WorkerError,
    ) -> Self {
        let status = match error {
            WorkerError::Timeout { .. } => TaskStatus::Timeout,
            WorkerError::Cancelled => TaskStatus::Cancelled,
            WorkerError::ValidationFailed { .. } => TaskStatus::ValidationFailed,
            _ => TaskStatus::Error,
        };

        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            task_id: task_id.into(),
            status,
            model: model.into(),
            output: None,
            raw_output: None,
            done_reason: None,
            metrics: None,
            error: Some(error.to_detail()),
        }
    }

    /// Returns the CLI process exit code corresponding to this task result.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self.status {
            TaskStatus::Success => exit_codes::SUCCESS,
            _ => self
                .error
                .as_ref()
                .map_or(exit_codes::INTERNAL_ERROR, |err| err.category.exit_code()),
        }
    }
}

/// Structured server health snapshot returned by the `health` command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReport {
    pub status: String,
    pub base_url: String,
    pub ollama_version: String,
    pub running_models: Vec<RunningModel>,
}

/// Coordinates validation, model verification, Ollama inference, and metrics.
#[derive(Debug, Clone)]
pub struct TaskExecutor<C> {
    client: C,
    verify_model: bool,
}

impl<C: OllamaClient> TaskExecutor<C> {
    /// Creates a new executor backed by any [`OllamaClient`] implementation.
    #[must_use]
    pub const fn new(client: C, verify_model: bool) -> Self {
        Self {
            client,
            verify_model,
        }
    }

    /// Checks Ollama liveness (`/api/version`) and currently loaded models (`/api/ps`).
    pub async fn health(&self, base_url: &str) -> WorkerResult<HealthReport> {
        let version = self.client.version().await?;
        let running_models = self.client.list_running_models().await?;

        Ok(HealthReport {
            status: "ok".to_owned(),
            base_url: base_url.to_owned(),
            ollama_version: version.version,
            running_models,
        })
    }

    /// Lists models installed on the Ollama server (`/api/tags`).
    pub async fn models(&self) -> WorkerResult<Vec<InstalledModel>> {
        self.client.list_models().await
    }

    /// Executes a single task and always produces a structured [`TaskResult`].
    pub async fn execute(&self, request: &TaskRequest) -> TaskResult {
        if let Err(err) = request.validate() {
            return TaskResult::from_error(&request.task_id, &request.model, &err);
        }

        let model = request.model.trim();
        let task_id = request.task_id.trim();

        if self.verify_model {
            match self.client.list_models().await {
                Ok(installed) => {
                    if !installed.iter().any(|m| model_matches(&m.name, model)) {
                        let err = WorkerError::ModelNotFound {
                            model: model.to_owned(),
                        };
                        return TaskResult::from_error(task_id, model, &err);
                    }
                }
                Err(err) => return TaskResult::from_error(task_id, model, &err),
            }
        }

        let system_prompt = request
            .system_prompt
            .as_deref()
            .unwrap_or(DEFAULT_SYSTEM_PROMPT);

        let chat_request = ChatRequest::stateless(
            model,
            Some(system_prompt),
            request.instruction.trim(),
            &request.options,
        );

        let started = Instant::now();
        let response = match self.client.chat(&chat_request).await {
            Ok(resp) => resp,
            Err(err) => return TaskResult::from_error(task_id, model, &err),
        };
        let wall_clock = started.elapsed();
        let metrics = InferenceMetrics::from_chat_response(&response, wall_clock);

        let raw_content = response.message.content;
        let trimmed = raw_content.trim();
        if trimmed.is_empty() {
            let err = WorkerError::ValidationFailed {
                message: "model returned an empty response".into(),
            };
            return TaskResult {
                schema_version: SCHEMA_VERSION.to_owned(),
                task_id: task_id.to_owned(),
                status: TaskStatus::ValidationFailed,
                model: response.model,
                output: None,
                raw_output: Some(raw_content),
                done_reason: response.done_reason,
                metrics: Some(metrics),
                error: Some(err.to_detail()),
            };
        }

        TaskResult {
            schema_version: SCHEMA_VERSION.to_owned(),
            task_id: task_id.to_owned(),
            status: TaskStatus::Success,
            model: response.model,
            output: Some(trimmed.to_owned()),
            raw_output: Some(raw_content),
            done_reason: response.done_reason,
            metrics: Some(metrics),
            error: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::error::ErrorCategory;
    use crate::ollama::{ChatMessage, ChatResponse, ChatRole, VersionResponse};

    struct FakeOllamaClient {
        installed: Vec<InstalledModel>,
        chat_result: Option<Result<ChatResponse, &'static str>>,
        chat_calls: AtomicUsize,
    }

    impl FakeOllamaClient {
        fn with_model(name: &str, response_content: &str) -> Self {
            Self {
                installed: vec![InstalledModel {
                    name: name.to_owned(),
                    model: Some(name.to_owned()),
                    modified_at: None,
                    size: 2_400_000_000,
                    digest: None,
                    details: None,
                }],
                chat_result: Some(Ok(ChatResponse {
                    model: name.to_owned(),
                    created_at: None,
                    message: ChatMessage {
                        role: ChatRole::Assistant,
                        content: response_content.to_owned(),
                    },
                    done: true,
                    done_reason: Some("stop".into()),
                    total_duration: Some(1_000_000_000),
                    load_duration: Some(100_000_000),
                    prompt_eval_count: Some(41),
                    prompt_eval_duration: Some(200_000_000),
                    eval_count: Some(30),
                    eval_duration: Some(600_000_000),
                })),
                chat_calls: AtomicUsize::new(0),
            }
        }
    }

    impl OllamaClient for FakeOllamaClient {
        async fn version(&self) -> WorkerResult<VersionResponse> {
            Ok(VersionResponse {
                version: "0.6.0".into(),
            })
        }

        async fn list_models(&self) -> WorkerResult<Vec<InstalledModel>> {
            Ok(self.installed.clone())
        }

        async fn list_running_models(&self) -> WorkerResult<Vec<RunningModel>> {
            Ok(vec![])
        }

        async fn chat(&self, _request: &ChatRequest) -> WorkerResult<ChatResponse> {
            self.chat_calls.fetch_add(1, Ordering::SeqCst);
            match &self.chat_result {
                Some(Ok(resp)) => Ok(resp.clone()),
                Some(Err("timeout")) => Err(WorkerError::Timeout { timeout_secs: 30 }),
                _ => Err(WorkerError::OllamaUnavailable {
                    message: "connection refused".into(),
                }),
            }
        }
    }

    #[tokio::test]
    async fn executes_valid_task_and_returns_structured_metrics() {
        let client = FakeOllamaClient::with_model(
            "qwen2.5-coder:3b",
            "  def is_prime(n: int) -> bool:\n      return n > 1  ",
        );
        let executor = TaskExecutor::new(client, true);
        let req = TaskRequest::new("task-001", "qwen2.5-coder:3b", "Write is_prime in Python");

        let result = executor.execute(&req).await;
        assert_eq!(result.status, TaskStatus::Success);
        assert_eq!(result.exit_code(), 0);
        assert_eq!(result.schema_version, SCHEMA_VERSION);
        assert_eq!(
            result.output.as_deref(),
            Some("def is_prime(n: int) -> bool:\n      return n > 1")
        );
        assert!(result.error.is_none());
        let metrics = result.metrics.expect("metrics should be populated");
        assert_eq!(metrics.prompt_tokens, 41);
        assert_eq!(metrics.generated_tokens, 30);
        assert_eq!(metrics.generation_tokens_per_second, 50.0);
    }

    #[tokio::test]
    async fn fails_fast_when_model_is_missing_without_calling_chat() {
        let client = FakeOllamaClient::with_model("qwen2.5-coder:3b", "ok");
        let executor = TaskExecutor::new(client, true);
        let req = TaskRequest::new("task-002", "unknown:7b", "Hello");

        let result = executor.execute(&req).await;
        assert_eq!(result.status, TaskStatus::Error);
        assert_eq!(result.exit_code(), 4);
        assert_eq!(
            result.error.as_ref().map(|e| e.category),
            Some(ErrorCategory::ModelNotFound)
        );
        assert_eq!(executor.client.chat_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn distinguishes_empty_output_as_validation_failed_preserving_metrics() {
        let client = FakeOllamaClient::with_model("qwen2.5-coder:3b", "   \n  ");
        let executor = TaskExecutor::new(client, true);
        let req = TaskRequest::new("task-003", "qwen2.5-coder:3b", "Write something");

        let result = executor.execute(&req).await;
        assert_eq!(result.status, TaskStatus::ValidationFailed);
        assert_eq!(result.exit_code(), 6);
        assert!(result.output.is_none());
        assert_eq!(result.raw_output.as_deref(), Some("   \n  "));
        assert!(result.metrics.is_some());
        assert_eq!(
            result.error.as_ref().map(|e| e.category),
            Some(ErrorCategory::ValidationFailed)
        );
    }

    #[tokio::test]
    async fn maps_timeout_and_invalid_input_cleanly() {
        let mut client = FakeOllamaClient::with_model("qwen2.5-coder:3b", "ignored");
        client.chat_result = Some(Err("timeout"));
        let executor = TaskExecutor::new(client, true);

        let timeout_res = executor
            .execute(&TaskRequest::new("task-004", "qwen2.5-coder:3b", "Slow"))
            .await;
        assert_eq!(timeout_res.status, TaskStatus::Timeout);
        assert_eq!(timeout_res.exit_code(), 5);

        let invalid_res = executor
            .execute(&TaskRequest::new("task-005", "qwen2.5-coder:3b", "   "))
            .await;
        assert_eq!(invalid_res.status, TaskStatus::Error);
        assert_eq!(invalid_res.exit_code(), 2);
    }
}
