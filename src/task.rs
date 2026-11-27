//! Task request/result contracts and the core [`TaskExecutor`] orchestrator.

use std::str::FromStr;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::InferenceOptions;
use crate::error::{ErrorDetail, WorkerError, WorkerResult, exit_codes};
use crate::metrics::InferenceMetrics;
use crate::ollama::{ChatRequest, InstalledModel, OllamaClient, RunningModel, model_matches};

/// Current structured schema version emitted in task requests and results.
pub const SCHEMA_VERSION: &str = "1";

/// Minimal, token-efficient system prompt used when a task does not specify one.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a concise, precise coding assistant. Follow instructions strictly and return only the requested output without unnecessary commentary.";

fn default_schema_version() -> String {
    SCHEMA_VERSION.to_owned()
}

fn default_true() -> bool {
    true
}

/// Supported coding task types for scoped local-model delegation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    #[default]
    Freeform,
    GenerateFunction,
    GenerateUnitTest,
    GenerateDocstring,
    ExplainCode,
    RefactorFunction,
    FixBug,
    GenerateJson,
}

impl TaskType {
    /// Returns the default expected output format when `OutputExpectation::format` is `Auto`.
    #[must_use]
    pub const fn default_output_format(self) -> OutputFormat {
        match self {
            Self::GenerateFunction
            | Self::GenerateUnitTest
            | Self::GenerateDocstring
            | Self::RefactorFunction
            | Self::FixBug => OutputFormat::Code,
            Self::GenerateJson => OutputFormat::Json,
            Self::Freeform | Self::ExplainCode => OutputFormat::Text,
        }
    }

    /// Returns the canonical snake_case identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Freeform => "freeform",
            Self::GenerateFunction => "generate_function",
            Self::GenerateUnitTest => "generate_unit_test",
            Self::GenerateDocstring => "generate_docstring",
            Self::ExplainCode => "explain_code",
            Self::RefactorFunction => "refactor_function",
            Self::FixBug => "fix_bug",
            Self::GenerateJson => "generate_json",
        }
    }
}

impl FromStr for TaskType {
    type Err = WorkerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "freeform" => Ok(Self::Freeform),
            "generate_function" => Ok(Self::GenerateFunction),
            "generate_unit_test" => Ok(Self::GenerateUnitTest),
            "generate_docstring" => Ok(Self::GenerateDocstring),
            "explain_code" => Ok(Self::ExplainCode),
            "refactor_function" => Ok(Self::RefactorFunction),
            "fix_bug" => Ok(Self::FixBug),
            "generate_json" => Ok(Self::GenerateJson),
            other => Err(WorkerError::InvalidInput {
                message: format!(
                    "unsupported task_type '{other}'; expected one of: freeform, generate_function, generate_unit_test, generate_docstring, explain_code, refactor_function, fix_bug, generate_json"
                ),
            }),
        }
    }
}

/// Desired output structure for static, non-executing validation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    #[default]
    Auto,
    Code,
    Json,
    Text,
}

/// Static validation expectations attached to a [`TaskRequest`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputExpectation {
    #[serde(default)]
    pub format: OutputFormat,
    #[serde(default = "default_true")]
    pub extract_code_block: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_chars: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_fields: Vec<String>,
}

impl Default for OutputExpectation {
    fn default() -> Self {
        Self {
            format: OutputFormat::Auto,
            extract_code_block: true,
            max_output_chars: None,
            json_schema: None,
            required_fields: Vec::new(),
        }
    }
}

impl OutputExpectation {
    /// Resolves `OutputFormat::Auto` using the task's [`TaskType`].
    #[must_use]
    pub const fn effective_format(&self, task_type: TaskType) -> OutputFormat {
        match self.format {
            OutputFormat::Auto => task_type.default_output_format(),
            explicit => explicit,
        }
    }

    /// Validates expectation constraints before executing inference.
    pub fn validate(&self) -> WorkerResult<()> {
        if let Some(max_chars) = self.max_output_chars
            && max_chars == 0
        {
            return Err(WorkerError::InvalidInput {
                message: "output_expectation.max_output_chars must be >= 1".into(),
            });
        }
        for field in &self.required_fields {
            if field.trim().is_empty() {
                return Err(WorkerError::InvalidInput {
                    message: "output_expectation.required_fields must not contain empty names"
                        .into(),
                });
            }
        }
        Ok(())
    }
}

/// Single static check outcome recorded in [`ValidationReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: Option<String>,
}

impl ValidationCheck {
    #[must_use]
    pub fn pass(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail: None,
        }
    }

    #[must_use]
    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail: Some(detail.into()),
        }
    }
}

/// Aggregated non-executing output validation summary returned in [`TaskResult`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationReport {
    pub passed: bool,
    pub extracted_code_block: bool,
    pub checks: Vec<ValidationCheck>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

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

/// Versioned, strongly typed task request for single-shot local inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskRequest {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub task_id: String,
    #[serde(default)]
    pub task_type: TaskType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub instruction: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub options: InferenceOptions,
    #[serde(default)]
    pub output_expectation: OutputExpectation,
}

impl TaskRequest {
    /// Creates a minimal [`TaskRequest`] with default schema version and inference options.
    #[must_use]
    pub fn new(
        task_id: impl Into<String>,
        model: impl Into<String>,
        instruction: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_owned(),
            task_id: task_id.into(),
            task_type: TaskType::Freeform,
            language: None,
            instruction: instruction.into(),
            context: String::new(),
            constraints: Vec::new(),
            model: model.into(),
            system_prompt: None,
            options: InferenceOptions::default(),
            output_expectation: OutputExpectation::default(),
        }
    }

    /// Validates schema version, required fields, inference options, and output expectations.
    pub fn validate(&self) -> WorkerResult<()> {
        if self.schema_version.trim() != SCHEMA_VERSION {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "unsupported schema_version '{}'; expected '{SCHEMA_VERSION}'",
                    self.schema_version
                ),
            });
        }
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
        self.options.validate()?;
        self.output_expectation.validate()
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationReport>,
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
            validation: None,
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
            let report = ValidationReport {
                passed: false,
                extracted_code_block: false,
                checks: vec![ValidationCheck::fail(
                    "non_empty",
                    "model returned an empty response",
                )],
                warnings: Vec::new(),
            };
            return TaskResult {
                schema_version: SCHEMA_VERSION.to_owned(),
                task_id: task_id.to_owned(),
                status: TaskStatus::ValidationFailed,
                model: response.model,
                output: None,
                raw_output: Some(raw_content),
                done_reason: response.done_reason,
                validation: Some(report),
                metrics: Some(metrics),
                error: Some(err.to_detail()),
            };
        }

        let report = ValidationReport {
            passed: true,
            extracted_code_block: false,
            checks: vec![ValidationCheck::pass("non_empty")],
            warnings: Vec::new(),
        };

        TaskResult {
            schema_version: SCHEMA_VERSION.to_owned(),
            task_id: task_id.to_owned(),
            status: TaskStatus::Success,
            model: response.model,
            output: Some(trimmed.to_owned()),
            raw_output: Some(raw_content),
            done_reason: response.done_reason,
            validation: Some(report),
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

    #[test]
    fn deserializes_spec_example_task_json_with_defaults() {
        let raw = r#"{
            "task_id": "task-001",
            "task_type": "generate_function",
            "language": "python",
            "instruction": "Implement a function that checks whether an integer is prime.",
            "context": "",
            "constraints": [
                "Use Python standard library only",
                "Return only the implementation",
                "Do not modify files"
            ],
            "model": "qwen2.5-coder:3b"
        }"#;

        let req: TaskRequest =
            serde_json::from_str(raw).expect("spec example TaskRequest JSON should deserialize");
        req.validate().expect("spec example task should be valid");
        assert_eq!(req.schema_version, "1");
        assert_eq!(req.task_type, TaskType::GenerateFunction);
        assert_eq!(req.language.as_deref(), Some("python"));
        assert_eq!(req.constraints.len(), 3);
        assert_eq!(
            req.output_expectation.effective_format(req.task_type),
            OutputFormat::Code
        );
    }

    #[test]
    fn rejects_unsupported_schema_version_and_invalid_task_type() {
        let mut req = TaskRequest::new("task-001", "qwen2.5-coder:3b", "Write code");
        req.schema_version = "99".into();
        assert_eq!(req.validate().expect_err("must reject v99").exit_code(), 2);

        assert!(TaskType::from_str("generate_function").is_ok());
        assert!(TaskType::from_str("generate_json").is_ok());
        assert!(TaskType::from_str("unknown_type").is_err());
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
        assert!(result.validation.as_ref().is_some_and(|v| v.passed));
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
        assert!(result.validation.as_ref().is_some_and(|v| !v.passed));
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
