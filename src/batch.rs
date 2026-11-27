//! Versioned sequential batch contracts and [`BatchExecutor`] orchestrator.
//!
//! Executes independent local inference tasks strictly sequentially (`concurrency = 1`)
//! by reusing [`TaskExecutor`]'s validation, prompt construction, Ollama client, metrics,
//! and non-executing static output validation pipeline.

use std::collections::HashSet;
use std::future::Future;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::DEFAULT_TIMEOUT_SECS;
use crate::client::OllamaClient;
use crate::config::{InferenceOptions, MAX_TIMEOUT_SECS};
use crate::error::{ErrorCategory, ErrorDetail, WorkerError, WorkerResult, exit_codes};
use crate::prompt::PromptBuilder;
use crate::task::{
    DryRunReport, OutputExpectation, SCHEMA_VERSION, TaskExecutor, TaskRequest, TaskResult,
    TaskStatus, TaskType,
};

/// Current structured schema version emitted in batch requests, results, and dry-run reports.
pub const BATCH_SCHEMA_VERSION: &str = "1";

/// Hard upper bound on the number of tasks in a single sequential batch.
pub const MAX_BATCH_TASKS: usize = 100;

/// Hard upper bound on overall batch wall-clock timeout (4 hours).
pub const MAX_BATCH_TIMEOUT_SECS: u64 = 14_400;

fn default_batch_schema_version() -> String {
    BATCH_SCHEMA_VERSION.to_owned()
}

/// Shared defaults inherited by items in a [`BatchTask`] unless overridden per task or via CLI.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<InferenceOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_expectation: Option<OutputExpectation>,
}

impl BatchDefaults {
    /// Validates optional batch-level defaults before executing any task.
    pub fn validate(&self) -> WorkerResult<()> {
        if let Some(model) = &self.model
            && model.trim().is_empty()
        {
            return Err(WorkerError::InvalidInput {
                message: "defaults.model must not be empty or whitespace".into(),
            });
        }
        if let Some(options) = &self.options {
            options.validate()?;
        }
        if let Some(expectation) = &self.output_expectation {
            expectation.validate()?;
            if let Some(schema) = &expectation.json_schema {
                jsonschema::validator_for(schema).map_err(|err| WorkerError::InvalidInput {
                    message: format!("invalid defaults.output_expectation.json_schema: {err}"),
                })?;
            }
        }
        Ok(())
    }
}

/// Execution policy governing failure handling and timeouts for a [`BatchTask`].
///
/// Note for library callers using [`crate::HttpOllamaClient`]: initialize [`crate::WorkerConfig`]
/// with at least [`BatchTask::max_effective_task_timeout_secs`] so `reqwest`'s client-level
/// timeout does not fire before longer per-task timeouts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPolicy {
    #[serde(default)]
    pub fail_fast: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch_timeout_secs: Option<u64>,
}

impl BatchPolicy {
    /// Validates timeout bounds on the batch policy.
    pub fn validate(&self) -> WorkerResult<()> {
        if let Some(secs) = self.task_timeout_secs
            && (secs == 0 || secs > MAX_TIMEOUT_SECS)
        {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "policy.task_timeout_secs must be between 1 and {MAX_TIMEOUT_SECS}, got {secs}"
                ),
            });
        }
        if let Some(secs) = self.batch_timeout_secs
            && (secs == 0 || secs > MAX_BATCH_TIMEOUT_SECS)
        {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "policy.batch_timeout_secs must be between 1 and {MAX_BATCH_TIMEOUT_SECS}, got {secs}"
                ),
            });
        }
        Ok(())
    }
}

/// Individual task entry inside [`BatchTask::tasks`].
///
/// Accepts every v0.1.0 [`TaskRequest`] JSON field verbatim while allowing `schema_version`,
/// `model`, `language`, `system_prompt`, `options`, and `output_expectation` to be omitted
/// and inherited from [`BatchTask::defaults`] without default-value shadowing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchTaskItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_version: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<InferenceOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_expectation: Option<OutputExpectation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

impl BatchTaskItem {
    /// Creates a minimal [`BatchTaskItem`] that inherits `model` and `options` from batch defaults.
    #[must_use]
    pub fn new(task_id: impl Into<String>, instruction: impl Into<String>) -> Self {
        Self {
            schema_version: None,
            task_id: task_id.into(),
            task_type: TaskType::Freeform,
            language: None,
            instruction: instruction.into(),
            context: String::new(),
            constraints: Vec::new(),
            model: None,
            system_prompt: None,
            options: None,
            output_expectation: None,
            timeout_secs: None,
        }
    }
}

impl From<TaskRequest> for BatchTaskItem {
    fn from(req: TaskRequest) -> Self {
        Self {
            schema_version: Some(req.schema_version),
            task_id: req.task_id,
            task_type: req.task_type,
            language: req.language,
            instruction: req.instruction,
            context: req.context,
            constraints: req.constraints,
            model: Some(req.model),
            system_prompt: req.system_prompt,
            options: Some(req.options),
            output_expectation: Some(req.output_expectation),
            timeout_secs: None,
        }
    }
}

/// Versioned sequential batch task contract (`schema_version = "1"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchTask {
    #[serde(default = "default_batch_schema_version")]
    pub schema_version: String,
    pub batch_id: String,
    #[serde(default)]
    pub defaults: BatchDefaults,
    #[serde(default)]
    pub policy: BatchPolicy,
    pub tasks: Vec<BatchTaskItem>,
}

impl BatchTask {
    /// Creates a [`BatchTask`] with default schema version, empty defaults, and failure isolation enabled.
    #[must_use]
    pub fn new(batch_id: impl Into<String>, tasks: Vec<BatchTaskItem>) -> Self {
        Self {
            schema_version: BATCH_SCHEMA_VERSION.to_owned(),
            batch_id: batch_id.into(),
            defaults: BatchDefaults::default(),
            policy: BatchPolicy::default(),
            tasks,
        }
    }

    /// Validates the batch envelope (version, `batch_id`, bounds, unique `task_id`s, policy/defaults).
    ///
    /// Per-task semantic checks (instruction content, secret scanning, context budget, JSON Schema)
    /// are intentionally deferred to per-task execution so individual task failures remain isolated.
    pub fn validate_envelope(&self) -> WorkerResult<()> {
        if self.schema_version.trim() != BATCH_SCHEMA_VERSION {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "unsupported batch schema_version '{}'; expected '{BATCH_SCHEMA_VERSION}'",
                    self.schema_version
                ),
            });
        }
        if self.batch_id.trim().is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "batch_id must not be empty".into(),
            });
        }
        if self.tasks.is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "batch tasks must contain at least one task".into(),
            });
        }
        if self.tasks.len() > MAX_BATCH_TASKS {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "batch contains {} tasks, exceeding maximum allowed limit of {MAX_BATCH_TASKS}",
                    self.tasks.len()
                ),
            });
        }

        self.defaults.validate()?;
        self.policy.validate()?;

        let mut seen_ids = HashSet::with_capacity(self.tasks.len());
        for (idx, item) in self.tasks.iter().enumerate() {
            let trimmed_id = item.task_id.trim();
            if trimmed_id.is_empty() {
                return Err(WorkerError::InvalidInput {
                    message: format!("tasks[{idx}].task_id must not be empty"),
                });
            }
            if !seen_ids.insert(trimmed_id) {
                return Err(WorkerError::InvalidInput {
                    message: format!("duplicate task_id '{trimmed_id}' in batch"),
                });
            }
            if let Some(secs) = item.timeout_secs
                && (secs == 0 || secs > MAX_TIMEOUT_SECS)
            {
                return Err(WorkerError::InvalidInput {
                    message: format!(
                        "tasks[{idx}].timeout_secs ('{trimmed_id}') must be between 1 and {MAX_TIMEOUT_SECS}, got {secs}"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Returns the effective `fail_fast` policy after applying runtime overrides.
    #[must_use]
    pub const fn effective_fail_fast(&self, overrides: &BatchRuntimeOverrides) -> bool {
        match overrides.fail_fast {
            Some(flag) => flag,
            None => self.policy.fail_fast,
        }
    }

    /// Returns the effective overall batch timeout after applying runtime overrides.
    #[must_use]
    pub fn effective_batch_timeout_secs(&self, overrides: &BatchRuntimeOverrides) -> Option<u64> {
        overrides
            .batch_timeout_secs
            .or(self.policy.batch_timeout_secs)
    }

    /// Resolves the effective per-task timeout for `item`.
    ///
    /// Precedence: `CLI --task-timeout > item.timeout_secs > policy.task_timeout_secs > global --timeout / env / default`.
    #[must_use]
    pub fn effective_task_timeout_secs(
        &self,
        item: &BatchTaskItem,
        overrides: &BatchRuntimeOverrides,
    ) -> u64 {
        overrides
            .task_timeout_secs
            .or(item.timeout_secs)
            .or(self.policy.task_timeout_secs)
            .unwrap_or(overrides.default_task_timeout_secs)
    }

    /// Returns the maximum effective per-task timeout across all items, bounded to `1..=MAX_TIMEOUT_SECS`.
    ///
    /// Used by the CLI to size the underlying `reqwest::Client` timeout so no valid per-task timeout
    /// is cut short by a smaller client-level HTTP timeout.
    #[must_use]
    pub fn max_effective_task_timeout_secs(&self, overrides: &BatchRuntimeOverrides) -> u64 {
        let max_secs = self
            .tasks
            .iter()
            .map(|item| self.effective_task_timeout_secs(item, overrides))
            .max()
            .unwrap_or(overrides.default_task_timeout_secs);
        max_secs.clamp(1, MAX_TIMEOUT_SECS)
    }

    /// Resolves a [`BatchTaskItem`] into a standard v0.1.0 [`TaskRequest`] using the documented precedence:
    /// `CLI overrides > Task item > Batch defaults > Environment fallback > Built-in defaults`.
    #[must_use]
    pub fn resolve_task_request(
        &self,
        item: &BatchTaskItem,
        overrides: &BatchRuntimeOverrides,
    ) -> TaskRequest {
        let model = if let Some(override_model) = overrides
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            override_model.to_owned()
        } else if let Some(item_model) = &item.model {
            // Preserve explicitly provided blank item.model so per-task validation rejects it
            // instead of silently falling back to defaults.model.
            item_model.trim().to_owned()
        } else {
            self.defaults
                .model
                .as_deref()
                .map(str::trim)
                .filter(|m| !m.is_empty())
                .or_else(|| {
                    overrides
                        .fallback_model
                        .as_deref()
                        .map(str::trim)
                        .filter(|m| !m.is_empty())
                })
                .unwrap_or_default()
                .to_owned()
        };

        let mut options = InferenceOptions::default();
        if let Some(default_opts) = &self.defaults.options {
            options = options.merge_override(default_opts);
        }
        if let Some(item_opts) = &item.options {
            options = options.merge_override(item_opts);
        }
        if let Some(cli_opts) = &overrides.options {
            options = options.merge_override(cli_opts);
        }

        let language = item
            .language
            .clone()
            .or_else(|| self.defaults.language.clone());
        let system_prompt = item
            .system_prompt
            .clone()
            .or_else(|| self.defaults.system_prompt.clone());
        let output_expectation = item
            .output_expectation
            .clone()
            .or_else(|| self.defaults.output_expectation.clone())
            .unwrap_or_default();

        TaskRequest {
            schema_version: item
                .schema_version
                .clone()
                .unwrap_or_else(|| SCHEMA_VERSION.to_owned()),
            task_id: item.task_id.clone(),
            task_type: item.task_type,
            language,
            instruction: item.instruction.clone(),
            context: item.context.clone(),
            constraints: item.constraints.clone(),
            model,
            system_prompt,
            options,
            output_expectation,
        }
    }
}

/// Optional CLI or caller-supplied runtime overrides applied on top of a [`BatchTask`].
#[derive(Debug, Clone, PartialEq)]
pub struct BatchRuntimeOverrides {
    pub model: Option<String>,
    pub fallback_model: Option<String>,
    pub fail_fast: Option<bool>,
    pub task_timeout_secs: Option<u64>,
    pub default_task_timeout_secs: u64,
    pub batch_timeout_secs: Option<u64>,
    pub options: Option<InferenceOptions>,
}

impl Default for BatchRuntimeOverrides {
    fn default() -> Self {
        Self {
            model: None,
            fallback_model: None,
            fail_fast: None,
            task_timeout_secs: None,
            default_task_timeout_secs: DEFAULT_TIMEOUT_SECS,
            batch_timeout_secs: None,
            options: None,
        }
    }
}

impl BatchRuntimeOverrides {
    /// Validates runtime override bounds before starting execution.
    pub fn validate(&self) -> WorkerResult<()> {
        if let Some(model) = &self.model
            && model.trim().is_empty()
        {
            return Err(WorkerError::InvalidInput {
                message: "override --model must not be empty or whitespace".into(),
            });
        }
        if self.default_task_timeout_secs == 0 || self.default_task_timeout_secs > MAX_TIMEOUT_SECS
        {
            return Err(WorkerError::InvalidConfig {
                message: format!(
                    "timeout must be between 1 and {MAX_TIMEOUT_SECS} seconds, got {}",
                    self.default_task_timeout_secs
                ),
            });
        }
        if let Some(secs) = self.task_timeout_secs
            && (secs == 0 || secs > MAX_TIMEOUT_SECS)
        {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "task_timeout_secs must be between 1 and {MAX_TIMEOUT_SECS}, got {secs}"
                ),
            });
        }
        if let Some(secs) = self.batch_timeout_secs
            && (secs == 0 || secs > MAX_BATCH_TIMEOUT_SECS)
        {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "batch_timeout_secs must be between 1 and {MAX_BATCH_TIMEOUT_SECS}, got {secs}"
                ),
            });
        }
        if let Some(opts) = &self.options {
            opts.validate()?;
        }
        Ok(())
    }
}

/// Overall execution status of a [`BatchResult`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    /// Intermediate checkpoint state written to `--output` while tasks are still executing.
    InProgress,
    /// All tasks executed and succeeded.
    Completed,
    /// At least one task succeeded and at least one task failed, timed out, or was skipped due to `fail_fast`.
    PartialSuccess,
    /// Zero tasks succeeded and the batch was not cancelled or batch-timed-out.
    Failed,
    /// Overall `batch_timeout_secs` elapsed before all tasks finished.
    TimedOut,
    /// Interrupted by `SIGINT`/`SIGTERM` before all tasks finished.
    Cancelled,
}

/// Aggregated task outcome counts for a [`BatchResult`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total_tasks: usize,
    pub executed_tasks: usize,
    pub succeeded_tasks: usize,
    pub validation_failed_tasks: usize,
    pub error_tasks: usize,
    pub timed_out_tasks: usize,
    pub cancelled_tasks: usize,
    pub skipped_tasks: usize,
}

impl BatchSummary {
    /// Computes task count totals from completed `results` and unexecuted `skipped_tasks`.
    #[must_use]
    pub fn from_results(total_tasks: usize, results: &[TaskResult], skipped_tasks: usize) -> Self {
        let mut summary = Self {
            total_tasks,
            executed_tasks: results.len(),
            skipped_tasks,
            ..Self::default()
        };

        for res in results {
            match res.status {
                TaskStatus::Success => summary.succeeded_tasks += 1,
                TaskStatus::ValidationFailed => summary.validation_failed_tasks += 1,
                TaskStatus::Error => summary.error_tasks += 1,
                TaskStatus::Timeout => summary.timed_out_tasks += 1,
                TaskStatus::Cancelled => summary.cancelled_tasks += 1,
            }
        }

        summary
    }
}

/// Aggregated timing and token metrics across all tasks in a batch.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BatchMetrics {
    pub batch_wall_clock_ms: u64,
    pub batch_overhead_ms: u64,
    #[serde(default)]
    pub unattributed_duration_ms: u64,
    pub total_worker_overhead_ms: u64,
    pub avg_worker_overhead_ms: f64,
    pub total_ollama_duration_ms: u64,
    pub total_load_duration_ms: u64,
    pub total_prompt_eval_duration_ms: u64,
    pub total_eval_duration_ms: u64,
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub avg_prompt_tokens_per_second: f64,
    pub avg_generation_tokens_per_second: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avg_ttft_ms: Option<f64>,
}

impl BatchMetrics {
    /// Aggregates per-task [`crate::InferenceMetrics`] alongside total batch wall-clock duration.
    ///
    /// Includes metrics from every task that produced Ollama telemetry (including `validation_failed`
    /// tasks) to reflect true CPU and token consumption.
    #[must_use]
    pub fn from_results(results: &[TaskResult], batch_wall_clock: Duration) -> Self {
        Self::from_results_with_unattributed(results, batch_wall_clock, 0)
    }

    /// Aggregates metrics while accounting separately for `unattributed_duration_ms` spent in
    /// tasks that did not return Ollama timing telemetry (for example, cancelled, timed-out,
    /// or transport-failed tasks), preventing `batch_overhead_ms` inflation.
    #[must_use]
    pub fn from_results_with_unattributed(
        results: &[TaskResult],
        batch_wall_clock: Duration,
        unattributed_duration_ms: u64,
    ) -> Self {
        let batch_wall_clock_ms = u64::try_from(batch_wall_clock.as_millis()).unwrap_or(u64::MAX);

        let mut metrics_count: u64 = 0;
        let mut total_worker_overhead_ms: u64 = 0;
        let mut total_ollama_duration_ms: u64 = 0;
        let mut total_load_duration_ms: u64 = 0;
        let mut total_prompt_eval_duration_ms: u64 = 0;
        let mut total_eval_duration_ms: u64 = 0;
        let mut total_prompt_tokens: u64 = 0;
        let mut total_generated_tokens: u64 = 0;
        let mut ttft_sum_ms: u64 = 0;
        let mut ttft_count: u64 = 0;

        for result in results {
            if let Some(m) = &result.metrics {
                metrics_count = metrics_count.saturating_add(1);
                total_worker_overhead_ms =
                    total_worker_overhead_ms.saturating_add(m.worker_overhead_ms);
                total_ollama_duration_ms =
                    total_ollama_duration_ms.saturating_add(m.total_duration_ms);
                total_load_duration_ms = total_load_duration_ms.saturating_add(m.load_duration_ms);
                total_prompt_eval_duration_ms =
                    total_prompt_eval_duration_ms.saturating_add(m.prompt_eval_duration_ms);
                total_eval_duration_ms = total_eval_duration_ms.saturating_add(m.eval_duration_ms);
                total_prompt_tokens = total_prompt_tokens.saturating_add(m.prompt_tokens);
                total_generated_tokens = total_generated_tokens.saturating_add(m.generated_tokens);
                if let Some(ttft) = m.ttft_ms {
                    ttft_sum_ms = ttft_sum_ms.saturating_add(ttft);
                    ttft_count = ttft_count.saturating_add(1);
                }
            }
        }

        let batch_overhead_ms = batch_wall_clock_ms
            .saturating_sub(total_ollama_duration_ms)
            .saturating_sub(unattributed_duration_ms);
        let avg_worker_overhead_ms = if metrics_count > 0 {
            round2((total_worker_overhead_ms as f64) / (metrics_count as f64))
        } else {
            0.0
        };
        let avg_prompt_tokens_per_second =
            weighted_tokens_per_second(total_prompt_tokens, total_prompt_eval_duration_ms);
        let avg_generation_tokens_per_second =
            weighted_tokens_per_second(total_generated_tokens, total_eval_duration_ms);
        let avg_ttft_ms =
            (ttft_count > 0).then(|| round2((ttft_sum_ms as f64) / (ttft_count as f64)));

        Self {
            batch_wall_clock_ms,
            batch_overhead_ms,
            unattributed_duration_ms,
            total_worker_overhead_ms,
            avg_worker_overhead_ms,
            total_ollama_duration_ms,
            total_load_duration_ms,
            total_prompt_eval_duration_ms,
            total_eval_duration_ms,
            total_prompt_tokens,
            total_generated_tokens,
            avg_prompt_tokens_per_second,
            avg_generation_tokens_per_second,
            avg_ttft_ms,
        }
    }
}

#[must_use]
fn weighted_tokens_per_second(tokens: u64, duration_ms: u64) -> f64 {
    if tokens == 0 || duration_ms == 0 {
        return 0.0;
    }
    let seconds = (duration_ms as f64) / 1_000.0;
    round2((tokens as f64) / seconds)
}

#[must_use]
fn round2(val: f64) -> f64 {
    if !val.is_finite() {
        return 0.0;
    }
    let scaled = val * 100.0;
    if !scaled.is_finite() {
        return 0.0;
    }
    scaled.round() / 100.0
}

/// Machine-readable result emitted by `ollama-worker batch` (`schema_version = "1"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BatchResult {
    pub schema_version: String,
    pub batch_id: String,
    pub status: BatchStatus,
    pub fail_fast: bool,
    pub summary: BatchSummary,
    pub metrics: BatchMetrics,
    pub results: Vec<TaskResult>,
    pub skipped_task_ids: Vec<String>,
    pub error: Option<ErrorDetail>,
}

impl BatchResult {
    /// Builds a structured batch-level failure result (for example, invalid envelope or config).
    #[must_use]
    pub fn from_error(
        batch_id: impl Into<String>,
        fail_fast: bool,
        skipped_task_ids: Vec<String>,
        error: &WorkerError,
    ) -> Self {
        let total_tasks = skipped_task_ids.len();
        let status = match error {
            WorkerError::Timeout { .. } => BatchStatus::TimedOut,
            WorkerError::Cancelled => BatchStatus::Cancelled,
            _ => BatchStatus::Failed,
        };
        Self {
            schema_version: BATCH_SCHEMA_VERSION.to_owned(),
            batch_id: batch_id.into(),
            status,
            fail_fast,
            summary: BatchSummary::from_results(total_tasks, &[], skipped_task_ids.len()),
            metrics: BatchMetrics::default(),
            results: Vec::new(),
            skipped_task_ids,
            error: Some(error.to_detail()),
        }
    }

    /// Returns the documented CLI process exit code for this batch result.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self.status {
            BatchStatus::Completed => exit_codes::SUCCESS,
            BatchStatus::Cancelled => exit_codes::CANCELLED,
            BatchStatus::TimedOut => exit_codes::TIMEOUT,
            BatchStatus::PartialSuccess => exit_codes::PARTIAL_FAILURE,
            BatchStatus::Failed => {
                if let Some(err) = &self.error {
                    return err.category.exit_code();
                }
                self.results
                    .iter()
                    .find(|r| r.status != TaskStatus::Success)
                    .map_or(exit_codes::INTERNAL_ERROR, TaskResult::exit_code)
            }
            BatchStatus::InProgress => exit_codes::INTERNAL_ERROR,
        }
    }
}

/// Per-task offline preview entry in [`BatchDryRunReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchDryRunTask {
    pub task_id: String,
    pub valid: bool,
    pub effective_timeout_secs: u64,
    pub report: Option<DryRunReport>,
    pub error: Option<ErrorDetail>,
}

/// Offline preview returned by `ollama-worker batch --dry-run` without contacting Ollama.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchDryRunReport {
    pub schema_version: String,
    pub batch_id: String,
    pub fail_fast: bool,
    pub total_tasks: usize,
    pub valid_tasks: usize,
    pub invalid_tasks: usize,
    pub total_estimated_prompt_tokens: usize,
    pub tasks: Vec<BatchDryRunTask>,
}

impl BatchDryRunReport {
    /// Returns exit `0` when all tasks pass dry-run checks, or exit `2` (`INVALID_INPUT`) otherwise.
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        if self.invalid_tasks == 0 {
            exit_codes::SUCCESS
        } else {
            exit_codes::INVALID_INPUT
        }
    }
}

/// Executes a [`BatchTask`] strictly sequentially (`concurrency = 1`) by reusing [`TaskExecutor`].
///
/// When backed by [`crate::HttpOllamaClient`], configure [`crate::WorkerConfig`] with a timeout
/// of at least [`BatchTask::max_effective_task_timeout_secs`] so client-level HTTP timeouts do
/// not preempt longer per-task timeouts.
#[derive(Debug, Clone)]
pub struct BatchExecutor<C> {
    task_executor: TaskExecutor<C>,
}

/// Validates a [`BatchTask`] and every task item offline without instantiating an [`OllamaClient`].
pub fn dry_run_batch(
    batch: &BatchTask,
    overrides: &BatchRuntimeOverrides,
    prompt_builder: &PromptBuilder,
) -> WorkerResult<BatchDryRunReport> {
    overrides.validate()?;
    batch.validate_envelope()?;

    let mut tasks = Vec::with_capacity(batch.tasks.len());
    let mut valid_tasks = 0usize;
    let mut invalid_tasks = 0usize;
    let mut total_estimated_prompt_tokens = 0usize;

    for item in &batch.tasks {
        let resolved = batch.resolve_task_request(item, overrides);
        let effective_timeout_secs = batch.effective_task_timeout_secs(item, overrides);
        match dry_run_single_task(&resolved, prompt_builder) {
            Ok(report) => {
                valid_tasks += 1;
                total_estimated_prompt_tokens =
                    total_estimated_prompt_tokens.saturating_add(report.estimated_prompt_tokens);
                tasks.push(BatchDryRunTask {
                    task_id: item.task_id.trim().to_owned(),
                    valid: true,
                    effective_timeout_secs,
                    report: Some(report),
                    error: None,
                });
            }
            Err(err) => {
                invalid_tasks += 1;
                tasks.push(BatchDryRunTask {
                    task_id: item.task_id.trim().to_owned(),
                    valid: false,
                    effective_timeout_secs,
                    report: None,
                    error: Some(err.to_detail()),
                });
            }
        }
    }

    Ok(BatchDryRunReport {
        schema_version: BATCH_SCHEMA_VERSION.to_owned(),
        batch_id: batch.batch_id.trim().to_owned(),
        fail_fast: batch.effective_fail_fast(overrides),
        total_tasks: batch.tasks.len(),
        valid_tasks,
        invalid_tasks,
        total_estimated_prompt_tokens,
        tasks,
    })
}

fn dry_run_single_task(
    request: &TaskRequest,
    prompt_builder: &PromptBuilder,
) -> WorkerResult<DryRunReport> {
    request.validate()?;
    crate::validation::OutputValidator::validate_request_schema(request)?;
    let built = prompt_builder.build(request)?;
    let json_format_enabled = request
        .output_expectation
        .effective_format(request.task_type)
        == crate::task::OutputFormat::Json;

    Ok(DryRunReport {
        schema_version: SCHEMA_VERSION.to_owned(),
        task_id: request.task_id.trim().to_owned(),
        model: request.model.trim().to_owned(),
        task_type: request.task_type,
        system_prompt: built.system_prompt,
        user_prompt: built.user_prompt,
        estimated_prompt_tokens: built.estimated_prompt_tokens,
        context_budget_tokens: built.context_budget_tokens,
        ollama_json_format_enabled: json_format_enabled,
        warnings: built.warnings,
    })
}

impl<C: OllamaClient> BatchExecutor<C> {
    /// Creates a new sequential batch executor backed by any [`OllamaClient`].
    #[must_use]
    pub fn new(client: C, verify_model: bool) -> Self {
        Self {
            task_executor: TaskExecutor::new(client, verify_model),
        }
    }

    /// Creates a [`BatchExecutor`] wrapping an existing [`TaskExecutor`].
    #[must_use]
    pub const fn from_task_executor(task_executor: TaskExecutor<C>) -> Self {
        Self { task_executor }
    }

    /// Overrides the [`PromptBuilder`] used for prompt rendering and context budget checks.
    #[must_use]
    pub fn with_prompt_builder(mut self, prompt_builder: PromptBuilder) -> Self {
        self.task_executor = self.task_executor.with_prompt_builder(prompt_builder);
        self
    }

    /// Validates the batch envelope and every task item offline without contacting Ollama.
    pub fn dry_run(
        &self,
        batch: &BatchTask,
        overrides: &BatchRuntimeOverrides,
    ) -> WorkerResult<BatchDryRunReport> {
        overrides.validate()?;
        batch.validate_envelope()?;

        let mut tasks = Vec::with_capacity(batch.tasks.len());
        let mut valid_tasks = 0usize;
        let mut invalid_tasks = 0usize;
        let mut total_estimated_prompt_tokens = 0usize;

        for item in &batch.tasks {
            let resolved = batch.resolve_task_request(item, overrides);
            let effective_timeout_secs = batch.effective_task_timeout_secs(item, overrides);
            match self.task_executor.dry_run(&resolved) {
                Ok(report) => {
                    valid_tasks += 1;
                    total_estimated_prompt_tokens = total_estimated_prompt_tokens
                        .saturating_add(report.estimated_prompt_tokens);
                    tasks.push(BatchDryRunTask {
                        task_id: item.task_id.trim().to_owned(),
                        valid: true,
                        effective_timeout_secs,
                        report: Some(report),
                        error: None,
                    });
                }
                Err(err) => {
                    invalid_tasks += 1;
                    tasks.push(BatchDryRunTask {
                        task_id: item.task_id.trim().to_owned(),
                        valid: false,
                        effective_timeout_secs,
                        report: None,
                        error: Some(err.to_detail()),
                    });
                }
            }
        }

        Ok(BatchDryRunReport {
            schema_version: BATCH_SCHEMA_VERSION.to_owned(),
            batch_id: batch.batch_id.trim().to_owned(),
            fail_fast: batch.effective_fail_fast(overrides),
            total_tasks: batch.tasks.len(),
            valid_tasks,
            invalid_tasks,
            total_estimated_prompt_tokens,
            tasks,
        })
    }

    /// Executes a [`BatchTask`] sequentially with no external cancellation signal.
    pub async fn execute(
        &self,
        batch: &BatchTask,
        overrides: &BatchRuntimeOverrides,
    ) -> BatchResult {
        self.execute_with_cancel(batch, overrides, std::future::pending::<()>())
            .await
    }

    /// Executes a [`BatchTask`] sequentially while observing `cancel` for interruption.
    pub async fn execute_with_cancel<F>(
        &self,
        batch: &BatchTask,
        overrides: &BatchRuntimeOverrides,
        cancel: F,
    ) -> BatchResult
    where
        F: Future<Output = ()> + Send,
    {
        self.execute_inner(batch, overrides, cancel, None).await
    }

    /// Executes a [`BatchTask`] sequentially with cancellation support and a per-task progress hook.
    ///
    /// `on_task_complete` is called with an up-to-date [`BatchResult`] snapshot after each executed
    /// task so callers (such as CLI `--output`) can atomically persist completed results incrementally.
    pub async fn execute_with_cancel_and_hook<F, H>(
        &self,
        batch: &BatchTask,
        overrides: &BatchRuntimeOverrides,
        cancel: F,
        mut on_task_complete: H,
    ) -> BatchResult
    where
        F: Future<Output = ()> + Send,
        H: FnMut(&BatchResult),
    {
        self.execute_inner(batch, overrides, cancel, Some(&mut on_task_complete))
            .await
    }

    async fn execute_inner<F>(
        &self,
        batch: &BatchTask,
        overrides: &BatchRuntimeOverrides,
        cancel: F,
        mut on_task_complete: Option<&mut dyn FnMut(&BatchResult)>,
    ) -> BatchResult
    where
        F: Future<Output = ()> + Send,
    {
        let fail_fast = batch.effective_fail_fast(overrides);
        let fallback_batch_id = batch.batch_id.trim().to_owned();
        let bounded_task_ids: Vec<String> = if batch.tasks.len() <= MAX_BATCH_TASKS {
            batch
                .tasks
                .iter()
                .map(|t| t.task_id.trim().to_owned())
                .collect()
        } else {
            Vec::new()
        };

        if let Err(err) = overrides.validate() {
            return BatchResult::from_error(fallback_batch_id, fail_fast, bounded_task_ids, &err);
        }
        if let Err(err) = batch.validate_envelope() {
            return BatchResult::from_error(fallback_batch_id, fail_fast, bounded_task_ids, &err);
        }

        let started = Instant::now();
        let batch_timeout_secs = batch.effective_batch_timeout_secs(overrides);
        let batch_deadline = batch_timeout_secs.map(|secs| started + Duration::from_secs(secs));

        let total_tasks = batch.tasks.len();
        let mut results = Vec::with_capacity(total_tasks);
        let mut cached_models = None;
        let mut unattributed_duration_ms: u64 = 0;
        let mut cancelled = false;
        let mut batch_timed_out = false;

        tokio::pin!(cancel);

        for (idx, item) in batch.tasks.iter().enumerate() {
            let pre_cancelled = tokio::select! {
                biased;
                () = cancel.as_mut() => true,
                () = std::future::ready(()) => false,
            };
            if pre_cancelled {
                cancelled = true;
                break;
            }

            let task_timeout_secs = batch.effective_task_timeout_secs(item, overrides);
            let task_timeout = Duration::from_secs(task_timeout_secs);
            let (effective_timeout, bounded_by_batch) = match batch_deadline {
                Some(deadline) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        batch_timed_out = true;
                        break;
                    }
                    if remaining < task_timeout {
                        (remaining, true)
                    } else {
                        (task_timeout, false)
                    }
                }
                None => (task_timeout, false),
            };

            let resolved_req = batch.resolve_task_request(item, overrides);
            let task_id = resolved_req.task_id.trim().to_owned();
            let model = resolved_req.model.trim().to_owned();
            let task_started = Instant::now();

            let outcome = tokio::time::timeout(
                effective_timeout,
                self.task_executor.execute_with_cancel_and_model_cache(
                    &resolved_req,
                    cancel.as_mut(),
                    &mut cached_models,
                ),
            )
            .await;

            let task_elapsed_ms =
                u64::try_from(task_started.elapsed().as_millis()).unwrap_or(u64::MAX);
            let mut stop_now = false;
            match outcome {
                Ok(task_result) => {
                    let status = task_result.status;
                    if task_result.metrics.is_none() {
                        unattributed_duration_ms =
                            unattributed_duration_ms.saturating_add(task_elapsed_ms);
                    }
                    results.push(task_result);

                    if status == TaskStatus::Cancelled {
                        cancelled = true;
                        stop_now = true;
                    } else if status == TaskStatus::Timeout
                        && batch_deadline.is_some_and(|deadline| Instant::now() >= deadline)
                    {
                        batch_timed_out = true;
                        stop_now = true;
                    } else if fail_fast && status != TaskStatus::Success {
                        stop_now = true;
                    }
                }
                Err(_) => {
                    unattributed_duration_ms =
                        unattributed_duration_ms.saturating_add(task_elapsed_ms);
                    let hit_batch_deadline = bounded_by_batch
                        || batch_deadline.is_some_and(|deadline| Instant::now() >= deadline);
                    if hit_batch_deadline {
                        let secs = batch_timeout_secs.unwrap_or(task_timeout_secs);
                        results.push(batch_timeout_task_result(&task_id, &model, secs));
                        batch_timed_out = true;
                        stop_now = true;
                    } else {
                        let err = WorkerError::Timeout {
                            timeout_secs: task_timeout_secs,
                        };
                        results.push(TaskResult::from_error(&task_id, &model, &err));
                        if fail_fast {
                            stop_now = true;
                        }
                    }
                }
            }

            // Only construct and clone an intermediate snapshot when a progress hook is registered.
            if !stop_now
                && idx + 1 < total_tasks
                && let Some(hook) = on_task_complete.as_deref_mut()
            {
                let remaining_ids: Vec<String> = batch.tasks[idx + 1..]
                    .iter()
                    .map(|t| t.task_id.trim().to_owned())
                    .collect();
                let snapshot = build_batch_result(
                    &fallback_batch_id,
                    BatchStatus::InProgress,
                    fail_fast,
                    total_tasks,
                    results.clone(),
                    remaining_ids,
                    started.elapsed(),
                    unattributed_duration_ms,
                    None,
                );
                hook(&snapshot);
            }

            if stop_now {
                break;
            }
        }

        let skipped_task_ids: Vec<String> = batch.tasks[results.len()..]
            .iter()
            .map(|t| t.task_id.trim().to_owned())
            .collect();

        let (final_status, batch_error) = if cancelled {
            (
                BatchStatus::Cancelled,
                Some(WorkerError::Cancelled.to_detail()),
            )
        } else if batch_timed_out {
            let secs = batch_timeout_secs.unwrap_or(0);
            (
                BatchStatus::TimedOut,
                Some(ErrorDetail {
                    category: ErrorCategory::Timeout,
                    message: format!("batch timed out after {secs}s"),
                    retryable: false,
                }),
            )
        } else {
            let succeeded = results
                .iter()
                .filter(|r| r.status == TaskStatus::Success)
                .count();
            if succeeded == total_tasks {
                (BatchStatus::Completed, None)
            } else if succeeded > 0 {
                (BatchStatus::PartialSuccess, None)
            } else {
                (BatchStatus::Failed, None)
            }
        };

        build_batch_result(
            &fallback_batch_id,
            final_status,
            fail_fast,
            total_tasks,
            results,
            skipped_task_ids,
            started.elapsed(),
            unattributed_duration_ms,
            batch_error,
        )
    }
}

fn batch_timeout_task_result(task_id: &str, model: &str, batch_timeout_secs: u64) -> TaskResult {
    TaskResult {
        schema_version: SCHEMA_VERSION.to_owned(),
        task_id: task_id.to_owned(),
        status: TaskStatus::Timeout,
        model: model.to_owned(),
        output: None,
        raw_output: None,
        done_reason: None,
        validation: None,
        metrics: None,
        error: Some(ErrorDetail {
            category: ErrorCategory::Timeout,
            message: format!(
                "task aborted: overall batch timeout of {batch_timeout_secs}s exceeded"
            ),
            retryable: false,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_batch_result(
    batch_id: &str,
    status: BatchStatus,
    fail_fast: bool,
    total_tasks: usize,
    results: Vec<TaskResult>,
    skipped_task_ids: Vec<String>,
    elapsed: Duration,
    unattributed_duration_ms: u64,
    error: Option<ErrorDetail>,
) -> BatchResult {
    let summary = BatchSummary::from_results(total_tasks, &results, skipped_task_ids.len());
    let metrics =
        BatchMetrics::from_results_with_unattributed(&results, elapsed, unattributed_duration_ms);
    BatchResult {
        schema_version: BATCH_SCHEMA_VERSION.to_owned(),
        batch_id: batch_id.to_owned(),
        status,
        fail_fast,
        summary,
        metrics,
        results,
        skipped_task_ids,
        error,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::client::{
        ChatMessage, ChatRequest, ChatResponse, ChatRole, GenerateRequest, GenerateResponse,
        InstalledModel, RunningModel, StreamedChatResponse, VersionResponse,
    };
    use crate::error::ErrorCategory;
    use crate::metrics::InferenceMetrics;

    enum FakeAction {
        Ok {
            content: String,
            prompt_tokens: u64,
            eval_tokens: u64,
            total_ns: u64,
            prompt_ns: u64,
            eval_ns: u64,
            load_ns: u64,
            delay_ms: u64,
            ttft_ms: Option<u64>,
        },
        Err(WorkerError),
    }

    impl FakeAction {
        fn ok(content: &str) -> Self {
            Self::Ok {
                content: content.to_owned(),
                prompt_tokens: 100,
                eval_tokens: 40,
                total_ns: 500_000_000,
                prompt_ns: 100_000_000,
                eval_ns: 400_000_000,
                load_ns: 0,
                delay_ms: 0,
                ttft_ms: None,
            }
        }
    }

    struct ScriptedOllamaClient {
        installed: Vec<InstalledModel>,
        list_models_errors: Mutex<VecDeque<Option<WorkerError>>>,
        actions: Mutex<VecDeque<FakeAction>>,
        list_models_calls: AtomicUsize,
        chat_calls: AtomicUsize,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        on_chat_complete: Option<Box<dyn Fn(usize) + Send + Sync>>,
    }

    impl ScriptedOllamaClient {
        fn new(models: &[&str], actions: Vec<FakeAction>) -> Self {
            Self {
                installed: models
                    .iter()
                    .map(|name| InstalledModel {
                        name: (*name).to_owned(),
                        model: Some((*name).to_owned()),
                        modified_at: None,
                        size: 2_400_000_000,
                        digest: None,
                        details: None,
                    })
                    .collect(),
                list_models_errors: Mutex::new(VecDeque::new()),
                actions: Mutex::new(VecDeque::from(actions)),
                list_models_calls: AtomicUsize::new(0),
                chat_calls: AtomicUsize::new(0),
                in_flight: AtomicUsize::new(0),
                max_in_flight: AtomicUsize::new(0),
                on_chat_complete: None,
            }
        }

        async fn run_next(
            &self,
            request: &ChatRequest,
        ) -> WorkerResult<(ChatResponse, Option<u64>)> {
            let active = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(active, Ordering::SeqCst);

            let call_idx = self.chat_calls.fetch_add(1, Ordering::SeqCst);
            let next = self
                .actions
                .lock()
                .expect("mutex should not be poisoned")
                .pop_front()
                .unwrap_or_else(|| FakeAction::ok("default ok"));

            let result = match next {
                FakeAction::Ok {
                    content,
                    prompt_tokens,
                    eval_tokens,
                    total_ns,
                    prompt_ns,
                    eval_ns,
                    load_ns,
                    delay_ms,
                    ttft_ms,
                } => {
                    if delay_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                    Ok((
                        ChatResponse {
                            model: request.model.clone(),
                            created_at: None,
                            message: ChatMessage {
                                role: ChatRole::Assistant,
                                content,
                            },
                            done: true,
                            done_reason: Some("stop".into()),
                            total_duration: Some(total_ns),
                            load_duration: Some(load_ns),
                            prompt_eval_count: Some(prompt_tokens),
                            prompt_eval_duration: Some(prompt_ns),
                            eval_count: Some(eval_tokens),
                            eval_duration: Some(eval_ns),
                        },
                        ttft_ms,
                    ))
                }
                FakeAction::Err(err) => Err(err),
            };

            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            if let Some(cb) = &self.on_chat_complete {
                cb(call_idx);
            }
            result
        }
    }

    impl OllamaClient for ScriptedOllamaClient {
        async fn version(&self) -> WorkerResult<VersionResponse> {
            Ok(VersionResponse {
                version: "0.6.2".into(),
            })
        }

        async fn list_models(&self) -> WorkerResult<Vec<InstalledModel>> {
            self.list_models_calls.fetch_add(1, Ordering::SeqCst);
            if let Some(Some(err)) = self
                .list_models_errors
                .lock()
                .expect("mutex should not be poisoned")
                .pop_front()
            {
                return Err(err);
            }
            Ok(self.installed.clone())
        }

        async fn list_running_models(&self) -> WorkerResult<Vec<RunningModel>> {
            Ok(vec![])
        }

        async fn chat(&self, request: &ChatRequest) -> WorkerResult<ChatResponse> {
            self.run_next(request).await.map(|(resp, _)| resp)
        }

        async fn chat_stream(&self, request: &ChatRequest) -> WorkerResult<StreamedChatResponse> {
            let (response, ttft_ms) = self.run_next(request).await?;
            Ok(StreamedChatResponse {
                response,
                ttft_ms: ttft_ms.or(Some(15)),
            })
        }

        async fn generate(&self, _request: &GenerateRequest) -> WorkerResult<GenerateResponse> {
            Err(WorkerError::Internal {
                message: "unused in batch tests".into(),
            })
        }
    }

    fn sample_batch(fail_fast: bool) -> BatchTask {
        let mut t1 = BatchTaskItem::new("t1", "Write is_prime");
        t1.task_type = TaskType::GenerateFunction;

        let mut t2 = BatchTaskItem::new("t2", "Write slugify");
        t2.task_type = TaskType::GenerateFunction;

        let mut t3 = BatchTaskItem::new("t3", "Write chunk_list");
        t3.task_type = TaskType::GenerateFunction;

        let mut batch = BatchTask::new("batch-001", vec![t1, t2, t3]);
        batch.defaults.model = Some("qwen2.5-coder:3b".into());
        batch.defaults.language = Some("python".into());
        batch.policy.fail_fast = fail_fast;
        batch
    }

    #[test]
    fn deserializes_batch_with_verbatim_v1_task_and_preserves_option_precedence() {
        let raw = r#"{
            "batch_id": "py-batch",
            "defaults": {
                "model": "qwen2.5-coder:3b",
                "language": "python",
                "options": {
                    "temperature": 0.0,
                    "max_tokens": 256,
                    "seed": 42
                }
            },
            "policy": {
                "fail_fast": false,
                "task_timeout_secs": 90,
                "batch_timeout_secs": 300
            },
            "tasks": [
                {
                    "task_id": "concise-item",
                    "task_type": "generate_function",
                    "instruction": "Implement is_prime"
                },
                {
                    "schema_version": "1",
                    "task_id": "v1-verbatim",
                    "task_type": "generate_function",
                    "language": "rust",
                    "instruction": "Implement is_prime in Rust",
                    "context": "",
                    "constraints": ["No unsafe"],
                    "model": "qwen2.5-coder:7b",
                    "options": {
                        "max_tokens": 384
                    },
                    "timeout_secs": 45
                }
            ]
        }"#;

        let batch: BatchTask = serde_json::from_str(raw).expect("BatchTask JSON should parse");
        batch.validate_envelope().expect("envelope should be valid");
        assert_eq!(batch.schema_version, BATCH_SCHEMA_VERSION);

        let overrides = BatchRuntimeOverrides::default();
        let req1 = batch.resolve_task_request(&batch.tasks[0], &overrides);
        assert_eq!(req1.model, "qwen2.5-coder:3b");
        assert_eq!(req1.language.as_deref(), Some("python"));
        // Batch default temperature (0.0) and max_tokens (256) must NOT be shadowed by TaskRequest defaults.
        assert_eq!(req1.options.temperature, Some(0.0));
        assert_eq!(req1.options.max_tokens, Some(256));
        assert_eq!(req1.options.seed, Some(42));
        assert_eq!(
            batch.effective_task_timeout_secs(&batch.tasks[0], &overrides),
            90
        );

        let req2 = batch.resolve_task_request(&batch.tasks[1], &overrides);
        assert_eq!(req2.model, "qwen2.5-coder:7b");
        assert_eq!(req2.language.as_deref(), Some("rust"));
        // Task item overrode max_tokens while inheriting temperature and seed from defaults.
        assert_eq!(req2.options.temperature, Some(0.0));
        assert_eq!(req2.options.max_tokens, Some(384));
        assert_eq!(req2.options.seed, Some(42));
        assert_eq!(
            batch.effective_task_timeout_secs(&batch.tasks[1], &overrides),
            45
        );
        assert_eq!(batch.max_effective_task_timeout_secs(&overrides), 90);

        // CLI overrides take highest precedence.
        let cli_overrides = BatchRuntimeOverrides {
            model: Some("override:3b".into()),
            task_timeout_secs: Some(30),
            options: Some(InferenceOptions {
                temperature: Some(0.2),
                num_ctx: None,
                max_tokens: None,
                seed: None,
                keep_alive: Some("10m".into()),
                stream: true,
            }),
            ..Default::default()
        };
        let req1_cli = batch.resolve_task_request(&batch.tasks[0], &cli_overrides);
        assert_eq!(req1_cli.model, "override:3b");
        assert_eq!(req1_cli.options.temperature, Some(0.2));
        assert_eq!(req1_cli.options.max_tokens, Some(256));
        assert_eq!(req1_cli.options.keep_alive.as_deref(), Some("10m"));
        assert!(req1_cli.options.stream);
        assert_eq!(
            batch.effective_task_timeout_secs(&batch.tasks[0], &cli_overrides),
            30
        );
    }

    #[test]
    fn envelope_validation_rejects_bad_version_empty_or_duplicate_ids_and_limits() {
        let mut bad_ver = sample_batch(false);
        bad_ver.schema_version = "99".into();
        assert_eq!(bad_ver.validate_envelope().unwrap_err().exit_code(), 2);

        let empty = BatchTask::new("b-empty", vec![]);
        assert_eq!(empty.validate_envelope().unwrap_err().exit_code(), 2);

        let dup = BatchTask::new(
            "b-dup",
            vec![
                BatchTaskItem::new("same-id", "A"),
                BatchTaskItem::new("same-id", "B"),
            ],
        );
        assert_eq!(dup.validate_envelope().unwrap_err().exit_code(), 2);

        let mut too_many = Vec::new();
        for i in 0..=MAX_BATCH_TASKS {
            too_many.push(BatchTaskItem::new(format!("t-{i}"), "task"));
        }
        let oversized = BatchTask::new("b-big", too_many);
        assert_eq!(oversized.validate_envelope().unwrap_err().exit_code(), 2);

        let mut bad_timeout = sample_batch(false);
        bad_timeout.policy.batch_timeout_secs = Some(MAX_BATCH_TIMEOUT_SECS + 1);
        assert_eq!(bad_timeout.validate_envelope().unwrap_err().exit_code(), 2);
    }

    #[test]
    fn metrics_aggregation_computes_weighted_rates_and_handles_zero_durations() {
        let make_res =
            |id: &str, status: TaskStatus, metrics: Option<InferenceMetrics>| TaskResult {
                schema_version: SCHEMA_VERSION.to_owned(),
                task_id: id.to_owned(),
                status,
                model: "qwen2.5-coder:3b".into(),
                output: (status == TaskStatus::Success).then(|| "ok".into()),
                raw_output: None,
                done_reason: Some("stop".into()),
                validation: None,
                metrics,
                error: None,
            };

        let results = vec![
            make_res(
                "t1",
                TaskStatus::Success,
                Some(InferenceMetrics {
                    worker_overhead_ms: 2,
                    ttft_ms: Some(100),
                    total_duration_ms: 1000,
                    load_duration_ms: 200,
                    prompt_eval_duration_ms: 250,
                    eval_duration_ms: 500,
                    prompt_tokens: 100,
                    generated_tokens: 50,
                    prompt_tokens_per_second: 400.0,
                    generation_tokens_per_second: 100.0,
                }),
            ),
            make_res(
                "t2",
                TaskStatus::ValidationFailed,
                Some(InferenceMetrics {
                    worker_overhead_ms: 4,
                    ttft_ms: Some(200),
                    total_duration_ms: 2000,
                    load_duration_ms: 0,
                    prompt_eval_duration_ms: 750,
                    eval_duration_ms: 1500,
                    prompt_tokens: 300,
                    generated_tokens: 150,
                    prompt_tokens_per_second: 400.0,
                    generation_tokens_per_second: 100.0,
                }),
            ),
            make_res("t3", TaskStatus::Error, None),
        ];

        let agg = BatchMetrics::from_results(&results, Duration::from_millis(3_015));
        assert_eq!(agg.batch_wall_clock_ms, 3_015);
        assert_eq!(agg.batch_overhead_ms, 15);
        assert_eq!(agg.total_worker_overhead_ms, 6);
        assert_eq!(agg.avg_worker_overhead_ms, 3.0);
        assert_eq!(agg.total_ollama_duration_ms, 3_000);
        assert_eq!(agg.total_load_duration_ms, 200);
        assert_eq!(agg.total_prompt_tokens, 400);
        assert_eq!(agg.total_generated_tokens, 200);
        assert_eq!(agg.avg_prompt_tokens_per_second, 400.0);
        assert_eq!(agg.avg_generation_tokens_per_second, 100.0);
        assert_eq!(agg.avg_ttft_ms, Some(150.0));

        let empty = BatchMetrics::from_results(&[], Duration::from_millis(5));
        assert_eq!(empty.avg_worker_overhead_ms, 0.0);
        assert_eq!(empty.avg_prompt_tokens_per_second, 0.0);
        assert_eq!(empty.avg_generation_tokens_per_second, 0.0);
        assert_eq!(empty.avg_ttft_ms, None);
    }

    #[test]
    fn dry_run_validates_all_tasks_without_contacting_ollama() {
        let client = ScriptedOllamaClient::new(&[], vec![]);
        let executor = BatchExecutor::new(client, true);

        let mut batch = sample_batch(false);
        batch.tasks[1].instruction = "Use token ghp_1234567890abcdef1234567890abcdef".into();

        let report = executor
            .dry_run(&batch, &BatchRuntimeOverrides::default())
            .expect("dry_run envelope should succeed");
        assert_eq!(report.total_tasks, 3);
        assert_eq!(report.valid_tasks, 2);
        assert_eq!(report.invalid_tasks, 1);
        assert_eq!(report.exit_code(), 2);
        assert!(report.tasks[0].valid);
        assert!(!report.tasks[1].valid);
        assert!(report.tasks[2].valid);
        assert!(report.total_estimated_prompt_tokens > 0);
    }

    #[tokio::test]
    async fn executes_sequentially_caches_model_check_and_invokes_progress_hook() {
        let client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def is_prime(n): return n > 1"),
                FakeAction::ok("def slugify(s): return s.lower()"),
                FakeAction::ok("def chunk_list(xs, n): return [xs]"),
            ],
        );
        let executor = BatchExecutor::new(client, true);
        let batch = sample_batch(false);

        let mut checkpoints = Vec::new();
        let result = executor
            .execute_with_cancel_and_hook(
                &batch,
                &BatchRuntimeOverrides::default(),
                std::future::pending::<()>(),
                |snapshot| {
                    checkpoints.push((
                        snapshot.status,
                        snapshot.results.len(),
                        snapshot.skipped_task_ids.len(),
                    ));
                },
            )
            .await;

        assert_eq!(result.status, BatchStatus::Completed);
        assert_eq!(result.exit_code(), 0);
        assert_eq!(result.summary.total_tasks, 3);
        assert_eq!(result.summary.succeeded_tasks, 3);
        assert_eq!(result.summary.skipped_tasks, 0);
        assert_eq!(result.metrics.total_prompt_tokens, 300);
        assert_eq!(result.metrics.total_generated_tokens, 120);
        // `/api/tags` called strictly once for all 3 tasks.
        assert_eq!(executor.task_executor.client_list_models_calls(), 1);
        // Two intermediate checkpoints (after task 1 and task 2).
        assert_eq!(
            checkpoints,
            vec![
                (BatchStatus::InProgress, 1, 2),
                (BatchStatus::InProgress, 2, 1),
            ]
        );
    }

    #[tokio::test]
    async fn supports_failure_isolation_and_fail_fast() {
        // 1. Failure isolation (`fail_fast = false`): task 2 fails validation, task 3 still runs.
        let iso_client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f1(): pass"),
                FakeAction::ok("   "), // validation_failed
                FakeAction::ok("def f3(): pass"),
            ],
        );
        let iso_exec = BatchExecutor::new(iso_client, true);
        let iso_res = iso_exec
            .execute(&sample_batch(false), &BatchRuntimeOverrides::default())
            .await;
        assert_eq!(iso_res.status, BatchStatus::PartialSuccess);
        assert_eq!(iso_res.exit_code(), exit_codes::PARTIAL_FAILURE);
        assert_eq!(iso_res.summary.succeeded_tasks, 2);
        assert_eq!(iso_res.summary.validation_failed_tasks, 1);
        assert_eq!(iso_res.summary.skipped_tasks, 0);

        // 2. Fail-fast (`fail_fast = true`): stops after task 2; task 3 skipped without calling chat.
        let ff_client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f1(): pass"),
                FakeAction::Err(WorkerError::OllamaUnavailable {
                    message: "connection reset".into(),
                }),
                FakeAction::ok("def f3(): pass"),
            ],
        );
        let ff_exec = BatchExecutor::new(ff_client, true);
        let ff_res = ff_exec
            .execute(&sample_batch(true), &BatchRuntimeOverrides::default())
            .await;
        assert_eq!(ff_res.status, BatchStatus::PartialSuccess);
        assert_eq!(ff_res.exit_code(), exit_codes::PARTIAL_FAILURE);
        assert_eq!(ff_res.summary.executed_tasks, 2);
        assert_eq!(ff_res.summary.succeeded_tasks, 1);
        assert_eq!(ff_res.summary.error_tasks, 1);
        assert_eq!(ff_res.skipped_task_ids, vec!["t3"]);
        assert_eq!(ff_exec.task_executor.client_chat_calls(), 2);
    }

    #[tokio::test]
    async fn preserves_completed_results_on_cancellation_and_batch_timeout() {
        // 1. Cancellation during task 2 preserves task 1 and skips task 3.
        let cancel_client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f1(): pass"),
                FakeAction::Ok {
                    content: "slow".into(),
                    prompt_tokens: 50,
                    eval_tokens: 20,
                    total_ns: 200_000_000,
                    prompt_ns: 50_000_000,
                    eval_ns: 150_000_000,
                    load_ns: 0,
                    delay_ms: 5_000,
                    ttft_ms: None,
                },
                FakeAction::ok("def f3(): pass"),
            ],
        );
        let cancel_exec = BatchExecutor::new(cancel_client, false);
        let cancel_res = cancel_exec
            .execute_with_cancel(
                &sample_batch(false),
                &BatchRuntimeOverrides::default(),
                tokio::time::sleep(Duration::from_millis(60)),
            )
            .await;
        assert_eq!(cancel_res.status, BatchStatus::Cancelled);
        assert_eq!(cancel_res.exit_code(), exit_codes::CANCELLED);
        assert_eq!(cancel_res.summary.succeeded_tasks, 1);
        assert_eq!(cancel_res.summary.cancelled_tasks, 1);
        assert_eq!(cancel_res.results[0].task_id, "t1");
        assert_eq!(cancel_res.results[0].status, TaskStatus::Success);
        assert_eq!(cancel_res.results[1].task_id, "t2");
        assert_eq!(cancel_res.results[1].status, TaskStatus::Cancelled);
        assert_eq!(cancel_res.skipped_task_ids, vec!["t3"]);
        assert_eq!(
            cancel_res.error.as_ref().map(|e| e.category),
            Some(ErrorCategory::Cancelled)
        );

        // 2. Overall batch_timeout_secs preserves task 1, times out task 2, skips task 3.
        let timeout_client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f1(): pass"),
                FakeAction::Ok {
                    content: "too slow".into(),
                    prompt_tokens: 50,
                    eval_tokens: 20,
                    total_ns: 200_000_000,
                    prompt_ns: 50_000_000,
                    eval_ns: 150_000_000,
                    load_ns: 0,
                    delay_ms: 2_500,
                    ttft_ms: None,
                },
                FakeAction::ok("def f3(): pass"),
            ],
        );
        let timeout_exec = BatchExecutor::new(timeout_client, false);
        let mut timeout_batch = sample_batch(false);
        timeout_batch.policy.batch_timeout_secs = Some(1);
        let timeout_res = timeout_exec
            .execute(&timeout_batch, &BatchRuntimeOverrides::default())
            .await;
        assert_eq!(timeout_res.status, BatchStatus::TimedOut);
        assert_eq!(timeout_res.exit_code(), exit_codes::TIMEOUT);
        assert_eq!(timeout_res.summary.succeeded_tasks, 1);
        assert_eq!(timeout_res.summary.timed_out_tasks, 1);
        assert_eq!(timeout_res.skipped_task_ids, vec!["t3"]);
        assert!(
            timeout_res.results[1]
                .error
                .as_ref()
                .is_some_and(|e| e.message.contains("overall batch timeout"))
        );
        // Unattributed duration captures the aborted task's wall time instead of inflating batch_overhead_ms.
        assert!(timeout_res.metrics.unattributed_duration_ms >= 900);
        assert!(timeout_res.metrics.batch_overhead_ms < 250);
    }

    #[tokio::test]
    async fn enforces_max_in_flight_one_and_handles_pre_task_cancellation() {
        use std::pin::Pin;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        use std::task::{Context, Poll};

        struct FlagCancel(Arc<AtomicBool>);
        impl Future for FlagCancel {
            type Output = ();
            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                if self.0.load(Ordering::SeqCst) {
                    Poll::Ready(())
                } else {
                    Poll::Pending
                }
            }
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        let flag_for_cb = Arc::clone(&cancel_flag);
        let mut client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f1(): pass"),
                FakeAction::ok("def f2(): pass"),
                FakeAction::ok("def f3(): pass"),
            ],
        );
        client.on_chat_complete = Some(Box::new(move |idx| {
            if idx == 0 {
                flag_for_cb.store(true, Ordering::SeqCst);
            }
        }));

        let executor = BatchExecutor::new(client, true);
        let res = executor
            .execute_with_cancel(
                &sample_batch(false),
                &BatchRuntimeOverrides::default(),
                FlagCancel(cancel_flag),
            )
            .await;

        assert_eq!(executor.task_executor.client_max_in_flight(), 1);
        assert_eq!(executor.task_executor.client_chat_calls(), 1);
        assert_eq!(res.status, BatchStatus::Cancelled);
        assert_eq!(res.exit_code(), exit_codes::CANCELLED);
        assert_eq!(res.summary.succeeded_tasks, 1);
        assert_eq!(res.summary.cancelled_tasks, 0);
        assert_eq!(res.results.len(), 1);
        assert_eq!(res.skipped_task_ids, vec!["t2", "t3"]);
    }

    #[tokio::test]
    async fn model_cache_recovers_after_tags_failure_and_isolates_mixed_models() {
        // 1. `/api/tags` fails on task 1, recovers on task 2, then cached for task 3.
        let client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![
                FakeAction::ok("def f2(): pass"),
                FakeAction::ok("def f3(): pass"),
            ],
        );
        client
            .list_models_errors
            .lock()
            .unwrap()
            .push_back(Some(WorkerError::OllamaUnavailable {
                message: "starting".into(),
            }));
        let executor = BatchExecutor::new(client, true);
        let res = executor
            .execute(&sample_batch(false), &BatchRuntimeOverrides::default())
            .await;

        assert_eq!(res.status, BatchStatus::PartialSuccess);
        assert_eq!(res.exit_code(), exit_codes::PARTIAL_FAILURE);
        assert_eq!(res.summary.error_tasks, 1);
        assert_eq!(res.summary.succeeded_tasks, 2);
        assert_eq!(executor.task_executor.client_list_models_calls(), 2);

        // 2. Mixed models: task 1 requests missing model, task 2 requests installed model -> `/api/tags` called ONCE.
        let mixed_client = ScriptedOllamaClient::new(
            &["qwen2.5-coder:3b"],
            vec![FakeAction::ok("def f2(): pass")],
        );
        let mixed_exec = BatchExecutor::new(mixed_client, true);
        let mut t1 = BatchTaskItem::new("m1", "Write f1");
        t1.model = Some("missing:7b".into());
        let mut t2 = BatchTaskItem::new("m2", "Write f2");
        t2.model = Some("qwen2.5-coder:3b".into());
        let mixed_batch = BatchTask::new("mixed-models", vec![t1, t2]);
        let mixed_res = mixed_exec
            .execute(&mixed_batch, &BatchRuntimeOverrides::default())
            .await;

        assert_eq!(mixed_res.status, BatchStatus::PartialSuccess);
        assert_eq!(
            mixed_res.results[0].exit_code(),
            exit_codes::MODEL_NOT_FOUND
        );
        assert_eq!(mixed_res.results[1].status, TaskStatus::Success);
        assert_eq!(mixed_exec.task_executor.client_list_models_calls(), 1);
    }

    #[test]
    fn validates_default_json_schema_blank_item_model_and_metrics_saturation() {
        // Invalid json_schema in defaults fails envelope validation immediately.
        let mut bad_schema = sample_batch(false);
        bad_schema.defaults.output_expectation = Some(OutputExpectation {
            json_schema: Some(serde_json::json!({ "type": 123 })),
            ..Default::default()
        });
        assert_eq!(bad_schema.validate_envelope().unwrap_err().exit_code(), 2);

        // Explicit blank item.model does NOT silently inherit defaults.model.
        let mut blank_item_batch = sample_batch(false);
        blank_item_batch.tasks[0].model = Some("   ".into());
        let resolved = blank_item_batch.resolve_task_request(
            &blank_item_batch.tasks[0],
            &BatchRuntimeOverrides::default(),
        );
        assert!(resolved.model.is_empty());
        assert_eq!(resolved.validate().unwrap_err().exit_code(), 2);

        // Saturation & non-finite guard in BatchMetrics.
        let huge = TaskResult {
            schema_version: SCHEMA_VERSION.to_owned(),
            task_id: "huge".into(),
            status: TaskStatus::Success,
            model: "m".into(),
            output: Some("ok".into()),
            raw_output: None,
            done_reason: Some("stop".into()),
            validation: None,
            metrics: Some(InferenceMetrics {
                worker_overhead_ms: u64::MAX,
                ttft_ms: Some(u64::MAX),
                total_duration_ms: u64::MAX,
                load_duration_ms: u64::MAX,
                prompt_eval_duration_ms: 1,
                eval_duration_ms: 1,
                prompt_tokens: u64::MAX,
                generated_tokens: u64::MAX,
                prompt_tokens_per_second: 0.0,
                generation_tokens_per_second: 0.0,
            }),
            error: None,
        };
        let agg = BatchMetrics::from_results_with_unattributed(
            &[huge.clone(), huge],
            Duration::from_millis(100),
            50,
        );
        assert_eq!(agg.total_prompt_tokens, u64::MAX);
        assert_eq!(agg.batch_overhead_ms, 0);
        assert!(agg.avg_prompt_tokens_per_second.is_finite());
        assert!(agg.avg_generation_tokens_per_second.is_finite());
        assert!(round2(f64::INFINITY).is_finite());
        assert!(round2(f64::NAN).is_finite());
    }

    impl TaskExecutor<ScriptedOllamaClient> {
        fn client_list_models_calls(&self) -> usize {
            self.client().list_models_calls.load(Ordering::SeqCst)
        }

        fn client_chat_calls(&self) -> usize {
            self.client().chat_calls.load(Ordering::SeqCst)
        }

        fn client_max_in_flight(&self) -> usize {
            self.client().max_in_flight.load(Ordering::SeqCst)
        }
    }
}
