//! Reproducible local-LLM benchmark runner separating worker overhead from model inference.

use std::str::FromStr;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::client::{ChatRequest, ChatResponse, GenerateRequest, OllamaClient, model_matches};
use crate::config::InferenceOptions;
use crate::error::{ErrorDetail, WorkerError, WorkerResult};
use crate::metrics::InferenceMetrics;
use crate::prompt::PromptBuilder;
use crate::task::{SCHEMA_VERSION, TaskRequest, TaskType};

/// Maximum iterations allowed per scenario in a single benchmark invocation.
pub const MAX_BENCHMARK_ITERATIONS: usize = 20;

/// Ollama endpoint(s) evaluated during a benchmark run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkEndpoint {
    #[default]
    Chat,
    Generate,
    Both,
}

impl FromStr for BenchmarkEndpoint {
    type Err = WorkerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "chat" => Ok(Self::Chat),
            "generate" => Ok(Self::Generate),
            "both" => Ok(Self::Both),
            other => Err(WorkerError::InvalidInput {
                message: format!(
                    "unsupported benchmark endpoint '{other}'; expected one of: chat, generate, both"
                ),
            }),
        }
    }
}

fn default_benchmark_max_tokens() -> u32 {
    256
}

/// Single reproducible prompt scenario in a [`BenchmarkSuite`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkScenario {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub task_type: TaskType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub instruction: String,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default = "default_benchmark_max_tokens")]
    pub max_tokens: u32,
}

/// Collection of benchmark scenarios covering small, medium, and larger prompts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub scenarios: Vec<BenchmarkScenario>,
}

impl BenchmarkSuite {
    /// Returns the built-in reproducible S1 (small), S2 (medium), S3 (large) suite.
    #[must_use]
    pub fn default_suite() -> Self {
        let medium_context = concat!(
            "from dataclasses import dataclass\n\n",
            "@dataclass\n",
            "class TokenBucket:\n",
            "    capacity: int\n",
            "    refill_per_sec: float\n",
            "    tokens: float\n",
            "    last_ts: float\n\n",
            "    def allow(self, now: float, cost: int = 1) -> bool:\n",
            "        if cost <= 0:\n",
            "            raise ValueError('cost must be positive')\n",
            "        elapsed = max(0.0, now - self.last_ts)\n",
            "        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_per_sec)\n",
            "        self.last_ts = now\n",
            "        if self.tokens >= cost:\n",
            "            self.tokens -= cost\n",
            "            return True\n",
            "        return False\n"
        );

        let mut large_context = String::from(
            "class CacheStore:\n    def __init__(self):\n        self._items = {}\n\n",
        );
        for idx in 1..=28 {
            large_context.push_str(&format!(
                "    def normalize_key_{idx:02}(self, raw: str) -> str:\n        cleaned = raw.strip().lower()\n        if not cleaned:\n            raise ValueError('empty key')\n        return f'ns{idx:02}:{{cleaned}}'\n\n"
            ));
        }
        large_context.push_str(
            "    def get_or_default(self, key: str, default: str) -> str:\n        # Bug: returns default when stored value is empty string\n        return self._items.get(key) or default\n",
        );

        Self {
            scenarios: vec![
                BenchmarkScenario {
                    id: "s1_small".into(),
                    label: "Small prompt (~100-200 tokens): isolated Python function".into(),
                    task_type: TaskType::GenerateFunction,
                    language: Some("python".into()),
                    instruction: "Implement `is_prime(n: int) -> bool` using only the Python standard library.".into(),
                    context: String::new(),
                    constraints: vec![
                        "Handle n < 2 cleanly".into(),
                        "Return only the function implementation".into(),
                    ],
                    max_tokens: 256,
                },
                BenchmarkScenario {
                    id: "s2_medium".into(),
                    label: "Medium prompt (~300-600 tokens): unit tests for TokenBucket".into(),
                    task_type: TaskType::GenerateUnitTest,
                    language: Some("python".into()),
                    instruction: "Write concise pytest unit tests for TokenBucket.allow covering refill, exhaustion, and invalid cost.".into(),
                    context: medium_context.into(),
                    constraints: vec![
                        "Use pytest style functions".into(),
                        "Return only test code".into(),
                    ],
                    max_tokens: 384,
                },
                BenchmarkScenario {
                    id: "s3_large".into(),
                    label: "Large prompt (~1500-2000 tokens): isolated bug fix in CacheStore".into(),
                    task_type: TaskType::FixBug,
                    language: Some("python".into()),
                    instruction: "Fix CacheStore.get_or_default so an explicitly stored empty string is returned instead of falling back to default.".into(),
                    context: large_context,
                    constraints: vec![
                        "Return only the fixed get_or_default method".into(),
                    ],
                    max_tokens: 256,
                },
            ],
        }
    }

    pub fn validate(&self) -> WorkerResult<()> {
        if self.scenarios.is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "benchmark suite must contain at least one scenario".into(),
            });
        }
        for scenario in &self.scenarios {
            if scenario.id.trim().is_empty() || scenario.instruction.trim().is_empty() {
                return Err(WorkerError::InvalidInput {
                    message: "each benchmark scenario requires non-empty id and instruction".into(),
                });
            }
        }
        Ok(())
    }
}

/// Parameters controlling a benchmark execution.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub model: String,
    pub endpoint: BenchmarkEndpoint,
    pub iterations: usize,
    pub stream: bool,
    pub cold_start: bool,
    pub temperature: f32,
    pub seed: Option<i64>,
    pub num_ctx: Option<u32>,
}

impl BenchmarkConfig {
    pub fn validate(&self) -> WorkerResult<()> {
        if self.model.trim().is_empty() {
            return Err(WorkerError::InvalidInput {
                message: "benchmark model must not be empty".into(),
            });
        }
        if self.iterations == 0 || self.iterations > MAX_BENCHMARK_ITERATIONS {
            return Err(WorkerError::InvalidInput {
                message: format!(
                    "benchmark iterations must be between 1 and {MAX_BENCHMARK_ITERATIONS}, got {}",
                    self.iterations
                ),
            });
        }
        let sample_opts = InferenceOptions {
            temperature: Some(self.temperature),
            num_ctx: self.num_ctx,
            max_tokens: Some(256),
            seed: self.seed,
            keep_alive: None,
            stream: self.stream,
        };
        sample_opts.validate()
    }
}

/// Metrics recorded for a single benchmark scenario iteration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioRunRecord {
    pub scenario_id: String,
    pub endpoint: String,
    pub iteration: usize,
    pub cold_start_requested: bool,
    pub status: String,
    pub wall_clock_ms: u64,
    pub estimated_prompt_tokens: usize,
    pub running_model_size_bytes: Option<u64>,
    pub metrics: Option<InferenceMetrics>,
    pub error: Option<ErrorDetail>,
}

/// Aggregated summary across all runs in a [`BenchmarkReport`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_runs: usize,
    pub succeeded_runs: usize,
    pub failed_runs: usize,
    pub avg_worker_overhead_ms: f64,
    pub avg_total_duration_ms: f64,
    pub avg_generation_tokens_per_second: f64,
}

/// Machine-readable report emitted by `ollama-worker benchmark`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: String,
    pub model: String,
    pub base_url: String,
    pub ollama_version: String,
    pub stream: bool,
    pub num_ctx: Option<u32>,
    pub runs: Vec<ScenarioRunRecord>,
    pub summary: BenchmarkSummary,
}

/// Executes a [`BenchmarkSuite`] against any [`OllamaClient`].
#[derive(Debug, Clone)]
pub struct BenchmarkRunner<C> {
    client: C,
    prompt_builder: PromptBuilder,
}

impl<C: OllamaClient> BenchmarkRunner<C> {
    #[must_use]
    pub fn new(client: C) -> Self {
        Self {
            client,
            prompt_builder: PromptBuilder::default(),
        }
    }

    /// Runs the configured benchmark scenarios and computes summary statistics.
    pub async fn run(
        &self,
        config: &BenchmarkConfig,
        suite: &BenchmarkSuite,
        base_url: &str,
    ) -> WorkerResult<BenchmarkReport> {
        config.validate()?;
        suite.validate()?;

        let model = config.model.trim();
        let version = self.client.version().await?.version;
        let installed = self.client.list_models().await?;
        if !installed.iter().any(|m| model_matches(&m.name, model)) {
            return Err(WorkerError::ModelNotFound {
                model: model.to_owned(),
            });
        }

        let endpoints: &[&str] = match config.endpoint {
            BenchmarkEndpoint::Chat => &["chat"],
            BenchmarkEndpoint::Generate => &["generate"],
            BenchmarkEndpoint::Both => &["chat", "generate"],
        };

        let mut runs = Vec::new();
        for scenario in &suite.scenarios {
            for &endpoint in endpoints {
                for iteration in 1..=config.iterations {
                    let is_cold = config.cold_start && iteration == 1;
                    if is_cold {
                        let unload = GenerateRequest::unload(model);
                        if let Err(err) = self.client.generate(&unload).await {
                            tracing::warn!(
                                model = model,
                                error = %err,
                                "failed to unload model before cold-start iteration"
                            );
                        }
                    }

                    let record = self
                        .run_single(config, scenario, endpoint, iteration, is_cold)
                        .await;
                    runs.push(record);
                }
            }
        }

        let summary = summarize_runs(&runs);
        Ok(BenchmarkReport {
            schema_version: SCHEMA_VERSION.to_owned(),
            model: model.to_owned(),
            base_url: base_url.to_owned(),
            ollama_version: version,
            stream: config.stream,
            num_ctx: config.num_ctx,
            runs,
            summary,
        })
    }

    async fn run_single(
        &self,
        config: &BenchmarkConfig,
        scenario: &BenchmarkScenario,
        endpoint: &str,
        iteration: usize,
        cold_start_requested: bool,
    ) -> ScenarioRunRecord {
        let model = config.model.trim();
        let mut req = TaskRequest::new(
            format!("bench-{}-{endpoint}-{iteration}", scenario.id),
            model,
            &scenario.instruction,
        );
        req.task_type = scenario.task_type;
        req.language.clone_from(&scenario.language);
        req.context.clone_from(&scenario.context);
        req.constraints.clone_from(&scenario.constraints);
        req.options = InferenceOptions {
            temperature: Some(config.temperature),
            num_ctx: config.num_ctx,
            max_tokens: Some(scenario.max_tokens),
            seed: config.seed,
            keep_alive: None,
            stream: config.stream,
        };

        let built = match self.prompt_builder.build(&req) {
            Ok(p) => p,
            Err(err) => {
                return ScenarioRunRecord {
                    scenario_id: scenario.id.clone(),
                    endpoint: endpoint.to_owned(),
                    iteration,
                    cold_start_requested,
                    status: "error".into(),
                    wall_clock_ms: 0,
                    estimated_prompt_tokens: 0,
                    running_model_size_bytes: None,
                    metrics: None,
                    error: Some(err.to_detail()),
                };
            }
        };

        let started = Instant::now();
        let outcome: WorkerResult<(ChatResponse, Option<u64>)> = if endpoint == "generate" {
            let gen_req = GenerateRequest::new(
                model,
                Some(&built.system_prompt),
                &built.user_prompt,
                &req.options,
            );
            self.client
                .generate(&gen_req)
                .await
                .map(|resp| (ChatResponse::from(resp), None))
        } else {
            let chat_req = ChatRequest::stateless(
                model,
                Some(&built.system_prompt),
                &built.user_prompt,
                &req.options,
            );
            if config.stream {
                self.client
                    .chat_stream(&chat_req)
                    .await
                    .map(|s| (s.response, s.ttft_ms))
            } else {
                self.client.chat(&chat_req).await.map(|r| (r, None))
            }
        };
        let wall_clock = started.elapsed();
        let wall_clock_ms = u64::try_from(wall_clock.as_millis()).unwrap_or(u64::MAX);

        let running_model_size_bytes =
            self.client
                .list_running_models()
                .await
                .ok()
                .and_then(|models| {
                    models
                        .into_iter()
                        .find(|m| model_matches(&m.name, model))
                        .map(|m| m.size)
                });

        match outcome {
            Ok((response, ttft_ms)) => {
                let metrics =
                    InferenceMetrics::from_chat_response_with_ttft(&response, wall_clock, ttft_ms);
                ScenarioRunRecord {
                    scenario_id: scenario.id.clone(),
                    endpoint: endpoint.to_owned(),
                    iteration,
                    cold_start_requested,
                    status: "success".into(),
                    wall_clock_ms,
                    estimated_prompt_tokens: built.estimated_prompt_tokens,
                    running_model_size_bytes,
                    metrics: Some(metrics),
                    error: None,
                }
            }
            Err(err) => ScenarioRunRecord {
                scenario_id: scenario.id.clone(),
                endpoint: endpoint.to_owned(),
                iteration,
                cold_start_requested,
                status: "error".into(),
                wall_clock_ms,
                estimated_prompt_tokens: built.estimated_prompt_tokens,
                running_model_size_bytes,
                metrics: None,
                error: Some(err.to_detail()),
            },
        }
    }
}

fn summarize_runs(runs: &[ScenarioRunRecord]) -> BenchmarkSummary {
    let total_runs = runs.len();
    let succeeded: Vec<&InferenceMetrics> = runs
        .iter()
        .filter(|r| r.status == "success")
        .filter_map(|r| r.metrics.as_ref())
        .collect();
    let succeeded_runs = succeeded.len();
    let failed_runs = total_runs.saturating_sub(succeeded_runs);

    if succeeded_runs == 0 {
        return BenchmarkSummary {
            total_runs,
            succeeded_runs: 0,
            failed_runs,
            avg_worker_overhead_ms: 0.0,
            avg_total_duration_ms: 0.0,
            avg_generation_tokens_per_second: 0.0,
        };
    }

    let denom = succeeded_runs as f64;
    let avg_overhead = round2(
        succeeded
            .iter()
            .map(|m| m.worker_overhead_ms as f64)
            .sum::<f64>()
            / denom,
    );
    let avg_total = round2(
        succeeded
            .iter()
            .map(|m| m.total_duration_ms as f64)
            .sum::<f64>()
            / denom,
    );
    let avg_gen_tps = round2(
        succeeded
            .iter()
            .map(|m| m.generation_tokens_per_second)
            .sum::<f64>()
            / denom,
    );

    BenchmarkSummary {
        total_runs,
        succeeded_runs,
        failed_runs,
        avg_worker_overhead_ms: avg_overhead,
        avg_total_duration_ms: avg_total,
        avg_generation_tokens_per_second: avg_gen_tps,
    }
}

fn round2(val: f64) -> f64 {
    (val * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_suite_covers_small_medium_and_large_prompts_within_budget() {
        let suite = BenchmarkSuite::default_suite();
        suite.validate().expect("default suite should be valid");
        assert_eq!(suite.scenarios.len(), 3);

        let builder = PromptBuilder::default();
        let mut token_counts = Vec::new();
        for s in &suite.scenarios {
            let mut req = TaskRequest::new(&s.id, "qwen2.5-coder:3b", &s.instruction);
            req.task_type = s.task_type;
            req.language.clone_from(&s.language);
            req.context.clone_from(&s.context);
            req.constraints.clone_from(&s.constraints);
            req.options.max_tokens = Some(s.max_tokens);
            let built = builder
                .build(&req)
                .unwrap_or_else(|err| panic!("scenario {} should fit budget: {err}", s.id));
            token_counts.push(built.estimated_prompt_tokens);
        }

        assert!(token_counts[0] < token_counts[1]);
        assert!(token_counts[1] < token_counts[2]);
        assert!(token_counts[2] > 1000);
    }

    #[test]
    fn endpoint_parser_validates_known_values() {
        assert_eq!(
            BenchmarkEndpoint::from_str("chat").unwrap(),
            BenchmarkEndpoint::Chat
        );
        assert_eq!(
            BenchmarkEndpoint::from_str("generate").unwrap(),
            BenchmarkEndpoint::Generate
        );
        assert_eq!(
            BenchmarkEndpoint::from_str("both").unwrap(),
            BenchmarkEndpoint::Both
        );
        assert!(BenchmarkEndpoint::from_str("invalid").is_err());
    }
}
