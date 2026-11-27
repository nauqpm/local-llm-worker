//! Inference timing and throughput metrics derived from Native Ollama responses.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::client::ChatResponse;

const NANOS_PER_MILLI: u64 = 1_000_000;
const NANOS_PER_SEC_F64: f64 = 1_000_000_000.0;

/// Performance metrics reported alongside a task result.
///
/// Explicitly separates `worker_overhead_ms` (client serialization, HTTP round-trip,
/// and validation) from Ollama's reported `total_duration_ms`, and includes optional
/// `ttft_ms` (Time to First Token) when streaming is enabled.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub worker_overhead_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    pub total_duration_ms: u64,
    pub load_duration_ms: u64,
    pub prompt_eval_duration_ms: u64,
    pub eval_duration_ms: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
    pub prompt_tokens_per_second: f64,
    pub generation_tokens_per_second: f64,
}

impl InferenceMetrics {
    /// Computes normalized millisecond durations and token throughput from an
    /// Ollama [`ChatResponse`] and the measured client-side `wall_clock` duration.
    #[must_use]
    pub fn from_chat_response(response: &ChatResponse, wall_clock: Duration) -> Self {
        Self::from_chat_response_with_ttft(response, wall_clock, None)
    }

    /// Computes metrics including optional client-measured Time to First Token (`ttft_ms`).
    #[must_use]
    pub fn from_chat_response_with_ttft(
        response: &ChatResponse,
        wall_clock: Duration,
        ttft_ms: Option<u64>,
    ) -> Self {
        let total_duration_ms = ns_to_ms(response.total_duration);
        let load_duration_ms = ns_to_ms(response.load_duration);
        let prompt_eval_duration_ms = ns_to_ms(response.prompt_eval_duration);
        let eval_duration_ms = ns_to_ms(response.eval_duration);

        let wall_ms = u64::try_from(wall_clock.as_millis()).unwrap_or(u64::MAX);
        let worker_overhead_ms = if response.total_duration.is_some() {
            wall_ms.saturating_sub(total_duration_ms)
        } else {
            wall_ms
        };

        let prompt_tokens = response.prompt_eval_count.unwrap_or(0);
        let generated_tokens = response.eval_count.unwrap_or(0);

        let prompt_tokens_per_second =
            tokens_per_second(prompt_tokens, response.prompt_eval_duration.unwrap_or(0));
        let generation_tokens_per_second =
            tokens_per_second(generated_tokens, response.eval_duration.unwrap_or(0));

        Self {
            worker_overhead_ms,
            ttft_ms,
            total_duration_ms,
            load_duration_ms,
            prompt_eval_duration_ms,
            eval_duration_ms,
            prompt_tokens,
            generated_tokens,
            prompt_tokens_per_second,
            generation_tokens_per_second,
        }
    }
}

#[must_use]
fn ns_to_ms(nanos: Option<u64>) -> u64 {
    nanos.unwrap_or(0) / NANOS_PER_MILLI
}

#[must_use]
fn tokens_per_second(tokens: u64, duration_ns: u64) -> f64 {
    if tokens == 0 || duration_ns == 0 {
        return 0.0;
    }
    let seconds = (duration_ns as f64) / NANOS_PER_SEC_F64;
    let rate = (tokens as f64) / seconds;
    (rate * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{ChatMessage, ChatRole};

    fn sample_response() -> ChatResponse {
        ChatResponse {
            model: "qwen2.5-coder:3b".into(),
            created_at: None,
            message: ChatMessage {
                role: ChatRole::Assistant,
                content: "ok".into(),
            },
            done: true,
            done_reason: Some("stop".into()),
            total_duration: Some(1_500_000_000),
            load_duration: Some(200_000_000),
            prompt_eval_count: Some(50),
            prompt_eval_duration: Some(250_000_000),
            eval_count: Some(20),
            eval_duration: Some(1_000_000_000),
        }
    }

    #[test]
    fn computes_durations_overhead_ttft_and_tokens_per_second() {
        let resp = sample_response();
        let metrics = InferenceMetrics::from_chat_response_with_ttft(
            &resp,
            Duration::from_millis(1_535),
            Some(420),
        );

        assert_eq!(metrics.total_duration_ms, 1_500);
        assert_eq!(metrics.worker_overhead_ms, 35);
        assert_eq!(metrics.ttft_ms, Some(420));
        assert_eq!(metrics.load_duration_ms, 200);
        assert_eq!(metrics.prompt_eval_duration_ms, 250);
        assert_eq!(metrics.eval_duration_ms, 1_000);
        assert_eq!(metrics.prompt_tokens, 50);
        assert_eq!(metrics.generated_tokens, 20);
        assert!((metrics.prompt_tokens_per_second - 200.0).abs() < f64::EPSILON);
        assert!((metrics.generation_tokens_per_second - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn handles_zero_and_missing_durations_without_nan_or_underflow() {
        let mut resp = sample_response();
        resp.total_duration = Some(2_000_000_000);
        resp.prompt_eval_duration = Some(0);
        resp.eval_duration = None;

        let metrics = InferenceMetrics::from_chat_response(&resp, Duration::from_millis(1_000));
        assert_eq!(metrics.worker_overhead_ms, 0);
        assert_eq!(metrics.ttft_ms, None);
        assert_eq!(metrics.prompt_tokens_per_second, 0.0);
        assert_eq!(metrics.generation_tokens_per_second, 0.0);
    }
}
