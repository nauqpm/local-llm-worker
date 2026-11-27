use std::process::Command;

use ollama_worker::{BenchmarkReport, BenchmarkSuite};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn cli_benchmark_runs_chat_generate_streaming_and_cold_start_against_mock() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "version": "0.6.2" })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{ "name": "qwen2.5-coder:3b", "size": 1929912432u64 }]
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{ "name": "qwen2.5-coder:3b", "size": 2400000000u64 }]
        })))
        .mount(&server)
        .await;

    let stream_body = concat!(
        "{\"model\":\"qwen2.5-coder:3b\",\"message\":{\"role\":\"assistant\",\"content\":\"def is_prime(n):\"},\"done\":false}\n",
        "{\"model\":\"qwen2.5-coder:3b\",\"message\":{\"role\":\"assistant\",\"content\":\" return n > 1\"},\"done\":true,\"done_reason\":\"stop\",\"total_duration\":500000000,\"load_duration\":50000000,\"prompt_eval_count\":45,\"prompt_eval_duration\":100000000,\"eval_count\":15,\"eval_duration\":300000000}\n"
    );
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(stream_body, "application/x-ndjson"))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "qwen2.5-coder:3b",
            "response": "def is_prime(n): return n > 1",
            "done": true,
            "done_reason": "stop",
            "total_duration": 450000000u64,
            "load_duration": 0u64,
            "prompt_eval_count": 40u64,
            "prompt_eval_duration": 90000000u64,
            "eval_count": 15u64,
            "eval_duration": 300000000u64
        })))
        .mount(&server)
        .await;

    let suite_path = "benchmarks/default_suite.json";
    let suite_raw = std::fs::read_to_string(suite_path).expect("default_suite.json should exist");
    let suite: BenchmarkSuite =
        serde_json::from_str(&suite_raw).expect("default_suite.json should parse");
    assert_eq!(suite.scenarios.len(), 2);

    let out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "benchmark",
            "--model",
            "qwen2.5-coder:3b",
            "--suite",
            suite_path,
            "--endpoint",
            "both",
            "--stream",
            "--cold-start",
            "--iterations",
            "1",
        ])
        .output()
        .expect("benchmark CLI should run");

    assert_eq!(out.status.code(), Some(0));
    let report: BenchmarkReport =
        serde_json::from_slice(&out.stdout).expect("stdout must be valid BenchmarkReport JSON");
    assert_eq!(report.model, "qwen2.5-coder:3b");
    assert!(report.stream);
    // 2 scenarios * 2 endpoints (chat + generate) * 1 iteration = 4 runs
    assert_eq!(report.runs.len(), 4);
    assert_eq!(report.summary.succeeded_runs, 4);
    assert_eq!(report.summary.failed_runs, 0);

    let chat_run = report
        .runs
        .iter()
        .find(|r| r.endpoint == "chat")
        .expect("chat run should exist");
    assert!(chat_run.cold_start_requested);
    assert_eq!(chat_run.running_model_size_bytes, Some(2_400_000_000));
    assert!(chat_run.metrics.as_ref().and_then(|m| m.ttft_ms).is_some());
}
