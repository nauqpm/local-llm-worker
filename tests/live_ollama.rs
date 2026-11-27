//! Opt-in live integration tests against a real Ollama instance.
//!
//! These tests are marked `#[ignore]` and never run during default `cargo test` or CI.
//! To execute manually against a local Ollama server:
//!
//! ```bash
//! OLLAMA_LIVE_TEST=1 cargo test --test live_ollama -- --ignored --nocapture
//! ```

use ollama_worker::{
    DEFAULT_OLLAMA_BASE_URL, HttpOllamaClient, TaskExecutor, TaskRequest, TaskStatus, WorkerConfig,
};

fn live_test_enabled() -> bool {
    std::env::var("OLLAMA_LIVE_TEST")
        .is_ok_and(|val| val == "1" || val.eq_ignore_ascii_case("true"))
}

fn live_executor() -> (TaskExecutor<HttpOllamaClient>, String) {
    let base_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_BASE_URL.to_owned());
    let allow_remote = std::env::var("OLLAMA_ALLOW_REMOTE").is_ok_and(|v| v == "1");
    let config = WorkerConfig::new(&base_url, 120, allow_remote, None)
        .expect("live test WorkerConfig should be valid");
    let client = HttpOllamaClient::new(config.clone()).expect("HTTP client should build");
    (
        TaskExecutor::new(client, true),
        config.base_url_str().to_owned(),
    )
}

#[tokio::test]
#[ignore = "opt-in live test; set OLLAMA_LIVE_TEST=1 and run with --ignored"]
async fn live_health_and_models_check() {
    if !live_test_enabled() {
        eprintln!("skipping live test: set OLLAMA_LIVE_TEST=1 to enable");
        return;
    }

    let (executor, base_url) = live_executor();
    let health = executor
        .health(&base_url)
        .await
        .expect("live Ollama server should respond to /api/version and /api/ps");
    assert_eq!(health.status, "ok");
    assert!(!health.ollama_version.is_empty());

    let models = executor
        .models()
        .await
        .expect("live Ollama server should respond to /api/tags");
    assert!(
        !models.is_empty(),
        "expected at least one installed model on live Ollama server"
    );
}

#[tokio::test]
#[ignore = "opt-in live test; set OLLAMA_LIVE_TEST=1 and run with --ignored"]
async fn live_run_simple_python_task() {
    if !live_test_enabled() {
        eprintln!("skipping live test: set OLLAMA_LIVE_TEST=1 to enable");
        return;
    }

    let model =
        std::env::var("OLLAMA_WORKER_MODEL").unwrap_or_else(|_| "qwen2.5-coder:3b".to_owned());
    let (executor, _) = live_executor();
    let req = TaskRequest::new(
        "live-001",
        &model,
        "Write a concise Python function `is_prime(n: int) -> bool`. Return only the code.",
    );

    let result = executor.execute(&req).await;
    assert_eq!(
        result.status,
        TaskStatus::Success,
        "live result: {result:?}"
    );
    let metrics = result.metrics.expect("live metrics should be present");
    assert!(metrics.prompt_tokens > 0);
    assert!(metrics.generated_tokens > 0);
}
