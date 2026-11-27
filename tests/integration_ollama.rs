use std::net::TcpListener;
use std::process::Command;
use std::time::Duration;

use ollama_worker::{
    ErrorCategory, HttpOllamaClient, TaskExecutor, TaskRequest, TaskResult, TaskStatus,
    WorkerConfig,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn tags_payload_with_qwen() -> Value {
    json!({
        "models": [
            {
                "name": "qwen2.5-coder:3b",
                "model": "qwen2.5-coder:3b",
                "size": 1929912432u64,
                "details": {
                    "format": "gguf",
                    "family": "qwen2",
                    "parameter_size": "3.1B",
                    "quantization_level": "Q4_K_M"
                }
            }
        ]
    })
}

fn chat_success_payload(content: &str) -> Value {
    json!({
        "model": "qwen2.5-coder:3b",
        "created_at": "2026-11-27T06:30:00Z",
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": true,
        "done_reason": "stop",
        "total_duration": 1_200_000_000u64,
        "load_duration": 150_000_000u64,
        "prompt_eval_count": 41u64,
        "prompt_eval_duration": 200_000_000u64,
        "eval_count": 32u64,
        "eval_duration": 800_000_000u64
    })
}

fn executor_for(
    base_url: &str,
    timeout_secs: u64,
    verify_model: bool,
) -> TaskExecutor<HttpOllamaClient> {
    let config = WorkerConfig::new(base_url, timeout_secs, false, None)
        .expect("wiremock loopback URL should be accepted")
        .with_verify_model(verify_model);
    let client = HttpOllamaClient::new(config).expect("HTTP client should initialize");
    TaskExecutor::new(client, verify_model)
}

#[tokio::test]
async fn health_and_models_succeed_against_mock_ollama() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "version": "0.6.2" })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {
                    "name": "qwen2.5-coder:3b",
                    "size": 2400000000u64,
                    "size_vram": 0u64,
                    "context_length": 8192u32
                }
            ]
        })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload_with_qwen()))
        .mount(&server)
        .await;

    let executor = executor_for(&server.uri(), 10, true);
    let health = executor
        .health(&server.uri())
        .await
        .expect("health should succeed");
    assert_eq!(health.status, "ok");
    assert_eq!(health.ollama_version, "0.6.2");
    assert_eq!(health.running_models.len(), 1);
    assert_eq!(health.running_models[0].name, "qwen2.5-coder:3b");

    let models = executor.models().await.expect("models should succeed");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "qwen2.5-coder:3b");
}

#[tokio::test]
async fn run_succeeds_and_returns_structured_metrics() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload_with_qwen()))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(chat_success_payload(
                "def is_prime(n: int) -> bool:\n    return n > 1",
            )),
        )
        .mount(&server)
        .await;

    let executor = executor_for(&server.uri(), 10, true);
    let req = TaskRequest::new("task-001", "qwen2.5-coder:3b", "Write is_prime in Python");

    let result = executor.execute(&req).await;
    assert_eq!(result.status, TaskStatus::Success);
    assert_eq!(result.exit_code(), 0);
    assert_eq!(result.done_reason.as_deref(), Some("stop"));
    assert!(result.error.is_none());
    let metrics = result.metrics.expect("metrics should exist");
    assert_eq!(metrics.prompt_tokens, 41);
    assert_eq!(metrics.generated_tokens, 32);
    assert_eq!(metrics.generation_tokens_per_second, 40.0);
}

#[tokio::test]
async fn run_reports_model_not_found_from_tags_and_from_404() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload_with_qwen()))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(404)
                .set_body_json(json!({ "error": "model 'missing:3b' not found" })),
        )
        .mount(&server)
        .await;

    let verified_exec = executor_for(&server.uri(), 10, true);
    let res_tags = verified_exec
        .execute(&TaskRequest::new("t-tags", "missing:3b", "hi"))
        .await;
    assert_eq!(res_tags.exit_code(), 4);
    assert_eq!(
        res_tags.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::ModelNotFound)
    );

    let direct_exec = executor_for(&server.uri(), 10, false);
    let res_404 = direct_exec
        .execute(&TaskRequest::new("t-404", "missing:3b", "hi"))
        .await;
    assert_eq!(res_404.exit_code(), 4);
    assert_eq!(
        res_404.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::ModelNotFound)
    );
}

#[tokio::test]
async fn run_handles_connection_failure_http_503_invalid_json_and_schema_mismatch() {
    // 1. Guaranteed-closed loopback port for connection failure.
    let listener = TcpListener::bind("127.0.0.1:0").expect("should bind ephemeral port");
    let closed_port = listener.local_addr().expect("should get local addr").port();
    drop(listener);

    let down_exec = executor_for(&format!("http://127.0.0.1:{closed_port}"), 5, false);
    let conn_res = down_exec
        .execute(&TaskRequest::new("t-conn", "qwen2.5-coder:3b", "hi"))
        .await;
    assert_eq!(conn_res.status, TaskStatus::Error);
    assert_eq!(conn_res.exit_code(), 3);
    assert_eq!(
        conn_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::OllamaUnavailable)
    );

    // 2. HTTP 503 Server Error.
    let server_503 = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({ "error": "server busy" })))
        .mount(&server_503)
        .await;
    let res_503 = executor_for(&server_503.uri(), 5, false)
        .execute(&TaskRequest::new("t-503", "qwen2.5-coder:3b", "hi"))
        .await;
    assert_eq!(res_503.exit_code(), 3);
    assert_eq!(res_503.error.as_ref().map(|e| e.retryable), Some(true));

    // 3. Invalid JSON body.
    let server_bad_json = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string("{not valid json"))
        .mount(&server_bad_json)
        .await;
    let res_bad_json = executor_for(&server_bad_json.uri(), 5, false)
        .execute(&TaskRequest::new("t-bad-json", "qwen2.5-coder:3b", "hi"))
        .await;
    assert_eq!(res_bad_json.exit_code(), 3);
    assert_eq!(
        res_bad_json.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::ProtocolError)
    );

    // 4. Unexpected JSON schema (missing `message`).
    let server_bad_schema = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "unexpected": true })))
        .mount(&server_bad_schema)
        .await;
    let res_bad_schema = executor_for(&server_bad_schema.uri(), 5, false)
        .execute(&TaskRequest::new("t-schema", "qwen2.5-coder:3b", "hi"))
        .await;
    assert_eq!(res_bad_schema.exit_code(), 3);
    assert_eq!(
        res_bad_schema.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::ProtocolError)
    );
}

#[tokio::test]
async fn run_handles_timeout_and_empty_model_output() {
    let server_slow = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(2))
                .set_body_json(chat_success_payload("late")),
        )
        .mount(&server_slow)
        .await;

    let res_timeout = executor_for(&server_slow.uri(), 1, false)
        .execute(&TaskRequest::new("t-timeout", "qwen2.5-coder:3b", "slow"))
        .await;
    assert_eq!(res_timeout.status, TaskStatus::Timeout);
    assert_eq!(res_timeout.exit_code(), 5);
    assert_eq!(
        res_timeout.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::Timeout)
    );

    let server_empty = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_success_payload("   ")))
        .mount(&server_empty)
        .await;

    let res_empty = executor_for(&server_empty.uri(), 5, false)
        .execute(&TaskRequest::new("t-empty", "qwen2.5-coder:3b", "blank"))
        .await;
    assert_eq!(res_empty.status, TaskStatus::ValidationFailed);
    assert_eq!(res_empty.exit_code(), 6);
    assert!(res_empty.metrics.is_some());
    assert_eq!(res_empty.raw_output.as_deref(), Some("   "));
}

#[tokio::test]
async fn cli_binary_enforces_stdout_json_and_exit_code_contracts() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/version"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "version": "0.6.2" })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/ps"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "models": [] })))
        .mount(&server)
        .await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload_with_qwen()))
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_success_payload("return True")))
        .mount(&server)
        .await;

    // 1. Successful `run` -> exit 0 + valid TaskResult JSON on stdout.
    let ok_out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "run",
            "--task-id",
            "cli-001",
            "--model",
            "qwen2.5-coder:3b",
            "--task",
            "Return True",
        ])
        .output()
        .expect("ollama-worker binary should execute");
    assert_eq!(ok_out.status.code(), Some(0));
    let parsed: TaskResult =
        serde_json::from_slice(&ok_out.stdout).expect("stdout must be valid TaskResult JSON");
    assert_eq!(parsed.status, TaskStatus::Success);
    assert_eq!(parsed.task_id, "cli-001");
    assert_eq!(parsed.output.as_deref(), Some("return True"));

    // 2. Missing model -> exit 4 + structured JSON on stdout.
    let missing_out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "run",
            "--model",
            "nonexistent:3b",
            "--task",
            "Hello",
        ])
        .output()
        .expect("ollama-worker binary should execute");
    assert_eq!(missing_out.status.code(), Some(4));
    let missing_res: TaskResult = serde_json::from_slice(&missing_out.stdout)
        .expect("error output must still be valid TaskResult JSON");
    assert_eq!(missing_res.status, TaskStatus::Error);
    assert_eq!(
        missing_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::ModelNotFound)
    );

    // 3. Remote URL blocked -> exit 2 + structured JSON on stdout.
    let remote_out = Command::new(bin)
        .args([
            "--base-url",
            "http://192.168.1.50:11434",
            "run",
            "--model",
            "qwen2.5-coder:3b",
            "--task",
            "Hello",
        ])
        .output()
        .expect("ollama-worker binary should execute");
    assert_eq!(remote_out.status.code(), Some(2));
    let remote_res: TaskResult = serde_json::from_slice(&remote_out.stdout)
        .expect("remote-block error must emit valid TaskResult JSON");
    assert_eq!(
        remote_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );

    // 4. `health --json` -> exit 0 + valid JSON.
    let health_out = Command::new(bin)
        .args(["--base-url", &server.uri(), "health", "--json"])
        .output()
        .expect("ollama-worker binary should execute");
    assert_eq!(health_out.status.code(), Some(0));
    let health_json: Value =
        serde_json::from_slice(&health_out.stdout).expect("health --json must emit valid JSON");
    assert_eq!(health_json["status"], "ok");
    assert_eq!(health_json["ollama_version"], "0.6.2");
}
