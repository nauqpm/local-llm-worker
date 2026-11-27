use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use ollama_worker::{
    DryRunReport, ErrorCategory, HttpOllamaClient, OutputFormat, TaskExecutor, TaskRequest,
    TaskResult, TaskStatus, TaskType, WorkerConfig,
};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after UNIX_EPOCH")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ollama-worker-test-{label}-{}-{nanos}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("temp test dir should be created");
    dir
}

fn tags_payload() -> Value {
    json!({
        "models": [
            { "name": "qwen2.5-coder:3b", "size": 1929912432u64 }
        ]
    })
}

fn chat_response(content: &str, done_reason: &str) -> Value {
    json!({
        "model": "qwen2.5-coder:3b",
        "message": { "role": "assistant", "content": content },
        "done": true,
        "done_reason": done_reason,
        "total_duration": 900_000_000u64,
        "load_duration": 50_000_000u64,
        "prompt_eval_count": 64u64,
        "prompt_eval_duration": 150_000_000u64,
        "eval_count": 40u64,
        "eval_duration": 700_000_000u64
    })
}

fn executor_for(base_url: &str, verify_model: bool) -> TaskExecutor<HttpOllamaClient> {
    let config = WorkerConfig::new(base_url, 10, false, None)
        .expect("mock URL should be valid")
        .with_verify_model(verify_model);
    let client = HttpOllamaClient::new(config).expect("client should initialize");
    TaskExecutor::new(client, verify_model)
}

#[tokio::test]
async fn all_seven_core_task_types_succeed_with_expected_validation() {
    let cases = [
        (
            TaskType::GenerateFunction,
            "```python\ndef is_prime(n: int) -> bool:\n    return n > 1\n```",
            "def is_prime(n: int) -> bool:\n    return n > 1",
            true,
        ),
        (
            TaskType::GenerateUnitTest,
            "```python\ndef test_is_prime():\n    assert is_prime(2)\n```",
            "def test_is_prime():\n    assert is_prime(2)",
            true,
        ),
        (
            TaskType::GenerateDocstring,
            "```python\n\"\"\"Check whether n is prime.\"\"\"\n```",
            "\"\"\"Check whether n is prime.\"\"\"",
            true,
        ),
        (
            TaskType::ExplainCode,
            "Checks whether the given integer is strictly greater than one.",
            "Checks whether the given integer is strictly greater than one.",
            false,
        ),
        (
            TaskType::RefactorFunction,
            "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
            "def add(a: int, b: int) -> int:\n    return a + b",
            true,
        ),
        (
            TaskType::FixBug,
            "```python\ndef divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('zero')\n    return a / b\n```",
            "def divide(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('zero')\n    return a / b",
            true,
        ),
        (
            TaskType::GenerateJson,
            "{\"ok\": true, \"items\": [1, 2]}",
            "{\"ok\": true, \"items\": [1, 2]}",
            false,
        ),
    ];

    for (task_type, raw_body, expected_output, expect_extracted) in cases {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(raw_body, "stop")))
            .mount(&server)
            .await;

        let mut req = TaskRequest::new(
            format!("task-{}", task_type.as_str()),
            "qwen2.5-coder:3b",
            "Perform the scoped coding task.",
        );
        req.task_type = task_type;
        req.language = Some("python".into());

        let result = executor_for(&server.uri(), false).execute(&req).await;
        assert_eq!(
            result.status,
            TaskStatus::Success,
            "task type {:?} failed: {:?}",
            task_type,
            result.error
        );
        assert_eq!(result.exit_code(), 0);
        assert_eq!(result.output.as_deref(), Some(expected_output));
        let validation = result.validation.expect("validation report should exist");
        assert!(validation.passed);
        assert_eq!(validation.extracted_code_block, expect_extracted);

        let received = server
            .received_requests()
            .await
            .expect("wiremock should record requests");
        let sent_json: Value =
            serde_json::from_slice(&received[0].body).expect("request body should be JSON");
        if task_type.default_output_format() == OutputFormat::Json {
            assert_eq!(sent_json["format"], "json");
        } else {
            assert!(sent_json.get("format").is_none());
        }
    }
}

#[tokio::test]
async fn cli_run_supports_input_file_output_file_and_quiet_flag() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload()))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "```python\ndef is_prime(n: int) -> bool:\n    return n > 1\n```",
            "stop",
        )))
        .mount(&server)
        .await;

    let temp_dir = unique_temp_dir("file-io");
    let input_path = temp_dir.join("task.json");
    let output_path = temp_dir.join("result.json");

    fs::write(
        &input_path,
        serde_json::to_string_pretty(&json!({
            "task_id": "file-001",
            "task_type": "generate_function",
            "language": "python",
            "instruction": "Implement is_prime(n: int) -> bool",
            "context": "",
            "constraints": ["Use Python standard library only"],
            "model": "qwen2.5-coder:3b"
        }))
        .expect("task JSON should serialize"),
    )
    .expect("should write input file");

    let out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "run",
            "--input",
            input_path.to_str().expect("utf8 path"),
            "--output",
            output_path.to_str().expect("utf8 path"),
            "--quiet",
        ])
        .output()
        .expect("CLI binary should run");

    assert_eq!(out.status.code(), Some(0));
    assert!(
        out.stdout.is_empty(),
        "expected empty stdout when --quiet is set"
    );

    let saved = fs::read_to_string(&output_path).expect("output file should be written");
    let result: TaskResult =
        serde_json::from_str(&saved).expect("output file should be valid JSON");
    assert_eq!(result.task_id, "file-001");
    assert_eq!(result.status, TaskStatus::Success);
    assert_eq!(
        result.output.as_deref(),
        Some("def is_prime(n: int) -> bool:\n    return n > 1")
    );
    assert!(
        result
            .validation
            .as_ref()
            .is_some_and(|v| v.passed && v.extracted_code_block)
    );

    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn cli_run_supports_stdin_and_enforces_json_schema_validation() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");

    let task_payload = serde_json::to_string(&json!({
        "task_id": "stdin-json-001",
        "task_type": "generate_json",
        "instruction": "Return function metadata as JSON.",
        "model": "qwen2.5-coder:3b",
        "output_expectation": {
            "format": "json",
            "required_fields": ["name", "complexity"],
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "complexity": { "type": "string", "enum": ["O(1)", "O(sqrt(n))", "O(n)"] }
                },
                "required": ["name", "complexity"],
                "additionalProperties": false
            }
        }
    }))
    .expect("task JSON should serialize");

    // 1. Valid JSON matching schema -> Exit 0.
    let ok_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "{\"name\": \"is_prime\", \"complexity\": \"O(sqrt(n))\"}",
            "stop",
        )))
        .mount(&ok_server)
        .await;

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &ok_server.uri(),
            "run",
            "--stdin",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("should spawn CLI");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(task_payload.as_bytes())
        .expect("write stdin");
    let ok_out = child.wait_with_output().expect("wait CLI");
    assert_eq!(ok_out.status.code(), Some(0));
    let ok_res: TaskResult = serde_json::from_slice(&ok_out.stdout).expect("valid TaskResult JSON");
    assert_eq!(ok_res.status, TaskStatus::Success);

    // 2. Schema violation -> Exit 6 with preserved raw_output and metrics.
    let bad_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "{\"name\": \"is_prime\", \"complexity\": \"O(2^n)\"}",
            "stop",
        )))
        .mount(&bad_server)
        .await;

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &bad_server.uri(),
            "run",
            "--stdin",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("should spawn CLI");
    child
        .stdin
        .take()
        .expect("stdin pipe")
        .write_all(task_payload.as_bytes())
        .expect("write stdin");
    let bad_out = child.wait_with_output().expect("wait CLI");
    assert_eq!(bad_out.status.code(), Some(6));
    let bad_res: TaskResult =
        serde_json::from_slice(&bad_out.stdout).expect("valid TaskResult JSON");
    assert_eq!(bad_res.status, TaskStatus::ValidationFailed);
    assert!(bad_res.raw_output.is_some());
    assert!(bad_res.metrics.is_some());
    assert!(bad_res.validation.as_ref().is_some_and(|v| {
        !v.passed
            && v.checks
                .iter()
                .any(|c| c.name == "json_schema" && !c.passed)
    }));
}

#[tokio::test]
async fn cli_dry_run_over_budget_and_secret_guard_never_contact_ollama() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    // 1. Valid --dry-run -> Exit 0 and DryRunReport JSON.
    let dry_out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "run",
            "--model",
            "qwen2.5-coder:3b",
            "--type",
            "generate_function",
            "--lang",
            "python",
            "--task",
            "Write is_prime(n: int) -> bool",
            "--dry-run",
        ])
        .output()
        .expect("dry-run should execute");
    assert_eq!(dry_out.status.code(), Some(0));
    let report: DryRunReport =
        serde_json::from_slice(&dry_out.stdout).expect("dry-run should emit DryRunReport JSON");
    assert_eq!(report.task_type, TaskType::GenerateFunction);
    assert!(report.estimated_prompt_tokens > 0);
    assert!(report.estimated_prompt_tokens < report.context_budget_tokens);

    // 2. Over context budget -> Exit 2 without calling Ollama.
    let big_context = "a".repeat(4000);
    let big_task = serde_json::to_string(&json!({
        "task_id": "over-budget-001",
        "model": "qwen2.5-coder:3b",
        "instruction": "Refactor this",
        "context": big_context,
        "options": { "num_ctx": 1024, "max_tokens": 512 }
    }))
    .expect("serialize big task");

    let mut child = Command::new(bin)
        .args(["--base-url", &server.uri(), "run", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn over-budget CLI");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(big_task.as_bytes())
        .expect("write stdin");
    let big_out = child.wait_with_output().expect("wait");
    assert_eq!(big_out.status.code(), Some(2));
    let big_res: TaskResult = serde_json::from_slice(&big_out.stdout).expect("valid error JSON");
    assert_eq!(
        big_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );

    // 3. Secret in prompt -> Exit 2, never leaks the raw secret value.
    let fake_secret = "ghp_1234567890abcdef1234567890abcdef";
    let sec_out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "run",
            "--model",
            "qwen2.5-coder:3b",
            "--task",
            &format!("Use token {fake_secret}"),
        ])
        .output()
        .expect("secret test should execute");
    assert_eq!(sec_out.status.code(), Some(2));
    let stdout_str = String::from_utf8_lossy(&sec_out.stdout);
    let stderr_str = String::from_utf8_lossy(&sec_out.stderr);
    assert!(!stdout_str.contains(fake_secret));
    assert!(!stderr_str.contains(fake_secret));
}

#[tokio::test]
async fn run_flags_truncated_length_and_unclosed_fence_as_validation_failed() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(chat_response("```python\ndef is_prime(n: int):", "length")),
        )
        .mount(&server)
        .await;

    let mut req = TaskRequest::new("trunc-001", "qwen2.5-coder:3b", "Write is_prime");
    req.task_type = TaskType::GenerateFunction;

    let result = executor_for(&server.uri(), false).execute(&req).await;
    assert_eq!(result.status, TaskStatus::ValidationFailed);
    assert_eq!(result.exit_code(), 6);
    assert_eq!(result.done_reason.as_deref(), Some("length"));
    let report = result.validation.expect("validation report should exist");
    assert!(!report.passed);
    assert!(
        report
            .checks
            .iter()
            .any(|c| c.name == "not_truncated" && !c.passed)
    );
    assert!(
        report
            .checks
            .iter()
            .any(|c| c.name == "code_fence" && !c.passed)
    );
}
