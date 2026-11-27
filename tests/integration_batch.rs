use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ollama_worker::{
    BatchDryRunReport, BatchExecutor, BatchResult, BatchRuntimeOverrides, BatchStatus, BatchTask,
    BatchTaskItem, ErrorCategory, HttpOllamaClient, TaskStatus, TaskType, WorkerConfig,
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
        "ollama-worker-batch-test-{label}-{}-{nanos}",
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

fn chat_response(content: &str, prompt_tokens: u64, eval_tokens: u64) -> Value {
    json!({
        "model": "qwen2.5-coder:3b",
        "message": { "role": "assistant", "content": content },
        "done": true,
        "done_reason": "stop",
        "total_duration": 1_000_000_000u64,
        "load_duration": 100_000_000u64,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": 200_000_000u64,
        "eval_count": eval_tokens,
        "eval_duration": 700_000_000u64
    })
}

fn batch_executor_for(
    base_url: &str,
    timeout_secs: u64,
    verify_model: bool,
) -> BatchExecutor<HttpOllamaClient> {
    let config = WorkerConfig::new(base_url, timeout_secs, false, None)
        .expect("mock URL should be valid")
        .with_verify_model(verify_model);
    let client = HttpOllamaClient::new(config).expect("client should initialize");
    BatchExecutor::new(client, verify_model)
}

#[tokio::test]
async fn cli_batch_executes_three_tasks_sequentially_caches_tags_and_aggregates_metrics() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    // `/api/tags` must be called strictly ONCE for all 3 tasks.
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload()))
        .expect(1)
        .mount(&server)
        .await;

    // Mount 3 sequential `/api/chat` responses in order.
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "```python\ndef is_prime(n: int) -> bool:\n    return n > 1\n```",
            120,
            40,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "```python\ndef slugify(text: str) -> str:\n    return text.lower()\n```",
            131,
            44,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "```python\ndef chunk_list(items: list, size: int) -> list[list]:\n    return [items[i:i+size] for i in range(0, len(items), size)]\n```",
            130,
            50,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    let temp_dir = unique_temp_dir("happy-path");
    let output_path = temp_dir.join("batch_result.json");

    let out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "batch",
            "--input",
            "examples/batches/python_utils_batch.json",
            "--output",
            output_path.to_str().expect("utf8 path"),
            "--quiet",
        ])
        .output()
        .expect("CLI batch should run");

    assert_eq!(out.status.code(), Some(0));
    assert!(out.stdout.is_empty(), "expected empty stdout with --quiet");

    let saved = fs::read_to_string(&output_path).expect("output file should exist");
    let result: BatchResult =
        serde_json::from_str(&saved).expect("output file must be valid BatchResult JSON");

    assert_eq!(result.schema_version, "1");
    assert_eq!(result.batch_id, "example-python-utils-001");
    assert_eq!(result.status, BatchStatus::Completed);
    assert_eq!(result.summary.total_tasks, 3);
    assert_eq!(result.summary.executed_tasks, 3);
    assert_eq!(result.summary.succeeded_tasks, 3);
    assert_eq!(result.summary.skipped_tasks, 0);
    assert!(result.skipped_task_ids.is_empty());
    assert!(result.error.is_none());

    // Aggregated tokens match 120 + 131 + 130 = 381 input, 40 + 44 + 50 = 134 output.
    assert_eq!(result.metrics.total_prompt_tokens, 381);
    assert_eq!(result.metrics.total_generated_tokens, 134);
    assert_eq!(result.metrics.total_ollama_duration_ms, 3_000);
    assert_eq!(result.metrics.total_load_duration_ms, 300);

    // Verify sequential order of task IDs and extracted code blocks.
    assert_eq!(result.results[0].task_id, "fn-is-prime");
    assert!(
        result.results[0]
            .output
            .as_deref()
            .is_some_and(|o| o.contains("def is_prime"))
    );
    assert_eq!(result.results[1].task_id, "fn-slugify");
    assert!(
        result.results[1]
            .output
            .as_deref()
            .is_some_and(|o| o.contains("def slugify"))
    );
    assert_eq!(result.results[2].task_id, "fn-chunk-list");
    assert!(
        result.results[2]
            .output
            .as_deref()
            .is_some_and(|o| o.contains("def chunk_list"))
    );

    // Verify exact sequential request ordering against `/api/chat` payloads.
    let received = server
        .received_requests()
        .await
        .expect("wiremock should record received requests");
    let chat_bodies: Vec<String> = received
        .iter()
        .filter(|req| req.url.path() == "/api/chat")
        .map(|req| String::from_utf8_lossy(&req.body).into_owned())
        .collect();
    assert_eq!(chat_bodies.len(), 3);
    assert!(chat_bodies[0].contains("is_prime"));
    assert!(chat_bodies[1].contains("slugify"));
    assert!(chat_bodies[2].contains("chunk_list"));

    // Verify no leftover temporary files exist in temp_dir.
    let files: Vec<_> = fs::read_dir(&temp_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(files.len(), 1);

    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn cli_batch_failure_isolation_fail_fast_and_total_failure_exit_codes() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");

    let payload = serde_json::to_string(&json!({
        "batch_id": "isolation-test",
        "defaults": { "model": "qwen2.5-coder:3b", "language": "python" },
        "tasks": [
            { "task_id": "t1", "task_type": "generate_function", "instruction": "Write f1" },
            { "task_id": "t2", "task_type": "generate_function", "instruction": "Write f2" },
            { "task_id": "t3", "task_type": "generate_function", "instruction": "Write f3" }
        ]
    }))
    .expect("batch JSON should serialize");

    // 1. Failure isolation (default fail_fast = false): task 2 returns blank -> Exit 7, task 3 runs.
    let iso_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            30,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&iso_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response("   ", 30, 2)))
        .up_to_n_times(1)
        .expect(1)
        .mount(&iso_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f3(): pass",
            30,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&iso_server)
        .await;

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &iso_server.uri(),
            "batch",
            "--stdin",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn batch CLI");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write stdin");
    let iso_out = child.wait_with_output().expect("wait CLI");
    assert_eq!(iso_out.status.code(), Some(7));
    let iso_res: BatchResult =
        serde_json::from_slice(&iso_out.stdout).expect("stdout must be valid BatchResult JSON");
    assert_eq!(iso_res.status, BatchStatus::PartialSuccess);
    assert_eq!(iso_res.summary.succeeded_tasks, 2);
    assert_eq!(iso_res.summary.validation_failed_tasks, 1);
    assert_eq!(iso_res.results[1].status, TaskStatus::ValidationFailed);
    assert!(iso_res.results[1].raw_output.is_some());
    assert!(iso_res.skipped_task_ids.is_empty());

    // 2. Fail-fast (`--fail-fast`): stops after task 2 failure; task 3 never hits `/api/chat`.
    let ff_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            30,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&ff_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_json(json!({ "error": "busy" })))
        .up_to_n_times(1)
        .expect(1)
        .mount(&ff_server)
        .await;

    let mut ff_child = Command::new(bin)
        .args([
            "--base-url",
            &ff_server.uri(),
            "batch",
            "--stdin",
            "--skip-model-check",
            "--fail-fast",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn fail-fast CLI");
    ff_child
        .stdin
        .take()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write stdin");
    let ff_out = ff_child.wait_with_output().expect("wait CLI");
    assert_eq!(ff_out.status.code(), Some(7));
    let ff_res: BatchResult =
        serde_json::from_slice(&ff_out.stdout).expect("stdout must be valid BatchResult JSON");
    assert_eq!(ff_res.status, BatchStatus::PartialSuccess);
    assert!(ff_res.fail_fast);
    assert_eq!(ff_res.summary.executed_tasks, 2);
    assert_eq!(ff_res.summary.succeeded_tasks, 1);
    assert_eq!(ff_res.summary.error_tasks, 1);
    assert_eq!(ff_res.skipped_task_ids, vec!["t3"]);

    // 3. Total failure with missing model -> Exit 4 (`MODEL_NOT_FOUND`).
    let missing_server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(tags_payload()))
        .expect(1)
        .mount(&missing_server)
        .await;

    let mut missing_child = Command::new(bin)
        .args([
            "--base-url",
            &missing_server.uri(),
            "batch",
            "--stdin",
            "--model",
            "missing:7b",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn missing-model CLI");
    missing_child
        .stdin
        .take()
        .expect("stdin")
        .write_all(payload.as_bytes())
        .expect("write stdin");
    let missing_out = missing_child.wait_with_output().expect("wait CLI");
    assert_eq!(missing_out.status.code(), Some(4));
    let missing_res: BatchResult =
        serde_json::from_slice(&missing_out.stdout).expect("stdout must be valid BatchResult JSON");
    assert_eq!(missing_res.status, BatchStatus::Failed);
    assert_eq!(missing_res.summary.succeeded_tasks, 0);
    assert_eq!(missing_res.summary.error_tasks, 3);
}

#[tokio::test]
async fn cli_batch_dry_run_validates_offline_and_blocks_secrets_without_calling_ollama() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&server)
        .await;

    // 1. Valid example fixture -> Exit 0.
    let ok_out = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "batch",
            "--input",
            "examples/batches/python_utils_batch.json",
            "--dry-run",
        ])
        .output()
        .expect("dry-run CLI should execute");
    assert_eq!(ok_out.status.code(), Some(0));
    let report: BatchDryRunReport =
        serde_json::from_slice(&ok_out.stdout).expect("stdout must be valid BatchDryRunReport");
    assert_eq!(report.total_tasks, 3);
    assert_eq!(report.valid_tasks, 3);
    assert_eq!(report.invalid_tasks, 0);
    assert!(report.total_estimated_prompt_tokens > 0);

    // 2. Batch with secret in task 2 -> Exit 2 and never leaks raw secret.
    let fake_secret = "ghp_1234567890abcdef1234567890abcdef";
    let bad_batch = serde_json::to_string(&json!({
        "batch_id": "secret-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "tasks": [
            { "task_id": "ok-1", "instruction": "Write add(a, b)" },
            { "task_id": "bad-2", "instruction": format!("Use token {fake_secret}") }
        ]
    }))
    .expect("serialize secret batch");

    let mut child = Command::new(bin)
        .args(["--base-url", &server.uri(), "batch", "--stdin", "--dry-run"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn secret dry-run");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(bad_batch.as_bytes())
        .expect("write stdin");
    let bad_out = child.wait_with_output().expect("wait CLI");
    assert_eq!(bad_out.status.code(), Some(2));
    let stdout_str = String::from_utf8_lossy(&bad_out.stdout);
    let stderr_str = String::from_utf8_lossy(&bad_out.stderr);
    assert!(!stdout_str.contains(fake_secret));
    assert!(!stderr_str.contains(fake_secret));
    let bad_report: BatchDryRunReport =
        serde_json::from_slice(&bad_out.stdout).expect("valid BatchDryRunReport JSON");
    assert_eq!(bad_report.valid_tasks, 1);
    assert_eq!(bad_report.invalid_tasks, 1);
}

#[tokio::test]
async fn batch_per_task_timeout_overall_timeout_and_cancellation_preserve_completed_results() {
    let make_batch = || {
        let mut t1 = BatchTaskItem::new("t1", "Write f1");
        t1.task_type = TaskType::GenerateFunction;
        let mut t2 = BatchTaskItem::new("t2", "Write f2");
        t2.task_type = TaskType::GenerateFunction;
        let mut t3 = BatchTaskItem::new("t3", "Write f3");
        t3.task_type = TaskType::GenerateFunction;
        let mut b = BatchTask::new("resilience-batch", vec![t1, t2, t3]);
        b.defaults.model = Some("qwen2.5-coder:3b".into());
        b
    };

    // 1. Per-task timeout (`task_timeout_secs = 1`, `fail_fast = false`): task 2 times out, task 3 succeeds.
    let task_timeout_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&task_timeout_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(3))
                .set_body_json(chat_response("late", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&task_timeout_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f3(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&task_timeout_server)
        .await;

    let mut per_task_batch = make_batch();
    per_task_batch.policy.task_timeout_secs = Some(1);
    let per_task_res = batch_executor_for(&task_timeout_server.uri(), 10, false)
        .execute(&per_task_batch, &BatchRuntimeOverrides::default())
        .await;
    assert_eq!(per_task_res.status, BatchStatus::PartialSuccess);
    assert_eq!(per_task_res.exit_code(), 7);
    assert_eq!(per_task_res.summary.succeeded_tasks, 2);
    assert_eq!(per_task_res.summary.timed_out_tasks, 1);
    assert_eq!(per_task_res.results[1].status, TaskStatus::Timeout);

    // 2. Overall `batch_timeout_secs = 1`: task 1 preserved, task 2 timed out, task 3 skipped.
    let batch_timeout_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&batch_timeout_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(3))
                .set_body_json(chat_response("late", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&batch_timeout_server)
        .await;

    let mut overall_batch = make_batch();
    overall_batch.policy.batch_timeout_secs = Some(1);
    let overall_res = batch_executor_for(&batch_timeout_server.uri(), 10, false)
        .execute(&overall_batch, &BatchRuntimeOverrides::default())
        .await;
    assert_eq!(overall_res.status, BatchStatus::TimedOut);
    assert_eq!(overall_res.exit_code(), 5);
    assert_eq!(overall_res.summary.succeeded_tasks, 1);
    assert_eq!(overall_res.summary.timed_out_tasks, 1);
    assert_eq!(overall_res.skipped_task_ids, vec!["t3"]);

    // 3. Cancellation aborts in-flight task 2 promptly, preserves task 1, and invokes checkpoint hook.
    let cancel_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&cancel_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_json(chat_response("cancelled", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&cancel_server)
        .await;

    let started = Instant::now();
    let mut checkpoint_statuses = Vec::new();
    let cancel_res = batch_executor_for(&cancel_server.uri(), 30, false)
        .execute_with_cancel_and_hook(
            &make_batch(),
            &BatchRuntimeOverrides::default(),
            tokio::time::sleep(Duration::from_millis(120)),
            |snap| checkpoint_statuses.push((snap.status, snap.results.len())),
        )
        .await;
    let elapsed = started.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "cancellation should abort promptly, took {elapsed:?}"
    );
    assert_eq!(checkpoint_statuses, vec![(BatchStatus::InProgress, 1)]);
    assert_eq!(cancel_res.status, BatchStatus::Cancelled);
    assert_eq!(cancel_res.exit_code(), 130);
    assert_eq!(cancel_res.summary.succeeded_tasks, 1);
    assert_eq!(cancel_res.summary.cancelled_tasks, 1);
    assert_eq!(cancel_res.results[0].status, TaskStatus::Success);
    assert_eq!(cancel_res.results[1].status, TaskStatus::Cancelled);
    assert_eq!(cancel_res.skipped_task_ids, vec!["t3"]);
}

#[tokio::test]
async fn cli_batch_streaming_and_envelope_error_contracts() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    let ndjson = concat!(
        "{\"model\":\"qwen2.5-coder:3b\",\"message\":{\"role\":\"assistant\",\"content\":\"def ok():\"},\"done\":false}\n",
        "{\"model\":\"qwen2.5-coder:3b\",\"message\":{\"role\":\"assistant\",\"content\":\" return 1\"},\"done\":true,\"done_reason\":\"stop\",\"total_duration\":400000000,\"load_duration\":0,\"prompt_eval_count\":25,\"prompt_eval_duration\":100000000,\"eval_count\":8,\"eval_duration\":250000000}\n"
    );

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(ndjson, "application/x-ndjson"))
        .expect(2)
        .mount(&server)
        .await;

    let stream_batch = serde_json::to_string(&json!({
        "batch_id": "stream-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "tasks": [
            { "task_id": "s1", "instruction": "Write ok" },
            { "task_id": "s2", "instruction": "Write ok again" }
        ]
    }))
    .expect("serialize stream batch");

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "batch",
            "--stdin",
            "--stream",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn stream batch");
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(stream_batch.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait CLI");
    assert_eq!(out.status.code(), Some(0));
    let res: BatchResult = serde_json::from_slice(&out.stdout).expect("valid BatchResult JSON");
    assert_eq!(res.status, BatchStatus::Completed);
    assert!(res.metrics.avg_ttft_ms.is_some());
    assert_eq!(res.metrics.total_prompt_tokens, 50);
    assert_eq!(res.metrics.total_generated_tokens, 16);

    // Duplicate task_id -> Exit 2 with structured BatchResult JSON.
    let dup_batch = serde_json::to_string(&json!({
        "batch_id": "dup-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "tasks": [
            { "task_id": "dup", "instruction": "A" },
            { "task_id": "dup", "instruction": "B" }
        ]
    }))
    .expect("serialize dup batch");

    let mut dup_child = Command::new(bin)
        .args(["--base-url", &server.uri(), "batch", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dup batch");
    dup_child
        .stdin
        .take()
        .expect("stdin")
        .write_all(dup_batch.as_bytes())
        .expect("write stdin");
    let dup_out = dup_child.wait_with_output().expect("wait CLI");
    assert_eq!(dup_out.status.code(), Some(2));
    let dup_res: BatchResult =
        serde_json::from_slice(&dup_out.stdout).expect("valid error BatchResult JSON");
    assert_eq!(dup_res.status, BatchStatus::Failed);
    assert_eq!(
        dup_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );

    // Remote Ollama endpoint blocked by default -> Exit 2 with structured BatchResult JSON.
    let remote_out = Command::new(bin)
        .args([
            "--base-url",
            "http://192.168.1.50:11434",
            "batch",
            "--input",
            "examples/batches/python_utils_batch.json",
        ])
        .output()
        .expect("spawn remote-blocked batch");
    assert_eq!(remote_out.status.code(), Some(2));
    let remote_res: BatchResult =
        serde_json::from_slice(&remote_out.stdout).expect("valid error BatchResult JSON");
    assert_eq!(remote_res.status, BatchStatus::Failed);
    assert_eq!(
        remote_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );
}

#[tokio::test]
async fn cli_batch_continue_on_error_explicit_timeout_oversized_input_and_output_errors() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");

    // 1. `--continue-on-error` overrides `policy.fail_fast: true` in JSON.
    let coe_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response("   ", 20, 2)))
        .up_to_n_times(1)
        .expect(1)
        .mount(&coe_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f2(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&coe_server)
        .await;

    let fail_fast_json = serde_json::to_string(&json!({
        "batch_id": "coe-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "policy": { "fail_fast": true, "task_timeout_secs": 60 },
        "tasks": [
            { "task_id": "t1", "instruction": "Write f1" },
            { "task_id": "t2", "instruction": "Write f2" }
        ]
    }))
    .expect("serialize");

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &coe_server.uri(),
            "batch",
            "--stdin",
            "--continue-on-error",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(fail_fast_json.as_bytes())
        .unwrap();
    let coe_out = child.wait_with_output().unwrap();
    assert_eq!(coe_out.status.code(), Some(7));
    let coe_res: BatchResult = serde_json::from_slice(&coe_out.stdout).unwrap();
    assert!(!coe_res.fail_fast);
    assert_eq!(coe_res.summary.executed_tasks, 2);
    assert_eq!(coe_res.summary.succeeded_tasks, 1);

    // 2. Explicit global `--timeout 1` overrides `policy.task_timeout_secs: 60` in JSON.
    let timeout_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(3))
                .set_body_json(chat_response("late", 20, 10)),
        )
        .expect(1)
        .mount(&timeout_server)
        .await;

    let single_task_json = serde_json::to_string(&json!({
        "batch_id": "global-timeout-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "policy": { "task_timeout_secs": 60 },
        "tasks": [{ "task_id": "t1", "instruction": "Slow task" }]
    }))
    .unwrap();

    let started = Instant::now();
    let mut t_child = Command::new(bin)
        .args([
            "--base-url",
            &timeout_server.uri(),
            "--timeout",
            "1",
            "batch",
            "--stdin",
            "--skip-model-check",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    t_child
        .stdin
        .take()
        .unwrap()
        .write_all(single_task_json.as_bytes())
        .unwrap();
    let t_out = t_child.wait_with_output().unwrap();
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "explicit --timeout 1 must override JSON policy.task_timeout_secs=60"
    );
    assert_eq!(t_out.status.code(), Some(5));
    let t_res: BatchResult = serde_json::from_slice(&t_out.stdout).unwrap();
    assert_eq!(t_res.results[0].status, TaskStatus::Timeout);

    // 3. Oversized (> 4 MiB) batch payload via stdin -> Exit 2 + valid BatchResult JSON.
    let oversized = vec![b' '; (4 * 1024 * 1024) + 64];
    let mut big_child = Command::new(bin)
        .args(["batch", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _ = big_child.stdin.take().unwrap().write_all(&oversized);
    let big_out = big_child.wait_with_output().unwrap();
    assert_eq!(big_out.status.code(), Some(2));
    let big_res: BatchResult =
        serde_json::from_slice(&big_out.stdout).expect("oversized error must emit BatchResult");
    assert_eq!(
        big_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );

    // 4. Combined invalid `--base-url` + invalid `--output` still emits structured JSON on stdout (Exit 2).
    let temp_dir = unique_temp_dir("bad-out");
    let missing_out = temp_dir.join("no_such_dir").join("out.json");
    let bad_combo = Command::new(bin)
        .args([
            "--base-url",
            "http://192.168.1.50:11434",
            "batch",
            "--input",
            "examples/batches/python_utils_batch.json",
            "--output",
            missing_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(bad_combo.status.code(), Some(2));
    let combo_res: BatchResult =
        serde_json::from_slice(&bad_combo.stdout).expect("stdout must contain valid error JSON");
    assert_eq!(
        combo_res.error.as_ref().map(|e| e.category),
        Some(ErrorCategory::InvalidInput)
    );
    let _ = fs::remove_dir_all(&temp_dir);
}

#[tokio::test]
async fn cli_batch_persists_in_progress_checkpoint_mid_batch_and_respects_no_incremental_output() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");

    let payload = serde_json::to_string(&json!({
        "batch_id": "checkpoint-disk-test",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "tasks": [
            { "task_id": "t1", "instruction": "Write f1" },
            { "task_id": "t2", "instruction": "Write f2" }
        ]
    }))
    .unwrap();

    // 1. Default incremental checkpointing: file appears with status=in_progress while task 2 runs.
    let inc_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&inc_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(1200))
                .set_body_json(chat_response("def f2(): pass", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&inc_server)
        .await;

    let temp_dir = unique_temp_dir("checkpoint-disk");
    let out_file = temp_dir.join("result.json");

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &inc_server.uri(),
            "batch",
            "--stdin",
            "--skip-model-check",
            "--output",
            out_file.to_str().unwrap(),
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();

    let mut observed_in_progress = false;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if let Ok(raw) = fs::read_to_string(&out_file)
            && let Ok(parsed) = serde_json::from_str::<BatchResult>(&raw)
            && parsed.status == BatchStatus::InProgress
        {
            assert_eq!(parsed.results.len(), 1);
            assert_eq!(parsed.results[0].task_id, "t1");
            assert_eq!(parsed.skipped_task_ids, vec!["t2"]);
            observed_in_progress = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }

    let final_out = child.wait_with_output().unwrap();
    assert_eq!(final_out.status.code(), Some(0));
    assert!(
        observed_in_progress,
        "expected atomic in_progress checkpoint on disk while task 2 was still running"
    );
    let final_res: BatchResult =
        serde_json::from_str(&fs::read_to_string(&out_file).unwrap()).unwrap();
    assert_eq!(final_res.status, BatchStatus::Completed);
    assert_eq!(final_res.results.len(), 2);

    // 2. With `--no-incremental-output`: output file must not exist mid-batch while task 2 sleeps.
    let no_inc_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&no_inc_server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_millis(500))
                .set_body_json(chat_response("def f2(): pass", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&no_inc_server)
        .await;

    let no_inc_file = temp_dir.join("no_inc_result.json");
    let mut child2 = Command::new(bin)
        .args([
            "--base-url",
            &no_inc_server.uri(),
            "batch",
            "--stdin",
            "--skip-model-check",
            "--output",
            no_inc_file.to_str().unwrap(),
            "--no-incremental-output",
            "--quiet",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child2
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();

    tokio::time::sleep(Duration::from_millis(150)).await;
    assert!(
        !no_inc_file.exists(),
        "--no-incremental-output must not write output file before completion"
    );

    let final_out2 = child2.wait_with_output().unwrap();
    assert_eq!(final_out2.status.code(), Some(0));
    assert!(no_inc_file.exists());
    let _ = fs::remove_dir_all(&temp_dir);
}

#[cfg(unix)]
#[tokio::test]
async fn cli_batch_unix_sigterm_preserves_completed_result_and_exits_130() {
    let bin = env!("CARGO_BIN_EXE_ollama-worker");
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(chat_response(
            "def f1(): pass",
            20,
            10,
        )))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(10))
                .set_body_json(chat_response("late", 20, 10)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    let payload = serde_json::to_string(&json!({
        "batch_id": "unix-sigterm-batch",
        "defaults": { "model": "qwen2.5-coder:3b" },
        "tasks": [
            { "task_id": "t1", "instruction": "Write f1" },
            { "task_id": "t2", "instruction": "Write f2" },
            { "task_id": "t3", "instruction": "Write f3" }
        ]
    }))
    .unwrap();

    let temp_dir = unique_temp_dir("sigterm");
    let out_file = temp_dir.join("sigterm_result.json");

    let mut child = Command::new(bin)
        .args([
            "--base-url",
            &server.uri(),
            "batch",
            "--stdin",
            "--skip-model-check",
            "--output",
            out_file.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(payload.as_bytes())
        .unwrap();

    // Wait for task 1 checkpoint on disk before sending SIGTERM while task 2 is in-flight.
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && !out_file.exists() {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(out_file.exists(), "expected checkpoint after task 1");

    let pid = child.id().to_string();
    let kill_status = Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .expect("kill command should succeed");
    assert!(kill_status.success());

    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(130));
    let stdout_res: BatchResult =
        serde_json::from_slice(&out.stdout).expect("stdout must contain valid cancelled JSON");
    assert_eq!(stdout_res.status, BatchStatus::Cancelled);
    assert_eq!(stdout_res.summary.succeeded_tasks, 1);
    assert_eq!(stdout_res.summary.cancelled_tasks, 1);
    assert_eq!(stdout_res.skipped_task_ids, vec!["t3"]);

    let file_res: BatchResult =
        serde_json::from_str(&fs::read_to_string(&out_file).unwrap()).unwrap();
    assert_eq!(file_res.status, BatchStatus::Cancelled);
    assert_eq!(file_res.summary.succeeded_tasks, 1);

    let _ = fs::remove_dir_all(&temp_dir);
}
