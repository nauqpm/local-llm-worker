# Contracts: Task Schema, CLI & Ollama Integration

## 1. Task Input Schema (`schema_version = "1"`)

All optional extension fields use `#[serde(default)]` so minimal inputs remain valid.

```json
{
  "schema_version": "1",
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
  "model": "qwen2.5-coder:3b",
  "options": {
    "temperature": 0.1,
    "num_ctx": 4096,
    "max_tokens": 512,
    "seed": 42
  },
  "output_expectation": {
    "format": "code",
    "extract_code_block": true,
    "max_output_chars": 8000
  }
}
```

### Task Type Rollout

- **M1:** `freeform` (direct task prompt via `--task` or minimal input).
- **M2:** `generate_function`, `generate_unit_test`, `generate_docstring`, `explain_code`, `refactor_function`, `fix_bug`, `generate_json`.
- **M3+:** `generate_sql`, `generate_regex`, `convert_language`.

## 2. Structured Task Result Schema

The result distinguishes **generation success** from **validation success** and preserves `raw_output` for diagnostics. All numeric values below are illustrative schema placeholders, not benchmark measurements.

```json
{
  "schema_version": "1",
  "task_id": "task-001",
  "status": "success",
  "model": "qwen2.5-coder:3b",
  "output": "def is_prime(n: int) -> bool:\n    ...",
  "raw_output": "```python\ndef is_prime(n: int) -> bool:\n    ...\n```",
  "done_reason": "stop",
  "validation": {
    "passed": true,
    "extracted_code_block": true,
    "checks": [
      { "name": "non_empty", "passed": true, "detail": null }
    ],
    "warnings": []
  },
  "metrics": {
    "worker_overhead_ms": 0,
    "ttft_ms": null,
    "total_duration_ms": 0,
    "load_duration_ms": 0,
    "prompt_eval_duration_ms": 0,
    "eval_duration_ms": 0,
    "prompt_tokens": 0,
    "generated_tokens": 0,
    "prompt_tokens_per_second": 0.0,
    "generation_tokens_per_second": 0.0
  },
  "error": null
}
```

### Allowed `status` Values

- `success`: Inference succeeded and all output validations passed.
- `validation_failed`: Inference succeeded, but static output validation failed (`raw_output` and `metrics` preserved).
- `timeout`: Request exceeded configured timeout.
- `cancelled`: Interrupted by `SIGINT`/`SIGTERM` (M3).
- `error`: Input, configuration, model discovery, transport, or protocol error.

## 3. CLI Interface (`ollama-worker`)

```bash
ollama-worker health [--json] [--base-url URL] [--allow-remote-ollama] [--pretty]
ollama-worker models [--json] [--base-url URL] [--pretty]
ollama-worker run --model qwen2.5-coder:3b --type generate_function --lang python --task "Write is_prime"
ollama-worker run --input task.json [--output result.json] [--quiet] [--stream] [--pretty]
cat task.json | ollama-worker run --stdin [--stream]
ollama-worker run --input task.json --dry-run [--pretty]
ollama-worker batch --input batch.json [--output batch_result.json] [--quiet] [--fail-fast|--continue-on-error] [--task-timeout SECS] [--batch-timeout SECS] [--stream] [--dry-run] [--pretty]
cat batch.json | ollama-worker batch --stdin [--output batch_result.json] [--fail-fast]
ollama-worker benchmark --model qwen2.5-coder:3b [--suite FILE] [--endpoint chat|generate|both] [--stream] [--cold-start] [--iterations N]
```

### Stream Separation & Exit Codes

- `stdout`: Strictly machine-readable JSON output for `run` and `batch` (and when `--json` is passed to `health`/`models`).
- `stderr`: Structured logs (`tracing`), diagnostics, and warnings.

| Exit Code | Meaning |
| :--- | :--- |
| `0` | Success (`status == "success"` for `run`, `status == "completed"` for `batch`) |
| `1` | Unexpected internal error |
| `2` | Invalid input, configuration error, or context budget exceeded |
| `3` | Ollama unavailable, network/HTTP 5xx error, or malformed protocol response |
| `4` | Requested model not found |
| `5` | Request or overall batch timeout (`status == "timeout"` / `"timed_out"`) |
| `6` | Output validation failed (`status == "validation_failed"`) |
| `7` | Partial batch completion (`batch` only: `status == "partial_success"`, >= 1 succeeded and >= 1 failed/timed out/skipped) |
| `130` | Cancelled via `SIGINT`/`SIGTERM` (`128 + 2`; preserves completed batch results) |

## 3.1 Sequential Batch Contracts (`v0.2.0`, `schema_version = "1"`)

`BatchTask` executes independent tasks strictly sequentially (`concurrency = 1`) through the existing single-task `TaskExecutor` pipeline. Any v0.1.0 `TaskRequest` JSON object is accepted verbatim in `tasks`, while concise items may omit `schema_version`, `model`, `language`, `system_prompt`, `options`, or `output_expectation` to inherit from `defaults`.

### `BatchTask` Input Schema

```json
{
  "schema_version": "1",
  "batch_id": "example-python-utils-001",
  "defaults": {
    "model": "qwen2.5-coder:3b",
    "language": "python",
    "options": { "temperature": 0.1, "max_tokens": 256, "seed": 42 }
  },
  "policy": {
    "fail_fast": false,
    "task_timeout_secs": 120,
    "batch_timeout_secs": 600
  },
  "tasks": [
    {
      "task_id": "fn-is-prime",
      "task_type": "generate_function",
      "instruction": "Implement `is_prime(n: int) -> bool`."
    }
  ]
}
```

### `BatchResult` Output Schema

- `status`: `"completed"`, `"partial_success"`, `"failed"`, `"timed_out"`, `"cancelled"` (or `"in_progress"` in intermediate atomic checkpoints written to `--output`).
- `summary`: `total_tasks`, `executed_tasks`, `succeeded_tasks`, `validation_failed_tasks`, `error_tasks`, `timed_out_tasks`, `cancelled_tasks`, `skipped_tasks`.
- `metrics`: `batch_wall_clock_ms`, `batch_overhead_ms`, `unattributed_duration_ms` (wall-clock ms spent in tasks without Ollama telemetry such as cancelled/timed-out/transport-errored tasks), `total_worker_overhead_ms`, `avg_worker_overhead_ms`, `total_ollama_duration_ms`, `total_load_duration_ms`, `total_prompt_eval_duration_ms`, `total_eval_duration_ms`, `total_prompt_tokens`, `total_generated_tokens`, `avg_prompt_tokens_per_second`, `avg_generation_tokens_per_second`, and optional `avg_ttft_ms`.
- `results`: Array of standard v0.1.0 `TaskResult` objects in execution order.
- `skipped_task_ids`: Ordered IDs of tasks not executed due to `fail_fast`, `batch_timeout_secs`, or cancellation.
- `error`: Optional batch-level `ErrorDetail` (envelope error, batch timeout, or cancellation).

### Batch Precedence & Persistence

- **Configuration precedence:** `CLI batch flags > Per-task JSON item > BatchTask.defaults/policy > Global CLI / Env > Built-in Defaults`.
  - Explicitly passing global `--timeout <SECS>` on the CLI also overrides JSON `policy.task_timeout_secs` and `item.timeout_secs` unless `--task-timeout` is supplied; otherwise the built-in/env timeout acts as fallback.
  - Partial `options` objects in `BatchTask` merge on top of `InferenceOptions::default()` (`temperature: 0.1`, `max_tokens: 512`) so CPU generation remains bounded; `stream` is additive (`||`).
- **Atomic incremental output:** When `--output <FILE>` is passed to `ollama-worker batch`, the destination must be a direct regular file (symlinks, directories, and device/special files are rejected). Completed results are atomically checkpointed after each task using an exclusive same-directory temp file (`.<name>.tmp.<pid>.<seq>`), `sync_all()`, and `rename` (disable with `--no-incremental-output`). If final file persistence fails, structured JSON is still emitted to `stdout` unless `--quiet` is set.
- **Bounds:** Maximum batch input size `4 MiB`, maximum `100` tasks per batch, maximum `batch_timeout_secs = 14400` (4 hours).

### Configuration Precedence

`CLI Flags > Task JSON options > Environment Variables > Built-in Defaults`

Supported environment variables:

- `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_WORKER_MODEL`
- `OLLAMA_WORKER_TIMEOUT_SECS` (default: `120`)
- `RUST_LOG` (default: `warn`)

## 4. Native Ollama API Integration

`ollama-worker` targets the **Native Ollama API** rather than `/v1` to retain nanosecond timing metrics and per-request inference options.

| Endpoint | Method | Purpose | Milestone |
| :--- | :--- | :--- | :--- |
| `/api/version` | `GET` | Lightweight server liveness and version check | M1 |
| `/api/tags` | `GET` | List installed models and verify model existence | M1 |
| `/api/ps` | `GET` | Inspect currently loaded models and memory usage | M1 / M3 |
| `/api/chat` | `POST` | Primary stateless 2-message (`system` + `user`) inference without `tools` (non-streaming & NDJSON streaming) | M1–M3 |
| `/api/generate` | `POST` | Benchmark comparison against `/api/chat`, cold-start unload (`keep_alive: "0"`), and future FIM (`suffix`) | M3 |
| `/api/show` | `POST` | Optional future enhancement for verbose model metadata inspection | Future |

### Per-Request Inference Defaults (CPU 3B Model)

- `temperature`: `0.1` for code/JSON tasks, `0.2` for explanations.
- `num_predict` (`max_tokens`): `512` default (configurable, bounded).
- `num_ctx`: Omitted by default in M1 to avoid forcing Ollama to unload/reload an already-loaded `8192`-context model; benchmarked explicitly (`4096` vs `8192`) in M3.
- `keep_alive`: Respects Ollama's default (`5m`) unless explicitly overridden per request.
- `stream`: `false` in M1–M2; optional `true` in M3 for TTFT and cancellation.

### Ollama Request Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Codex / LoopForge / User
    participant Worker as ollama-worker
    participant Ollama as Ollama (127.0.0.1:11434)

    Caller->>Worker: run --input task.json
    Worker->>Worker: Validate TaskRequest & Build Prompt
    Worker->>Worker: Estimate Tokens vs Context Budget

    alt --verify-model enabled (default)
        Worker->>Ollama: GET /api/tags
        Ollama-->>Worker: 200 OK { models: [...] }
        Worker->>Worker: Verify requested model exists
    end

    Worker->>Ollama: POST /api/chat (stream: false, options: {temperature, num_predict, seed})
    Note over Ollama: Load model if cold, prefill prompt, generate tokens on CPU
    Ollama-->>Worker: 200 OK { message.content, done_reason, *_duration, *_count }

    Worker->>Worker: Compute Metrics (tokens/s, worker_overhead_ms)
    Worker->>Worker: Run Non-Executing OutputValidator
    Worker-->>Caller: stdout: Structured Result JSON (Exit 0 or 6)
```
