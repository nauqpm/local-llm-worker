# Testing Strategy & Performance Benchmark Plan

## 1. Testing Strategy

| Layer | Mechanism | Scope | CI Default? | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- |
| **Unit Tests** | `cargo test --lib` | Config precedence, loopback check, prompt templates, context budget guard, metrics calculations (zero-duration safe), output validators, error mapping | Yes | 100% pass on Windows and Linux; zero panics on edge cases |
| **Integration Tests (Mock HTTP)** | `wiremock` + CLI subprocess tests | Ollama 200 OK, 404 model not found, 500/503 errors, connection refused, timeout, malformed JSON, empty output, `done_reason: "length"` | Yes (offline) | 100% pass without live Ollama; verifies exact `stdout` JSON contract and process exit codes |
| **Live Tests (Opt-in)** | `cargo test -- --ignored` (`OLLAMA_LIVE_TEST=1`) | Real Home Server Ollama (`health`, `models`, single `qwen2.5-coder:3b` task) | No (manual) | Returns `status: "success"`, `prompt_tokens > 0`, `generated_tokens > 0` |
| **End-to-End Tests (M4)** | Codex + `ollama-worker` + LoopForge Native Runner | Full delegation loop from task JSON generation through review, automated testing, and LoopForge artifact persistence | Manual (M4) | Complete run succeeds with verified artifacts and exit codes |

## 2. Performance Benchmark Plan (M3)

No benchmark numbers are assumed or fabricated prior to empirical execution on the Home Server.

### Systems Under Comparison

1. **Direct `curl` → Native Ollama API** (lower-bound baseline).
2. **`ollama-worker` → Native Ollama API** (target system).
3. **OpenCode → Ollama (`qwen2.5-coder:3b`)** (guarded by a protective timeout to prevent host starvation).

### Controlled Variables

- Identical model (`qwen2.5-coder:3b`), identical prompts, identical `temperature`, `seed`, and `num_predict`.
- Explicit separation of worker overhead from model inference duration:
  - `worker_overhead_ms = wall_clock_ms - (ollama.total_duration_ns / 1_000_000)`

### Benchmark Scenarios

| Scenario | Target Prompt Size | Description |
| :--- | :--- | :--- |
| **S1 — Small** | ~50–150 tokens | Isolated helper function (`is_prime` in Python) |
| **S2 — Medium** | ~300–600 tokens | Unit tests or refactoring for a ~40-line snippet |
| **S3 — Large** | ~1,500–2,500 tokens | Stress test CPU prefill scaling on i5-1135G7 |
| **Cold vs. Warm** | S1 unloaded (`keep_alive: 0`) vs. preloaded (`/api/ps`) | Isolates `load_duration` from steady-state inference |
| **Context & Endpoint Matrix** | `num_ctx = 4096` vs. `8192`; `/api/chat` vs. `/api/generate` | Empirically validates whether `num_ctx` changes trigger model reloads and compares prompt token counts |

### Metrics Collected

- `wall_clock_ms`, `worker_overhead_ms`, `ttft_ms` (streaming mode)
- `load_duration_ms`, `prompt_eval_duration_ms`, `eval_duration_ms`
- `prompt_tokens`, `prompt_tokens_per_second`, `generated_tokens`, `generation_tokens_per_second`
- Peak RSS of `ollama-worker` (`/usr/bin/time -v`), Ollama container CPU/RAM (`docker stats --no-stream` and `/api/ps`)
