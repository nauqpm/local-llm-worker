# Roadmap & Architecture Decision Records

## 1. Architecture Decision Records (ADRs)

### ADR-001: Minimal Async Runtime (`tokio` `current_thread` + `reqwest 0.13`)

- **Status:** Accepted
- **Decision:** Use `tokio` (minimal features: `rt`, `macros`, `time`, adding `signal` in M3) with `reqwest 0.13` (`default-features = false`, `features = ["json"]` in M1) and `wiremock` for mock HTTP integration tests.
- **Rationale:** Aligns M1 with M3 streaming/cancellation and potential M5 library reuse (`axum`/MCP) while avoiding `tokio`'s `full` feature bloat and `reqwest 0.13`'s default `aws-lc` C/CMake build requirement on Windows when communicating over local HTTP.
- **Consequences:** Slightly longer initial compile than blocking `ureq`; explicit TLS feature selection will be documented if remote HTTPS support is enabled later.

### ADR-002: Native Ollama API with Stateless `POST /api/chat`

- **Status:** Accepted
- **Decision:** Use Native Ollama endpoints (`/api/version`, `/api/tags`, `/api/ps`, `/api/chat`) instead of `/v1/chat/completions`. Send stateless 2-message (`[system, user]`) requests without `tools`.
- **Rationale:** Exposes nanosecond timing metrics (`load_duration`, `prompt_eval_duration`, `eval_duration`), per-request `options`, and `keep_alive`, while honoring `qwen2.5-coder:3b`'s native ChatML instruction template without multi-turn bloat. `/api/generate` will be benchmarked alongside `/api/chat` in M3.

### ADR-003: Single Cargo Package (`lib` + `bin`) for M1–M4

- **Status:** Accepted
- **Decision:** Keep `src/lib.rs` (`ollama_worker`) and `src/main.rs` (`ollama-worker`) in a single Cargo package until M5.
- **Rationale:** Preserves clean library reusability without premature Cargo workspace complexity.

### ADR-004: Strict Separation of Generation, Validation, and Execution

- **Status:** Accepted
- **Decision:** `ollama-worker` performs only generation and non-executing static validation; code execution and testing are delegated to independent, explicitly authorized runners.

### ADR-005: LoopForge `v0.4.3` Integration via Native Sandbox on Personal Runner `v0.13.2`

- **Status:** Accepted
- **Decision:** Deploy the static Linux release binary to the existing native LoopForge Runner host and run Script Nodes in a Native Sandbox profile, exchanging task/result JSON via workspace files and `ArtifactOutputs`.
- **Rationale:** Directly reaches `http://127.0.0.1:11434` without modifying Ollama, Docker networking, Tailscale, or LoopForge source code.

### ADR-006: Dual License (`MIT OR Apache-2.0`) & Consolidated Documentation

- **Status:** Accepted
- **Decision:** Use standard Rust dual licensing (`MIT OR Apache-2.0`) and 7 cohesive documents in `docs/`.

---

## 2. Milestone Roadmap

### M0 — Research and Architecture *(Completed)*
- Requirements analysis, LoopForge `v0.4.3` source inspection, Ollama API & crate verification, architecture design, ADRs, and user approval.

### M1 — Minimum Viable Worker *(In Progress)*
- `M1.1`: Git initialization, dual licenses, consolidated `docs/`, `Cargo.toml`, `lib`/`bin` skeleton, CI workflow.
- `M1.2`: Typed `error.rs` (with exit codes) and `config.rs` (env/CLI precedence + loopback guard).
- `M1.3`: `ollama.rs` (`OllamaClient` trait + `reqwest` implementation for `/api/version`, `/api/tags`, `/api/ps`, `/api/chat`).
- `M1.4`: `metrics.rs` and MVP `task.rs` request/result models.
- `M1.5`: `cli.rs` & `main.rs` (`health`, `models`, `run --model --task`, `stdout` JSON / `stderr` logs).
- `M1.6`: `wiremock` integration tests, CLI tests, and opt-in live test (`#[ignore]`).

### M2 — Task-Oriented Worker
- `M2.1`: Versioned `TaskRequest` v1, 7 core `TaskType`s, `TaskResult` with `validation` block.
- `M2.2`: `prompt.rs` templates, delimiter isolation, token estimator, and context budget enforcement.
- `M2.3`: `validation.rs` (non-empty, length, code-block extractor, JSON / `jsonschema`, `done_reason` check).
- `M2.4`: `--input`, `--output`, `--stdin`, `--dry-run`, and basic secret-pattern protection.
- `M2.5`: Comprehensive unit & mock integration tests for all task types and validation failures.
- `M2.6`: Standalone Codex delegation guide and example task fixtures.

### M3 — Performance and Reliability
- `M3.1`: Modularize `src/client/`, add NDJSON streaming and TTFT metrics.
- `M3.2`: Graceful `SIGINT`/`SIGTERM` cancellation (Exit `130`).
- `M3.3`: Bounded exponential backoff retry for idempotent/connect failures only.
- `M3.4`: `benchmark` subcommand and reproducible benchmark scripts (`curl` vs `ollama-worker` vs OpenCode).
- `M3.5`: GitHub Actions `release.yml` for static `x86_64-unknown-linux-musl` binaries + `SHA256SUMS`.
- `M3.6`: Run empirical benchmarks on Home Server and document measured findings.

### M4 — LoopForge Integration
- `M4.1`: Verify Runner `v0.13.2` Native Sandbox environment (`PATH`, env vars, `127.0.0.1:11434`).
- `M4.2`: Install verified release binary on Runner host.
- `M4.3`: Configure sample LOOP workflow with Script Node and `ArtifactOutputs`.
- `M4.4`: End-to-end validation (Codex → `ollama-worker` → Codex review → automated tests → approval).

### M5 — Advanced Integrations *(Optional)*
- Cargo workspace split, `stdio` MCP server, lightweight `axum` HTTP service, bounded concurrency queue, and multi-model routing.
