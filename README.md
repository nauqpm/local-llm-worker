# local-llm-worker (`ollama-worker`)

A lightweight, resource-efficient, strongly typed Local LLM Worker written in Rust for AI agent orchestration.

`ollama-worker` enables senior AI agents (such as Codex or LoopForge workflows) to delegate small, clearly scoped coding tasks to local Ollama models (e.g. `qwen2.5-coder:3b` running on CPU) without heavyweight agent frameworks or unreliable tool-calling loops.

## Core Design Principles

- **Low overhead:** Single-shot Rust CLI with minimal RAM footprint and no background daemon required.
- **Native Ollama API:** Communicates directly with Ollama (`/api/version`, `/api/tags`, `/api/ps`, `/api/chat`) to preserve nanosecond inference metrics and per-request inference controls.
- **Deterministic contracts:** Structured JSON on `stdout`, structured logs on `stderr`, and explicit process exit codes (`0, 1, 2, 3, 4, 5, 6, 130`).
- **Context & secret safety:** Estimates prompt size against the configured context budget without silent truncation, blocks non-loopback endpoints unless explicitly opted in (`--allow-remote-ollama`), and checks for common accidental secret patterns before sending prompts.
- **Strict security boundary:** Never executes generated code or shell commands; performs static, non-executing output validation (non-empty, truncation check, markdown code fence extraction, max length, JSON & JSON Schema).
- **Reusable core:** Clean separation between the `ollama_worker` library crate (`src/lib.rs`) and the `ollama-worker` CLI binary (`src/main.rs`).

## Quick Start

```bash
# Verify formatting, lints, and all offline unit + mock integration tests
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets

# Inspect CLI help
cargo run -- --help

# Validate a task JSON and preview prompt + token estimate without calling Ollama
cargo run -- run --input examples/tasks/prime_function.json --dry-run --pretty

# Check local Ollama health and installed models
ollama-worker health --json
ollama-worker models --json

# Run a task from JSON file and save the structured result
ollama-worker run --input examples/tasks/prime_function.json --output result.json

# Pipe a task from stdin (convenient for Codex / automation)
cat examples/tasks/structured_json.json | ollama-worker run --stdin
```

### Opt-in Live Tests (Requires Local Ollama)

```bash
OLLAMA_LIVE_TEST=1 cargo test --test live_ollama -- --ignored --nocapture
```

## Examples & Documentation

- Task fixtures: [examples/tasks/](examples/tasks/)
- Standalone Codex delegation template: [examples/codex/DELEGATION_RULES.md](examples/codex/DELEGATION_RULES.md)
- [Architecture & Requirements](docs/ARCHITECTURE.md)
- [Task Schema, CLI & Ollama Contracts](docs/CONTRACTS.md)
- [LoopForge & Codex Integrations](docs/INTEGRATIONS.md)
- [Security Model & Boundaries](docs/SECURITY.md)
- [Testing & Performance Benchmark Plan](docs/QUALITY_AND_PERF.md)
- [Development & Deployment Guide](docs/DEPLOYMENT.md)
- [Roadmap & Architecture Decision Records](docs/ROADMAP_AND_ADRS.md)

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
