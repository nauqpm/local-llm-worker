# local-llm-worker (`ollama-worker`)

A lightweight, resource-efficient, strongly typed Local LLM Worker written in Rust for AI agent orchestration.

`ollama-worker` enables senior AI agents (such as Codex or LoopForge workflows) to delegate small, clearly scoped coding tasks to local Ollama models (e.g. `qwen2.5-coder:3b` running on CPU) without heavyweight agent frameworks or unreliable tool-calling loops.

## Core Design Principles

- **Low overhead:** Single-shot Rust CLI with minimal RAM footprint and no background daemon required.
- **Native Ollama API:** Communicates directly with Ollama (`/api/version`, `/api/tags`, `/api/ps`, `/api/chat`) to preserve nanosecond inference metrics and per-request inference controls.
- **Deterministic contracts:** Structured JSON on `stdout`, structured logs on `stderr`, and explicit process exit codes.
- **Strict security boundary:** Never executes generated code or shell commands; treats all model output as untrusted data.
- **Reusable core:** Clean separation between the `ollama_worker` library crate (`src/lib.rs`) and the `ollama-worker` CLI binary (`src/main.rs`).

## Quick Start (Development)

```bash
# Format check
cargo fmt --check

# Lint check
cargo clippy --all-targets -- -D warnings

# Run unit and offline mock integration tests
cargo test

# Show CLI help
cargo run -- --help
```

## Documentation

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
