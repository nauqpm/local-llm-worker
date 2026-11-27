# Architecture & Requirements

## 1. Project Overview

`local-llm-worker` (CLI binary: `ollama-worker`, library crate: `ollama_worker`) is a lightweight Rust worker that bridges senior AI agents (such as Codex or LoopForge orchestrators) and a local Ollama server.

### Target Environment

- **Home Server:** Linux `amd64`, Intel Core i5-1135G7 (4 cores / 8 threads), 8 GB RAM, 8 GB Swap.
- **Local Inference:** Ollama in Docker accessible on host loopback `http://127.0.0.1:11434`, running `qwen2.5-coder:3b` (Q4_K_M, ~2.4 GB loaded memory, CPU inference).
- **Co-located Services:** LoopForge `v0.4.3` (Docker), LoopForge Runner `v0.13.2` (native `systemd` user service), Tailscale, Muse, Crafty Controller.

### Problem Statement

Running a 3B CPU-bound model through heavyweight autonomous coding-agent harnesses (such as OpenCode) injects large system prompts and tool schemas (~4,098 prompt tokens observed vs. ~41 prompt tokens for a direct request), causing sustained ~400% CPU utilization, >10-minute stalls, and failed tool-calling outputs.

`ollama-worker` replaces heavyweight agent loops with narrow, stateless, structured task delegation while leaving architecture, repository-wide reasoning, review, and execution to senior agents and dedicated test runners.

## 2. Functional & Non-Functional Requirements

### Functional Requirements

1. **Ollama Client:** Check server health, list installed models, inspect running models, validate target model existence, execute non-streaming inference (streaming in M3), handle HTTP/network/timeout errors, and extract performance metrics.
2. **Task Management:** Accept strongly typed task inputs via CLI flags, JSON file, or `stdin`, starting with a minimal MVP input and expanding to scoped coding task types in M2.
3. **Prompt Construction:** Build concise, task-specific prompts with explicit context budget checks and **no silent truncation**.
4. **Structured Results:** Emit predictable JSON to `stdout`, logs to `stderr`, explicit exit codes, and separate generation success from output validation success.
5. **Non-Executing Validation:** Validate outputs statically (non-empty, length, code block extraction, JSON schema) without ever executing generated code.

### Non-Functional Requirements

- Minimal runtime RAM and fast CLI startup.
- Zero modifications to existing Home Server services or global Ollama settings.
- Full offline testability using mock HTTP servers without requiring a live LLM.
- Independent of LoopForge at its core; reusable as a Rust library (`ollama_worker`).

## 3. System Architecture

```mermaid
flowchart TD
    Senior["Senior AI Agent (Codex / Human Dev)"]
    Decomp["Task Decomposition & Scoped Context"]
    CLI["ollama-worker CLI (stdin / file / flags)"]
    Core["ollama_worker Core Library"]
    Ollama["Ollama HTTP API (127.0.0.1:11434)"]
    Model["qwen2.5-coder:3b (CPU, Q4_K_M)"]
    Result["Structured JSON Result (stdout / file)"]
    Review["Senior Agent Review & Independent Test Runner"]

    Senior --> Decomp
    Decomp -->|"Task JSON"| CLI
    CLI --> Core
    Core -->|"Native HTTP JSON"| Ollama
    Ollama --> Model
    Model --> Ollama
    Ollama -->|"Response + Metrics (ns)"| Core
    Core -->|"stdout: Result JSON / stderr: Logs"| Result
    Result --> Review
```

## 4. Component Relationships

```mermaid
flowchart LR
    subgraph Bin["CLI Layer (src/main.rs, cli.rs)"]
        Args["Argument & Env Parser (clap)"]
        IO["Stdin / File / Stdout / Exit Codes"]
    end

    subgraph Lib["Core Library (src/lib.rs)"]
        Batch["BatchExecutor (Sequential, concurrency = 1)"]
        Exec["TaskExecutor (Orchestrator)"]
        Config["WorkerConfig & InferenceOptions"]
        Task["TaskSchema & Input Validator"]
        Prompt["PromptBuilder & Budget Estimator"]
        Client["OllamaClient Trait + HttpOllamaClient"]
        Valid["OutputValidator (Non-executing)"]
        Metrics["MetricsCalculator & BatchMetrics"]
        Err["WorkerError (Typed Categories)"]
    end

    Args --> Exec
    Args --> Batch
    IO --> Exec
    IO --> Batch
    Batch --> Exec
    Batch --> Metrics
    Exec --> Config
    Exec --> Task
    Exec --> Prompt
    Exec --> Client
    Exec --> Valid
    Exec --> Metrics
    Task -.-> Err
    Prompt -.-> Err
    Client -.-> Err
    Valid -.-> Err
```

## 5. Task Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Received: Read CLI args / file / stdin
    Received --> InputValidation: Deserialize TaskRequest
    InputValidation --> FailedInput: Schema / Empty / Secret / Budget Error
    InputValidation --> PromptBuilding: Valid TaskRequest
    PromptBuilding --> FailedBudget: Exceeds context budget (no silent truncation)
    PromptBuilding --> ModelChecking: Built Prompt + Token Estimate
    ModelChecking --> FailedModel: Server unreachable / Model not found
    ModelChecking --> Inferring: Model verified
    Inferring --> TimedOut: Request timeout exceeded
    Inferring --> Cancelled: SIGINT / SIGTERM received (M3)
    Inferring --> FailedTransport: Connection / HTTP 5xx / Malformed JSON
    Inferring --> OutputValidation: Raw generation received + Metrics parsed
    OutputValidation --> ValidationFailed: Empty / Invalid JSON / Fence error / Length violation
    OutputValidation --> Succeeded: Generation & Validation passed
    FailedInput --> [*]: Emit Result JSON + Exit 2
    FailedBudget --> [*]: Emit Result JSON + Exit 2
    FailedModel --> [*]: Emit Result JSON + Exit 3 or 4
    FailedTransport --> [*]: Emit Result JSON + Exit 3
    TimedOut --> [*]: Emit Result JSON + Exit 5
    Cancelled --> [*]: Emit Result JSON + Exit 130
    ValidationFailed --> [*]: Emit Result JSON + Exit 6
    Succeeded --> [*]: Emit Result JSON + Exit 0
```

## 6. Error Handling & Reliability Flow

```mermaid
flowchart TD
    Op["Operation Executed"] --> Check{"Result Ok?"}
    Check -->|Yes| Next["Next Stage / Emit Success JSON"]
    Check -->|No| Cat{"Categorize WorkerError"}

    Cat -->|"Config / Invalid Task / Over Budget"| EInput["Category: invalid_input (No Retry, Exit 2)"]
    Cat -->|"Model Not Found (HTTP 404)"| EModel["Category: model_not_found (No Retry, Exit 4)"]
    Cat -->|"Connection Refused / Reset / HTTP 503"| ETrans{"Transient & Safe to Retry?"}
    Cat -->|"Timeout Elapsed"| ETime["Category: timeout (No Retry for Inference, Exit 5)"]
    Cat -->|"Malformed Ollama Response"| EProto["Category: protocol_error (No Retry, Exit 3)"]
    Cat -->|"Output Validation Failed"| EValid["Category: validation_failed (Preserve raw_output, Exit 6)"]

    ETrans -->|"Health / Tags / Connect before send"| Retry["Exponential Backoff (max 2-3 retries, bounded)"]
    ETrans -->|"Inference already in-flight"| NoRetry["Do NOT auto-retry expensive CPU inference (Exit 3)"]
    Retry -->|"Still failing"| EUnavail["Category: ollama_unavailable (Exit 3)"]
```

### Reliability vs. Inference Cost Trade-off

Expensive CPU inference requests are **never automatically retried** once sent or timed out, because retrying a slow CPU request stacks load on Ollama and can starve co-located services on an 8 GB RAM host. Bounded exponential backoff retries (M3) apply exclusively to cheap, idempotent metadata/health endpoints or pre-request connection setup.
