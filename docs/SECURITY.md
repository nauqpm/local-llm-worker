# Security Model & Boundaries

## 1. Security Boundary

```text
[ Untrusted Zone: Local LLM Output ]
        |
        v
[ ollama-worker: Parse & Static Validation ONLY ]
        |
        v
[ Senior Agent Review ]
        |
        v
[ Authorized Execution / Test Sandbox ]
```

`ollama-worker` is strictly a **generation and static-validation component**. It never bridges into code execution.

## 2. Security Invariants

1. **No Automatic Code or Command Execution:** The worker never invokes a shell, interpreter, or compiler on model-generated output.
2. **Restricted Filesystem Access:**
   - Only reads the explicit `--input` path (or `stdin`), enforcing a maximum input size (default 1 MiB).
   - Only writes to the explicit `--output` path when requested; never crawls repositories or mutates project files on its own.
3. **Loopback-First Network Policy:**
   - Defaults to `http://127.0.0.1:11434`.
   - Rejects non-loopback `OLLAMA_BASE_URL` targets unless an explicit opt-in flag (`--allow-remote-ollama`) is provided, preventing accidental data exfiltration to external endpoints.
   - Never exposes unauthenticated Ollama endpoints to public networks.
4. **Secret Hygiene:**
   - Logs go exclusively to `stderr` and omit full prompt bodies and raw model outputs at default log levels (`warn`/`info`).
   - M2 introduces pre-flight detection/warnings for common secret patterns (e.g., API keys, private key headers) in task inputs.
5. **Prompt Injection Mitigation:**
   - Prompt templates isolate user instructions and code context with explicit delimiters.
   - Model responses are always treated as untrusted data and validated structurally before returning `status: "success"`.
6. **Supply-Chain Hygiene:**
   - Minimize dependencies and disable unused crate features (e.g., `reqwest` `default-features = false`, selective `tokio` features).
   - Commit `Cargo.lock` for reproducible builds and run `cargo audit` in CI.
