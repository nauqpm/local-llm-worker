# Local LLM Delegation Rules (Codex / Senior Agent Template)

Copy the section below into a repository's `AGENTS.md` (or agent instructions) to let a senior AI agent delegate small, isolated coding tasks to `ollama-worker` independently of LoopForge.

---

## Delegating Scoped Subtasks to `ollama-worker`

A local `ollama-worker` CLI is available for offloading small, self-contained drafting tasks to a local Ollama model (default: `qwen2.5-coder:3b`).

### When to Delegate

Delegate **only** when all of the following hold:
- The task is self-contained (<= 1–2 short code snippets, target prompt ~100–500 tokens).
- The task falls into a narrow drafting category: `generate_function`, `generate_unit_test`, `generate_docstring`, `explain_code`, `refactor_function`, `fix_bug`, or `generate_json`.
- The code is **not** security-critical (no authentication, authorization, cryptography, credential handling, or untrusted input parsing).
- No secrets, tokens, or private repository credentials appear in `instruction` or `context`.

### How to Invoke

Pipe a versioned `TaskRequest` JSON via `stdin` and read the structured `TaskResult` JSON from `stdout`:

```bash
ollama-worker run --stdin <<'EOF'
{
  "schema_version": "1",
  "task_id": "subtask-001",
  "task_type": "generate_function",
  "language": "python",
  "instruction": "Implement is_prime(n: int) -> bool.",
  "context": "",
  "constraints": ["Standard library only", "Return only the implementation"],
  "model": "qwen2.5-coder:3b"
}
EOF
```

Use `--dry-run` first if you want to verify prompt size against the context budget without running CPU inference.

### Mandatory Review & Error Handling Rules

1. Always check the process exit code and the `status` and `validation.passed` fields:
   - Exit `0` (`status == "success"`): Candidate draft passed static validation.
   - Exit `6` (`status == "validation_failed"`): Generation completed, but static checks failed (inspect `validation.checks` and `raw_output`, then handle directly or narrow the task).
   - Exit `2` (`invalid_input` / `context_budget_exceeded`): Fix input or reduce context size; do not retry blindly.
   - Exit `3` / `4` / `5` (Ollama unavailable / missing model / timeout): Fall back to implementing the task directly instead of looping retries.
2. **Never trust generated code blindly:** Always review the returned `output` for correctness, edge cases, and style before applying it to the repository, and run the project's automated test suite afterward.
