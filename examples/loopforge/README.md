# LoopForge `v0.4.3` + Runner `v0.13.2` Integration Guide

This directory contains ready-to-use scripts and node contracts for integrating `ollama-worker` into a LoopForge `v0.4.3` workflow using a **Native Sandbox** on your existing **LoopForge Runner `v0.13.2`** (`systemd` user service) — with **zero modifications** to Ollama, Docker, Tailscale, or LoopForge source code.

## 1. Install the Verified Static Release Binary on the Home Server

Download the tagged `x86_64-unknown-linux-musl` release archive and checksum file from GitHub Releases, verify SHA-256 integrity, and install to `/usr/local/bin/ollama-worker` (or `~/.local/bin/ollama-worker`):

```bash
VERSION="v0.1.0"
ARCHIVE="ollama-worker-${VERSION}-x86_64-unknown-linux-musl.tar.gz"
BASE="https://github.com/nauqpm/local-llm-worker/releases/download/${VERSION}"

mkdir -p /tmp/ollama-worker-install && cd /tmp/ollama-worker-install
curl -fsSLO "${BASE}/${ARCHIVE}"
curl -fsSLO "${BASE}/SHA256SUMS.txt"
sha256sum -c SHA256SUMS.txt

tar -xzf "${ARCHIVE}" ollama-worker
chmod 0755 ollama-worker
sudo mv ollama-worker /usr/local/bin/ollama-worker
ollama-worker --version
```

> **Note on `PATH` inside LoopForge Native Sandboxes:**
> LoopForge `v0.4.3` runs native sandbox commands under `controlledBaseEnvironment(os.Environ())`. Placing the binary in `/usr/local/bin/ollama-worker` (or setting `OLLAMA_WORKER_BIN=/home/<user>/.local/bin/ollama-worker` in the Native Sandbox profile's environment variables) ensures Script Nodes always find it reliably.

## 2. Run Pre-Flight Verification

Test both directly on the Runner host and once inside a test Script Node (Native Sandbox):

```bash
./examples/loopforge/preflight_runner.sh
```

Expected result: exit code `0`, confirming `ollama-worker` is executable, `http://127.0.0.1:11434` is reachable, and `qwen2.5-coder:3b` is installed.

## 3. Configure the LoopForge LOOP Workflow

Bind the workflow (or Script Node) to a **Native Sandbox** profile executing on your Personal Runner `v0.13.2`:

| Node | Type | Responsibility | Declared `ArtifactOutputs` |
| :--- | :--- | :--- | :--- |
| **Node 1** | Agent (Codex) | Analyze requirement, decompose narrow subtask, write `.worker/task.json` | `name: worker-task`, `path: .worker/task.json`, `mediaType: application/json`, `required: true` |
| **Node 2** | Script (`shell`) | Run `examples/loopforge/run_worker_node.sh` (invokes `ollama-worker run`) | `name: worker-result`, `path: .worker/result.json`, `mediaType: application/json`, `required: true` |
| **Node 3** | Agent (Codex) | Read `.worker/result.json`, inspect `status` + `validation.checks`, review & integrate code | Repository patch / diff |
| **Node 4** | Script (`shell`) | Run independent project tests (`pytest`, `cargo test`, linters) | Test log / report |
| **Node 5** | Approval | Human review and approval gate | — |

### Handling Failures (`OnFailure` / `RejectTo`)

- Exit `0`: Generation and static validation passed → proceed to Codex Review (Node 3).
- Exit `2` (invalid input / over budget / secret blocked) or Exit `6` (static validation failed): `.worker/result.json` still contains the structured `error` and `validation.checks` (`raw_output` preserved on Exit `6`). Route via `OnFailure: reject` back to Codex to narrow the task or implement directly.
- Exit `3` / `4` / `5` (Ollama down / missing model / timeout): Do **not** blindly retry expensive CPU inference (`MaxAttempts: 1`); fall back to Codex directly.
