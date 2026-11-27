#!/usr/bin/env bash
# Read-only pre-flight check for LoopForge Runner v0.13.2 Native Sandbox.
# Verifies binary path, loopback Ollama reachability, and target model presence.

set -euo pipefail

WORKER_BIN="${OLLAMA_WORKER_BIN:-$(command -v ollama-worker || echo /usr/local/bin/ollama-worker)}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
TARGET_MODEL="${OLLAMA_WORKER_MODEL:-qwen2.5-coder:3b}"

echo "[preflight] Host: $(hostname) | User: $(id -un)"
echo "[preflight] PATH: ${PATH}"
echo "[preflight] Binary: ${WORKER_BIN}"

if [[ ! -x "${WORKER_BIN}" ]]; then
  echo "[preflight] ERROR: ollama-worker binary not found or not executable at '${WORKER_BIN}'" >&2
  exit 1
fi

"${WORKER_BIN}" --version
echo "[preflight] Checking Ollama health at ${BASE_URL}..."
"${WORKER_BIN}" --base-url "${BASE_URL}" --timeout 15 health --json

echo "[preflight] Checking installed models for '${TARGET_MODEL}'..."
MODELS_JSON="$("${WORKER_BIN}" --base-url "${BASE_URL}" --timeout 15 models --json)"
echo "${MODELS_JSON}"

if ! printf '%s' "${MODELS_JSON}" | grep -Fq "\"${TARGET_MODEL}\""; then
  echo "[preflight] ERROR: model '${TARGET_MODEL}' not found in /api/tags" >&2
  exit 4
fi

echo "[preflight] OK: Native Sandbox can execute ollama-worker and reach '${TARGET_MODEL}' at ${BASE_URL}."
