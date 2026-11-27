#!/usr/bin/env bash
# LoopForge v0.4.3 Script Node entrypoint for delegating a scoped task to ollama-worker.
#
# Expected workspace input:  .worker/task.json   (written by upstream Codex node)
# Declared workspace output: .worker/result.json (read by downstream Codex review node)

set -euo pipefail

WORKER_BIN="${OLLAMA_WORKER_BIN:-$(command -v ollama-worker || echo /usr/local/bin/ollama-worker)}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
TIMEOUT_SECS="${OLLAMA_WORKER_TIMEOUT_SECS:-120}"
TASK_FILE="${WORKER_TASK_FILE:-.worker/task.json}"
RESULT_FILE="${WORKER_RESULT_FILE:-.worker/result.json}"

if [[ ! -x "${WORKER_BIN}" ]]; then
  echo "[ollama-worker-node] ERROR: binary not found at '${WORKER_BIN}'" >&2
  exit 1
fi

if [[ ! -f "${TASK_FILE}" ]]; then
  echo "[ollama-worker-node] ERROR: missing task file '${TASK_FILE}'" >&2
  exit 2
fi

mkdir -p "$(dirname "${RESULT_FILE}")"

set +e
"${WORKER_BIN}" \
  --base-url "${BASE_URL}" \
  --timeout "${TIMEOUT_SECS}" \
  run \
  --input "${TASK_FILE}" \
  --output "${RESULT_FILE}" \
  --quiet
EXIT_CODE=$?
set -e

echo "[ollama-worker-node] finished with exit_code=${EXIT_CODE} result_file=${RESULT_FILE}" >&2
exit "${EXIT_CODE}"
