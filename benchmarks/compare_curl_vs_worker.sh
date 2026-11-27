#!/usr/bin/env bash
# Reproducible Home Server benchmark script comparing:
#   1. Direct curl -> Native Ollama /api/chat
#   2. ollama-worker -> Native Ollama /api/chat (streaming + non-streaming)
#   3. Optional OpenCode baseline (disabled unless RUN_OPENCODE=1, guarded by timeout)
#
# Does NOT modify Ollama, Docker, or LoopForge configuration.

set -euo pipefail

MODEL="${OLLAMA_WORKER_MODEL:-qwen2.5-coder:3b}"
BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
WORKER_BIN="${OLLAMA_WORKER_BIN:-ollama-worker}"
OUT_DIR="${BENCH_OUT_DIR:-benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)}"
PROMPT="Implement a Python function \`is_prime(n: int) -> bool\` using only the standard library. Return only the code."

mkdir -p "$OUT_DIR"

echo "=== Environment Snapshot ==="
echo "Model: $MODEL | Base URL: $BASE_URL | Output: $OUT_DIR"
curl -sS "$BASE_URL/api/version" > "$OUT_DIR/ollama_version.json"
curl -sS "$BASE_URL/api/ps" > "$OUT_DIR/ollama_ps_before.json"
if command -v docker >/dev/null 2>&1; then
  docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' > "$OUT_DIR/docker_stats_before.txt" || true
fi

echo "=== 1. Direct curl baseline (/api/chat, non-streaming, seed=42, temp=0.0) ==="
PAYLOAD=$(cat <<EOF
{
  "model": "$MODEL",
  "stream": false,
  "options": { "temperature": 0.0, "num_predict": 256, "seed": 42 },
  "messages": [
    { "role": "system", "content": "You are a concise, precise coding assistant. Follow instructions strictly and return only the requested output without unnecessary commentary." },
    { "role": "user", "content": "$PROMPT" }
  ]
}
EOF
)
START_NS=$(date +%s%N)
/usr/bin/time -v -o "$OUT_DIR/curl_time.txt" \
  curl -sS -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" > "$OUT_DIR/curl_response.json"
END_NS=$(date +%s%N)
echo "curl_wall_clock_ms=$(( (END_NS - START_NS) / 1000000 ))" | tee "$OUT_DIR/curl_wall_ms.txt"

echo "=== 2. ollama-worker single task (measuring Peak RSS via /usr/bin/time -v) ==="
/usr/bin/time -v -o "$OUT_DIR/worker_single_time.txt" \
  "$WORKER_BIN" --base-url "$BASE_URL" run \
  --model "$MODEL" \
  --task "$PROMPT" \
  --temperature 0.0 --max-tokens 256 --seed 42 \
  --skip-model-check \
  --output "$OUT_DIR/worker_single_result.json" --quiet

echo "=== 3. ollama-worker full suite (S1/S2/S3, streaming TTFT + chat vs generate) ==="
"$WORKER_BIN" --base-url "$BASE_URL" benchmark \
  --model "$MODEL" \
  --endpoint both \
  --stream \
  --iterations 2 \
  --output "$OUT_DIR/worker_suite_report.json" --quiet --pretty

curl -sS "$BASE_URL/api/ps" > "$OUT_DIR/ollama_ps_after.json"
if command -v docker >/dev/null 2>&1; then
  docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' > "$OUT_DIR/docker_stats_after.txt" || true
fi

if [[ "${RUN_OPENCODE:-0}" == "1" ]] && command -v opencode >/dev/null 2>&1; then
  echo "=== 4. Optional OpenCode baseline (guarded with 180s timeout) ==="
  START_NS=$(date +%s%N)
  timeout 180s opencode run "$PROMPT" > "$OUT_DIR/opencode_output.txt" 2>&1 || echo "opencode_exit=$?" >> "$OUT_DIR/opencode_output.txt"
  END_NS=$(date +%s%N)
  echo "opencode_wall_clock_ms=$(( (END_NS - START_NS) / 1000000 ))" | tee "$OUT_DIR/opencode_wall_ms.txt"
fi

echo "Done. Results written to: $OUT_DIR"
