# Performance Benchmark Suite

This directory contains reproducible benchmark definitions and scripts to compare `ollama-worker`, direct `curl` calls against Native Ollama API, and (optionally) OpenCode on the target Linux Home Server.

## Quick Usage

### 1. Run Built-in Suite (S1 Small, S2 Medium, S3 Large)

```bash
# Warm-start streaming benchmark (measures TTFT and generation throughput)
ollama-worker benchmark \
  --model qwen2.5-coder:3b \
  --stream \
  --iterations 2 \
  --pretty

# Compare /api/chat vs /api/generate with explicit num_ctx
ollama-worker benchmark \
  --model qwen2.5-coder:3b \
  --endpoint both \
  --num-ctx 4096 \
  --output bench_4096.json --pretty

# Cold-start measurement (unloads model via keep_alive=0 before iteration 1)
ollama-worker benchmark \
  --model qwen2.5-coder:3b \
  --cold-start \
  --stream \
  --output bench_cold.json --pretty
```

### 2. Full Comparison Script on Linux Home Server (`curl` vs `ollama-worker` vs OpenCode)

```bash
chmod +x benchmarks/compare_curl_vs_worker.sh
./benchmarks/compare_curl_vs_worker.sh

# Optionally include OpenCode (guarded by a 180s timeout)
RUN_OPENCODE=1 ./benchmarks/compare_curl_vs_worker.sh
```

Results are written to `benchmarks/results/<timestamp>/` (git-ignored) and capture wall-clock duration, Ollama nanosecond metrics, TTFT, `/usr/bin/time -v` Peak RSS, `/api/ps` model memory, and `docker stats --no-stream`.
