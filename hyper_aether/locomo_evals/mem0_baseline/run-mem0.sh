#!/usr/bin/env bash
set -euo pipefail

# One-command runner:
# 1) starts vLLM (if not already running)
# 2) runs mem0 baseline end-to-end (add -> search -> eval -> scores)
# 3) stops vLLM started by this script (unless KEEP_VLLM_ALIVE=1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

TOP_K="${TOP_K:-30}"
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-3B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_HEALTH_TIMEOUT_SECS="${VLLM_HEALTH_TIMEOUT_SECS:-180}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${RESULTS_DIR}/vllm_server.log}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"

# Optional: provide a fully custom serve command.
# Example:
#   VLLM_SERVE_CMD='vllm serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000'
VLLM_SERVE_CMD="${VLLM_SERVE_CMD:-vllm serve ${VLLM_MODEL} --host ${VLLM_HOST} --port ${VLLM_PORT}}"

mkdir -p "${RESULTS_DIR}"
VLLM_PID=""
STARTED_VLLM=0

log() {
  echo "[run_all_with_vllm] $*"
}

is_vllm_healthy() {
  curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1
}

cleanup() {
  if [[ "${STARTED_VLLM}" -eq 1 && -n "${VLLM_PID}" && "${KEEP_VLLM_ALIVE}" != "1" ]]; then
    log "Stopping vLLM process ${VLLM_PID}..."
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

log "Project root: ${PROJECT_ROOT}"
log "Using VLLM_BASE_URL=${VLLM_BASE_URL}"
log "Using VLLM_MODEL=${VLLM_MODEL}"
log "Using TOP_K=${TOP_K}"

if is_vllm_healthy; then
  log "Detected healthy vLLM at ${VLLM_BASE_URL}; reusing existing server."
else
  log "Starting vLLM in background..."
  log "Command: ${VLLM_SERVE_CMD}"
  bash -lc "${VLLM_SERVE_CMD}" >"${VLLM_LOG_PATH}" 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1
  log "vLLM PID=${VLLM_PID}; logs -> ${VLLM_LOG_PATH}"

  start_ts="$(date +%s)"
  while ! is_vllm_healthy; do
    now_ts="$(date +%s)"
    elapsed="$((now_ts - start_ts))"
    if [[ "${elapsed}" -ge "${VLLM_HEALTH_TIMEOUT_SECS}" ]]; then
      log "Timed out waiting for vLLM health after ${VLLM_HEALTH_TIMEOUT_SECS}s."
      log "Check logs: ${VLLM_LOG_PATH}"
      exit 1
    fi
    sleep 2
  done
  log "vLLM is healthy."
fi

export VLLM_BASE_URL
export VLLM_MODEL

log "Running mem0 baseline (all phases)..."
cd "${PROJECT_ROOT}"
uv run -- python -m locomo_evals.mem0_baseline.run --method all --top_k "${TOP_K}"
log "Pipeline completed successfully."

