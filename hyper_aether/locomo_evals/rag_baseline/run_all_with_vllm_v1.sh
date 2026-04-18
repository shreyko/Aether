#!/bin/bash/env bash

# One-command runner for the RAG baseline:
# 1) starts vLLM (if not already running)
# 2) runs the RAG baseline end-to-end (index -> search -> eval -> scores)
# 3) stops vLLM started by this script (unless KEEP_VLLM_ALIVE=1)
#
# vLLM lives in this project's venv (optional extra `vllm`). Make sure it is
# installed on a GPU node with a recent CUDA module:
#     module load cuda/12.8.1
#     uv sync --extra vllm --extra mem0-baseline
# then run this script on a GPU node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

CHUNK_SIZE="${CHUNK_SIZE:-512}"
TOP_K="${TOP_K:-2}"
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_HEALTH_TIMEOUT_SECS="${VLLM_HEALTH_TIMEOUT_SECS:-1200}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${RESULTS_DIR}/vllm_server.log}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
UV_BIN="${UV_BIN:-uv}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VLLM_DEVICE="${VLLM_DEVICE:-auto}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
VLLM_SERVE_CMD="${VLLM_SERVE_CMD:-}"

mkdir -p "${RESULTS_DIR}"
VLLM_PID=""
STARTED_VLLM=0

log() {
  echo "[run_all_with_vllm] $*"
}

is_vllm_healthy() {
  curl -fsS "${VLLM_BASE_URL}/models" >/dev/null 2>&1
}

vllm_supports_device_flag() {
  "${UV_BIN}" run -- vllm serve -h 2>/dev/null | grep -q -- "--device"
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
log "Using CHUNK_SIZE=${CHUNK_SIZE}"
log "Using TOP_K=${TOP_K}"
log "Using UV_BIN=${UV_BIN}"

if [[ "${VLLM_DEVICE}" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    VLLM_DEVICE="cuda"
  else
    VLLM_DEVICE="cpu"
  fi
fi
log "Using VLLM_DEVICE=${VLLM_DEVICE}"

if [[ -z "${VLLM_SERVE_CMD}" ]]; then
  chat_template_arg=""
  device_arg=""
  if [[ -n "${VLLM_CHAT_TEMPLATE}" ]]; then
    chat_template_arg="--chat-template ${VLLM_CHAT_TEMPLATE}"
  fi
  if vllm_supports_device_flag; then
    device_arg="--device ${VLLM_DEVICE}"
  else
    log "Detected vLLM CLI without --device support; skipping device flag."
  fi
  VLLM_SERVE_CMD="${UV_BIN} run -- vllm serve ${VLLM_MODEL} --host ${VLLM_HOST} --port ${VLLM_PORT} ${device_arg} ${chat_template_arg} ${VLLM_EXTRA_ARGS}"
fi

if is_vllm_healthy; then
  log "Detected healthy vLLM at ${VLLM_BASE_URL}; reusing existing server."
else
  log "Starting vLLM in background..."
  log "Command: ${VLLM_SERVE_CMD}"
  # PYTHONUNBUFFERED=1 so progress (weight download, CUDA graph capture) shows
  # up in the log in real time instead of being flushed only on exit.
  ( cd "${PROJECT_ROOT}" && PYTHONUNBUFFERED=1 bash -c "${VLLM_SERVE_CMD}" ) >"${VLLM_LOG_PATH}" 2>&1 &
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
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
      log "vLLM process ${VLLM_PID} exited before becoming healthy."
      log "Check logs: ${VLLM_LOG_PATH}"
      exit 1
    fi
    sleep 2
  done
  log "vLLM is healthy."
fi

export VLLM_BASE_URL
export VLLM_MODEL

log "Running RAG baseline (all phases)..."
cd "${PROJECT_ROOT}"
"${UV_BIN}" run -- python -m locomo_evals.rag_baseline.run --method all --chunk_size "${CHUNK_SIZE}" --top_k "${TOP_K}"
log "Pipeline completed successfully."
