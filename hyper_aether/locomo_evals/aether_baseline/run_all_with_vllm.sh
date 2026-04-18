#!/bin/bash
#SBATCH --job-name=aether
#SBATCH --output=./job_logs/aether_baseline_%j.log
#SBATCH --nodes=1
#SBATCH --partition=spgpu2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --account=jjparkcv_owned1

source ~/.bashrc
module load gcc/11
module load cuda/12.8
source /home/gpranav/pranav_work/scratch/sca/Aether/hyper_aether/.venv/bin/activate
set -euo pipefail

# HF_TOKEN must be set in your shell env (e.g. in ~/.bashrc) before sbatch.
# Never hardcode the token in this file — GitHub push-protection will block
# the push and the token will leak into slurm logs / shared fs snapshots.
export HF_TOKEN=""
export HF_HOME="/home/gpranav/pranav_work/scratch/hf_cache"
export CACHE_DIR="/home/gpranav/pranav_work/scratch/hf_cache"

# One-command runner for the Aether baseline:
# 1) starts vLLM (if not already running)
# 2) runs the aether baseline end-to-end (add -> search -> eval -> scores)
# 3) stops vLLM started by this script (unless KEEP_VLLM_ALIVE=1)
#
# vLLM lives in this project's venv (optional extra `vllm`). Make sure it is
# installed on a GPU node with a recent CUDA module:
#     module load cuda/12.8.1
#     uv sync --extra vllm --extra mem0-baseline
# then run this script on a GPU node, e.g.:
#     sbatch locomo_evals/aether_baseline/run_all_with_vllm.sh

# Under sbatch, Slurm copies this script to /var/spool/slurmd.spool/jobXXX/
# so ${BASH_SOURCE[0]} no longer resolves to the repo file. Prefer an
# explicit override, then $SLURM_SUBMIT_DIR (where sbatch was invoked),
# then fall back to the repo path implied by the venv we just sourced.
PROJECT_ROOT="${PROJECT_ROOT:-/home/gpranav/pranav_work/scratch/sca/Aether/hyper_aether}"
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/locomo_evals/aether_baseline}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results}"

TOP_K="${TOP_K:-30}"
VLLM_MODEL="${VLLM_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
VLLM_HEALTH_TIMEOUT_SECS="${VLLM_HEALTH_TIMEOUT_SECS:-1200}"
VLLM_LOG_PATH="${VLLM_LOG_PATH:-${RESULTS_DIR}/vllm_server_${SLURM_JOB_ID:-local}.log}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
UV_BIN="${UV_BIN:-uv}"
# Tuned for L40S (48GB, native BF16). max-model-len=32k is well over the
# ANSWER_PROMPT worst case (top_k=30 memories) and frees up KV cache for
# high concurrency; max-num-seqs=128 lets vLLM actually batch the parallel
# client requests we issue from the ADD/SEARCH/EVAL phases.
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---dtype bfloat16 --max-model-len 32768 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9}"
VLLM_DEVICE="${VLLM_DEVICE:-auto}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
VLLM_SERVE_CMD="${VLLM_SERVE_CMD:-}"

# Client-side concurrency for the aether pipeline. Tune down if you see
# vLLM OOM / queue buildup; tune up if vLLM KV cache stays under-utilized.
# AETHER_ADD_BATCH_SIZE=4 matches mem0's fact-extraction sweet spot — big
# enough to amortize per-call overhead, small enough to keep the guided
# JSON output under the extractor's token budget.
AETHER_ADD_WORKERS="${AETHER_ADD_WORKERS:-24}"
AETHER_ADD_BATCH_SIZE="${AETHER_ADD_BATCH_SIZE:-4}"
AETHER_SEARCH_WORKERS="${AETHER_SEARCH_WORKERS:-32}"
AETHER_EVAL_WORKERS="${AETHER_EVAL_WORKERS:-32}"
AETHER_EMBEDDER_DEVICE="${AETHER_EMBEDDER_DEVICE:-auto}"

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

# Keep the embedder on CPU by default. On Slurm clusters like spgpu2 the
# GPU is in EXCLUSIVE_PROCESS compute mode, so once vLLM has claimed a
# CUDA context, any second process trying to .to('cuda') dies with
# "CUDA-capable device(s) is/are busy or unavailable". MiniLM is tiny
# (~22M params) — CPU embedding is not the bottleneck once the LLM
# calls are parallelized.
if [[ "${AETHER_EMBEDDER_DEVICE}" == "auto" ]]; then
  AETHER_EMBEDDER_DEVICE="cpu"
fi
log "Using AETHER_EMBEDDER_DEVICE=${AETHER_EMBEDDER_DEVICE}"
log "Using AETHER_ADD_WORKERS=${AETHER_ADD_WORKERS} AETHER_ADD_BATCH_SIZE=${AETHER_ADD_BATCH_SIZE}"
log "Using AETHER_SEARCH_WORKERS=${AETHER_SEARCH_WORKERS} AETHER_EVAL_WORKERS=${AETHER_EVAL_WORKERS}"

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
export AETHER_EMBEDDER_DEVICE
export AETHER_ADD_WORKERS
export AETHER_ADD_BATCH_SIZE
export AETHER_SEARCH_WORKERS
export AETHER_EVAL_WORKERS

# PHASES controls which pipeline steps to run. Default "all" runs the
# end-to-end pipeline in one process (add -> search -> eval -> scores).
# To resume after a partial failure (e.g. ADD succeeded but SEARCH
# crashed), set PHASES="search eval scores" — the persistent aether_db/
# kernels from the previous ADD run are reused. Also set
# AETHER_SKIP_DELETE_ALL=1 when rerunning ADD if you want to keep the
# existing kernels.
PHASES="${PHASES:-all}"
log "Using PHASES=${PHASES}"

log "Running aether baseline..."
cd "${PROJECT_ROOT}"
for phase in ${PHASES}; do
  log "==== Phase: ${phase} ===="
  "${UV_BIN}" run -- python -u -m locomo_evals.aether_baseline.run --method "${phase}" --top_k "${TOP_K}"
done
log "Pipeline completed successfully."
