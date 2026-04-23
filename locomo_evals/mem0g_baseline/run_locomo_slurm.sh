#!/usr/bin/env bash
#SBATCH --job-name=locomo-mem0g
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#
# Full LOCOMO pipeline for mem0 **graph** (Kuzu): add -> search -> eval -> scores.
#
# Requires mem0ai 1.x with graph extras (``uv sync --extra mem0g-baseline``). Do not mix
# with an environment that only has mem0ai 2.x for the vector baseline.
#
# Prerequisites: vLLM at VLLM_BASE_URL or START_VLLM=1.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-4B}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_PORT="${VLLM_PORT:-8000}"
TOP_K="${TOP_K:-30}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/locomo_evals/datasets/locomo10.json}"

cd "${REPO_ROOT}"

if [[ "${START_VLLM:-0}" == "1" ]]; then
  echo "[slurm/mem0g] Starting vLLM (${VLLM_MODEL}) on port ${VLLM_PORT}..."
  if ! command -v vllm >/dev/null 2>&1; then
    echo "[slurm/mem0g] ERROR: vllm not on PATH." >&2
    exit 1
  fi
  vllm serve "${VLLM_MODEL}" --host 127.0.0.1 --port "${VLLM_PORT}" --dtype auto &
  VLLM_PID=$!
  cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; }
  trap cleanup EXIT
  export VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
  export VLLM_PORT
  python3 -c "
import os, time, urllib.request
port = os.environ['VLLM_PORT']
url = f'http://127.0.0.1:{port}/v1/models'
for _ in range(600):
    try:
        urllib.request.urlopen(url, timeout=5)
        print('[slurm/mem0g] vLLM ready')
        break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit('timeout waiting for vLLM')
"
fi

echo "[slurm/mem0g] Running pipeline (top_k=${TOP_K})..."
python3 -m locomo_evals.mem0g_baseline.run \
  --method all \
  --top_k "${TOP_K}" \
  --data_path "${DATA_PATH}" \
  "$@"

echo "[slurm/mem0g] Done."
