import os

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets", "locomo10.json")
# Override AETHER_DB_PATH to keep V1/V2 (or any A/B) hypergraph kernels in
# isolated directories so reruns don't reuse stale pickles.
AETHER_DB_PATH = os.getenv("AETHER_DB_PATH", os.path.join(BASE_DIR, "aether_db_qwen"))

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
#VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")

EMBEDDER_MODEL = os.getenv("AETHER_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
EMBEDDER_DEVICE = os.getenv("AETHER_EMBEDDER_DEVICE", "cpu")

AETHER_SIMILARITY_THRESHOLD = float(os.getenv("AETHER_SIMILARITY_THRESHOLD", "0.65"))

AETHER_ADD_BATCH_SIZE = int(os.getenv("AETHER_ADD_BATCH_SIZE", "4"))
AETHER_ADD_WORKERS = int(os.getenv("AETHER_ADD_WORKERS", "24"))
AETHER_SEARCH_WORKERS = int(os.getenv("AETHER_SEARCH_WORKERS", "32"))
AETHER_EVAL_WORKERS = int(os.getenv("AETHER_EVAL_WORKERS", "32"))

DEFAULT_TOP_K = int(os.getenv("AETHER_TOP_K", "30"))
DEFAULT_MAX_TOKENS = int(os.getenv("AETHER_MAX_TOKENS", "512"))
DEFAULT_EXTRACT_MAX_TOKENS = int(os.getenv("AETHER_EXTRACT_MAX_TOKENS", "2048"))


def get_vllm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
