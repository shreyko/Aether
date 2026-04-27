import os

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets", "locomo10.json")
RAG_DB_PATH = os.path.join(BASE_DIR, "mem_db")
RAG_CHROMA_COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION_NAME", "locomo_rag")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# Default Qwen3.5: needs `qwen3_5` in Transformers (Aether pins vllm 0.19+ / transformers 5.5+).
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-4B")
EMBEDDER_MODEL = os.getenv("RAG_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
# The embedder is tiny (~90 MB); default to CPU so it never fights vLLM for
# the GPU when both are running on the same node.
EMBEDDER_DEVICE = os.getenv("RAG_EMBEDDER_DEVICE", "cpu")

CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "2"))

# vLLM can batch many concurrent requests; running LLM calls serially wastes
# almost all of the server's throughput. These knobs control concurrency and
# output length for the search/eval phases.
DEFAULT_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "256"))
DEFAULT_LLM_CONCURRENCY = int(os.getenv("RAG_LLM_CONCURRENCY", "32"))


def get_vllm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
