import os

from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(os.path.dirname(BASE_DIR), "datasets", "locomo10.json")
RAG_DB_PATH = os.path.join(BASE_DIR, "mem_db")
RAG_CHROMA_COLLECTION_NAME = os.getenv("RAG_CHROMA_COLLECTION_NAME", "locomo_rag")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
EMBEDDER_MODEL = os.getenv("RAG_EMBEDDER_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "512"))
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "2"))


def get_vllm_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
