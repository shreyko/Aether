"""Mem0 **graph** baseline: mem0ai 1.x with embedded Kuzu (KG) + Chroma vector store."""

import os

from openai import OpenAI

from ..mem0_baseline.config import CUSTOM_INSTRUCTIONS, VLLM_BASE_URL, VLLM_MODEL

_BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_KUZU_DIR = os.path.join(_BASELINE_DIR, "mem0g_kuzu")
os.makedirs(_KUZU_DIR, exist_ok=True)

# mem0ai 1.x MemoryConfig expects ``custom_fact_extraction_prompt`` (not ``custom_instructions``).
MEM0G_CONFIG = {
    "version": "v1.1",
    "llm": {
        "provider": "vllm",
        "config": {
            "model": VLLM_MODEL,
            "vllm_base_url": VLLM_BASE_URL,
            "temperature": 0.0,
            "max_tokens": 4000,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "embedding_dims": 384,
            "model_kwargs": {"device": os.getenv("MEM0_EMBEDDER_DEVICE", "cpu")},
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "locomo_mem0g",
            "path": os.path.join(_BASELINE_DIR, "mem0g_db"),
        },
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db": os.path.join(_KUZU_DIR, "mem0_graph.kuzu"),
        },
    },
    "custom_fact_extraction_prompt": CUSTOM_INSTRUCTIONS,
}


def get_vllm_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
