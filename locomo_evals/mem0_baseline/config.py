import os

from openai import OpenAI

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

MEM0_CONFIG = {
    "llm": {
        "provider": "vllm",
        "config": {
            "model": VLLM_MODEL,
            "vllm_base_url": VLLM_BASE_URL,
            "temperature": 0.0,
            # 4000 gives mem0's fact-extractor enough headroom to finish
            # the JSON response for batch_size up to ~8; with 2000 we were
            # seeing frequent mid-JSON truncation ("Expecting ',' delimiter"
            # at char ~6800-8800) which silently drops the batch's memories.
            "max_tokens": 4000,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2",
            "embedding_dims": 384,
            # Pin the sentence-transformer to CPU so it doesn't fight vLLM for
            # the GPU (vLLM grabs ~90% of VRAM by default). MiniLM is tiny, so
            # CPU embedding is plenty fast for LOCOMO-scale workloads.
            "model_kwargs": {"device": os.getenv("MEM0_EMBEDDER_DEVICE", "cpu")},
        },
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "locomo_mem0",
            "path": os.path.join(os.path.dirname(__file__), "mem0_db"),
        },
    },
}

CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


def get_vllm_client() -> OpenAI:
    """Return an OpenAI client pointed at the local vLLM server."""
    return OpenAI(base_url=VLLM_BASE_URL, api_key="unused")
