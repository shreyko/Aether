import json
import os
from enum import Enum

import ollama
from pydantic import BaseModel


class Backend(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"


class MemoryEntry(BaseModel):
    node_id: str
    abstraction: str
    value: str
    contexts: list[str]


class ExtractionResult(BaseModel):
    memories: list[MemoryEntry]


EXTRACTION_PROMPT = """
Analyze the following conversation chunk. Extract factual memories that could be useful for a persistent AI assistant.
For each memory, provide:
1. node_id: A unique snake_case identifier
2. abstraction: A high-level summary (Primary Abstraction)
3. value: The raw, specific detail mentioned (Memory Value)
4. contexts: A list of 1-3 situational envelopes or themes this belongs to. Use consistent theme names.

Conversation Chunk:
"{chunk}"
"""


def _extract_via_ollama(prompt: str, model_name: str) -> list[MemoryEntry]:
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=ExtractionResult.model_json_schema(),
        options={"temperature": 0.0},
    )
    result = ExtractionResult.model_validate_json(response.message.content)
    return result.memories


def _extract_via_vllm(prompt: str, llm) -> list[MemoryEntry]:
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    guided = GuidedDecodingParams(json=ExtractionResult.model_json_schema())
    params = SamplingParams(temperature=0.0, guided_decoding=guided)

    outputs = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        sampling_params=params,
    )
    text = outputs[0].outputs[0].text
    result = ExtractionResult.model_validate_json(text)
    return result.memories


_DEFAULT_OLLAMA_EXTRACT = os.getenv("OLLAMA_EXTRACT_MODEL", os.getenv("OLLAMA_MODEL", "qwen3.5:4b"))


def extract_hypergraph_nodes(
    transcript_chunk: str,
    model_name: str | None = None,
    backend: Backend | str = Backend.OLLAMA,
    llm=None,
) -> list[MemoryEntry]:
    """
    Extract nodes and hypergraph contexts from a conversation chunk.

    Args:
        transcript_chunk: The conversation text to analyze.
        model_name: Model identifier (e.g. ``qwen3.5:4b`` for Ollama,
            ``Qwen/Qwen3.5-4B`` for vLLM). Defaults to ``OLLAMA_EXTRACT_MODEL``
            / ``OLLAMA_MODEL`` / ``qwen3.5:4b``.
        backend: Inference backend — "ollama" or "vllm".
        llm: A pre-initialized ``vllm.LLM`` instance (required when
             backend is "vllm").  Create once and reuse across calls
             to avoid reloading the model every time.
    """
    if model_name is None:
        model_name = _DEFAULT_OLLAMA_EXTRACT
    backend = Backend(backend)
    prompt = EXTRACTION_PROMPT.format(chunk=transcript_chunk)

    try:
        if backend == Backend.OLLAMA:
            return _extract_via_ollama(prompt, model_name)

        if llm is None:
            raise ValueError(
                "A vllm.LLM instance must be passed via the `llm` argument "
                "when using the vllm backend."
            )
        return _extract_via_vllm(prompt, llm)
    except Exception as e:
        print(f"[Extractor Error] {e}")
        return []
