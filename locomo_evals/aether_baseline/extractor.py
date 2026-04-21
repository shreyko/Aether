"""Memory extractor for the Aether LOCOMO baseline.

Ports ``hyper_aether/extractor.py`` to use a running vLLM HTTP server via
the OpenAI-compatible client, the same pattern used by the mem0 / RAG
baselines. This avoids loading the model twice in a single process (once
as an OpenAI server and once via ``vllm.LLM``) and lets us share one vLLM
instance across the ADD phase's worker pool.

Structured JSON outputs are enforced with vLLM's ``guided_json`` extra body
parameter, which is equivalent to calling ``SamplingParams(guided_decoding=
GuidedDecodingParams(json=...))`` with the in-process API.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from .config import DEFAULT_EXTRACT_MAX_TOKENS, VLLM_MODEL, get_vllm_client


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

Respond ONLY with a JSON object of the form {{"memories": [{{"node_id": "...", "abstraction": "...", "value": "...", "contexts": ["...", "..."]}}, ...]}}

Conversation Chunk:
\"{chunk}\"
""".strip()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_json(text: str) -> dict | None:
    """Best-effort extraction of the first JSON object from ``text``."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def extract_hypergraph_nodes(
    transcript_chunk: str,
    client=None,
    model_name: str = VLLM_MODEL,
    max_tokens: int = DEFAULT_EXTRACT_MAX_TOKENS,
    retries: int = 2,
) -> list[MemoryEntry]:
    """Extract hypergraph memory nodes from ``transcript_chunk`` via vLLM.

    Args:
        transcript_chunk: The conversation text to analyze.
        client: An ``openai.OpenAI`` client pointed at a vLLM server. If
            ``None`` a fresh client is created from the module config.
        model_name: The served vLLM model name.
        max_tokens: Output budget for the extraction response.
        retries: Number of retries if the JSON fails to parse.
    """
    if client is None:
        client = get_vllm_client()

    prompt = EXTRACTION_PROMPT.format(chunk=transcript_chunk)
    schema = ExtractionResult.model_json_schema()

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"guided_json": schema},
            )
            text = response.choices[0].message.content or ""
            payload = _coerce_json(text)
            if payload is None:
                raise ValueError("extractor returned non-JSON response")
            result = ExtractionResult.model_validate(payload)
            return result.memories
        except Exception as e:  # pragma: no cover - network/model error path
            last_err = e
            if attempt >= retries:
                break

    print(f"[Extractor Error] {last_err}")
    return []
