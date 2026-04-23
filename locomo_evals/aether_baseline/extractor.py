"""Typed memory extractor (V2) for the Aether LOCOMO baseline.

This talks to a running vLLM HTTP server via the OpenAI-compatible client,
the same pattern used by the mem0 / RAG baselines. It asks the model to
emit a JSON array of *typed* memory blocks -- one of:

    Temporal/Date, Entity/Person/Pet, Static Fact, Preference/Trait,
    Relationship, Goal/Intention, Spatial/Location, State Change/Update

Plus the usual ``node_id`` / ``abstraction`` / ``value`` / ``contexts``
fields and a small set of type-specific optional fields (``raw_date``,
``entity_name``, ``source_entity``, ``previous_state``, etc.).

Structured JSON outputs are enforced with vLLM's ``guided_json`` extra
body parameter, which is equivalent to calling
``SamplingParams(guided_decoding=GuidedDecodingParams(json=...))`` with
the in-process vLLM API.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from .config import DEFAULT_EXTRACT_MAX_TOKENS, VLLM_MODEL, get_vllm_client


# The 8 V2 block types, plus "Generic" as a safe fallback when the model
# produces something it is not sure about.
BlockType = Literal[
    "Temporal/Date",
    "Entity/Person/Pet",
    "Static Fact",
    "Preference/Trait",
    "Relationship",
    "Goal/Intention",
    "Spatial/Location",
    "State Change/Update",
    "Generic",
]


class MemoryEntry(BaseModel):
    """A single typed memory block emitted by the extractor."""

    node_id: str
    abstraction: str
    value: str
    contexts: list[str] = Field(default_factory=list)
    block_type: BlockType = "Generic"

    # Optional per-type fields. All are Optional[str] so vLLM's guided_json
    # can freely omit (null) them for block types that do not need them.
    raw_date: str | None = None  # Temporal/Date
    entity_name: str | None = None  # Entity/Person/Pet
    entity_type: str | None = None  # Entity/Person/Pet
    category: str | None = None  # Static Fact
    source_entity: str | None = None  # Relationship
    target_entity: str | None = None  # Relationship
    status: str | None = None  # Goal/Intention
    previous_state: str | None = None  # State Change/Update


class ExtractionResult(BaseModel):
    memories: list[MemoryEntry]


EXTRACTION_PROMPT = """
Analyze the following conversation chunk and extract factual, long-term memories useful to a persistent AI assistant.

For each memory, provide:
1. node_id: A unique snake_case identifier (descriptive, not random).
2. abstraction: A high-level summary (primary abstraction).
3. value: The raw, specific detail mentioned in the chunk.
4. contexts: A list of 1-3 situational envelopes / themes this belongs to. Use consistent theme names across memories.
5. block_type: EXACTLY one of:
   - "Temporal/Date"          (anything anchored to a specific date/time, event scheduling, timelines)
   - "Entity/Person/Pet"      (a named person, pet, or other entity in the speaker's life)
   - "Static Fact"            (a stable, objective fact about the speaker: profession, age, allergies, etc.)
   - "Preference/Trait"       (likes, dislikes, personal tastes, personality traits)
   - "Relationship"           (a connection between two entities: family, friends, coworkers)
   - "Goal/Intention"         (a future plan, active goal, or intention)
   - "Spatial/Location"       (a location or spatial relationship)
   - "State Change/Update"    (an update to a previously-established fact, e.g. moved cities, changed jobs)
   - "Generic"                (use only if none of the above fit)

Optional fields -- set them when they fit the block_type, otherwise leave as null:
   - Temporal/Date:        raw_date
   - Entity/Person/Pet:    entity_name, entity_type
   - Static Fact:          category
   - Relationship:         source_entity, target_entity
   - Goal/Intention:       status   (e.g. "Active", "Completed", "Abandoned")
   - State Change/Update:  previous_state

Respond ONLY with a JSON object of the form:
{{"memories": [{{"node_id": "...", "abstraction": "...", "value": "...", "contexts": ["..."], "block_type": "...", "raw_date": null, "entity_name": null, "entity_type": null, "category": null, "source_entity": null, "target_entity": null, "status": null, "previous_state": null}}]}}

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
    """Extract typed hypergraph memory blocks from ``transcript_chunk`` via vLLM."""
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
