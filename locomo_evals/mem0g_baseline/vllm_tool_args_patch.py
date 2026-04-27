"""Work around mem0 VllmLLM tool JSON parsing when the model emits extra text after the first object.

Qwen3 (and similar) occasionally return tool ``function.arguments`` where a valid JSON object
is followed by trailing text or a second value. ``json.loads(extract_json(...))`` then raises
``JSONDecodeError: Extra data``. Parsing with :class:`json.JSONDecoder` ``raw_decode`` keeps
only the first complete JSON value, matching what the graph pipeline needs.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED = False


def _first_json_value(s: str) -> Any:
    from mem0.memory.utils import extract_json

    raw = (extract_json(s) or "").strip()
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(raw)
        if end < len(raw):
            logger.debug(
                "mem0g: took first JSON value only; dropped %d trailing chars in tool arguments",
                len(raw) - end,
            )
        return obj
    except json.JSONDecodeError:
        return json.loads(raw)


def apply_vllm_tool_args_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from mem0.llms.vllm import VllmLLM

    def _parse_response(self, response, tools):
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": _first_json_value(tool_call.function.arguments or ""),
                        }
                    )
            return processed_response
        return response.choices[0].message.content

    VllmLLM._parse_response = _parse_response
    _PATCHED = True


apply_vllm_tool_args_patch()
