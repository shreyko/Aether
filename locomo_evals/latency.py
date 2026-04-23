"""Shared latency helpers: per-sample timings and p50/p95/p99 summaries for LOCOMO baselines."""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping

import numpy as np


def percentile_summary_seconds(values: list[float] | Iterable[float]) -> dict[str, float] | None:
    """Return p50/p95/p99 in seconds, or None if empty."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return None
    qs = np.percentile(arr, [50, 95, 99])
    return {"p50": float(qs[0]), "p95": float(qs[1]), "p99": float(qs[2])}


def search_results_sidecar_path(search_results_json: str) -> str:
    """Path for latency summary JSON next to a ``*_search_results*.json`` file."""
    base, _ext = os.path.splitext(search_results_json)
    return f"{base}.latency_summary.json"


def collect_qa_latency_rows(results: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _k, items in results.items():
        for r in items:
            if isinstance(r, dict) and "error" not in r:
                rows.append(r)
    return rows


def summarize_qa_latencies(results: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate search / generation / total latency percentiles from per-QA result dicts."""
    rows = collect_qa_latency_rows(results)
    search_v: list[float] = []
    gen_v: list[float] = []
    total_v: list[float] = []
    for r in rows:
        if "search_latency_sec" in r:
            search_v.append(float(r["search_latency_sec"]))
        if "generation_latency_sec" in r:
            gen_v.append(float(r["generation_latency_sec"]))
        if "total_latency_sec" in r:
            total_v.append(float(r["total_latency_sec"]))
    out: dict[str, Any] = {"n_qa": len(rows)}
    for key, vals in (
        ("search_latency_sec", search_v),
        ("generation_latency_sec", gen_v),
        ("total_latency_sec", total_v),
    ):
        ps = percentile_summary_seconds(vals)
        if ps is not None:
            out[key] = ps
        out[f"{key}_count"] = len(vals)
    return out


def write_search_latency_summary(search_results_json: str, results: Mapping[str, list[dict[str, Any]]], baseline: str) -> str:
    """Write latency summary next to search results; returns path written."""
    payload = {"baseline": baseline, "phase": "search", **summarize_qa_latencies(results)}
    path = search_results_sidecar_path(search_results_json)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_add_latency_summary(
    output_path: str,
    baseline: str,
    per_batch_seconds: list[float],
    *,
    primary_key: str = "per_batch_add_sec",
    extra_sections: dict[str, list[float]] | None = None,
    **meta: Any,
) -> str:
    """Write add/index phase latency percentiles (one sample per ingest batch by default).

    ``primary_key`` names the percentile block for ``per_batch_seconds`` (e.g. RAG uses
    ``per_conversation_index_sec`` when each sample is one conversation's embed+upsert).
    """
    sections: dict[str, Any] = {"baseline": baseline, "phase": "add", **meta}
    sections[f"{primary_key}_n"] = len(per_batch_seconds)
    ps = percentile_summary_seconds(per_batch_seconds)
    if ps is not None:
        sections[primary_key] = ps
    if extra_sections:
        for name, vals in extra_sections.items():
            sections[f"{name}_n"] = len(vals)
            p = percentile_summary_seconds(vals)
            if p is not None:
                sections[name] = p
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2)
    return output_path
