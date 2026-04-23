"""Compare LOCOMO evaluation metrics across the RAG, mem0, and Aether baselines.

Reads each baseline's ``evaluation_metrics.json`` (produced by its own
``run.py`` eval phase) and emits:

  * ``comparison.json`` — machine-readable aggregates:
        overall / per-category / per-conversation means for BLEU, F1,
        LLM-judge accuracy, and per-baseline QA counts.
  * ``comparison.md`` — human-readable markdown report with the same
    tables formatted for eyeballing / pasting into docs.

All three baselines share the same LOCOMO QA set and the same
``ANSWER_PROMPT`` / ``llm_as_a_judge`` scoring, so any differences
isolate the retrieval-pipeline effect (flat chunks vs flat extracted
memories vs hypergraph envelope).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd

LOCOMO_EVALS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LOCOMO QA category codes (see mem0ai/evals/locomo and the dataset card).
CATEGORY_LABELS: dict[int, str] = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "open-domain",
    5: "adversarial",
}

BASELINES: dict[str, str] = {
    "rag": os.path.join(LOCOMO_EVALS_ROOT, "rag_baseline", "results", "evaluation_metrics.json"),
    "mem0": os.path.join(LOCOMO_EVALS_ROOT, "mem0_baseline", "results", "evaluation_metrics.json"),
    "aether": os.path.join(LOCOMO_EVALS_ROOT, "aether_baseline", "results", "evaluation_metrics.json"),
}

METRICS = ["bleu_score", "f1_score", "llm_score"]


def _flatten(path: str, baseline: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw: dict[str, list[dict[str, Any]]] = json.load(f)

    rows: list[dict[str, Any]] = []
    for conv_id, qa_list in raw.items():
        for qa in qa_list:
            row = {
                "baseline": baseline,
                "conversation": int(conv_id),
                "category": int(qa.get("category", -1)),
                "bleu_score": float(qa.get("bleu_score", 0.0)),
                "f1_score": float(qa.get("f1_score", 0.0)),
                "llm_score": int(qa.get("llm_score", 0)),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_frame(df: pd.DataFrame, group_cols: list[str] | None) -> pd.DataFrame:
    agg = {m: "mean" for m in METRICS}
    agg["category"] = "count"  # any column works for count
    if group_cols:
        out = df.groupby(group_cols).agg(
            bleu=("bleu_score", "mean"),
            f1=("f1_score", "mean"),
            llm=("llm_score", "mean"),
            n=("category", "count"),
        )
    else:
        out = df.agg(
            bleu=("bleu_score", "mean"),
            f1=("f1_score", "mean"),
            llm=("llm_score", "mean"),
            n=("category", "count"),
        )
    return out.round(4)


def _pivot(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    """Produce a pivot with one row per ``index_col`` and columns (metric, baseline)."""
    g = df.groupby(["baseline", index_col]).agg(
        bleu=("bleu_score", "mean"),
        f1=("f1_score", "mean"),
        llm=("llm_score", "mean"),
        n=("category", "count"),
    )
    pivot = g.unstack("baseline").round(4)
    return pivot


def _overall_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("baseline").agg(
        bleu=("bleu_score", "mean"),
        f1=("f1_score", "mean"),
        llm=("llm_score", "mean"),
        n=("category", "count"),
    )
    return g.round(4)


def _to_markdown_row(vals: list[str]) -> str:
    return "| " + " | ".join(vals) + " |"


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_comparison(
    baselines: dict[str, str] = BASELINES,
    out_dir: str | None = None,
) -> dict[str, Any]:
    frames = []
    for name, path in baselines.items():
        if not os.path.exists(path):
            print(f"[compare] WARNING: missing {path}; skipping '{name}'")
            continue
        frames.append(_flatten(path, name))
    if not frames:
        raise FileNotFoundError("No baseline evaluation_metrics.json files found.")

    df = pd.concat(frames, ignore_index=True)

    overall = _overall_table(df)
    by_category = _pivot(df, "category")
    by_conversation = _pivot(df, "conversation")

    present_baselines = sorted(df["baseline"].unique())
    comparison: dict[str, Any] = {
        "baselines": present_baselines,
        "category_labels": {str(k): v for k, v in CATEGORY_LABELS.items()},
        "overall": {
            baseline: {
                "bleu": float(overall.loc[baseline, "bleu"]),
                "f1": float(overall.loc[baseline, "f1"]),
                "llm_accuracy": float(overall.loc[baseline, "llm"]),
                "n": int(overall.loc[baseline, "n"]),
            }
            for baseline in overall.index
        },
        "per_category": {},
        "per_conversation": {},
    }

    for cat in sorted(df["category"].unique()):
        cat_label = CATEGORY_LABELS.get(cat, f"cat_{cat}")
        comparison["per_category"][str(cat)] = {
            "label": cat_label,
            "scores": {
                baseline: {
                    "bleu": float(by_category.loc[cat, ("bleu", baseline)])
                    if ("bleu", baseline) in by_category.columns
                    else None,
                    "f1": float(by_category.loc[cat, ("f1", baseline)])
                    if ("f1", baseline) in by_category.columns
                    else None,
                    "llm_accuracy": float(by_category.loc[cat, ("llm", baseline)])
                    if ("llm", baseline) in by_category.columns
                    else None,
                    "n": int(by_category.loc[cat, ("n", baseline)])
                    if ("n", baseline) in by_category.columns
                    and not pd.isna(by_category.loc[cat, ("n", baseline)])
                    else 0,
                }
                for baseline in present_baselines
            },
        }

    for conv in sorted(df["conversation"].unique()):
        comparison["per_conversation"][str(conv)] = {
            baseline: {
                "bleu": float(by_conversation.loc[conv, ("bleu", baseline)])
                if ("bleu", baseline) in by_conversation.columns
                and not pd.isna(by_conversation.loc[conv, ("bleu", baseline)])
                else None,
                "f1": float(by_conversation.loc[conv, ("f1", baseline)])
                if ("f1", baseline) in by_conversation.columns
                and not pd.isna(by_conversation.loc[conv, ("f1", baseline)])
                else None,
                "llm_accuracy": float(by_conversation.loc[conv, ("llm", baseline)])
                if ("llm", baseline) in by_conversation.columns
                and not pd.isna(by_conversation.loc[conv, ("llm", baseline)])
                else None,
                "n": int(by_conversation.loc[conv, ("n", baseline)])
                if ("n", baseline) in by_conversation.columns
                and not pd.isna(by_conversation.loc[conv, ("n", baseline)])
                else 0,
            }
            for baseline in present_baselines
        }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "comparison.json")
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2)
        md_path = os.path.join(out_dir, "comparison.md")
        with open(md_path, "w") as f:
            f.write(_render_markdown(comparison))
        print(f"[compare] Wrote {json_path}")
        print(f"[compare] Wrote {md_path}")

    _print_tables(comparison)

    return comparison


def _render_markdown(comparison: dict[str, Any]) -> str:
    baselines = comparison["baselines"]
    lines: list[str] = []
    lines.append("# LOCOMO Baseline Comparison")
    lines.append("")
    lines.append(
        "All three baselines share the same LOCOMO QA set, the same "
        "`ANSWER_PROMPT` at generation time, and the same `llm_as_a_judge` "
        "scoring step. Differences therefore isolate the **retrieval "
        "pipeline** (raw chunks vs. LLM-extracted sentences vs. hypergraph "
        "envelope)."
    )
    lines.append("")
    lines.append("Metrics:")
    lines.append("- **BLEU**: standard n-gram overlap against the gold answer.")
    lines.append("- **F1**: token-level F1 against the gold answer.")
    lines.append(
        "- **LLM accuracy**: 0/1 per question from the Llama-3 judge "
        "(`llm_as_a_judge.py`), averaged."
    )
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    header = ["baseline", "n", "BLEU", "F1", "LLM accuracy"]
    lines.append(_to_markdown_row(header))
    lines.append(_to_markdown_row(["---"] * len(header)))
    for baseline in baselines:
        row = comparison["overall"][baseline]
        lines.append(
            _to_markdown_row(
                [
                    baseline,
                    str(row["n"]),
                    _fmt(row["bleu"]),
                    _fmt(row["f1"]),
                    _fmt(row["llm_accuracy"]),
                ]
            )
        )
    lines.append("")

    lines.append("## Per category")
    lines.append("")
    header = ["category", "label", "n (per baseline)"]
    for m in ["BLEU", "F1", "LLM"]:
        for baseline in baselines:
            header.append(f"{m} {baseline}")
    lines.append(_to_markdown_row(header))
    lines.append(_to_markdown_row(["---"] * len(header)))
    for cat_id in sorted(comparison["per_category"].keys(), key=int):
        entry = comparison["per_category"][cat_id]
        label = entry["label"]
        scores = entry["scores"]
        ns = "/".join(str(scores[b]["n"]) for b in baselines)
        row = [cat_id, label, ns]
        for metric_key in ["bleu", "f1", "llm_accuracy"]:
            for baseline in baselines:
                v = scores[baseline][metric_key]
                row.append("-" if v is None else _fmt(v))
        lines.append(_to_markdown_row(row))
    lines.append("")

    lines.append("## Winner by category (LLM accuracy)")
    lines.append("")
    header = ["category", "label"] + [f"{b} LLM" for b in baselines] + ["best"]
    lines.append(_to_markdown_row(header))
    lines.append(_to_markdown_row(["---"] * len(header)))
    for cat_id in sorted(comparison["per_category"].keys(), key=int):
        entry = comparison["per_category"][cat_id]
        scores = entry["scores"]
        vals: list[tuple[str, float]] = []
        row = [cat_id, entry["label"]]
        for baseline in baselines:
            v = scores[baseline]["llm_accuracy"]
            row.append("-" if v is None else _fmt(v))
            if v is not None:
                vals.append((baseline, v))
        winner = max(vals, key=lambda x: x[1])[0] if vals else "-"
        row.append(winner)
        lines.append(_to_markdown_row(row))
    lines.append("")

    lines.append("## Per conversation (LLM accuracy)")
    lines.append("")
    header = ["conv"] + [f"{b}" for b in baselines] + ["best"]
    lines.append(_to_markdown_row(header))
    lines.append(_to_markdown_row(["---"] * len(header)))
    for conv_id in sorted(comparison["per_conversation"].keys(), key=int):
        entry = comparison["per_conversation"][conv_id]
        row = [conv_id]
        vals: list[tuple[str, float]] = []
        for baseline in baselines:
            v = entry[baseline]["llm_accuracy"]
            row.append("-" if v is None else _fmt(v))
            if v is not None:
                vals.append((baseline, v))
        winner = max(vals, key=lambda x: x[1])[0] if vals else "-"
        row.append(winner)
        lines.append(_to_markdown_row(row))
    lines.append("")

    return "\n".join(lines)


def _print_tables(comparison: dict[str, Any]) -> None:
    print("\n=== Overall ===")
    overall_df = pd.DataFrame(comparison["overall"]).T
    overall_df = overall_df[["n", "bleu", "f1", "llm_accuracy"]]
    print(overall_df.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Per category (LLM accuracy) ===")
    cats = sorted(comparison["per_category"].keys(), key=int)
    baselines = comparison["baselines"]
    rows = []
    for cat_id in cats:
        entry = comparison["per_category"][cat_id]
        r = {"category": cat_id, "label": entry["label"]}
        for b in baselines:
            r[b] = entry["scores"][b]["llm_accuracy"]
        rows.append(r)
    print(
        pd.DataFrame(rows)
        .set_index(["category", "label"])
        .to_string(float_format=lambda x: f"{x:.4f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LOCOMO baselines.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(LOCOMO_EVALS_ROOT, "results"),
        help="Directory to write comparison.json and comparison.md into.",
    )
    args = parser.parse_args()
    build_comparison(out_dir=args.out_dir)


if __name__ == "__main__":
    main()
