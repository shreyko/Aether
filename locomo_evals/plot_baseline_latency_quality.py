"""Visualize LOCOMO baseline latency summaries and LLM-as-judge scores.

Loads per-baseline:
  - ``**/results/*.latency_summary.json`` with ``phase == "search"`` (QA path)
  - ``**/results/add_latency_summary.json`` (ingest/index path where present)
  - ``**/results/evaluation_metrics.json`` for mean ``llm_score`` (0/1 per QA)

Writes figures under ``locomo_evals/results/plots/`` by default.

Dependencies::
    pip install pandas matplotlib seaborn
  or ``uv pip install pandas matplotlib seaborn``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LOCOMO_EVALS_ROOT = Path(__file__).resolve().parent

DEFAULT_RESULTS: dict[str, dict[str, Path]] = {
    "rag": {
        "results": LOCOMO_EVALS_ROOT / "rag_baseline" / "results",
    },
    "mem0": {
        "results": LOCOMO_EVALS_ROOT / "mem0_baseline" / "results",
    },
    "aether": {
        "results": LOCOMO_EVALS_ROOT / "aether_baseline" / "results",
    },
}

PERCENTILES = ("p50", "p95", "p99")

PALETTE_ORDER = {"rag": "#4C72B0", "mem0": "#DD8452", "aether": "#55A868"}
BASELINE_ORDER = ["rag", "mem0", "aether"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _find_search_latency_summary(results_dir: Path) -> Path | None:
    """Prefer the lone ``*.latency_summary.json`` with ``phase == "search"``."""
    candidates: list[Path] = []
    for p in sorted(results_dir.glob("*.latency_summary.json")):
        try:
            data = _load_json(p)
        except OSError:
            continue
        if data.get("phase") == "search":
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple search latency summaries in {results_dir}: "
            + ", ".join(c.name for c in candidates)
        )
    return None


def _search_latency_long_frame(
    baseline: str,
    data: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = {"search_latency_sec": "search", "generation_latency_sec": "generation", "total_latency_sec": "total"}
    for block, comp in labels.items():
        node = data.get(block)
        if not isinstance(node, dict):
            continue
        for p in PERCENTILES:
            if p in node:
                rows.append(
                    {
                        "baseline": baseline,
                        "component": comp,
                        "percentile": p.upper(),
                        "seconds": float(node[p]),
                    }
                )
    return pd.DataFrame(rows)


def _add_phase_rows(results_dir: Path, baseline: str) -> pd.DataFrame:
    path = results_dir / "add_latency_summary.json"
    if not path.is_file():
        return pd.DataFrame()
    data = _load_json(path)
    rows: list[dict[str, Any]] = []

    batch = data.get("per_batch_add_sec")
    if isinstance(batch, dict):
        metric = "per_batch_add"
        note = "per ingest batch"
    elif data.get("per_conversation_index_sec") or data.get("sample_unit") == "conversation":
        batch = data.get("per_conversation_index_sec", {})
        metric = "per_conversation_index"
        note = "per conversation embed+upsert"
    else:
        batch = {}

    if isinstance(batch, dict) and batch:
        for p in PERCENTILES:
            if p in batch:
                rows.append(
                    {
                        "baseline": baseline,
                        "metric_label": metric,
                        "units_note": note,
                        "percentile": p.upper(),
                        "seconds": float(batch[p]),
                    }
                )

    return pd.DataFrame(rows)


def _mean_llm_score(eval_path: Path) -> tuple[float, int]:
    """Return mean llm_score in [0,1] and QA count."""
    raw = _load_json(eval_path)
    scores: list[int] = []
    for qa_list in raw.values():
        for qa in qa_list:
            scores.append(int(qa.get("llm_score", 0)))
    n = len(scores)
    if n == 0:
        return 0.0, 0
    return sum(scores) / n, n


def plot_search_latency_bars(search_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bars: facet by component (search / generation / total), hue = percentile."""
    if search_df.empty:
        print("[plot] No search-phase latency rows; skipping search_latency bar chart.")
        return

    df = search_df.copy()
    df["baseline"] = pd.Categorical(df["baseline"], categories=BASELINE_ORDER, ordered=True)
    pct_order = ["P50", "P95", "P99"]
    df["percentile"] = pd.Categorical(df["percentile"], categories=pct_order, ordered=True)

    g = sns.catplot(
        data=df,
        kind="bar",
        x="baseline",
        y="seconds",
        hue="percentile",
        col="component",
        col_order=["search", "generation", "total"],
        order=BASELINE_ORDER,
        height=4.5,
        aspect=0.9,
        palette="viridis",
        legend_out=False,
        errorbar=None,
    )
    g.set_axis_labels("Baseline", "Latency (seconds)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        for c in ax.containers:
            ax.bar_label(c, fmt="%.2f", padding=2, fontsize=7)
    g.figure.suptitle("QA phase: retrieval + generation latency (percentiles)", y=1.02, fontsize=13)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def plot_add_latency_bars(add_df: pd.DataFrame, out_path: Path) -> None:
    if add_df.empty:
        print("[plot] No add-phase latency rows; skipping add_latency bar chart.")
        return

    df = add_df.copy()
    df["baseline"] = pd.Categorical(df["baseline"], categories=BASELINE_ORDER, ordered=True)
    pct_order = ["P50", "P95", "P99"]
    df["percentile"] = pd.Categorical(df["percentile"], categories=pct_order, ordered=True)

    g = sns.catplot(
        data=df,
        kind="bar",
        x="baseline",
        y="seconds",
        hue="percentile",
        height=5,
        aspect=1.15,
        order=BASELINE_ORDER,
        palette="crest",
        legend_out=False,
        errorbar=None,
    )
    ax = g.ax

    notes = sorted({str(x) for x in df.get("units_note", pd.Series(dtype=object)).dropna().unique()})
    title = "Ingest / index phase latency (percentiles)"
    if notes:
        title += "\nUnits: " + "; ".join(notes)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Latency (seconds)")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(g.figure)


def plot_flat_all_metrics(search_df: pd.DataFrame, out_path: Path) -> None:
    """One clustered bar chart: metric = component × percentile string, grouped by baseline."""
    if search_df.empty:
        return
    df = search_df.copy()
    df["metric"] = df["component"] + " " + df["percentile"]
    metric_order = [f"{c} {p}" for c in ["search", "generation", "total"] for p in ["P50", "P95", "P99"]]
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
    df["baseline"] = pd.Categorical(df["baseline"], categories=BASELINE_ORDER, ordered=True)

    fig_h = max(6, len(metric_order) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.barplot(
        data=df,
        y="metric",
        x="seconds",
        hue="baseline",
        order=metric_order,
        hue_order=BASELINE_ORDER,
        palette=[PALETTE_ORDER[b] for b in BASELINE_ORDER],
        ax=ax,
        errorbar=None,
    )
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("")
    ax.set_title("All QA-phase latency metrics (horizontal bars)")
    ax.legend(title="baseline", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_p99(
    summary_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    p99_metric: str = "total_latency_sec",
) -> None:
    """Scatter: x = mean LLM judge accuracy, y = chosen p99 latency (seconds)."""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        print("[plot] No rows for accuracy vs p99; skipping.")
        return

    df["baseline"] = pd.Categorical(df["baseline"], categories=BASELINE_ORDER, ordered=True)

    metric_label = {"total_latency_sec": "total", "search_latency_sec": "search", "generation_latency_sec": "generation"}.get(
        p99_metric,
        p99_metric,
    )
    col = "p99_total_sec"
    if p99_metric == "search_latency_sec":
        col = "p99_search_sec"
    elif p99_metric == "generation_latency_sec":
        col = "p99_gen_sec"

    fig, ax = plt.subplots(figsize=(8, 5.5))

    seen: set[str] = set()

    for _, row in df.iterrows():
        b = str(row["baseline"])
        show_label = b not in seen
        seen.add(b)
        ax.scatter(
            row["mean_llm_accuracy"],
            row[col],
            s=280,
            c=PALETTE_ORDER.get(b, "#333333"),
            label=b if show_label else None,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            b,
            (row["mean_llm_accuracy"], row[col]),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=10,
            ha="left",
        )

    ax.set_xlabel("Mean LLM-as-judge accuracy (0–1)")
    ax.set_ylabel(f"P99 latency — {metric_label} (seconds)")
    ax.set_title("Quality vs tail latency across baselines")
    ax.grid(True, alpha=0.35)
    # Optional log scale if extreme spread
    ymax = float(df[col].max())
    positive = df.loc[df[col] > 0, col]
    ymin = float(positive.min()) if not positive.empty else 0.0
    if ymax / max(ymin, 1e-9) > 50:
        ax.set_yscale("log")
        ax.set_ylabel(f"P99 latency — {metric_label} (seconds, log scale)")

    ax.set_xlim(-0.05, max(1.05, df["mean_llm_accuracy"].max() * 1.02))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="baseline", loc="best")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_data(defaults: dict[str, dict[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    search_parts: list[pd.DataFrame] = []
    add_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for baseline, paths in defaults.items():
        results_dir = paths["results"]
        if not results_dir.is_dir():
            print(f"[plot] WARN: missing results dir {results_dir}")

        latency_path = _find_search_latency_summary(results_dir)
        eval_path = results_dir / "evaluation_metrics.json"

        if latency_path:
            latency = _load_json(latency_path)
            search_parts.append(_search_latency_long_frame(baseline, latency))
        else:
            print(f"[plot] WARN: no search latency summary under {results_dir}")
            latency = {}

        llm_mean, n_qa = (0.0, 0)
        if eval_path.is_file():
            llm_mean, n_qa = _mean_llm_score(eval_path)
        else:
            print(f"[plot] WARN: missing {eval_path}")

        tl = latency.get("total_latency_sec") or {}
        sl = latency.get("search_latency_sec") or {}
        gl = latency.get("generation_latency_sec") or {}

        summary_rows.append(
            {
                "baseline": baseline,
                "mean_llm_accuracy": llm_mean,
                "n_qa": n_qa,
                "p99_total_sec": float(tl.get("p99", float("nan"))),
                "p99_search_sec": float(sl.get("p99", float("nan"))),
                "p99_gen_sec": float(gl.get("p99", float("nan"))),
            }
        )

        add_parts.append(_add_phase_rows(results_dir, baseline))

    search_df = pd.concat(search_parts, ignore_index=True) if search_parts else pd.DataFrame()
    add_df = pd.concat(add_parts, ignore_index=True) if add_parts else pd.DataFrame()
    return search_df, add_df, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LOCOMO baseline latency and LLM judge accuracy.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=LOCOMO_EVALS_ROOT / "results" / "plots",
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--p99-metric",
        choices=("total_latency_sec", "search_latency_sec", "generation_latency_sec"),
        default="total_latency_sec",
        help="Which search-phase p99 latency to plot on Y for the scatter.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)

    search_df, add_df, summary_rows = collect_data(DEFAULT_RESULTS)

    out_dir = args.out_dir
    plot_search_latency_bars(search_df, out_dir / "latency_qa_faceted.png")
    plot_flat_all_metrics(search_df, out_dir / "latency_qa_all_metrics_horizontal.png")

    dup = search_df.copy()
    dup["_dup"] = 1  # seaborn facets need unique grouping; simplified: skip redundant stacked sheet
    plot_add_latency_bars(add_df, out_dir / "latency_add_phase.png")
    plot_accuracy_vs_p99(summary_rows, out_dir / "llm_accuracy_vs_p99_latency.png", p99_metric=args.p99_metric)

    summ_path = out_dir / "baseline_latency_quality_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summ_path, index=False)
    print(f"[plot] Wrote plots under {out_dir}")
    print(f"[plot] Wrote summary table {summ_path}")


if __name__ == "__main__":
    main()
