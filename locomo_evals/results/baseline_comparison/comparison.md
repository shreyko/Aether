# LOCOMO baseline comparison

Aggregated from `evaluation_metrics.json` and `*latency_summary.json` under each baseline’s `results/`. Metrics are macro-averaged over QAs present in each file (category **5** is often omitted from eval outputs).

Sources: **rag**, **mem0**, **mem0g** (mem0 graph / Kuzu), **aether**.

---

## Overall scores

Mean BLEU-1 / F1 / LLM judge over all graded QAs in each `evaluation_metrics.json`.

| Baseline | BLEU | F1 | LLM judge | n_qa |
|----------|------:|-----:|----------:|------:|
| rag | 0.0600 | 0.0875 | 0.3656 | 1540 |
| mem0 | 0.0470 | 0.0743 | 0.3847 | 1518 |
| mem0g | 0.2709 | 0.3349 | **0.6877** | 1540 |
| aether | 0.0537 | 0.0863 | 0.5513 | 1540 |

---

## Scores by LOCOMO category

### BLEU (mean)

| Cat | Label | rag | mem0 | mem0g | aether |
|-----|-------|-----:|------:|-------:|-------:|
| 1 | Single-hop factual | 0.0745 | 0.0738 | 0.2586 | 0.0703 |
| 2 | Temporal | 0.0331 | 0.0219 | 0.2385 | 0.0346 |
| 3 | Multi-hop abstraction | 0.0306 | 0.0561 | 0.1370 | 0.0557 |
| 4 | Open-domain | 0.0688 | 0.0466 | **0.3026** | 0.0551 |

### F1 (mean)

| Cat | Label | rag | mem0 | mem0g | aether |
|-----|-------|-----:|------:|-------:|-------:|
| 1 | Single-hop factual | 0.0897 | 0.0878 | **0.3297** | 0.0826 |
| 2 | Temporal | 0.0358 | 0.0353 | **0.3075** | 0.0613 |
| 3 | Multi-hop abstraction | 0.0299 | **0.0884** | 0.1555 | 0.0830 |
| 4 | Open-domain | 0.1131 | 0.0830 | **0.3677** | 0.0976 |

### LLM judge (mean, 0–1)

| Cat | Label | rag | mem0 | mem0g | aether |
|-----|-------|-----:|------:|-------:|-------:|
| 1 | Single-hop factual | 0.4291 | 0.5771 | **0.8121** | 0.4716 |
| 2 | Temporal | 0.1807 | 0.1956 | **0.4330** | 0.4424 |
| 3 | Multi-hop abstraction | 0.2292 | 0.5684 | **0.5625** | 0.5833 |
| 4 | Open-domain | 0.4304 | 0.3712 | **0.7574** | 0.6159 |

Bold = best LLM judge in that category (ties broken by table order only).

### Counts (graded QAs per category in each metrics file)

| Cat | rag (n) | mem0 (n) | mem0g (n) | aether (n) |
|-----|---------|---------|----------|-----------|
| 1 | 282 | 279 | 282 | 282 |
| 2 | 321 | 317 | 321 | 321 |
| 3 | 96 | 95 | 96 | 96 |
| 4 | 841 | 827 | 841 | 841 |

---

## Latency — search phase (seconds)

Retrieval/embed path + answer generation totals in the search pipeline (`n_qa` ≈ dataset questions evaluated).

| Baseline | search p50 | search p95 | search p99 | gen p50 | gen p95 | gen p99 | total p50 | total p95 | total p99 | n_qa |
|---------|----------:|----------:|----------:|--------:|--------:|--------:|----------:|----------:|----------:|-----:|
| rag | 2.54 | 2.92 | 3.07 | 0.15 | 1.40 | 4.80 | 2.81 | 3.97 | 5.11 | 1986 |
| mem0 | 0.94 | 3.07 | 43.13 | 2.70 | 12.88 | 26.02 | 3.77 | 21.90 | 55.58 | 1986 |
| mem0g | 4.64 | 7.76 | 9.33 | 1.73 | 6.03 | 22.56 | 6.56 | 12.28 | 28.11 | 1986 |
| aether | 8.68 | 16.28 | 20.10 | 0.83 | 6.60 | 7.75 | 9.84 | 17.65 | 21.80 | 1986 |

RAG search config: BM25 chunks; **mem0**/**mem0g**/**aether** use retrieved-memory top‑k settings as in each baseline run (e.g. top‑30 where applicable).

---

## Latency — add / ingest phase (seconds)

Units differ by baseline tooling.

### RAG (per conversation index build)

Measured over **10** conversations.

| p50 | p95 | p99 |
|---:|---:|---:|
| 0.51 | 1.53 | 2.14 |

### mem0, mem0g, aether — per batch (`n` batches ≈ micro-batches written during ADD)

All three report **3142** batches in the summarized run.

| Baseline | p50 | p95 | p99 |
|----------|---:|---:|---:|
| mem0 | 14.69 | 35.68 | 45.88 |
| mem0g | 36.32 | 61.58 | 78.61 |
| aether | 19.40 | 33.66 | 42.05 |

---

## Regenerating

From the repo root:

```bash
PYTHONPATH=. python -m locomo_evals.compare_baselines_tables --html
```

Updates `scores_*.csv`, `latencies.csv`, `baseline_comparison_report.html`; refresh this markdown if you edit the script outputs.
