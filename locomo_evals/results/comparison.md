# LOCOMO Baseline Comparison

All three baselines share the same LOCOMO QA set, the same `ANSWER_PROMPT` at generation time, and the same `llm_as_a_judge` scoring step. Differences therefore isolate the **retrieval pipeline** (raw chunks vs. LLM-extracted sentences vs. hypergraph envelope).

Metrics:
- **BLEU**: standard n-gram overlap against the gold answer.
- **F1**: token-level F1 against the gold answer.
- **LLM accuracy**: 0/1 per question from the Llama-3 judge (`llm_as_a_judge.py`), averaged.

## Overall

| baseline | n | BLEU | F1 | LLM accuracy |
| --- | --- | --- | --- | --- |
| aether | 1540 | 0.0537 | 0.0863 | 0.6864 |
| mem0 | 1518 | 0.0470 | 0.0743 | 0.6344 |
| rag | 1540 | 0.0603 | 0.0852 | 0.3643 |

## Per category

| category | label | n (per baseline) | BLEU aether | BLEU mem0 | BLEU rag | F1 aether | F1 mem0 | F1 rag | LLM aether | LLM mem0 | LLM rag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | single-hop | 282/279/282 | 0.0703 | 0.0738 | 0.0690 | 0.0826 | 0.0878 | 0.0762 | 0.7305 | 0.7312 | 0.4326 |
| 2 | multi-hop | 321/317/321 | 0.0346 | 0.0219 | 0.0247 | 0.0613 | 0.0353 | 0.0242 | 0.7259 | 0.6782 | 0.2274 |
| 3 | temporal | 96/95/96 | 0.0557 | 0.0561 | 0.0304 | 0.0830 | 0.0884 | 0.0310 | 0.6250 | 0.8000 | 0.2500 |
| 4 | open-domain | 841/827/841 | 0.0551 | 0.0466 | 0.0744 | 0.0976 | 0.0830 | 0.1177 | 0.6635 | 0.5659 | 0.4067 |

## Winner by category (LLM accuracy)

| category | label | aether LLM | mem0 LLM | rag LLM | best |
| --- | --- | --- | --- | --- | --- |
| 1 | single-hop | 0.7305 | 0.7312 | 0.4326 | mem0 |
| 2 | multi-hop | 0.7259 | 0.6782 | 0.2274 | aether |
| 3 | temporal | 0.6250 | 0.8000 | 0.2500 | mem0 |
| 4 | open-domain | 0.6635 | 0.5659 | 0.4067 | aether |

## Per conversation (LLM accuracy)

| conv | aether | mem0 | rag | best |
| --- | --- | --- | --- | --- |
| 0 | 0.7763 | 0.6233 | 0.4474 | aether |
| 1 | 0.7160 | 0.7000 | 0.4568 | aether |
| 2 | 0.6579 | 0.7000 | 0.3158 | mem0 |
| 3 | 0.7186 | 0.6701 | 0.4070 | aether |
| 4 | 0.6742 | 0.6294 | 0.3427 | aether |
| 5 | 0.6911 | 0.6585 | 0.3415 | aether |
| 6 | 0.6800 | 0.5933 | 0.3733 | aether |
| 7 | 0.6230 | 0.5550 | 0.2775 | aether |
| 8 | 0.6410 | 0.5256 | 0.3590 | aether |
| 9 | 0.7089 | 0.7342 | 0.3734 | mem0 |
