# RAG Baseline for LOCOMO

This folder contains the retrieval-augmented generation (RAG) baseline implementation for the LOCOMO benchmark.

## What it does
- indexes full conversation sessions as fixed-size token chunks
- embeds chunks with a sentence-transformers model
- stores chunk vectors in Chroma
- retrieves top-k chunks for each question
- generates answers via a local vLLM server using an OpenAI-compatible client
- evaluates answers with BLEU, token F1, and an LLM judge

## Files
- `config.py` — baseline configuration, vLLM client helper, dataset location
- `chunk_index.py` — build the chunk index and persist it in Chroma
- `rag_search.py` — retrieve top-k chunks and generate answers
- `metrics.py` — BLEU, F1, and LLM judge scoring
- `eval.py` — evaluate generated answers
- `generate_scores.py` — aggregate scores by category
- `run.py` — CLI wrapper for baseline phases
- `run_all_with_vllm.sh` — end-to-end runner with vLLM startup/shutdown

## Usage
Run the RAG baseline end-to-end:

```bash
cd hyper_aether
bash locomo_evals/rag_baseline/run_all_with_vllm.sh
```

Run individual phases:

```bash
cd hyper_aether
python -m locomo_evals.rag_baseline.run --method index --chunk_size 512
python -m locomo_evals.rag_baseline.run --method search --chunk_size 512 --top_k 2
python -m locomo_evals.rag_baseline.run --method eval --input_file hyper_aether/locomo_evals/rag_baseline/results/rag_search_results_chunksize512_top2.json
python -m locomo_evals.rag_baseline.run --method scores --input_file hyper_aether/locomo_evals/rag_baseline/results/evaluation_metrics.json
```

## Notes
- The shell driver only runs the RAG baseline, not `mem0_baseline`.
- Use `TOP_K=1` or `TOP_K=2` for the paper-style comparison.
- `CHUNK_SIZE` is configurable; the default is `512` tokens.

## Requirements
This baseline assumes the same optional dependencies as the mem0 baseline:
- `openai`
- `chromadb`
- `sentence-transformers`
- `nltk`
- `pandas`
- `tqdm`

Make sure the vLLM service is available at `http://localhost:8000/v1` or configure `VLLM_BASE_URL`.
