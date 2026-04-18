"""
Evaluate search-phase results: compute BLEU, F1, and LLM judge scores.

Follows the structure of
https://github.com/mem0ai/mem0/blob/main/evaluation/evals.py
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .metrics import calculate_bleu_scores, calculate_f1, evaluate_llm_judge


def _score_item(conv_key: str, item: dict) -> tuple[str, dict] | None:
    """Compute BLEU/F1/LLM-judge for a single QA item. Returns (conv_key, row) or None if skipped."""
    category = str(item["category"])
    if category == "5":
        return None

    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
    question = str(item["question"])

    bleu = calculate_bleu_scores(pred_answer, gt_answer)
    f1 = calculate_f1(pred_answer, gt_answer)
    llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

    return conv_key, {
        "question": question,
        "answer": gt_answer,
        "response": pred_answer,
        "category": category,
        "bleu_score": bleu["bleu1"],
        "f1_score": f1,
        "llm_score": llm_score,
    }


def evaluate_results(input_file: str, output_file: str, max_workers: int | None = None):
    with open(input_file, "r") as f:
        data = json.load(f)

    if max_workers is None:
        max_workers = int(os.getenv("MEM0_EVAL_WORKERS", "32"))

    # Flatten into (conv_key, item) pairs so we can parallelize the LLM-judge
    # calls across the whole dataset, not just within a single conversation.
    jobs: list[tuple[str, dict]] = []
    for conv_key, items in data.items():
        for item in items:
            jobs.append((conv_key, item))

    results: dict[str, list] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_score_item, ck, it) for ck, it in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            out = fut.result()
            if out is None:
                continue
            conv_key, row = out
            results[conv_key].append(row)

    # Deterministic ordering: sort each conversation's rows by question text so
    # repeated runs produce diff-friendly output despite the unordered futures.
    for conv_key in results:
        results[conv_key].sort(key=lambda r: r["question"])

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate mem0 baseline results")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluation_metrics.json")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Concurrent LLM-judge calls (default: $MEM0_EVAL_WORKERS or 32).",
    )
    args = parser.parse_args()

    evaluate_results(args.input_file, args.output_file, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
