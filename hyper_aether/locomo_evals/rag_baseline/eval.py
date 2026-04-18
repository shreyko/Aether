"""Evaluate RAG search-phase results with BLEU, F1, and LLM judge scoring."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .config import DEFAULT_LLM_CONCURRENCY
from .metrics import calculate_bleu_scores, calculate_f1, evaluate_llm_judge


def evaluate_results(
    input_file: str,
    output_file: str,
    max_workers: int = DEFAULT_LLM_CONCURRENCY,
) -> None:
    with open(input_file, "r") as f:
        data = json.load(f)

    # Flatten work so we can fan out every (conv, item) pair through the pool.
    # BLEU/F1 are cheap and computed inline in the worker alongside the LLM
    # judge call, since the judge is the only real latency sink.
    tasks: list[tuple[str, int, dict[str, object]]] = []
    for conv_key, items in data.items():
        for pos, item in enumerate(items):
            if str(item.get("category", "")) == "5":
                continue
            tasks.append((conv_key, pos, item))

    def _score_one(item: dict[str, object]) -> dict[str, object]:
        gt_answer = str(item["answer"])
        pred_answer = str(item["response"])
        category = str(item["category"])
        question = str(item["question"])

        bleu = calculate_bleu_scores(pred_answer, gt_answer)
        f1 = calculate_f1(pred_answer, gt_answer)
        llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

        return {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu["bleu1"],
            "f1_score": f1,
            "llm_score": llm_score,
        }

    # Preserve per-conversation ordering by dropping results into pre-sized slots.
    per_conv_len: dict[str, int] = defaultdict(int)
    for conv_key, pos, _ in tasks:
        per_conv_len[conv_key] = max(per_conv_len[conv_key], pos + 1)
    slots: dict[str, list[object | None]] = {
        k: [None] * n for k, n in per_conv_len.items()
    }

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
        future_to_meta = {
            pool.submit(_score_one, item): (conv_key, pos)
            for conv_key, pos, item in tasks
        }
        for future in tqdm(
            as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="LLM-judge eval",
        ):
            conv_key, pos = future_to_meta[future]
            try:
                slots[conv_key][pos] = future.result()
            except Exception as exc:
                slots[conv_key][pos] = {"error": f"{type(exc).__name__}: {exc}"}

    results: dict[str, list[dict[str, object]]] = {
        k: [it for it in v if it is not None] for k, v in slots.items()
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG baseline results")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluation_metrics.json")
    args = parser.parse_args()
    evaluate_results(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
