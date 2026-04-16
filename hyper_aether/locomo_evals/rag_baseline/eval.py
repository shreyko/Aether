"""Evaluate RAG search-phase results with BLEU, F1, and LLM judge scoring."""

import argparse
import json
from collections import defaultdict

from tqdm import tqdm

from .metrics import calculate_bleu_scores, calculate_f1, evaluate_llm_judge


def evaluate_results(input_file: str, output_file: str) -> None:
    with open(input_file, "r") as f:
        data = json.load(f)

    results: dict[str, list[dict[str, object]]] = defaultdict(list)

    for conv_key, items in tqdm(data.items(), desc="Evaluating conversations"):
        for item in items:
            gt_answer = str(item["answer"])
            pred_answer = str(item["response"])
            category = str(item["category"])
            question = str(item["question"])

            if category == "5":
                continue

            bleu = calculate_bleu_scores(pred_answer, gt_answer)
            f1 = calculate_f1(pred_answer, gt_answer)
            llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

            results[conv_key].append(
                {
                    "question": question,
                    "answer": gt_answer,
                    "response": pred_answer,
                    "category": category,
                    "bleu_score": bleu["bleu1"],
                    "f1_score": f1,
                    "llm_score": llm_score,
                }
            )

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
