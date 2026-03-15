"""
CLI entry point for the mem0 baseline on LOCOMO.

Usage:
    # ADD phase  – ingest conversations into mem0
    python -m locomo_evals.mem0_baseline.run --method add

    # SEARCH phase – retrieve memories and generate answers
    python -m locomo_evals.mem0_baseline.run --method search --top_k 30

    # EVAL phase – compute BLEU / F1 / LLM-judge scores
    python -m locomo_evals.mem0_baseline.run --method eval \
        --input_file locomo_evals/mem0_baseline/results/mem0_search_results.json

    # SCORES phase – aggregate scores by category
    python -m locomo_evals.mem0_baseline.run --method scores \
        --input_file locomo_evals/mem0_baseline/results/evaluation_metrics.json
"""

import argparse
import os

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASELINE_DIR, "results")
DATASET_PATH = os.path.join(os.path.dirname(BASELINE_DIR), "datasets", "locomo10.json")


def main():
    parser = argparse.ArgumentParser(description="mem0 baseline on LOCOMO")
    parser.add_argument(
        "--method",
        choices=["add", "search", "eval", "scores"],
        required=True,
        help="Pipeline phase to run",
    )
    parser.add_argument("--data_path", type=str, default=DATASET_PATH, help="Path to locomo10.json")
    parser.add_argument("--top_k", type=int, default=30, help="Number of memories to retrieve")
    parser.add_argument("--input_file", type=str, default=None, help="Input file for eval / scores phases")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR, help="Directory for output files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "add":
        from .mem0_add import MemoryADD

        manager = MemoryADD(data_path=args.data_path)
        manager.process_all_conversations()

    elif args.method == "search":
        from .mem0_search import MemorySearch

        output_file = os.path.join(args.output_dir, f"mem0_search_results_top{args.top_k}.json")
        searcher = MemorySearch(output_path=output_file, top_k=args.top_k)
        searcher.process_data_file(args.data_path)

    elif args.method == "eval":
        from .eval import evaluate_results

        input_file = args.input_file
        if input_file is None:
            input_file = os.path.join(args.output_dir, f"mem0_search_results_top{args.top_k}.json")
        output_file = os.path.join(args.output_dir, "evaluation_metrics.json")
        evaluate_results(input_file, output_file)

    elif args.method == "scores":
        from .generate_scores import generate_scores

        input_file = args.input_file
        if input_file is None:
            input_file = os.path.join(args.output_dir, "evaluation_metrics.json")
        generate_scores(input_file)


if __name__ == "__main__":
    main()
