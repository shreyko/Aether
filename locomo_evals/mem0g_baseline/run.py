"""
CLI entry point for the **mem0 graph (Kuzu)** baseline on LOCOMO.

Requires ``mem0ai[graph]>=1.0.0,<2.0.0`` (mem0ai 2.x removed external graph stores).
Install: ``uv sync --extra mem0g-baseline`` (see ``pyproject.toml``).

Usage mirrors ``locomo_evals.mem0_baseline.run``; outputs use the ``mem0g_`` prefix.
"""

import argparse
import os

from ..mem0_baseline.eval import evaluate_results
from ..mem0_baseline.generate_scores import generate_scores

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASELINE_DIR, "results")
DATASET_PATH = os.path.join(os.path.dirname(BASELINE_DIR), "datasets", "locomo10.json")


def main():
    parser = argparse.ArgumentParser(description="mem0g (Kuzu graph) baseline on LOCOMO")
    parser.add_argument(
        "--method",
        choices=["add", "search", "eval", "scores", "all"],
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

        MemoryADD(data_path=args.data_path).process_all_conversations()

    elif args.method == "search":
        from .mem0_search import MemorySearch

        output_file = os.path.join(args.output_dir, f"mem0g_search_results_top{args.top_k}.json")
        MemorySearch(output_path=output_file, top_k=args.top_k).process_data_file(args.data_path)

    elif args.method == "eval":
        input_file = args.input_file
        if input_file is None:
            input_file = os.path.join(args.output_dir, f"mem0g_search_results_top{args.top_k}.json")
        output_file = os.path.join(args.output_dir, "evaluation_metrics.json")
        evaluate_results(input_file, output_file)

    elif args.method == "scores":
        input_file = args.input_file or os.path.join(args.output_dir, "evaluation_metrics.json")
        generate_scores(input_file)

    elif args.method == "all":
        from .mem0_add import MemoryADD
        from .mem0_search import MemorySearch

        search_output_file = os.path.join(args.output_dir, f"mem0g_search_results_top{args.top_k}.json")
        eval_output_file = os.path.join(args.output_dir, "evaluation_metrics.json")

        print("[mem0g ALL] Step 1/4: ADD")
        MemoryADD(data_path=args.data_path).process_all_conversations()

        print("[mem0g ALL] Step 2/4: SEARCH")
        MemorySearch(output_path=search_output_file, top_k=args.top_k).process_data_file(args.data_path)

        print("[mem0g ALL] Step 3/4: EVAL")
        evaluate_results(search_output_file, eval_output_file)

        print("[mem0g ALL] Step 4/4: SCORES")
        generate_scores(eval_output_file)


if __name__ == "__main__":
    main()
