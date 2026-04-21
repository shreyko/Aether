"""CLI entry point for RAG baseline on LOCOMO."""

import argparse
import os

from .chunk_index import RAGIndexer
from .eval import evaluate_results
from .generate_scores import generate_scores
from .rag_search import RAGSearch

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASELINE_DIR, "results")
DATASET_PATH = os.path.join(os.path.dirname(BASELINE_DIR), "datasets", "locomo10.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG baseline on LOCOMO")
    parser.add_argument(
        "--method",
        choices=["index", "search", "eval", "scores", "all"],
        required=True,
        help="Pipeline phase to run",
    )
    parser.add_argument("--data_path", type=str, default=DATASET_PATH, help="Path to locomo10.json")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--top_k", type=int, default=2, help="Number of chunks to retrieve")
    parser.add_argument("--input_file", type=str, default=None, help="Input file for eval / scores phases")
    parser.add_argument("--output_dir", type=str, default=RESULTS_DIR, help="Directory for output files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "index":
        indexer = RAGIndexer(dataset_path=args.data_path, chunk_size=args.chunk_size)
        indexer.build()

    elif args.method == "search":
        output_file = os.path.join(
            args.output_dir,
            f"rag_search_results_chunksize{args.chunk_size}_top{args.top_k}.json",
        )
        searcher = RAGSearch(output_path=output_file, top_k=args.top_k, chunk_size=args.chunk_size)
        searcher.process_data_file(args.data_path)

    elif args.method == "eval":
        input_file = args.input_file
        if input_file is None:
            input_file = os.path.join(
                args.output_dir,
                f"rag_search_results_chunksize{args.chunk_size}_top{args.top_k}.json",
            )
        output_file = os.path.join(args.output_dir, "evaluation_metrics.json")
        evaluate_results(input_file, output_file)

    elif args.method == "scores":
        input_file = args.input_file or os.path.join(args.output_dir, "evaluation_metrics.json")
        generate_scores(input_file)

    elif args.method == "all":
        search_output_file = os.path.join(
            args.output_dir,
            f"rag_search_results_chunksize{args.chunk_size}_top{args.top_k}.json",
        )
        eval_output_file = os.path.join(args.output_dir, "evaluation_metrics.json")

        print("[ALL] Step 1/4: INDEX")
        indexer = RAGIndexer(dataset_path=args.data_path, chunk_size=args.chunk_size)
        indexer.build()

        print("[ALL] Step 2/4: SEARCH")
        searcher = RAGSearch(output_path=search_output_file, top_k=args.top_k, chunk_size=args.chunk_size)
        searcher.process_data_file(args.data_path)

        print("[ALL] Step 3/4: EVAL")
        evaluate_results(search_output_file, eval_output_file)

        print("[ALL] Step 4/4: SCORES")
        generate_scores(eval_output_file)


if __name__ == "__main__":
    main()
