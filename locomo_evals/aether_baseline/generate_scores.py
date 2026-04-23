"""Aggregate evaluation metrics by category for the Aether baseline."""

import argparse
import json

import pandas as pd


def generate_scores(input_file: str) -> None:
    with open(input_file, "r") as f:
        data = json.load(f)

    all_items = []
    for items in data.values():
        all_items.extend(items)

    df = pd.DataFrame(all_items)
    df["category"] = pd.to_numeric(df["category"])

    result = (
        df.groupby("category")
        .agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"})
        .round(4)
    )
    result["count"] = df.groupby("category").size()

    print("Mean Scores Per Category:")
    print(result)

    overall = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)
    print("\nOverall Mean Scores:")
    print(overall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aggregate Aether baseline scores")
    parser.add_argument("--input_file", type=str, default="evaluation_metrics.json")
    args = parser.parse_args()
    generate_scores(args.input_file)


if __name__ == "__main__":
    main()
