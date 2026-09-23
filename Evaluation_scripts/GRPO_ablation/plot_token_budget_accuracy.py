#!/usr/bin/env python3
"""Plot checkpoint accuracy as a function of inference token budget."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_evaluation(value: str) -> tuple[int, Path]:
    try:
        token_text, path_text = value.split("=", 1)
        tokens = int(token_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected TOKENS=EVALUATION_DIR") from exc
    if tokens < 1:
        raise argparse.ArgumentTypeError("Token budget must be positive")
    return tokens, Path(path_text)


def read_accuracy(path: Path) -> dict[str, float]:
    csv_path = path.expanduser().resolve() / "checkpoint_accuracy.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["checkpoint"]: float(row["accuracy_percent"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=parse_evaluation,
        required=True,
        help="Repeat TOKENS=EVALUATION_DIR for every evaluated token budget.",
    )
    parser.add_argument("--checkpoints", nargs="+", default=[
        "baseline", "checkpoint-1500", "checkpoint-2000",
        "checkpoint-2500", "checkpoint-3000",
    ])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="MapWise Test Accuracy vs. Inference Token Budget")
    args = parser.parse_args()

    evaluations = sorted(args.evaluation, key=lambda item: item[0])
    if len({tokens for tokens, _ in evaluations}) != len(evaluations):
        raise ValueError("Duplicate token budgets")
    accuracy_by_tokens = {tokens: read_accuracy(path) for tokens, path in evaluations}
    for tokens, values in accuracy_by_tokens.items():
        missing = set(args.checkpoints) - set(values)
        if missing:
            raise ValueError(f"Token budget {tokens} is missing checkpoints: {sorted(missing)}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token_budgets = [tokens for tokens, _ in evaluations]
    checkpoint_labels = [
        checkpoint.replace("checkpoint-", "step ") for checkpoint in args.checkpoints
    ]

    table_rows = []
    fig, axis = plt.subplots(figsize=(8, 5.5))
    for tokens in token_budgets:
        values = [accuracy_by_tokens[tokens][checkpoint] for checkpoint in args.checkpoints]
        axis.plot(checkpoint_labels, values, marker="o", linewidth=2.2,
                  label=f"{tokens} tokens")
        for checkpoint, accuracy in zip(args.checkpoints, values):
            table_rows.append({
                "checkpoint": checkpoint,
                "max_new_tokens": tokens,
                "accuracy_percent": accuracy,
            })

    axis.set_xlabel("Checkpoint")
    axis.set_ylabel("Acc (%)")
    axis.set_title(args.title)
    axis.grid(True, alpha=0.25)
    axis.legend(title="Maximum completion length")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_token_budget.png", dpi=220)
    fig.savefig(output_dir / "accuracy_vs_token_budget.pdf")
    plt.close(fig)

    with (output_dir / "accuracy_vs_token_budget.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
