#!/usr/bin/env python3
"""Compute binary Yes/No metrics from checkpoint evaluation responses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_binary(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [row for row in rows if row.get("ground_truth_type") == "Binary"]


def metrics(rows: list[dict]) -> dict:
    counts = {
        "yes_yes": 0, "yes_no": 0, "yes_abstain": 0,
        "no_yes": 0, "no_no": 0, "no_abstain": 0,
    }
    truncated = 0
    for row in rows:
        gold = str(row.get("normalized_ground_truth") or row.get("ground_truth", "")).strip().lower()
        prediction = str(
            row.get("normalized_prediction")
            or row.get("evaluated_answer")
            or row.get("final_answer", "")
        ).strip().lower()
        if gold not in {"yes", "no"}:
            raise ValueError(f"Unsupported normalized Binary ground truth: {gold!r}")
        predicted_class = prediction if prediction in {"yes", "no"} else "abstain"
        counts[f"{gold}_{predicted_class}"] += 1
        truncated += not bool(row.get("terminated", row.get("generation_status") == "complete"))

    total = len(rows)
    predicted = total - counts["yes_abstain"] - counts["no_abstain"]
    correct = counts["yes_yes"] + counts["no_no"]

    yes_precision = divide(counts["yes_yes"], counts["yes_yes"] + counts["no_yes"])
    yes_recall = divide(
        counts["yes_yes"], counts["yes_yes"] + counts["yes_no"] + counts["yes_abstain"]
    )
    no_precision = divide(counts["no_no"], counts["no_no"] + counts["yes_no"])
    no_recall = divide(
        counts["no_no"], counts["no_no"] + counts["no_yes"] + counts["no_abstain"]
    )
    yes_f1 = f1(yes_precision, yes_recall)
    no_f1 = f1(no_precision, no_recall)

    return {
        "binary_total": total,
        "gold_yes": counts["yes_yes"] + counts["yes_no"] + counts["yes_abstain"],
        "gold_no": counts["no_yes"] + counts["no_no"] + counts["no_abstain"],
        "tp_yes": counts["yes_yes"],
        "fp_yes": counts["no_yes"],
        "fn_yes_as_no": counts["yes_no"],
        "tn_no": counts["no_no"],
        "abstain_gold_yes": counts["yes_abstain"],
        "abstain_gold_no": counts["no_abstain"],
        "coverage_percent": 100 * divide(predicted, total),
        "strict_accuracy_percent": 100 * divide(correct, total),
        "accuracy_on_parseable_percent": 100 * divide(correct, predicted),
        "truncated_percent": 100 * divide(truncated, total),
        "yes_precision_percent": 100 * yes_precision,
        "yes_recall_percent": 100 * yes_recall,
        "yes_f1_percent": 100 * yes_f1,
        "no_precision_percent": 100 * no_precision,
        "no_recall_percent": 100 * no_recall,
        "no_f1_percent": 100 * no_f1,
        "macro_precision_percent": 50 * (yes_precision + no_precision),
        "macro_recall_percent": 50 * (yes_recall + no_recall),
        "macro_f1_percent": 50 * (yes_f1 + no_f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    output = args.output or evaluation_dir / "binary_classification_metrics.csv"
    results = []
    for checkpoint in args.checkpoints:
        response_path = evaluation_dir / checkpoint / "responses.jsonl"
        results.append({"checkpoint": checkpoint, **metrics(load_binary(response_path))})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
