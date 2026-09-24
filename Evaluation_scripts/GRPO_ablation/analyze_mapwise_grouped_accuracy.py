#!/usr/bin/env python3
"""Summarize MapWise accuracy by ability level and answer type."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.expanduser().resolve()
    qa_rows = json.loads(args.qa_json.expanduser().read_text(encoding="utf-8-sig"))
    output = args.output or evaluation_dir / "grouped_accuracy_baseline_checkpoint2500.csv"
    results: list[dict] = []

    for checkpoint in args.checkpoints:
        response_path = evaluation_dir / checkpoint / "responses.jsonl"
        rows = read_jsonl(response_path)
        for row in rows:
            index = int(row["sample_index"])
            if index < 0 or index >= len(qa_rows):
                raise IndexError(f"Invalid sample_index {index} in {response_path}")
            meta = qa_rows[index]
            row["ability_level"] = meta["ability_level"]
            row["answer_type"] = row.get("ground_truth_type") or meta["ground_truth_type"]

        for field, theme in (("ability_level", "ability_level"), ("answer_type", "answer_type")):
            for category in sorted({str(row[field]) for row in rows}):
                selected = [row for row in rows if str(row[field]) == category]
                correct = sum(int(row.get("strict_exact_match", 0)) for row in selected)
                results.append(
                    {
                        "checkpoint": checkpoint,
                        "theme": theme,
                        "category": category,
                        "total": len(selected),
                        "correct": correct,
                        "incorrect": len(selected) - correct,
                        "accuracy_percent": 100 * correct / len(selected),
                    }
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(output.resolve())


if __name__ == "__main__":
    main()
