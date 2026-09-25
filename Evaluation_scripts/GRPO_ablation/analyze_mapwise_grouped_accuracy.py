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

    if len(args.checkpoints) == 2:
        baseline_name, target_name = args.checkpoints
        baseline_rows = {int(row["sample_index"]): row for row in read_jsonl(
            evaluation_dir / baseline_name / "responses.jsonl"
        )}
        target_rows = {int(row["sample_index"]): row for row in read_jsonl(
            evaluation_dir / target_name / "responses.jsonl"
        )}
        if set(baseline_rows) != set(target_rows):
            raise ValueError("Baseline and target do not contain identical sample indices")

        transition_rows = []
        for level in sorted({str(qa_rows[index]["ability_level"]) for index in baseline_rows}):
            indices = [index for index in baseline_rows
                       if str(qa_rows[index]["ability_level"]) == level]
            before_correct = lambda index: bool(baseline_rows[index].get("strict_exact_match", 0))
            after_correct = lambda index: bool(target_rows[index].get("strict_exact_match", 0))
            before_terminated = lambda index: bool(baseline_rows[index].get(
                "terminated", baseline_rows[index].get("generation_status") == "complete"
            ))
            after_terminated = lambda index: bool(target_rows[index].get(
                "terminated", target_rows[index].get("generation_status") == "complete"
            ))
            wrong_to_correct = [index for index in indices
                                if not before_correct(index) and after_correct(index)]
            transition_rows.append({
                "ability_level": level,
                "total": len(indices),
                "baseline_correct": sum(before_correct(index) for index in indices),
                "target_correct": sum(after_correct(index) for index in indices),
                "wrong_to_correct": len(wrong_to_correct),
                "wrong_to_correct_from_truncated_baseline": sum(
                    not before_terminated(index) for index in wrong_to_correct
                ),
                "both_terminated_wrong_to_correct": sum(
                    before_terminated(index) and after_terminated(index)
                    for index in wrong_to_correct
                ),
                "correct_to_wrong": sum(
                    before_correct(index) and not after_correct(index) for index in indices
                ),
                "baseline_truncated": sum(not before_terminated(index) for index in indices),
                "target_truncated": sum(not after_terminated(index) for index in indices),
            })
        transition_output = output.with_name(
            output.stem + "_transitions_by_ability.csv"
        )
        with transition_output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0]))
            writer.writeheader()
            writer.writerows(transition_rows)
        print(transition_output.resolve())


if __name__ == "__main__":
    main()
