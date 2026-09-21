#!/usr/bin/env python3
"""Compare greedy checkpoint responses, separating truncation recovery from learned corrections."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_correct(row: dict) -> bool:
    return bool(row.get("strict_exact_match", row.get("primary_score", 0)))


def is_terminated(row: dict) -> bool:
    return bool(row.get("terminated", row.get("generation_status") == "complete"))


def checkpoint_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(is_correct(row) for row in rows)
    terminated = [row for row in rows if is_terminated(row)]
    truncated = total - len(terminated)
    return {
        "total": total,
        "correct": correct,
        "accuracy_percent": 100 * correct / total,
        "terminated": len(terminated),
        "truncated": truncated,
        "truncated_percent": 100 * truncated / total,
        "correct_among_terminated": sum(is_correct(row) for row in terminated),
        "accuracy_among_terminated_percent": (
            100 * sum(is_correct(row) for row in terminated) / len(terminated)
            if terminated else None
        ),
        "mean_generated_tokens": mean(int(row.get("generated_tokens", 0)) for row in rows),
        "mean_generated_tokens_terminated": (
            mean(int(row.get("generated_tokens", 0)) for row in terminated)
            if terminated else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--checkpoints", nargs="+", default=["baseline", "checkpoint-2000", "checkpoint-3000"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    output_dir = (args.output_dir or evaluation_dir / "reasoning_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict[str, dict]] = {}
    for checkpoint in args.checkpoints:
        path = evaluation_dir / checkpoint / "responses.jsonl"
        rows = read_jsonl(path)
        indexed = {str(row["qa_id"]): row for row in rows}
        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate qa_id values in {path}")
        records[checkpoint] = indexed

    common_ids = set.intersection(*(set(rows) for rows in records.values()))
    if any(len(rows) != len(common_ids) for rows in records.values()):
        raise ValueError("Checkpoint response files do not contain the same QA IDs")

    baseline_name = args.checkpoints[0]
    baseline = records[baseline_name]
    summary = {
        "checkpoints": {
            checkpoint: checkpoint_metrics(list(records[checkpoint].values()))
            for checkpoint in args.checkpoints
        },
        "comparisons": {},
    }

    comparison_rows: list[dict] = []
    raw_path = output_dir / "per_qa_reasoning_comparison.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_handle:
        for qa_id in sorted(common_ids):
            base = baseline[qa_id]
            flat = {
                "qa_id": qa_id,
                "country": base.get("country", ""),
                "template_no": base.get("template_no", ""),
                "ground_truth_type": base.get("ground_truth_type", ""),
                "question": base.get("question", ""),
                "ground_truth": base.get("ground_truth", ""),
            }
            raw = {key: flat[key] for key in flat}
            for checkpoint in args.checkpoints:
                row = records[checkpoint][qa_id]
                prefix = checkpoint.replace("checkpoint-", "step_").replace("-", "_")
                flat[f"{prefix}_correct"] = int(is_correct(row))
                flat[f"{prefix}_terminated"] = int(is_terminated(row))
                flat[f"{prefix}_generated_tokens"] = int(row.get("generated_tokens", 0))
                flat[f"{prefix}_final_answer"] = row.get("final_answer", "")
                raw[checkpoint] = {
                    "correct": is_correct(row),
                    "terminated": is_terminated(row),
                    "generated_tokens": int(row.get("generated_tokens", 0)),
                    "final_answer": row.get("final_answer", ""),
                    "raw_response": row.get("raw_response", ""),
                    "evaluation_note": row.get("evaluation_note", ""),
                }
            comparison_rows.append(flat)
            raw_handle.write(json.dumps(raw, ensure_ascii=False) + "\n")

    for checkpoint in args.checkpoints[1:]:
        target = records[checkpoint]
        transitions = Counter()
        grouped = defaultdict(Counter)
        for qa_id in common_ids:
            before = baseline[qa_id]
            after = target[qa_id]
            before_correct = is_correct(before)
            after_correct = is_correct(after)
            before_term = is_terminated(before)
            after_term = is_terminated(after)
            accuracy_transition = (
                "correct_to_correct" if before_correct and after_correct else
                "correct_to_wrong" if before_correct else
                "wrong_to_correct" if after_correct else
                "wrong_to_wrong"
            )
            termination_transition = (
                f"{'terminated' if before_term else 'truncated'}_to_"
                f"{'terminated' if after_term else 'truncated'}"
            )
            transitions[accuracy_transition] += 1
            transitions[termination_transition] += 1
            if not before_correct and after_correct:
                transitions[
                    "wrong_to_correct_from_terminated_baseline"
                    if before_term else "wrong_to_correct_from_truncated_baseline"
                ] += 1
            if before_term and after_term:
                transitions[f"both_terminated_{accuracy_transition}"] += 1
            for field in ("country", "template_no", "ground_truth_type"):
                grouped[field, str(before.get(field, ""))][accuracy_transition] += 1

        summary["comparisons"][f"{baseline_name}_vs_{checkpoint}"] = {
            "transitions": dict(transitions),
            "net_correct_gain": transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
            "strong_learning_candidates": transitions["both_terminated_wrong_to_correct"],
            "truncation_recovery_candidates": transitions["wrong_to_correct_from_truncated_baseline"],
            "grouped_accuracy_transitions": {
                f"{field}={value}": dict(counts)
                for (field, value), counts in sorted(grouped.items())
            },
        }

    (output_dir / "reasoning_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "per_qa_transition.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
