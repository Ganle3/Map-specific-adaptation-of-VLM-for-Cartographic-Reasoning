#!/usr/bin/env python3
"""Re-score and compare MapVerse checkpoint evaluations across directories.

Each --input argument has the form LABEL=PATH_TO_RESPONSES_JSONL.  The first
label is the comparison baseline.  Scores are recomputed with the current
MapVerse evaluator instead of trusting historical strict_exact_match fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from mapverse_evaluation_exact import evaluate_exact


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(path)


def load(label: str, path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.expanduser().resolve().open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            # Standard evaluation contains one greedy row per QA.  Ignore any
            # optional sampled diagnostic rows if both are present.
            if row.get("decoding", "greedy") != "greedy":
                continue
            qa_id = str(row["qa_id"])
            if qa_id in rows:
                raise ValueError(f"Duplicate greedy qa_id={qa_id!r} for {label}")
            result = evaluate_exact(
                row.get("final_answer", row.get("raw_response", "")),
                row.get("ground_truth", ""),
                row.get("ground_truth_type", ""),
            )
            row["rescored_correct"] = int(result["correct"])
            row["stored_correct"] = int(bool(row.get("strict_exact_match", 0)))
            rows[qa_id] = row
    if not rows:
        raise ValueError(f"No greedy responses found for {label}: {path}")
    return rows


def terminated(row: dict) -> bool:
    return bool(row.get("terminated", row.get("generation_status") == "complete"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) < 2:
        parser.error("At least two --input LABEL=PATH arguments are required")

    labels = [label for label, _ in args.input]
    if len(labels) != len(set(labels)):
        parser.error("Input labels must be unique")
    records = {label: load(label, path) for label, path in args.input}
    baseline_label = labels[0]
    baseline = records[baseline_label]
    baseline_ids = set(baseline)
    for label, rows in records.items():
        if set(rows) != baseline_ids:
            missing = sorted(baseline_ids - set(rows))
            extra = sorted(set(rows) - baseline_ids)
            raise ValueError(
                f"QA IDs differ for {label}: missing={len(missing)}, extra={len(extra)}"
            )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for label in labels:
        rows = list(records[label].values())
        correct = sum(row["rescored_correct"] for row in rows)
        complete = sum(terminated(row) for row in rows)
        metric_rows.append({
            "checkpoint": label,
            "total": len(rows),
            "correct": correct,
            "accuracy_percent": 100 * correct / len(rows),
            "terminated": complete,
            "truncated": len(rows) - complete,
            "truncated_percent": 100 * (len(rows) - complete) / len(rows),
            "stored_to_rescored_changes": sum(
                row["stored_correct"] != row["rescored_correct"] for row in rows
            ),
        })

    transition_rows: list[dict] = []
    evidence_rows: list[dict] = []
    for label in labels[1:]:
        target = records[label]
        counts: Counter[str] = Counter()
        for qa_id in sorted(baseline_ids):
            before, after = baseline[qa_id], target[qa_id]
            bc, ac = bool(before["rescored_correct"]), bool(after["rescored_correct"])
            bt, at = terminated(before), terminated(after)
            transition = (
                "correct_to_correct" if bc and ac else
                "correct_to_wrong" if bc else
                "wrong_to_correct" if ac else "wrong_to_wrong"
            )
            counts[transition] += 1
            source = ""
            if transition == "wrong_to_correct":
                if bt and at:
                    source = "stable_completed_wrong_to_correct"
                elif not bt and at:
                    source = "truncation_recovery"
                else:
                    source = "other_wrong_to_correct"
                counts[source] += 1
                evidence_rows.append({
                    "comparison": f"{baseline_label}_vs_{label}",
                    "gain_source": source,
                    "qa_id": qa_id,
                    "answer_type": after.get("ground_truth_type", ""),
                    "question": after.get("question", ""),
                    "ground_truth": after.get("ground_truth", ""),
                    "baseline_terminated": int(bt),
                    "target_terminated": int(at),
                    "baseline_answer": before.get("final_answer", ""),
                    "target_answer": after.get("final_answer", ""),
                    "baseline_response": before.get("raw_response", ""),
                    "target_response": after.get("raw_response", ""),
                })
        transition_rows.append({
            "baseline": baseline_label,
            "checkpoint": label,
            "wrong_to_correct": counts["wrong_to_correct"],
            "correct_to_wrong": counts["correct_to_wrong"],
            "net_correct_gain": counts["wrong_to_correct"] - counts["correct_to_wrong"],
            "stable_completed_wrong_to_correct": counts[
                "stable_completed_wrong_to_correct"
            ],
            "truncation_recovery": counts["truncation_recovery"],
            "other_wrong_to_correct": counts["other_wrong_to_correct"],
            "correct_to_correct": counts["correct_to_correct"],
            "wrong_to_wrong": counts["wrong_to_wrong"],
        })

    write_csv(output / "checkpoint_metrics_rescored.csv", metric_rows)
    write_csv(output / "transition_summary_rescored.csv", transition_rows)
    write_csv(output / "wrong_to_correct_evidence_rescored.csv", evidence_rows)
    summary = {
        "evaluator": str(Path(__file__).with_name("mapverse_evaluation_exact.py")),
        "baseline": baseline_label,
        "metrics": metric_rows,
        "transitions": transition_rows,
    }
    (output / "analysis_summary_rescored.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
