#!/usr/bin/env python3
"""Decompose matched MapWise-to-MapVerse rollout gains by truncation and QA."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> dict[tuple[int, int], dict]:
    rows = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("decoding") == "sampling":
                rows[(int(row["sample_index"]), int(row["sampling_seed"]))] = row
    return rows


def correct(row: dict) -> bool:
    return bool(row.get("strict_exact_match", 0))


def terminated(row: dict) -> bool:
    return bool(row.get("terminated", row.get("generation_status") == "complete"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--checkpoint", default="checkpoint-3000")
    args = parser.parse_args()
    root = args.evaluation_dir.resolve()
    baseline = load(root / args.baseline / "responses.jsonl")
    checkpoint = load(root / args.checkpoint / "responses.jsonl")
    if set(baseline) != set(checkpoint):
        raise ValueError("Baseline and checkpoint sampling keys differ")

    rollout_rows = []
    per_type: dict[str, Counter] = defaultdict(Counter)
    per_qa: dict[int, list[dict]] = defaultdict(list)
    for key in sorted(baseline):
        before, after = baseline[key], checkpoint[key]
        bc, ac = correct(before), correct(after)
        bt, at = terminated(before), terminated(after)
        transition = ("correct_to_correct" if bc and ac else
                      "correct_to_wrong" if bc else
                      "wrong_to_correct" if ac else "wrong_to_wrong")
        source = ""
        if transition == "wrong_to_correct":
            source = "stable_terminated_correction" if bt and at else (
                "truncation_recovery" if not bt and at else "other_wrong_to_correct"
            )
        answer_type = str(after.get("ground_truth_type", ""))
        row = {
            "sample_index": key[0], "sampling_seed": key[1],
            "qa_id": after.get("qa_id", ""), "answer_type": answer_type,
            "transition": transition, "gain_source": source,
            "baseline_terminated": int(bt), "checkpoint_terminated": int(at),
            "baseline_generated_tokens": before.get("generated_tokens", ""),
            "checkpoint_generated_tokens": after.get("generated_tokens", ""),
            "question": after.get("question", ""),
            "ground_truth": after.get("ground_truth", ""),
            "baseline_final_answer": before.get("final_answer", ""),
            "checkpoint_final_answer": after.get("final_answer", ""),
        }
        rollout_rows.append(row); per_qa[key[0]].append(row)
        per_type[answer_type][transition] += 1
        if source: per_type[answer_type][source] += 1

    candidates = []
    for sample_index, rows in sorted(per_qa.items()):
        counts = Counter(row["transition"] for row in rows)
        sources = Counter(row["gain_source"] for row in rows if row["gain_source"])
        first = rows[0]
        candidates.append({
            "sample_index": sample_index, "qa_id": first["qa_id"],
            "answer_type": first["answer_type"], "question": first["question"],
            "ground_truth": first["ground_truth"],
            "baseline_correct_rollouts": counts["correct_to_correct"] + counts["correct_to_wrong"],
            "checkpoint_correct_rollouts": counts["correct_to_correct"] + counts["wrong_to_correct"],
            "net_correct_delta": counts["wrong_to_correct"] - counts["correct_to_wrong"],
            "stable_terminated_corrections": sources["stable_terminated_correction"],
            "truncation_recoveries": sources["truncation_recovery"],
            "correct_to_wrong": counts["correct_to_wrong"],
        })
    candidates.sort(key=lambda row: (
        -row["stable_terminated_corrections"], -row["net_correct_delta"], row["sample_index"]
    ))

    type_rows = []
    transition_fields = (
        "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong",
        "stable_terminated_correction", "truncation_recovery", "other_wrong_to_correct",
    )
    for answer_type, counts in sorted(per_type.items()):
        total = sum(counts[name] for name in (
            "correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"
        ))
        type_rows.append({
            "answer_type": answer_type, "rollouts": total,
            **{name: counts[name] for name in transition_fields},
        })

    for name, rows in (
        ("rollout_transition_sources.csv", rollout_rows),
        ("transition_summary_by_answer_type.csv", type_rows),
        ("strong_transfer_candidates.csv", candidates),
    ):
        with (root / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(root / name)


if __name__ == "__main__":
    main()
