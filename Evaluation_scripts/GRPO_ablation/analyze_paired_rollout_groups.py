#!/usr/bin/env python3
"""Summarize matched four-rollout groups and extract new-correct evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_sampling(path: Path) -> dict[int, dict[int, dict]]:
    grouped: dict[int, dict[int, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("decoding") != "sampling":
                continue
            grouped[int(row["sample_index"])][int(row["sampling_seed"])] = row
    return dict(grouped)


def classify(rewards: list[int]) -> str:
    if sum(rewards) == 0:
        return "all_wrong"
    if sum(rewards) == len(rewards):
        return "all_correct"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--checkpoint", default="checkpoint-3000")
    parser.add_argument("--expected-draws", type=int, default=4)
    args = parser.parse_args()

    root = args.evaluation_dir.expanduser().resolve()
    model_rows = {
        args.baseline: read_sampling(root / args.baseline / "responses.jsonl"),
        args.checkpoint: read_sampling(root / args.checkpoint / "responses.jsonl"),
    }
    sample_ids = sorted(set(model_rows[args.baseline]) | set(model_rows[args.checkpoint]))
    group_rows = []
    evidence = []

    for sample_index in sample_ids:
        seed_sets = [set(model_rows[label].get(sample_index, {})) for label in model_rows]
        if any(len(seeds) != args.expected_draws for seeds in seed_sets):
            raise ValueError(f"Sample {sample_index} does not have {args.expected_draws} draws per model")
        if seed_sets[0] != seed_sets[1]:
            raise ValueError(f"Sample {sample_index} has unmatched sampling seeds")

        per_model = {}
        for label, grouped in model_rows.items():
            rows = [grouped[sample_index][seed] for seed in sorted(seed_sets[0])]
            rewards = [int(row.get("strict_exact_match", 0)) for row in rows]
            per_model[label] = (rows, rewards)
            group_rows.append({
                "model": label,
                "sample_index": sample_index,
                "qa_id": rows[0].get("qa_id", ""),
                "answer_type": rows[0].get("ground_truth_type", ""),
                "correct_rollouts": sum(rewards),
                "num_rollouts": len(rewards),
                "rollout_accuracy_percent": 100 * sum(rewards) / len(rewards),
                "group_class": classify(rewards),
            })

        base_rows, base_rewards = per_model[args.baseline]
        ckpt_rows, ckpt_rewards = per_model[args.checkpoint]
        for base, ckpt, before, after in zip(base_rows, ckpt_rows, base_rewards, ckpt_rewards):
            if before == 0 and after == 1:
                evidence.append({
                    "sample_index": sample_index,
                    "qa_id": ckpt.get("qa_id", ""),
                    "sampling_seed": ckpt["sampling_seed"],
                    "question": ckpt.get("question", ""),
                    "ground_truth": ckpt.get("ground_truth", ""),
                    "answer_type": ckpt.get("ground_truth_type", ""),
                    "resolved_image_path": ckpt.get("resolved_image_path", ""),
                    "baseline_final_answer": base.get("final_answer", ""),
                    "checkpoint_final_answer": ckpt.get("final_answer", ""),
                    "baseline_raw_response": base.get("raw_response", ""),
                    "checkpoint_raw_response": ckpt.get("raw_response", ""),
                    "baseline_terminated": base.get("terminated"),
                    "checkpoint_terminated": ckpt.get("terminated"),
                })

    summary_rows = []
    for label in model_rows:
        subset = [row for row in group_rows if row["model"] == label]
        counts = {kind: sum(row["group_class"] == kind for row in subset)
                  for kind in ("all_wrong", "mixed", "all_correct")}
        correct = sum(row["correct_rollouts"] for row in subset)
        total = sum(row["num_rollouts"] for row in subset)
        summary_rows.append({
            "model": label,
            "questions": len(subset),
            "rollouts": total,
            "correct_rollouts": correct,
            "rollout_accuracy_percent": 100 * correct / total,
            **{name: counts[name] for name in counts},
            "all_wrong_percent": 100 * counts["all_wrong"] / len(subset),
            "mixed_percent": 100 * counts["mixed"] / len(subset),
            "all_correct_percent": 100 * counts["all_correct"] / len(subset),
        })

    with (root / "rollout_group_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)
    with (root / "per_qa_rollout_groups.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader(); writer.writerows(group_rows)
    with (root / "new_correct_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "diagnostic_summary.json").write_text(json.dumps({
        "matched_sampling": True,
        "draws_per_question": args.expected_draws,
        "summary": summary_rows,
        "wrong_to_correct_rollouts": len(evidence),
        "evidence_file": "new_correct_evidence.jsonl",
        "interpretation_limit": (
            "New-correct records are candidates for qualitative image/reasoning review; "
            "their existence alone does not establish grounded transfer."
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrong-to-correct matched rollouts: {len(evidence)}")


if __name__ == "__main__":
    main()
