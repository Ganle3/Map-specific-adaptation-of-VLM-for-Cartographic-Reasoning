#!/usr/bin/env python3
"""Summarize matched four-rollout groups and extract new-correct evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_sampling(path: Path) -> dict[int, dict[int, dict]]:
    grouped: dict[int, dict[int, dict]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig") as handle:
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
    parser.add_argument(
        "--checkpoint-only", action="store_true",
        help="Summarize one checkpoint without requiring or comparing a baseline.",
    )
    args = parser.parse_args()

    root = args.evaluation_dir.expanduser().resolve()
    labels = [args.checkpoint] if args.checkpoint_only else [args.baseline, args.checkpoint]
    model_rows = {
        label: read_sampling(root / label / "responses.jsonl") for label in labels
    }
    sample_ids = sorted(set().union(*(set(rows) for rows in model_rows.values())))
    group_rows = []
    paired_group_rows = []
    evidence = []
    rollout_transitions = Counter()
    group_transitions = Counter()

    for sample_index in sample_ids:
        seed_sets = [set(model_rows[label].get(sample_index, {})) for label in model_rows]
        if any(len(seeds) != args.expected_draws for seeds in seed_sets):
            raise ValueError(f"Sample {sample_index} does not have {args.expected_draws} draws per model")
        if len(seed_sets) == 2 and seed_sets[0] != seed_sets[1]:
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
                "truncated_rollouts": sum(
                    not bool(row.get("terminated", row.get("generation_status") == "complete"))
                    for row in rows
                ),
            })

        if not args.checkpoint_only:
            base_rows, base_rewards = per_model[args.baseline]
            ckpt_rows, ckpt_rewards = per_model[args.checkpoint]
            base_class = classify(base_rewards)
            ckpt_class = classify(ckpt_rewards)
            group_transitions[f"{base_class}_to_{ckpt_class}"] += 1
            paired_group_rows.append({
                "sample_index": sample_index,
                "qa_id": ckpt_rows[0].get("qa_id", ""),
                "answer_type": ckpt_rows[0].get("ground_truth_type", ""),
                "baseline_correct_rollouts": sum(base_rewards),
                "checkpoint_correct_rollouts": sum(ckpt_rewards),
                "correct_rollout_delta": sum(ckpt_rewards) - sum(base_rewards),
                "baseline_group_class": base_class,
                "checkpoint_group_class": ckpt_class,
                "baseline_truncated_rollouts": sum(
                    not bool(row.get("terminated", row.get("generation_status") == "complete"))
                    for row in base_rows
                ),
                "checkpoint_truncated_rollouts": sum(
                    not bool(row.get("terminated", row.get("generation_status") == "complete"))
                    for row in ckpt_rows
                ),
            })
            for base, ckpt, before, after in zip(base_rows, ckpt_rows, base_rewards, ckpt_rewards):
                transition = (
                    "correct_to_correct" if before and after else
                    "correct_to_wrong" if before else
                    "wrong_to_correct" if after else
                    "wrong_to_wrong"
                )
                rollout_transitions[transition] += 1
                if before == 0 and after == 1:
                    base_terminated = bool(base.get(
                        "terminated", base.get("generation_status") == "complete"
                    ))
                    checkpoint_terminated = bool(ckpt.get(
                        "terminated", ckpt.get("generation_status") == "complete"
                    ))
                    rollout_transitions[
                        "wrong_to_correct_from_terminated_baseline"
                        if base_terminated else "wrong_to_correct_from_truncated_baseline"
                    ] += 1
                    if base_terminated and checkpoint_terminated:
                        rollout_transitions["both_terminated_wrong_to_correct"] += 1
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
    if paired_group_rows:
        with (root / "paired_qa_group_transitions.csv").open(
                "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired_group_rows[0]))
            writer.writeheader(); writer.writerows(paired_group_rows)
    with (root / "new_correct_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for row in evidence:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (root / "diagnostic_summary.json").write_text(json.dumps({
        "matched_sampling": not args.checkpoint_only,
        "draws_per_question": args.expected_draws,
        "summary": summary_rows,
        "wrong_to_correct_rollouts": len(evidence),
        "rollout_transitions": dict(rollout_transitions),
        "qa_group_transitions": dict(group_transitions),
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
