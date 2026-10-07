#!/usr/bin/env python3
"""Summarize four-rollout consistency from evaluate_grpo.py responses.jsonl.

Generation stays in the established GRPO evaluator.  This file intentionally
only scores/canonicalizes the saved responses and computes group metrics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--evaluation-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-draws", type=int, default=4)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rollout_exact_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_score(evaluator, row: Mapping[str, Any]) -> tuple[bool, Any]:
    """Use the same evaluator API selected by evaluate_grpo.py for this test."""
    if hasattr(evaluator, "evaluate_exact"):  # MapVerse
        # evaluate_grpo.py already persisted the reward obtained with its
        # established protocol.  For answer agreement, canonicalize the
        # extracted final answer rather than the whole reasoning trace.
        result = evaluator.evaluate_exact(
            row.get("final_answer") or row.get("raw_response", ""), row.get("ground_truth", ""),
            row.get("ground_truth_type", ""),
        )
        return bool(row.get("strict_exact_match", result["correct"])), result.get("prediction_canonical")
    result = evaluator.evaluate_sample({**row, "final_answer": ""})  # MapWise
    return bool(result["strict_exact_match"]), result.get("normalized_prediction")


def answer_key(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "<unparseable>"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def main() -> None:
    args = parse_args()
    if args.expected_draws != 4:
        raise ValueError("The consistency protocol is fixed at four rollouts per question.")
    source = args.responses_jsonl.expanduser().resolve()
    evaluator_path = args.evaluation_script.expanduser().resolve()
    if not source.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError("responses JSONL and evaluation script must both exist.")
    evaluator = load_module(evaluator_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("decoding") != "sampling":
            continue
        correct, canonical = canonical_score(evaluator, row)
        groups[str(row["qa_id"])].append({
            "rollout_index": row.get("sampling_seed"), "correct": correct,
            "canonical_answer": answer_key(canonical), "generated_tokens": row.get("generated_tokens"),
        })

    details = []
    for qa_id, rollouts in groups.items():
        if len(rollouts) != args.expected_draws:
            raise ValueError(f"{qa_id} has {len(rollouts)} sampled rollouts, expected {args.expected_draws}.")
        counts = Counter(r["canonical_answer"] for r in rollouts)
        entropy = -sum((n / len(rollouts)) * math.log2(n / len(rollouts)) for n in counts.values())
        details.append({"qa_id": qa_id, "rollouts": rollouts, "correct_count": sum(r["correct"] for r in rollouts),
                        "answer_counts": dict(sorted(counts.items())), "answer_consistency": max(counts.values()) / len(rollouts),
                        "answer_entropy_bits": entropy, "normalized_answer_entropy": entropy / math.log2(len(rollouts))})
    if not details:
        raise ValueError("No sampled rollouts found; evaluate with --sampling-draws 4 --skip-greedy.")
    total_rollouts = len(details) * args.expected_draws
    summary = {
        "questions": len(details), "rollouts_per_question": args.expected_draws,
        "rollout_accuracy": sum(d["correct_count"] for d in details) / total_rollouts,
        "pass_rate": sum(d["correct_count"] > 0 for d in details) / len(details),
        "all_rollouts_correct_rate": sum(d["correct_count"] == args.expected_draws for d in details) / len(details),
        "answer_consistency": sum(d["answer_consistency"] for d in details) / len(details),
        "answer_entropy_bits": sum(d["answer_entropy_bits"] for d in details) / len(details),
        "normalized_answer_entropy": sum(d["normalized_answer_entropy"] for d in details) / len(details),
        "responses_jsonl": str(source), "evaluation_script": str(evaluator_path),
    }
    output_dir = args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "question_metrics.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
