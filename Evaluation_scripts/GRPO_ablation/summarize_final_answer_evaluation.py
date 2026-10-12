#!/usr/bin/env python3
"""Summarize greedy evaluation responses, including answerability and latency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ABILITY_LEVELS = tuple(f"L{level}" for level in range(6))


def nonempty_final_answer(row: dict[str, Any]) -> bool:
    return bool(str(row.get("final_answer", "") or "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    response_path = args.responses_jsonl.expanduser().resolve()
    rows = [json.loads(line) for line in response_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"No responses found in {response_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("ability_level", "") or "Unknown")].append(row)

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        extractable = [row for row in group if nonempty_final_answer(row)]
        timed = []
        for row in extractable:
            try:
                seconds = float(row.get("inference_seconds"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(seconds) and seconds >= 0:
                timed.append(seconds)
        total = len(group)
        correct = sum(int(bool(row.get("strict_exact_match", 0))) for row in group)
        return {
            "total": total,
            "correct": correct,
            "accuracy_percent": 100 * correct / total if total else None,
            "extractable_final_answers": len(extractable),
            "ordinary_answer_rate": len(extractable) / total if total else None,
            "timed_extractable_final_answers": len(timed),
            "average_inference_seconds_extractable_final_answers": (
                sum(timed) / len(timed) if timed else None
            ),
        }

    overall = summarize(rows)
    by_ability = {
        level: summarize(groups[level])
        for level in (*ABILITY_LEVELS, *(level for level in sorted(groups) if level not in ABILITY_LEVELS))
    }
    summary = {
        "responses_jsonl": str(response_path),
        "overall": overall,
        "by_ability_level": by_ability,
        "definitions": {
            "ordinary_answer_rate": "Fraction with a non-empty extracted final_answer.",
            "average_inference_seconds_extractable_final_answers": (
                "Mean inference_seconds only among rows with a non-empty extracted final_answer."
            ),
        },
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "accuracy_by_ability_level.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["ability_level", *next(iter(by_ability.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for level, values in by_ability.items():
            writer.writerow({"ability_level": level, **values})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
