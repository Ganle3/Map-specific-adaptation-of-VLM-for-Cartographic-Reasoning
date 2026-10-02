#!/usr/bin/env python3
"""Evaluate inference_mapwise_trl JSON predictions with the MapVerse scorer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mapverse_evaluation_exact import evaluate_exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    predictions = json.loads(args.predictions_json.read_text(encoding="utf-8-sig"))
    if not isinstance(predictions, list):
        raise TypeError("Predictions JSON must contain a top-level list.")

    details = []
    for row in predictions:
        answer_type = row.get("ground_truth_type", row.get("answer_type", ""))
        ground_truth = row.get("ground_truth", row.get("correct_answer", ""))
        result = evaluate_exact(
            row.get("final_answer", row.get("raw_response", "")),
            ground_truth,
            answer_type,
        )
        details.append({
            **row,
            "prediction_raw": result["prediction_raw"],
            "ground_truth_raw": result["ground_truth_raw"],
            "prediction_canonical": result["prediction_canonical"],
            "ground_truth_canonical": result["ground_truth_canonical"],
            "correct": bool(result["correct"]),
            "reward": float(result["reward"]),
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if details:
        with (output / "evaluation_details.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            fields = [
                "qa_id", "source_index", "image_name", "question", "answer_type",
                "ground_truth", "final_answer", "prediction_canonical",
                "ground_truth_canonical", "generated_tokens", "generation_status",
                "correct", "reward", "raw_response",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(details)

    correct = sum(row["correct"] for row in details)
    timed_rows = [
        row for row in details
        if isinstance(row.get("inference_seconds"), (int, float))
    ]
    # A generation that reaches its configured token cap has an unusually
    # long runtime by construction. Exclude it from runtime aggregates so
    # averages compare actual completed generations rather than token-limit
    # outliers. Keep the count explicit in the summary.
    capped_rows = [
        row for row in timed_rows
        if isinstance(row.get("max_new_tokens"), (int, float))
        and isinstance(row.get("generated_tokens"), (int, float))
        and row["generated_tokens"] >= row["max_new_tokens"]
    ]
    inference_times = [
        float(row["inference_seconds"])
        for row in timed_rows
        if row not in capped_rows
    ]
    nontruncated_final_answer = sum(
        bool(str(row.get("final_answer", "")).strip())
        and row.get("generation_status") != "truncated"
        for row in details
    )
    summary = {
        "total": len(details),
        "correct": correct,
        "accuracy_percent": 100 * correct / len(details) if details else 0.0,
        "truncated": sum(row.get("generation_status") == "truncated" for row in details),
        "truncated_fraction": (
            sum(row.get("generation_status") == "truncated" for row in details) / len(details)
            if details else 0.0
        ),
        "final_answer_nontruncated": nontruncated_final_answer,
        "final_answer_nontruncated_fraction": (
            nontruncated_final_answer / len(details) if details else 0.0
        ),
        "inference_timed_records": len(inference_times),
        "inference_capped_records_excluded": len(capped_rows),
        "inference_capped_records_excluded_fraction": (
            len(capped_rows) / len(timed_rows) if timed_rows else 0.0
        ),
        "inference_total_seconds": sum(inference_times),
        "inference_average_seconds": (
            sum(inference_times) / len(inference_times) if inference_times else 0.0
        ),
        "inference_min_seconds": min(inference_times) if inference_times else 0.0,
        "inference_max_seconds": max(inference_times) if inference_times else 0.0,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Details: {output / 'evaluation_details.json'}")


if __name__ == "__main__":
    main()
