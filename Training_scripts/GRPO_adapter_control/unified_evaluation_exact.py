"""Dataset-aware exact scorer for mixed MapWise and MapVerse JSON files."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "Evaluation_scripts" / "GRPO_ablation"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mapverse_evaluation_exact import evaluate_exact as mapverse_exact
from mapwise_evaluation_exact import evaluate_sample as mapwise_sample


def evaluate_sample(record):
    if str(record.get("country", "")).casefold() != "mapverse":
        return mapwise_sample(record)
    result = mapverse_exact(
        record.get("raw_response", ""),
        record.get("ground_truth", ""),
        record.get("ground_truth_type", ""),
    )
    output = dict(record)
    output.update(
        strict_exact_match=int(bool(result.get("correct", result.get("reward", 0)))),
        evaluated_answer=result.get("prediction_raw", ""),
        normalized_prediction=result.get("prediction_canonical", ""),
        normalized_ground_truth=result.get("ground_truth_canonical", ""),
        metric="mapverse_exact",
        evaluation_note="",
    )
    return output
