#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the complete FRIEDA baseline pipeline:

    1. Qwen3-VL inference
    2. FRIEDA evaluation
    3. Save predictions and evaluation results

Expected project structure
--------------------------
Thesis/
├── Evaluation_scripts/
│   ├── run_frieda.py
│   ├── inference_frieda.py
│   ├── frieda_evaluation.py
│   └── orientation.pkl
│
├── Test_data/
│   └── FRIEDA_test/
│       ├── frieda_test.json
│       └── image/
│
└── Evaluation_results/
    └── FRIEDA_Qwen3VL/
        ├── frieda_predictions.json
        ├── evaluation_summary.json
        ├── evaluation_details.json
        └── evaluation_details.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from inference_frieda import run_frieda_inference
from frieda_evaluation import evaluate_frieda, print_summary


# ============================================================
# 1. Experiment configuration
# ============================================================

EXPERIMENT_NAME = "FRIEDA_Qwen3VL"

MODEL_NAME = (
    "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
)

MAX_NEW_TOKENS = 3072

DISTANCE_TOLERANCE = 0.20

# None means deterministic textual matching only.
#
# To use a local LLM judge later, replace None with a model name, for example:
#
JUDGE_MODEL = "mistralai/Ministral-8B-Instruct-2410"
#
# Loading the inference model and judge simultaneously may exceed GPU memory.
# Therefore, it is safer to run evaluation after inference has released
# the Qwen model.
# JUDGE_MODEL: Optional[str] = None


# ============================================================
# 2. Automatic project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# The scripts are assumed to be inside:
# Thesis/Evaluation_scripts/
PROJECT_ROOT = SCRIPT_DIR.parent

FRIEDA_ROOT = PROJECT_ROOT / "Test_data" / "FRIEDA_test"
QA_JSON = FRIEDA_ROOT / "frieda_test.json"
IMAGE_ROOT = FRIEDA_ROOT / "image"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Evaluation_results"
    / EXPERIMENT_NAME
)

PREDICTION_JSON = OUTPUT_DIR / "frieda_predictions.json"

# The official FRIEDA orientation file is preferred when present.
ORIENTATION_PKL = SCRIPT_DIR / "orientation.pkl"


# ============================================================
# 3. Validation utilities
# ============================================================

def validate_project_paths() -> None:
    """Check all required input files and directories before execution."""
    required_paths = {
        "FRIEDA QA JSON": QA_JSON,
        "FRIEDA image directory": IMAGE_ROOT,
    }

    missing: list[str] = []

    for label, path in required_paths.items():
        if not path.exists():
            missing.append(f"{label}:\n  {path}")

    if missing:
        joined = "\n\n".join(missing)
        raise FileNotFoundError(
            "The following required FRIEDA paths were not found:\n\n"
            f"{joined}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def count_qa_samples(qa_json: Path) -> int:
    """Return the number of QA records in the FRIEDA JSON."""
    with qa_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(
            f"FRIEDA QA JSON must contain a list: {qa_json}"
        )

    return len(data)


def count_prediction_records(prediction_json: Path) -> int:
    """Return the number of prediction records already saved."""
    if not prediction_json.exists():
        return 0

    with prediction_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return len(data)

    if isinstance(data, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(data.get(key), list):
                return len(data[key])
        return len(data)

    return 0


def resolve_orientation_path() -> Optional[Path]:
    """
    Use the official orientation.pkl when available.

    If the file is absent, frieda_evaluation.py falls back to its built-in
    direction mapping.
    """
    if ORIENTATION_PKL.exists():
        return ORIENTATION_PKL.resolve()

    print(
        "\nWarning: orientation.pkl was not found.",
        file=sys.stderr,
    )
    print(
        "The built-in orientation mapping in "
        "frieda_evaluation.py will be used instead.",
        file=sys.stderr,
    )
    return None


def print_pipeline_configuration() -> None:
    """Print the resolved experiment configuration."""
    print("=" * 80)
    print("FRIEDA BASELINE PIPELINE")
    print("=" * 80)
    print(f"Experiment:       {EXPERIMENT_NAME}")
    print(f"Model:            {MODEL_NAME}")
    print(f"QA JSON:          {QA_JSON}")
    print(f"Image directory:  {IMAGE_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Prediction JSON:  {PREDICTION_JSON}")
    print(f"Max new tokens:   {MAX_NEW_TOKENS}")
    print(f"Distance tol.:    {DISTANCE_TOLERANCE:.0%}")

    if ORIENTATION_PKL.exists():
        print(f"Orientation map:  {ORIENTATION_PKL}")
    else:
        print("Orientation map:  built-in mapping")

    if JUDGE_MODEL:
        print(f"Text judge:       {JUDGE_MODEL}")
    else:
        print("Text judge:       deterministic matching only")


# ============================================================
# 4. Pipeline stages
# ============================================================

def run_inference_stage(
    *,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
) -> Path:
    """Run FRIEDA inference and return the prediction JSON path."""
    print("\n" + "=" * 80)
    print("STAGE 1: FRIEDA INFERENCE")
    print("=" * 80)

    prediction_path = run_frieda_inference(
        model_name=MODEL_NAME,
        qa_json=QA_JSON,
        image_root=IMAGE_ROOT,
        output_json=PREDICTION_JSON,
        max_new_tokens=MAX_NEW_TOKENS,
        start_index=start_index,
        end_index=end_index,
        resume=resume,
        overwrite=overwrite,
        save_every=1,
        print_every=1,
    )

    if not prediction_path.exists():
        raise FileNotFoundError(
            "Inference completed without producing a prediction file:\n"
            f"{prediction_path}"
        )

    return prediction_path


def run_evaluation_stage(
    prediction_json: Path,
) -> dict[str, Any]:
    """Evaluate the prediction file and return the aggregate summary."""
    print("\n" + "=" * 80)
    print("STAGE 2: FRIEDA EVALUATION")
    print("=" * 80)

    orientation_path = resolve_orientation_path()

    summary = evaluate_frieda(
        prediction_json=prediction_json,
        qa_json=QA_JSON,
        output_dir=OUTPUT_DIR,
        orientation_pkl=orientation_path,
        distance_tolerance=DISTANCE_TOLERANCE,
        judge_model=JUDGE_MODEL,
        judge_load_in_4bit=True,
    )

    return summary


# ============================================================
# 5. Complete pipeline
# ============================================================

def run_frieda_pipeline(
    *,
    inference_only: bool = False,
    evaluation_only: bool = False,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
) -> Optional[dict[str, Any]]:
    """
    Run inference, evaluation, or the complete FRIEDA pipeline.

    Returns
    -------
    dict or None
        Evaluation summary when evaluation is run. Returns None for
        inference-only execution.
    """
    if inference_only and evaluation_only:
        raise ValueError(
            "--inference-only and --evaluation-only cannot be used together."
        )

    validate_project_paths()
    print_pipeline_configuration()

    qa_count = count_qa_samples(QA_JSON)
    existing_predictions = count_prediction_records(PREDICTION_JSON)

    print(f"QA samples:       {qa_count}")
    print(f"Saved predictions:{existing_predictions:>6}")

    if evaluation_only:
        if not PREDICTION_JSON.exists():
            raise FileNotFoundError(
                "Evaluation-only mode requires an existing prediction file:\n"
                f"{PREDICTION_JSON}"
            )

        prediction_path = PREDICTION_JSON

    else:
        prediction_path = run_inference_stage(
            start_index=start_index,
            end_index=end_index,
            resume=resume,
            overwrite=overwrite,
        )

    if inference_only:
        print("\nInference-only mode completed.")
        print(f"Predictions saved to: {prediction_path}")
        return None

    summary = run_evaluation_stage(prediction_path)

    print("\n")
    print_summary(summary)

    print("\n" + "=" * 80)
    print("FRIEDA PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Predictions: {prediction_path.resolve()}")
    print(
        "Summary:     "
        f"{OUTPUT_DIR / 'evaluation_summary.json'}"
    )
    print(
        "Details:     "
        f"{OUTPUT_DIR / 'evaluation_details.json'}"
    )
    print(
        "CSV:         "
        f"{OUTPUT_DIR / 'evaluation_details.csv'}"
    )

    return summary


# ============================================================
# 6. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL inference and FRIEDA evaluation "
            "as one reproducible baseline pipeline."
        )
    )

    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--inference-only",
        action="store_true",
        help="Run inference without evaluation.",
    )

    mode_group.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "Skip model inference and evaluate the existing "
            "frieda_predictions.json."
        ),
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Inclusive starting index for inference.",
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive ending index for inference.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse previously completed predictions.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing prediction file before inference.",
    )

    return parser.parse_args()


def main() -> Optional[dict[str, Any]]:
    args = parse_args()

    return run_frieda_pipeline(
        inference_only=args.inference_only,
        evaluation_only=args.evaluation_only,
        start_index=args.start_index,
        end_index=args.end_index,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nPipeline interrupted by the user.",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as error:
        print("\nFRIEDA pipeline failed.", file=sys.stderr)
        print(
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise