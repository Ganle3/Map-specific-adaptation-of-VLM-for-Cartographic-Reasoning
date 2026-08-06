#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from inference_frieda import run_frieda_inference
from frieda_evaluation import evaluate_frieda, print_summary


# ============================================================
# 1. Stable experiment settings
# ============================================================

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
DISTANCE_TOLERANCE = 0.20
JUDGE_MODEL: Optional[str] = "mistralai/Ministral-8B-Instruct-2410"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ORIENTATION_PKL = SCRIPT_DIR / "orientation.pkl"


# ============================================================
# 2. Split-aware default paths
# ============================================================

def default_data_paths(split: str) -> tuple[Path, Path]:
    """Return the default original-format QA JSON and image directory."""
    if split == "validation":
        root = PROJECT_ROOT / "Train_Val_data" / "FRIEDA"
        return root / "frieda_validation.json", root / "image"

    if split == "test":
        root = PROJECT_ROOT / "Test_data" / "FRIEDA_test"
        return root / "frieda_test.json", root / "image"

    raise ValueError(f"Unsupported split: {split}")


def default_experiment_name(split: str, adapter_path: Optional[Path]) -> str:
    variant = "adapted" if adapter_path is not None else "baseline"
    return f"FRIEDA_{split}_{variant}"


# ============================================================
# 3. Validation and reporting utilities
# ============================================================

def validate_project_paths(
    qa_json: Path,
    image_root: Path,
    adapter_path: Optional[Path],
    output_dir: Path,
) -> None:
    required_paths = {
        "FRIEDA QA JSON": qa_json,
        "FRIEDA image directory": image_root,
    }

    if adapter_path is not None:
        required_paths["LoRA adapter directory"] = adapter_path
        required_paths["LoRA adapter config"] = adapter_path / "adapter_config.json"

    missing = [
        f"{label}:\n  {path}"
        for label, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required paths were not found:\n\n"
            + "\n\n".join(missing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)


def count_qa_samples(qa_json: Path) -> int:
    with qa_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"FRIEDA QA JSON must contain a list: {qa_json}")

    return len(data)


def count_prediction_records(prediction_json: Path) -> int:
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
    if ORIENTATION_PKL.exists():
        return ORIENTATION_PKL.resolve()

    print(
        "\nWarning: orientation.pkl was not found; the built-in mapping will be used.",
        file=sys.stderr,
    )
    return None


def print_pipeline_configuration(
    *,
    split: str,
    experiment_name: str,
    model_name: str,
    adapter_path: Optional[Path],
    qa_json: Path,
    image_root: Path,
    output_dir: Path,
    prediction_json: Path,
) -> None:
    print("=" * 80)
    print("FRIEDA EVALUATION PIPELINE")
    print("=" * 80)
    print(f"Split:            {split}")
    print(f"Experiment:       {experiment_name}")
    print(f"Base model:       {model_name}")
    print(
        f"Adapter:          {adapter_path if adapter_path is not None else 'None (baseline)'}"
    )
    print(f"QA JSON:          {qa_json}")
    print(f"Image directory:  {image_root}")
    print(f"Output directory: {output_dir}")
    print(f"Prediction JSON:  {prediction_json}")
    print(f"Max new tokens:   {MAX_NEW_TOKENS}")
    print(f"Distance tol.:    {DISTANCE_TOLERANCE:.0%}")
    print(
        f"Orientation map:  {ORIENTATION_PKL if ORIENTATION_PKL.exists() else 'built-in mapping'}"
    )
    print(f"Text judge:       {JUDGE_MODEL or 'deterministic matching only'}")


# ============================================================
# 4. Pipeline stages
# ============================================================

def run_inference_stage(
    *,
    model_name: str,
    adapter_path: Optional[Path],
    qa_json: Path,
    image_root: Path,
    prediction_json: Path,
    start_index: int,
    end_index: Optional[int],
    resume: bool,
    overwrite: bool,
) -> Path:
    print("\n" + "=" * 80)
    print("STAGE 1: FRIEDA INFERENCE")
    print("=" * 80)

    prediction_path = run_frieda_inference(
        model_name=model_name,
        adapter_path=adapter_path,
        qa_json=qa_json,
        image_root=image_root,
        output_json=prediction_json,
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
    *,
    prediction_json: Path,
    qa_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("STAGE 2: FRIEDA EVALUATION")
    print("=" * 80)

    return evaluate_frieda(
        prediction_json=prediction_json,
        qa_json=qa_json,
        output_dir=output_dir,
        orientation_pkl=resolve_orientation_path(),
        distance_tolerance=DISTANCE_TOLERANCE,
        judge_model=JUDGE_MODEL,
        judge_load_in_4bit=True,
    )


# ============================================================
# 5. Complete pipeline
# ============================================================

def run_frieda_pipeline(
    *,
    split: str,
    experiment_name: str,
    model_name: str,
    adapter_path: Optional[Path],
    qa_json: Path,
    image_root: Path,
    output_dir: Path,
    inference_only: bool = False,
    evaluation_only: bool = False,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
) -> Optional[dict[str, Any]]:
    if inference_only and evaluation_only:
        raise ValueError(
            "--inference-only and --evaluation-only cannot be used together."
        )

    qa_json = qa_json.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    adapter_path = (
        adapter_path.expanduser().resolve() if adapter_path is not None else None
    )
    prediction_json = output_dir / "frieda_predictions.json"

    validate_project_paths(qa_json, image_root, adapter_path, output_dir)
    print_pipeline_configuration(
        split=split,
        experiment_name=experiment_name,
        model_name=model_name,
        adapter_path=adapter_path,
        qa_json=qa_json,
        image_root=image_root,
        output_dir=output_dir,
        prediction_json=prediction_json,
    )

    print(f"QA samples:       {count_qa_samples(qa_json)}")
    print(f"Saved predictions:{count_prediction_records(prediction_json):>6}")

    if evaluation_only:
        if not prediction_json.exists():
            raise FileNotFoundError(
                "Evaluation-only mode requires an existing prediction file:\n"
                f"{prediction_json}"
            )
        prediction_path = prediction_json
    else:
        prediction_path = run_inference_stage(
            model_name=model_name,
            adapter_path=adapter_path,
            qa_json=qa_json,
            image_root=image_root,
            prediction_json=prediction_json,
            start_index=start_index,
            end_index=end_index,
            resume=resume,
            overwrite=overwrite,
        )

    if inference_only:
        print("\nInference-only mode completed.")
        print(f"Predictions saved to: {prediction_path}")
        return None

    summary = run_evaluation_stage(
        prediction_json=prediction_path,
        qa_json=qa_json,
        output_dir=output_dir,
    )

    print("\n")
    print_summary(summary)
    print("\n" + "=" * 80)
    print("FRIEDA PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Predictions: {prediction_path.resolve()}")
    print(f"Summary:     {output_dir / 'evaluation_summary.json'}")
    print(f"Details:     {output_dir / 'evaluation_details.json'}")
    print(f"CSV:         {output_dir / 'evaluation_details.csv'}")
    return summary


# ============================================================
# 6. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FRIEDA validation/test inference and evaluation with either "
            "the baseline Qwen3-VL model or a PEFT/LoRA adapter."
        )
    )

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Dataset split to evaluate. Default: test.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT/LoRA adapter directory. Omit for baseline evaluation.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Output experiment name. If omitted, uses "
            "FRIEDA_<split>_<baseline|adapted>."
        ),
    )
    parser.add_argument(
        "--qa-json",
        type=Path,
        default=None,
        help="Optional QA JSON override for the selected split.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional image-directory override for the selected split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output-directory override.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--inference-only", action="store_true")
    mode_group.add_argument("--evaluation-only", action="store_true")

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> Optional[dict[str, Any]]:
    args = parse_args()

    default_qa_json, default_image_root = default_data_paths(args.split)
    qa_json = args.qa_json or default_qa_json
    image_root = args.image_root or default_image_root

    experiment_name = args.experiment_name or default_experiment_name(
        args.split,
        args.adapter_path,
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "Evaluation_results" / experiment_name
    )

    return run_frieda_pipeline(
        split=args.split,
        experiment_name=experiment_name,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        qa_json=qa_json,
        image_root=image_root,
        output_dir=output_dir,
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
        print("\nPipeline interrupted by the user.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print("\nFRIEDA pipeline failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise