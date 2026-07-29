# python3
# -*- coding: utf-8 -*-
"""Run MapWise inference and/or deterministic evaluation."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Optional

import torch

from inference_mapwise import (
    DEFAULT_OUTPUT_JSON,
    MAPWISE_IMAGE_ROOT,
    MAPWISE_JSON,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    run_mapwise_inference,
)
from mapwise_evaluation import evaluate_mapwise, print_summary

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPERIMENT_NAME = "Mapwise_Qwen3VL"
QA_JSON = MAPWISE_JSON
IMAGE_ROOT = MAPWISE_IMAGE_ROOT
OUTPUT_DIR = PROJECT_ROOT / "Evaluation_results" / EXPERIMENT_NAME
PREDICTION_JSON = DEFAULT_OUTPUT_JSON


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_inputs(qa_json: Path, image_root: Path,
                    prediction_json: Path, evaluation_only: bool) -> None:
    if not qa_json.exists():
        raise FileNotFoundError(f"QA JSON not found:\n{qa_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found:\n{image_root}")
    if evaluation_only and not prediction_json.exists():
        raise FileNotFoundError(
            "Evaluation-only mode requires an existing prediction JSON:\n"
            f"{prediction_json}"
        )


def run_mapwise_pipeline(
    *,
    model_name: str = MODEL_NAME,
    qa_json: Path = QA_JSON,
    image_root: Path = IMAGE_ROOT,
    prediction_json: Path = PREDICTION_JSON,
    output_dir: Path = OUTPUT_DIR,
    max_new_tokens: int = MAX_NEW_TOKENS,
    start_index: int = 0,
    end_index: Optional[int] = None,
    inference_only: bool = False,
    evaluation_only: bool = False,
    overwrite: bool = False,
) -> Optional[dict]:
    if inference_only and evaluation_only:
        raise ValueError("--inference-only and --evaluation-only cannot be combined.")

    qa_json = Path(qa_json).resolve()
    image_root = Path(image_root).resolve()
    prediction_json = Path(prediction_json).resolve()
    output_dir = Path(output_dir).resolve()
    validate_inputs(qa_json, image_root, prediction_json, evaluation_only)

    print("=" * 80)
    print("MAPWISE EXPERIMENT WORKFLOW")
    print("=" * 80)
    print(f"Experiment:      {EXPERIMENT_NAME}")
    print(f"Model:           {model_name}")
    print(f"QA JSON:         {qa_json}")
    print(f"Image root:      {image_root}")
    print(f"Prediction JSON: {prediction_json}")
    print(f"Output dir:      {output_dir}")
    print(f"Max tokens:      {max_new_tokens}")

    if not evaluation_only:
        print("\n" + "#" * 80)
        print("# STAGE 1: MAPWISE INFERENCE")
        print("#" * 80)
        prediction_json = run_mapwise_inference(
            model_name=model_name,
            qa_json=qa_json,
            image_root=image_root,
            output_json=prediction_json,
            max_new_tokens=max_new_tokens,
            start_index=start_index,
            end_index=end_index,
            resume=not overwrite,
            overwrite=overwrite,
        )
        release_cuda_memory()

    if inference_only:
        print("\nInference-only mode completed.")
        return None

    print("\n" + "#" * 80)
    print("# STAGE 2: MAPWISE EVALUATION")
    print("#" * 80)
    summary = evaluate_mapwise(
        prediction_json=prediction_json,
        qa_json=qa_json,
        output_dir=output_dir,
    )
    print_summary(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MapWise inference and/or evaluation.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--qa-json", type=Path, default=QA_JSON)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--predictions-json", type=Path, default=PREDICTION_JSON)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> Optional[dict]:
    args = parse_args()
    return run_mapwise_pipeline(
        model_name=args.model_name,
        qa_json=args.qa_json,
        image_root=args.image_root,
        prediction_json=args.predictions_json,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        start_index=args.start_index,
        end_index=args.end_index,
        inference_only=args.inference_only,
        evaluation_only=args.evaluation_only,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nProgram failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise