#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run MapWise inference + evaluation for a base model and optional GRPO adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from inference_mapwise import (
    MODEL_NAME,
    MAX_NEW_TOKENS,
    MAPWISE_JSON,
    MAPWISE_IMAGE_ROOT,
    DEFAULT_OUTPUT_JSON,
    run_mapwise_inference,
)
from mapwise_evaluation import evaluate_mapwise


DEFAULT_EVALUATION_OUTPUT_DIR = None


def validate_inputs(
    qa_json: Path,
    image_root: Path,
    prediction_json: Optional[Path],
    adapter_path: Optional[Path],
    evaluation_only: bool,
) -> None:
    if not qa_json.exists():
        raise FileNotFoundError(f"QA JSON not found:\n{qa_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found:\n{image_root}")

    if adapter_path is not None:
        if not adapter_path.exists() or not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter checkpoint not found:\n{adapter_path}")
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(
                "Adapter checkpoint is missing adapter_config.json:\n"
                f"{adapter_path}"
            )

    if evaluation_only:
        if prediction_json is None:
            raise ValueError(
                "--evaluation-only requires an explicit --predictions-json."
            )
        if not prediction_json.exists():
            raise FileNotFoundError(
                "Evaluation-only mode requires an existing prediction JSON:\n"
                f"{prediction_json}"
            )


def run_mapwise_pipeline(
    *,
    model_name: str = MODEL_NAME,
    adapter_path: Optional[Path] = None,
    qa_json: Path = MAPWISE_JSON,
    image_root: Path = MAPWISE_IMAGE_ROOT,
    prediction_json: Optional[Path] = DEFAULT_OUTPUT_JSON,
    output_dir: Optional[Path] = DEFAULT_EVALUATION_OUTPUT_DIR,
    max_new_tokens: int = MAX_NEW_TOKENS,
    thinking_mode: str = "auto",
    start_index: int = 0,
    end_index: Optional[int] = None,
    overwrite: bool = False,
    evaluation_only: bool = False,
) -> Path:
    qa_json = Path(qa_json).resolve()
    image_root = Path(image_root).resolve()
    adapter_path = (
        Path(adapter_path).expanduser().resolve()
        if adapter_path is not None
        else None
    )

    if prediction_json is not None:
        prediction_json = Path(prediction_json).resolve()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()

    validate_inputs(
        qa_json=qa_json,
        image_root=image_root,
        prediction_json=prediction_json,
        adapter_path=adapter_path,
        evaluation_only=evaluation_only,
    )

    if not evaluation_only:
        print("=" * 80)
        print("Stage 1/2: MapWise checkpoint inference")
        print("=" * 80)
        print(f"Base model: {model_name}")
        print(f"Adapter:    {adapter_path if adapter_path else 'None (baseline)'}")

        prediction_json = run_mapwise_inference(
            model_name=model_name,
            adapter_path=adapter_path,
            qa_json=qa_json,
            image_root=image_root,
            output_json=prediction_json,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode,
            start_index=start_index,
            end_index=end_index,
            overwrite=overwrite,
        )
        prediction_json = Path(prediction_json).resolve()

    if prediction_json is None:
        raise RuntimeError(
            "Prediction JSON path was not resolved. Provide --predictions-json "
            "or run inference first."
        )
    if not prediction_json.exists():
        raise FileNotFoundError(
            f"Prediction JSON not found after inference:\n{prediction_json}"
        )

    if output_dir is None:
        output_dir = prediction_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("Stage 2/2: MapWise evaluation")
    print("=" * 80)
    print(f"Predictions:       {prediction_json}")
    print(f"Evaluation output: {output_dir}")

    evaluate_mapwise(
        prediction_json=prediction_json,
        output_dir=output_dir,
    )
    return prediction_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic MapWise inference/evaluation for a base model "
            "plus an optional GRPO LoRA checkpoint."
        )
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help=(
            "PEFT/LoRA checkpoint directory, e.g. .../checkpoint-15. "
            "Omit to evaluate the baseline model."
        ),
    )
    parser.add_argument("--qa-json", type=Path, default=MAPWISE_JSON)
    parser.add_argument("--image-root", type=Path, default=MAPWISE_IMAGE_ROOT)
    parser.add_argument("--predictions-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVALUATION_OUTPUT_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--thinking",
        dest="thinking_mode",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Skip inference and evaluate an existing --predictions-json.",
    )
    return parser


def main() -> Path:
    args = build_arg_parser().parse_args()
    try:
        return run_mapwise_pipeline(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            qa_json=args.qa_json,
            image_root=args.image_root,
            prediction_json=args.predictions_json,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            thinking_mode=args.thinking_mode,
            start_index=args.start_index,
            end_index=args.end_index,
            overwrite=args.overwrite,
            evaluation_only=args.evaluation_only,
        )
    except Exception as exc:
        print("\nProgram failed.")
        print(f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
