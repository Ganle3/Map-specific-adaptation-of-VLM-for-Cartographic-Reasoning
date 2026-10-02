#!/usr/bin/env python3
"""Run MapWise inference and MapVerse exact evaluation in one command.

This is a convenience wrapper for local diagnosis runs.  It preserves the raw
inference JSON and writes the evaluator's JSON/CSV/summary beside it.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
INFER = HERE / "inference_mapwise_trl.py"
EVALUATE = HERE / "evaluate_mapverse_predictions.py"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT/LoRA checkpoint. Omit this argument for baseline inference.",
    )
    p.add_argument("--qa-json", required=True, type=Path)
    p.add_argument("--image-root", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--min-pixels", type=int, default=65536)
    p.add_argument("--max-pixels", type=int, default=1000000)
    p.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_tag = "baseline" if args.adapter_path is None else args.adapter_path.name
    predictions = out / f"predictions_{model_tag}_tokens{args.max_new_tokens}.json"
    detail_dir = out / "evaluation"

    infer = [sys.executable, str(INFER), "--model-name", args.model_name,
             "--qa-json", str(args.qa_json),
             "--image-root", str(args.image_root), "--output-json", str(predictions),
             "--max-new-tokens", str(args.max_new_tokens), "--thinking", args.thinking,
             "--min-pixels", str(args.min_pixels), "--max-pixels", str(args.max_pixels),
             "--save-every", str(args.save_every), "--print-every", str(args.print_every)]
    if args.adapter_path is not None:
        infer.extend(["--adapter-path", str(args.adapter_path)])
    if args.no_resume:
        infer.append("--no-resume")
    if args.overwrite:
        infer.append("--overwrite")
    subprocess.run(infer, check=True)

    evaluate = [sys.executable, str(EVALUATE), "--predictions-json", str(predictions),
                "--output-dir", str(detail_dir)]
    subprocess.run(evaluate, check=True)
    print(f"Raw predictions: {predictions}")
    print(f"Evaluation outputs: {detail_dir}")


if __name__ == "__main__":
    main()
