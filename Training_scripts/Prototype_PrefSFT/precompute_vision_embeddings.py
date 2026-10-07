#!/usr/bin/env python3
"""Cache frozen final-ViT states once per question, before Qwen's final merger.

The cache is valid only for the exact frozen base+GRPO adapter and image
processor configuration used to create it. Do not reuse it after changing
either visual LoRA weights or processor pixel limits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from prototype_module import CachedVisionForward
from utils import (image_for_question, load_backbone, load_questions, make_trajectory_inputs,
                   model_device, qwen_visual, vision_cache_key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True); p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--mapwise-image-root", type=Path, required=True); p.add_argument("--mapverse-image-root", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path, required=True); p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--full-precision", action="store_true"); args = p.parse_args()
    questions = load_questions(args.rollouts)
    model, processor = load_backbone(args.model_name, args.adapter_path, load_in_4bit=not args.full_precision)
    model.eval(); device = model_device(model); cache = CachedVisionForward(qwen_visual(model), args.cache_dir)
    try:
        for index, question in enumerate(questions, 1):
            key = vision_cache_key(question)
            if cache.has(key): continue
            image = image_for_question(question, args.mapwise_image_root, args.mapverse_image_root)
            # Any response is sufficient: vision encoding depends only on image pixels/grid.
            inputs = make_trajectory_inputs(processor, question, question["trajectories"][0], image, device)
            inputs.pop("labels")
            with cache.capture(key):
                with torch.no_grad(): model(**inputs)
            print(f"[{index}/{len(questions)}] cached {question['qa_id']}", flush=True)
    finally:
        cache.remove()


if __name__ == "__main__": main()
