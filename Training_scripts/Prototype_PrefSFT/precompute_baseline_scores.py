#!/usr/bin/env python3
"""Compute frozen M0 length-normalized rollout scores once and atomically cache them."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from contextlib import nullcontext
from prototype_module import CachedVisionForward
from utils import (collate_trajectory_inputs, image_for_question, load_backbone, load_questions,
                   make_trajectory_inputs, model_device, normalized_logprob, trajectory_key, qwen_visual,
                   vision_cache_key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mapwise-image-root", type=Path, required=True); p.add_argument("--mapverse-image-root", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path, required=True); p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--trajectory-micro-batch-size", type=int, default=1)
    p.add_argument("--full-precision", action="store_true")
    p.add_argument("--vision-cache-dir", type=Path, default=None)
    p.add_argument("--min-pixels", type=int, default=65536); p.add_argument("--max-pixels", type=int, default=1000000)
    p.add_argument("--min-correct", type=int, default=1); p.add_argument("--max-correct", type=int, default=7)
    args = p.parse_args()
    existing = json.loads(args.output.read_text(encoding="utf-8")).get("scores", {}) if args.output.exists() else {}
    questions = load_questions(args.rollouts, min_correct=args.min_correct, max_correct=args.max_correct)
    model, processor = load_backbone(args.model_name, args.adapter_path, load_in_4bit=not args.full_precision,
                                     min_pixels=args.min_pixels, max_pixels=args.max_pixels); model.eval(); device = model_device(model)
    vision_cache = CachedVisionForward(qwen_visual(model), args.vision_cache_dir) if args.vision_cache_dir else None
    for qi, q in enumerate(questions, 1):
        image = image_for_question(q, args.mapwise_image_root, args.mapverse_image_root)
        pending = [t for t in q["trajectories"] if trajectory_key(q, t) not in existing]
        for start in range(0, len(pending), args.trajectory_micro_batch_size):
            ts = pending[start:start + args.trajectory_micro_batch_size]
            rows = [make_trajectory_inputs(processor, q, t, image, device) for t in ts]
            batch = collate_trajectory_inputs(rows, processor.tokenizer.pad_token_id)
            labels = batch.pop("labels")
            cache_context = vision_cache.use([vision_cache_key(q)] * len(ts)) if vision_cache else nullcontext()
            with cache_context:
                with torch.no_grad(): scores = normalized_logprob(model(**batch).logits, labels)
            for t, score in zip(ts, scores):
                existing[trajectory_key(q, t)] = {"baseline_normalized_logprob": float(score.cpu())}
        if qi % 10 == 0 or qi == len(questions):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            tmp = args.output.with_suffix(args.output.suffix + ".tmp")
            tmp.write_text(json.dumps({"schema_version": "prototype_baseline_scores_v1", "scores": existing}, indent=2), encoding="utf-8")
            tmp.replace(args.output); print(f"cached {len(existing)} trajectories", flush=True)
    if vision_cache: vision_cache.remove()

if __name__ == "__main__": main()
