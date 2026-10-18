#!/usr/bin/env python3
"""Evaluate baseline, fixed-grid, learned-mean, or random ROI rollouts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from PIL import Image
from .data import evaluate_answer, load_mapverse_questions
from .roi_policy import DynamicROIPolicy, ROIPolicyConfig
from .rollout import RolloutConfig, collect_rollout
from .spatial_mask import SoftROIMask, VisualTokenGrid
from .train_roi_grpo import build_inputs


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--image-root", required=True); p.add_argument("--checkpoint"); p.add_argument("--adapter")
    p.add_argument("--mode", choices=("baseline", "fixed_grid", "learned_roi", "random_roi"), default="learned_roi")
    p.add_argument("--limit", type=int); p.add_argument("--max-new-tokens", type=int, default=256); p.add_argument("--output", required=True)
    args = p.parse_args(); sparse = Path(__file__).resolve().parents[2] / "SparseVLM_eliminateCONTEXT"; sys.path.insert(0, str(sparse))
    from rolling_sparse_qwen3vl import load_model
    model, processor = load_model(args.model, Path(args.adapter) if args.adapter else None); device = model.get_input_embeddings().weight.device
    policy = DynamicROIPolicy(ROIPolicyConfig(model.config.text_config.hidden_size)).to(device)
    if args.checkpoint: policy.load_state_dict(torch.load(args.checkpoint, map_location=device)["policy"])
    if args.mode == "baseline":
        # A near-zero learnt policy is used only to drive the cache loop; its
        # strengths are zeroed by this explicit head bias.
        with torch.no_grad(): policy.mean_head.bias[4::5].fill_(-30)
    elif args.mode == "fixed_grid":
        with torch.no_grad(): policy.mean_head.bias[4::5].fill_(0.0)
    items = load_mapverse_questions(Path(args.dataset), Path(args.image_root), limit=args.limit); rows = []
    for item in items:
        with Image.open(item.image_path) as im: inputs = build_inputs(processor, im.convert("RGB"), item.question, device)
        _, h, w = inputs["image_grid_thw"][0].tolist(); merge = model.config.vision_config.spatial_merge_size
        rollout = collect_rollout(model, processor, inputs, policy, SoftROIMask(VisualTokenGrid(h//merge, w//merge)),
                                  lambda text: evaluate_answer(text, item)["reward"], RolloutConfig(args.max_new_tokens, eos_token_id=processor.tokenizer.eos_token_id),
                                  deterministic_roi=args.mode != "random_roi")
        rows.append({"sample_id": item.sample_id, "answer_type": item.answer_type, **rollout})
    Path(args.output).write_text("\n".join(json.dumps(x, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) for x in rows), encoding="utf-8")
    print(json.dumps({"accuracy": sum(x["reward"] for x in rows)/max(1,len(rows)), "count": len(rows)}))

if __name__ == "__main__": main()
