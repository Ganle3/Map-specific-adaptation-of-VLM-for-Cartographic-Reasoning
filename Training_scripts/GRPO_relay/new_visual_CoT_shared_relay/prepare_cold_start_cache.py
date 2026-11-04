#!/usr/bin/env python3
"""Cache frozen Qwen3-VL merger embeddings for Relay4 cold-start training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
RELAY_ROOT = THIS_DIR.parent
VLM_ROOT = RELAY_ROOT.parents[1]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (RELAY_ROOT, SPARSE_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt
from rolling_sparse_qwen3vl import load_model
from utils import encode_global_image, qwen_lm


def records(path: Path):
    unique = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            unique[str(value["sample_id"])] = value
    return list(unique.values())


def cache_name(record: dict) -> str:
    grid = record["global_image"]["grid"]
    return f"{record['global_image']['sha256_prefix']}_{grid[0]}x{grid[1]}.pt"


def parse_args():
    default_teacher = VLM_ROOT / "Training_outputs/gradattn_teacher"
    default_adapter = (VLM_ROOT / "Training_outputs" /
                       "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500" /
                       "checkpoint-2500")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, default=default_teacher)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, default=default_adapter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--correct-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.cache_dir = args.cache_dir or args.teacher_root / "global_merger"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.teacher_root / "manifest.jsonl"
    values = records(manifest)
    if args.correct_only:
        values = [v for v in values if float(v["evaluation"]["reward"]) > 0]
    # One vision encoding is enough for every question sharing an image.
    by_cache = {}
    for value in values:
        by_cache.setdefault(cache_name(value), value)
    pending = [(name, value) for name, value in by_cache.items()
               if not (args.cache_dir / name).is_file()]
    pending_total = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]
    print(json.dumps({"records": len(values), "unique_images": len(by_cache),
                      "already_cached": len(by_cache) - pending_total,
                      "missing_total": pending_total,
                      "to_encode_this_run": len(pending)}, indent=2), flush=True)
    if not pending:
        return

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = qwen_lm(model).get_input_embeddings().weight.device
    for index, (name, record) in enumerate(pending, start=1):
        image_path = Path(record["image"])
        with Image.open(image_path) as handle:
            original = handle.convert("RGB")
        max_pixels = int(json.loads((args.teacher_root / "config.json").read_text())["global_max_pixels"])
        submitted = _resize_to_pixel_cap(original, max_pixels)
        inputs = _content_inputs(
            processor, [submitted], build_cartographic_prompt(record["question"]),
            device, max_pixels,
        )
        features = encode_global_image(model, inputs)
        memory = features.pooler_output
        memory = memory if isinstance(memory, torch.Tensor) else torch.cat(tuple(memory), dim=0)
        grid = tuple(map(int, record["global_image"]["grid"]))
        if memory.shape != (grid[0] * grid[1], memory.shape[-1]):
            raise RuntimeError(f"Merger/grid mismatch for {record['sample_id']}: {memory.shape} vs {grid}")
        destination = args.cache_dir / name
        torch.save({
            "memory_bf16": memory.detach().to("cpu", dtype=torch.bfloat16),
            "grid": torch.tensor(grid, dtype=torch.int32),
            "image_sha256_prefix": record["global_image"]["sha256_prefix"],
            "source_image": str(image_path),
        }, destination)
        print(f"[CACHE {index:04d}/{len(pending):04d}] {destination.name} {tuple(memory.shape)}", flush=True)


if __name__ == "__main__":
    main()
