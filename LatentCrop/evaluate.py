#!/usr/bin/env python3
"""Measure reconstruction/STOP on held-out maps using a saved extractor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import LatentCropDataset, build_examples, collate_steps
from latent_visual_extractor import ExtractorConfig, LatentVisualExtractor
from train import run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Training output containing map_split.json.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ExtractorConfig(**checkpoint["model_config"])
    training_config = json.loads((args.run_dir / "run_config.json").read_text(encoding="utf-8"))
    map_split = json.loads((args.run_dir / "map_split.json").read_text(encoding="utf-8"))
    all_examples, stats = build_examples(args.source_root, args.teacher_root, config.k_max)
    allowed_maps = set(map_split[args.split])
    selected = [example for example in all_examples if example.map_id in allowed_maps]
    if not selected or set(map_split["train"]) & allowed_maps:
        raise ValueError("Evaluation split is empty or overlaps training maps.")
    model = LatentVisualExtractor(config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(LatentCropDataset(selected, stats["merger_dim"]), batch_size=args.batch_size,
                        collate_fn=collate_steps, shuffle=False, num_workers=0)
    with torch.inference_mode():
        metrics = run_epoch(model, loader, None, device, float(training_config["cosine_weight"]),
                            float(training_config["stop_weight"]), 1, 1.0, train=False)
    result = {"split": args.split, "checkpoint": str(args.checkpoint.resolve()),
              "unique_maps": len({example.map_id for example in selected}), "metrics": metrics}
    destination = args.output or args.run_dir / f"{args.split}_evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
