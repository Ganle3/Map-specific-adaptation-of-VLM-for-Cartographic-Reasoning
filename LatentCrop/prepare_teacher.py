#!/usr/bin/env python3
"""Encode unique original-resolution Grad-Att crops into frozen main-merger targets.

Run this after the crop collector finishes.  It never forwards a crop through
the language decoder and never trains/updates the Qwen model or its LoRA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

SPARSE = Path(__file__).resolve().parents[1] / "SparseVLM_eliminateCONTEXT"
if str(SPARSE) not in sys.path:
    sys.path.insert(0, str(SPARSE))

from continuous_cache_qwen3vl import _content_inputs, _vl_backbone  # noqa: E402
from rolling_sparse_qwen3vl import load_model  # noqa: E402


def collect_unique_crop_paths(manifest_path: Path) -> list[str]:
    paths = set()
    for line in manifest_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            for step in row["steps"]:
                paths.update(crop["path"] for crop in step["crops"])
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Existing gradattn_teacher directory with manifest.jsonl and crops/.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-crops", type=int, default=None, help="Optional small encoding smoke test.")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.max_pixels < 65_536 or (args.max_crops is not None and args.max_crops < 1):
        parser.error("--max-pixels must be >= 65536; --max-crops must be positive if supplied.")
    source, destination = args.source_root.resolve(), args.output_dir.resolve()
    manifest = source / "manifest.jsonl"
    if not manifest.is_file():
        parser.error(f"Missing collector manifest: {manifest}")
    collector_config_file = source / "config.json"
    collector_adapter_label = None
    if collector_config_file.is_file():
        collector_config = json.loads(collector_config_file.read_text(encoding="utf-8"))
        # Collector JSON written on Windows may be consumed on Linux.  Compare
        # the checkpoint label rather than interpreting a Windows absolute path
        # as a Linux path.  This checks identity by directory name, not bytes;
        # users must copy the same adapter checkpoint to Euler.
        collector_parts = collector_config["adapter"].replace("\\", "/").rstrip("/").split("/")
        collector_adapter_label = "/".join(collector_parts[-2:])
        current_adapter_label = "/".join(args.adapter.parts[-2:])
        if collector_adapter_label != current_adapter_label or bool(collector_config["load_in_4bit"]) != args.load_in_4bit:
            parser.error("Teacher NF4 setting or adapter checkpoint label differs from the Grad-Att collector.")
    destination.mkdir(parents=True, exist_ok=True)
    config = {"adapter_checkpoint_label": collector_adapter_label or "/".join(args.adapter.parts[-2:]),
              "model": args.model, "max_pixels": args.max_pixels, "load_in_4bit": args.load_in_4bit,
              "target": "Qwen3-VL main visual merger pooler_output; no DeepStack"}
    config_file = destination / "teacher_config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        parser.error(f"Teacher cache configuration differs from {config_file}; choose another output directory.")
    config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")

    index_file = destination / "targets.jsonl"
    completed = {}
    if index_file.exists():
        for line in index_file.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                completed[row["crop_path"]] = row
    crop_paths = collect_unique_crop_paths(manifest)
    if args.max_crops is not None:
        crop_paths = crop_paths[:args.max_crops]
    todo = []
    for relative in crop_paths:
        image_path = source / relative
        if not image_path.is_file():
            raise FileNotFoundError(f"Collector crop disappeared: {image_path}")
        stat = image_path.stat()
        previous = completed.get(relative)
        if previous and previous.get("source_size") == stat.st_size and previous.get("source_mtime_ns") == stat.st_mtime_ns \
                and (destination / previous["target_file"]).is_file():
            continue
        todo.append((relative, image_path, stat))
    print(f"Unique crop files={len(crop_paths)}, targets already valid={len(crop_paths)-len(todo)}, to encode={len(todo)}")
    if not todo:
        return

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    backbone = _vl_backbone(model)
    device = model.get_input_embeddings().weight.device
    (destination / "embeddings").mkdir(exist_ok=True)
    with index_file.open("a", encoding="utf-8") as index_handle:
        for number, (relative, image_path, stat) in enumerate(todo, start=1):
            with Image.open(image_path) as opened:
                crop = opened.convert("RGB")
            # This is the same image preprocessing route as a real crop turn.
            # The stored file remains the original-pixel crop; Qwen's processor
            # performs its mandatory dynamic-resolution resizing in memory.
            inputs = _content_inputs(processor, [crop], "visual target", device, args.max_pixels)
            grid = inputs["image_grid_thw"]
            with torch.inference_mode():
                features = backbone.get_image_features(inputs["pixel_values"], grid, return_dict=True)
            embedding = features.pooler_output[0].detach().to("cpu", dtype=torch.bfloat16).contiguous()
            thw = grid[0].detach().to("cpu", dtype=torch.int32)
            merge = int(backbone.visual.spatial_merge_size)
            expected = int(thw[0] * (thw[1] // merge) * (thw[2] // merge))
            if embedding.ndim != 2 or embedding.shape[0] != expected:
                raise RuntimeError(f"Main merger/grid mismatch for {image_path}: {embedding.shape} vs {thw.tolist()}")
            digest = hashlib.sha256(f"{relative}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
            target_relative = f"embeddings/{digest}.pt"
            target_path = destination / target_relative
            temporary = target_path.with_suffix(".pt.tmp")
            torch.save({"embedding": embedding, "image_grid_thw": thw,
                        "source_crop": relative, "original_size": [crop.width, crop.height]}, temporary)
            temporary.replace(target_path)
            row = {"crop_path": relative, "target_file": target_relative,
                   "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                   "token_count": int(embedding.shape[0]), "embedding_dim": int(embedding.shape[1]),
                   "image_grid_thw": thw.tolist(), "original_size": [crop.width, crop.height]}
            index_handle.write(json.dumps(row) + "\n")
            index_handle.flush()
            if number == 1 or number % 100 == 0 or number == len(todo):
                print(f"[{number}/{len(todo)}] {relative}: grid={thw.tolist()} merger_tokens={len(embedding)}")


if __name__ == "__main__":
    main()
