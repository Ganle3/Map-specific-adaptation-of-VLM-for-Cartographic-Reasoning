#!/usr/bin/env python3
"""Visualize both fixed Relay1 tokens' prefill image attention at step 4000."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
RELAY_ROOT = HERE.parent
VLM_ROOT = HERE.parents[2]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (HERE, RELAY_ROOT, SPARSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from capture import RelayPrefillAttentionCapture
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import MapQuestion, build_cartographic_prompt
from evaluate_core import load_evaluation_questions, load_policy
from only_visual_relay.layout import build_prefill
from relay_module import RelayModule
from render import render_question
from rolling_sparse_qwen3vl import load_model
from utils import decoder_hidden_size, encode_global_image, generate_from_relays, qwen_lm

DEFAULT_RUN = VLM_ROOT / "Training_outputs/Prototype/relay1/checkpoint"
DEFAULT_OUTPUT = VLM_ROOT / "Training_outputs/Prototype/relay1_visualization_step4000"


@dataclass(frozen=True)
class Selected:
    dataset: str
    index: int
    item: MapQuestion
    folder: str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=HERE / "questions.json")
    parser.add_argument("--mapverse-dataset", type=Path, default=VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json")
    parser.add_argument("--mapwise-dataset", type=Path, default=VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json")
    parser.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    parser.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint-step", type=int, default=4000)
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.checkpoint_step != 4000:
        parser.error("This visualization is intentionally fixed to checkpoint 4000")
    return args


def selected_questions(args) -> list[Selected]:
    specs = json.loads(args.selection.read_text(encoding="utf-8"))
    pools = {
        "mapverse": load_evaluation_questions(args.mapverse_dataset, args.mapverse_root, args.mapwise_root),
        "mapwise": load_evaluation_questions(args.mapwise_dataset, args.mapverse_root, args.mapwise_root),
    }
    result = []
    for spec in specs:
        dataset, index = str(spec["dataset"]), int(spec["array_index_0based"])
        item = pools[dataset][index]
        result.append(Selected(dataset, index, item, f"{dataset}_qa{index:03d}"))
    return result


def prepare(model, processor, row, args):
    with Image.open(row.item.image_path) as handle:
        original = handle.convert("RGB")
        original_size = original.size
        image = _resize_to_pixel_cap(original, args.max_pixels)
    source = _content_inputs(processor, [image], build_cartographic_prompt(row.item.question),
                             model.get_input_embeddings().weight.device, args.max_pixels)
    features = encode_global_image(model, source)
    return source, features, original_size, image.size


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    checkpoint = args.run_dir / "checkpoint-4000.pt"
    adapter = args.run_dir / "checkpoint-4000"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Incomplete Relay1 checkpoint: {args.run_dir}")
    rows = selected_questions(args)
    args.output = args.output.resolve()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use --overwrite: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    policy = RelayModule(decoder_hidden_size(model), num_relays=2).to(
        model.get_input_embeddings().weight.device
    ).eval()
    state = load_policy(policy, checkpoint, "after_vision_end", "only_visual")
    if int(state.get("step", -1)) != 4000 or int(state.get("num_relay_tokens", -1)) != 2:
        raise ValueError("Relay1 checkpoint metadata does not describe step-4000 with two relays")

    pages = []
    try:
        for row in rows:
            print(f"[Relay1 attention] {row.dataset}[{row.index}] {row.item.question}", flush=True)
            destination = args.output / row.folder
            destination.mkdir(parents=True)
            source, features, original_size, submitted_size = prepare(model, processor, row, args)
            augmented, _ = build_prefill(model, processor.tokenizer, source, features,
                                         policy, with_grad=False)
            qwen = qwen_lm(model)
            image_positions = augmented["input_ids"][0].eq(int(qwen.config.image_token_id)).nonzero().flatten()
            t, h, w = map(int, augmented["image_grid_thw"][0].tolist())
            merge = int(qwen.config.vision_config.spatial_merge_size)
            grid = (h // merge, w // merge)
            if t != 1 or image_positions.numel() != grid[0] * grid[1]:
                raise RuntimeError("Expected one complete merged still-image grid")
            relay_positions = augmented["relay_positions"].flatten()
            with RelayPrefillAttentionCapture(model, image_positions, relay_positions, grid) as capture:
                # One token is selected directly from prefill logits, so no second
                # decoder forward occurs and only the prefill attention is captured.
                generate_from_relays(model, **augmented, do_sample=False, num_beams=1,
                                     max_new_tokens=1, use_cache=True)
            maps = capture.arrays()
            np.savez_compressed(destination / "attention.npz", relay_attention=maps)
            record = {
                "variant": "relay1_only_visual", "checkpoint": str(checkpoint.resolve()),
                "dataset": row.dataset, "array_index_0based": row.index,
                "qa_id": row.item.sample_id, "question": row.item.question,
                "ground_truth": row.item.answer, "image": str(row.item.image_path.resolve()),
                "num_relay_tokens": 2, "layers": int(maps.shape[1]),
                "relay_positions": relay_positions.tolist(), "merged_grid_hw": list(grid),
                "image_grid_thw": [t, h, w], "original_size_wh": list(original_size),
                "submitted_size_wh": list(submitted_size),
                "map_axes": ["relay_token", "decoder_layer", "patch_y", "patch_x"],
                "semantics": "Post-softmax relay-query attention to image-token keys, averaged over heads. Relay states are computed once during prefill and then cached per layer.",
            }
            (destination / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            pages.append(render_question(destination))
            print(f"  maps={maps.shape} relay_positions={relay_positions.tolist()}", flush=True)
            del source, features, augmented, maps
    finally:
        del policy, model, processor
        gc.collect()
        torch.cuda.empty_cache()
    (args.output / "manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint.resolve()), "questions": [str(page) for page in pages]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    links = "\n".join(
        f'<li><a href="{page.relative_to(args.output).as_posix()}">{row.dataset} '
        f'{row.index}: {row.item.question}</a></li>'
        for row, page in zip(rows, pages)
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Relay1 attention</title>'
        '<style>body{font:18px system-ui;max-width:1000px;margin:40px auto;line-height:1.6}'
        'a{color:#0759b8}</style><h1>Relay1 step-4000 prefill attention</h1><ul>'
        + links + '</ul>', encoding="utf-8"
    )
    print("\nHTML pages:", flush=True)
    for page in pages:
        print(page, flush=True)


if __name__ == "__main__":
    main()
