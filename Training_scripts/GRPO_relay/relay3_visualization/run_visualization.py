#!/usr/bin/env python3
"""Compare step-4000 relay3 and adapter-only decoder image attention."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from dataclasses import asdict, dataclass
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

from capture import ImageAttentionCapture
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import MapQuestion, build_cartographic_prompt
from evaluate_core import load_evaluation_questions, load_policy
from relay_decode import generate_relay_with_capture
from relay_module import RelayModule
from render import render_question
from rolling_sparse_qwen3vl import load_model
from utils import (
    build_adapter_only_prefill,
    decoder_hidden_size,
    encode_global_image,
    generate_from_relays,
    qwen_lm,
)
from Image_question_CoT_shared_relay.layout import build_prefill as build_relay_prefill


DEFAULT_OUTPUT = VLM_ROOT / "Training_outputs/Prototype/relay3_visualization_step4000"
DEFAULT_RELAY_RUN = VLM_ROOT / "Training_outputs/Prototype/relay3/checkpoint"
DEFAULT_ADAPTER_RUN = VLM_ROOT / "Training_outputs/Prototype/adapter_only/checkpoint"


@dataclass(frozen=True)
class SelectedQuestion:
    dataset: str
    array_index_0based: int
    item: MapQuestion
    folder: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=HERE / "questions.json")
    parser.add_argument(
        "--mapverse-dataset", type=Path,
        default=VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json",
    )
    parser.add_argument(
        "--mapwise-dataset", type=Path,
        default=VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json",
    )
    parser.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    parser.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    parser.add_argument("--adapter-run-dir", type=Path, default=DEFAULT_ADAPTER_RUN)
    parser.add_argument("--relay-run-dir", type=Path, default=DEFAULT_RELAY_RUN)
    parser.add_argument("--checkpoint-step", type=int, default=4000)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--num-relay-tokens", type=int, default=1)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.checkpoint_step != 4000:
        parser.error("This comparison is intentionally fixed to checkpoint step 4000")
    if args.max_new_tokens < 1 or args.max_pixels < 1024 or args.num_relay_tokens < 1:
        parser.error("Token, pixel and relay counts must be positive")
    return args


def load_selected(args) -> list[SelectedQuestion]:
    specs = json.loads(args.selection.read_text(encoding="utf-8"))
    pools = {
        "mapverse": load_evaluation_questions(
            args.mapverse_dataset, args.mapverse_root, args.mapwise_root
        ),
        "mapwise": load_evaluation_questions(
            args.mapwise_dataset, args.mapverse_root, args.mapwise_root
        ),
    }
    selected = []
    for spec in specs:
        dataset = str(spec["dataset"])
        index = int(spec["array_index_0based"])
        if dataset not in pools or not 0 <= index < len(pools[dataset]):
            raise IndexError(f"Invalid selection {dataset}[{index}]")
        item = pools[dataset][index]
        folder = f"{dataset}_qa{index:03d}"
        selected.append(SelectedQuestion(dataset, index, item, folder))
    if len({row.folder for row in selected}) != len(selected):
        raise ValueError("Selection contains duplicate questions")
    return selected


def grid_and_positions(model, source_inputs):
    qwen = qwen_lm(model)
    positions = source_inputs["input_ids"][0].eq(int(qwen.config.image_token_id)).nonzero().flatten()
    t, h, w = map(int, source_inputs["image_grid_thw"][0].tolist())
    merge = int(qwen.config.vision_config.spatial_merge_size)
    grid = (h // merge, w // merge)
    if t != 1 or h % merge or w % merge or positions.numel() != grid[0] * grid[1]:
        raise RuntimeError("Expected one complete still-image merged patch grid")
    return positions, grid, [t, h, w]


def prepare(model, processor, selected: SelectedQuestion, args):
    with Image.open(selected.item.image_path) as handle:
        original = handle.convert("RGB")
        original_size = original.size
        image = _resize_to_pixel_cap(original, args.max_pixels)
    source = _content_inputs(
        processor,
        [image],
        build_cartographic_prompt(selected.item.question),
        model.get_input_embeddings().weight.device,
        args.max_pixels,
    )
    features = encode_global_image(model, source)
    positions, grid, image_grid = grid_and_positions(model, source)
    return source, features, positions, grid, image_grid, original_size, image.size


def completion_record(model, processor, selected, source, generated, maps, grid,
                      image_grid, original_size, submitted_size, checkpoint, variant):
    prefix = source["input_ids"].shape[1]
    ids = generated[0, prefix:]
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    return {
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "dataset": selected.dataset,
        "array_index_0based": selected.array_index_0based,
        "qa_id": selected.item.sample_id,
        "question": selected.item.question,
        "ground_truth": selected.item.answer,
        "image": str(selected.item.image_path.resolve()),
        "response": processor.tokenizer.decode(ids, skip_special_tokens=True),
        "raw_response": processor.tokenizer.decode(ids, skip_special_tokens=False),
        "token_ids": ids.tolist(),
        "tokens": [processor.tokenizer.decode([int(token)]) for token in ids],
        "status": "eos" if ids.numel() and int(ids[-1]) in eos_ids else "token_limit",
        "steps": int(ids.numel()),
        "layers": int(maps.shape[1]),
        "merged_grid_hw": list(grid),
        "image_grid_thw": image_grid,
        "original_size_wh": list(original_size),
        "submitted_size_wh": list(submitted_size),
        "map_axes": ["inference_step", "decoder_layer", "patch_y", "patch_x"],
        "frame_semantics": (
            "Frame t is the decoder query that predicts generated token t. "
            "Relay3 frames jointly process [x_t,r_t]; adapter-only frame 0 is "
            "the last prompt query and later frames query the preceding generated token."
        ),
    }


def save_variant(destination: Path, record: dict, hidden: np.ndarray,
                 relay: np.ndarray | None = None):
    destination.mkdir(parents=True, exist_ok=False)
    arrays = {"hidden_attention": hidden.astype(np.float32)}
    if relay is not None:
        arrays["relay_attention"] = relay.astype(np.float32)
    np.savez_compressed(destination / "attention.npz", **arrays)
    (destination / "result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "response.txt").write_text(record["response"], encoding="utf-8")


def run_adapter(selected, args, output):
    checkpoint = args.adapter_run_dir / f"checkpoint-{args.checkpoint_step}.pt"
    adapter = args.adapter_run_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter-only step-4000 checkpoint below {args.adapter_run_dir}")
    state = torch.load(checkpoint, map_location="cpu")
    if state.get("placement") != "none" or state.get("policy_kind") != "adapter_only":
        raise ValueError("Adapter-only checkpoint metadata is incompatible")
    del state
    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        for row in selected:
            print(f"[adapter-only] {row.folder}", flush=True)
            source, features, positions, grid, image_grid, original_size, submitted_size = prepare(
                model, processor, row, args
            )
            augmented, _ = build_adapter_only_prefill(source, features)
            with ImageAttentionCapture(model, positions, grid, query_kind="adapter") as capture:
                generated = generate_from_relays(
                    model, **augmented, do_sample=False, num_beams=1,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                )
            token_count = generated.shape[1] - source["input_ids"].shape[1]
            hidden, _ = capture.arrays(token_count, allow_one_unused_tail=True)
            record = completion_record(
                model, processor, row, source, generated, hidden, grid, image_grid,
                original_size, submitted_size, checkpoint, "adapter_only",
            )
            save_variant(output / row.folder / "adapter_only", record, hidden)
            print(f"  steps={token_count} status={record['status']}", flush=True)
            del source, features, augmented, generated, hidden
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_relay(selected, args, output):
    checkpoint = args.relay_run_dir / f"checkpoint-{args.checkpoint_step}.pt"
    adapter = args.relay_run_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing relay3 step-4000 checkpoint below {args.relay_run_dir}")
    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = RelayModule(
        decoder_hidden_size(model), num_relays=args.num_relay_tokens
    ).to(device).eval()
    state = load_policy(
        policy, checkpoint, "dynamic_cot_relay", "image_question_cot_shared"
    )
    if int(state.get("step", -1)) != args.checkpoint_step:
        raise ValueError("Relay checkpoint step metadata mismatch")
    if int(state.get("num_relay_tokens", -1)) != args.num_relay_tokens:
        raise ValueError("Relay count does not match checkpoint metadata")
    try:
        for row in selected:
            print(f"[relay3] {row.folder}", flush=True)
            source, features, positions, grid, image_grid, original_size, submitted_size = prepare(
                model, processor, row, args
            )
            augmented, _ = build_relay_prefill(
                model, processor.tokenizer, source, features, policy, with_grad=False
            )
            generated, hidden, relay = generate_relay_with_capture(
                model, augmented, positions, grid, args.max_new_tokens
            )
            record = completion_record(
                model, processor, row, source, generated, hidden, grid, image_grid,
                original_size, submitted_size, checkpoint, "relay3",
            )
            save_variant(output / row.folder / "relay3", record, hidden, relay)
            print(f"  steps={record['steps']} status={record['status']}", flush=True)
            del source, features, augmented, generated, hidden, relay
    finally:
        del policy, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    selected = load_selected(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    existing = [args.output / row.folder for row in selected if (args.output / row.folder).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Use --overwrite or a fresh --output; found {existing[0]}")
    if args.overwrite:
        for path in existing:
            shutil.rmtree(path)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_step": args.checkpoint_step,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "num_relay_tokens": args.num_relay_tokens,
        "selection": [
            {"dataset": row.dataset, "array_index_0based": row.array_index_0based,
             "qa_id": row.item.sample_id, "folder": row.folder}
            for row in selected
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_adapter(selected, args, args.output)
    run_relay(selected, args, args.output)
    pages = [render_question(args.output / row.folder) for row in selected]
    print("\nHTML pages:", flush=True)
    for page in pages:
        print(page, flush=True)


if __name__ == "__main__":
    main()
