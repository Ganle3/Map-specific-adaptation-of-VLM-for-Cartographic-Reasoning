#!/usr/bin/env python3
"""Verify controller crop/replay insertion against a recorded continuous run.

This test deliberately bypasses WHEN and WHERE. Crop bboxes and replay text
come from an existing ``continuous_result.json``; only the frozen HOW path is
exercised. It replays Window 0 then every recorded crop window, using the
recorded crop/replay state but recomputing the same global attention gate.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT / "SparseVLM_eliminateCONTEXT"
for directory in (str(Path(__file__).resolve().parent), str(SPARSE)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from continuous_cache_qwen3vl import (
    ContinuousConfig, _append_crop_then_assistant_replay, _content_inputs,
    _crop_images_with_budget, _generate_segment, _image_geometries,
    _make_anchor_cache, _resize_to_pixel_cap, load_model,
)
from rolling_sparse_qwen3vl import PixelRegion, crop_region


def _prompt(final: bool) -> str:
    return """The attached crop is fresh visual evidence for the assistant reasoning already in progress.
Continue that exact unfinished reasoning; do not restart the task or reintroduce the legend. """ + (
        "Conclude exactly with: Final answer: <answer>." if final else
        "Add the next concrete visual observation and proceed to the next unresolved check; do not give a final answer yet."
    )


def _controller_how_inputs(processor, images, instruction, replay, device, per_crop_pixels):
    """The adapter boundary: it must delegate to the validated helper verbatim."""
    return _append_crop_then_assistant_replay(processor, images, instruction, replay, device, per_crop_pixels)


def _equal_tensors(left: dict, right: dict) -> list[str]:
    issues = []
    if set(left) != set(right):
        issues.append(f"input keys differ: {set(left) ^ set(right)}")
    for key in set(left) & set(right):
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and not torch.equal(a, b):
            issues.append(f"tensor differs: {key}, {tuple(a.shape)} vs {tuple(b.shape)}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-json", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--window", type=int, default=None,
                        help="Optional single crop window. Omit to validate every recorded crop window.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.recorded_json.read_text(encoding="utf-8"))
    recorded_config = record["config"]
    cfg = ContinuousConfig(**{key: value for key, value in recorded_config.items()
                              if key in ContinuousConfig.__dataclass_fields__})
    windows = record["windows"]
    if args.window is not None and (args.window <= 0 or args.window >= len(windows)):
        parser.error("--window must identify an existing crop window (>0).")
    model, processor = load_model(args.model, args.adapter)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(record["image"]).convert("RGB")
    # Pixel budgets were introduced after some recorded trajectories.  Their
    # absence means the historical run passed original-resolution PIL images
    # directly to the processor; do not silently apply today's 1M default.
    has_pixel_budgets = "global_max_pixels" in recorded_config and "crop_total_max_pixels" in recorded_config
    global_image = _resize_to_pixel_cap(original, cfg.global_max_pixels) if has_pixel_budgets else original
    initial_prompt = f"""Inspect the complete map carefully. Question: {record['question']}
Reason step by step. Do not give a final answer until explicitly asked."""
    initial = _content_inputs(processor, [global_image], initial_prompt, device,
                              cfg.global_max_pixels if has_pixel_budgets else None)
    global_local, global_grids = _image_geometries(model, initial)
    anchor, anchor_cursor = _make_anchor_cache(model, initial)
    zero_gate = torch.zeros(global_local[0].numel(), dtype=torch.float32, device=device)
    # Recreate Window 0 only to obtain the exact initial global-gate vector.
    initial_text, _initial_past, _initial_cursor, initial_maps = _generate_segment(
        model, processor, initial, None, 0, global_local[0], global_local, global_grids,
        cfg.window_tokens, zero_gate, 0.0, cfg.visual_score,
    )
    global_gate, _retained = __import__("continuous_cache_qwen3vl")._patch_level_global_bias(
        initial_maps[0], cfg, global_local[0].numel())
    selected_windows = [args.window] if args.window is not None else list(range(1, len(windows)))
    checks = []
    for index in selected_windows:
        chosen = windows[index]
        regions = [PixelRegion(**row) for row in chosen["input_regions"]]
        if has_pixel_budgets:
            images, budget = _crop_images_with_budget(original, regions, cfg.crop_total_max_pixels)
            per_crop = cfg.crop_total_max_pixels // len(images)
        else:
            images = [crop_region(original, region) for region in regions]
            budget = [{"source_size": [image.width, image.height], "submitted_size": [image.width, image.height],
                       "source_pixels": image.width * image.height, "submitted_pixels": image.width * image.height,
                       "per_crop_max_pixels": None} for image in images]
            per_crop = None
        expected_budget = (chosen.get("input_visual_budget") or {}).get("crops")
        replay = chosen.get("replay_tail_before")
        if replay is None:
            raise RuntimeError(f"Recorded Window {index} has no assistant replay.")
        instruction = _prompt(index == len(windows) - 1)
        reference = _append_crop_then_assistant_replay(processor, images, instruction, replay, device, per_crop)
        controller = _controller_how_inputs(processor, images, instruction, replay, device, per_crop)
        issues = _equal_tensors(reference, controller)
        local, grids = _image_geometries(model, controller)
        crop_absolute = [span + anchor.get_seq_length() for span in local]
        bias_active = index >= cfg.global_bias_from_window and (index - cfg.global_bias_from_window) % cfg.global_bias_interval == 0
        text, _past, _cursor, maps = _generate_segment(
            model, processor, controller, copy.deepcopy(anchor), anchor_cursor,
            global_local[0], [global_local[0], *crop_absolute], [global_grids[0], *grids],
            cfg.final_window_tokens if index == len(windows) - 1 else cfg.window_tokens,
            global_gate if bias_active else zero_gate, cfg.crop_key_bias, cfg.visual_score,
        )
        global_gate, retained = __import__("continuous_cache_qwen3vl")._patch_level_global_bias(
            maps[0], cfg, global_local[0].numel())
        checks.append({
            "window": index, "input_tensor_equivalent": not issues, "input_issues": issues,
            "crop_budget_matches_record": None if expected_budget is None else budget == expected_budget,
            "recorded_crop_budget_available": expected_budget is not None,
            "computed_crop_budget": budget, "anchor_cache_length": anchor.get_seq_length(),
            "recorded_cache_length_before": chosen.get("cache_length_before"),
            "generated_cache_length_after": _past.get_seq_length(),
            "recorded_cache_length_after": chosen.get("cache_length_after"),
            "crop_image_grids": [list(grid) for grid in grids], "global_bias_active": bias_active,
            "retained_global_patches_for_next": retained,
            "generated_text": text, "recorded_text": chosen["text"],
            "generated_text_matches_record": text == chosen["text"],
        })
    report = {
        "recorded_json": str(args.recorded_json.resolve()),
        "legacy_unbounded_pixel_inputs": not has_pixel_budgets,
        "window0_generated_text_matches_record": initial_text == windows[0]["text"],
        "windows": checks,
        "all_input_tensors_equivalent": all(row["input_tensor_equivalent"] for row in checks),
        "all_generated_text_matches_record": all(row["generated_text_matches_record"] for row in checks),
        "reconstructed_reasoning": initial_text + "".join(row["generated_text"] for row in checks),
        "recorded_reasoning": windows[0]["text"] + "".join(windows[row["window"]]["text"] for row in checks),
    }
    report["reconstructed_reasoning_matches_record"] = (
        report["reconstructed_reasoning"] == report["recorded_reasoning"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"window0_matches": report["window0_generated_text_matches_record"],
                      "input_tensors_match": report["all_input_tensors_equivalent"],
                      "all_windows_text_match": report["all_generated_text_matches_record"],
                      "windows_checked": len(checks)}, indent=2))


if __name__ == "__main__":
    main()
