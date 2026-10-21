#!/usr/bin/env python3
"""Pause greedy Qwen3-VL inference at a phrase, then inject hand-drawn ROI bias.

Unlike a two-process prefix replay, this keeps the actual live ``past_key_values``
in GPU memory.  After the terminal pause, selected original-image boxes are
mapped to existing full-image visual keys and receive a logit bias only on
future decoder forwards.  No ViT pass, anchor prefill, or prefix KV replay is
performed after the intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import GlobalVisualKeyBias, _content_inputs, _resize_to_pixel_cap, _segment_positions
from interactive_roi_drawer import draw_rois
from rolling_sparse_qwen3vl import _image_geometries, load_model


def _prompt(question: str) -> str:
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


def _parse_boxes(value: str, image_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    boxes = []
    width, height = image_size
    for raw in value.split(";"):
        values = [int(part.strip()) for part in raw.strip().split()]
        if len(values) != 4:
            raise ValueError("Each ROI must be four integers: left top right bottom; separate boxes with ';'.")
        left, top, right, bottom = values
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"ROI {(left, top, right, bottom)} is outside original image {(width, height)}.")
        boxes.append((left, top, right, bottom))
    if not boxes:
        raise ValueError("At least one ROI is required.")
    return boxes


def _roi_bias(grid: tuple[int, int], original_size: tuple[int, int], boxes, device, magnitude: float) -> torch.Tensor:
    rows, cols = map(int, grid)
    width, height = original_size
    yy = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) / rows * height
    xx = (torch.arange(cols, device=device, dtype=torch.float32) + 0.5) / cols * width
    selected = torch.zeros((rows, cols), device=device, dtype=torch.bool)
    for left, top, right, bottom in boxes:
        selected |= ((xx[None, :] >= left) & (xx[None, :] < right)
                     & (yy[:, None] >= top) & (yy[:, None] < bottom))
    if not selected.any():
        raise RuntimeError("The boxes cover no merged visual-token centers; enlarge them.")
    bias = torch.zeros(rows * cols, dtype=torch.float32, device=device)
    bias[selected.flatten()] = magnitude
    return bias


def _refresh_anchor_trace(model, anchor_inputs, trace_ids: list[int], global_positions, bias):
    """Current LookBetter anchor-trace HOW, without changing visible tokens."""
    anchor_positions = _segment_positions(model, anchor_inputs, 0)
    with GlobalVisualKeyBias(global_positions, bias, True), torch.inference_mode():
        anchor = model(**anchor_inputs, position_ids=anchor_positions, use_cache=True, return_dict=True)
    cursor = int(anchor_positions.max().item()) + 1
    replay = torch.tensor([trace_ids], device=global_positions.device, dtype=anchor_inputs["input_ids"].dtype)
    mask = torch.ones((1, anchor.past_key_values.get_seq_length() + replay.shape[1]),
                      device=global_positions.device, dtype=torch.long)
    replay_positions = torch.arange(cursor, cursor + replay.shape[1], device=global_positions.device,
                                    dtype=torch.long).view(1, 1, -1).expand(3, -1, -1)
    with GlobalVisualKeyBias(global_positions, bias, True), torch.inference_mode():
        output = model(input_ids=replay, attention_mask=mask, position_ids=replay_positions,
                       past_key_values=anchor.past_key_values, use_cache=True, return_dict=True)
    return output.logits[:, -1].argmax(dim=-1), output.past_key_values, cursor + replay.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pause-after", action="append", required=True,
                        help="Case-insensitive phrase required before pausing. Repeat to require multiple phrases.")
    parser.add_argument("--roi-key-bias", type=float, default=3.079442)
    parser.add_argument("--how-refresh", choices=("roi_only", "roi_biased_full_anchor_trace"), default="roi_only",
                        help="roi_only keeps live prefix KV; anchor_trace re-prefills the anchor and teacher-forces the visible trace under ROI bias.")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-json", type=Path, default=None,
                        help="Reuse roi_original_pixels from an earlier manual ROI JSON instead of opening the canvas.")
    parser.add_argument("--roi-input", choices=("canvas", "terminal"), default="canvas",
                        help="canvas opens a mouse ROI picker while preserving live KV; terminal accepts numeric boxes.")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    image = _resize_to_pixel_cap(original, args.global_max_pixels)
    prompt = _prompt(args.question)
    inputs = _content_inputs(processor, [image], prompt, device, args.global_max_pixels)
    global_local, grids = _image_geometries(model, inputs)
    global_positions, grid = global_local[0], grids[0]
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True, return_dict=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(positions.max().item()) + 1
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    ids: list[int] = []
    active_bias = torch.zeros(global_positions.numel(), dtype=torch.float32, device=device)
    paused = False
    boxes = []
    pause_token = None
    teacher_forced_replay_tokens = 0
    for _ in range(args.max_new_tokens):
        ids.append(int(pending.item()))
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        if not paused and all(phrase.casefold() in text.casefold() for phrase in args.pause_after):
            paused, pause_token = True, len(ids)
            print("\n\n=== PAUSED: live KV cache is retained on GPU ===")
            print(text)
            print(f"\nOriginal image size: {original.width} x {original.height}")
            if args.roi_json is not None:
                saved = json.loads(args.roi_json.read_text(encoding="utf-8"))
                saved_boxes = saved.get("roi_original_pixels")
                if not isinstance(saved_boxes, list) or not saved_boxes:
                    raise ValueError(f"No roi_original_pixels found in {args.roi_json}.")
                boxes = _parse_boxes("; ".join(" ".join(map(str, box)) for box in saved_boxes), original.size)
                print(f"Reused {len(boxes)} ROI(s) from {args.roi_json}.")
            elif args.roi_input == "canvas":
                boxes = draw_rois(original)
            else:
                raw = input("Enter ROI boxes as 'left top right bottom; left top right bottom': ")
                boxes = _parse_boxes(raw, original.size)
            active_bias = _roi_bias(grid, original.size, boxes, device, args.roi_key_bias)
            print(f"Applied +{args.roi_key_bias:g} to {int((active_bias > 0).sum())} / {active_bias.numel()} global visual keys.\n")
            if args.how_refresh == "roi_biased_full_anchor_trace":
                # ``ids`` are the literal visible prefix.  Their text is not
                # regenerated; only its decoder hidden states/KV are rebuilt
                # under the selected ROI key bias.
                teacher_forced_replay_tokens = len(ids)
                pending, past, cursor = _refresh_anchor_trace(model, inputs, ids, global_positions, active_bias)
                print(f"Teacher-forced replay rebuilt {teacher_forced_replay_tokens} reasoning-token KV states under ROI bias.\n")
                # The refreshed ``pending`` predicts the next unseen token;
                # do not feed it in this same iteration or it would skip that
                # token in this loop's append-then-forward ordering.
                continue
        if int(pending.item()) in eos_ids:
            break
        mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=device)
        next_positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=device)
        with GlobalVisualKeyBias(global_positions, active_bias, bool(active_bias.any())), torch.inference_mode():
            output = model(input_ids=pending.view(1, 1), attention_mask=mask, position_ids=next_positions,
                           past_key_values=past, use_cache=True, return_dict=True)
        pending, past, cursor = output.logits[:, -1].argmax(dim=-1), output.past_key_values, cursor + 1

    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    result = {"image": str(args.image.resolve()), "question": args.question, "prompt": prompt,
              "pause_after": args.pause_after, "pause_reached": paused, "pause_token": pause_token,
              "roi_original_pixels": boxes, "roi_key_bias": args.roi_key_bias,
              "roi_visual_tokens": int((active_bias > 0).sum().item()), "response": response,
              "backbone_precision": "nf4_4bit_bf16_compute" if args.load_in_4bit else "torch_dtype_auto",
              "how": args.how_refresh,
              "teacher_forced_replay_tokens": teacher_forced_replay_tokens}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== FINAL RESPONSE ===\n" + response)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
