#!/usr/bin/env python3
"""Compare late ROI-only bias against an ROI-biased anchor-trace refresh.

This deliberately bypasses WHEN/WHERE.  Both conditions use the same image,
question, assistant prefix, ROI and greedy decoding.  The only intervention is
whether the ROI bias is active while that prefix is prefixed (A), or starts
only after its KV cache has already been built (B).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import (
    GlobalVisualKeyBias,
    _content_inputs,
    _generate_segment,
    _image_geometries,
    _resize_to_pixel_cap,
    _segment_positions,
)
from rolling_sparse_qwen3vl import load_model


def append_assistant_prefix(inputs, tokenizer, prefix: str):
    """Append literal unfinished assistant tokens after the chat header."""
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not ids:
        raise ValueError("The prefix file contains no tokenizer-visible text.")
    ids = torch.tensor([ids], device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
    for name in ("attention_mask", "mm_token_type_ids"):
        if name not in inputs:
            inputs[name] = torch.zeros_like(inputs["input_ids"][:, :-ids.shape[1]])
        extension = torch.ones_like(ids) if name == "attention_mask" else torch.zeros_like(ids)
        inputs[name] = torch.cat([inputs[name], extension], dim=1)
    return inputs


def roi_mask(grid: tuple[int, int], original_size: tuple[int, int], rois: list[list[int]], device: torch.device):
    """Use the same merged-token-centre mapping as the ROI cold-start path."""
    height, width = map(int, grid)
    original_width, original_height = original_size
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * original_height
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * original_width
    chosen = torch.zeros((height, width), device=device, dtype=torch.bool)
    for roi in rois:
        left, top, right, bottom = roi
        if not (0 <= left < right <= original_width and 0 <= top < bottom <= original_height):
            raise ValueError(f"Invalid ROI {roi} for original image {original_size}.")
        chosen |= ((xx[None, :] >= left) & (xx[None, :] < right)
                   & (yy[:, None] >= top) & (yy[:, None] < bottom))
    chosen = chosen.flatten()
    if not chosen.any():
        raise RuntimeError("ROI selects zero merged visual tokens; enlarge the box.")
    return chosen


def late_bias_continuation(model, processor, inputs, global_positions, grid, biases, budget: int):
    """Build prefix KV ungated, then enable ROI bias only for later queries."""
    positions = _segment_positions(model, inputs, 0)
    local_length = inputs["input_ids"].shape[1]
    mask = torch.ones((1, local_length), device=inputs["input_ids"].device, dtype=torch.long)
    prefill_inputs = dict(inputs)
    prefill_inputs.pop("attention_mask", None)
    with torch.inference_mode():
        output = model(**prefill_inputs, attention_mask=mask, position_ids=positions, use_cache=True, return_dict=True)

    past = output.past_key_values
    cursor = int(positions.max().item()) + 1
    token = output.logits[:, -1].argmax(dim=-1)
    token_ids: list[int] = []
    eos = processor.tokenizer.eos_token_id

    # The first token is predicted from the ungated prefix.  Every following
    # prediction is made after the immediately preceding token has queried the
    # biased visual keys.  This is the causal analogue of a mid-trajectory LOOK.
    for _ in range(budget):
        token_ids.append(int(token.item()))
        if int(token.item()) == eos:
            break
        attention_mask = torch.ones((1, past.get_seq_length() + 1), device=token.device, dtype=torch.long)
        position_ids = torch.full((3, 1, 1), cursor, device=token.device, dtype=torch.long)
        with GlobalVisualKeyBias(global_positions, biases, True), torch.inference_mode():
            output = model(input_ids=token.view(1, 1), attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=past,
                           use_cache=True, return_dict=True)
        past = output.past_key_values
        cursor += 1
        token = output.logits[:, -1].argmax(dim=-1)
    return {
        "continuation": processor.tokenizer.decode(token_ids, skip_special_tokens=True),
        "token_ids": token_ids,
        "terminated_by_eos": bool(token_ids and token_ids[-1] == eos),
        "prefix_cache_length": local_length,
    }


def refreshed_anchor_trace_continuation(model, processor, anchor_inputs, prefix: str,
                                        global_positions, biases, budget: int):
    """Exact current HOW: biased anchor prefill + forced assistant-trace replay.

    This deliberately follows ``rollout._refresh_trace_from_anchor`` rather
    than putting prefix tokens in the multimodal prefill.  It is therefore the
    direct counterpart of a successful cold-start LOOK with revision_tokens=0.
    """
    ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
    if not ids:
        raise ValueError("The prefix file contains no tokenizer-visible text.")
    positions = _segment_positions(model, anchor_inputs, 0)
    with GlobalVisualKeyBias(global_positions, biases, True), torch.inference_mode():
        anchor = model(**anchor_inputs, position_ids=positions, use_cache=True, return_dict=True)
    cursor = int(positions.max().item()) + 1
    forced = torch.tensor([ids], device=global_positions.device, dtype=anchor_inputs["input_ids"].dtype)
    mask = torch.ones((1, anchor.past_key_values.get_seq_length() + forced.shape[1]),
                      device=global_positions.device, dtype=torch.long)
    forced_positions = torch.arange(cursor, cursor + forced.shape[1], device=global_positions.device,
                                   dtype=torch.long).view(1, 1, -1).expand(3, -1, -1)
    with GlobalVisualKeyBias(global_positions, biases, True), torch.inference_mode():
        output = model(input_ids=forced, attention_mask=mask, position_ids=forced_positions,
                       past_key_values=anchor.past_key_values, use_cache=True, return_dict=True)
    past, cursor = output.past_key_values, cursor + forced.shape[1]
    token = output.logits[:, -1].argmax(dim=-1)
    token_ids: list[int] = []
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    for _ in range(budget):
        token_ids.append(int(token.item()))
        if int(token.item()) in eos_ids:
            break
        attention_mask = torch.ones((1, past.get_seq_length() + 1), device=token.device, dtype=torch.long)
        position_ids = torch.full((3, 1, 1), cursor, device=token.device, dtype=torch.long)
        with GlobalVisualKeyBias(global_positions, biases, True), torch.inference_mode():
            output = model(input_ids=token.view(1, 1), attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=past, use_cache=True, return_dict=True)
        past, cursor = output.past_key_values, cursor + 1
        token = output.logits[:, -1].argmax(dim=-1)
    return {"continuation": processor.tokenizer.decode(token_ids, skip_special_tokens=True),
            "token_ids": token_ids, "terminated_by_eos": bool(token_ids and token_ids[-1] in eos_ids),
            "forced_replay_tokens": len(ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix-file", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, action="append", required=True,
                        help="Original-image ROI. Repeat for each hand-drawn evidence region.")
    parser.add_argument("--roi-key-bias", type=float, default=2.079442)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--prompt-mode", choices=("cartographic", "legacy"), default="cartographic",
                        help="cartographic exactly matches the controller rollout prompt.")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit",
                        help="Use legacy non-NF4 loading; default matches Debug_GRPO QLoRA.")
    parser.add_argument("--conditions", nargs="+", choices=("roi_bias_only", "anchor_trace_refresh"),
                        default=("roi_bias_only", "anchor_trace_refresh"),
                        help="Which HOW interventions to execute.")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    prefix = args.prefix_file.read_text(encoding="utf-8")
    if args.prompt_mode == "cartographic":
        prompt = ("This is a cartographic reasoning question.\n\n"
                  "Use only the supplied map image to answer the question. Carefully inspect the\n"
                  "map legend, labels, colors, boundaries, spatial relationships, and other\n"
                  "relevant visual information.\n\n"
                  f"Question:\n{args.question}\n\n"
                  "Provide your answer on a separate line using exactly this format:\n"
                  "Final answer: <answer>")
    else:
        prompt = (f"Inspect the complete map carefully. Question: {args.question}\n"
                  "Reason step by step. Do not give a final answer until explicitly asked.")
    image = _resize_to_pixel_cap(original, args.global_max_pixels)

    base = _content_inputs(processor, [image], prompt, device, args.global_max_pixels)
    visual_positions, grids = _image_geometries(model, base)
    global_positions, grid = visual_positions[0], grids[0]
    selected = roi_mask(grid, original.size, args.roi, device)
    biases = torch.zeros(global_positions.numel(), device=device, dtype=torch.float32)
    biases[selected] = args.roi_key_bias

    refreshed = None
    if "anchor_trace_refresh" in args.conditions:
        refreshed = refreshed_anchor_trace_continuation(
            model, processor, _content_inputs(processor, [image], prompt, device, args.global_max_pixels), prefix,
            global_positions, biases, args.max_new_tokens)

    # Same prefix input IDs and same ROI keys; only its already-cached prefix
    # representation was built with no ROI bias.
    late = None
    if "roi_bias_only" in args.conditions:
        late_inputs = append_assistant_prefix(
            _content_inputs(processor, [image], prompt, device, args.global_max_pixels), processor.tokenizer, prefix)
        late = late_bias_continuation(model, processor, late_inputs, global_positions, grid, biases, args.max_new_tokens)

    result = {
        "design": {
        "roi_bias_only": "assistant prefix prefilled ungated; ROI bias starts only after its KV cache exists",
        "anchor_trace_refresh": "ROI bias active while recomputing the full-image anchor and assistant-prefix KV, then during continuation",
        },
        "prompt": prompt,
        "roi_original_pixels": args.roi,
        "global_grid": list(map(int, grid)),
        "roi_visual_tokens": int(selected.sum().item()),
        "global_visual_tokens": int(selected.numel()),
        "roi_key_bias": args.roi_key_bias,
        "equivalent_key_multiplier": math.exp(args.roi_key_bias),
        "prefix": prefix,
        "roi_bias_only": late,
        "anchor_trace_refresh": refreshed,
        "prompt_mode": args.prompt_mode,
        "backbone_precision": "nf4_4bit_bf16_compute" if args.load_in_4bit else "torch_dtype_auto",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if late is not None:
        print("\n=== ROI bias only (after existing prefix KV) ===\n" + late["continuation"])
    if refreshed is not None:
        print("\n=== Anchor-trace refresh (ROI bias during anchor + prefix KV rebuild) ===\n" + refreshed["continuation"])
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
