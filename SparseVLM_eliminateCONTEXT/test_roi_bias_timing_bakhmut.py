#!/usr/bin/env python3
"""Isolate *when* an ROI-KV logit bias begins to affect Qwen3-VL.

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


def roi_mask(grid: tuple[int, int], original_size: tuple[int, int], roi: list[int], device: torch.device):
    """Use the same merged-token-centre mapping as the ROI cold-start path."""
    height, width = map(int, grid)
    original_width, original_height = original_size
    left, top, right, bottom = roi
    if not (0 <= left < right <= original_width and 0 <= top < bottom <= original_height):
        raise ValueError(f"Invalid ROI {roi} for original image {original_size}.")
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * original_height
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * original_width
    chosen = ((xx[None, :] >= left) & (xx[None, :] < right)
              & (yy[:, None] >= top) & (yy[:, None] < bottom)).flatten()
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--prefix-file", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, default=[544, 355, 800, 485])
    parser.add_argument("--roi-key-bias", type=float, default=2.079442)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, processor = load_model(args.model, args.adapter)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    prefix = args.prefix_file.read_text(encoding="utf-8")
    prompt = (f"Inspect the complete map carefully. Question: {args.question}\n"
              "Reason step by step. Do not give a final answer until explicitly asked.")
    image = _resize_to_pixel_cap(original, args.global_max_pixels)

    base = _content_inputs(processor, [image], prompt, device, args.global_max_pixels)
    visual_positions, grids = _image_geometries(model, base)
    global_positions, grid = visual_positions[0], grids[0]
    selected = roi_mask(grid, original.size, args.roi, device)
    biases = torch.zeros(global_positions.numel(), device=device, dtype=torch.float32)
    biases[selected] = args.roi_key_bias

    # A: exactly the old ROI test semantics. The hook covers both prefix prefill
    # and all generated tokens because _generate_segment calls model.generate.
    conditioned_inputs = append_assistant_prefix(
        _content_inputs(processor, [image], prompt, device, args.global_max_pixels), processor.tokenizer, prefix)
    with torch.inference_mode():
        early_text, _, _, _ = _generate_segment(
            model, processor, conditioned_inputs, None, 0, global_positions,
            [global_positions], [grid], args.max_new_tokens, biases, 0.0, "attention")

    # B: same prefix input IDs and same ROI keys; only its already-cached
    # prefix representation was built with no ROI bias.
    late_inputs = append_assistant_prefix(
        _content_inputs(processor, [image], prompt, device, args.global_max_pixels), processor.tokenizer, prefix)
    late = late_bias_continuation(model, processor, late_inputs, global_positions, grid, biases, args.max_new_tokens)

    result = {
        "design": {
            "A": "ROI bias active during assistant-prefix prefill and continuation",
            "B": "assistant prefix prefilled ungated; ROI bias active only after the prefix cache exists",
        },
        "prompt": prompt,
        "roi_original_pixels": args.roi,
        "global_grid": list(map(int, grid)),
        "roi_visual_tokens": int(selected.sum().item()),
        "global_visual_tokens": int(selected.numel()),
        "roi_key_bias": args.roi_key_bias,
        "equivalent_key_multiplier": math.exp(args.roi_key_bias),
        "prefix": prefix,
        "A_bias_during_prefix_prefill": {"continuation": early_text},
        "B_bias_after_prefix_cache": late,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== A: bias during prefix prefill ===\n" + early_text)
    print("\n=== B: bias after prefix cache ===\n" + late["continuation"])
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
