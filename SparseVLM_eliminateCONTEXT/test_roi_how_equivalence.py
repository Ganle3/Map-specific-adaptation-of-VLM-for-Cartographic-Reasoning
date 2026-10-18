#!/usr/bin/env python3
"""Verify ROI-HOW equivalence: joint prefix prefill vs anchor + forced replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import GlobalVisualKeyBias, _content_inputs, _resize_to_pixel_cap, _segment_positions
from rolling_sparse_qwen3vl import _image_geometries, load_model
from test_roi_kv_bias_bakhmut import append_assistant_prefix, roi_patch_mask


def prompt(question: str) -> str:
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--prefix-file", type=Path, required=True)
    p.add_argument("--roi", type=int, nargs=4, action="append", required=True)
    p.add_argument("--roi-key-bias", type=float, required=True)
    p.add_argument("--global-max-pixels", type=int, default=1_000_000)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model, processor = load_model(None, a.adapter); model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(a.image).convert("RGB")
    image = _resize_to_pixel_cap(original, a.global_max_pixels)
    text = prompt(a.question)
    prefix = a.prefix_file.read_text(encoding="utf-8").rstrip("\r\n")
    base = _content_inputs(processor, [image], text, device, a.global_max_pixels)
    global_positions, grids = _image_geometries(model, base)
    positions = global_positions[0]
    union = torch.zeros(positions.numel(), dtype=torch.bool)
    for box in a.roi:
        union |= roi_patch_mask(grids[0], original.size, box)
    bias = torch.zeros(positions.numel(), device=device, dtype=torch.float32)
    bias[union.to(device)] = a.roi_key_bias

    # A: one joint forward, identical to test_roi_kv_bias_bakhmut.py.
    joint = append_assistant_prefix(_content_inputs(processor, [image], text, device, a.global_max_pixels),
                                   processor.tokenizer, prefix)
    joint_pos = _segment_positions(model, joint, 0)
    with GlobalVisualKeyBias(positions, bias, True), torch.inference_mode():
        j = model(**joint, position_ids=joint_pos, use_cache=True, return_dict=True)

    # B: current cold-start HOW mechanics: anchor first *without* an ROI
    # bias, then teacher-force the assistant prefix under the ROI bias.
    base_pos = _segment_positions(model, base, 0)
    with torch.inference_mode():
        anchor = model(**base, position_ids=base_pos, use_cache=True, return_dict=True)
    ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
    forced = torch.tensor([ids], device=device, dtype=base["input_ids"].dtype)
    cursor = int(base_pos.max().item()) + 1
    forced_pos = torch.arange(cursor, cursor + forced.shape[1], device=device, dtype=torch.long).view(1, 1, -1).expand(3, -1, -1)
    mask = torch.ones((1, anchor.past_key_values.get_seq_length() + forced.shape[1]), device=device, dtype=torch.long)
    with GlobalVisualKeyBias(positions, bias, True), torch.inference_mode():
        s = model(input_ids=forced, attention_mask=mask, position_ids=forced_pos,
                  past_key_values=anchor.past_key_values, use_cache=True, return_dict=True)

    # C: split cache execution with the ROI bias also active while building the
    # full-image anchor.  This isolates timing from cache splitting: if C=A,
    # the only causal difference between A and B is the ungated anchor prefill.
    with GlobalVisualKeyBias(positions, bias, True), torch.inference_mode():
        biased_anchor = model(**base, position_ids=base_pos, use_cache=True, return_dict=True)
    biased_mask = torch.ones(
        (1, biased_anchor.past_key_values.get_seq_length() + forced.shape[1]),
        device=device, dtype=torch.long,
    )
    with GlobalVisualKeyBias(positions, bias, True), torch.inference_mode():
        c = model(input_ids=forced, attention_mask=biased_mask, position_ids=forced_pos,
                  past_key_values=biased_anchor.past_key_values, use_cache=True, return_dict=True)

    # ``position_ids`` is [rope_axis, batch, sequence].  Slice the *sequence*
    # axis explicitly; the old two-index expression accidentally sliced the
    # singleton batch axis and compared the whole joint prompt with the replay.
    joint_prefix_pos = joint_pos[:, :, -forced.shape[1]:]
    result = {
        "prefix_token_count": forced.shape[1], "roi_visual_tokens": int(union.sum().item()),
        "roi_visual_token_indices": torch.nonzero(union, as_tuple=False).flatten().tolist(),
        "roi_cache_key_positions": positions[union.to(positions.device)].detach().cpu().tolist(),
        "joint_vs_split_prefix_positions_equal": bool(torch.equal(joint_prefix_pos, forced_pos)),
        "base_last_position": base_pos[:, 0, -1].tolist(),
        "base_position_max": int(base_pos.max().item()),
        "joint_prefix_first_position": joint_prefix_pos[:, 0, 0].tolist(),
        "joint_prefix_last_position": joint_prefix_pos[:, 0, -1].tolist(),
        "split_prefix_first_position": forced_pos[:, 0, 0].tolist(),
        "split_prefix_last_position": forced_pos[:, 0, -1].tolist(),
        "max_abs_logit_difference": float((j.logits[:, -1] - s.logits[:, -1]).abs().max().item()),
        "max_abs_logit_difference_biased_anchor_split": float((j.logits[:, -1] - c.logits[:, -1]).abs().max().item()),
        "joint_next_token": processor.tokenizer.decode([int(j.logits[:, -1].argmax().item())]),
        "split_next_token": processor.tokenizer.decode([int(s.logits[:, -1].argmax().item())]),
        "biased_anchor_split_next_token": processor.tokenizer.decode([int(c.logits[:, -1].argmax().item())]),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
