#!/usr/bin/env python3
"""Greedily stop Qwen3-VL at a manual ROI intervention boundary.

This is phase 1 of the manual ROI-only experiment.  It never changes visual
attention.  Once every requested trigger phrase is visible in the generated
assistant text, it writes that exact decoded prefix for phase 2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap, _segment_positions
from rolling_sparse_qwen3vl import load_model


def cartographic_prompt(question: str) -> str:
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--stop-after", action="append", required=True,
                        help="Case-insensitive phrase required in the emitted reasoning. Repeat to require several phrases.")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--prefix-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    image = _resize_to_pixel_cap(original, args.global_max_pixels)
    prompt = cartographic_prompt(args.question)
    inputs = _content_inputs(processor, [image], prompt, device, args.global_max_pixels)
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True, return_dict=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(positions.max().item()) + 1
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    token_ids: list[int] = []
    found = False
    for _ in range(args.max_new_tokens):
        token_ids.append(int(pending.item()))
        text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        if all(phrase.casefold() in text.casefold() for phrase in args.stop_after):
            found = True
            break
        if int(pending.item()) in eos_ids:
            break
        mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=device)
        text_positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(input_ids=pending.view(1, 1), attention_mask=mask, position_ids=text_positions,
                           past_key_values=past, use_cache=True, return_dict=True)
        pending, past, cursor = output.logits[:, -1].argmax(dim=-1), output.past_key_values, cursor + 1

    prefix = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    args.prefix_output.parent.mkdir(parents=True, exist_ok=True)
    args.prefix_output.write_text(prefix, encoding="utf-8")
    report = {"image": str(args.image.resolve()), "question": args.question, "triggers": args.stop_after,
              "trigger_found": found, "prefix_tokens": len(token_ids), "prefix": prefix,
              "backbone_precision": "nf4_4bit_bf16_compute" if args.load_in_4bit else "torch_dtype_auto"}
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(prefix)
    print(f"\nTrigger found: {found}; tokens: {len(token_ids)}")
    print(f"Prefix: {args.prefix_output}")


if __name__ == "__main__":
    main()
