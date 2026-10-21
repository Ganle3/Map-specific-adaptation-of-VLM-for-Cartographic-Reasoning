#!/usr/bin/env python3
"""Append a real ViT crop to live Qwen3-VL KV after a generated trigger.

This is a causal HOW experiment, not a rolling-window procedure.  It generates
normally from one full map, pauses only after the trigger token is emitted,
commits that token to the existing cache, and appends a fresh crop as a new
multimodal user turn.  Existing global visual KV and reasoning KV remain in
place; no anchor refresh or assistant-prefix replay occurs.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import (
    _append_turn_inputs, _content_inputs, _resize_to_pixel_cap, _segment_positions,
)
from interactive_roi_drawer import draw_rois
from rolling_sparse_qwen3vl import _image_geometries, load_model


@contextmanager
def _suppress_crop_deepstack(model):
    """Keep the live crop ViT/merger path, but zero its DeepStack decoder input.

    This hook is active only for the crop append forward.  The original full
    image prefill, all earlier reasoning KV, and later text-only forwards keep
    their normal behaviour.  Three zero tensors preserve the expected visual
    placeholder mask and decoder call protocol.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    modules = [module for module in model.modules() if isinstance(module, Qwen3VLTextModel)]
    if len(modules) != 1:
        raise RuntimeError(f"Expected one Qwen3-VL text decoder, found {len(modules)}")

    def before_decoder(_module, args, kwargs):
        features = kwargs.get("deepstack_visual_embeds")
        if features is None:
            raise RuntimeError("Crop append did not supply DeepStack features to the decoder")
        kwargs["deepstack_visual_embeds"] = [torch.zeros_like(feature) for feature in features]
        return args, kwargs

    handle = modules[0].register_forward_pre_hook(before_decoder, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def _prompt(question: str) -> str:
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


def _forward_token(model, token, past, cursor: int):
    """Commit one already-emitted assistant token to cache and predict the next."""
    mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=token.device)
    positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=token.device)
    with torch.inference_mode():
        output = model(input_ids=token.view(1, 1), attention_mask=mask, position_ids=positions,
                       past_key_values=past, use_cache=True, return_dict=True)
    return output.logits[:, -1].argmax(dim=-1), output.past_key_values, cursor + 1


def _append_crop(model, processor, crops: list[Image.Image], instruction: str, past, cursor: int, device,
                 max_pixels: int, no_crop_deepstack: bool = False):
    """Encode crops through Qwen ViT/merger, then append their chat segment to ``past``."""
    inputs = _append_turn_inputs(processor, crops, instruction, device, max_pixels=max_pixels)
    local_positions = _segment_positions(model, inputs, cursor)
    prior = past.get_seq_length()
    full_mask = torch.ones((1, prior + inputs["input_ids"].shape[1]), dtype=torch.long, device=device)
    kwargs = dict(inputs)
    kwargs.pop("attention_mask", None)
    context = _suppress_crop_deepstack(model) if no_crop_deepstack else nullcontext()
    with context, torch.inference_mode():
        output = model(**kwargs, attention_mask=full_mask, position_ids=local_positions,
                       past_key_values=past, use_cache=True, return_dict=True)
    crop_local, crop_grids = _image_geometries(model, inputs)
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            int(local_positions.max().item()) + 1,
            [span + prior for span in crop_local], crop_grids, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pause-after", action="append", default=["Bakhmut"],
                        help="Generated case-insensitive phrase required before crop insertion.")
    parser.add_argument("--crop-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-instruction", default="Continue the unfinished reasoning using this additional visual evidence.",
                        help="Minimal text accompanying the appended crop; it must not encode task-specific evidence.")
    parser.add_argument("--no-crop-deepstack", action="store_true",
                        help="At the live crop append only, zero the three ViT DeepStack streams before the decoder; keep the main merger output.")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    global_image = _resize_to_pixel_cap(original, args.global_max_pixels)
    inputs = _content_inputs(processor, [global_image], _prompt(args.question), device, args.global_max_pixels)
    global_local, global_grids = _image_geometries(model, inputs)
    initial_positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=initial_positions, use_cache=True, return_dict=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(initial_positions.max().item()) + 1
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    ids: list[int] = []
    inserted = False
    boxes: list[tuple[int, int, int, int]] = []
    trigger_token = None
    crop_record = None

    for _ in range(args.max_new_tokens):
        ids.append(int(pending.item()))
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        if not inserted and all(phrase.casefold() in text.casefold() for phrase in args.pause_after):
            trigger_token = len(ids)
            print("\n\n=== PAUSED AFTER TRIGGER: live global + reasoning KV retained ===")
            print(text)
            print(f"\nDraw one or more evidence crop(s) on the original {original.width} x {original.height} image.")
            boxes = draw_rois(original)
            # The just-emitted trigger token belongs to the assistant prefix;
            # commit it before opening a later user crop turn.
            _, past, cursor = _forward_token(model, pending, past, cursor)
            crops = [_resize_to_pixel_cap(original.crop(box), max(1, args.crop_max_pixels // len(boxes))) for box in boxes]
            pending, past, cursor, crop_positions, crop_grids, crop_inputs = _append_crop(
                model, processor, crops, args.crop_instruction, past, cursor, device,
                max(1, args.crop_max_pixels // len(boxes)), args.no_crop_deepstack)
            crop_record = {
                "original_boxes": boxes,
                "source_sizes": [[box[2] - box[0], box[3] - box[1]] for box in boxes],
                "submitted_sizes": [[crop.width, crop.height] for crop in crops],
                "image_grid_thw": crop_inputs["image_grid_thw"].detach().cpu().tolist(),
                "merged_grids": [list(map(int, grid)) for grid in crop_grids],
                "crop_cache_key_positions": [span.detach().cpu().tolist() for span in crop_positions],
                "cache_length_after_crop_turn": int(past.get_seq_length()),
                "instruction": args.crop_instruction,
                "crop_deepstack": "zeroed" if args.no_crop_deepstack else "native",
            }
            inserted = True
            print(f"Inserted {len(crops)} ViT+merger crop(s); existing full-map/reasoning KV was retained.\n")
            # pending now predicts the first new assistant token after the
            # crop turn; do not consume it in this iteration.
            continue
        if int(pending.item()) in eos_ids:
            break
        pending, past, cursor = _forward_token(model, pending, past, cursor)

    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    result = {
        "image": str(args.image.resolve()), "question": args.question,
        "pause_after": args.pause_after, "trigger_token": trigger_token,
        "pre_crop_reasoning": processor.tokenizer.decode(ids[:trigger_token] if trigger_token else ids, skip_special_tokens=True),
        "crop": crop_record, "response": response,
        "how": "live_kv_append_real_vit_merger_crop_no_anchor_refresh_no_assistant_replay",
        "global_visual_grid": list(map(int, global_grids[0])),
        "backbone_precision": "nf4_4bit_bf16_compute" if args.load_in_4bit else "torch_dtype_auto",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== FINAL RESPONSE ===\n" + response)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
