#!/usr/bin/env python3
"""Append a spatial slice of the *existing full-image merger embeddings*.

The drawn rectangle selects tokens from the global image's already-computed
main-merger output. No original-image pixel crop is encoded at the trigger.
The selected embeddings enter a new multimodal turn and pass through the
decoder, which appends fresh KV while retaining all preceding reasoning KV.
This is an experiment, not a claim that slicing global features equals
re-encoding a real crop through the vision encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from continuous_cache_qwen3vl import (
    _append_turn_inputs, _content_inputs, _resize_to_pixel_cap, _segment_positions, _vl_backbone,
)
from interactive_roi_drawer import draw_rois
from live_crop_append_interactive import _forward_token, _prompt
from rolling_sparse_qwen3vl import _image_geometries, load_model


def select_merger_rectangles(merger: torch.Tensor, grid: tuple[int, int],
                             original_size: tuple[int, int], boxes: list[tuple[int, int, int, int]]):
    """Map original-pixel boxes to contiguous row-major merged-token grids."""
    height, width = map(int, grid)
    original_width, original_height = original_size
    if merger.ndim != 2 or merger.shape[0] != height * width:
        raise ValueError(f"Global merger tokens {tuple(merger.shape)} do not match grid {grid}.")
    yy = (torch.arange(height, dtype=torch.float64) + 0.5) * original_height / height
    xx = (torch.arange(width, dtype=torch.float64) + 0.5) * original_width / width
    feature_grid = merger.reshape(height, width, -1)
    selected, records = [], []
    for box in boxes:
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= original_width and 0 <= y1 < y2 <= original_height):
            raise ValueError(f"Invalid ROI {box} for original image {original_size}.")
        row_indices = torch.nonzero((yy >= y1) & (yy < y2)).flatten()
        col_indices = torch.nonzero((xx >= x1) & (xx < x2)).flatten()
        if not len(row_indices) or not len(col_indices):
            raise ValueError(f"ROI {box} contains no full-image merger token centers; draw a larger rectangle.")
        top, bottom = int(row_indices[0]), int(row_indices[-1]) + 1
        left, right = int(col_indices[0]), int(col_indices[-1]) + 1
        feature = feature_grid[top:bottom, left:right].reshape(-1, merger.shape[-1]).contiguous()
        rows = torch.arange(top, bottom)
        cols = torch.arange(left, right)
        global_indices = (rows[:, None] * width + cols[None, :]).flatten().tolist()
        selected.append(feature)
        records.append({
            "original_box": list(box), "global_grid_rectangle": [top, left, bottom, right],
            "selected_grid": [bottom - top, right - left], "selected_token_count": len(global_indices),
            "global_merger_token_indices": global_indices,
        })
    return selected, records


def build_roi_turn(processor, model, selected: list[torch.Tensor], records: list[dict],
                   instruction: str, device):
    """Create a normal image-token chat layout without encoding any real ROI pixels."""
    # Blank images provide only Qwen's chat/template scaffolding. Their pixel
    # values are discarded; each placeholder span is replaced with the actual
    # selected merger-token count and its corresponding 2-D grid.
    dummy_images = [Image.new("RGB", (64, 64)) for _ in selected]
    inputs = _append_turn_inputs(processor, dummy_images, instruction, device)
    dummy_spans, _ = _image_geometries(model, inputs)
    image_token_id = int(model.config.image_token_id)
    names = ("input_ids", "attention_mask", "mm_token_type_ids")
    for name in names:
        source = inputs[name][0]
        parts = []
        previous = 0
        for span, feature in zip(dummy_spans, selected):
            start, end = int(span[0]), int(span[-1]) + 1
            parts.append(source[previous:start])
            fill = image_token_id if name == "input_ids" else int(source[start])
            parts.append(source.new_full((feature.shape[0],), fill))
            previous = end
        parts.append(source[previous:])
        inputs[name] = torch.cat(parts).unsqueeze(0)
    merge = int(model.config.vision_config.spatial_merge_size)
    inputs["image_grid_thw"] = torch.tensor(
        [[1, row["selected_grid"][0] * merge, row["selected_grid"][1] * merge] for row in records],
        dtype=torch.long, device=device,
    )
    inputs.pop("pixel_values", None)
    spans, grids = _image_geometries(model, inputs)
    for span, feature in zip(spans, selected):
        if len(span) != feature.shape[0]:
            raise RuntimeError("Synthetic placeholder count does not match selected merger tokens.")
    return inputs, spans, grids


def append_merger_roi(model, processor, selected: list[torch.Tensor], records: list[dict],
                      instruction: str, past, cursor: int, device):
    inputs, local_spans, grids = build_roi_turn(processor, model, selected, records, instruction, device)
    position_ids = _segment_positions(model, inputs, cursor)
    previous_length = past.get_seq_length()
    full_mask = torch.ones((1, previous_length + inputs["input_ids"].shape[1]),
                           dtype=torch.long, device=device)
    # Qwen's decoder expects three DeepStack feature lists when image tokens
    # are present. Zeroing them isolates the already-computed main merger.
    deepstack_count = len(model.config.vision_config.deepstack_visual_indexes)
    visual = BaseModelOutputWithDeepstackFeatures(
        pooler_output=tuple(selected),
        deepstack_features=[tuple(torch.zeros_like(feature) for feature in selected)
                            for _ in range(deepstack_count)],
    )
    kwargs = dict(inputs)
    kwargs.pop("attention_mask", None)
    with torch.inference_mode():
        output = model(**kwargs, mm_encoder_outputs={"image": visual},
                       attention_mask=full_mask, position_ids=position_ids,
                       past_key_values=past, use_cache=True, return_dict=True)
    expected_length = previous_length + inputs["input_ids"].shape[1]
    if output.past_key_values.get_seq_length() != expected_length:
        raise RuntimeError("Decoder did not append the expected number of KV positions.")
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            int(position_ids.max().item()) + 1,
            [span + previous_length for span in local_spans], grids, inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--pause-after", action="append", default=None,
                        help="Generated phrase required before drawing the ROI. Can be repeated.")
    parser.add_argument("--roi", nargs=4, type=int, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                        help="Optional repeatable original-pixel bbox; omit to draw interactively.")
    parser.add_argument("--no-append", action="store_true",
                        help="Matched greedy control: keep generating without an ROI turn.")
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--crop-instruction", default="Continue the unfinished reasoning using this additional visual evidence.")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.global_max_pixels < 1 or args.max_new_tokens < 1:
        parser.error("--global-max-pixels and --max-new-tokens must be positive.")
    triggers = args.pause_after or ["Bakhmut"]

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    device = model.get_input_embeddings().weight.device
    original = Image.open(args.image).convert("RGB")
    submitted = _resize_to_pixel_cap(original, args.global_max_pixels)
    inputs = _content_inputs(processor, [submitted], _prompt(args.question), device, args.global_max_pixels)
    global_spans, global_grids = _image_geometries(model, inputs)
    backbone = _vl_backbone(model)
    with torch.inference_mode():
        global_features = backbone.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True)
    merger = global_features.pooler_output[0]
    if merger.shape[0] != len(global_spans[0]):
        raise RuntimeError("Full-image merger output and decoder image placeholders are misaligned.")
    initial_positions = _segment_positions(model, inputs, 0)
    initial_kwargs = dict(inputs)
    initial_kwargs.pop("pixel_values", None)
    with torch.inference_mode():
        output = model(**initial_kwargs, mm_encoder_outputs={"image": global_features},
                       position_ids=initial_positions, use_cache=True, return_dict=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(initial_positions.max().item()) + 1
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(x) for x in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    ids: list[int] = []
    insertion = None

    for _ in range(args.max_new_tokens):
        ids.append(int(pending.item()))
        text = processor.tokenizer.decode(ids, skip_special_tokens=True)
        if insertion is None and not args.no_append and all(phrase.casefold() in text.casefold() for phrase in triggers):
            trigger_token = len(ids)
            print("\n=== PAUSED AFTER TRIGGER: global + reasoning KV retained ===\n" + text)
            if args.roi:
                boxes = [tuple(box) for box in args.roi]
                print(f"\nUsing supplied original-pixel ROI(s): {boxes}")
            else:
                print(f"\nDraw ROI(s) on the original {original.width} x {original.height} image.")
                boxes = draw_rois(original)
            selected, records = select_merger_rectangles(merger, global_grids[0], original.size, boxes)
            _, past, cursor = _forward_token(model, pending, past, cursor)
            pending, past, cursor, spans, grids, turn = append_merger_roi(
                model, processor, selected, records, args.crop_instruction, past, cursor, device)
            insertion = {
                "trigger_token": trigger_token, "original_boxes": [list(box) for box in boxes],
                "selected_regions": records,
                "image_grid_thw": turn["image_grid_thw"].detach().cpu().tolist(),
                "append_merged_grids": [list(grid) for grid in grids],
                "appended_cache_key_positions": [span.detach().cpu().tolist() for span in spans],
                "cache_length_after_turn": int(past.get_seq_length()),
                "instruction": args.crop_instruction, "deepstack": "zeroed",
            }
            print(f"Appended {sum(feature.shape[0] for feature in selected)} existing merger tokens "
                  f"as {len(selected)} new image span(s); no ROI pixel crop or second ViT pass.\n")
            continue
        if int(pending.item()) in eos_ids:
            break
        pending, past, cursor = _forward_token(model, pending, past, cursor)

    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    result = {
        "image": str(args.image.resolve()), "question": args.question, "pause_after": triggers,
        "global_original_size": list(original.size), "global_submitted_size": list(submitted.size),
        "global_visual_grid": list(global_grids[0]), "global_merger_token_count": len(merger),
        "insertion": insertion, "response": response,
        "how": ("matched_greedy_no_append" if args.no_append else
                "live_append_global_merger_roi_via_decoder_no_roi_vit_no_anchor_refresh"),
        "backbone_precision": "nf4_4bit_bf16_compute" if args.load_in_4bit else "torch_dtype_auto",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== FINAL RESPONSE ===\n" + response)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
