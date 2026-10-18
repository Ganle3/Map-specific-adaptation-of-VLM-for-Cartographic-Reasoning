#!/usr/bin/env python3
"""Controlled Qwen3-VL ROI-KV attention-bias experiment.

Place beside continuous_cache_qwen3vl.py, dynamic_attention.py,
attention_policy.py, rolling_sparse_qwen3vl.py. Uses original model loader.
No training and no new image encoding in ROI-bias conditions.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from PIL import Image

from continuous_cache_qwen3vl import (
    GlobalVisualKeyBias,
    _content_inputs, _image_geometries, _generate_segment,
    _make_anchor_cache, _append_crop_then_assistant_replay,
    _resize_to_pixel_cap,
    _segment_positions,
)
from rolling_sparse_qwen3vl import load_model


def append_assistant_prefix(inputs, tokenizer, prefix):
    """Add unfinished assistant text after the chat template's assistant header."""
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if not ids:
        raise ValueError("The prefix must not be empty.")
    ids = torch.tensor([ids], device=inputs["input_ids"].device,
                       dtype=inputs["input_ids"].dtype)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
    for name in ("attention_mask", "mm_token_type_ids"):
        if name not in inputs:
            inputs[name] = torch.zeros_like(inputs["input_ids"][:, :-ids.shape[1]])
        fill = torch.ones_like(ids) if name == "attention_mask" else torch.zeros_like(ids)
        inputs[name] = torch.cat([inputs[name], fill], dim=1)
    return inputs


def roi_patch_mask(grid, original_size, box):
    """Map ORIGINAL image pixel bbox onto Qwen merged visual-token grid."""
    h, w = map(int, grid)
    ow, oh = original_size
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= ow and 0 <= y1 < y2 <= oh):
        raise ValueError(f"Invalid ROI {box} for image {(ow, oh)}")
    yy = (torch.arange(h, dtype=torch.float32) + 0.5) / h * oh
    xx = (torch.arange(w, dtype=torch.float32) + 0.5) / w * ow
    mask = ((xx[None, :] >= x1) & (xx[None, :] < x2)
            & (yy[:, None] >= y1) & (yy[:, None] < y2)).flatten()
    if not mask.any():
        raise RuntimeError("ROI contains no visual-token centers; enlarge ROI.")
    return mask


def recorded_look_prefix(tokenizer, jsonl_path: Path, rollout_index: int, look_number: int,
                         backoff_tokens: int = 0):
    """Recover the exact decoded-prefix boundary recorded by a cold rollout.

    JSONL stores decoded text rather than token IDs. Re-tokenising it is the
    closest recoverable representation; the recovered token count and selected
    boxes are written to the output for audit.
    """
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("This diagnostic expects a JSONL containing one question row.")
    runs = rows[0].get("rollouts", [])
    matching = [run for run in runs if int(run.get("rollout_index", -1)) == rollout_index]
    if len(matching) != 1:
        raise ValueError(f"Could not find rollout_index={rollout_index}.")
    events = matching[0].get("look_events", [])
    if not 1 <= look_number <= len(events):
        raise ValueError(f"look_number must be in 1..{len(events)} for rollout {rollout_index}.")
    event = events[look_number - 1]
    recorded_token = int(event["token"])
    boundary = recorded_token - backoff_tokens
    if boundary < 1:
        raise ValueError(f"prefix backoff {backoff_tokens} leaves no assistant token before LOOK {recorded_token}.")
    ids = tokenizer.encode(matching[0]["response"], add_special_tokens=False)
    if boundary > len(ids):
        raise RuntimeError(f"Recorded LOOK token {boundary} exceeds recovered response length {len(ids)}.")
    prefix_ids = ids[:boundary]
    return tokenizer.decode(prefix_ids, skip_special_tokens=True), event, len(ids), boundary


def generate(model, processor, inputs, positions, grids, biases, max_tokens,
             past=None, cursor=0):
    text, _, _, maps = _generate_segment(
        model, processor, inputs, past, cursor,
        positions,
        [positions],  # list[tensor]
        grids,
        max_tokens,
        biases,
        0.0,
        "attention",
    )
    return text, maps


def move_dynamic_cache(cache, device: torch.device):
    """Move DynamicCache K/V tensors without changing their sequence contents."""
    for layer in cache.layers:
        if getattr(layer, "is_initialized", False):
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            layer.device = torch.device(device)
    return cache


def replay_recorded_schedule(model, processor, initial, global_positions, grid, original_size,
                             events, roi_key_bias: float, token_budget: int,
                             recorded_response: str | None = None):
    """Generate with fixed recorded WHEN/WHERE events and no controller.

    At each recorded token index, the corresponding stored regions replace the
    active ROI. When ``recorded_response`` is supplied, the exact stored
    reasoning tokens up to that event replace the live prefix before refresh.
    Thus every ROI is tested against the actual cold-start reasoning context
    shown in its Markdown report, rather than a potentially divergent replay.
    """
    positions = _segment_positions(model, initial, 0)
    initial_mask = torch.ones((1, initial["input_ids"].shape[1]), device=global_positions.device, dtype=torch.long)
    prefill = dict(initial)
    prefill.pop("attention_mask", None)
    with torch.inference_mode():
        output = model(**prefill, attention_mask=initial_mask, position_ids=positions, use_cache=True, return_dict=True)
    # The strict replay needs an immutable anchor, but keeping two full global
    # caches on a 20-GB GPU can cause a silent driver OOM. Store that immutable
    # copy on CPU; it is moved back only after the mutable working cache is
    # released at a recorded LOOK. This changes no K/V values or positions.
    anchor_past = move_dynamic_cache(copy.deepcopy(output.past_key_values), torch.device("cpu"))
    anchor_cursor = int(positions.max().item()) + 1
    past, cursor = output.past_key_values, anchor_cursor
    pending = output.logits[:, -1].argmax(dim=-1)
    # Do not leave a third Python reference to the mutable working cache.
    # Keeping it would make the first anchor refresh hold three full global-KV
    # copies and can trigger a driver-level OOM without a Python traceback.
    del output
    token_ids = [int(pending.item())]
    active_bias = torch.zeros(global_positions.numel(), device=global_positions.device, dtype=torch.float32)
    events_by_token = {int(event["token"]): event for event in events}
    recorded_ids = (processor.tokenizer.encode(recorded_response, add_special_tokens=False)
                    if recorded_response is not None else None)
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    applied = []

    def refresh_from_anchor(biases, context_ids):
        refreshed = move_dynamic_cache(copy.deepcopy(anchor_past), global_positions.device)
        tokens = torch.tensor([context_ids], device=global_positions.device, dtype=torch.long)
        length = tokens.shape[1]
        mask = torch.ones((1, refreshed.get_seq_length() + length), device=global_positions.device, dtype=torch.long)
        text_positions = torch.arange(anchor_cursor, anchor_cursor + length, device=global_positions.device, dtype=torch.long)
        text_position_ids = text_positions.view(1, 1, length).expand(3, -1, -1)
        with GlobalVisualKeyBias(global_positions, biases, True), torch.inference_mode():
            refreshed_output = model(input_ids=tokens, attention_mask=mask, position_ids=text_position_ids,
                                     past_key_values=refreshed, use_cache=True, return_dict=True)
        return refreshed_output.logits[:, -1].argmax(dim=-1), refreshed_output.past_key_values, anchor_cursor + length

    token_index = 1
    while token_index < token_budget and int(pending.item()) not in eos_ids:
        event = events_by_token.get(token_index)
        if event is not None:
            if recorded_ids is not None:
                if token_index > len(recorded_ids):
                    raise RuntimeError(f"LOOK token {token_index} exceeds recovered stored reasoning length {len(recorded_ids)}.")
                # Exact recorded reasoning context at this LOOK, not the live
                # schedule replay that may have diverged after an earlier ROI.
                token_ids = recorded_ids[:token_index]
            selected = torch.zeros_like(active_bias)
            boxes = []
            for region in event["regions"]:
                box = [int(region[key]) for key in ("left", "top", "right", "bottom")]
                boxes.append(box)
                selected[roi_patch_mask(grid, original_size, box).to(selected.device)] = roi_key_bias
            # Replace the working cache exactly as the anchor-trace HOW does.
            # ``output`` from the preceding one-token forward aliases ``past``.
            # Release both before cloning the immutable anchor.
            if "output" in locals():
                del output
            del past
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            active_bias = selected
            print(f"[fixed schedule] LOOK at token {token_index}: {len(boxes)} ROI(s), "
                  f"{int((active_bias > 0).sum().item())} visual keys", flush=True)
            pending, past, cursor = refresh_from_anchor(active_bias, token_ids)
            applied.append({"token": token_index, "boxes": boxes,
                            "roi_visual_tokens": int((active_bias > 0).sum().item()),
                            "prefix_token_count": len(token_ids),
                            "recorded_prefix": processor.tokenizer.decode(token_ids, skip_special_tokens=True),
                            "first_token_after_roi_refresh": processor.tokenizer.decode(
                                [int(pending.item())], skip_special_tokens=True)})
        else:
            mask = torch.ones((1, past.get_seq_length() + 1), device=global_positions.device, dtype=torch.long)
            pos = torch.full((3, 1, 1), cursor, device=global_positions.device, dtype=torch.long)
            with GlobalVisualKeyBias(global_positions, active_bias, bool(active_bias.any())), torch.inference_mode():
                output = model(input_ids=pending.view(1, 1), attention_mask=mask, position_ids=pos,
                               past_key_values=past, use_cache=True, return_dict=True)
            pending, past, cursor = output.logits[:, -1].argmax(dim=-1), output.past_key_values, cursor + 1
        if int(pending.item()) in eos_ids:
            break
        token_ids.append(int(pending.item()))
        token_index += 1
    return {"continuation": processor.tokenizer.decode(token_ids, skip_special_tokens=True),
            "token_ids": token_ids, "terminated_by_eos": int(pending.item()) in eos_ids,
            "applied_looks": applied,
            "look_context_source": "recorded_rollout_reasoning" if recorded_ids is not None else "live_replay"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--question", required=True)
    prefix_source = ap.add_mutually_exclusive_group(required=True)
    prefix_source.add_argument("--prefix-file", type=Path,
                               help="UTF-8 file containing the exact assistant reasoning prefix.")
    prefix_source.add_argument("--rollout-jsonl", type=Path,
                               help="One-question cold-start JSONL from which to recover a LOOK prefix.")
    prefix_source.add_argument("--replay-schedule-jsonl", type=Path,
                               help="One-question cold-start JSONL: force its complete recorded LOOK schedule.")
    ap.add_argument("--rollout-index", type=int, default=None,
                    help="Required with --rollout-jsonl.")
    ap.add_argument("--look-number", type=int, default=None,
                    help="1-based LOOK event required with --rollout-jsonl.")
    ap.add_argument("--prefix-backoff-tokens", type=int, default=0,
                    help="Inject a recorded LOOK's fixed ROI this many reasoning tokens before its recorded boundary.")
    ap.add_argument("--roi", type=int, nargs=4, action="append", default=None,
                    help="One original-image ROI. Repeat --roi to reproduce all regions from one LOOK.")
    ap.add_argument("--prompt-mode", choices=("legacy", "cartographic"), default="legacy",
                    help="Use cartographic to exactly match the current cold-start prompt.")
    ap.add_argument("--biases", type=float, nargs="+", default=[0, 0.693147, 1.386294, 2.079442, 4.0])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--global-max-pixels", type=int, default=1_000_000)
    ap.add_argument("--with-highres-crop", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    model, processor = load_model(args.model, args.adapter)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    schedule_row = None
    if args.replay_schedule_jsonl is not None:
        if args.rollout_index is None:
            ap.error("--replay-schedule-jsonl requires --rollout-index.")
        rows = [json.loads(line) for line in args.replay_schedule_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 1:
            ap.error("--replay-schedule-jsonl must contain exactly one question row.")
        matching = [run for run in rows[0].get("rollouts", []) if int(run.get("rollout_index", -1)) == args.rollout_index]
        if len(matching) != 1:
            ap.error(f"Could not find rollout_index={args.rollout_index} in schedule JSONL.")
        schedule_row = matching[0]
        prefix, recorded_event, recovered_response_tokens, prefix_boundary = "", None, None, None
        boxes = None
    elif args.rollout_jsonl is not None:
        if args.rollout_index is None or args.look_number is None:
            ap.error("--rollout-jsonl requires --rollout-index and --look-number.")
        if args.prefix_backoff_tokens < 0:
            ap.error("--prefix-backoff-tokens must be non-negative.")
        prefix, recorded_event, recovered_response_tokens, prefix_boundary = recorded_look_prefix(
            processor.tokenizer, args.rollout_jsonl, args.rollout_index, args.look_number,
            args.prefix_backoff_tokens)
        boxes = [[int(region[key]) for key in ("left", "top", "right", "bottom")]
                 for region in recorded_event["regions"]]
    else:
        # Text fixtures conventionally end with a newline; that file-format
        # newline is not part of the intended assistant prefix.
        prefix = args.prefix_file.read_text(encoding="utf-8").rstrip("\r\n")
        recorded_event, recovered_response_tokens, prefix_boundary = None, None, None
        boxes = args.roi or [[544, 355, 800, 485]]
    if args.roi is not None:
        if schedule_row is not None:
            ap.error("--roi is incompatible with --replay-schedule-jsonl; schedule boxes are fixed from JSONL.")
        boxes = args.roi
    device = model.get_input_embeddings().weight.device

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
    global_image = _resize_to_pixel_cap(image, args.global_max_pixels)
    base_inputs = _content_inputs(processor, [global_image], prompt, device,
                                  args.global_max_pixels)
    global_local, grids = _image_geometries(model, base_inputs)
    positions = global_local[0]
    if schedule_row is not None:
        result = {
            "condition": "fixed_recorded_when_where_schedule",
            "prompt": prompt,
            "prompt_mode": args.prompt_mode,
            "source_jsonl": str(args.replay_schedule_jsonl),
            "rollout_index": args.rollout_index,
            "recorded_response": schedule_row["response"],
            "recorded_look_events": schedule_row["look_events"],
            "roi_key_bias": args.biases[-1] if len(args.biases) == 1 else args.biases,
        }
        # A schedule is an end-to-end WHEN/WHERE replay, so one nonzero bias
        # is required rather than the independent A/B prefix conditions.
        if len(args.biases) != 1 or args.biases[0] < 0:
            ap.error("--replay-schedule-jsonl requires exactly one non-negative --biases value.")
        replay = replay_recorded_schedule(model, processor, base_inputs, positions, grids[0], image.size,
                                          schedule_row["look_events"], args.biases[0], args.max_new_tokens,
                                          recorded_response=schedule_row["response"])
        result["replay"] = replay
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== FIXED RECORDED WHEN/WHERE SCHEDULE ===\n" + replay["continuation"])
        print(f"\nSaved: {args.output}")
        return

    mask = torch.zeros(positions.numel(), dtype=torch.bool)
    for box in boxes:
        mask |= roi_patch_mask(grids[0], image.size, box)
    assert mask.numel() == positions.numel(), (mask.numel(), positions.numel())

    results = {
        "roi_original_pixels": boxes,
        "global_grid": list(map(int, grids[0])),
        "roi_token_count": int(mask.sum()),
        "global_token_count": int(mask.numel()),
        "prefix": prefix,
        "prefix_source": ({"rollout_jsonl": str(args.rollout_jsonl), "rollout_index": args.rollout_index,
                           "look_number": args.look_number, "recorded_event": recorded_event,
                           "recorded_look_token": int(recorded_event["token"]),
                           "prefix_token_boundary": prefix_boundary,
                           "prefix_backoff_tokens": args.prefix_backoff_tokens,
                           "recovered_response_tokens": recovered_response_tokens}
                          if args.rollout_jsonl is not None else {"prefix_file": str(args.prefix_file)}),
        "prompt_mode": args.prompt_mode,
        "conditions": [],
    }

    for bias in args.biases:
        inputs = _content_inputs(processor, [global_image], prompt, device,
                                 args.global_max_pixels)
        inputs = append_assistant_prefix(inputs, processor.tokenizer, prefix)
        weights = torch.zeros(positions.numel(), device=positions.device,
                              dtype=torch.float32)
        weights[mask.to(weights.device)] = float(bias)
        with torch.inference_mode():
            text, _ = generate(model, processor, inputs, positions, grids,
                               weights, args.max_new_tokens)
        record = {"condition": "global_roi_bias", "bias": bias,
                  "equivalent_copies": math.exp(bias),
                  "continuation": text}
        results["conditions"].append(record)
        print(f"\n=== ROI bias={bias:.3f} ===\n{text}", flush=True)

    if args.with_highres_crop:
        if len(boxes) != 1:
            raise ValueError("--with-highres-crop supports one --roi only; it is not used for ROI-KV tests.")
        # Anchor + fresh high-resolution crop + exact same assistant prefix.
        # Unlike ROI-bias conditions, this reruns the ViT for the crop.
        x1,y1,x2,y2 = boxes[0]
        crop = image.crop((x1,y1,x2,y2))
        anchor_inputs = _content_inputs(processor, [global_image], prompt, device,
                                        args.global_max_pixels)
        with torch.inference_mode():
            anchor_past, anchor_cursor = _make_anchor_cache(model, anchor_inputs)
        crop_inputs = _append_crop_then_assistant_replay(
            processor, [crop],
            "Fresh visual evidence for the unfinished reasoning. Continue the same reasoning.",
            prefix, device, max_pixels=None)
        crop_local, crop_grids = _image_geometries(model, crop_inputs)
        crop_positions = crop_local[0] + anchor_past.get_seq_length()
        zero = torch.zeros_like(positions, dtype=torch.float32)
        # Global original image and new crop can both be attended to.
        with torch.inference_mode():
            text, _, _, _ = _generate_segment(
                model, processor, crop_inputs, anchor_past, anchor_cursor,
                positions, [positions, crop_positions],
                [grids[0], crop_grids[0]],
                args.max_new_tokens, zero, 0.0, "attention")
        results["conditions"].append({
            "condition": "new_highres_crop_anchor_replay", "continuation": text})
        print(f"\n=== HIGHRES CROP ===\n{text}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
