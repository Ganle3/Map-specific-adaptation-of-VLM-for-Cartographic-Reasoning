#!/usr/bin/env python3
"""Minimal full-image ROI-logit-bias sweep for Bakhmut colour recognition.

This is intentionally *not* a LOOK/WHEN/WHERE rollout.  Each condition gets
one fresh full-image prefill and normal greedy Qwen3-VL generation.  The only
intervention is an additive decoder-attention logit bias on existing global
visual KV keys whose spatial centres fall inside the Bakhmut ROI.  No crop is
encoded, no user turn is appended, and no reasoning prefix is replayed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from continuous_cache_qwen3vl import (
    GlobalVisualKeyBias,
    _content_inputs,
    _generate_segment,
    _image_geometries,
    _resize_to_pixel_cap,
    _segment_positions,
)
from rolling_sparse_qwen3vl import load_model


DEFAULT_QUESTION = "What is the surrounding color of the black dot for Bakhmut?"
DEFAULT_LOGIT_PREFIX = "The surrounding color of Bakhmut's black dot is"


def roi_patch_mask(grid: tuple[int, int], original_size: tuple[int, int], box: list[int]) -> torch.Tensor:
    """Map an original-image box to the immutable full-image visual-token grid."""
    height, width = map(int, grid)
    image_width, image_height = original_size
    left, top, right, bottom = box
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError(f"Invalid ROI {box} for image size {original_size}.")
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) / height * image_height
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) / width * image_width
    mask = ((xs[None, :] >= left) & (xs[None, :] < right) &
            (ys[:, None] >= top) & (ys[:, None] < bottom)).flatten()
    if not mask.any():
        raise RuntimeError("ROI includes no visual-token centres; enlarge the bbox.")
    return mask


def template_bbox(image: Image.Image, template: Image.Image, scale: float, padding: float) -> tuple[list[int], float]:
    """Locate a JPEG detail template using normalized grayscale correlation.

    Matching is intentionally performed only once before model loading.  The
    reduced resolution keeps it inexpensive; the resulting coordinates are
    mapped back to the original image and enlarged by ``padding``.
    """
    if not 0 < scale <= 1:
        raise ValueError("--template-scale must be in (0, 1].")
    resized_image = image.convert("L").resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    resized_template = template.convert("L").resize(
        (max(2, round(template.width * scale)), max(2, round(template.height * scale))), Image.Resampling.BILINEAR)
    if resized_template.width >= resized_image.width or resized_template.height >= resized_image.height:
        raise ValueError("Template must be smaller than image after resizing.")

    # Normalized cross-correlation implemented with torch convolution.  This
    # tolerates a uniform brightness shift between the WebP and JPEG copies.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.tensor(list(resized_image.getdata()), dtype=torch.float32, device=device).view(1, 1, resized_image.height, resized_image.width)
    needle = torch.tensor(list(resized_template.getdata()), dtype=torch.float32, device=device).view(1, 1, resized_template.height, resized_template.width)
    needle = needle - needle.mean()
    needle_norm = needle.square().sum().sqrt().clamp_min(1e-6)
    ones = torch.ones_like(needle)
    count = float(needle.numel())
    numerator = F.conv2d(source, needle)
    local_sum = F.conv2d(source, ones)
    local_square_sum = F.conv2d(source.square(), ones)
    local_variance_sum = (local_square_sum - local_sum.square() / count).clamp_min(1e-6)
    score_map = numerator / (local_variance_sum.sqrt() * needle_norm)
    flat = int(score_map.flatten().argmax().item())
    row = flat // score_map.shape[-1]
    col = flat % score_map.shape[-1]
    score = float(score_map[0, 0, row, col].item())

    left = col / scale
    top = row / scale
    right = (col + resized_template.width) / scale
    bottom = (row + resized_template.height) / scale
    pad_x, pad_y = (right - left) * padding, (bottom - top) * padding
    box = [max(0, math.floor(left - pad_x)), max(0, math.floor(top - pad_y)),
           min(image.width, math.ceil(right + pad_x)), min(image.height, math.ceil(bottom + pad_y))]
    return box, score


def render_roi(image: Image.Image, box: list[int], destination: Path, label: str) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline="#ff9d00", width=max(3, image.width // 300))
    draw.text((box[0] + 4, max(2, box[1] - 24)), label, fill="#111111", stroke_width=2, stroke_fill="#ffffff")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def render_logit_curve(rows: list[dict], destination: Path) -> None:
    """Render the decisive log-probability margin (pink minus white)."""
    width, height, margin = 960, 480, 72
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    values = [row["pink_minus_white_logprob"] for row in rows]
    low, high = min(values + [0.0]), max(values + [0.0])
    if math.isclose(low, high):
        low, high = low - 1.0, high + 1.0
    pad = 0.10 * (high - low)
    low, high = low - pad, high + pad
    biases = [row["bias"] for row in rows]
    min_bias, max_bias = min(biases), max(biases)
    if math.isclose(min_bias, max_bias):
        min_bias, max_bias = min_bias - 1.0, max_bias + 1.0
    x = lambda value: margin + (value - min_bias) / (max_bias - min_bias) * (width - 2 * margin)
    y = lambda value: height - margin - (value - low) / (high - low) * (height - 2 * margin)
    draw.line((margin, margin, margin, height - margin), fill="#334155", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="#334155", width=2)
    draw.line((margin, y(0.0), width - margin, y(0.0)), fill="#94a3b8", width=2)
    points = [(x(row["bias"]), y(row["pink_minus_white_logprob"])) for row in rows]
    draw.line(points, fill="#c026d3", width=4)
    for row, (px, py) in zip(rows, points):
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#c026d3")
        draw.text((px - 12, height - margin + 12), f"{row['bias']:.2g}", fill="#334155")
    draw.text((margin, 18), "Fixed-prefix next-token evidence margin: log P(pink) − log P(white)", fill="#111827")
    draw.text((margin, height - 40), "ROI key-logit bias", fill="#334155")
    draw.text((margin + 6, y(0.0) - 18), "equal pink / white probability", fill="#64748b")
    draw.text((width - 270, 44), "positive → pink favoured", fill="#a21caf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def fixed_prefix_logit_sweep(model, processor, global_image: Image.Image, prompt: str,
                             global_positions: torch.Tensor, grid: tuple[int, int], mask: torch.Tensor,
                             biases: list[float], max_pixels: int, device: torch.device,
                             prefix: str) -> list[dict]:
    """Measure the *same* next-token decision for every ROI bias.

    This intentionally has no free reasoning continuation.  The assistant
    prefix is part of the initial multimodal prefill, so the requested visual
    reweighting is present when Qwen forms its hidden state for the colour
    token.  We score only the first BPE token of ``' pink'`` and ``' white'``;
    their token IDs are exported for audit.
    """
    pink_ids = processor.tokenizer.encode(" pink", add_special_tokens=False)
    white_ids = processor.tokenizer.encode(" white", add_special_tokens=False)
    if not pink_ids or not white_ids:
        raise RuntimeError("Qwen tokenizer returned an empty candidate encoding.")
    rows = []
    for bias in biases:
        inputs = _content_inputs(processor, [global_image], prompt, device, max_pixels)
        # The chat template already ends at the assistant header.  This is a
        # literal assistant prefix, not a quoted user instruction or replay.
        prefix_ids = processor.tokenizer.encode(prefix, add_special_tokens=False)
        append_ids = inputs["input_ids"].new_tensor(prefix_ids).unsqueeze(0)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], append_ids), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], inputs["attention_mask"].new_ones((1, append_ids.shape[1]))), dim=1)
        if "mm_token_type_ids" in inputs:
            inputs["mm_token_type_ids"] = torch.cat((inputs["mm_token_type_ids"], inputs["mm_token_type_ids"].new_zeros((1, append_ids.shape[1]))), dim=1)
        positions = _segment_positions(model, inputs, 0)
        full_mask = torch.ones((1, inputs["input_ids"].shape[1]), dtype=torch.long, device=device)
        weights = torch.zeros(global_positions.numel(), dtype=torch.float32, device=device)
        weights[mask] = float(bias)
        model_inputs = dict(inputs)
        model_inputs.pop("attention_mask", None)
        with GlobalVisualKeyBias(global_positions, weights, bool(bias)), torch.inference_mode():
            output = model(**model_inputs, attention_mask=full_mask, position_ids=positions,
                           use_cache=False, return_dict=True)
        logits = output.logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        pink_id, white_id = int(pink_ids[0]), int(white_ids[0])
        row = {"bias": float(bias), "equivalent_key_multiplier": math.exp(float(bias)),
               "pink_token_ids": pink_ids, "white_token_ids": white_ids,
               "pink_first_token": processor.tokenizer.decode([pink_id]),
               "white_first_token": processor.tokenizer.decode([white_id]),
               "pink_first_token_logit": float(logits[pink_id]),
               "white_first_token_logit": float(logits[white_id]),
               "pink_first_token_logprob": float(log_probs[pink_id]),
               "white_first_token_logprob": float(log_probs[white_id]),
               "pink_minus_white_logprob": float(log_probs[pink_id] - log_probs[white_id])}
        rows.append(row)
        print(f"bias={bias:.6g}: logP(pink)={row['pink_first_token_logprob']:.4f}, "
              f"logP(white)={row['white_first_token_logprob']:.4f}, "
              f"pink−white={row['pink_minus_white_logprob']:.4f}", flush=True)
    return rows


def load_nf4_model(model_name: str | None, adapter_path: Path):
    """Load the adapted NF4/bf16 model exactly as the attention viewer does."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if model_name is None:
        config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        model_name = config["base_model_name_or_path"]
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        model_name, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
        quantization_config=quantization, local_files_only=True,
    ).eval()
    return PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False).eval(), processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Use the attention-viewer NF4 weights with bf16 compute instead of the GRPO loader.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--template", type=Path,
                        help="Bakhmut detail JPEG used for automatic ROI matching.")
    parser.add_argument("--roi", type=int, nargs=4,
                        help="Manual original-image bbox LEFT TOP RIGHT BOTTOM; overrides --template.")
    parser.add_argument("--template-scale", type=float, default=0.5)
    parser.add_argument("--template-padding", type=float, default=0.20,
                        help="Fractional padding around automatic template bbox.")
    parser.add_argument("--biases", type=float, nargs="+", default=[0.0, 1.079442, 2.079442, 3.079442])
    parser.add_argument("--fixed-prefix-logit-diagnostic", action="store_true",
                        help="Do not freely generate. Measure pink/white next-token logits after a fixed assistant prefix.")
    parser.add_argument("--diagnostic-prefix", default=DEFAULT_LOGIT_PREFIX,
                        help="Literal assistant prefix for --fixed-prefix-logit-diagnostic.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.roi is None and args.template is None:
        parser.error("Specify either --template or --roi.")
    if args.roi is not None and args.template is not None:
        parser.error("Use one ROI source: --roi or --template.")
    if args.template_padding < 0:
        parser.error("--template-padding must be non-negative.")

    original = Image.open(args.image).convert("RGB")
    if args.roi is not None:
        box, match_score, roi_source = list(args.roi), None, "manual"
    else:
        box, match_score = template_bbox(original, Image.open(args.template).convert("RGB"),
                                         args.template_scale, args.template_padding)
        roi_source = "template_match"
    roi_preview = args.output.with_name(args.output.stem + "_roi.png")
    render_roi(original, box, roi_preview,
               f"Bakhmut ROI ({roi_source}{'' if match_score is None else f', NCC={match_score:.3f}'})")

    model, processor = (load_nf4_model(args.model, args.adapter) if args.load_in_4bit
                        else load_model(args.model, args.adapter))
    model.eval()
    device = model.get_input_embeddings().weight.device
    global_image = _resize_to_pixel_cap(original, args.global_max_pixels)
    # This is deliberately identical across bias conditions.  The question is
    # direct, so there is no assistant-prefix injection/replay path at all.
    prompt = ("Use only the supplied map image to answer the question. Inspect the local visual evidence "
              "carefully. Give a concise answer after thinking.\n\n"
              f"Question: {args.question}")
    reference_inputs = _content_inputs(processor, [global_image], prompt, device, args.global_max_pixels)
    global_positions, grids = _image_geometries(model, reference_inputs)
    mask = roi_patch_mask(grids[0], original.size, box).to(device)
    results = {"question": args.question, "image": str(args.image.resolve()), "roi_source": roi_source,
               "roi_original_pixels": box, "template": str(args.template) if args.template else None,
               "template_ncc": match_score, "roi_visual_token_count": int(mask.sum().item()),
               "global_visual_token_count": int(mask.numel()), "global_grid": list(map(int, grids[0])),
               "roi_preview": str(roi_preview.resolve()), "mechanism": "existing_global_visual_KV_roi_logit_bias_only",
               "model_precision_path": "nf4_4bit_bf16_compute" if args.load_in_4bit else "grpo_loader_torch_dtype_auto",
               "no_crop_reencoding": True, "no_crop_user_turn": True, "no_reasoning_replay": True,
               "conditions": []}
    if args.fixed_prefix_logit_diagnostic:
        rows = fixed_prefix_logit_sweep(model, processor, global_image, prompt, global_positions[0], grids[0], mask,
                                        args.biases, args.global_max_pixels, device, args.diagnostic_prefix)
        curve = args.output.with_name(args.output.stem + "_pink_vs_white_curve.png")
        render_logit_curve(rows, curve)
        results.update({"experiment": "fixed_assistant_prefix_next_token_logit_sweep",
                        "fixed_assistant_prefix": args.diagnostic_prefix,
                        "candidate_interpretation": "First BPE token of the literal continuations ' pink' and ' white'.",
                        "curve": str(curve.resolve()), "conditions": rows})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nCurve: {curve}\nResults: {args.output}")
        return
    for bias in args.biases:
        inputs = _content_inputs(processor, [global_image], prompt, device, args.global_max_pixels)
        weights = torch.zeros(global_positions[0].numel(), dtype=torch.float32, device=device)
        weights[mask] = bias
        text, _, _, _ = _generate_segment(model, processor, inputs, None, 0, global_positions[0],
                                           global_positions, grids, args.max_new_tokens, weights, 0.0, "attention")
        record = {"bias": float(bias), "equivalent_key_multiplier": math.exp(float(bias)), "output": text}
        results["conditions"].append(record)
        print(f"\n=== Bakhmut ROI bias {bias:.6g} (exp={math.exp(float(bias)):.2f}) ===\n{text}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nROI preview: {roi_preview}\nResults: {args.output}")


if __name__ == "__main__":
    main()
