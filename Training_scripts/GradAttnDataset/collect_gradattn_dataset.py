#!/usr/bin/env python3
"""Collect frozen-Qwen3-VL Grad-Att supervision records from map questions.

The language model and its LoRA adapter are frozen.  This collector never
injects crops back into the active generation: crops are *labels/evidence for a
later teacher*, extracted from the original resolution image.  Every selected
reasoning token receives a decoder hidden state; a step with no localized
Grad-Att proposal records an empty crop list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT / "SparseVLM_eliminateCONTEXT"
LOOKBETTER = ROOT / "Training_scripts" / "GRPO_lookbetter"
for folder in (str(Path(__file__).resolve().parent), str(SPARSE), str(LOOKBETTER)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from attention_policy import salient_regions
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap, _segment_positions, load_model
from data import build_cartographic_prompt, evaluate_answer, load_mapverse_questions
from rolling_sparse_qwen3vl import PixelRegion, _image_geometries

from grad_attribution import DecoderGradAttentionCapture


@dataclass(frozen=True)
class CropSettings:
    attention_mass: float
    max_crops: int
    min_concentration: float
    min_local_contrast: float
    min_patches: int
    padding_patches: int
    max_area_fraction: float
    min_original_side: int
    merge_iou: float
    reuse_iou: float
    min_color_entropy: float
    min_edge_density: float


def _detach_cache(cache):
    """Cut a completed Grad-Att graph while retaining the numerical KV cache."""
    for layer in cache.layers:
        if getattr(layer, "is_initialized", False):
            layer.keys = layer.keys.detach()
            layer.values = layer.values.detach()
    return cache


def _enable_input_grads(model):
    """Make frozen-NF4 forwards differentiable with respect to attention maps."""
    enable = getattr(model, "enable_input_require_grads", None)
    if enable is None:
        enable = getattr(getattr(model, "base_model", None), "enable_input_require_grads", None)
    if enable is None:
        raise RuntimeError("This Transformers/PEFT model does not expose enable_input_require_grads().")
    enable()


def _normalised_entropy(scores: np.ndarray) -> tuple[float, float]:
    flat = np.maximum(np.asarray(scores, dtype=np.float64).reshape(-1), 0.0)
    total = float(flat.sum())
    if total <= np.finfo(np.float64).tiny:
        return 0.0, 0.0
    p = flat / total
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(max(2, p.size)))
    return 1.0 - entropy, float(np.sort(p)[-max(1, math.ceil(p.size * 0.10)):].sum())


def _expand_component(component, grid: tuple[int, int], cfg: CropSettings):
    """Pad a Grad-Att component and make it a legal, non-degenerate crop."""
    height, width = grid
    min_side = max(2, cfg.min_patches)
    cy = (component.top + component.bottom) / 2
    cx = (component.left + component.right) / 2
    patch_h = max(min_side, component.bottom - component.top + 2 * cfg.padding_patches)
    patch_w = max(min_side, component.right - component.left + 2 * cfg.padding_patches)
    max_area = max(1, int(math.floor(height * width * cfg.max_area_fraction)))
    while patch_h * patch_w > max_area and (patch_h > min_side or patch_w > min_side):
        if patch_h >= patch_w and patch_h > min_side:
            patch_h -= 1
        elif patch_w > min_side:
            patch_w -= 1
        else:
            break
    if patch_h * patch_w > max_area:
        return None
    top = max(0, min(height - patch_h, int(round(cy - patch_h / 2))))
    left = max(0, min(width - patch_w, int(round(cx - patch_w / 2))))
    return top, left, top + patch_h, left + patch_w


def _project_region(box, grid: tuple[int, int], original: Image.Image, score: float, cfg: CropSettings):
    top, left, bottom, right = box
    grid_h, grid_w = grid
    x0 = int(math.floor(left * original.width / grid_w))
    x1 = int(math.ceil(right * original.width / grid_w))
    y0 = int(math.floor(top * original.height / grid_h))
    y1 = int(math.ceil(bottom * original.height / grid_h))
    min_side = min(cfg.min_original_side, original.width, original.height)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = max(0, min(original.width - min_side, int(round(cx - min_side / 2))))
    y0 = max(0, min(original.height - min_side, int(round(cy - min_side / 2))))
    x1, y1 = max(x0 + min_side, x1), max(y0 + min_side, y1)
    x1, y1 = min(original.width, x1), min(original.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return PixelRegion(x0, y0, x1, y1, float(score))


def _iou(a: PixelRegion, b: PixelRegion) -> float:
    x0, y0 = max(a.left, b.left), max(a.top, b.top)
    x1, y1 = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = ((a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter)
    return inter / union if union else 0.0


def _merge_overlapping_regions(regions: list[PixelRegion], merge_iou: float) -> list[PixelRegion]:
    """Merge substantially overlapping proposals from one token's Grad-Att map.

    This is intentionally scoped to a *single* token step.  A later token may
    discover a related but larger region, but retroactively replacing an older
    crop with that union would leak future visual evidence into the old teacher
    state.
    """
    pending = sorted(regions, key=lambda region: region.score, reverse=True)
    merged: list[PixelRegion] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            remainder = []
            for other in pending:
                if _iou(current, other) > merge_iou:
                    current = PixelRegion(
                        min(current.left, other.left), min(current.top, other.top),
                        max(current.right, other.right), max(current.bottom, other.bottom),
                        current.score + other.score,
                    )
                    changed = True
                else:
                    remainder.append(other)
            pending = remainder
        merged.append(current)
    return sorted(merged, key=lambda region: region.score, reverse=True)


def _visual_detail_metrics(image: Image.Image, region: PixelRegion) -> dict:
    """Measure whether an original-pixel crop has usable local visual detail.

    The crop is only downsampled for this inexpensive *diagnostic*.  The image
    later saved for the teacher remains the exact original-resolution crop.
    Colour entropy catches uniform fills; edge density preserves labels, dots,
    borders, and other sparse but useful evidence on a largely uniform region.
    """
    crop = image.crop((region.left, region.top, region.right, region.bottom))
    scale = min(1.0, 256.0 / max(crop.width, crop.height))
    if scale < 1.0:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.BILINEAR)
    rgb = np.asarray(crop, dtype=np.uint8)
    # 16 bins/channel is deliberately coarse: JPG/WebP noise in a truly flat
    # fill should not turn it into a high-entropy crop.
    bins = (rgb >> 4).astype(np.int32)
    codes = bins[..., 0] * 256 + bins[..., 1] * 16 + bins[..., 2]
    counts = np.bincount(codes.reshape(-1), minlength=4096).astype(np.float64)
    probability = counts[counts > 0] / counts.sum()
    color_entropy = float(-(probability * np.log(probability)).sum() / np.log(4096))
    luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    # A map label/boundary has a sharp luminance transition even if it covers
    # only a small fraction of a mostly flat cartographic color.
    horizontal = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    vertical = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    edge_density = float((horizontal + vertical >= 24.0).mean())
    return {"color_entropy": color_entropy, "edge_density": edge_density,
            "detail_probe_size": [int(crop.width), int(crop.height)]}


def _has_visual_detail(metrics: dict, cfg: CropSettings) -> bool:
    # Retain the crop if *either* colour diversity or sparse structural detail
    # is present. Requiring both would reject text on a uniform control area.
    return (metrics["color_entropy"] >= cfg.min_color_entropy
            or metrics["edge_density"] >= cfg.min_edge_density)


def propose_crops(scores: np.ndarray, grid: tuple[int, int], original: Image.Image, cfg: CropSettings):
    """Return localizable Grad-Att crop regions plus auditable gating metrics."""
    concentration, top10_mass = _normalised_entropy(scores)
    metrics = {"concentration": concentration, "top10_mass": top10_mass, "saliency_sum": float(scores.sum())}
    if scores.sum() <= np.finfo(np.float32).tiny:
        return [], metrics | {"crop_decision": "no_positive_gradatt"}
    if concentration < cfg.min_concentration:
        return [], metrics | {"crop_decision": "too_diffuse"}
    regions = []
    for component in salient_regions(scores, cfg.max_crops * 4, cfg.attention_mass, min_patches=1):
        box = _expand_component(component, grid, cfg)
        if box is None:
            continue
        top, left, bottom, right = box
        area_fraction = (bottom - top) * (right - left) / (grid[0] * grid[1])
        mass = float(scores[top:bottom, left:right].sum())
        contrast = mass / max(area_fraction, 1e-12)
        if contrast < cfg.min_local_contrast:
            continue
        region = _project_region(box, grid, original, mass, cfg)
        if region is not None and all(_iou(region, old) < 0.98 for old in regions):
            regions.append(region)
        # Keep a small surplus before IoU union; otherwise a high-ranked pair
        # could consume the quota and hide a distinct third evidence region.
        if len(regions) == cfg.max_crops * 4:
            break
    regions = _merge_overlapping_regions(regions, cfg.merge_iou)
    return regions[:cfg.max_crops], metrics | {"crop_decision": "localized" if regions else "no_legal_component"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def _quantize_int8(tensor: torch.Tensor):
    value = tensor.detach().to("cpu", dtype=torch.float32)
    scale = value.abs().max().clamp_min(1e-8) / 127.0
    return torch.round(value / scale).clamp(-127, 127).to(torch.int8), float(scale.item())


def save_global_visual_kv(cache, positions, destination: Path, metadata: dict, mode: str):
    """Store global-image keys/values once, never prompt/reasoning KV entries."""
    if mode == "none":
        return None
    if destination.exists():
        return destination
    payload = {"metadata": metadata | {"format": mode, "global_key_positions": positions.cpu()}}
    for index, layer in enumerate(cache.layers):
        if not getattr(layer, "is_initialized", False):
            continue
        key = layer.keys.index_select(-2, positions.to(layer.keys.device))
        value = layer.values.index_select(-2, positions.to(layer.values.device))
        if mode == "fp16":
            payload[f"layer_{index:02d}_key"] = key.detach().to("cpu", dtype=torch.float16)
            payload[f"layer_{index:02d}_value"] = value.detach().to("cpu", dtype=torch.float16)
        elif mode == "int8":
            qk, sk = _quantize_int8(key)
            qv, sv = _quantize_int8(value)
            payload[f"layer_{index:02d}_key"] = qk
            payload[f"layer_{index:02d}_key_scale"] = sk
            payload[f"layer_{index:02d}_value"] = qv
            payload[f"layer_{index:02d}_value_scale"] = sv
        else:
            raise ValueError(f"Unknown global KV mode: {mode}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _continuation_forward(model, token, past, cursor: int, visual_positions, grid, args):
    """Commit one emitted token, obtaining its hidden state and Grad-Att map."""
    device = token.device
    mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=device)
    positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    with DecoderGradAttentionCapture(model, visual_positions, grid, layer_start=args.grad_layer_start) as capture:
        output = model(input_ids=token.view(1, 1), attention_mask=mask, position_ids=positions,
                       past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        next_token = output.logits[:, -1].argmax(dim=-1)
        objective = torch.log_softmax(output.logits[:, -1].float(), dim=-1)[0, int(next_token.item())]
        objective.backward()
        layers, pooled = capture.maps(args.grad_aggregation)
        hidden = output.hidden_states[-1][0, -1].detach().to("cpu", dtype=torch.bfloat16)
        next_past = _detach_cache(output.past_key_values)
    model.zero_grad(set_to_none=True)
    return next_token.detach(), next_past, layers, pooled, hidden


def collect_one(model, processor, item, args, output_root: Path):
    device = model.get_input_embeddings().weight.device
    original = Image.open(item.image_path).convert("RGB")
    submitted = _resize_to_pixel_cap(original, args.global_max_pixels)
    prompt = build_cartographic_prompt(item.question)
    inputs = _content_inputs(processor, [submitted], prompt, device, args.global_max_pixels)
    visual_spans, grids = _image_geometries(model, inputs)
    if len(visual_spans) != 1:
        raise RuntimeError("Expected exactly one global image span.")
    visual_positions, grid = visual_spans[0], grids[0]
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        prefill = model(**inputs, position_ids=positions, use_cache=True, return_dict=True)
    image_hash = _file_hash(item.image_path)
    kv_file = output_root / "global_visual_kv" / f"{image_hash}_{grid[0]}x{grid[1]}_{args.global_kv_mode}.pt"
    saved_kv = save_global_visual_kv(
        prefill.past_key_values, visual_positions, kv_file,
        {"image_sha256_prefix": image_hash, "source_image": str(item.image_path),
         "original_size": [original.width, original.height], "submitted_size": [submitted.width, submitted.height],
         "image_grid": list(grid), "model": args.model or "adapter_base", "adapter": str(args.adapter)},
        args.global_kv_mode,
    )
    past = prefill.past_key_values
    cursor = int(positions.max().item()) + 1
    pending = prefill.logits[:, -1].argmax(dim=-1)
    del prefill

    settings = CropSettings(args.attention_mass, args.max_crops_per_step, args.min_concentration,
                            args.min_local_contrast, args.min_patches, args.padding_patches,
                            args.max_area_fraction, args.min_original_side, args.merge_iou, args.reuse_iou,
                            args.min_crop_color_entropy, args.min_crop_edge_density)
    all_ids, steps, hiddens, maps, crop_history = [], [], [], [], []
    thinking_closed = False
    eos_ids = model.generation_config.eos_token_id
    eos_ids = {int(x) for x in eos_ids} if isinstance(eos_ids, (list, tuple, set)) else {int(eos_ids)}
    for token_index in range(args.max_generation_tokens):
        if int(pending.item()) in eos_ids:
            break
        all_ids.append(int(pending.item()))
        collect_grad = token_index < args.grad_token_limit and token_index % args.grad_stride == 0 and not thinking_closed
        if collect_grad:
            next_token, past, layers, score_map, hidden = _continuation_forward(
                model, pending, past, cursor, visual_positions, grid, args
            )
            regions, metrics = propose_crops(score_map, grid, original, settings)
            accepted_regions, rejected_regions = [], []
            for region in regions:
                detail = _visual_detail_metrics(original, region)
                if _has_visual_detail(detail, settings):
                    accepted_regions.append(region)
                else:
                    rejected_regions.append(asdict(region) | detail | {"reason": "low_visual_detail"})
            # A single proposed background region means this token has no
            # useful local target. With multiple proposals, however, reject
            # only the background candidates and keep the independently useful
            # evidence regions. This avoids turning one bad secondary proposal
            # into a false empty label.
            if rejected_regions and (
                args.detail_filter_policy == "all_candidates"
                or (args.detail_filter_policy == "singleton_empty" and len(accepted_regions) + len(rejected_regions) == 1)
            ):
                regions = []
                metrics["crop_decision"] = "token_rejected_single_low_detail_candidate"
            else:
                regions = accepted_regions
            metrics["filtered_low_detail_regions"] = rejected_regions
            metrics["pre_detail_filter_crop_count"] = len(accepted_regions) + len(rejected_regions)
            metrics["accepted_crop_count"] = len(regions)
            saved_regions = []
            for rank, region in enumerate(regions):
                prior = next((entry for entry in crop_history if _iou(region, entry["region"]) >= args.reuse_iou), None)
                if prior is None:
                    crop_path = output_root / "crops" / f"{item.sample_id}_{image_hash}" / f"token_{token_index:04d}_r{rank:02d}.png"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    original.crop((region.left, region.top, region.right, region.bottom)).save(crop_path)
                    crop_history.append({"region": region, "path": crop_path})
                else:
                    crop_path = prior["path"]
                saved_regions.append(asdict(region) | {"path": _relative(crop_path, output_root)})
            steps.append({
                "token_index": token_index, "token_id": int(pending.item()),
                "token_text": processor.tokenizer.decode([int(pending.item())], skip_special_tokens=False),
                "metrics": metrics, "crops": saved_regions,
            })
            hiddens.append(hidden)
            if args.store_grad_maps:
                maps.append(score_map)
        else:
            with torch.inference_mode():
                mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=device)
                position_ids = torch.full((3, 1, 1), cursor, dtype=torch.long, device=device)
                output = model(input_ids=pending.view(1, 1), attention_mask=mask, position_ids=position_ids,
                               past_key_values=past, use_cache=True, return_dict=True)
            next_token, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
        pending = next_token
        cursor += 1
        decoded = processor.tokenizer.decode(all_ids, skip_special_tokens=False)
        thinking_closed = "</think>" in decoded

    response = processor.tokenizer.decode(all_ids, skip_special_tokens=True)
    sample_dir = output_root / "steps"
    sample_dir.mkdir(parents=True, exist_ok=True)
    state_file = sample_dir / f"{item.sample_id}_{image_hash}.pt"
    torch.save({
        "hidden_states_bf16": torch.stack(hiddens) if hiddens else torch.empty((0, 0), dtype=torch.bfloat16),
        "token_indices": torch.tensor([step["token_index"] for step in steps], dtype=torch.int32),
        "token_ids": torch.tensor([step["token_id"] for step in steps], dtype=torch.int64),
        "image_grid": torch.tensor(grid, dtype=torch.int32),
    }, state_file)
    map_file = None
    if args.store_grad_maps:
        map_dir = output_root / "grad_maps"
        map_dir.mkdir(parents=True, exist_ok=True)
        map_file = map_dir / f"{item.sample_id}_{image_hash}.npz"
        np.savez_compressed(map_file, maps=np.stack(maps) if maps else np.empty((0, *grid), np.float32),
                            token_indices=np.asarray([step["token_index"] for step in steps], dtype=np.int32))
    return {
        "sample_id": item.sample_id, "dataset": item.metadata.get("dataset", "mapverse"), "image": str(item.image_path),
        "question": item.question, "gold_answer": item.answer, "answer_type": item.answer_type,
        "response": response, "evaluation": evaluate_answer(response, item),
        "global_image": {"sha256_prefix": image_hash, "original_size": [original.width, original.height],
                         "submitted_size": [submitted.width, submitted.height], "grid": list(grid),
                         "visual_kv": _relative(saved_kv, output_root) if saved_kv else None},
        "hidden_state_file": _relative(state_file, output_root),
        "grad_map_file": _relative(map_file, output_root) if map_file else None,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=1, help="1-based inclusive dataset index.")
    parser.add_argument("--end-index", type=int, default=None, help="1-based inclusive dataset index.")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000)
    parser.add_argument("--max-generation-tokens", type=int, default=800)
    parser.add_argument("--grad-token-limit", type=int, default=100)
    parser.add_argument("--grad-stride", type=int, default=1)
    parser.add_argument("--grad-layer-start", type=int, default=0, help="Use layers >= this index; 0 means all layers.")
    parser.add_argument("--grad-aggregation", choices=("last", "mean", "max"), default="mean")
    parser.add_argument("--global-kv-mode", choices=("none", "int8", "fp16"), default="int8")
    parser.add_argument("--store-grad-maps", action="store_true")
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--max-crops-per-step", type=int, default=2)
    parser.add_argument("--min-concentration", type=float, default=0.12)
    parser.add_argument("--min-local-contrast", type=float, default=2.0)
    parser.add_argument("--min-patches", type=int, default=2)
    parser.add_argument("--padding-patches", type=int, default=1)
    parser.add_argument("--max-area-fraction", type=float, default=0.25)
    parser.add_argument("--min-original-side", type=int, default=64)
    parser.add_argument("--merge-iou", type=float, default=0.50,
                        help="Merge overlapping crop proposals from the same token step above this IoU.")
    parser.add_argument("--reuse-iou", type=float, default=0.90)
    parser.add_argument("--min-crop-color-entropy", type=float, default=0.06,
                        help="Reject a crop only when both its colour entropy and edge density are below their gates.")
    parser.add_argument("--min-crop-edge-density", type=float, default=0.012,
                        help="Fraction of detail-probe pixels with a strong luma edge required to rescue a low-entropy crop.")
    parser.add_argument("--detail-filter-policy", choices=("singleton_empty", "all_candidates", "per_crop"), default="singleton_empty",
                        help="singleton_empty (default) empties a token only when its sole proposal is low-detail; per_crop keeps other valid proposals.")
    args = parser.parse_args()
    if args.grad_stride < 1 or args.grad_token_limit < 0 or args.max_generation_tokens < 1:
        parser.error("Generation and Grad-Att token limits must be positive; --grad-stride must be >= 1.")
    if not 0 < args.merge_iou <= 1 or not 0 < args.reuse_iou <= 1:
        parser.error("--merge-iou and --reuse-iou must be in (0, 1].")
    if not 0 <= args.min_crop_color_entropy <= 1 or not 0 <= args.min_crop_edge_density <= 1:
        parser.error("Crop entropy and edge-density gates must be in [0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_mapverse_questions(args.dataset_json, args.mapverse_image_root, mapwise_root=args.mapwise_root)
    end = args.end_index or len(records)
    if not (1 <= args.start_index <= end <= len(records)):
        parser.error(f"Invalid inclusive range [{args.start_index}, {end}] for {len(records)} records.")
    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    _enable_input_grads(model)
    manifest = args.output_dir / "manifest.jsonl"
    config = vars(args).copy()
    config["dataset_json"] = str(args.dataset_json.resolve())
    config["mapverse_image_root"] = str(args.mapverse_image_root.resolve())
    config["mapwise_root"] = str(args.mapwise_root.resolve())
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    with manifest.open("a", encoding="utf-8") as handle:
        for dataset_index in range(args.start_index, end + 1):
            item = records[dataset_index - 1]
            row = collect_one(model, processor, item, args, args.output_dir)
            row["dataset_index"] = dataset_index
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{dataset_index}/{end}] {item.sample_id}: steps={len(row['steps'])} "
                  f"crops={sum(len(step['crops']) for step in row['steps'])} reward={row['evaluation']['reward']}")


if __name__ == "__main__":
    main()
