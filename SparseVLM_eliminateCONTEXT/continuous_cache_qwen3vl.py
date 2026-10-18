#!/usr/bin/env python3
"""Experimental continuous-KV rolling crops for Qwen3-VL.

Unlike ``rolling_sparse_qwen3vl.py``, this keeps the initial full-image and
generated-reasoning cache alive.  Every later window appends a new user turn
containing crop images.  The optional global-key bias is applied only to the
initial full-image key positions; it never removes them from the cache.

This prototype targets the Qwen3-VL Transformers implementation installed in
the VisionGRPO environment.  It uses explicit M-RoPE positions for appended
multimodal turns and commits the final generated token into the cache before
the next turn is appended.
"""
from __future__ import annotations

import argparse
import copy
import json
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from attention_policy import aggregate_decoder_layers
from dynamic_attention import MultiImageAttentionCapture
from rolling_sparse_qwen3vl import (
    PixelRegion,
    _extract_final_answer,
    _image_geometries,
    _refine_regions,
    crop_region,
    load_model,
    render_window_crops,
    RollingConfig,
    _update_evidence_ledger,
)


@dataclass(frozen=True)
class ContinuousConfig:
    windows: int = 6
    window_tokens: int = 80
    final_window_tokens: int = 128
    regions: int = 4
    attention_mass: float = 0.60
    aggregation: str = "mean"
    visual_score: str = "attention"
    # Default: every next window is selected afresh in original-image
    # coordinates by the ungated global scout, including Window 1.
    full_reselect_from_window: int = 0
    full_reselect_interval: int = 1
    global_bias_from_window: int = 3
    global_bias_interval: int = 1
    global_key_bias: float = -4.0
    global_retain_attention_mass: float = 0.60
    min_crop_global_patches: int = 2
    ledger_tokens: int = 128
    replay_tail_tokens: int = 0
    crop_key_bias: float = 0.0
    global_scout_tokens: int = 24
    bootstrap_windows: int = 2
    bootstrap_grid_rows: int = 3
    bootstrap_grid_cols: int = 3
    novelty_iou: float = 0.35
    global_max_pixels: int = 1_000_000
    crop_total_max_pixels: int = 1_000_000


class GlobalVisualKeyBias(AbstractContextManager):
    """Bias only initial-global visual keys; text/reasoning keys are untouched."""

    def __init__(self, global_positions, biases, enabled: bool, crop_positions=None, crop_key_bias: float = 0.0,
                 capture=None):
        self.global_positions = global_positions
        self.biases = biases
        self.enabled = enabled
        self.crop_positions = crop_positions
        self.crop_key_bias = crop_key_bias
        self.capture = capture
        self.module = None
        self.original = None

    def __enter__(self):
        if not self.enabled:
            return self
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen

        self.module, self.original = qwen, qwen.eager_attention_forward
        positions = self.global_positions
        original = self.original

        def biased_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            # Only decoder text attention has keys spanning the full language cache.
            if isinstance(module, qwen.Qwen3VLTextAttention) and key.shape[-2] > int(positions.max()):
                penalty = query.new_zeros((1, 1, 1, key.shape[-2]))
                penalty[..., positions.to(key.device)] = self.biases.to(key.device, key.dtype)
                # Current crop keys alone receive this positive bias.  It does
                # not affect text/reasoning keys or cached global keys.
                if self.crop_positions is not None and self.crop_key_bias:
                    crop = self.crop_positions.to(key.device)
                    crop = crop[crop < key.shape[-2]]
                    penalty[..., crop] += self.crop_key_bias
                attention_mask = penalty if attention_mask is None else attention_mask + penalty
            result = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if self.capture is not None and isinstance(module, qwen.Qwen3VLTextAttention):
                self.capture.record_eager(module.layer_idx, module, result[1], value)
            return result

        qwen.eager_attention_forward = biased_eager_attention
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.module.eager_attention_forward = self.original


def _resize_to_pixel_cap(image: Image.Image, max_pixels: int) -> Image.Image:
    """Downscale only, preserving source coordinates and aspect ratio."""
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive.")
    source_pixels = image.width * image.height
    if source_pixels <= max_pixels:
        return image
    scale = (max_pixels / source_pixels) ** 0.5
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _crop_images_with_budget(original: Image.Image, regions: list[PixelRegion], total_max_pixels: int):
    """Crop in original coordinates, then split a visual-pixel budget evenly.

    Each local crop still covers exactly its original bounding box.  Resizing
    happens only after that crop is made, so selecting regions and rendering
    overlays remain in the original-map coordinate system.
    """
    if not regions:
        return [], []
    per_crop_cap = max(1, total_max_pixels // len(regions))
    images, metadata = [], []
    for region in regions:
        source = crop_region(original, region)
        resized = _resize_to_pixel_cap(source, per_crop_cap)
        images.append(resized)
        metadata.append({
            "source_size": [source.width, source.height],
            "submitted_size": [resized.width, resized.height],
            "source_pixels": source.width * source.height,
            "submitted_pixels": resized.width * resized.height,
            "per_crop_max_pixels": per_crop_cap,
        })
    return images, metadata


def _content_inputs(processor, images: list[Image.Image], text: str, device, max_pixels: int | None = None):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    # Transformers 5.19 routes multimodal preprocessing kwargs through
    # ``processor_kwargs``.  Passing max_pixels directly is both deprecated
    # and ignored by apply_chat_template in newer releases.
    if max_pixels is None:
        kwargs = {}
    else:
        size = dict(processor.image_processor.size)
        size["longest_edge"] = max_pixels
        kwargs = {"processor_kwargs": {"images_kwargs": {"size": size}}}
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt", **kwargs,
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


def _append_turn_inputs(processor, images: list[Image.Image], text: str, device, max_pixels: int | None = None):
    """Build only the tokens that follow an unfinished assistant response."""
    standalone = _content_inputs(processor, images, text, device, max_pixels)
    tokenizer = processor.tokenizer
    user_prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    transition = tokenizer.encode("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    prefix_len = len(user_prefix)
    if standalone["input_ids"][0, :prefix_len].tolist() != user_prefix:
        raise RuntimeError("Unexpected Qwen chat template; cannot safely append a cache turn.")
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key not in standalone:
            continue
        tail = standalone[key][:, prefix_len:]
        if key == "input_ids":
            head = tail.new_tensor(transition).unsqueeze(0)
        elif key == "attention_mask":
            head = tail.new_ones((1, len(transition)))
        else:
            head = tail.new_zeros((1, len(transition)))
        standalone[key] = __import__("torch").cat((head, tail), dim=1)
    # Text-only scout turns may not receive this field from the processor,
    # while Qwen's M-RoPE helper expects it unconditionally.
    if "mm_token_type_ids" not in standalone:
        standalone["mm_token_type_ids"] = standalone["input_ids"].new_zeros(standalone["input_ids"].shape)
    return standalone


def _append_crop_then_assistant_replay(processor, images: list[Image.Image], instruction: str,
                                       assistant_prefix: str, device, max_pixels: int | None = None):
    """Append crops in a user turn, then replay text as actual assistant tokens.

    The earlier version put the trace inside a user XML block.  Qwen therefore
    treated it as a quoted document and began a fresh answer.  Here the chat
    template has already opened the next assistant turn; appending the trace
    after that header makes the next generated token a literal continuation of
    the replayed assistant reasoning, while the crop precedes it causally.
    """
    import torch

    inputs = _append_turn_inputs(processor, images, instruction, device, max_pixels)
    replay = inputs["input_ids"].new_tensor(
        processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)
    ).unsqueeze(0)
    if replay.shape[1] == 0:
        return inputs
    inputs["input_ids"] = torch.cat((inputs["input_ids"], replay), dim=1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], inputs["attention_mask"].new_ones((1, replay.shape[1]))), dim=1
    )
    if "mm_token_type_ids" in inputs:
        inputs["mm_token_type_ids"] = torch.cat(
            (inputs["mm_token_type_ids"], inputs["mm_token_type_ids"].new_zeros((1, replay.shape[1]))), dim=1
        )
    return inputs


def _vl_backbone(model):
    """Return the Qwen3VLModel beneath optional PEFT and LM wrappers."""
    candidate = model
    # PeftModel.get_base_model() returns Qwen3VLForConditionalGeneration;
    # its `.model` is then the Qwen3VLModel that owns get_rope_index().
    if hasattr(candidate, "get_base_model"):
        candidate = candidate.get_base_model()
    for _ in range(4):
        if hasattr(candidate, "get_rope_index"):
            return candidate
        if not hasattr(candidate, "model"):
            break
        next_candidate = candidate.model
        if next_candidate is candidate:
            break
        candidate = next_candidate
    raise AttributeError("Could not locate Qwen3VLModel.get_rope_index beneath the loaded model wrapper.")


def _segment_positions(model, inputs, cursor: int):
    """Compute local Qwen 3D positions then shift them after the cached sequence."""
    position_ids, _ = _vl_backbone(model).get_rope_index(
        inputs["input_ids"], inputs["mm_token_type_ids"],
        image_grid_thw=inputs.get("image_grid_thw"), attention_mask=inputs.get("attention_mask"),
    )
    return position_ids + cursor


def _commit_last_token(model, past, token, cache_length: int, position: int):
    """Generation caches the token used to predict the last token, not the last token itself."""
    import torch

    if past.get_seq_length() == cache_length:
        return past
    if past.get_seq_length() != cache_length - 1:
        raise RuntimeError(f"Unexpected cache length {past.get_seq_length()}, expected {cache_length - 1} or {cache_length}.")
    mask = torch.ones((1, cache_length), dtype=torch.long, device=token.device)
    pos = torch.full((3, 1, 1), position, dtype=torch.long, device=token.device)
    with torch.inference_mode():
        output = model(input_ids=token.view(1, 1), attention_mask=mask, position_ids=pos,
                       past_key_values=past, use_cache=True, return_dict=True)
    return output.past_key_values


def _generate_segment(model, processor, inputs, past, cursor: int, global_positions, tracked_positions, grids,
                      token_budget: int, global_biases, crop_key_bias: float = 0.0,
                      visual_score: str = "attention"):
    """Generate one cached segment and collect global/current-crop attention maps."""
    import torch

    local_len = inputs["input_ids"].shape[1]
    prior_len = 0 if past is None else past.get_seq_length()
    positions = _segment_positions(model, inputs, cursor)
    full_mask = torch.ones((1, prior_len + local_len), dtype=torch.long, device=inputs["input_ids"].device)
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.max_new_tokens = token_budget
    generation.use_cache = True
    generation.return_dict_in_generate = True
    # The capture needs absolute cache key positions: old global tokens plus
    # current crop tokens offset by the cache length before this appended turn.
    with MultiImageAttentionCapture(model, tracked_positions, grids, visual_score) as capture, \
            GlobalVisualKeyBias(global_positions, global_biases,
                                bool(global_biases.abs().max().item() > 0) or bool(crop_key_bias) or visual_score != "attention",
                                __import__("torch").cat(tracked_positions[1:]) if len(tracked_positions) > 1 else None,
                                crop_key_bias, capture), torch.inference_mode():
        model_inputs = dict(inputs)
        model_inputs.pop("attention_mask", None)
        output = model.generate(
            **model_inputs, attention_mask=full_mask, position_ids=positions,
            past_key_values=past, generation_config=generation,
        )
    generated = output.sequences[0, local_len:].tolist()
    if not generated:
        raise RuntimeError("Cached generation returned no tokens.")
    raw_maps, _ = capture.summaries(len(generated))
    expected_length = prior_len + local_len + len(generated)
    last_position = int(positions.max().item()) + len(generated) - 1
    past = _commit_last_token(model, output.past_key_values,
                              output.sequences[0, -1], expected_length, last_position)
    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return text, past, last_position + 1, raw_maps


def _make_anchor_cache(model, inputs):
    """Cache full-image question once, before any assistant reasoning tokens.

    A replay window clones this immutable prefix then recomputes the active
    reasoning suffix together with its new crop.  Thus old reasoning tokens
    are not frozen relative to the newly supplied visual evidence.
    """
    import torch

    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True, return_dict=True)
    return output.past_key_values, int(positions.max().item()) + 1


def _tail_tokens(tokenizer, text: str, limit: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[-limit:], skip_special_tokens=True) if len(ids) > limit else text


def _iou(a: PixelRegion, b: PixelRegion) -> float:
    left, top = max(a.left, b.left), max(a.top, b.top)
    right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    union = ((a.right - a.left) * (a.bottom - a.top)
             + (b.right - b.left) * (b.bottom - b.top) - inter)
    return inter / union if union else 0.0


def _coverage_regions(original: Image.Image, attention_regions: list[PixelRegion], cfg: ContinuousConfig) -> list[PixelRegion]:
    """Use a spatial safety net before attention-only refinement begins."""
    tiles = []
    for row in range(cfg.bootstrap_grid_rows):
        for col in range(cfg.bootstrap_grid_cols):
            tiles.append(PixelRegion(
                col * original.width // cfg.bootstrap_grid_cols,
                row * original.height // cfg.bootstrap_grid_rows,
                (col + 1) * original.width // cfg.bootstrap_grid_cols,
                (row + 1) * original.height // cfg.bootstrap_grid_rows,
                0.0,
            ))
    # Put the decoder's strongest proposal first, then guarantee broad map
    # coverage.  A small proposal has low IoU with its containing tile, so both
    # can coexist and receive independent high-resolution visual encodings.
    result = list(attention_regions[:1]) + tiles
    return result[:cfg.regions]


def _novel_regions(candidates: list[PixelRegion], visited: list[PixelRegion], cfg: ContinuousConfig) -> list[PixelRegion]:
    """Avoid repeatedly scouting the same legend/saliency attractor."""
    selected = []
    for candidate in candidates:
        if all(_iou(candidate, prior) <= cfg.novelty_iou for prior in [*visited, *selected]):
            selected.append(candidate)
            if len(selected) == cfg.regions:
                return selected
    # Preserve progress even if every candidate overlaps a previous check.
    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
            if len(selected) == cfg.regions:
                break
    return selected


def _ungated_global_scout(model, processor, focus_past, focus_cursor: int, global_positions,
                          global_grid, device, cfg: ContinuousConfig):
    """Read original global visual KV without altering the formal reasoning cache.

    ``focus_past`` contains the just-completed reasoning segment, so the scout
    query is conditioned on what Window n reasoned.  The clone is disposable:
    it contributes neither text nor KV to Window n+1.  Crucially, all global
    key biases and crop boosts are zero in this pass, so its map is suitable
    for selecting a fresh original-image crop.
    """
    prompt = """Privately plan the next visual check for the reasoning already completed. Inspect the original full map held in context. Identify the unresolved map location most useful to verify next. Do not restate the legend or answer the task."""
    inputs = _append_turn_inputs(processor, [], prompt, device)
    zero_bias = __import__("torch").zeros(global_positions.numel(), dtype=__import__("torch").float32,
                                             device=global_positions.device)
    text, _discarded, _cursor, maps = _generate_segment(
        model, processor, inputs, copy.deepcopy(focus_past), focus_cursor,
        global_positions, [global_positions], [global_grid], cfg.global_scout_tokens,
        zero_bias, 0.0, cfg.visual_score,
    )
    return text, maps[0]


def _mass(map_layers: np.ndarray, aggregation: str) -> float:
    weights = np.linspace(1.0, 2.0, map_layers.shape[0]).tolist() if aggregation == "weighted_mean" else None
    return float(aggregate_decoder_layers(map_layers, aggregation, weights).sum())


def _patch_level_global_bias(map_layers: np.ndarray, cfg: ContinuousConfig, count: int):
    """Keep high-attention global patches accessible; suppress only the rest.

    Scores come from the preceding window's actual attention to the initial
    image.  The returned vector aligns exactly with the original image-token
    cache positions in raster order.
    """
    import torch

    weights = np.linspace(1.0, 2.0, map_layers.shape[0]).tolist() if cfg.aggregation == "weighted_mean" else None
    scores = np.maximum(aggregate_decoder_layers(map_layers, cfg.aggregation, weights).reshape(-1), 0)
    if scores.size != count:
        raise RuntimeError(f"Global patch score count {scores.size} != global cache key count {count}.")
    keep = np.zeros(count, dtype=bool)
    if scores.sum() > 0:
        order = np.argsort(scores)[::-1]
        cutoff = np.searchsorted(np.cumsum(scores[order]), scores.sum() * cfg.global_retain_attention_mass, side="left") + 1
        keep[order[:cutoff]] = True
    else:
        keep[:] = True
    # High-support patches are unmodified (0); only low-support visual keys get
    # the negative logit bias.  No text/reasoning cache position appears here.
    return torch.tensor(np.where(keep, 0.0, cfg.global_key_bias), dtype=torch.float32), int(keep.sum())


def _enforce_min_global_patch_extent(regions, original: Image.Image, global_grid, min_patches: int):
    """Expand tiny recursive crops to at least N-by-N initial global patches.

    A crop is re-resized before every VLM call, so a two-cell component in a
    later crop grid can otherwise map to only a few *original* pixels.  The
    initial global grid is the stable physical reference frame.
    """
    if min_patches < 1:
        return regions
    gh, gw = global_grid
    min_w = min(original.width, int(np.ceil(original.width * min_patches / gw)))
    min_h = min(original.height, int(np.ceil(original.height * min_patches / gh)))
    result = []
    for region in regions:
        width, height = region.right - region.left, region.bottom - region.top
        target_w, target_h = max(width, min_w), max(height, min_h)
        cx, cy = (region.left + region.right) // 2, (region.top + region.bottom) // 2
        left = max(0, min(original.width - target_w, cx - target_w // 2))
        top = max(0, min(original.height - target_h, cy - target_h // 2))
        result.append(PixelRegion(left, top, left + target_w, top + target_h, region.score))
    return result


def run(model, processor, image_path: Path, question: str, cfg: ContinuousConfig) -> dict:
    import torch

    original = Image.open(image_path).convert("RGB")
    full = PixelRegion(0, 0, original.width, original.height, 1.0)
    device = model.get_input_embeddings().weight.device
    global_image = _resize_to_pixel_cap(original, cfg.global_max_pixels)
    global_visual_budget = {
        "source_size": [original.width, original.height],
        "submitted_size": [global_image.width, global_image.height],
        "source_pixels": original.width * original.height,
        "submitted_pixels": global_image.width * global_image.height,
        "max_pixels": cfg.global_max_pixels,
    }
    initial_prompt = f"""Inspect the complete map carefully. Question: {question}
Reason step by step. Do not give a final answer until explicitly asked."""
    initial = _content_inputs(processor, [global_image], initial_prompt, device, cfg.global_max_pixels)
    global_local, global_grids = _image_geometries(model, initial)
    global_positions = global_local[0]
    global_gate_biases = torch.zeros(global_positions.numel(), dtype=torch.float32, device=global_positions.device)
    text, past, cursor, initial_maps = _generate_segment(
        model, processor, initial, None, 0, global_positions, global_local, global_grids,
        cfg.window_tokens, global_gate_biases, 0.0, cfg.visual_score,
    )
    initial_cache_length = past.get_seq_length()
    anchor_past, anchor_cursor = (None, None)
    reasoning_trace = text
    if cfg.replay_tail_tokens:
        # Do this once only.  ``past`` above contains generated thinking; this
        # immutable cache contains just full image + question + assistant cue.
        # Replay windows do not use the first generated cache. Release it
        # before building the anchor, otherwise two large global KV caches
        # overlap with the second vision forward.
        del past
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        anchor_past, anchor_cursor = _make_anchor_cache(model, initial)
    # This is deliberately generated outside ``past``: the ledger is a compact
    # controller state, while ``past`` remains the unbroken original model
    # trajectory.  The ledger text is appended to each subsequent crop turn.
    # Hence it makes the next unresolved check explicit without rebuilding or
    # suppressing the preceding reasoning KV.
    ledger_cfg = RollingConfig(ledger_tokens=cfg.ledger_tokens)
    ledger = None if cfg.replay_tail_tokens else _update_evidence_ledger(model, processor, question, "", text, ledger_cfg)
    attention_regions = _refine_regions(initial_maps, [full], global_grids, RollingConfig(
        regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
    ))
    attention_regions = _enforce_min_global_patch_extent(attention_regions, original, global_grids[0], cfg.min_crop_global_patches)
    regions = (_coverage_regions(original, attention_regions, cfg)
               if cfg.bootstrap_windows > 0 else attention_regions)
    global_gate_biases, retained_global_patches = _patch_level_global_bias(
        initial_maps[0], cfg, global_positions.numel()
    )
    initial_scout_text, initial_scout_regions = None, None
    # Do not let an early legend-biased scout override the coverage safety net.
    if cfg.full_reselect_from_window == 0 and cfg.bootstrap_windows == 0:
        initial_scout_past = anchor_past if cfg.replay_tail_tokens else past
        initial_scout_cursor = anchor_cursor if cfg.replay_tail_tokens else cursor
        initial_scout_text, initial_scout_map = _ungated_global_scout(
            model, processor, initial_scout_past, initial_scout_cursor, global_positions, global_grids[0], device, cfg
        )
        initial_scout_regions = _refine_regions([initial_scout_map], [full], global_grids, RollingConfig(
            regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
        ))
        regions = _enforce_min_global_patch_extent(
            initial_scout_regions, original, global_grids[0], cfg.min_crop_global_patches
        )
        initial_scout_regions = regions
    visited_scout_regions: list[PixelRegion] = []
    windows = [{"window": 0, "text": text, "ledger_before": None, "ledger_after": ledger,
                "replay_tail_after": _tail_tokens(processor.tokenizer, reasoning_trace, cfg.replay_tail_tokens) if cfg.replay_tail_tokens else None,
                "input_regions": [asdict(full)], "input_visual_budget": global_visual_budget,
                "next_regions": [asdict(r) for r in regions], "global_bias": 0.0,
                "selected_regions_source": "ungated_global_scout" if initial_scout_regions else ("bootstrap_coverage" if cfg.bootstrap_windows > 0 else "initial_focus"),
                "ungated_global_scout_text": initial_scout_text,
                "ungated_global_scout_regions": [asdict(r) for r in initial_scout_regions] if initial_scout_regions else None,
                "global_retained_patches_for_next": retained_global_patches,
                "cache_length_before": 0, "cache_length_after": initial_cache_length,
                "attention_mass": {"global": _mass(initial_maps[0], cfg.aggregation), "crop": None}}]
    for index in range(1, cfg.windows):
        final = index == cfg.windows - 1
        use_global = index >= cfg.full_reselect_from_window and (index - cfg.full_reselect_from_window) % cfg.full_reselect_interval == 0
        bias_active = index >= cfg.global_bias_from_window and (index - cfg.global_bias_from_window) % cfg.global_bias_interval == 0
        ledger_before = ledger
        if cfg.replay_tail_tokens:
            replay_before = _tail_tokens(processor.tokenizer, reasoning_trace, cfg.replay_tail_tokens)
            prompt = """The attached crop is fresh visual evidence for the assistant reasoning already in progress.
Continue that exact unfinished reasoning; do not restart the task or reintroduce the legend. """ + (
                "Conclude exactly with: Final answer: <answer>." if final else
                "Add the next concrete visual observation and proceed to the next unresolved check; do not give a final answer yet."
            )
            segment_past, segment_cursor = copy.deepcopy(anchor_past), anchor_cursor
        else:
            replay_before = None
            prompt = f"""This is a continuation of one unfinished visual reasoning trace, not a new attempt.
The model's preceding reasoning KV is still present. The new image crop is only additional evidence.

The following compact evidence ledger was generated after the preceding window. Treat VERIFIED facts as already
settled unless the crop directly contradicts them. Execute exactly the item(s) in NEXT before considering anything
else. Do NOT restate the question, legend, city list, or any VERIFIED fact. Your response must begin with the
new visual observation and then state the next unresolved check; it must add a delta, rather than restart analysis.
<evidence_ledger>
{ledger_before}
</evidence_ledger>

{"Now use the ledger and the crop to answer exactly one line: Final answer: <answer>." if final else "Do not give the final answer yet; report only the next factual evidence delta."}"""
            segment_past, segment_cursor = past, cursor
        input_regions = list(regions)
        images, crop_budget = _crop_images_with_budget(original, input_regions, cfg.crop_total_max_pixels)
        per_crop_budget = cfg.crop_total_max_pixels // max(1, len(input_regions))
        inputs = (_append_crop_then_assistant_replay(processor, images, prompt, replay_before, device, per_crop_budget)
                  if cfg.replay_tail_tokens else _append_turn_inputs(processor, images, prompt, device, per_crop_budget))
        crop_local, crop_grids = _image_geometries(model, inputs)
        crop_absolute = [span + segment_past.get_seq_length() for span in crop_local]
        tracked = [global_positions, *crop_absolute]
        grids = [global_grids[0], *crop_grids]
        active_biases = global_gate_biases if bias_active else torch.zeros_like(global_gate_biases)
        cache_length_before = segment_past.get_seq_length()
        text, past, cursor, maps = _generate_segment(
            model, processor, inputs, segment_past, segment_cursor, global_positions, tracked, grids,
            cfg.final_window_tokens if final else cfg.window_tokens,
            active_biases, cfg.crop_key_bias, cfg.visual_score,
        )
        if cfg.replay_tail_tokens:
            reasoning_trace += "\n" + text
            ledger = None
        else:
            ledger = _update_evidence_ledger(model, processor, question, ledger_before, text, ledger_cfg)
        global_mass = _mass(maps[0], cfg.aggregation)
        crop_mass = sum(_mass(item, cfg.aggregation) for item in maps[1:])
        global_gate_biases, retained_global_patches = _patch_level_global_bias(
            maps[0], cfg, global_positions.numel()
        )
        scout_text, scout_regions = None, None
        did_global_scout = False
        selection_source = "crop_refine"
        # The focus pass intentionally uses the gated global image.  Never use
        # that distorted map to rediscover a location on the full image: run a
        # disposable, ungated scout after the current reasoning instead.
        if index < cfg.bootstrap_windows:
            # Window 1 repeats the broad evidence set once.  This gives small
            # labels/city markers a second high-resolution observation before
            # free-form attention is allowed to narrow the search.
            regions = _coverage_regions(original, attention_regions, cfg)
            selection_source = "bootstrap_coverage"
        elif use_global:
            did_global_scout = True
            scout_text, scout_map = _ungated_global_scout(
                model, processor, past, cursor, global_positions, global_grids[0], device, cfg
            )
            # Ask for extra candidates then reject repetitive high-saliency
            # locations (typically the legend) before taking the final budget.
            scout_candidates = _refine_regions([scout_map], [full], global_grids, RollingConfig(
                regions=cfg.regions * 4, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
            ))
            scout_regions = _novel_regions(scout_candidates, visited_scout_regions, cfg)
            regions = scout_regions
            selection_source = "ungated_global_scout_novel"
        else:
            regions = _refine_regions(maps[1:], input_regions, crop_grids, RollingConfig(
                regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
            ))
        regions = _enforce_min_global_patch_extent(regions, original, global_grids[0], cfg.min_crop_global_patches)
        if did_global_scout:
            scout_regions = regions
            visited_scout_regions.extend(scout_regions)
        windows.append({"window": index, "text": text, "ledger_before": ledger_before, "ledger_after": ledger,
                        "replay_tail_before": replay_before,
                        "replay_tail_after": _tail_tokens(processor.tokenizer, reasoning_trace, cfg.replay_tail_tokens) if cfg.replay_tail_tokens else None,
                        "input_regions": [asdict(r) for r in input_regions],
                        "input_visual_budget": {"total_max_pixels": cfg.crop_total_max_pixels,
                                                "per_crop_max_pixels": per_crop_budget,
                                                "crops": crop_budget},
                        "next_regions": [asdict(r) for r in regions], "global_reselect_for_next": did_global_scout,
                        "selected_regions_source": selection_source,
                        "ungated_global_scout_text": scout_text,
                        "ungated_global_scout_regions": [asdict(r) for r in scout_regions] if scout_regions else None,
                        "global_bias_low_support": cfg.global_key_bias if bias_active else 0.0,
                        "global_retained_patches_for_next": retained_global_patches,
                        "cache_length_before": cache_length_before, "cache_length_after": past.get_seq_length(),
                        "attention_mass": {"global": global_mass, "crop": crop_mass,
                                           "global_share": global_mass / max(global_mass + crop_mass, 1e-12)}})
    final_text = windows[-1]["text"]
    return {"image": str(image_path.resolve()), "question": question, "config": asdict(cfg),
            "protocol": "anchor KV + replayed active reasoning suffix" if cfg.replay_tail_tokens else "continuous KV append; initial global image KV and reasoning KV retained; crop turns appended; optional global attention-logit bias",
            "windows": windows, "final_text": final_text, "final_answer": _extract_final_answer(final_text)}


def main():
    parser = argparse.ArgumentParser(description="Experimental continuous-cache Qwen3-VL crop reasoning.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--window-tokens", type=int, default=80)
    parser.add_argument("--final-window-tokens", type=int, default=128)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--attention-mass", type=float, default=0.60)
    parser.add_argument("--min-crop-global-patches", type=int, default=2,
                        help="Minimum crop width and height measured in initial full-image patch cells.")
    parser.add_argument("--aggregation", choices=("last", "mean", "max", "weighted_mean"), default="mean")
    parser.add_argument("--visual-score", choices=("attention", "value_norm", "projected_value_norm"), default="attention",
                        help="Patch score used for crop ranking: raw attention, attention×||V||, or attention×||V W_O||.")
    parser.add_argument("--full-reselect-from-window", type=int, default=0,
                        help="0 means use ungated original-map scout immediately to choose Window 1.")
    parser.add_argument("--full-reselect-interval", type=int, default=1,
                        help="Run the ungated original-map scout after every N windows.")
    parser.add_argument("--global-bias-from-window", type=int, default=3)
    parser.add_argument("--global-bias-interval", type=int, default=1)
    parser.add_argument("--global-key-bias", type=float, default=-4.0)
    parser.add_argument("--global-retain-attention-mass", type=float, default=0.60,
                        help="Attention mass of initial-image patches left unpenalized by the next window's gate.")
    parser.add_argument("--ledger-tokens", type=int, default=128,
                        help="Maximum tokens for the per-window factual evidence ledger.")
    parser.add_argument("--replay-tail-tokens", type=int, default=0,
                        help="When >0, preserve full-image anchor KV but recompute this many recent reasoning tokens with every new crop.")
    parser.add_argument("--crop-key-bias", type=float, default=0.0,
                        help="Add this attention-logit bias only to current crop visual keys; ln(5)=1.609 approximates five copies.")
    parser.add_argument("--global-scout-tokens", type=int, default=24,
                        help="Tokens for the disposable ungated global-crop planning pass.")
    parser.add_argument("--bootstrap-windows", type=int, default=2,
                        help="Early crop windows that use broad grid coverage before attention-only scouting.")
    parser.add_argument("--bootstrap-grid-rows", type=int, default=3)
    parser.add_argument("--bootstrap-grid-cols", type=int, default=3)
    parser.add_argument("--novelty-iou", type=float, default=0.35,
                        help="Reject global-scout candidates overlapping previously selected scout regions above this IoU.")
    parser.add_argument("--global-max-pixels", type=int, default=1_000_000,
                        help="Maximum source pixels submitted for the initial global-map vision pass.")
    parser.add_argument("--crop-total-max-pixels", type=int, default=1_000_000,
                        help="Per-window total source-pixel budget split evenly across all crop images.")
    args = parser.parse_args()
    if min(args.windows, args.window_tokens, args.final_window_tokens, args.regions, args.min_crop_global_patches, args.full_reselect_interval, args.global_bias_interval, args.ledger_tokens, args.global_scout_tokens, args.bootstrap_grid_rows, args.bootstrap_grid_cols, args.global_max_pixels, args.crop_total_max_pixels) < 1 or args.replay_tail_tokens < 0 or args.bootstrap_windows < 0:
        parser.error("window counts, token budgets, regions, and intervals must be positive.")
    if args.full_reselect_from_window < 0:
        parser.error("full-reselect-from-window must be non-negative.")
    if not 0 < args.attention_mass <= 1 or not 0 < args.global_retain_attention_mass <= 1:
        parser.error("attention-mass and global-retain-attention-mass must be in (0, 1].")
    if not 0 <= args.novelty_iou <= 1:
        parser.error("novelty-iou must be in [0, 1].")
    model, processor = load_model(args.model, args.adapter)
    cfg = ContinuousConfig(**{key: getattr(args, key) for key in ContinuousConfig.__dataclass_fields__})
    result = run(model, processor, args.image, args.question, cfg)
    output = args.output / "continuous_result.json" if not args.output.suffix else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    render_window_crops(args.image, result, output.parent / f"{output.stem}_windows")
    print(f"Saved continuous-cache trajectory to {output}")


if __name__ == "__main__":
    main()
