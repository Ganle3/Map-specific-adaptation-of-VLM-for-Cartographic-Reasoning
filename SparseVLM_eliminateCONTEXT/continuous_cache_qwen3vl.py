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
)


@dataclass(frozen=True)
class ContinuousConfig:
    windows: int = 6
    window_tokens: int = 80
    final_window_tokens: int = 128
    regions: int = 4
    attention_mass: float = 0.60
    aggregation: str = "mean"
    full_reselect_from_window: int = 3
    full_reselect_interval: int = 3
    global_bias_from_window: int = 3
    global_bias_interval: int = 1
    global_key_bias: float = -4.0


class GlobalVisualKeyBias(AbstractContextManager):
    """Add a constant logit bias to initial-global visual keys in eager attention."""

    def __init__(self, global_positions, bias: float, enabled: bool):
        self.global_positions = global_positions
        self.bias = bias
        self.enabled = enabled and bias != 0.0
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
                penalty[..., positions.to(key.device)] = self.bias
                attention_mask = penalty if attention_mask is None else attention_mask + penalty
            return original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        qwen.eager_attention_forward = biased_eager_attention
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.module.eager_attention_forward = self.original


def _content_inputs(processor, images: list[Image.Image], text: str, device):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


def _append_turn_inputs(processor, images: list[Image.Image], text: str, device):
    """Build only the tokens that follow an unfinished assistant response."""
    standalone = _content_inputs(processor, images, text, device)
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
    return standalone


def _segment_positions(model, inputs, cursor: int):
    """Compute local Qwen 3D positions then shift them after the cached sequence."""
    position_ids, _ = model.model.get_rope_index(
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
                      token_budget: int, global_bias: float):
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
    with MultiImageAttentionCapture(model, tracked_positions, grids) as capture, \
            GlobalVisualKeyBias(global_positions, global_bias, global_bias != 0.0), torch.inference_mode():
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


def _mass(map_layers: np.ndarray, aggregation: str) -> float:
    weights = np.linspace(1.0, 2.0, map_layers.shape[0]).tolist() if aggregation == "weighted_mean" else None
    return float(aggregate_decoder_layers(map_layers, aggregation, weights).sum())


def run(model, processor, image_path: Path, question: str, cfg: ContinuousConfig) -> dict:
    import torch

    original = Image.open(image_path).convert("RGB")
    full = PixelRegion(0, 0, original.width, original.height, 1.0)
    device = model.get_input_embeddings().weight.device
    initial_prompt = f"""Inspect the complete map carefully. Question: {question}
Reason step by step. Do not give a final answer until explicitly asked."""
    initial = _content_inputs(processor, [original], initial_prompt, device)
    global_local, global_grids = _image_geometries(model, initial)
    global_positions = global_local[0]
    text, past, cursor, initial_maps = _generate_segment(
        model, processor, initial, None, 0, global_positions, global_local, global_grids,
        cfg.window_tokens, 0.0,
    )
    regions = _refine_regions(initial_maps, [full], global_grids, RollingConfig(
        regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
    ))
    windows = [{"window": 0, "text": text, "input_regions": [asdict(full)],
                "next_regions": [asdict(r) for r in regions], "global_bias": 0.0,
                "attention_mass": {"global": _mass(initial_maps[0], cfg.aggregation), "crop": None}}]
    for index in range(1, cfg.windows):
        final = index == cfg.windows - 1
        use_global = index >= cfg.full_reselect_from_window and (index - cfg.full_reselect_from_window) % cfg.full_reselect_interval == 0
        bias_active = index >= cfg.global_bias_from_window and (index - cfg.global_bias_from_window) % cfg.global_bias_interval == 0
        prompt = (f"Continue the same reasoning. Inspect the newly supplied crop evidence for: {question}. "
                  + ("Output exactly one line: Final answer: <answer>." if final else "Do not give the final answer yet."))
        input_regions = list(regions)
        inputs = _append_turn_inputs(processor, [crop_region(original, r) for r in input_regions], prompt, device)
        crop_local, crop_grids = _image_geometries(model, inputs)
        crop_absolute = [span + past.get_seq_length() for span in crop_local]
        tracked = [global_positions, *crop_absolute]
        grids = [global_grids[0], *crop_grids]
        text, past, cursor, maps = _generate_segment(
            model, processor, inputs, past, cursor, global_positions, tracked, grids,
            cfg.final_window_tokens if final else cfg.window_tokens,
            cfg.global_key_bias if bias_active else 0.0,
        )
        global_mass = _mass(maps[0], cfg.aggregation)
        crop_mass = sum(_mass(item, cfg.aggregation) for item in maps[1:])
        # A scheduled global reselect uses the same continuous-cache attention:
        # global image is never re-encoded and is never sent as a new formal input.
        if use_global:
            regions = _refine_regions([maps[0]], [full], global_grids, RollingConfig(
                regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
            ))
        else:
            regions = _refine_regions(maps[1:], input_regions, crop_grids, RollingConfig(
                regions=cfg.regions, aggregation=cfg.aggregation, attention_mass=cfg.attention_mass
            ))
        windows.append({"window": index, "text": text, "input_regions": [asdict(r) for r in input_regions],
                        "next_regions": [asdict(r) for r in regions], "global_reselect_for_next": use_global,
                        "global_bias": cfg.global_key_bias if bias_active else 0.0,
                        "attention_mass": {"global": global_mass, "crop": crop_mass,
                                           "global_share": global_mass / max(global_mass + crop_mass, 1e-12)}})
    final_text = windows[-1]["text"]
    return {"image": str(image_path.resolve()), "question": question, "config": asdict(cfg),
            "protocol": "continuous KV append; initial global image KV and reasoning KV retained; crop turns appended; optional global attention-logit bias",
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
    parser.add_argument("--aggregation", choices=("last", "mean", "max", "weighted_mean"), default="mean")
    parser.add_argument("--full-reselect-from-window", type=int, default=3)
    parser.add_argument("--full-reselect-interval", type=int, default=3)
    parser.add_argument("--global-bias-from-window", type=int, default=3)
    parser.add_argument("--global-bias-interval", type=int, default=1)
    parser.add_argument("--global-key-bias", type=float, default=-4.0)
    args = parser.parse_args()
    if min(args.windows, args.window_tokens, args.final_window_tokens, args.regions, args.full_reselect_interval, args.global_bias_interval) < 1:
        parser.error("window counts, token budgets, regions, and intervals must be positive.")
    if not 0 < args.attention_mass <= 1:
        parser.error("attention-mass must be in (0, 1].")
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
