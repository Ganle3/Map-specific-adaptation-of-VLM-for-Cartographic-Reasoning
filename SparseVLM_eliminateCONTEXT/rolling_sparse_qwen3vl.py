#!/usr/bin/env python3
"""Rolling visual re-input for a Qwen3-VL-8B-Thinking PEFT adapter.

Window 0 sees the complete image.  Later windows can focus on dynamically
ranked crops.  An optional scout→focus mode uses the complete image only in a
no-cache attention scout, then performs actual reasoning with crops alone.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from attention_policy import Region, aggregate_decoder_layers, salient_regions
from dynamic_attention import MultiImageAttentionCapture


@dataclass(frozen=True)
class RollingConfig:
    windows: int = 4
    window_tokens: int = 512
    transcript_tokens: int = 1200
    regions: int = 4
    region_side: int = 4
    aggregation: str = "last"  # last | mean | max | weighted_mean
    attention_mass: float = 0.60
    max_pixels: int = 1280 * 1280
    include_full_image_after_first: bool = False
    scout_full_image_from_window: int | None = None
    scout_full_image_interval: int = 1
    full_image_scout_only: bool = False
    scout_tokens: int = 80
    state_mode: str = "ledger"  # ledger | raw
    ledger_tokens: int = 192
    final_window_tokens: int = 96


@dataclass(frozen=True)
class PixelRegion:
    """A region in the coordinate system of the original image."""
    left: int
    top: int
    right: int
    bottom: int
    score: float


def crop_region(image: Image.Image, region: PixelRegion) -> Image.Image:
    return image.crop((region.left, region.top, region.right, region.bottom))


def _prompt(question: str, transcript: str, window: int, region_count: int, allow_final: bool,
            evidence_override: str | None = None) -> str:
    evidence = evidence_override or ("the complete original map" if window == 0 else (
        "only high-priority cropped map regions, in descending visual priority"
    ))
    return f"""You are solving a cartographic reasoning problem with a Qwen3-VL thinking model.
Question: {question}

This is rolling reasoning window {window}. You are given {evidence}. Inspect them again;
the images are authoritative and must override a mistaken earlier observation.

Persistent evidence state from earlier windows:
<rolling_state>
{transcript or '(none; begin by inspecting the complete map)'}
</rolling_state>

{"This is the final window. Recheck the supplied crops against the evidence state, then output exactly one line: Final answer: <answer>. Do not restart a long analysis, do not list the evidence, and do not emit any text after that line." if allow_final else "Record only a provisional visual reasoning step. Do NOT emit Final answer yet; unresolved evidence will be carried forward in a structured ledger."}"""


def _build_inputs(processor, images: list[Image.Image], prompt: str, device):
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs.to(device)


def _image_geometries(model, inputs) -> tuple[list, list[tuple[int, int]]]:
    """Split Qwen's image placeholder tokens into one span per supplied image."""
    positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    merge = model.config.vision_config.spatial_merge_size
    grids, spans, offset = [], [], 0
    for t, h, w in inputs["image_grid_thw"].tolist():
        if h % merge or w % merge:
            raise RuntimeError("Image grid is not divisible by Qwen spatial_merge_size.")
        grid = (h // merge, w // merge)
        size = t * grid[0] * grid[1]
        grids.append(grid)
        spans.append(positions[offset:offset + size])
        offset += size
    if offset != positions.numel() or any(not len(span) for span in spans):
        raise RuntimeError("Could not align Qwen image placeholders with image_grid_thw.")
    return spans, grids


def _generate_with_attention(model, processor, images: list[Image.Image], prompt: str, cfg: RollingConfig,
                             max_new_tokens: int | None = None):
    import torch

    device = model.get_input_embeddings().weight.device
    inputs = _build_inputs(processor, images, prompt, device)
    positions, grids = _image_geometries(model, inputs)
    prompt_len = inputs["input_ids"].shape[1]
    gen = copy.deepcopy(model.generation_config)
    gen.do_sample = False
    gen.num_beams = 1
    gen.max_new_tokens = max_new_tokens if max_new_tokens is not None else cfg.window_tokens
    gen.use_cache = True
    gen.return_dict_in_generate = False
    with MultiImageAttentionCapture(model, positions, grids) as capture, torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=gen)
    ids = sequences[0, prompt_len:].tolist()
    if not ids:
        raise RuntimeError("Generation returned no tokens; cannot rank visual evidence.")
    raw, conditional = capture.summaries(len(ids))
    text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    return text, raw, conditional, grids


def _update_evidence_ledger(model, processor, question: str, prior_ledger: str, window_text: str,
                            cfg: RollingConfig) -> str:
    """Compress a visual reasoning segment into a bounded, reusable state."""
    import torch

    prior_ledger = prior_ledger or "VERIFIED: none\nUNRESOLVED: none\nNEXT: inspect the image"
    prompt = f"""Maintain a factual evidence ledger for this visual question:
{question}

Previous ledger:
<ledger>
{prior_ledger}
</ledger>

New visual reasoning segment (possibly truncated; do not infer unsupported facts):
<segment>
{window_text}
</segment>

Return ONLY this compact format, preserving verified facts and unresolved items:
VERIFIED:
- ...
UNRESOLVED:
- ...
NEXT:
- ...
Do not answer the original question and do not include think tags."""
    device = model.get_input_embeddings().weight.device
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(device)
    prompt_len = inputs["input_ids"].shape[1]
    gen = copy.deepcopy(model.generation_config)
    gen.do_sample = False
    gen.num_beams = 1
    gen.max_new_tokens = cfg.ledger_tokens
    gen.use_cache = False
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=gen)
    ledger = processor.tokenizer.decode(output[0, prompt_len:].tolist(), skip_special_tokens=True).strip()
    if not ledger:
        raise RuntimeError("Evidence-ledger update generated no text.")
    return ledger


def _scout_full_image(model, processor, image: Image.Image, prompt: str,
                      cfg: RollingConfig) -> tuple[str, list[np.ndarray], list[tuple[int, int]]]:
    """Generate a private visual plan, then discard its full-image KV cache.

    Attention is aggregated over several actual generated planning tokens,
    matching the formal-window attention mechanism.  The planning text and its
    cache are not passed to the formal crop-only generation call.
    """
    import torch

    device = model.get_input_embeddings().weight.device
    inputs = _build_inputs(processor, [image], prompt, device)
    positions, grids = _image_geometries(model, inputs)
    prompt_len = inputs["input_ids"].shape[1]
    gen = copy.deepcopy(model.generation_config)
    gen.do_sample = False
    gen.num_beams = 1
    gen.max_new_tokens = cfg.scout_tokens
    gen.use_cache = True
    gen.return_dict_in_generate = False
    with MultiImageAttentionCapture(model, positions, grids) as capture, torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=gen)
    ids = sequences[0, prompt_len:].tolist()
    if not ids:
        raise RuntimeError("Full-image scout generated no visual-planning tokens.")
    raw, _conditional = capture.summaries(len(ids))
    planning_text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    del sequences, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return planning_text, raw, grids


def _pixel_iou(a: PixelRegion, b: PixelRegion) -> float:
    overlap_w = max(0, min(a.right, b.right) - max(a.left, b.left))
    overlap_h = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))
    intersection = overlap_w * overlap_h
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection
    return intersection / union if union else 0.0


def _project_component(parent: PixelRegion, local: Region, grid: tuple[int, int]) -> PixelRegion:
    """Project a non-empty patch component to non-empty original pixel bounds.

    Floor on the left/top and ceiling on the right/bottom is essential when a
    small parent crop is internally resized to a much denser Qwen patch grid.
    Using floor for both sides could turn one selected patch into a 0-pixel
    image, which PIL correctly refuses to save.
    """
    gh, gw = grid
    width, height = parent.right - parent.left, parent.bottom - parent.top
    left = parent.left + local.left * width // gw
    top = parent.top + local.top * height // gh
    right = parent.left + (local.right * width + gw - 1) // gw
    bottom = parent.top + (local.bottom * height + gh - 1) // gh
    return PixelRegion(left, top, max(left + 1, right), max(top + 1, bottom), local.score)


def _refine_regions(raw_maps: list[np.ndarray], parents: list[PixelRegion], grids: list[tuple[int, int]], cfg: RollingConfig) -> list[PixelRegion]:
    """Map current-window attention from each input crop back to original pixels."""
    candidates: list[PixelRegion] = []
    for maps, parent, grid in zip(raw_maps, parents, grids):
        weights = np.linspace(1.0, 2.0, maps.shape[0]).tolist() if cfg.aggregation == "weighted_mean" else None
        score_map = aggregate_decoder_layers(maps, cfg.aggregation, weights)
        for local in salient_regions(score_map, cfg.regions, cfg.attention_mass):
            candidates.append(_project_component(parent, local, grid))
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[PixelRegion] = []
    for candidate in candidates:
        if all(_pixel_iou(candidate, previous) <= 0.25 for previous in selected):
            selected.append(candidate)
            if len(selected) == cfg.regions:
                break
    return selected


def render_window_crops(image_path: Path, result: dict, destination: Path) -> None:
    """Export exact per-window crop inputs plus original-image overlays.

    ``window_XX_input_rank_YY.png`` are the images actually fed to that
    window.  ``window_XX_next_rank_YY.png`` are the dynamically selected
    patches that will feed the next window.  The overlay records their source
    coordinates on the unmodified original, so resized processor inputs do not
    obscure what was selected.
    """
    original = Image.open(image_path).convert("RGB")
    destination.mkdir(parents=True, exist_ok=True)
    palette = [(231, 76, 60), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (26, 188, 156)]
    for window in result["windows"]:
        index = int(window["window"])
        for kind, label in (("input_regions", "input"), ("next_regions", "next")):
            regions = [PixelRegion(**row) for row in window[kind]]
            overlay = original.copy()
            draw = ImageDraw.Draw(overlay)
            for rank, region in enumerate(regions, 1):
                color = palette[(rank - 1) % len(palette)]
                draw.rectangle((region.left, region.top, region.right, region.bottom), outline=color, width=5)
                draw.text((region.left + 4, region.top + 4), f"{label} {rank}", fill=color, stroke_width=1, stroke_fill="black")
                crop_region(original, region).save(destination / f"window_{index:02d}_{label}_rank_{rank:02d}.png")
            overlay.save(destination / f"window_{index:02d}_{label}_overlay.png")


def _extract_final_answer(text: str) -> str | None:
    matches = re.findall(r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else None


def run(model, processor, image_path: Path, question: str, cfg: RollingConfig) -> dict:
    """Run the rolling protocol and return serialisable evidence/provenance."""
    original = Image.open(image_path).convert("RGB")
    rolling_state = ""
    windows: list[dict] = []
    full_region = PixelRegion(0, 0, original.width, original.height, 1.0)
    regions: list[PixelRegion] = [full_region]
    for index in range(cfg.windows):
        is_final = index == cfg.windows - 1
        prompt = _prompt(question, rolling_state, index, len(regions), is_final)
        scout_record = None
        scout_active = (
            cfg.full_image_scout_only
            and cfg.scout_full_image_from_window is not None
            and index >= cfg.scout_full_image_from_window
            and (index - cfg.scout_full_image_from_window) % cfg.scout_full_image_interval == 0
        )
        if scout_active:
            scout_prompt = _prompt(
                question, rolling_state, index, len(regions), False,
                evidence_override=(
                    "the complete original map. Generate a short private visual search plan: "
                    "identify the legend and the specific map regions needed to resolve the currently unresolved evidence. "
                    "Do not give the final answer."
                ),
            )
            scout_text, scout_maps, scout_grids = _scout_full_image(model, processor, original, scout_prompt, cfg)
            parents = _refine_regions(scout_maps, [full_region], scout_grids, cfg)
            if not parents:
                raise RuntimeError("Full-image scout selected no attention regions.")
            scout_record = {
                "input_region": asdict(full_region),
                "attention_grids": [list(grid) for grid in scout_grids],
                "selected_regions": [asdict(region) for region in parents],
                "planning_text_discarded_before_formal_generation": scout_text,
                "cache_policy": "scout KV is used during planning then discarded; formal generation starts a new crop-only KV cache",
            }
        else:
            parents = ([full_region] + regions) if index and cfg.include_full_image_after_first else regions
        images = [crop_region(original, parent) for parent in parents]
        text, raw_maps, _conditional_maps, grids = _generate_with_attention(
            model, processor, images, prompt, cfg,
            max_new_tokens=cfg.final_window_tokens if is_final else None,
        )
        regions = _refine_regions(raw_maps, parents, grids, cfg)
        ledger_after = None
        if not is_final:
            if cfg.state_mode == "ledger":
                rolling_state = _update_evidence_ledger(model, processor, question, rolling_state, text, cfg)
            else:
                rolling_state = processor.tokenizer.decode(
                    processor.tokenizer.encode((rolling_state + "\n" + text).strip(), add_special_tokens=False)[-cfg.transcript_tokens:],
                    skip_special_tokens=True,
                )
            ledger_after = rolling_state
        windows.append({"window": index, "text": text, "attention_grids": [list(grid) for grid in grids],
                        "aggregation": cfg.aggregation, "input_regions": [asdict(parent) for parent in parents],
                        "next_regions": [asdict(region) for region in regions], "scout": scout_record,
                        "ledger_after": ledger_after})
    protocol = "window 0 uses full image; later windows use only dynamically re-ranked crops unless include_full_image_after_first is enabled"
    if cfg.full_image_scout_only:
        protocol += "; at the configured scout threshold, full image is used only by a disposable planning generation before crop-only formal generation"
    final_text = windows[-1]["text"] if windows else ""
    return {"image": str(image_path.resolve()), "question": question, "config": asdict(cfg),
            "protocol": protocol,
            "windows": windows, "rolling_state": rolling_state,
            "final_answer": _extract_final_answer(final_text), "final_text": final_text}


def load_model(model_name: str | None, adapter_path: Path | None):
    import torch
    # The project's 4-bit Unsloth checkpoints patch Transformers at import time.
    # Keep the normal Hugging Face path dependency-free for standard adapters.
    if model_name and model_name.lower().startswith("unsloth/"):
        import unsloth  # noqa: F401
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if adapter_path:
        config_file = adapter_path / "adapter_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"Missing PEFT adapter config: {config_file}")
        adapter_base = json.loads(config_file.read_text(encoding="utf-8")).get("base_model_name_or_path")
        if model_name is None:
            model_name = adapter_base
        elif adapter_base and model_name != adapter_base:
            raise ValueError(f"Adapter expects base {adapter_base!r}, but --model is {model_name!r}.")
    model_name = model_name or "Qwen/Qwen3-VL-8B-Thinking"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", attn_implementation="eager"
    ).eval()
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path)).eval()
    return model, AutoProcessor.from_pretrained(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling visual re-input for Qwen3-VL-Thinking + PEFT adapter.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=None, help="Base model; inferred from adapter_config.json when omitted.")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--window-tokens", type=int, default=512)
    parser.add_argument("--transcript-tokens", type=int, default=1200)
    parser.add_argument("--state-mode", choices=("ledger", "raw"), default="ledger",
                        help="Persistent state: bounded factual ledger (default) or legacy raw transcript.")
    parser.add_argument("--ledger-tokens", type=int, default=192,
                        help="Maximum tokens for each text-only evidence-ledger update.")
    parser.add_argument("--final-window-tokens", type=int, default=96,
                        help="Maximum tokens reserved for the final one-line answer mode.")
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--region-side", type=int, default=4,
                        help="Deprecated compatibility option; variable-sized attention components are used instead.")
    parser.add_argument("--aggregation", choices=("last", "mean", "max", "weighted_mean"), default="last")
    parser.add_argument("--attention-mass", type=float, default=0.60,
                        help="Cumulative attention mass retained as dynamic patch components (0, 1].")
    parser.add_argument("--include-full-image-after-first", action="store_true",
                        help="Ablation: retain the complete image after window 0. Default is focus-only crops.")
    parser.add_argument("--scout-full-image-from-window", type=int, default=None,
                        help="Zero-based window index at which to begin disposable full-image scout-to-crop-only focus.")
    parser.add_argument("--scout-full-image-interval", type=int, default=1,
                        help="Run a full-image scout every N windows after its start (1 means every window).")
    parser.add_argument("--full-image-scout-only", action="store_true",
                        help="Require full images after the scout threshold to be used only in a disposable scout pass, never in formal generation.")
    parser.add_argument("--scout-tokens", type=int, default=80,
                        help="Private full-image visual-planning tokens before attention-based re-cropping.")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Do not export per-window crop PNGs and original-image overlays.")
    args = parser.parse_args()
    if (args.windows < 1 or args.window_tokens < 1 or args.transcript_tokens < 1
            or args.ledger_tokens < 1 or args.final_window_tokens < 1 or args.scout_tokens < 1
            or not 0 < args.attention_mass <= 1):
        parser.error("token budgets and windows must be positive; attention-mass must be in (0, 1].")
    if args.scout_full_image_from_window is not None and args.scout_full_image_from_window < 0:
        parser.error("scout-full-image-from-window must be non-negative.")
    if args.scout_full_image_interval < 1:
        parser.error("scout-full-image-interval must be positive.")
    if args.full_image_scout_only != (args.scout_full_image_from_window is not None):
        parser.error("Use --scout-full-image-from-window and --full-image-scout-only together.")
    if args.full_image_scout_only and args.include_full_image_after_first:
        parser.error("--full-image-scout-only conflicts with --include-full-image-after-first.")
    model, processor = load_model(args.model, args.adapter)
    config = RollingConfig(args.windows, args.window_tokens, args.transcript_tokens,
                           args.regions, args.region_side, args.aggregation,
                           args.attention_mass, include_full_image_after_first=args.include_full_image_after_first,
                           scout_full_image_from_window=args.scout_full_image_from_window,
                           scout_full_image_interval=args.scout_full_image_interval,
                           full_image_scout_only=args.full_image_scout_only,
                           scout_tokens=args.scout_tokens,
                           state_mode=args.state_mode, ledger_tokens=args.ledger_tokens,
                           final_window_tokens=args.final_window_tokens)
    result = run(model, processor, args.image, args.question, config)
    # A suffix-less --output is treated as an output directory, which matches
    # the command that produced the PermissionError in the pilot run.
    output_path = args.output / "rolling_result.json" if not args.output.suffix else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist trajectory first: visualization must never make an expensive
    # completed generation unrecoverable.
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_visualize:
        visualization_dir = output_path.parent / f"{output_path.stem}_windows"
        try:
            render_window_crops(args.image, result, visualization_dir)
            result["visualization_dir"] = str(visualization_dir.resolve())
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as error:
            print(f"Warning: trajectory was saved, but crop visualization failed: {error}")
    print(f"Saved rolling trajectory to {output_path}")


if __name__ == "__main__":
    main()
