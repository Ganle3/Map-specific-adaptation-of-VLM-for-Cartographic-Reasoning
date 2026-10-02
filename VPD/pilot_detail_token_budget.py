"""Increase one Qwen VL image's actual token budget; preserve other image tensors.

Place in VLM_adaptation/VPD next to pilot_multimage_inference.py.
Uses its existing loader and multimodal helper. The selected image alone is
reprocessed; its image_pad tokens, grid and pixel payload are replaced together.
No repeated images, reasoning prefix or additional cue are inserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parent
DEFAULT_ADAPTER = VLM_DIR / "Training_outputs" / "MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12" / "checkpoints" / "checkpoint-3000"
CONTEXT = (
    "You are given multiple views of the same map. Image 1 is the full map; "
    "the later image(s) are detail crops for visual verification. Use and only according to all "
    "images, especially the detail crop, before answering.\n\n"
)


def pad_runs(ids, pad_id):
    runs, i = [], 0
    while i < len(ids):
        if ids[i] != pad_id:
            i += 1
            continue
        start = i
        while i < len(ids) and ids[i] == pad_id:
            i += 1
        runs.append((start, i))
    return runs


def visual_counts(grids, merge_size):
    counts = []
    for t, h, w in grids:
        raw = t * h * w
        if h % merge_size or w % merge_size:
            raise ValueError("Grid dimensions must be divisible by merge_size")
        counts.append(raw // merge_size ** 2)
    return counts


def one_special_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != text:
        raise ValueError(f"Unsupported tokenizer: {text} is not a single special token")
    return ids[0]


def replace_selected(inputs, new_image, index, processor, torch):
    """Maintain image/token alignment; only selected visual payload changes."""
    allowed = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids",
               "pixel_values", "image_grid_thw"}
    if set(inputs) - allowed:
        raise ValueError(f"Unsupported helper fields: {sorted(set(inputs) - allowed)}")
    base = {k: v.detach().cpu().clone() for k, v in inputs.items()}
    if base["input_ids"].shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")
    if "attention_mask" in base and not bool(base["attention_mask"].eq(1).all()):
        raise ValueError("This single-example pilot expects unpadded input")
    grids = base["image_grid_thw"].tolist()
    merge = int(processor.image_processor.merge_size)
    counts = visual_counts(grids, merge)
    image_id = one_special_id(processor.tokenizer, "<|image_pad|>")
    runs = pad_runs(base["input_ids"][0].tolist(), image_id)
    if len(runs) != len(grids) or [b - a for a, b in runs] != counts:
        raise ValueError("Original image grids do not match image_pad spans")
    if not 0 <= index < len(grids):
        raise ValueError("Detail index out of range")
    if new_image["image_grid_thw"].shape[0] != 1:
        raise ValueError("Expected exactly one reprocessed image")
    new_grid = new_image["image_grid_thw"][0].detach().cpu()
    new_count = visual_counts([new_grid.tolist()], merge)[0]
    old_raw = [math.prod(g) for g in grids]
    if base["pixel_values"].shape[0] != sum(old_raw):
        raise ValueError("Unsupported pixel_values layout; expected raw patch rows")
    new_pixels = new_image["pixel_values"].detach().cpu().to(base["pixel_values"].dtype)
    if new_pixels.shape[0] != math.prod(new_grid.tolist()):
        raise ValueError("Reprocessed pixel layout does not match grid")
    if new_pixels.shape[1:] != base["pixel_values"].shape[1:]:
        raise ValueError("Original/new image patch feature shapes differ")
    a, b = runs[index]
    result = dict(base)
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
        if key not in base:
            continue
        # Preserve the original selected-image modality ID, rather than assume it.
        fill = image_id if key == "input_ids" else (1 if key == "attention_mask" else int(base[key][0, a]))
        mid = torch.full((1, new_count), fill, dtype=base[key].dtype)
        result[key] = torch.cat([base[key][:, :a], mid, base[key][:, b:]], dim=1)
    patch_start = sum(old_raw[:index])
    patch_end = patch_start + old_raw[index]
    result["pixel_values"] = torch.cat(
        [base["pixel_values"][:patch_start], new_pixels, base["pixel_values"][patch_end:]], dim=0)
    result["image_grid_thw"] = base["image_grid_thw"].clone()
    result["image_grid_thw"][index] = new_grid.to(result["image_grid_thw"].dtype)
    # Verify every OTHER image's actual tensor and grid, including the full map.
    new_offset = 0
    old_offset = 0
    unchanged = []
    for i, old_size in enumerate(old_raw):
        new_size = math.prod(result["image_grid_thw"][i].tolist())
        if i != index:
            ok = torch.equal(base["image_grid_thw"][i], result["image_grid_thw"][i]) and torch.equal(
                base["pixel_values"][old_offset:old_offset + old_size],
                result["pixel_values"][new_offset:new_offset + new_size])
            if not ok:
                raise AssertionError(f"Untargeted image {i + 1} changed")
            unchanged.append(i + 1)
        old_offset += old_size
        new_offset += new_size
    actual_counts = visual_counts(result["image_grid_thw"].tolist(), merge)
    new_runs = pad_runs(result["input_ids"][0].tolist(), image_id)
    if [y - x for x, y in new_runs] != actual_counts:
        raise AssertionError("Final image token alignment failed")
    return result, counts, actual_counts, unchanged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--detail-index", type=int, default=2, help="1-based image number to change (default 2)")
    p.add_argument("--detail-tokens", type=int, default=0, help="Approximate target token count; 0 = original preprocessing")
    p.add_argument("--detail-multiplier", type=float, help="Target this multiple of measured baseline detail tokens; mutually exclusive with nonzero --detail-tokens")
    p.add_argument("--prompt-file", type=Path, help="Exact FULL existing prompt, UTF-8; avoids changing wording during comparison")
    p.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--thinking-mode", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--inspect-only", action="store_true", help="Print/save preprocessing metadata without generating an answer")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    if args.detail_tokens < 0 or args.max_new_tokens < 1:
        raise ValueError("Invalid token budget")
    if args.detail_multiplier is not None and (args.detail_multiplier <= 0 or args.detail_tokens):
        raise ValueError("Use positive --detail-multiplier OR nonzero --detail-tokens")
    paths = [x.expanduser().resolve() for x in args.images]
    index = args.detail_index - 1
    if not 0 <= index < len(paths):
        raise ValueError("--detail-index must identify an input image")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"))
    import inference_mapwise_trl as inference
    import torch
    import transformers
    from PIL import Image
    model, processor, loaded_adapter = inference.load_model_and_processor(
        adapter_path=args.adapter_path.expanduser().resolve())
    model.eval()
    prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else CONTEXT + inference.build_mapwise_prompt(args.question)
    images = []
    try:
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        inputs = inference.prepare_multimodal_inputs_multi(
            processor, images, prompt, thinking_mode=args.thinking_mode)
        ip = processor.image_processor
        merge = int(ip.merge_size)
        patch = int(ip.patch_size)
        original_grids = inputs["image_grid_thw"].detach().cpu().tolist()
        before = visual_counts(original_grids, merge)
        if len(before) != len(paths):
            raise ValueError("Helper image count does not match input paths")
        target = args.detail_tokens
        if args.detail_multiplier is not None:
            target = math.ceil(before[index] * args.detail_multiplier)
        unchanged = list(range(1, len(paths) + 1))
        after = before
        if target:
            factor = patch * merge
            # In Qwen image processors these size fields are PIXEL AREAS, not
            # side lengths. A narrow min/max range forces small crops to upscale.
            pixel_budget = target * factor * factor
            new_image = ip(images=[images[index]], return_tensors="pt", do_resize=True,
                           size={"shortest_edge": pixel_budget, "longest_edge": pixel_budget})
            inputs, before, after, unchanged = replace_selected(inputs, new_image, index, processor, torch)
            if target > before[index] and after[index] <= before[index]:
                raise RuntimeError("Detail token count did not increase; requested size may have been ignored")
        grids = inputs["image_grid_thw"].detach().cpu().tolist()
        info = [{"image_number": i + 1, "path": str(path), "original_width_height": list(im.size),
                 "original_grid_thw": original_grids[i], "actual_grid_thw": grids[i],
                 "original_visual_tokens": before[i], "actual_visual_tokens": after[i],
                 "processed_height_width": [grids[i][1] * patch, grids[i][2] * patch]}
                for i, (path, im) in enumerate(zip(paths, images))]
    finally:
        for im in images:
            im.close()
    for item in info:
        print(f"Image {item['image_number']}: {item['original_visual_tokens']} -> {item['actual_visual_tokens']} visual tokens; grid={item['actual_grid_thw']}", flush=True)
    result = {"question": args.question, "prompt": prompt, "images": info,
              "source_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
              "adapter_path": str(loaded_adapter) if loaded_adapter else None,
              "thinking_mode": args.thinking_mode, "requested_detail_tokens": target,
              "detail_multiplier": args.detail_multiplier,
              "untargeted_image_tensors_verified_unchanged": unchanged,
              "patch_size": patch, "merge_size": merge,
              "total_visual_tokens": sum(after),
              "input_tokens": inputs["input_ids"].shape[1],
              "torch_version": torch.__version__, "transformers_version": transformers.__version__,
              "note": "Upsampling allocates tokens to existing evidence; it does not create new captured detail."}
    if not args.inspect_only:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        for module in model.modules():
            if "rope_deltas" in vars(module):
                module.rope_deltas = None
        inputs = inference.move_inputs_to_model_device(inputs, model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                                       num_beams=1, num_return_sequences=1, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(tokens[None, :], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        result.update(raw_response=raw, final_answer=inference.extract_final_answer(raw),
                      generated_tokens=int(tokens.numel()),
                      generation_status="complete" if len(tokens) and int(tokens[-1]) in eos_ids else "token_limit_or_other_stop",
                      inference_seconds=round(time.perf_counter() - started, 3))
        print(raw, flush=True)
    destination = args.output_json.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
