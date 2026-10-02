"""Qwen3-VL assistant-prefix continuation pilot; place beside the original pilot.

Uses the same inference_mapwise_trl loader/preprocessor as pilot_multimage_inference.py.
Keep is exact token-prefix replay. Remove deletes complete vision spans and rebuilds
the cache. Refresh inserts another processed crop INSIDE the assistant stream.
Refresh is an exploratory intervention, not a reproduction of TVC/PVC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parent
DEFAULT_ADAPTER = VLM_DIR / "Training_outputs" / "MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12" / "checkpoints" / "checkpoint-3000"
DEFAULT_CONTEXT = (
    "You are given multiple views of the same map. Image 1 is the full map; "
    "the later image(s) are detail crops for visual verification. Use and only according to all "
    "images, especially the detail crop, before answering.\n\n"
)
DEFAULT_CUE = (
    "\nRecheck the visible map before the next judgment. Use the black dot as the "
    "city location, not the text label. Read the fill immediately surrounding the dot.\n"
)


def vision_keep_indices(ids, start_id, end_id):
    """Delete entire vision spans, including delimiters; retain all other tokens."""
    keep, inside, spans = [], False, 0
    for i, token in enumerate(ids):
        if token == start_id:
            if inside:
                raise ValueError("Nested vision spans")
            inside = True
            spans += 1
        elif token == end_id:
            if not inside:
                raise ValueError("Unmatched vision_end")
            inside = False
        elif not inside:
            keep.append(i)
    if inside:
        raise ValueError("Unclosed vision span")
    if not spans:
        raise ValueError("No vision spans found; this script requires Qwen VL tokens")
    return keep, spans


def token_boundary_before_text(ids, tokenizer, needle):
    """Locate a decoded match but slice ORIGINAL ids, never retokenize the prefix.

    Select the largest token boundary not crossing the match. If a token contains
    both preceding whitespace and the match, omit that token as a whole.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    target = text.find(needle)
    if target < 0:
        raise ValueError(f"Cut marker absent from baseline: {needle!r}")
    best = 0
    for n in range(len(ids) + 1):
        decoded = tokenizer.decode(ids[:n], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if len(decoded) <= target and text.startswith(decoded):
            best = n
    return best


def reasoning_limit(ids, tokenizer):
    """Stop before think closure/EOS; do not generate from a completed answer."""
    limits = [len(ids)]
    for marker in ("</think>", "<|im_end|>", "<|endoftext|>"):
        text = tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        if marker in text:
            limits.append(token_boundary_before_text(ids, tokenizer, marker))
    return min(limits)


def special_id(tokenizer, token):
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1 or tokenizer.convert_ids_to_tokens(encoded[0]) != token:
        raise ValueError(f"Expected Qwen special token {token!r}")
    return encoded[0]


def cpu_inputs(inputs):
    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw",
               "mm_token_type_ids", "token_type_ids"}
    unexpected = set(inputs) - allowed
    if unexpected:
        raise ValueError(f"Unsupported preprocessor fields: {sorted(unexpected)}; inspect before running")
    out = {k: v.detach().cpu().clone() for k, v in inputs.items()}
    if out["input_ids"].shape[0] != 1:
        raise ValueError("Only batch size 1 is supported")
    return out


def append_inputs(base, extension_ids, torch, visual_block=None):
    """Rebuild sequence fields and concatenate image payloads in token order."""
    out = {k: v.clone() for k, v in base.items()}
    extra = torch.tensor([extension_ids], dtype=out["input_ids"].dtype)
    out["input_ids"] = torch.cat([out["input_ids"], extra], dim=1)
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in base:
            if visual_block is not None:
                # Modern Qwen uses mm_token_type_ids (1=image). Older releases
                # infer modalities from image_token_id and do not return this field.
                block_types = visual_block.get(key)
                if block_types is None:
                    raise ValueError(f"Refresh processor did not provide {key}")
                if block_types.shape[1] != len(extension_ids):
                    raise ValueError("Refresh token-type length mismatch")
                types = block_types.to(base[key].dtype)
            else:
                types = torch.zeros((1, len(extension_ids)), dtype=base[key].dtype)
            out[key] = torch.cat([base[key], types], dim=1)
    if visual_block is not None:
        for key in ("pixel_values", "image_grid_thw"):
            if key not in visual_block or key not in out:
                raise ValueError(f"Missing image payload {key}")
            out[key] = torch.cat([out[key], visual_block[key]], dim=0)
    return out


def remove_images(base, tokenizer, torch):
    indices, spans = vision_keep_indices(
        base["input_ids"][0].tolist(), special_id(tokenizer, "<|vision_start|>"),
        special_id(tokenizer, "<|vision_end|>"))
    out = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
        if key in base:
            out[key] = base[key][:, indices].clone()
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    return out, spans


def reset_rope_state(model):
    # Qwen implementations can retain rope_deltas on a nested module across calls.
    # Always rebuild positions for each condition; never reuse past_key_values.
    for module in model.modules():
        if "rope_deltas" in vars(module):
            module.rope_deltas = None


def run_generation(model, processor, inputs, inference, torch, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    reset_rope_state(model)
    moved = inference.move_inputs_to_model_device(
        {k: v.clone() for k, v in inputs.items()}, model)
    kwargs = dict(do_sample=args.sample, max_new_tokens=args.max_new_tokens,
                  num_beams=1, num_return_sequences=1, use_cache=True)
    if args.sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**moved, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ids = generated[0, moved["input_ids"].shape[1]:].detach().cpu().tolist()
    raw = processor.tokenizer.decode(ids, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return {"token_ids": ids, "raw_response": raw,
            "generated_tokens": len(ids), "seed": seed,
            "generation_status": "complete" if ids and ids[-1] in eos else "token_limit_or_other_stop",
            "inference_seconds": round(time.perf_counter() - started, 3)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="+", type=Path, required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--thinking-mode", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--context-file", type=Path, help="Replace ONLY the multi-view introduction with exact old wording")
    p.add_argument("--fractions", nargs="+", type=float, default=[0, .25, .5, .75])
    p.add_argument("--cut-before", action="append", default=[], help="First exact occurrence in baseline reasoning, e.g. Bakhmut:")
    p.add_argument("--only-markers", action="store_true", help="Skip fraction checkpoints")
    p.add_argument("--conditions", nargs="+", choices=("keep", "remove", "cue", "refresh"),
                   default=["keep", "remove", "cue", "refresh"])
    p.add_argument("--refresh-image", type=Path, help="Default: last original image; specify the existing detail crop")
    p.add_argument("--cue-file", type=Path)
    p.add_argument("--probe-question", help="Optional SEPARATE diagnostic question after each keep prefix, and directly with full views / crop")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--sample", action="store_true")
    p.add_argument("--temperature", type=float, default=.8)
    p.add_argument("--top-p", type=float, default=.95)
    p.add_argument("--reuse-baseline", type=Path, help="baseline_state.pt produced by THIS script; avoids regenerating baseline")
    p.add_argument("--baseline-only", action="store_true", help="Generate/save baseline first; inspect it before selecting exact cut markers")
    return p


def main():
    args = parser().parse_args()
    if args.max_new_tokens < 1 or any(not 0 <= f < 1 for f in args.fractions):
        raise ValueError("Require max-new-tokens > 0 and 0 <= fractions < 1")
    if args.only_markers and not args.cut_before:
        raise ValueError("--only-markers requires --cut-before")
    if not args.sample and len(args.seeds) != 1:
        raise ValueError("Multiple seeds require --sample; greedy repeats are redundant")
    if not args.temperature > 0 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    paths = [p.expanduser().resolve() for p in args.images]
    refresh_path = (args.refresh_image or paths[-1]).expanduser().resolve()
    for path in paths + [refresh_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    outdir = args.output_dir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"))
    import inference_mapwise_trl as inference
    import torch
    import transformers
    from PIL import Image

    model, processor, loaded = inference.load_model_and_processor(
        adapter_path=args.adapter_path.expanduser().resolve())
    model.eval()
    tokenizer = processor.tokenizer
    context = args.context_file.read_text(encoding="utf-8") if args.context_file else DEFAULT_CONTEXT
    cue = (
        "\n" + args.cue_file.read_text(encoding="utf-8-sig").strip() + "\n"
        if args.cue_file else DEFAULT_CUE
    )
    prompt = context + inference.build_mapwise_prompt(args.question)
    def prepare(image_paths, text):
        images = []
        try:
            for path in image_paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            return cpu_inputs(inference.prepare_multimodal_inputs_multi(
                processor, images, text, thinking_mode=args.thinking_mode))
        finally:
            for im in images:
                im.close()

    metadata = {"images": [str(p) for p in paths], "question": args.question,
                "prompt": prompt, "adapter_path": str(loaded) if loaded else None,
                "requested_adapter_path": str(args.adapter_path.expanduser().resolve()),
                "thinking_mode": args.thinking_mode,
                "image_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
    if args.reuse_baseline:
        state = torch.load(args.reuse_baseline, map_location="cpu", weights_only=True)
        if state["metadata"] != metadata:
            raise ValueError("Baseline metadata mismatch (images/question/prompt/adapter/thinking)")
        base, baseline = state["inputs"], state["baseline"]
        # Check tokenizer consistency before using token IDs saved elsewhere.
        if state["special_ids"] != {t: special_id(tokenizer, t) for t in state["special_ids"]}:
            raise ValueError("Baseline tokenizer mismatch")
    else:
        base = prepare(paths, prompt)
        baseline = run_generation(model, processor, base, inference, torch, args, args.seeds[0])
        baseline["final_answer"] = inference.extract_final_answer(baseline["raw_response"])
        state = {"metadata": metadata, "inputs": base, "baseline": baseline,
                 "special_ids": {t: special_id(tokenizer, t) for t in
                                 ("<|vision_start|>", "<|vision_end|>", "<|im_start|>", "<|im_end|>")}}
        torch.save(state, outdir / "baseline_state.pt")
    write_json(outdir / "baseline.json", {**metadata, **baseline})
    print("Baseline:", baseline["final_answer"], flush=True)
    if args.baseline_only:
        print(f"Saved baseline to {outdir}; no interventions run", flush=True)
        return
    ids = baseline["token_ids"]
    limit = reasoning_limit(ids, tokenizer)
    checkpoints = {}
    if not args.only_markers:
        for f in args.fractions:
            checkpoints.setdefault(math.floor(limit * f), []).append(f"fraction_{f:g}")
    for marker in args.cut_before:
        n = token_boundary_before_text(ids, tokenizer, marker)
        if n >= limit:
            raise ValueError(f"Marker {marker!r} is outside the open reasoning")
        checkpoints.setdefault(n, []).append(f"before_{marker}")
    removed, span_count = remove_images(base, tokenizer, torch)
    refresh_block = None
    if "refresh" in args.conditions:
        with Image.open(refresh_path) as im:
            image = im.convert("RGB")
        try:
            # No chat template here: this block is inserted in the OPEN assistant
            # stream, after the preserved reasoning tokens, with no role transition.
            refresh_block = cpu_inputs(processor(
                text=[cue + "<|vision_start|><|image_pad|><|vision_end|>\n"],
                images=[image], return_tensors="pt", padding=False,
                add_special_tokens=False))
        finally:
            image.close()
    report = {**metadata, "transformers_version": transformers.__version__,
              "torch_version": torch.__version__, "baseline": baseline,
              "removed_vision_spans": span_count, "refresh_image": str(refresh_path),
              "cue": cue, "sampling": args.sample,
              "warnings": ["No automatic visual-forgetting verdict.",
                           "Remove changes sequence length/positions and retains visual facts in text.",
                           "Refresh adds tokens and is not an exact PVC implementation.",
                           "A probe is a separate prompted branch, not an observation of hidden state."],
              "runs": [], "probes": []}
    # Incremental output protects results if a later condition fails.
    def save():
        write_json(outdir / "results.json", report)
    save()
    if args.probe_question:
        probe_text = context + inference.build_mapwise_prompt(args.probe_question)
        for name, image_paths in (("direct_full_views", paths), ("direct_crop", [refresh_path])):
            pi = prepare(image_paths, probe_text if name == "direct_full_views" else
                         inference.build_mapwise_prompt(args.probe_question))
            for seed in args.seeds:
                result = run_generation(model, processor, pi, inference, torch, args, seed)
                report["probes"].append({"condition": name, "question": args.probe_question,
                                          **result, "final_answer": inference.extract_final_answer(result["raw_response"])})
                save()
    for n, names in sorted(checkpoints.items()):
        prefix_ids = ids[:n]
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
        keep = append_inputs(base, prefix_ids, torch)
        for condition in args.conditions:
            if condition == "keep":
                ci = keep
            elif condition == "remove":
                ci = append_inputs(removed, prefix_ids, torch)
            elif condition == "cue":
                ci = append_inputs(keep, tokenizer.encode(cue, add_special_tokens=False), torch)
            else:
                ci = append_inputs(keep, refresh_block["input_ids"][0].tolist(), torch,
                                   visual_block=refresh_block)
            for seed in args.seeds:
                result = run_generation(model, processor, ci, inference, torch, args, seed)
                combined = tokenizer.decode(prefix_ids + result["token_ids"],
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False).strip()
                record = {"checkpoint_tokens": n, "checkpoint_labels": names,
                          "prefix_text": prefix, "prefix_token_ids": prefix_ids,
                          "condition": condition, "input_tokens": ci["input_ids"].shape[1],
                          **result, "combined_response": combined,
                          "final_answer": inference.extract_final_answer(combined)}
                if condition == "keep" and not args.sample:
                    expected = ids[n:n + len(result["token_ids"])]
                    # Baseline may have exhausted its budget; only compare overlap.
                    overlap = min(len(expected), len(result["token_ids"]))
                    actual = result["token_ids"][:overlap]
                    record["baseline_overlap_tokens"] = overlap
                    record["baseline_overlap_matches"] = actual == expected[:overlap]
                    record["first_divergence_token"] = next(
                        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
                report["runs"].append(record)
                save()
                print(f"cut={n} {condition} seed={seed}: {record['final_answer']}", flush=True)
        if args.probe_question:
            # Close the current assistant and ask an independent diagnostic. It
            # deliberately changes the task, so it is NEVER counted as continuation.
            open_text = tokenizer.decode(keep["input_ids"][0].tolist(), skip_special_tokens=False)
            close_think = "</think>\n" if open_text.rfind("<think>") > open_text.rfind("</think>") else ""
            tail = (close_think + "<|im_end|>\n<|im_start|>user\n" +
                    inference.build_mapwise_prompt(args.probe_question) +
                    "<|im_end|>\n<|im_start|>assistant\n")
            # Match the thinking opener of the ORIGINAL generation header.
            original_text = tokenizer.decode(base["input_ids"][0].tolist(), skip_special_tokens=False)
            header = original_text.rsplit("<|im_start|>assistant", 1)[-1]
            if "<think>" in header:
                tail += "<think>\n"
            pi = append_inputs(keep, tokenizer.encode(tail, add_special_tokens=False), torch)
            for seed in args.seeds:
                result = run_generation(model, processor, pi, inference, torch, args, seed)
                report["probes"].append({"condition": "after_prefix_new_user_question",
                                          "checkpoint_tokens": n, "prefix_text": prefix,
                                          "question": args.probe_question, **result,
                                          "final_answer": inference.extract_final_answer(result["raw_response"])})
                save()
    print(f"Saved {outdir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
