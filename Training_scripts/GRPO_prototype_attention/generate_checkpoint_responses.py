#!/usr/bin/env python3
"""Generate full greedy reasoning text from one prototype-attention checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[1]
for path in (VLM_ROOT / "SparseVLM_eliminateCONTEXT",
             VLM_ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt, evaluate_answer, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model

from prototype_cross_attention_policy import PrototypeCrossAttentionPolicy, decoder_hidden_size
from prototype_prefill import (build_prototype_prefill, encode_global_image,
                               generate_from_preencoded_vision, qwen_lm)


def render_readable(results: list[dict]) -> str:
    """Render responses without relying on adjacent-string precedence."""
    sections = []
    for row in results:
        lines = [
            "=" * 100,
            f"DATASET INDEX: {row['dataset_index']}",
            f"QA ID: {row['qa_id']}",
            f"QUESTION: {row['question']}",
            f"GROUND TRUTH: {row['ground_truth']}",
            f"REWARD: {row['reward']}",
            f"GENERATED TOKENS: {row['generated_tokens']}",
            f"TERMINATED: {row['terminated']}",
            "-" * 100,
            row["response"].rstrip(),
        ]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-indices", type=int, nargs="+", default=[245, 246, 247, 248])
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    if state.get("placement") != "after_question":
        raise ValueError(f"Expected after_question checkpoint, got {state.get('placement')!r}")
    policy_state = state["policy"]
    if any(key.startswith("question_attention.") for key in policy_state):
        raise ValueError("This script invocation expects the image_only prototype policy")
    num_prototypes = int(state.get("num_prototypes", policy_state["prototypes"].shape[0]))
    num_heads = int(state.get("attention_heads", 8))

    records = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root
    )
    for index in args.dataset_indices:
        if not 1 <= index <= len(records):
            parser.error(f"dataset index {index} is outside 1..{len(records)}")

    model, processor = load_model(args.model, args.adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = PrototypeCrossAttentionPolicy(
        decoder_hidden_size(model), num_prototypes=num_prototypes,
        num_heads=num_heads, train_projections=False,
    ).to(device).eval()
    missing, unexpected = policy.load_state_dict(policy_state, strict=False)
    allowed_missing = {"cross_attention.in_proj_weight", "cross_attention.out_proj.weight"}
    if unexpected or set(missing) - allowed_missing:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset_index in args.dataset_indices:
        item = records[dataset_index - 1]
        with Image.open(item.image_path) as image_file:
            image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
        source = _content_inputs(
            processor, [image], build_cartographic_prompt(item.question), device, args.max_pixels
        )
        global_features = encode_global_image(model, source)
        augmented, attention = build_prototype_prefill(
            model, processor.tokenizer, source, global_features,
            policy, "after_question", with_grad=False,
        )
        prefix_length = augmented["input_ids"].shape[1]
        generated = generate_from_preencoded_vision(
            model, **augmented, do_sample=False, num_beams=1,
            max_new_tokens=args.max_new_tokens, use_cache=True,
        )
        completion = generated[0, prefix_length:]
        response = processor.tokenizer.decode(completion, skip_special_tokens=True)
        eos = qwen_lm(model).generation_config.eos_token_id
        eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
        evaluation = evaluate_answer(response, item)
        row = {
            "checkpoint": str(args.checkpoint),
            "dataset_index": dataset_index,
            "qa_id": item.sample_id,
            "dataset": item.metadata.get("dataset", "mapverse"),
            "question": item.question,
            "ground_truth": item.answer,
            "response": response,
            "reward": float(evaluation["reward"]),
            "evaluation": evaluation,
            "generated_tokens": int(completion.numel()),
            "terminated": bool(completion.numel() and int(completion[-1]) in eos_ids),
            "prototype_attention_entropy": float(
                -(attention.float().clamp_min(1e-12)
                  * attention.float().clamp_min(1e-12).log()).sum(dim=-1).mean()
            ),
        }
        results.append(row)
        print(f"[GREEDY] index={dataset_index} qa_id={item.sample_id} "
              f"reward={row['reward']} tokens={row['generated_tokens']}", flush=True)

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.output_text.write_text(render_readable(results), encoding="utf-8")
    print(f"Saved JSONL: {args.output_jsonl}", flush=True)
    print(f"Saved readable text: {args.output_text}", flush=True)


if __name__ == "__main__":
    main()
