#!/usr/bin/env python3
"""Evaluate causal Relay3 inference ablations with one frozen checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
RELAY_ROOT = HERE.parent
VLM_ROOT = HERE.parents[2]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (HERE, RELAY_ROOT, SPARSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ablation_decode import ALL_MODES, generate_relay3_ablation
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt, evaluate_answer
from evaluate_core import load_evaluation_questions, load_policy
from Image_question_CoT_shared_relay.dynamic_decode import generate_with_cot_relay
from Image_question_CoT_shared_relay.layout import build_prefill as build_relay_prefill
from relay_module import RelayModule
from rolling_sparse_qwen3vl import load_model
from utils import (
    build_adapter_only_prefill,
    decoder_hidden_size,
    encode_global_image,
    generate_from_relays,
    qwen_lm,
)


VARIANTS = ("full", "relay_off", "no_image_read", "no_current_x_read", "static_relay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", choices=("mapverse", "mapwise"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=4000)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--num-relay-tokens", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.checkpoint_step < 1 or args.num_relay_tokens < 1:
        parser.error("checkpoint-step and num-relay-tokens must be positive")
    if args.max_new_tokens < 1 or args.max_pixels < 1:
        parser.error("generation limits must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must not contain duplicates")
    return args


def eos_ids(model) -> set[int]:
    value = qwen_lm(model).generation_config.eos_token_id
    return set(map(int, value)) if isinstance(value, (list, tuple, set)) else {int(value)}


def completed_rows(path: Path, overwrite: bool) -> tuple[dict[tuple[str, int], dict], str]:
    if overwrite or not path.is_file():
        return {}, "w"
    rows: dict[tuple[str, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["variant"]), int(row["dataset_index"]))
        if key in rows:
            raise ValueError(f"Duplicate completed row {key}")
        rows[key] = row
    return rows, "a"


def generate_variant(model, variant: str, source, features, augmented):
    if variant == "relay_off":
        prompt, _ = build_adapter_only_prefill(source, features)
        return generate_from_relays(model, **prompt, do_sample=False, num_beams=1,
                                    max_new_tokens=ARGS.max_new_tokens, use_cache=True)
    if variant == "full":
        return generate_with_cot_relay(
            model, **augmented, do_sample=False, num_beams=1,
            max_new_tokens=ARGS.max_new_tokens, use_cache=True,
        )
    if variant not in ALL_MODES:
        raise ValueError(variant)
    return generate_relay3_ablation(
        model, **augmented, ablation_mode=variant, do_sample=False, num_beams=1,
        max_new_tokens=ARGS.max_new_tokens, use_cache=True,
    )


def write_summary(path: Path, rows: list[dict], variants: list[str]) -> None:
    question_count = len({int(row["dataset_index"]) for row in rows})
    summary = {"dataset": ARGS.dataset_name, "checkpoint_step": ARGS.checkpoint_step,
               "questions": question_count, "variants": {}}
    for variant in variants:
        part = [row for row in rows if row["variant"] == variant]
        if not part:
            continue
        terminated = [row for row in part if row["terminated"]]
        correct_terminated = [row for row in part if row["terminated"] and row["reward"] == 1.0]
        summary["variants"][variant] = {
            "count": len(part),
            "accuracy": sum(row["reward"] for row in part) / len(part),
            "termination_rate": len(terminated) / len(part),
            "mean_generated_tokens": sum(row["generated_tokens"] for row in part) / len(part),
            "mean_tokens_correct_terminated": (
                sum(row["generated_tokens"] for row in correct_terminated) / len(correct_terminated)
                if correct_terminated else None
            ),
            "mean_inference_seconds": sum(row["inference_seconds"] for row in part) / len(part),
        }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global ARGS, RECORDS
    ARGS = parse_args()
    checkpoint = ARGS.run_dir / f"checkpoint-{ARGS.checkpoint_step}.pt"
    adapter = ARGS.run_dir / f"checkpoint-{ARGS.checkpoint_step}"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Incomplete Relay3 checkpoint under {ARGS.run_dir}")

    RECORDS = load_evaluation_questions(ARGS.dataset, ARGS.mapverse_root, ARGS.mapwise_root)
    end = len(RECORDS) if ARGS.end_index is None else min(ARGS.end_index, len(RECORDS))
    if not 0 <= ARGS.start_index < end:
        raise ValueError(f"Invalid row interval [{ARGS.start_index},{end})")
    selected = list(enumerate(RECORDS[ARGS.start_index:end], start=ARGS.start_index + 1))

    destination = ARGS.output_dir / ARGS.dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    response_path = destination / "responses.jsonl"
    completed, mode = completed_rows(response_path, ARGS.overwrite)

    model, processor = load_model(ARGS.model, adapter, load_in_4bit=ARGS.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = RelayModule(
        decoder_hidden_size(model), num_relays=ARGS.num_relay_tokens
    ).to(device).eval()
    state = load_policy(
        policy, checkpoint, "dynamic_cot_relay", "image_question_cot_shared"
    )
    saved_count = int(state.get("num_relay_tokens", ARGS.num_relay_tokens))
    if saved_count != ARGS.num_relay_tokens:
        raise ValueError(
            f"Checkpoint relay count {saved_count} != requested {ARGS.num_relay_tokens}"
        )
    manifest = {
        "dataset": str(ARGS.dataset), "dataset_name": ARGS.dataset_name,
        "run_dir": str(ARGS.run_dir), "checkpoint": str(checkpoint),
        "adapter": str(adapter), "checkpoint_step": ARGS.checkpoint_step,
        "num_relay_tokens": ARGS.num_relay_tokens, "variants": ARGS.variants,
        "max_new_tokens": ARGS.max_new_tokens, "max_pixels": ARGS.max_pixels,
        "row_interval_1based_inclusive": [ARGS.start_index + 1, end],
        "definitions": {
            "full": "native trained Relay3 generation",
            "relay_off": "same trained adapter, no relay positions or relay computation",
            "no_image_read": "dynamic relay cannot query image KV; x_t keeps native image access",
            "no_current_x_read": "dynamic relay cannot query current x_t; x_t still queries relay",
            "static_relay": "compute relay on first decoding step, retain its layer-wise KV, never refresh",
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    try:
        with response_path.open(mode, encoding="utf-8") as handle:
            for ordinal, (dataset_index, item) in enumerate(selected, 1):
                pending = [v for v in ARGS.variants if (v, dataset_index) not in completed]
                if not pending:
                    print(f"[{ordinal}/{len(selected)}] skip index={dataset_index}", flush=True)
                    continue
                image_started = time.perf_counter()
                with Image.open(item.image_path) as image_file:
                    image = _resize_to_pixel_cap(image_file.convert("RGB"), ARGS.max_pixels)
                source = _content_inputs(
                    processor, [image], build_cartographic_prompt(item.question),
                    device, ARGS.max_pixels,
                )
                features = encode_global_image(model, source)
                image_seconds = time.perf_counter() - image_started
                augmented, _ = build_relay_prefill(
                    model, processor.tokenizer, source, features, policy, with_grad=False
                )
                prefix_length = source["input_ids"].shape[1]

                # Rotate order across questions so warm-up or thermal effects do not
                # systematically favor one timing condition.
                offset = (dataset_index - 1) % len(ARGS.variants)
                ordered = ARGS.variants[offset:] + ARGS.variants[:offset]
                ordered = [v for v in ordered if v in pending]
                for order_index, variant in enumerate(ordered):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generation_started = time.perf_counter()
                    with torch.inference_mode():
                        generated = generate_variant(model, variant, source, features, augmented)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    generation_seconds = time.perf_counter() - generation_started
                    tokens = generated[0, prefix_length:]
                    response = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                    evaluation = evaluate_answer(response, item)
                    terminated = bool(tokens.numel() and int(tokens[-1]) in eos_ids(model))
                    row = {
                        "dataset_name": ARGS.dataset_name,
                        "dataset_index": dataset_index,
                        "qa_id": item.sample_id,
                        "variant": variant,
                        "variant_order_within_question": order_index,
                        "question": item.question,
                        "ground_truth": item.answer,
                        "response": response,
                        "reward": float(evaluation["reward"]),
                        "evaluation": evaluation,
                        "generated_tokens": int(tokens.numel()),
                        "terminated": terminated,
                        "shared_image_encoding_seconds": image_seconds,
                        "generation_seconds": generation_seconds,
                        "inference_seconds": image_seconds + generation_seconds,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[(variant, dataset_index)] = row
                    print(
                        f"[{ordinal}/{len(selected)}] index={dataset_index} {variant} "
                        f"reward={row['reward']:.0f} tokens={row['generated_tokens']} "
                        f"generation_s={generation_seconds:.3f}", flush=True,
                    )
                    del generated, tokens
                del source, features, augmented
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        del policy, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = list(completed.values())
    write_summary(destination / "summary.json", rows, list(ARGS.variants))


if __name__ == "__main__":
    main()
