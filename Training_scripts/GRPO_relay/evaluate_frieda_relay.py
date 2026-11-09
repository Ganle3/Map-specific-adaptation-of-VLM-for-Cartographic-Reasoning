#!/usr/bin/env python3
"""Greedy FRIEDA evaluation shared by trained relay schemes 1, 2 and 3."""
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
RELAY_ROOT = HERE
VLM_ROOT = RELAY_ROOT.parents[1]
EVALUATION_ROOT = VLM_ROOT / "Evaluation_scripts"
for path in (RELAY_ROOT, EVALUATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_core import checkpoint_adapter_path, load_policy
from inference_frieda import (
    build_frieda_prompt,
    build_prediction_record,
    completed_question_refs,
    load_existing_predictions,
    load_json_list,
    replace_or_append_prediction,
    resolve_frieda_images,
    save_predictions,
)
from frieda_evaluation import evaluate_frieda, print_summary
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from rolling_sparse_qwen3vl import load_model
from relay_module import RelayModule
from utils import decoder_hidden_size, encode_global_image, generate_from_relays
from only_visual_relay.layout import build_prefill as build_only_visual_prefill
from Image_question_shared_relay.layout import build_prefill as build_shared_prefill
from Image_question_CoT_shared_relay.dynamic_decode import generate_with_cot_relay
from Image_question_CoT_shared_relay.layout import build_prefill as build_cot_prefill


SCHEMES = {
    "only_visual": {
        "placement": "after_vision_end",
        "builder": build_only_visual_prefill,
        "generator": generate_from_relays,
    },
    "image_question_shared": {
        "placement": "shared_dual_placement",
        "builder": build_shared_prefill,
        "generator": generate_from_relays,
    },
    "image_question_cot_shared": {
        "placement": "dynamic_cot_relay",
        "builder": build_cot_prefill,
        "generator": generate_with_cot_relay,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-kind", choices=tuple(SCHEMES), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--qa-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-relay-tokens", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=10_000)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if args.checkpoint_step < 1:
        parser.error("checkpoint-step must identify a trained checkpoint")
    if args.num_relay_tokens < 1 or args.max_new_tokens < 1 or args.max_pixels < 1:
        parser.error("relay count and token/pixel limits must be positive")
    return args


def main() -> None:
    args = parse_args()
    scheme = SCHEMES[args.policy_kind]
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = run_dir / f"checkpoint-{args.checkpoint_step}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    adapter = checkpoint_adapter_path(checkpoint)
    qa_json = args.qa_json.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prediction_json = output_dir / "frieda_predictions.json"
    for required in (qa_json, image_root, Path(args.judge_model)):
        if not required.exists():
            raise FileNotFoundError(required)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_json_list(qa_json)
    end_index = len(samples) if args.end_index is None else args.end_index
    if not 0 <= args.start_index <= end_index <= len(samples):
        raise ValueError(
            f"Invalid sample interval [{args.start_index}, {end_index}) for {len(samples)} rows"
        )
    if args.overwrite and prediction_json.exists():
        prediction_json.unlink()
    predictions = ([] if args.no_resume or args.overwrite
                   else load_existing_predictions(prediction_json))
    completed = completed_question_refs(predictions)

    print("=" * 80, flush=True)
    print("FRIEDA RELAY EVALUATION", flush=True)
    print(f"Policy kind:    {args.policy_kind}", flush=True)
    print(f"Placement:      {scheme['placement']}", flush=True)
    print(f"Run directory:  {run_dir}", flush=True)
    print(f"Checkpoint:     {checkpoint}", flush=True)
    print(f"Adapter:        {adapter}", flush=True)
    print(f"Relay tokens:   {args.num_relay_tokens}", flush=True)
    print(f"QA JSON:        {qa_json}", flush=True)
    print(f"Images:         {image_root}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}", flush=True)
    print(f"Max pixels:     {args.max_pixels}", flush=True)
    print(f"Predictions:    {prediction_json}", flush=True)
    print("=" * 80, flush=True)

    model, processor = load_model(
        args.model, adapter, load_in_4bit=args.load_in_4bit
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    policy = RelayModule(
        decoder_hidden_size(model), num_relays=args.num_relay_tokens,
    ).to(device).eval()
    state = load_policy(
        policy, checkpoint, scheme["placement"], args.policy_kind
    )
    if int(state.get("step", -1)) != args.checkpoint_step:
        raise ValueError(
            f"Checkpoint metadata step={state.get('step')}; expected {args.checkpoint_step}"
        )
    if int(state.get("num_relay_tokens", -1)) != args.num_relay_tokens:
        raise ValueError(
            "Checkpoint relay count does not match --num-relay-tokens: "
            f"{state.get('num_relay_tokens')} != {args.num_relay_tokens}"
        )

    try:
        for absolute_index in range(args.start_index, end_index):
            sample = samples[absolute_index]
            question_ref = str(sample.get("question_ref", f"frieda_{absolute_index}"))
            if not args.no_resume and question_ref in completed:
                print(
                    f"[{absolute_index + 1}/{len(samples)}] {question_ref}: skipped",
                    flush=True,
                )
                continue

            started = time.perf_counter()
            image_paths = resolve_frieda_images(sample, image_root)
            images = []
            for image_path in image_paths:
                with Image.open(image_path) as image_file:
                    images.append(
                        _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
                    )
            question = str(sample["question_text"]).strip()
            prompt = build_frieda_prompt(question)
            source_inputs = _content_inputs(
                processor, images, prompt, device, args.max_pixels
            )
            with torch.inference_mode():
                global_features = encode_global_image(model, source_inputs)
                augmented, _ = scheme["builder"](
                    model, processor.tokenizer, source_inputs, global_features,
                    policy, with_grad=False,
                )
                prefix_length = augmented["input_ids"].shape[1]
                generated = scheme["generator"](
                    model, **augmented, do_sample=False, num_beams=1,
                    max_new_tokens=args.max_new_tokens, use_cache=True,
                )
            completion = generated[:, prefix_length:]
            raw_response = processor.batch_decode(
                completion, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            record = build_prediction_record(
                sample=sample,
                sample_index=absolute_index,
                image_paths=image_paths,
                raw_response=raw_response,
                model_name=args.model or "Qwen/Qwen3-VL-8B-Thinking",
                adapter_path=adapter,
                max_new_tokens=args.max_new_tokens,
                inference_seconds=time.perf_counter() - started,
            )
            record.update({
                "relay_scheme": args.policy_kind,
                "relay_placement": scheme["placement"],
                "relay_checkpoint": str(checkpoint),
                "relay_checkpoint_step": args.checkpoint_step,
                "num_relay_tokens": args.num_relay_tokens,
                "max_pixels": args.max_pixels,
                "generated_tokens": int(completion.shape[1]),
            })
            replace_or_append_prediction(predictions, record)
            save_predictions(predictions, prediction_json)
            completed.add(question_ref)
            preview = record["final_answer"].replace("\n", " ")[:120]
            print(
                f"[{absolute_index + 1}/{len(samples)}] {question_ref}: "
                f"tokens={completion.shape[1]} answer={preview!r}",
                flush=True,
            )
            del source_inputs, global_features, augmented, generated, completion, images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del policy, model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_refs = {
        str(samples[index].get("question_ref", f"frieda_{index}"))
        for index in range(args.start_index, end_index)
    }
    completed_selected = completed.intersection(selected_refs)
    if len(completed_selected) != len(selected_refs):
        raise RuntimeError(
            f"Inference completed {len(completed_selected)}/{len(selected_refs)} selected samples"
        )

    summary = evaluate_frieda(
        prediction_json=prediction_json,
        qa_json=qa_json,
        output_dir=output_dir,
        orientation_pkl=(EVALUATION_ROOT / "orientation.pkl"
                         if (EVALUATION_ROOT / "orientation.pkl").exists() else None),
        distance_tolerance=0.20,
        judge_model=args.judge_model,
        judge_load_in_4bit=True,
    )
    print_summary(summary)
    print(json.dumps({
        "predictions": str(prediction_json),
        "summary": str(output_dir / "evaluation_summary.json"),
        "checkpoint_step": args.checkpoint_step,
        "num_relay_tokens": args.num_relay_tokens,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
