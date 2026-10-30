#!/usr/bin/env python3
"""Greedy evaluator shared by the native and dynamic relay schemes."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[1]
for path in (VLM_ROOT / "SparseVLM_eliminateCONTEXT",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import (MapQuestion, build_cartographic_prompt, evaluate_answer,
                  load_mapverse_questions)
from rolling_sparse_qwen3vl import load_model

from utils import (build_adapter_only_prefill, decoder_hidden_size,
                   encode_global_image, generate_from_relays)
from relay_module import RelayModule
from only_visual_relay.layout import build_prefill as build_only_visual_prefill
from Image_question_shared_relay.layout import build_prefill as build_shared_prefill
from Image_question_CoT_shared_relay.layout import build_prefill as build_cot_prefill
from Image_question_CoT_shared_relay.dynamic_decode import generate_with_cot_relay


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def resolve_mapwise_image(row: dict, root: Path) -> Path:
    country = str(row["country"]).strip().lower()
    map_no = str(row["map_no"]).strip()
    variant = str(row.get("map_type") or "with_annotations")
    base = root / country / "images" / variant / map_no
    candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in IMAGE_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Missing MapWise image country={country!r}, map_no={map_no!r}, variant={variant!r}"
    )


def load_evaluation_questions(dataset: Path, mapverse_root: Path,
                              mapwise_root: Path) -> list[MapQuestion]:
    rows = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {dataset}")
    # Unified easier and MapVerse files already use the training schema.
    if all("correct_answer" in row and "image_name" in row for row in rows):
        return load_mapverse_questions(dataset, mapverse_root, mapwise_root=mapwise_root)
    # Formal MapWise test400 uses ground_truth/map_no/country instead.
    required = {"country", "map_no", "question", "ground_truth", "ground_truth_type"}
    records = []
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"MapWise row {index} is missing {sorted(missing)}")
        metadata = dict(row)
        metadata["dataset"] = "mapwise"
        metadata.setdefault("qa_id", f"mapwise_{row['country']}_{row['map_no']}_{index}")
        records.append(MapQuestion(
            sample_id=str(metadata["qa_id"]),
            image_path=resolve_mapwise_image(row, mapwise_root),
            question=str(row["question"]), answer=str(row["ground_truth"]),
            answer_type=str(row["ground_truth_type"]), metadata=metadata,
        ))
    return records


def checkpoint_path(run_dir: Path, step: int) -> Path:
    path = run_dir / f"checkpoint-{step}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def checkpoint_adapter_path(checkpoint: Path) -> Path:
    """Return the PEFT directory saved beside a joint-training checkpoint."""
    state = torch.load(checkpoint, map_location="cpu")
    if not state.get("train_adapter", False):
        raise ValueError(f"Checkpoint did not train/save an adapter: {checkpoint}")
    # Use the path next to the checkpoint instead of the serialized absolute
    # path, so copied/moved Euler run directories remain evaluable.
    adapter = checkpoint.with_suffix("")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing checkpoint adapter directory: {adapter}")
    return adapter


def load_policy(policy, path: Path, expected_placement: str,
                expected_policy_kind: str) -> dict:
    state = torch.load(path, map_location="cpu")
    if state.get("placement") != expected_placement:
        raise ValueError(
            f"Checkpoint {path} placement={state.get('placement')!r}; expected {expected_placement!r}"
        )
    if state.get("policy_kind", "image_only") != expected_policy_kind:
        raise ValueError(
            f"Checkpoint {path} policy_kind={state.get('policy_kind')!r}; "
            f"expected {expected_policy_kind!r}"
        )
    saved = state["policy"]
    missing, unexpected = policy.load_state_dict(saved, strict=False)
    trainable_names = {name for name, parameter in policy.named_parameters()
                       if parameter.requires_grad}
    missing_trainable = sorted(trainable_names.intersection(missing))
    if missing_trainable or unexpected:
        raise ValueError(
            f"Policy state mismatch in {path}: missing trainable={missing_trainable}, "
            f"unexpected={unexpected}"
        )
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", required=True,
                        choices=("after_vision_end", "shared_dual_placement",
                                 "dynamic_cot_relay"))
    parser.add_argument("--policy-kind",
                        choices=("only_visual", "image_question_shared",
                                 "image_question_cot_shared"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--num-relay-tokens", type=int, default=2)
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    expected_placement = {
        "only_visual": "after_vision_end",
        "image_question_shared": "shared_dual_placement",
        "image_question_cot_shared": "dynamic_cot_relay",
    }[args.policy_kind]
    if args.placement != expected_placement:
        parser.error(f"{args.policy_kind} requires --placement {expected_placement}")
    if args.num_relay_tokens < 1:
        parser.error("num-relay-tokens must be positive")
    if args.policy_kind == "image_question_cot_shared" and args.num_relay_tokens != 1:
        parser.error("image_question_cot_shared requires --num-relay-tokens 1")

    steps = list(dict.fromkeys(args.checkpoint_steps))
    # Step zero is deliberately the native Qwen + input adapter baseline.  It
    # does not load checkpoint-0.pt and therefore inserts no random relay.
    checkpoints = {step: checkpoint_path(args.run_dir, step)
                   for step in steps if step != 0}
    records = load_evaluation_questions(args.dataset, args.mapverse_root, args.mapwise_root)
    output = args.run_dir / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    response_path = output / "responses.jsonl"
    completed: dict[tuple[int, int], dict] = {}
    if response_path.exists():
        for line in response_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[(int(row["checkpoint_step"]), int(row["dataset_index"]))] = row

    with response_path.open("a", encoding="utf-8") as response_file:
        # Adapter LoRA changes the ViT, merger and language decoder.  Evaluate
        # one complete checkpoint at a time and load its matching PEFT state;
        # visual features cannot be reused between checkpoints.
        for step in steps:
            pending = [index for index in range(1, len(records) + 1)
                       if (step, index) not in completed]
            if not pending:
                continue
            adapter_path = (args.adapter if step == 0
                            else checkpoint_adapter_path(checkpoints[step]))
            print(f"[EVAL LOAD] step={step} adapter={adapter_path}", flush=True)
            model, processor = load_model(
                args.model, adapter_path, load_in_4bit=args.load_in_4bit
            )
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            device = model.get_input_embeddings().weight.device
            policy = None
            if step != 0:
                policy = RelayModule(
                    decoder_hidden_size(model), num_relays=args.num_relay_tokens,
                ).to(device).eval()
                state = load_policy(
                    policy, checkpoints[step], args.placement, args.policy_kind
                )
                if int(state.get("step", -1)) != step:
                    raise ValueError(
                        f"Checkpoint metadata step mismatch: {checkpoints[step]}"
                    )

            for dataset_index, item in enumerate(records, start=1):
                if (step, dataset_index) in completed:
                    continue
                with Image.open(item.image_path) as image_file:
                    image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
                source_inputs = _content_inputs(
                    processor, [image], build_cartographic_prompt(item.question),
                    device, args.max_pixels
                )
                global_features = encode_global_image(model, source_inputs)
                if step == 0:
                    augmented, attention = build_adapter_only_prefill(
                        source_inputs, global_features
                    )
                    generator = generate_from_relays
                else:
                    builder = {
                        "only_visual": build_only_visual_prefill,
                        "image_question_shared": build_shared_prefill,
                        "image_question_cot_shared": build_cot_prefill,
                    }[args.policy_kind]
                    augmented, attention = builder(
                        model, processor.tokenizer, source_inputs, global_features,
                        policy, with_grad=False,
                    )
                    generator = (generate_with_cot_relay
                                 if args.policy_kind == "image_question_cot_shared"
                                 else generate_from_relays)
                builder = {
                    "initial_adapter_without_relay": step == 0,
                    "trained_checkpoint_with_relay": step != 0,
                }
                prefix_length = augmented["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = generator(
                        model,
                        **augmented, do_sample=False, num_beams=1,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                    )
                tokens = generated[0, prefix_length:]
                response = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                eos = model.generation_config.eos_token_id
                eos = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
                terminated = bool(tokens.numel() and int(tokens[-1]) in eos)
                evaluation = evaluate_answer(response, item)
                row = {
                    "checkpoint_step": step, "dataset_index": dataset_index,
                    "qa_id": item.sample_id,
                    "dataset": item.metadata.get("dataset", "mapverse"),
                    "country": item.metadata.get("country", ""),
                    "question": item.question, "ground_truth": item.answer,
                    "response": response, "reward": float(evaluation["reward"]),
                    "evaluation": evaluation, "generated_tokens": int(tokens.numel()),
                    "terminated": terminated, **builder,
                }
                if attention is not None:
                    weights = attention.detach().float()
                    row.update({
                        "prototype_attention_entropy": float(
                            -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
                        ),
                        "prototype_attention_max": float(weights.max(dim=-1).values.mean()),
                    })
                response_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                response_file.flush()
                completed[(step, dataset_index)] = row
                print(f"[EVAL step={step:05d}] {dataset_index}/{len(records)} "
                      f"qa_id={item.sample_id} reward={row['reward']}", flush=True)
                del generated, tokens
                del source_inputs, global_features, augmented
            del policy, model, processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected = len(records)
    summaries = []
    for step in steps:
        part = [completed[(step, index)] for index in range(1, expected + 1)
                if (step, index) in completed]
        if len(part) != expected:
            raise RuntimeError(f"Step {step} has {len(part)}/{expected} completed evaluations")
        correct = sum(row["reward"] for row in part)
        summaries.append({
            "checkpoint": "initial" if step == 0 else f"checkpoint-{step}",
            "step": step, "total": expected, "correct": int(correct),
            "accuracy": correct / expected,
            "truncated": sum(not row["terminated"] for row in part),
        })
    (output / "accuracy.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"evaluation": str(args.dataset), "summary": summaries},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

