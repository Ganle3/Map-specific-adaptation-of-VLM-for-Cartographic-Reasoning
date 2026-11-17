#!/usr/bin/env python3
"""Capture compact post-RoPE hidden-Q/image-K statistics on a test split."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
RELAY_ROOT = HERE.parent
VLM_ROOT = HERE.parents[2]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
VIS_ROOT = RELAY_ROOT / "relay3_visualization"
for path in (HERE, RELAY_ROOT, SPARSE_ROOT, VIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compact_capture import CompactQKCapture, METRIC_NAMES
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import MapQuestion, build_cartographic_prompt
from evaluate_core import load_evaluation_questions, load_policy
from relay_module import RelayModule
from rolling_sparse_qwen3vl import load_model
from utils import (
    _decoder_and_head,
    _vl_backbone,
    build_adapter_only_prefill,
    decoder_hidden_size,
    encode_global_image,
    explicit_relay_decoder_inputs,
    generate_from_relays,
    qwen_lm,
)
from Image_question_CoT_shared_relay.dynamic_decode import (
    _decoder_layers,
    _group_position_ids,
    _step_attention_mask,
)
from Image_question_CoT_shared_relay.layout import build_prefill as build_relay_prefill


DEFAULT_OUTPUT = VLM_ROOT / "Training_outputs/Prototype/relay3_global_qk_step4000"
DEFAULT_RUNS = {
    "adapter_only": VLM_ROOT / "Training_outputs/Prototype/adapter_only/checkpoint",
    "relay3": VLM_ROOT / "Training_outputs/Prototype/relay3/checkpoint",
}
DEFAULT_EVALUATIONS = {
    ("adapter_only", "mapverse"): VLM_ROOT / "Training_outputs/Prototype/adapter_only/evaluation_mapverse_test600_checkpoint_2000_4000/responses.jsonl",
    ("adapter_only", "mapwise"): VLM_ROOT / "Training_outputs/Prototype/adapter_only/evaluation_mapwise_test400_checkpoint_2000_4000/responses.jsonl",
    ("relay3", "mapverse"): VLM_ROOT / "Training_outputs/Prototype/relay3/evaluation_mapverse_test600_initial_without_relay_2000_4000_5000/responses.jsonl",
    ("relay3", "mapwise"): VLM_ROOT / "Training_outputs/Prototype/relay3/evaluation_mapwise_test400_initial_without_relay_2000_4000_5000/responses.jsonl",
}


@dataclass(frozen=True)
class Question:
    dataset: str
    index0: int
    item: MapQuestion
    raw: dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=("adapter_only", "relay3"), required=True)
    p.add_argument("--dataset-name", choices=("mapverse", "mapwise"), required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    p.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--checkpoint-step", type=int, default=4000)
    p.add_argument("--model", default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--start-index", type=int, default=0, help="Inclusive zero-based dataset row")
    p.add_argument("--end-index", type=int, default=None, help="Exclusive zero-based dataset row")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--max-pixels", type=int, default=1_000_000)
    p.add_argument("--num-relay-tokens", type=int, default=1)
    p.add_argument("--projection-dim", type=int, default=64)
    p.add_argument("--key-samples", type=int, default=32)
    p.add_argument("--projection-seed", type=int, default=20261117)
    p.add_argument("--projection-steps", type=int, nargs="+", default=(0, 4, 9, 19, 39, 79, 159, 319))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    p.set_defaults(load_in_4bit=True)
    a = p.parse_args()
    if a.run_dir is None:
        a.run_dir = DEFAULT_RUNS[a.variant]
    if a.evaluation is None:
        a.evaluation = DEFAULT_EVALUATIONS[(a.variant, a.dataset_name)]
    if a.checkpoint_step != 4000:
        p.error("This controlled comparison is fixed to checkpoint 4000")
    if min(a.max_new_tokens, a.max_pixels, a.projection_dim, a.key_samples) < 1:
        p.error("Capture sizes must be positive")
    return a


def read_official(path: Path, step: int) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["checkpoint_step"]) != step:
            continue
        index = int(row["dataset_index"])
        if index in rows:
            raise ValueError(f"Duplicate evaluation dataset_index={index}")
        rows[index] = row
    return rows


def load_questions(args) -> tuple[list[Question], dict[int, dict]]:
    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    parsed = load_evaluation_questions(args.dataset, args.mapverse_root, args.mapwise_root)
    if len(raw) != len(parsed):
        raise RuntimeError("Raw and parsed dataset lengths differ")
    end = len(raw) if args.end_index is None else min(args.end_index, len(raw))
    if not 0 <= args.start_index < end:
        raise ValueError(f"Invalid row range [{args.start_index},{end})")
    official = read_official(args.evaluation, args.checkpoint_step)
    expected = set(range(1, len(raw) + 1))
    if set(official) != expected:
        raise ValueError("Official evaluation is incomplete or uses unexpected indexing")
    return [Question(args.dataset_name, i, parsed[i], raw[i])
            for i in range(args.start_index, end)], official


def grid_and_positions(model, source):
    qwen = qwen_lm(model)
    positions = source["input_ids"][0].eq(int(qwen.config.image_token_id)).nonzero().flatten()
    t, h, w = map(int, source["image_grid_thw"][0].tolist())
    merge = int(qwen.config.vision_config.spatial_merge_size)
    grid = (h // merge, w // merge)
    if t != 1 or positions.numel() != grid[0] * grid[1]:
        raise RuntimeError("Expected one complete merged still-image grid")
    return positions, grid


def prepare(model, processor, question: Question, max_pixels: int):
    with Image.open(question.item.image_path) as handle:
        image = _resize_to_pixel_cap(handle.convert("RGB"), max_pixels)
    source = _content_inputs(
        processor, [image], build_cartographic_prompt(question.item.question),
        model.get_input_embeddings().weight.device, max_pixels,
    )
    features = encode_global_image(model, source)
    positions, grid = grid_and_positions(model, source)
    return source, features, positions, grid


def _decode_group(model, token, relay, position, past, visible):
    language_model, lm_head = _decoder_and_head(model)
    token_embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
    relay_embed = relay.to(token_embed.device, token_embed.dtype).unsqueeze(0)
    group = torch.cat((token_embed, relay_embed), dim=1)
    relay_count = relay.shape[0]
    old_length = past.get_seq_length()
    mask = _step_attention_mask(old_length, visible, group.dtype, group.device, relay_count)
    hidden = _decoder_layers(
        language_model, group, _group_position_ids(position, relay_count), mask,
        past_key_values=past,
    )
    logits = lm_head(hidden[:, 0, :])
    past.crop(-relay_count)
    if past.get_seq_length() != old_length + 1:
        raise RuntimeError("Relay cache crop did not retain exactly x_t")
    return logits, past


@torch.no_grad()
def generate_relay(model, augmented, capture: CompactQKCapture, max_new_tokens: int):
    input_ids = augmented["input_ids"]
    attention_mask = augmented["attention_mask"]
    relay = augmented["relay_embeddings"]
    visual = augmented["mm_encoder_outputs"]["image"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(model, input_ids, visual)
    positions, _ = _vl_backbone(model).get_rope_index(
        input_ids, augmented["mm_token_type_ids"],
        image_grid_thw=augmented["image_grid_thw"], attention_mask=attention_mask,
    )
    prefix_length = input_ids.shape[1] - 1
    language_model, _ = _decoder_and_head(model)
    prefill = language_model(
        input_ids=None, inputs_embeds=prompt_embeds[:, :prefix_length],
        attention_mask=attention_mask[:, :prefix_length],
        position_ids=positions[..., :prefix_length],
        visual_pos_masks=visual_mask[:, :prefix_length],
        deepstack_visual_embeds=deepstack, use_cache=True, return_dict=True,
    )
    past = prefill.past_key_values
    visible = augmented["relay_visible_prompt_mask"][:prefix_length].to(input_ids.device)
    token = input_ids[0, -1]
    position = positions[..., -1:]
    cursor = int(positions.max().item()) + 1
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    generated = []
    with capture:
        for _ in range(max_new_tokens):
            logits, past = _decode_group(model, token, relay, position, past, visible)
            sampled = logits.float().argmax(dim=-1)
            generated.append(sampled)
            if int(sampled.item()) in eos_ids:
                break
            token = sampled[0]
            visible = torch.cat((visible, visible.new_zeros(1)), dim=0)
            position = torch.full_like(position, cursor)
            cursor += 1
    completion = torch.stack(generated, dim=1)
    return torch.cat((input_ids, completion), dim=1)


def completion_ids(model, source, generated):
    ids = generated[0, source["input_ids"].shape[1]:]
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    terminated = bool(ids.numel() and int(ids[-1]) in eos_ids)
    return ids, terminated


def save_question(path: Path, arrays: dict, metadata: dict):
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "qk_stats.npz", **arrays)
    (path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    questions, official = load_questions(args)
    checkpoint = args.run_dir / f"checkpoint-{args.checkpoint_step}.pt"
    adapter = args.run_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint below {args.run_dir}")
    destination = args.output / args.dataset_name / args.variant
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variant": args.variant, "dataset": args.dataset_name,
        "checkpoint_step": args.checkpoint_step, "metric_names": METRIC_NAMES,
        "row_range_0based_half_open": [args.start_index,
                                        len(questions) + args.start_index],
        "projection_dim": args.projection_dim, "key_samples": args.key_samples,
        "projection_seed": args.projection_seed,
        "projection_steps_0based": args.projection_steps,
        "distance_definition": "1 - mean-head cosine using actual post-RoPE hidden Q and cached image K",
    }
    (destination / f"manifest_{args.start_index:04d}_{args.start_index + len(questions):04d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    policy = None
    if args.variant == "relay3":
        policy = RelayModule(decoder_hidden_size(model), num_relays=args.num_relay_tokens).to(
            model.get_input_embeddings().weight.device
        ).eval()
        state = load_policy(policy, checkpoint, "dynamic_cot_relay", "image_question_cot_shared")
        if int(state.get("num_relay_tokens", -1)) != args.num_relay_tokens:
            raise ValueError("Relay count differs from checkpoint")
    try:
        for n, question in enumerate(questions, 1):
            folder = destination / f"index{question.index0 + 1:04d}"
            if (folder / "qk_stats.npz").is_file() and not args.overwrite:
                print(f"[{n}/{len(questions)}] skip {folder.name}", flush=True)
                continue
            print(f"[{n}/{len(questions)}] {args.variant} {question.dataset} index={question.index0 + 1}", flush=True)
            source, features, image_positions, grid = prepare(
                model, processor, question, args.max_pixels
            )
            capture = CompactQKCapture(
                model, image_positions, query_kind=("relay" if policy is not None else "adapter"),
                projection_dim=args.projection_dim, key_samples=args.key_samples,
                projection_seed=args.projection_seed,
                projection_steps=tuple(args.projection_steps),
            )
            if policy is None:
                augmented, _ = build_adapter_only_prefill(source, features)
                with capture:
                    generated = generate_from_relays(
                        model, **augmented, do_sample=False, num_beams=1,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                    )
            else:
                augmented, _ = build_relay_prefill(
                    model, processor.tokenizer, source, features, policy, with_grad=False
                )
                generated = generate_relay(model, augmented, capture, args.max_new_tokens)
            ids, terminated = completion_ids(model, source, generated)
            arrays = capture.arrays(int(ids.numel()), allow_one_unused_tail=(policy is None))
            off = official[question.index0 + 1]
            if (str(off["qa_id"]) != str(question.item.sample_id)
                    or off["question"] != question.item.question):
                raise ValueError("Official evaluation row is misaligned")
            metadata = {
                "variant": args.variant, "dataset": question.dataset,
                "dataset_index": question.index0 + 1, "array_index_0based": question.index0,
                "qa_id": str(question.item.sample_id), "question": question.item.question,
                "ground_truth": question.item.answer,
                "ability_level": question.raw.get("ability_level"),
                "answer_type": question.raw.get("answer_type", question.raw.get("ground_truth_type")),
                "image": str(question.item.image_path), "grid_hw": list(grid),
                "generated_tokens": int(ids.numel()), "terminated": terminated,
                "response": processor.tokenizer.decode(ids, skip_special_tokens=True),
                "official_reward": float(off["reward"]),
                "official_generated_tokens": int(off["generated_tokens"]),
                "official_terminated": bool(off["terminated"]),
                "matches_official_response": processor.tokenizer.decode(
                    ids, skip_special_tokens=True
                ).strip() == str(off["response"]).strip(),
            }
            save_question(folder, arrays, metadata)
            del source, features, augmented, generated, ids, arrays, capture
    finally:
        del policy, model, processor
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

