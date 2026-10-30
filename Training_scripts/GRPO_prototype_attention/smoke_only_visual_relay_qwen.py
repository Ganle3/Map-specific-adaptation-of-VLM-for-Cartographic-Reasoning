#!/usr/bin/env python3
"""One real Qwen3-VL relay prefill, generation, and gradient regression."""
from __future__ import annotations

import argparse
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
from data import build_cartographic_prompt, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model

from language_grpo import completion_log_probs
from only_visual_relay_policy import OnlyVisualRelayPolicy
from only_visual_relay_prefill import (build_only_visual_relay_prefill,
                                       generate_from_visual_relays)
from prototype_cross_attention_policy import decoder_hidden_size
from prototype_prefill import encode_global_image, qwen_lm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=245)
    parser.add_argument("--max-pixels", type=int, default=100_000)
    parser.add_argument("--num-relay-tokens", type=int, default=2)
    args = parser.parse_args()

    records = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root,
        limit=args.dataset_index,
    )
    item = records[args.dataset_index - 1]
    model, processor = load_model(None, args.adapter, load_in_4bit=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    with Image.open(item.image_path) as image_file:
        image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
    source = _content_inputs(
        processor, [image], build_cartographic_prompt(item.question), device,
        args.max_pixels,
    )
    features = encode_global_image(model, source)
    policy = OnlyVisualRelayPolicy(
        decoder_hidden_size(model), num_relays=args.num_relay_tokens,
    ).to(device)

    augmented, attention = build_only_visual_relay_prefill(
        model, processor.tokenizer, source, features, policy, with_grad=True,
    )
    target = torch.tensor(
        processor.tokenizer.encode(" So", add_special_tokens=False)[:1],
        dtype=torch.long,
    )
    logp = completion_log_probs(model, augmented, target)
    (-logp.mean()).backward()
    relay_grad = float(policy.prototypes.grad.norm())
    backbone_grads = sum(parameter.grad is not None for parameter in model.parameters())
    if attention is not None or relay_grad <= 0 or backbone_grads:
        raise RuntimeError(
            f"Bad relay gradient route: grad={relay_grad}, backbone_grads={backbone_grads}"
        )

    rollout_prompt, _ = build_only_visual_relay_prefill(
        model, processor.tokenizer, source, features, policy, with_grad=False,
    )
    with torch.no_grad():
        generated = generate_from_visual_relays(
            model, **rollout_prompt, do_sample=True, temperature=1.0,
            top_p=1.0, top_k=0, max_new_tokens=2,
            return_dict_in_generate=True, output_scores=True,
        )
    generated_count = generated.sequences.shape[1] - rollout_prompt["input_ids"].shape[1]
    image_id = int(qwen_lm(model).config.image_token_id)
    if generated_count != len(generated.scores) or generated_count < 1:
        raise RuntimeError("Generation score alignment failed")
    if int(rollout_prompt["input_ids"].eq(image_id).sum()) != int(
            source["input_ids"].eq(image_id).sum()):
        raise RuntimeError("Relay path changed the original image placeholder count")
    print({
        "status": "PASS",
        "policy_kind": "only_visual_relay",
        "placement": "after_vision_end",
        "prompt_tokens_before": int(source["input_ids"].shape[1]),
        "prompt_tokens_after": int(rollout_prompt["input_ids"].shape[1]),
        "relay_mm_types": rollout_prompt["mm_token_type_ids"][
            0, rollout_prompt["relay_positions"]
        ].tolist(),
        "trainable_parameter_count": policy.prototypes.numel(),
        "relay_gradient_norm": relay_grad,
        "backbone_parameter_grad_count": backbone_grads,
        "sampled_generation_tokens": generated_count,
    }, flush=True)


if __name__ == "__main__":
    main()
