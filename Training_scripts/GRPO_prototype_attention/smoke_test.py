#!/usr/bin/env python3
"""One-question, no-update proof of both prefill placements and gradient routing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[1]
THESIS_ROOT = VLM_ROOT.parent
for path in (VLM_ROOT / "SparseVLM_eliminateCONTEXT",
             VLM_ROOT / "Training_scripts" / "GRPO_lookbetter"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model
from language_grpo import completion_log_probs
from prototype_cross_attention_policy import PrototypeCrossAttentionPolicy, decoder_hidden_size
from prototype_prefill import build_prototype_prefill, encode_global_image, encode_question_context
from question_conditioned_policy import QuestionConditionedPrototypePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=247)
    parser.add_argument("--max-pixels", type=int, default=100_000)
    args = parser.parse_args()

    records = load_mapverse_questions(args.dataset, args.mapverse_root,
                                      mapwise_root=args.mapwise_root, limit=args.dataset_index)
    item = records[args.dataset_index - 1]
    model, processor = load_model(None, args.adapter, load_in_4bit=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    with Image.open(item.image_path) as image_file:
        image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
    source = _content_inputs(processor, [image], build_cartographic_prompt(item.question),
                             device, args.max_pixels)
    global_features = encode_global_image(model, source)
    target = torch.tensor(processor.tokenizer.encode(" So", add_special_tokens=False)[:1], dtype=torch.long)
    if not target.numel():
        raise RuntimeError("Smoke-test target tokenization is empty")

    for placement in ("after_image", "after_question"):
        policy = PrototypeCrossAttentionPolicy(
            decoder_hidden_size(model), num_prototypes=9, num_heads=8
        ).to(device)
        augmented, weights = build_prototype_prefill(
            model, processor.tokenizer, source, global_features, policy, placement, with_grad=True
        )
        logp = completion_log_probs(model, augmented, target)
        (-logp.mean()).backward()
        prototype_grad = float(policy.prototypes.grad.norm())
        backbone_grads = sum(parameter.grad is not None for parameter in model.parameters())
        if not prototype_grad > 0 or backbone_grads:
            raise RuntimeError(
                f"Bad gradient route for {placement}: prototype_grad={prototype_grad}, "
                f"backbone_parameter_grads={backbone_grads}"
            )
        rollout_prompt, _ = build_prototype_prefill(
            model, processor.tokenizer, source, global_features, policy, placement, with_grad=False
        )
        with torch.no_grad():
            generated = model.generate(
                **rollout_prompt, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                max_new_tokens=2, return_dict_in_generate=True, output_scores=True,
            )
        generated_count = generated.sequences.shape[1] - rollout_prompt["input_ids"].shape[1]
        if generated_count != len(generated.scores) or generated_count < 1:
            raise RuntimeError(f"Generation score alignment failed for {placement}")
        print({"placement": placement,
               "prompt_tokens": int(augmented["input_ids"].shape[1]),
               "attention_shape": tuple(weights.shape),
               "prototype_gradient_norm": prototype_grad,
               "backbone_parameter_grad_count": backbone_grads,
               "sampled_generation_tokens": int(generated_count)}, flush=True)

    question_tokens = encode_question_context(
        model, processor.tokenizer, source, global_features, item.question
    )
    policy = QuestionConditionedPrototypePolicy(
        decoder_hidden_size(model), num_prototypes=9, num_heads=8
    ).to(device)
    augmented, image_weights = build_prototype_prefill(
        model, processor.tokenizer, source, global_features, policy, "after_question",
        with_grad=True, question_tokens=question_tokens,
    )
    logp = completion_log_probs(model, augmented, target)
    (-logp.mean()).backward()
    prototype_grad = float(policy.prototypes.grad.norm())
    backbone_grads = sum(parameter.grad is not None for parameter in model.parameters())
    if not prototype_grad > 0 or backbone_grads:
        raise RuntimeError("Question-conditioned policy gradient routing failed")
    print({"policy_kind": "question_conditioned", "placement": "after_question",
           "question_tokens": int(question_tokens.shape[0]),
           "question_attention_shape": tuple(policy.last_question_attention.shape),
           "image_attention_shape": tuple(image_weights.shape),
           "prototype_gradient_norm": prototype_grad,
           "backbone_parameter_grad_count": backbone_grads}, flush=True)


if __name__ == "__main__":
    main()
