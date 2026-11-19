#!/usr/bin/env python3
"""One-question generation and GRPO replay check for all layer modes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

RELAY_ROOT = Path(__file__).resolve().parents[1]
VLM_ROOT = RELAY_ROOT.parents[1]
for path in (RELAY_ROOT, VLM_ROOT / "SparseVLM_eliminateCONTEXT"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model
from relay_module import RelayModule
from utils import decoder_hidden_size, encode_global_image
from Image_question_CoT_shared_relay.layout import build_prefill
from Image_question_CoT_shared_relay.dynamic_decode import (
    RELAY_LAYER_MODES, cot_relay_completion_log_probs, generate_with_cot_relay,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=262144)
    args = parser.parse_args()
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL smoke")
    model, processor = load_model(None, args.adapter, load_in_4bit=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    policy = RelayModule(decoder_hidden_size(model), num_relays=4).to(
        model.get_input_embeddings().weight.device
    )
    item = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root, limit=1
    )[0]
    with Image.open(item.image_path) as file:
        image = _resize_to_pixel_cap(file.convert("RGB"), args.max_pixels)
    source = _content_inputs(
        processor, [image], build_cartographic_prompt(item.question),
        model.get_input_embeddings().weight.device, args.max_pixels,
    )
    features = encode_global_image(model, source)
    for mode in RELAY_LAYER_MODES:
        augmented, _ = build_prefill(
            model, processor.tokenizer, source, features, policy, with_grad=True
        )
        augmented["relay_layer_mode"] = mode
        output = generate_with_cot_relay(
            model, **augmented, do_sample=False, max_new_tokens=2,
            return_dict_in_generate=True, output_scores=True,
        )
        completion = output.sequences[0, source["input_ids"].shape[1]:]
        logps = cot_relay_completion_log_probs(model, augmented, completion)
        sampled_logps = torch.stack([
            torch.log_softmax(scores[0].float(), dim=-1)[token]
            for scores, token in zip(output.scores, completion)
        ])
        maximum_logprob_difference = float(
            (logps.detach() - sampled_logps).abs().max()
        )
        differences = (logps.detach() - sampled_logps).float().tolist()
        if maximum_logprob_difference > 0.1:
            raise RuntimeError(
                f"Generation and current-policy replay disagree for {mode}: "
                f"differences={differences}"
            )
        policy.zero_grad(set_to_none=True)
        logps.sum().backward()
        row_norms = policy.prototypes.grad.float().norm(dim=-1).tolist()
        if not all(torch.isfinite(policy.prototypes.grad).flatten().tolist()):
            raise RuntimeError(f"Nonfinite relay gradients for {mode}")
        if mode == "reset_stage1" and not all(norm > 0 for norm in row_norms):
            raise RuntimeError(f"One or more stage parameters received no gradient: {row_norms}")
        print(json.dumps({
            "mode": mode, "relay_parameter_shape": list(policy.prototypes.shape),
            "generated_tokens": int(completion.numel()),
            "relay_gradient_norm_by_row": row_norms,
            "replay_logprob_has_grad": bool(logps.requires_grad),
            "max_generation_replay_logprob_difference": maximum_logprob_difference,
            "generation_replay_logprob_differences": differences,
        }), flush=True)
    print("PASS: generation and differentiable replay for all three relay layer modes")


if __name__ == "__main__":
    main()
