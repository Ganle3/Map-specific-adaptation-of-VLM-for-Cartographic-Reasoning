#!/usr/bin/env python3
"""One-token real-Qwen gradient audit shared by both relay layouts."""
from __future__ import annotations

import argparse
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
from data import build_cartographic_prompt, load_mapverse_questions
from rolling_sparse_qwen3vl import load_model
from utils import (build_relay_prefill, completion_log_probs,
                   decoder_hidden_size, encode_global_image_with_grad)
from only_visual_relay.policy import OnlyVisualRelayPolicy
from Image_question_shared_relay.policy import ImageQuestionSharedRelayPolicy


def main(*, scheme: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mapverse-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, default=245)
    parser.add_argument("--max-pixels", type=int, default=100_000)
    parser.add_argument("--num-relay-tokens", type=int, default=2)
    args = parser.parse_args()

    item = load_mapverse_questions(
        args.dataset, args.mapverse_root, mapwise_root=args.mapwise_root,
        limit=args.dataset_index,
    )[args.dataset_index - 1]
    model, processor = load_model(None, args.adapter, load_in_4bit=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = []
    for name, parameter in model.named_parameters():
        if any(marker in name for marker in (".lora_A.", ".lora_B.",
                                              ".lora_embedding_A.",
                                              ".lora_embedding_B.")):
            parameter.requires_grad_(True)
            adapter.append((name, parameter))
    if not adapter:
        raise RuntimeError("No LoRA tensors found in loaded adapter")

    device = model.get_input_embeddings().weight.device
    with Image.open(item.image_path) as image_file:
        image = _resize_to_pixel_cap(image_file.convert("RGB"), args.max_pixels)
    source = _content_inputs(
        processor, [image], build_cartographic_prompt(item.question), device,
        args.max_pixels,
    )
    policy_type = (OnlyVisualRelayPolicy if scheme == "only_visual"
                   else ImageQuestionSharedRelayPolicy)
    policy = policy_type(
        decoder_hidden_size(model), num_relays=args.num_relay_tokens,
    ).to(device)
    features = encode_global_image_with_grad(model, source)
    augmented, _ = build_relay_prefill(
        model, processor.tokenizer, source, features, policy,
        scheme=scheme, with_grad=True,
    )
    target = torch.tensor(
        processor.tokenizer.encode(" So", add_special_tokens=False)[:1],
        dtype=torch.long,
    )
    (-completion_log_probs(model, augmented, target).mean()).backward()

    adapter_names = {name for name, _ in adapter}
    def nonzero(part: str) -> int:
        return sum(part in name and parameter.grad is not None
                   and bool(parameter.grad.count_nonzero())
                   for name, parameter in adapter)

    positions = augmented["relay_positions"]
    result = {
        "status": "PASS", "scheme": scheme,
        "relay_position_groups": positions.detach().cpu().tolist(),
        "shared_parameter_object_count": 1,
        "relay_gradient_norm": float(policy.prototypes.grad.norm())
        if policy.prototypes.grad is not None else 0.0,
        "visual_nonzero_gradient_tensors": nonzero(".visual."),
        "merger_nonzero_gradient_tensors": sum(
            (".visual.merger." in name or ".visual.deepstack_merger_list." in name)
            and parameter.grad is not None and bool(parameter.grad.count_nonzero())
            for name, parameter in adapter
        ),
        "language_nonzero_gradient_tensors": nonzero(".language_model."),
        "unexpected_base_gradient_tensors": sum(
            parameter.grad is not None for name, parameter in model.named_parameters()
            if name not in adapter_names
        ),
    }
    expected_groups = 1 if scheme == "only_visual" else 2
    if (positions.shape != (expected_groups, args.num_relay_tokens)
            or result["relay_gradient_norm"] <= 0
            or result["visual_nonzero_gradient_tensors"] == 0
            or result["merger_nonzero_gradient_tensors"] == 0
            or result["language_nonzero_gradient_tensors"] == 0
            or result["unexpected_base_gradient_tensors"] != 0):
        result["status"] = "FAIL"
        raise RuntimeError(result)
    print(result, flush=True)

