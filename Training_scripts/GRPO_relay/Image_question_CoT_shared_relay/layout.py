"""Scheme 3 prompt metadata; this scheme inserts no persistent prefill token."""
from __future__ import annotations

import torch

from utils import qwen_lm


def relay_visible_prompt_positions(model, tokenizer, source_inputs: dict) -> torch.Tensor:
    """Mark cached image and user-question positions visible to a step relay."""
    ids = source_inputs["input_ids"][0]
    types = source_inputs["mm_token_type_ids"][0]
    config = qwen_lm(model).config

    image_positions = ids.eq(int(config.image_token_id)).nonzero().flatten()
    if not len(image_positions):
        raise RuntimeError("Processor input contains no image placeholder tokens")
    final_image = int(image_positions[-1])
    later_end = ids[final_image + 1:].eq(int(config.vision_end_token_id)).nonzero().flatten()
    if not len(later_end):
        raise RuntimeError("Cannot locate <vision_end> after the final image placeholder")
    vision_end = final_image + 1 + int(later_end[0])

    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end) != 1:
        raise RuntimeError("Unexpected Qwen <|im_end|> tokenization")
    user_ends = ids.eq(int(im_end[0])).nonzero().flatten()
    if not len(user_ends):
        raise RuntimeError("Cannot locate the user-turn end before the assistant prefix")
    user_end = int(user_ends[-1])
    if user_end <= vision_end:
        raise RuntimeError("User question span must follow <vision_end>")

    visible = types.eq(1)  # image placeholders, whose K/V contain merged visual features
    visible[vision_end + 1:user_end + 1] = True  # question and user-turn terminator
    return visible


def build_prefill(model, tokenizer, source_inputs: dict, global_features,
                  relay_module, *, with_grad: bool):
    """Keep the native prompt and attach one shared relay for dynamic steps."""
    relays = (relay_module.prototypes if with_grad
              else relay_module.prototypes.detach())
    if relays.shape[0] != 1:
        raise ValueError("Scheme 3 currently requires exactly one relay token")
    return {
        "input_ids": source_inputs["input_ids"],
        "attention_mask": source_inputs["attention_mask"],
        "mm_token_type_ids": source_inputs["mm_token_type_ids"],
        "image_grid_thw": source_inputs["image_grid_thw"],
        "mm_encoder_outputs": {"image": global_features},
        "relay_embeddings": relays,
        "relay_visible_prompt_mask": relay_visible_prompt_positions(
            model, tokenizer, source_inputs
        ),
        "policy_kind": "image_question_cot_shared",
    }, None

