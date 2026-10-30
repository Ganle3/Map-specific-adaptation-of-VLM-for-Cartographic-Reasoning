"""Scheme 1 sequence layout: one relay span after ``<vision_end>``."""
from __future__ import annotations

from utils import build_relay_prefill_at_indices, qwen_lm


def visual_relay_insertion_index(model, source_inputs: dict) -> int:
    ids = source_inputs["input_ids"][0]
    config = qwen_lm(model).config
    image_positions = ids.eq(int(config.image_token_id)).nonzero().flatten()
    if not len(image_positions):
        raise RuntimeError("Processor input contains no image placeholder tokens")
    final_image = int(image_positions[-1])
    later_end = ids[final_image + 1:].eq(int(config.vision_end_token_id)).nonzero().flatten()
    if not len(later_end):
        raise RuntimeError("Cannot locate <vision_end> after the final image placeholder")
    return final_image + 1 + int(later_end[0]) + 1


def build_prefill(model, tokenizer, source_inputs: dict, global_features,
                  relay_module, *, with_grad: bool):
    index = visual_relay_insertion_index(model, source_inputs)
    return build_relay_prefill_at_indices(
        model, tokenizer, source_inputs, global_features, relay_module,
        insertion_indices=[index], policy_kind="only_visual", with_grad=with_grad,
    )

