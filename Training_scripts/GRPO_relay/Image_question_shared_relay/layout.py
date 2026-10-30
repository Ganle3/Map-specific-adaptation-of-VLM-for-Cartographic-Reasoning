"""Scheme 2 sequence layout: one shared relay span in two placements."""
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


def question_relay_insertion_index(tokenizer, source_inputs: dict) -> int:
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end) != 1:
        raise RuntimeError("Unexpected Qwen <|im_end|> tokenization")
    positions = source_inputs["input_ids"][0].eq(int(im_end[0])).nonzero().flatten()
    if not len(positions):
        raise RuntimeError("Cannot locate the user-turn end before the assistant prefix")
    return int(positions[-1])


def build_prefill(model, tokenizer, source_inputs: dict, global_features,
                  relay_module, *, with_grad: bool):
    visual_index = visual_relay_insertion_index(model, source_inputs)
    question_index = question_relay_insertion_index(tokenizer, source_inputs)
    if question_index <= visual_index:
        raise RuntimeError("Question relay position must follow the visual relay position")
    return build_relay_prefill_at_indices(
        model, tokenizer, source_inputs, global_features, relay_module,
        insertion_indices=[visual_index, question_index],
        policy_kind="image_question_shared", with_grad=with_grad,
    )

