"""Greedy relay3 generation with eager attention capture on dynamic steps."""
from __future__ import annotations

import torch

from capture import ImageAttentionCapture, set_decoder_attention_implementation
from Image_question_CoT_shared_relay.dynamic_decode import (
    _decoder_layers,
    _group_position_ids,
    _step_attention_mask,
)
from utils import (
    _decoder_and_head,
    _vl_backbone,
    explicit_relay_decoder_inputs,
    qwen_lm,
)


def _decode_group(model, token, relay, position, past, visible):
    language_model, lm_head = _decoder_and_head(model)
    token_embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
    relay_embed = relay.to(token_embed.device, token_embed.dtype).unsqueeze(0)
    group = torch.cat((token_embed, relay_embed), dim=1)
    relay_count = relay.shape[0]
    past_length = past.get_seq_length()
    mask = _step_attention_mask(
        past_length, visible, group.dtype, group.device, relay_count
    )
    hidden = _decoder_layers(
        language_model,
        group,
        _group_position_ids(position, relay_count),
        mask,
        past_key_values=past,
    )
    if past.get_seq_length() != past_length + 1 + relay_count:
        raise RuntimeError("Dynamic cache did not append x_t and relay positions")
    logits = lm_head(hidden[:, 0, :])
    past.crop(-relay_count)
    if past.get_seq_length() != past_length + 1:
        raise RuntimeError("Relay cache crop did not retain exactly x_t")
    return logits, past


@torch.no_grad()
def generate_relay_with_capture(model, augmented: dict, image_positions,
                                grid: tuple[int, int], max_new_tokens: int):
    input_ids = augmented["input_ids"]
    attention_mask = augmented["attention_mask"]
    relay = augmented["relay_embeddings"]
    visual = augmented["mm_encoder_outputs"]["image"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, input_ids, visual
    )
    positions, _ = _vl_backbone(model).get_rope_index(
        input_ids,
        augmented["mm_token_type_ids"],
        image_grid_thw=augmented["image_grid_thw"],
        attention_mask=attention_mask,
    )
    prefix_length = input_ids.shape[1] - 1
    language_model, _ = _decoder_and_head(model)
    # Prompt prefill can be quadratic and is not one of the requested dynamic
    # x_t/r_t frames. Keep it on SDPA, then switch dynamic q=1+R steps to eager.
    set_decoder_attention_implementation(model, "sdpa")
    prefill = language_model(
        input_ids=None,
        inputs_embeds=prompt_embeds[:, :prefix_length],
        attention_mask=attention_mask[:, :prefix_length],
        position_ids=positions[..., :prefix_length],
        visual_pos_masks=visual_mask[:, :prefix_length],
        deepstack_visual_embeds=deepstack,
        use_cache=True,
        return_dict=True,
    )
    past = prefill.past_key_values
    if past.get_seq_length() != prefix_length:
        raise RuntimeError("Relay3 prompt cache length mismatch")

    visible = augmented["relay_visible_prompt_mask"][:prefix_length].to(input_ids.device)
    token = input_ids[0, -1]
    position = positions[..., -1:]
    cursor = int(positions.max().item()) + 1
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    generated: list[torch.Tensor] = []
    with ImageAttentionCapture(
        model, image_positions, grid, query_kind="relay"
    ) as capture:
        for _ in range(max_new_tokens):
            logits, past = _decode_group(
                model, token, relay, position, past, visible
            )
            sampled = logits.float().argmax(dim=-1)
            generated.append(sampled)
            if int(sampled.item()) in eos_ids:
                break
            token = sampled[0]
            visible = torch.cat((visible, visible.new_zeros(1)), dim=0)
            position = torch.full_like(position, cursor)
            cursor += 1
    completion = torch.stack(generated, dim=1)
    hidden, relay_maps = capture.arrays(completion.shape[1])
    assert relay_maps is not None
    return torch.cat((input_ids, completion), dim=1), hidden, relay_maps

