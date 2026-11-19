"""Relay3 generation variants that cut one information path at a time.

All variants use the same trained Relay3 LoRA adapter and relay parameter bank.
No parameter is changed or retrained here.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from utils import (
    _decoder_and_head,
    _sampling_logits,
    _vl_backbone,
    explicit_relay_decoder_inputs,
    qwen_lm,
)
from Image_question_CoT_shared_relay.dynamic_decode import (
    _decoder_layers,
    _group_position_ids,
    _set_custom_mask_attention,
)


DYNAMIC_MODES = {"full", "no_image_read", "no_current_x_read"}
ALL_MODES = DYNAMIC_MODES | {"static_relay"}


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if allowed.dtype != torch.bool or allowed.ndim != 2:
        raise ValueError("allowed must be a 2-D bool tensor")
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


def step_attention_mask(
    past_length: int,
    relay_visible_past: torch.Tensor,
    dtype: torch.dtype,
    device,
    num_relays: int,
    *,
    relay_reads_current_x: bool,
) -> torch.Tensor:
    """Build the current ``[x_t, r_t,1, ...]`` query/key mask.

    The language row always reads causal history and the complete current
    group. Relay rows read only the selected prompt history and the current
    relay bank. Their access to current ``x_t`` is the controlled ablation.
    """
    if relay_visible_past.shape != (past_length,):
        raise RuntimeError(
            f"Relay-visible mask length {relay_visible_past.numel()} != cache {past_length}"
        )
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    group_size = 1 + num_relays
    allowed = torch.zeros(
        (group_size, past_length + group_size), dtype=torch.bool, device=device
    )
    allowed[0, :] = True
    allowed[1:, :past_length] = relay_visible_past.to(
        device=device, dtype=torch.bool
    ).unsqueeze(0)
    # Relay-to-relay access, including each relay's own position.
    allowed[1:, past_length + 1:] = True
    if relay_reads_current_x:
        allowed[1:, past_length] = True
    return _additive_mask(allowed, dtype)


def _decode_group(
    model,
    *,
    token: torch.Tensor,
    relay: torch.Tensor,
    position: torch.Tensor,
    past,
    relay_visible_past: torch.Tensor,
    relay_reads_current_x: bool,
    keep_relay_cache: bool,
):
    language_model, lm_head = _decoder_and_head(model)
    token_embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
    relay_embed = relay.to(token_embed.device, token_embed.dtype).unsqueeze(0)
    group = torch.cat((token_embed, relay_embed), dim=1)
    num_relays = int(relay.shape[0])
    past_length = past.get_seq_length()
    mask = step_attention_mask(
        past_length,
        relay_visible_past,
        group.dtype,
        group.device,
        num_relays,
        relay_reads_current_x=relay_reads_current_x,
    )
    _set_custom_mask_attention(language_model)
    hidden = _decoder_layers(
        language_model,
        group,
        _group_position_ids(position, num_relays),
        mask,
        past_key_values=past,
    )
    expected = past_length + 1 + num_relays
    if past.get_seq_length() != expected:
        raise RuntimeError(f"DynamicCache length {past.get_seq_length()} != {expected}")
    logits = lm_head(hidden[:, 0, :])
    if not keep_relay_cache:
        past.crop(-num_relays)
        if past.get_seq_length() != past_length + 1:
            raise RuntimeError("Relay cache crop did not retain exactly x_t")
    return logits, past


def _sample(logits, *, do_sample, temperature, top_p, top_k):
    scores = _sampling_logits(logits, temperature, top_p, top_k)
    token = (
        torch.distributions.Categorical(logits=scores).sample()
        if do_sample
        else scores.argmax(dim=-1)
    )
    return token, scores


@torch.no_grad()
def generate_relay3_ablation(
    model,
    *,
    ablation_mode: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor,
    mm_encoder_outputs: dict,
    relay_embeddings: torch.Tensor,
    relay_visible_prompt_mask: torch.Tensor,
    policy_kind: str,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    num_beams: int = 1,
    max_new_tokens: int = 500,
    return_dict_in_generate: bool = False,
    output_scores: bool = False,
    use_cache: bool = True,
):
    if ablation_mode not in ALL_MODES:
        raise ValueError(f"Unknown Relay3 ablation: {ablation_mode!r}")
    if policy_kind != "image_question_cot_shared":
        raise ValueError(f"Unexpected policy_kind={policy_kind!r}")
    if input_ids.shape[0] != 1 or num_beams != 1 or not use_cache:
        raise ValueError("Relay3 ablation requires batch=1, beams=1 and use_cache=True")
    if relay_embeddings.ndim != 2 or relay_embeddings.shape[0] < 1:
        raise ValueError("relay_embeddings must have shape [R,H], R >= 1")

    visual = mm_encoder_outputs["image"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, input_ids, visual
    )
    positions, _ = _vl_backbone(model).get_rope_index(
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    prefix_length = input_ids.shape[1] - 1
    language_model, lm_head = _decoder_and_head(model)
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
        raise RuntimeError("Relay3 prefill cache length mismatch")

    visible = relay_visible_prompt_mask[:prefix_length].to(input_ids.device).clone()
    if ablation_mode == "no_image_read":
        image_positions = mm_token_type_ids[0, :prefix_length].eq(1)
        visible[image_positions] = False

    token = input_ids[0, -1]
    position = positions[..., -1:]
    cursor = int(positions.max().item()) + 1
    generated: list[torch.Tensor] = []
    score_rows: list[torch.Tensor] = []
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}

    # Static Relay computes the full relay once at the first reasoning step,
    # retains those layer-wise relay K/V entries, and never creates another.
    static = ablation_mode == "static_relay"
    logits, past = _decode_group(
        model,
        token=token,
        relay=relay_embeddings,
        position=position,
        past=past,
        relay_visible_past=visible,
        relay_reads_current_x=(static or ablation_mode != "no_current_x_read"),
        keep_relay_cache=static,
    )

    for step in range(max_new_tokens):
        sampled, scores = _sample(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated.append(sampled)
        if output_scores:
            score_rows.append(scores)
        if int(sampled.item()) in eos_ids:
            break
        token = sampled[0]
        position = torch.full_like(position, cursor)
        cursor += 1

        if static:
            full_mask = torch.ones(
                (1, past.get_seq_length() + 1),
                dtype=attention_mask.dtype,
                device=input_ids.device,
            )
            out = language_model(
                input_ids=token.view(1, 1),
                attention_mask=full_mask,
                position_ids=position,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = out.past_key_values
            logits = lm_head(out.last_hidden_state[:, -1, :])
        else:
            visible = torch.cat((visible, visible.new_zeros(1)), dim=0)
            logits, past = _decode_group(
                model,
                token=token,
                relay=relay_embeddings,
                position=position,
                past=past,
                relay_visible_past=visible,
                relay_reads_current_x=ablation_mode != "no_current_x_read",
                keep_relay_cache=False,
            )

    completion = torch.stack(generated, dim=1)
    sequences = torch.cat((input_ids, completion), dim=1)
    if return_dict_in_generate:
        return SimpleNamespace(sequences=sequences, scores=tuple(score_rows))
    return sequences
