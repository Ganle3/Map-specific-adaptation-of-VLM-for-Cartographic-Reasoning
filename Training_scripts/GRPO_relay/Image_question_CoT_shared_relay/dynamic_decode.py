"""Dynamic per-step relay decoding and differentiable current-policy replay.

Each decision processes ``[x_t, r_t]`` jointly.  A custom block mask lets the
language query read the current relay and lets the relay query read only image
K/V, question K/V, its current language state, and itself.  During generation
both positions are temporarily appended to every DynamicCache layer; after the
decoder step ``cache.crop(-1)`` removes the relay K/V and retains ``x_t``.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from utils import (_decoder_and_head, _sampling_logits, _vl_backbone,
                   explicit_relay_decoder_inputs, qwen_lm)


def _set_custom_mask_attention(language_model) -> None:
    """Permanently select SDPA for this scheme's custom 4-D block mask.

    FlashAttention's compact causal-mask path cannot express the two-way
    current [x_t,r_t] block.  SDPA preserves the custom mask without
    materializing eager per-head attention weights, which matters on a 4090.
    The setting must remain active through backward because non-reentrant
    gradient checkpointing recomputes each decoder layer during backward.
    """
    for layer in language_model.layers:
        layer.self_attn.config._attn_implementation = "sdpa"


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if allowed.dtype != torch.bool or allowed.ndim != 2:
        raise ValueError("allowed attention mask must be a 2-D bool tensor")
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None, :, :]


def _step_attention_mask(past_length: int, relay_visible_past: torch.Tensor,
                         dtype: torch.dtype, device) -> torch.Tensor:
    if relay_visible_past.shape != (past_length,):
        raise RuntimeError(
            f"Relay-visible mask length {relay_visible_past.numel()} != cache {past_length}"
        )
    allowed = torch.zeros((2, past_length + 2), dtype=torch.bool, device=device)
    allowed[0, :] = True  # x_t reads all causal history, itself, and current r_t
    allowed[1, :past_length] = relay_visible_past.to(device=device, dtype=torch.bool)
    allowed[1, past_length:] = True  # r_t reads current x_t and itself
    return _additive_mask(allowed, dtype)


def _full_attention_mask(prefix_length: int, target_count: int,
                         relay_visible_prefix: torch.Tensor,
                         dtype: torch.dtype, device) -> torch.Tensor:
    """Mask native prompt prefix plus ``target_count`` [predictor, relay] pairs."""
    total = prefix_length + 2 * target_count
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    prefix_visible = relay_visible_prefix.to(device=device, dtype=torch.bool)
    if prefix_visible.shape != (prefix_length,):
        raise RuntimeError("Relay-visible prefix mask does not match the prompt prefix")
    for query in range(prefix_length):
        allowed[query, :query + 1] = True
    prior_language_positions: list[int] = []
    for index in range(target_count):
        language_position = prefix_length + 2 * index
        relay_position = language_position + 1
        allowed[language_position, :prefix_length] = True
        if prior_language_positions:
            allowed[language_position, prior_language_positions] = True
        allowed[language_position, language_position] = True
        allowed[language_position, relay_position] = True
        allowed[relay_position, :prefix_length] = prefix_visible
        allowed[relay_position, language_position] = True
        allowed[relay_position, relay_position] = True
        prior_language_positions.append(language_position)
    return _additive_mask(allowed, dtype)


def _decoder_layers(language_model, hidden_states: torch.Tensor,
                    position_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                    past_key_values=None, visual_pos_masks=None,
                    deepstack_visual_embeds=None, checkpoint_layers: bool = False):
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    for layer_index, layer in enumerate(language_model.layers):
        def run_layer(states, layer=layer):
            return layer(
                states, attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, use_cache=past_key_values is not None,
            )
        hidden_states = (checkpoint(run_layer, hidden_states, use_reentrant=False)
                         if checkpoint_layers else run_layer(hidden_states))
        if (deepstack_visual_embeds is not None
                and layer_index < len(deepstack_visual_embeds)):
            hidden_states = language_model._deepstack_process(
                hidden_states, visual_pos_masks,
                deepstack_visual_embeds[layer_index],
            )
    return language_model.norm(hidden_states)


def _pair_position_ids(position: torch.Tensor) -> torch.Tensor:
    """Give x_t and temporary r_t the same logical multimodal RoPE position."""
    if position.ndim != 3 or position.shape[1:] != (1, 1):
        raise ValueError(f"Expected position [D,1,1], got {tuple(position.shape)}")
    return position.expand(-1, -1, 2)


def _decode_pair(model, *, token: torch.Tensor, relay: torch.Tensor,
                 position: torch.Tensor, past, relay_visible_past: torch.Tensor):
    language_model, lm_head = _decoder_and_head(model)
    token_embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
    relay_embed = relay.to(token_embed.device, token_embed.dtype).view(1, 1, -1)
    pair = torch.cat((token_embed, relay_embed), dim=1)
    past_length = past.get_seq_length()
    mask = _step_attention_mask(
        past_length, relay_visible_past, pair.dtype, pair.device
    )
    _set_custom_mask_attention(language_model)
    hidden = _decoder_layers(
        language_model, pair, _pair_position_ids(position), mask,
        past_key_values=past,
    )
    if past.get_seq_length() != past_length + 2:
        raise RuntimeError("DynamicCache did not append both x_t and r_t positions")
    logits = lm_head(hidden[:, 0, :])
    past.crop(-1)
    if past.get_seq_length() != past_length + 1:
        raise RuntimeError("cache.crop(-1) did not retain x_t and remove r_t")
    return logits, past


@torch.no_grad()
def generate_with_cot_relay(
        model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        mm_encoder_outputs: dict, relay_embeddings: torch.Tensor,
        relay_visible_prompt_mask: torch.Tensor, policy_kind: str,
        do_sample: bool = False, temperature: float = 1.0,
        top_p: float = 1.0, top_k: int = 0, num_beams: int = 1,
        max_new_tokens: int, return_dict_in_generate: bool = False,
        output_scores: bool = False, use_cache: bool = True):
    if policy_kind != "image_question_cot_shared":
        raise ValueError(f"Unexpected policy_kind={policy_kind!r}")
    if input_ids.shape[0] != 1 or num_beams != 1 or relay_embeddings.shape[0] != 1:
        raise ValueError("Dynamic relay generation requires batch=1, beams=1, one relay")
    if input_ids.shape[1] < 2 or max_new_tokens < 1 or not use_cache:
        raise ValueError("Prompt must have >=2 tokens; max_new_tokens>0 and cache are required")

    visual = mm_encoder_outputs["image"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, input_ids, visual
    )
    full_positions, _ = _vl_backbone(model).get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    prefix_length = input_ids.shape[1] - 1
    language_model, _ = _decoder_and_head(model)
    prefill = language_model(
        input_ids=None, inputs_embeds=prompt_embeds[:, :prefix_length],
        attention_mask=attention_mask[:, :prefix_length],
        position_ids=full_positions[..., :prefix_length],
        visual_pos_masks=visual_mask[:, :prefix_length],
        deepstack_visual_embeds=deepstack, use_cache=True, return_dict=True,
    )
    past = prefill.past_key_values
    if past.get_seq_length() != prefix_length:
        raise RuntimeError("Scheme-3 prefill cache length mismatch")

    visible = relay_visible_prompt_mask[:prefix_length].to(input_ids.device)
    token = input_ids[0, -1]
    position = full_positions[..., -1:]
    cursor = int(full_positions.max().item()) + 1
    tokens: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}

    for _ in range(max_new_tokens):
        next_logits, past = _decode_pair(
            model, token=token, relay=relay_embeddings[0], position=position,
            past=past, relay_visible_past=visible,
        )
        step_scores = _sampling_logits(next_logits, temperature, top_p, top_k)
        sampled = (torch.distributions.Categorical(logits=step_scores).sample()
                   if do_sample else step_scores.argmax(dim=-1))
        tokens.append(sampled)
        if output_scores:
            scores.append(step_scores)
        if int(sampled.item()) in eos_ids:
            break
        token = sampled[0]
        visible = torch.cat((visible, visible.new_zeros(1)), dim=0)
        position = torch.full_like(position, cursor)
        cursor += 1

    completion = torch.stack(tokens, dim=1)
    sequences = torch.cat((input_ids, completion), dim=1)
    if return_dict_in_generate:
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))
    return sequences


def cot_relay_completion_log_probs(model, augmented_prompt: dict,
                                   completion_ids: torch.Tensor) -> torch.Tensor:
    """Replay a trajectory with [language predictor, relay] at every decision."""
    prompt_ids = augmented_prompt["input_ids"]
    targets = completion_ids.to(prompt_ids.device).view(1, -1)
    if targets.shape[1] < 1:
        return torch.empty(0, device=prompt_ids.device)
    relay = augmented_prompt["relay_embeddings"]
    if relay.shape[0] != 1 or not relay.requires_grad:
        raise RuntimeError("Current-policy scheme-3 replay requires one attached relay")

    visual = augmented_prompt["mm_encoder_outputs"]["image"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, prompt_ids, visual
    )
    positions, _ = _vl_backbone(model).get_rope_index(
        prompt_ids, augmented_prompt["mm_token_type_ids"],
        image_grid_thw=augmented_prompt["image_grid_thw"],
        attention_mask=augmented_prompt["attention_mask"],
    )
    prefix_length = prompt_ids.shape[1] - 1
    predictors = torch.cat((prompt_ids[:, -1:], targets[:, :-1]), dim=1)
    predictor_embeds = qwen_lm(model).get_input_embeddings()(predictors)
    relay_embed = relay.to(predictor_embeds.device, predictor_embeds.dtype).view(1, 1, -1)
    relay_embeds = relay_embed.expand(1, targets.shape[1], -1)
    pairs = torch.stack((predictor_embeds, relay_embeds), dim=2).reshape(
        1, 2 * targets.shape[1], -1
    )
    hidden = torch.cat((prompt_embeds[:, :prefix_length], pairs), dim=1)

    first_position = positions[..., -1:]
    extra_positions = [first_position]
    cursor = int(positions.max().item()) + 1
    for _ in range(1, targets.shape[1]):
        extra_positions.append(torch.full_like(first_position, cursor))
        cursor += 1
    paired_positions = torch.cat(
        [position.expand(-1, -1, 2) for position in extra_positions], dim=-1
    )
    all_positions = torch.cat((positions[..., :prefix_length], paired_positions), dim=-1)
    visible_prefix = augmented_prompt["relay_visible_prompt_mask"][:prefix_length]
    block_mask = _full_attention_mask(
        prefix_length, targets.shape[1], visible_prefix,
        hidden.dtype, hidden.device,
    )
    extended_visual_mask = torch.cat((
        visual_mask[:, :prefix_length],
        torch.zeros((1, 2 * targets.shape[1]), dtype=torch.bool, device=hidden.device),
    ), dim=1)
    language_model, lm_head = _decoder_and_head(model)
    _set_custom_mask_attention(language_model)
    with torch.enable_grad():
        decoded = _decoder_layers(
            language_model, hidden, all_positions, block_mask,
            visual_pos_masks=extended_visual_mask,
            deepstack_visual_embeds=deepstack, checkpoint_layers=True,
        )
        predictor_positions = prefix_length + 2 * torch.arange(
            targets.shape[1], device=hidden.device
        )
        logits = lm_head(decoded[:, predictor_positions, :])
        selected = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(0).squeeze(-1)
    if not selected.requires_grad:
        raise RuntimeError("Scheme-3 current-policy log probabilities are detached")
    return selected
