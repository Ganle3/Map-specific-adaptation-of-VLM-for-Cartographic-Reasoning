"""Dynamic per-step relay decoding and differentiable current-policy replay.

Each decision processes ``[x_t, r_t,1, ..., r_t,R]`` jointly.  A custom block
mask lets the language query read every current relay and lets each relay query
read only image K/V, question K/V, its current language state, and the current
relay bank.  During generation all positions are temporarily appended to every
DynamicCache layer; after the decoder step the relay K/V positions are removed
and only ``x_t`` is retained.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from utils import (_decoder_and_head, _sampling_logits, _vl_backbone,
                   explicit_relay_decoder_inputs, qwen_lm)


RELAY_LAYER_MODES = ("continuous", "reset_shared4", "reset_stage1")
_STAGE_STARTS = (0, 8, 16, 24)


def _active_relay(bank: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in RELAY_LAYER_MODES:
        raise ValueError(f"Unknown relay layer mode: {mode}")
    if mode != "continuous" and bank.shape[0] != 4:
        raise ValueError(f"{mode} requires exactly four trainable relay tokens")
    return bank[:1] if mode == "reset_stage1" else bank


def _stage_relay(bank: torch.Tensor, mode: str, layer_index: int) -> torch.Tensor:
    if mode == "reset_shared4":
        return bank
    if mode == "reset_stage1":
        stage_index = max(i for i, start in enumerate(_STAGE_STARTS)
                          if start <= layer_index)
        return bank[stage_index:stage_index + 1]
    raise ValueError(f"No stage seed for mode {mode}")


def _reset_relay_states(hidden: torch.Tensor, bank: torch.Tensor, mode: str,
                        layer_index: int, prefix_length: int,
                        target_count: int, num_relays: int) -> torch.Tensor:
    """Replace only relay rows; preserve prompt and language hidden states."""
    seed = _stage_relay(bank, mode, layer_index).to(hidden.device, hidden.dtype)
    if seed.shape != (num_relays, hidden.shape[-1]):
        raise RuntimeError("Stage relay shape differs from decoder group width")
    group_size = 1 + num_relays
    if hidden.shape[1] != prefix_length + group_size * target_count:
        raise RuntimeError("Stage reset does not match packed decoder sequence")
    groups = hidden[:, prefix_length:].reshape(
        hidden.shape[0], target_count, group_size, hidden.shape[-1]
    )
    repeated = seed.reshape(1, 1, num_relays, -1).expand(
        hidden.shape[0], target_count, -1, -1
    )
    updated = torch.cat((groups[:, :, :1, :], repeated), dim=2).reshape(
        hidden.shape[0], target_count * group_size, hidden.shape[-1]
    )
    return torch.cat((hidden[:, :prefix_length], updated), dim=1)


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
                         dtype: torch.dtype, device,
                         num_relays: int = 1) -> torch.Tensor:
    if relay_visible_past.shape != (past_length,):
        raise RuntimeError(
            f"Relay-visible mask length {relay_visible_past.numel()} != cache {past_length}"
        )
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    group_size = 1 + num_relays
    allowed = torch.zeros((group_size, past_length + group_size),
                          dtype=torch.bool, device=device)
    allowed[0, :] = True  # x_t reads all causal history, itself, and current r_t
    allowed[1:, :past_length] = relay_visible_past.to(
        device=device, dtype=torch.bool
    ).unsqueeze(0)
    allowed[1:, past_length:] = True  # each relay reads current x_t + relay bank
    return _additive_mask(allowed, dtype)


def _full_attention_mask(prefix_length: int, target_count: int,
                         relay_visible_prefix: torch.Tensor,
                         dtype: torch.dtype, device,
                         num_relays: int = 1) -> torch.Tensor:
    """Mask prompt prefix plus target groups of one predictor and R relays."""
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    group_size = 1 + num_relays
    total = prefix_length + group_size * target_count
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    prefix_visible = relay_visible_prefix.to(device=device, dtype=torch.bool)
    if prefix_visible.shape != (prefix_length,):
        raise RuntimeError("Relay-visible prefix mask does not match the prompt prefix")
    for query in range(prefix_length):
        allowed[query, :query + 1] = True
    prior_language_positions: list[int] = []
    for index in range(target_count):
        language_position = prefix_length + group_size * index
        relay_positions = slice(language_position + 1,
                                language_position + group_size)
        allowed[language_position, :prefix_length] = True
        if prior_language_positions:
            allowed[language_position, prior_language_positions] = True
        allowed[language_position, language_position] = True
        allowed[language_position, relay_positions] = True
        allowed[relay_positions, :prefix_length] = prefix_visible.unsqueeze(0)
        allowed[relay_positions, language_position:language_position + group_size] = True
        prior_language_positions.append(language_position)
    return _additive_mask(allowed, dtype)


def _decoder_layers(language_model, hidden_states: torch.Tensor,
                    position_ids: torch.Tensor, attention_mask: torch.Tensor, *,
                    past_key_values=None, visual_pos_masks=None,
                    deepstack_visual_embeds=None, checkpoint_layers: bool = False,
                    relay_bank: torch.Tensor | None = None,
                    relay_layer_mode: str = "continuous",
                    relay_prefix_length: int = 0,
                    relay_target_count: int = 1,
                    relay_count: int = 1):
    if relay_layer_mode not in RELAY_LAYER_MODES:
        raise ValueError(f"Unknown relay layer mode: {relay_layer_mode}")
    if relay_layer_mode != "continuous":
        if relay_bank is None or len(language_model.layers) <= _STAGE_STARTS[-1]:
            raise ValueError("Layer-segment relay requires a bank and at least 25 decoder layers")
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    for layer_index, layer in enumerate(language_model.layers):
        if (relay_layer_mode != "continuous"
                and layer_index in _STAGE_STARTS):
            hidden_states = _reset_relay_states(
                hidden_states, relay_bank, relay_layer_mode, layer_index,
                relay_prefix_length, relay_target_count, relay_count,
            )
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


def _group_position_ids(position: torch.Tensor, num_relays: int) -> torch.Tensor:
    """Give x_t and all temporary relays the same multimodal RoPE position."""
    if position.ndim != 3 or position.shape[1:] != (1, 1):
        raise ValueError(f"Expected position [D,1,1], got {tuple(position.shape)}")
    if num_relays < 1:
        raise ValueError("num_relays must be positive")
    return position.expand(-1, -1, 1 + num_relays)


def _decode_pair(model, *, token: torch.Tensor, relay: torch.Tensor,
                 position: torch.Tensor, past, relay_visible_past: torch.Tensor,
                 relay_layer_mode: str = "continuous"):
    language_model, lm_head = _decoder_and_head(model)
    token_embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
    if relay.ndim != 2 or relay.shape[0] < 1:
        raise ValueError("relay must have shape [R,H], R >= 1")
    active_relay = _active_relay(relay, relay_layer_mode)
    relay_embed = active_relay.to(token_embed.device, token_embed.dtype).unsqueeze(0)
    pair = torch.cat((token_embed, relay_embed), dim=1)
    num_relays = active_relay.shape[0]
    past_length = past.get_seq_length()
    mask = _step_attention_mask(
        past_length, relay_visible_past, pair.dtype, pair.device, num_relays
    )
    _set_custom_mask_attention(language_model)
    hidden = _decoder_layers(
        language_model, pair, _group_position_ids(position, num_relays), mask,
        past_key_values=past, relay_bank=relay,
        relay_layer_mode=relay_layer_mode, relay_count=num_relays,
    )
    if past.get_seq_length() != past_length + 1 + num_relays:
        raise RuntimeError("DynamicCache did not append x_t and every relay position")
    logits = lm_head(hidden[:, 0, :])
    past.crop(-num_relays)
    if past.get_seq_length() != past_length + 1:
        raise RuntimeError("cache crop did not retain x_t and remove every relay")
    return logits, past


@torch.no_grad()
def generate_with_cot_relay(
        model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        mm_encoder_outputs: dict, relay_embeddings: torch.Tensor,
        relay_visible_prompt_mask: torch.Tensor, policy_kind: str,
        relay_layer_mode: str = "continuous",
        do_sample: bool = False, temperature: float = 1.0,
        top_p: float = 1.0, top_k: int = 0, num_beams: int = 1,
        max_new_tokens: int, return_dict_in_generate: bool = False,
        output_scores: bool = False, use_cache: bool = True):
    if policy_kind != "image_question_cot_shared":
        raise ValueError(f"Unexpected policy_kind={policy_kind!r}")
    if (input_ids.shape[0] != 1 or num_beams != 1
            or relay_embeddings.ndim != 2 or relay_embeddings.shape[0] < 1):
        raise ValueError("Dynamic relay generation requires batch=1, beams=1, relays [R,H]")
    if input_ids.shape[1] < 2 or max_new_tokens < 1 or not use_cache:
        raise ValueError("Prompt must have >=2 tokens; max_new_tokens>0 and cache are required")
    _active_relay(relay_embeddings, relay_layer_mode)

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
            model, token=token, relay=relay_embeddings, position=position,
            past=past, relay_visible_past=visible,
            relay_layer_mode=relay_layer_mode,
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
    if relay.ndim != 2 or relay.shape[0] < 1 or not relay.requires_grad:
        raise RuntimeError("Current-policy scheme-3 replay requires attached relays [R,H]")
    relay_layer_mode = augmented_prompt.get("relay_layer_mode", "continuous")
    active_relay = _active_relay(relay, relay_layer_mode)
    num_relays = active_relay.shape[0]
    group_size = 1 + num_relays

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
    relay_embed = active_relay.to(predictor_embeds.device, predictor_embeds.dtype)
    relay_embeds = relay_embed.view(1, 1, num_relays, -1).expand(
        1, targets.shape[1], num_relays, -1
    )
    groups = torch.cat((predictor_embeds.unsqueeze(2), relay_embeds), dim=2).reshape(
        1, group_size * targets.shape[1], -1
    )
    hidden = torch.cat((prompt_embeds[:, :prefix_length], groups), dim=1)

    first_position = positions[..., -1:]
    extra_positions = [first_position]
    cursor = int(positions.max().item()) + 1
    for _ in range(1, targets.shape[1]):
        extra_positions.append(torch.full_like(first_position, cursor))
        cursor += 1
    paired_positions = torch.cat(
        [position.expand(-1, -1, group_size) for position in extra_positions], dim=-1
    )
    all_positions = torch.cat((positions[..., :prefix_length], paired_positions), dim=-1)
    visible_prefix = augmented_prompt["relay_visible_prompt_mask"][:prefix_length]
    block_mask = _full_attention_mask(
        prefix_length, targets.shape[1], visible_prefix,
        hidden.dtype, hidden.device, num_relays,
    )
    extended_visual_mask = torch.cat((
        visual_mask[:, :prefix_length],
        torch.zeros((1, group_size * targets.shape[1]),
                    dtype=torch.bool, device=hidden.device),
    ), dim=1)
    language_model, lm_head = _decoder_and_head(model)
    _set_custom_mask_attention(language_model)
    with torch.enable_grad():
        decoded = _decoder_layers(
            language_model, hidden, all_positions, block_mask,
            visual_pos_masks=extended_visual_mask,
            deepstack_visual_embeds=deepstack, checkpoint_layers=True,
            relay_bank=relay, relay_layer_mode=relay_layer_mode,
            relay_prefix_length=prefix_length,
            relay_target_count=targets.shape[1], relay_count=num_relays,
        )
        predictor_positions = prefix_length + group_size * torch.arange(
            targets.shape[1], device=hidden.device
        )
        logits = lm_head(decoded[:, predictor_positions, :])
        selected = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(0).squeeze(-1)
    if not selected.requires_grad:
        raise RuntimeError("Scheme-3 current-policy log probabilities are detached")
    return selected
