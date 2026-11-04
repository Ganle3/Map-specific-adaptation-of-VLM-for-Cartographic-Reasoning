"""Greedy Relay4 decoding with final-state selection and layer-0 refinement."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Relay4DecodeState:
    active_actions: list[int] = field(default_factory=list)
    active_local_memories: list[torch.Tensor] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


def _flatten_pooler(image_features) -> torch.Tensor:
    value = image_features.pooler_output
    return value if isinstance(value, torch.Tensor) else torch.cat(tuple(value), dim=0)


def _all_visible_mask(past_length: int, dtype: torch.dtype, device) -> torch.Tensor:
    return torch.zeros((1, 1, 1, past_length + 1), dtype=dtype, device=device)


def _crop_cache(cache, length: int) -> None:
    """Remove provisional current-token KV while retaining causal history."""
    crop = getattr(cache, "crop", None)
    if crop is not None:
        remove = cache.get_seq_length() - length
        if remove < 0:
            raise RuntimeError("Requested cache rollback length exceeds current cache")
        if remove:
            crop(-remove)
    else:
        for layer in cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.keys = layer.keys[..., :length, :]
                layer.values = layer.values[..., :length, :]
    if cache.get_seq_length() != length:
        raise RuntimeError(
            f"Could not roll provisional cache back to {length}; got {cache.get_seq_length()}"
        )


def _decoder_step(language_model, hidden, step_mask, position_embeddings, past):
    for layer in language_model.layers:
        hidden = layer(
            hidden, attention_mask=step_mask,
            position_embeddings=position_embeddings,
            past_key_values=past, use_cache=True,
        )
    return language_model.norm(hidden)


@torch.no_grad()
def generate_with_new_visual_relay(
        model, *, source_inputs: dict, global_features, policy, coarse_grid,
        tile_cache, tokenizer, max_new_tokens: int = 500,
        lookup_start_token: int = 10, lookup_every: int = 10,
        relay_layer: int = 0, sample_tiles: bool = False):
    """Generate with a provisional and refined pass on every LOOK step.

    The provisional pass obtains the current token's final decoder hidden
    state. That state selects and reads original-resolution evidence. Its new
    KV is discarded. The same token is then rerun with the visual readout
    injected before decoder layer 0; only refined KV remains persistent.
    """
    if max_new_tokens < 1 or lookup_start_token < 1 or lookup_every < 1:
        raise ValueError("token limits and lookup interval must be positive")
    if relay_layer != 0:
        raise ValueError("Two-pass Relay4 requires relay_layer=0")

    from utils import (_decoder_and_head, _sampling_logits, _vl_backbone,
                       explicit_relay_decoder_inputs, qwen_lm)

    input_ids = source_inputs["input_ids"]
    attention_mask = source_inputs["attention_mask"]
    mm_types = source_inputs["mm_token_type_ids"]
    grid_thw = source_inputs["image_grid_thw"]
    if input_ids.shape[0] != 1:
        raise ValueError("Relay4 demo supports one question at a time")

    global_memory = _flatten_pooler(global_features).to(
        input_ids.device, qwen_lm(model).get_input_embeddings().weight.dtype
    ).unsqueeze(0)
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, input_ids, global_features
    )
    positions, _ = _vl_backbone(model).get_rope_index(
        input_ids, mm_types, image_grid_thw=grid_thw,
        attention_mask=attention_mask,
    )
    language_model, lm_head = _decoder_and_head(model)
    prefill = language_model(
        input_ids=None, inputs_embeds=prompt_embeds,
        attention_mask=attention_mask, position_ids=positions,
        visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack,
        use_cache=True, return_dict=True,
    )
    past = prefill.past_key_values
    next_logits = lm_head(prefill.last_hidden_state[:, -1, :])
    cursor = int(positions.max().item()) + 1
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    generated: list[torch.Tensor] = []
    state = Relay4DecodeState()

    for generated_count in range(1, max_new_tokens + 1):
        token = _sampling_logits(next_logits, 1.0, 1.0, 0).argmax(dim=-1)
        generated.append(token)
        if int(token.item()) in eos_ids or generated_count == max_new_tokens:
            break

        hidden = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
        position = torch.full(
            (3, 1, 1), cursor, dtype=torch.long, device=input_ids.device,
        )
        position_embeddings = language_model.rotary_emb(hidden, position)
        past_length = past.get_seq_length()
        step_mask = _all_visible_mask(past_length, hidden.dtype, hidden.device)
        should_lookup = generated_count >= lookup_start_token and (
            (generated_count - lookup_start_token) % lookup_every == 0
        )

        if should_lookup:
            provisional = _decoder_step(
                language_model, hidden.clone(), step_mask, position_embeddings, past
            )
            _crop_cache(past, past_length)
            selection = policy.select_global_patches(
                provisional, global_memory, sample=sample_tiles
            )
            actions = [int(value) for value in selection["actions"][0].tolist()]
            state.active_actions = actions
            state.active_local_memories = []
            event = {
                "generated_token_count": generated_count,
                "affects_prediction_from_token": generated_count + 1,
                "controller_hidden_source": "provisional_final_decoder_state",
                "visual_injection_point": "before_decoder_layer_0",
                "decoder_layer_zero_based": 0,
                "reasoning_so_far": tokenizer.decode(
                    torch.stack(generated, dim=1)[0], skip_special_tokens=True
                ),
                "relay_selections": [],
            }
            for relay_index, action in enumerate(actions):
                geometry = coarse_grid.geometry(action)
                local_memory, cache_hit, metadata = tile_cache.get(geometry)
                state.active_local_memories.append(local_memory.unsqueeze(0))
                probabilities = selection["patch_probabilities"][0, relay_index]
                top_values, top_indices = probabilities.topk(min(5, probabilities.numel()))
                event["relay_selections"].append({
                    "relay_index": relay_index,
                    "selected_probability": float(probabilities[action]),
                    "selected_log_probability": float(
                        selection["selected_log_probs"][0, relay_index]
                    ),
                    "top5_actions": [int(value) for value in top_indices.cpu().tolist()],
                    "top5_probabilities": [float(value) for value in top_values.cpu().tolist()],
                    "cache_hit": bool(cache_hit),
                    **metadata,
                })
                print(
                    f"[RELAY4 LOOKUP token={generated_count} relay={relay_index}] "
                    f"action={action} row={geometry.row} col={geometry.col} "
                    f"bbox={list(geometry.bbox)} cache_hit={cache_hit} "
                    f"local_tokens={metadata['local_visual_tokens']}",
                    flush=True,
                )

            local_memories = [
                memory.to(hidden.device, hidden.dtype)
                for memory in state.active_local_memories
            ]
            visual_relays, _, local_attentions = policy.read_assigned_local_memories(
                provisional, local_memories
            )
            hidden, _, relay_attention = policy.fuse_hidden_state(hidden, visual_relays)
            for relay_event, local_attention in zip(
                    event["relay_selections"], local_attentions):
                relay_event["local_attention_entropy"] = float(
                    -(local_attention.clamp_min(1e-12)
                      * local_attention.clamp_min(1e-12).log()).sum(dim=-1).mean()
                )
            event["hidden_to_relay_probabilities"] = [
                float(value) for value in relay_attention[0, 0].cpu().tolist()
            ]
            state.trace.append(event)

        hidden = _decoder_step(
            language_model, hidden, step_mask, position_embeddings, past
        )
        next_logits = lm_head(hidden[:, -1, :])
        cursor += 1

    completion = torch.stack(generated, dim=1)
    response = tokenizer.decode(completion[0], skip_special_tokens=True)
    return completion, response, state
