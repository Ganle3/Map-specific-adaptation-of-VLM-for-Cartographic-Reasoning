"""Native Qwen3-VL prefill for trainable visual relay tokens.

Sequence layout::

    <vision_start> image placeholders <vision_end> R1 ... Rn question ...

R1..Rn use multimodal token type 0 (text/ordinary decoder positions).  The
processor's image placeholder count, ``image_grid_thw`` and DeepStack tensors
remain unchanged.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from prototype_prefill import _sampling_logits, _vl_backbone, qwen_lm


def _image_tuple(value):
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def relay_insertion_index(model, source_inputs: dict) -> int:
    """Return the position immediately after the original vision-end token."""
    ids = source_inputs["input_ids"][0]
    config = qwen_lm(model).config
    image_positions = ids.eq(int(config.image_token_id)).nonzero().flatten()
    if not len(image_positions):
        raise RuntimeError("Processor input contains no image placeholder tokens")
    final_image = int(image_positions[-1])
    vision_end = int(config.vision_end_token_id)
    later_end = ids[final_image + 1:].eq(vision_end).nonzero().flatten()
    if not len(later_end):
        raise RuntimeError("Cannot locate <vision_end> after the final image placeholder")
    end_position = final_image + 1 + int(later_end[0])
    return end_position + 1


def _carrier_token_id(model, tokenizer) -> int:
    """Choose an existing vocabulary ID whose embedding is always replaced."""
    qwen = qwen_lm(model)
    candidate = tokenizer.pad_token_id
    if candidate is None:
        candidate = getattr(qwen.config, "pad_token_id", None)
    if candidate is None:
        candidate = tokenizer.eos_token_id
    if isinstance(candidate, (list, tuple, set)):
        candidate = next(iter(candidate))
    if candidate is None:
        raise RuntimeError("Tokenizer has neither a pad nor EOS carrier token")
    forbidden = {
        int(qwen.config.image_token_id), int(qwen.config.video_token_id),
        int(qwen.config.vision_start_token_id), int(qwen.config.vision_end_token_id),
    }
    if int(candidate) in forbidden:
        raise RuntimeError("Chosen relay carrier collides with a multimodal special token")
    return int(candidate)


def build_only_visual_relay_prefill(model, tokenizer, source_inputs: dict,
                                    global_features, policy, *, with_grad: bool) -> tuple[dict, None]:
    """Insert relay positions without creating synthetic visual placeholders."""
    # This is the nn.Parameter.
    # Rollout construction uses an explicit detached view; current-policy recomputation keeps the graph.
    relays = policy.prototypes if with_grad else policy.prototypes.detach()
    count = int(relays.shape[0])
    index = relay_insertion_index(model, source_inputs)
    carrier = _carrier_token_id(model, tokenizer)

    source_ids = source_inputs["input_ids"][0]
    source_types = source_inputs["mm_token_type_ids"][0]
    source_mask = source_inputs["attention_mask"][0]
    relay_ids = source_ids.new_full((count,), carrier)
    # Official Qwen3-VL semantics: 0=text, 1=image, 2=video.  Relays are
    # ordinary decoder positions whose input embeddings are overridden below.
    relay_types = source_types.new_zeros((count,))
    relay_mask = source_mask.new_ones((count,))

    def splice(source: torch.Tensor, addition: torch.Tensor) -> torch.Tensor:
        return torch.cat((source[:index], addition, source[index:]), dim=0).unsqueeze(0)

    augmented = {
        "input_ids": splice(source_ids, relay_ids),
        "attention_mask": splice(source_mask, relay_mask),
        "mm_token_type_ids": splice(source_types, relay_types),
        "image_grid_thw": source_inputs["image_grid_thw"],
        "mm_encoder_outputs": {"image": global_features},
        "relay_embeddings": relays,
        "relay_positions": torch.arange(index, index + count, device=source_ids.device),
        "policy_kind": "only_visual_relay",
    }
    return augmented, None


def explicit_relay_decoder_inputs(model, input_ids: torch.Tensor, image_features,
                                  relay_embeddings: torch.Tensor,
                                  relay_positions: torch.Tensor):
    """Create decoder embeddings while preserving gradient to relay embeddings."""
    qwen = qwen_lm(model)
    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    positions = relay_positions.to(input_ids.device, dtype=torch.long).flatten()
    if positions.numel() != relay_embeddings.shape[0]:
        raise RuntimeError("Relay position/embedding count mismatch")
    if positions.numel() and not torch.equal(
            positions, torch.arange(int(positions[0]), int(positions[0]) + positions.numel(),
                                    device=positions.device)):
        raise RuntimeError("Relay positions must form one contiguous span")
    start = int(positions[0])
    end = start + int(positions.numel())
    relays = relay_embeddings.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(0)
    inputs_embeds = torch.cat(
        (inputs_embeds[:, :start], relays, inputs_embeds[:, end:]), dim=1
    )

    pooled = torch.cat(_image_tuple(image_features.pooler_output), dim=0).to(
        inputs_embeds.device, inputs_embeds.dtype
    )
    visual_pos_masks = input_ids.eq(int(qwen.config.image_token_id))
    if int(visual_pos_masks.sum()) != int(pooled.shape[0]):
        raise RuntimeError(
            f"Original image placeholder/feature mismatch: "
            f"tokens={int(visual_pos_masks.sum())}, embeddings={pooled.shape[0]}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(
        visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds), pooled
    )
    deepstack = [
        torch.cat(_image_tuple(level), dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        for level in image_features.deepstack_features
    ]
    return inputs_embeds, visual_pos_masks, deepstack


def _decoder_and_head(model):
    qwen = qwen_lm(model)
    vl_model = getattr(qwen, "model", None)
    language_model = getattr(vl_model, "language_model", None)
    lm_head = getattr(qwen, "lm_head", None)
    if language_model is None or lm_head is None:
        raise RuntimeError("Could not locate Qwen3-VL language decoder and LM head")
    return language_model, lm_head


@torch.no_grad()
def generate_from_visual_relays(
        model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        mm_encoder_outputs: dict, relay_embeddings: torch.Tensor,
        relay_positions: torch.Tensor, policy_kind: str = "only_visual_relay",
        do_sample: bool = False, temperature: float = 1.0,
        top_p: float = 1.0, top_k: int = 0, num_beams: int = 1,
        max_new_tokens: int, return_dict_in_generate: bool = False,
        output_scores: bool = False, use_cache: bool = True):
    """Generate after an explicit native prefill containing relay embeddings."""
    if policy_kind != "only_visual_relay":
        raise ValueError(f"Unexpected policy_kind={policy_kind!r}")
    if input_ids.shape[0] != 1 or num_beams != 1:
        raise ValueError("Visual-relay generation supports batch size 1 and num_beams=1")
    if max_new_tokens < 1 or not use_cache:
        raise ValueError("max_new_tokens must be positive and use_cache must be True")

    visual = mm_encoder_outputs["image"]
    inputs_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, input_ids, visual, relay_embeddings, relay_positions
    )
    position_ids, _ = _vl_backbone(model).get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    language_model, lm_head = _decoder_and_head(model)
    output = language_model(
        input_ids=None, inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, position_ids=position_ids,
        visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack,
        use_cache=True, return_dict=True,
    )
    past = output.past_key_values
    if past.get_seq_length() != input_ids.shape[1]:
        raise RuntimeError("Relay prefill cache length does not match augmented prompt length")
    next_logits = lm_head(output.last_hidden_state[:, -1, :])
    cursor = int(position_ids.max().item()) + 1
    tokens: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    eos = qwen_lm(model).generation_config.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}

    for _ in range(max_new_tokens):
        step_scores = _sampling_logits(next_logits, temperature, top_p, top_k)
        token = (torch.distributions.Categorical(logits=step_scores).sample()
                 if do_sample else step_scores.argmax(dim=-1))
        tokens.append(token)
        if output_scores:
            scores.append(step_scores)
        if int(token.item()) in eos_ids:
            break
        full_mask = torch.ones(
            (1, past.get_seq_length() + 1), dtype=attention_mask.dtype,
            device=input_ids.device,
        )
        token_position = torch.full(
            (3, 1, 1), cursor, dtype=torch.long, device=input_ids.device,
        )
        output = language_model(
            input_ids=token.view(1, 1), attention_mask=full_mask,
            position_ids=token_position, past_key_values=past,
            use_cache=True, return_dict=True,
        )
        past = output.past_key_values
        cursor += 1
        next_logits = lm_head(output.last_hidden_state[:, -1, :])

    completion = torch.stack(tokens, dim=1)
    sequences = torch.cat((input_ids, completion), dim=1)
    if return_dict_in_generate:
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))
    return sequences
