"""Language-token PPO/GRPO loss whose only trainable source is prototype memory."""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint


def _qwen_language_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
                          image_features, *, checkpoint_decoder: bool = True,
                          relay_embeddings: torch.Tensor | None = None,
                          relay_positions: torch.Tensor | None = None) -> torch.Tensor:
    """Run Qwen's decoder from differentiable pre-encoded visual embeddings.

    Some Transformers versions used on Euler accept ``mm_encoder_outputs``
    but detach those cached tensors inside the multimodal wrapper.  Build the
    exact decoder inputs explicitly so the prototype-memory graph is retained.
    Frozen Qwen/PEFT weights still participate in autograd with respect to the
    inputs; they remain excluded from the optimizer.
    """
    qwen = model.get_base_model() if hasattr(model, "get_base_model") else model
    vl_model = getattr(qwen, "model", None)
    language_model = getattr(vl_model, "language_model", None)
    lm_head = getattr(qwen, "lm_head", None)
    if vl_model is None or language_model is None or lm_head is None:
        raise RuntimeError("Could not locate Qwen3-VL language decoder and LM head")

    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    if (relay_embeddings is None) != (relay_positions is None):
        raise ValueError("relay_embeddings and relay_positions must be supplied together")
    if relay_embeddings is not None:
        positions = relay_positions.to(input_ids.device, dtype=torch.long).flatten()
        if positions.numel() != relay_embeddings.shape[0]:
            raise RuntimeError("Relay position/embedding count mismatch")
        start = int(positions[0])
        expected = torch.arange(start, start + positions.numel(), device=positions.device)
        if not torch.equal(positions, expected):
            raise RuntimeError("Relay positions must form one contiguous span")
        end = start + int(positions.numel())
        relays = relay_embeddings.to(inputs_embeds.device, inputs_embeds.dtype).unsqueeze(0)
        inputs_embeds = torch.cat(
            (inputs_embeds[:, :start], relays, inputs_embeds[:, end:]), dim=1
        )
    pooled = torch.cat(tuple(image_features.pooler_output), dim=0).to(
        device=inputs_embeds.device, dtype=inputs_embeds.dtype
    )
    image_token_id = int(qwen.config.image_token_id)
    visual_pos_masks = input_ids.eq(image_token_id)
    if int(visual_pos_masks.sum()) != int(pooled.shape[0]):
        raise RuntimeError(
            "Visual placeholder/embedding mismatch: "
            f"tokens={int(visual_pos_masks.sum())}, embeddings={pooled.shape[0]}"
        )
    scatter_mask = visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(scatter_mask, pooled)
    if not inputs_embeds.requires_grad:
        raise RuntimeError("Differentiable prototype memory did not reach Qwen inputs_embeds")

    deepstack = []
    for level in image_features.deepstack_features:
        tensors = (level,) if isinstance(level, torch.Tensor) else tuple(level)
        if any(tensor.ndim != 2 for tensor in tensors):
            raise RuntimeError(
                "Expected each DeepStack image feature to have shape [N,H], got "
                f"{[tuple(tensor.shape) for tensor in tensors]}"
            )
        deepstack.append(
            torch.cat(tensors, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        )
    rope_owner = qwen
    for _ in range(4):
        if hasattr(rope_owner, "get_rope_index"):
            break
        rope_owner = getattr(rope_owner, "model", None)
        if rope_owner is None:
            raise RuntimeError("Could not locate Qwen3-VL get_rope_index")
    position_ids, _ = rope_owner.get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    # Only the prototype input requires gradients; all decoder weights are
    # frozen.  A full 500-token completion nevertheless retains every decoder
    # layer's activations unless the layers are checkpointed, which exceeds a
    # 24 GiB 4090 for large maps.  Temporarily activate the Transformers layer
    # checkpoint wrapper for this differentiable current-policy pass.  Setting
    # the parent layer's flag directly leaves all frozen child modules in eval
    # mode and therefore does not introduce dropout.  Rollout generation never
    # enters this function and remains unchanged.
    layer_states = []
    if checkpoint_decoder:
        checkpoint_func = partial(checkpoint, use_reentrant=False)
        for layer in language_model.layers:
            layer_states.append((
                layer,
                layer.training,
                getattr(layer, "gradient_checkpointing", False),
                getattr(layer, "_gradient_checkpointing_func", None),
            ))
            layer.training = True
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = checkpoint_func
    try:
        decoder_output = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for layer, was_training, was_checkpointing, old_checkpoint_func in layer_states:
            layer.training = was_training
            layer.gradient_checkpointing = was_checkpointing
            if old_checkpoint_func is None:
                try:
                    delattr(layer, "_gradient_checkpointing_func")
                except AttributeError:
                    pass
            else:
                layer._gradient_checkpointing_func = old_checkpoint_func
    logits = lm_head(decoder_output.last_hidden_state)
    if not logits.requires_grad:
        raise RuntimeError("Qwen decoder logits are detached from explicit prototype inputs_embeds")
    return logits


def group_advantages(rewards: list[float], device) -> torch.Tensor:
    values = torch.tensor(rewards, dtype=torch.float32, device=device)
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-6)


def completion_log_probs(model, augmented_prompt: dict, completion_ids: torch.Tensor) -> torch.Tensor:
    """Recompute sampled-language log probabilities with a graph back to prototypes."""
    prefix_ids = augmented_prompt["input_ids"]
    completion_ids = completion_ids.to(prefix_ids.device).view(1, -1)
    full_ids = torch.cat((prefix_ids, completion_ids), dim=1)
    full_mask = torch.ones_like(full_ids)
    mm_types = torch.cat((augmented_prompt["mm_token_type_ids"],
                          torch.zeros_like(completion_ids)), dim=1)
    visual = augmented_prompt["mm_encoder_outputs"]["image"]
    is_relay = augmented_prompt.get("policy_kind") == "only_visual_relay"
    if is_relay:
        prototype_memory = augmented_prompt["relay_embeddings"]
        relay_positions = augmented_prompt["relay_positions"]
        if not prototype_memory.requires_grad:
            raise RuntimeError("Visual relay embeddings are detached before current-policy forward")
    else:
        prototype_memory = visual.pooler_output[-1]
        relay_positions = None
        if not prototype_memory.requires_grad:
            raise RuntimeError(
                "Prototype visual memory is detached before the Qwen current-policy forward"
            )
    with torch.enable_grad():
        logits = _qwen_language_logits(
            model, full_ids, full_mask, mm_types,
            augmented_prompt["image_grid_thw"], visual,
            relay_embeddings=prototype_memory if is_relay else None,
            relay_positions=relay_positions,
        )
    prefix_length = prefix_ids.shape[1]
    # Logit at prefix_length-1 predicts completion token 0.
    logits = logits[:, prefix_length - 1:prefix_length + completion_ids.shape[1] - 1, :].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(0).squeeze(-1)


def clipped_language_grpo_loss(current_logps: torch.Tensor, old_logps: torch.Tensor,
                               advantage: torch.Tensor, completion_mask: torch.Tensor,
                               clip_ratio: float = 0.2) -> torch.Tensor:
    """Independent token ratios; normalize within the trajectory, then across its group."""
    old_logps = old_logps.to(current_logps.device, dtype=current_logps.dtype)
    mask = completion_mask.to(current_logps.device, dtype=current_logps.dtype)
    ratio = torch.exp(current_logps - old_logps)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    surrogate = torch.minimum(unclipped, clipped)
    return -(surrogate * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def off_policy_sequence_keep(current_logps: torch.Tensor, old_logps: torch.Tensor,
                             advantage: torch.Tensor, completion_mask: torch.Tensor,
                             threshold: float | None) -> tuple[bool, float]:
    """DeepSeek/TRL negative-sequence off-policy mask.

    The rollout-time log probabilities are the sampling policy.  Positive
    advantage trajectories are always retained.  A negative trajectory is
    dropped only when its mean forward-KL estimate exceeds ``threshold``::

        mean_t(log pi_old(a_t) - log pi_current(a_t)) > threshold

    This intentionally mirrors ``GRPOTrainer.get_off_policy_mask`` while
    operating on one variable-length trajectory at a time.
    """
    mask = completion_mask.to(current_logps.device, dtype=current_logps.dtype)
    old = old_logps.to(current_logps.device, dtype=current_logps.dtype)
    mean_kl = float((((old - current_logps.detach()) * mask).sum()
                     / mask.sum().clamp_min(1.0)).cpu())
    if threshold is None:
        return True, mean_kl
    keep = float(advantage) >= 0.0 or mean_kl <= threshold
    return keep, mean_kl
