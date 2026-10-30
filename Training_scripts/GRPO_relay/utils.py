"""Insert cross-attended prototype memory once during Qwen3-VL prefill."""
from __future__ import annotations

import math
from functools import partial
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


def qwen_lm(model):
    """Return Qwen3VLForConditionalGeneration beneath an optional PEFT wrapper."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _vl_backbone(model):
    """Locate the Qwen3VLModel that owns multimodal RoPE construction."""
    candidate = qwen_lm(model)
    for _ in range(4):
        if hasattr(candidate, "get_rope_index"):
            return candidate
        candidate = getattr(candidate, "model", None)
        if candidate is None:
            break
    raise AttributeError("Could not locate Qwen3VLModel.get_rope_index")


def _sampling_logits(logits: torch.Tensor, temperature: float,
                     top_p: float, top_k: int) -> torch.Tensor:
    """Apply the three sampling transforms used by the rollout configuration."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    scores = logits.float() / temperature
    if top_k:
        keep = min(top_k, scores.shape[-1])
        cutoff = scores.topk(keep, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        remove = sorted_scores.softmax(dim=-1).cumsum(dim=-1) > top_p
        # Keep the first token above the cumulative threshold, as HF does.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(
            -1, sorted_indices, sorted_scores
        )
    return scores


@torch.no_grad()
def generate_from_preencoded_vision(
        model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        mm_encoder_outputs: dict, do_sample: bool = False,
        temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
        num_beams: int = 1, max_new_tokens: int,
        return_dict_in_generate: bool = False, output_scores: bool = False,
        use_cache: bool = True):
    """Single-example generation that accepts pre-encoded Qwen visual features.

    Transformers versions used by the Euler container reject the valid
    ``mm_encoder_outputs`` kwarg inside ``GenerationMixin`` before Qwen's
    forward is reached. This explicit autoregressive loop calls Qwen/PEFT
    forward directly: prefill consumes the cached visual features once, and
    subsequent tokens reuse the returned KV cache. Its result matches the two
    output forms used by this project (plain sequences or sequences+scores).
    """
    if input_ids.shape[0] != 1:
        raise ValueError("Cached prototype generation currently supports batch size 1")
    if num_beams != 1:
        raise ValueError("Cached prototype generation supports num_beams=1 only")
    if max_new_tokens < 1 or not use_cache:
        raise ValueError("max_new_tokens must be positive and use_cache must be True")

    position_ids, _ = _vl_backbone(model).get_rope_index(
        input_ids, mm_token_type_ids, image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )
    output = model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw, mm_encoder_outputs=mm_encoder_outputs,
        use_cache=True, return_dict=True,
    )
    past = output.past_key_values
    if past.get_seq_length() != input_ids.shape[1]:
        raise RuntimeError(
            f"Prefill cache length {past.get_seq_length()} does not match prompt "
            f"length {input_ids.shape[1]}"
        )
    cursor = int(position_ids.max().item()) + 1
    next_logits = output.logits[:, -1, :]
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
        output = model(
            input_ids=token.view(1, 1), attention_mask=full_mask,
            position_ids=token_position, past_key_values=past,
            use_cache=True, return_dict=True,
        )
        past = output.past_key_values
        cursor += 1
        next_logits = output.logits[:, -1, :]

    completion = torch.stack(tokens, dim=1)
    sequences = torch.cat((input_ids, completion), dim=1)
    if return_dict_in_generate:
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))
    return sequences


def encode_global_image(model, processor_inputs: dict):
    """Run the frozen ViT+merger once; returned tensors are detached reusable constants."""
    base = qwen_lm(model)
    with torch.no_grad():
        features = base.get_image_features(
            processor_inputs["pixel_values"], processor_inputs["image_grid_thw"], return_dict=True
        )
    def image_tuple(value):
        # Transformers releases differ here: one returns a tuple containing
        # one [N,H] tensor per image, another returns the [N,H] tensor itself.
        # Never iterate a tensor, which would turn it into N separate [H] rows.
        if isinstance(value, torch.Tensor):
            return (value.detach().clone(),)
        return tuple(t.detach().clone() for t in value)

    return BaseModelOutputWithDeepstackFeatures(
        pooler_output=image_tuple(features.pooler_output),
        deepstack_features=[image_tuple(level) for level in features.deepstack_features],
    )


def encode_global_image_with_grad(model, processor_inputs: dict):
    """Run ViT+merger with autograd for joint relay + visual-LoRA updates."""
    base = qwen_lm(model)
    vl_model = getattr(base, "model", None)
    visual = getattr(vl_model, "visual", None)
    blocks = getattr(visual, "blocks", None)
    if blocks is None:
        raise RuntimeError("Could not locate Qwen3-VL visual blocks for checkpointing")

    # The joint adapter includes every ViT block. At a one-million-pixel cap,
    # retaining all 27 eager-attention block activations exceeds a 20/24 GiB
    # GPU. Match the original adapter trainer's checkpointed training behavior.
    block_states = []
    checkpoint_func = partial(checkpoint, use_reentrant=False)
    for block in blocks:
        block_states.append((
            block,
            block.training,
            getattr(block, "gradient_checkpointing", False),
            getattr(block, "_gradient_checkpointing_func", None),
        ))
        block.training = True
        block.gradient_checkpointing = True
        block._gradient_checkpointing_func = checkpoint_func
    try:
        features = base.get_image_features(
            processor_inputs["pixel_values"], processor_inputs["image_grid_thw"],
            return_dict=True,
        )
    finally:
        for block, was_training, was_checkpointing, old_checkpoint_func in block_states:
            block.training = was_training
            block.gradient_checkpointing = was_checkpointing
            if old_checkpoint_func is None:
                try:
                    delattr(block, "_gradient_checkpointing_func")
                except AttributeError:
                    pass
            else:
                block._gradient_checkpointing_func = old_checkpoint_func

    def image_tuple(value):
        return (value,) if isinstance(value, torch.Tensor) else tuple(value)

    result = BaseModelOutputWithDeepstackFeatures(
        pooler_output=image_tuple(features.pooler_output),
        deepstack_features=[image_tuple(level) for level in features.deepstack_features],
    )
    tensors = [*result.pooler_output,
               *(tensor for level in result.deepstack_features for tensor in level)]
    if not any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "Differentiable ViT/merger recomputation produced no trainable visual graph"
        )
    return result




def _image_tuple(value):
    return (value,) if isinstance(value, torch.Tensor) else tuple(value)


def decoder_hidden_size(model) -> int:
    """Read the language hidden size from the loaded Qwen3-VL configuration."""
    config = qwen_lm(model).config
    text_config = getattr(config, "text_config", None)
    if text_config is None and hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
    if text_config is None or not getattr(text_config, "hidden_size", None):
        raise RuntimeError("Qwen3-VL text hidden size is unavailable")
    return int(text_config.hidden_size)


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


def question_relay_insertion_index(tokenizer, source_inputs: dict) -> int:
    """Return the position before the user ``<|im_end|>``/assistant prefix.

    The cartographic prompt contains a single user turn.  Using its final
    ``<|im_end|>`` keeps the relay after all question/instruction tokens while
    leaving the assistant generation prefix intact.
    """
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end) != 1:
        raise RuntimeError("Unexpected Qwen <|im_end|> tokenization")
    positions = source_inputs["input_ids"][0].eq(int(im_end[0])).nonzero().flatten()
    if not len(positions):
        raise RuntimeError("Cannot locate the user-turn end before the assistant prefix")
    return int(positions[-1])


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


def build_relay_prefill(model, tokenizer, source_inputs: dict, global_features,
                        policy, *, scheme: str, with_grad: bool) -> tuple[dict, None]:
    """Insert one or two placements backed by the same relay parameter bank.

    ``only_visual`` produces ``image <vision_end> P question``.
    ``image_question_shared`` produces
    ``image <vision_end> P question P assistant``.  The two spans in the
    second layout reference the same ``policy.prototypes`` tensor.
    """
    if scheme not in {"only_visual", "image_question_shared"}:
        raise ValueError(f"Unknown relay scheme: {scheme!r}")
    # This is the nn.Parameter.
    # Rollout construction uses an explicit detached view; current-policy recomputation keeps the graph.
    relays = policy.prototypes if with_grad else policy.prototypes.detach()
    count = int(relays.shape[0])
    visual_index = relay_insertion_index(model, source_inputs)
    original_indices = [visual_index]
    if scheme == "image_question_shared":
        question_index = question_relay_insertion_index(tokenizer, source_inputs)
        if question_index <= visual_index:
            raise RuntimeError("Question relay position must follow the visual relay position")
        original_indices.append(question_index)
    carrier = _carrier_token_id(model, tokenizer)

    source_ids = source_inputs["input_ids"][0]
    source_types = source_inputs["mm_token_type_ids"][0]
    source_mask = source_inputs["attention_mask"][0]
    relay_ids = source_ids.new_full((count,), carrier)
    # Official Qwen3-VL semantics: 0=text, 1=image, 2=video.  Relays are
    # ordinary decoder positions whose input embeddings are overridden below.
    relay_types = source_types.new_zeros((count,))
    relay_mask = source_mask.new_ones((count,))

    def splice_all(source: torch.Tensor, addition: torch.Tensor) -> torch.Tensor:
        result = source
        for index in reversed(original_indices):
            result = torch.cat((result[:index], addition, result[index:]), dim=0)
        return result.unsqueeze(0)

    visual_positions = torch.arange(
        visual_index, visual_index + count, device=source_ids.device
    )
    position_groups = [visual_positions]
    if scheme == "image_question_shared":
        # Inserting the earlier visual span shifts the later question span.
        question_start = original_indices[1] + count
        position_groups.append(torch.arange(
            question_start, question_start + count, device=source_ids.device
        ))
    relay_positions = torch.stack(position_groups, dim=0)

    augmented = {
        "input_ids": splice_all(source_ids, relay_ids),
        "attention_mask": splice_all(source_mask, relay_mask),
        "mm_token_type_ids": splice_all(source_types, relay_types),
        "image_grid_thw": source_inputs["image_grid_thw"],
        "mm_encoder_outputs": {"image": global_features},
        "relay_embeddings": relays,
        "relay_positions": relay_positions,
        "policy_kind": scheme,
    }
    return augmented, None


def build_adapter_only_prefill(source_inputs: dict, global_features) -> tuple[dict, None]:
    """Use the unmodified processor sequence for the matched LoRA-only control."""
    return {
        "input_ids": source_inputs["input_ids"],
        "attention_mask": source_inputs["attention_mask"],
        "mm_token_type_ids": source_inputs["mm_token_type_ids"],
        "image_grid_thw": source_inputs["image_grid_thw"],
        "mm_encoder_outputs": {"image": global_features},
        "policy_kind": "adapter_only",
    }, None


def explicit_relay_decoder_inputs(model, input_ids: torch.Tensor, image_features,
                                  relay_embeddings: torch.Tensor | None = None,
                                  relay_positions: torch.Tensor | None = None):
    """Create decoder embeddings while preserving gradient to relay embeddings."""
    qwen = qwen_lm(model)
    inputs_embeds = qwen.get_input_embeddings()(input_ids)
    if (relay_embeddings is None) != (relay_positions is None):
        raise ValueError("relay_embeddings and relay_positions must be supplied together")
    if relay_embeddings is not None:
        groups = relay_positions.to(input_ids.device, dtype=torch.long)
        if groups.ndim == 1:
            groups = groups.unsqueeze(0)
        if groups.ndim != 2 or groups.shape[1] != relay_embeddings.shape[0]:
            raise RuntimeError("Each relay placement must contain one position per shared relay")
        flat_positions = groups.flatten()
        if flat_positions.unique().numel() != flat_positions.numel():
            raise RuntimeError("Relay placements overlap")
        relays = relay_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
        repeated_relays = relays.unsqueeze(0).expand(groups.shape[0], -1, -1).reshape(
            1, flat_positions.numel(), relays.shape[-1]
        )
        inputs_embeds = inputs_embeds.index_copy(1, flat_positions, repeated_relays)

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
def generate_from_relays(
        model, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        mm_token_type_ids: torch.Tensor, image_grid_thw: torch.Tensor,
        mm_encoder_outputs: dict, policy_kind: str,
        relay_embeddings: torch.Tensor | None = None,
        relay_positions: torch.Tensor | None = None,
        do_sample: bool = False, temperature: float = 1.0,
        top_p: float = 1.0, top_k: int = 0, num_beams: int = 1,
        max_new_tokens: int, return_dict_in_generate: bool = False,
        output_scores: bool = False, use_cache: bool = True):
    """Generate after an explicit native prefill containing relay embeddings."""
    if policy_kind not in {"adapter_only", "only_visual", "image_question_shared"}:
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
        groups = relay_positions.to(input_ids.device, dtype=torch.long)
        if groups.ndim == 1:
            groups = groups.unsqueeze(0)
        if groups.ndim != 2 or groups.shape[1] != relay_embeddings.shape[0]:
            raise RuntimeError("Each relay placement must contain one position per shared relay")
        flat_positions = groups.flatten()
        relays = relay_embeddings.to(inputs_embeds.device, inputs_embeds.dtype)
        repeated = relays.unsqueeze(0).expand(groups.shape[0], -1, -1).reshape(
            1, flat_positions.numel(), relays.shape[-1]
        )
        inputs_embeds = inputs_embeds.index_copy(1, flat_positions, repeated)
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
    policy_kind = augmented_prompt.get("policy_kind")
    is_relay = policy_kind in {
        "only_visual", "image_question_shared"
    }
    if is_relay:
        prototype_memory = augmented_prompt["relay_embeddings"]
        relay_positions = augmented_prompt["relay_positions"]
        if not prototype_memory.requires_grad:
            raise RuntimeError("Visual relay embeddings are detached before current-policy forward")
    elif policy_kind == "adapter_only":
        prototype_memory = None
        relay_positions = None
        visual_tensors = [*visual.pooler_output,
                          *(tensor for level in visual.deepstack_features for tensor in level)]
        if not any(tensor.requires_grad for tensor in visual_tensors):
            raise RuntimeError("Adapter-only visual features are detached before current-policy forward")
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
