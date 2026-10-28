"""Insert cross-attended prototype memory once during Qwen3-VL prefill."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
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


def _prototype_block(model, source_inputs: dict, count: int):
    config = qwen_lm(model).config
    image_positions = source_inputs["input_ids"][0].eq(int(config.image_token_id)).nonzero().flatten()
    if not len(image_positions):
        raise RuntimeError("Original processor input contains no image placeholder tokens")
    image_type = int(source_inputs["mm_token_type_ids"][0, int(image_positions[0])])
    ids = source_inputs["input_ids"].new_tensor(
        [int(config.vision_start_token_id)] + [int(config.image_token_id)] * count
        + [int(config.vision_end_token_id)]
    )
    mm_types = source_inputs["mm_token_type_ids"].new_tensor([0] + [image_type] * count + [0])
    return ids, mm_types


def _insertion_index(model, tokenizer, source_inputs: dict, placement: str) -> int:
    ids = source_inputs["input_ids"][0]
    config = qwen_lm(model).config
    if placement == "after_image":
        image_positions = ids.eq(int(config.image_token_id)).nonzero().flatten()
        index = int(image_positions[-1]) + 1
        if index < ids.numel() and int(ids[index]) == int(config.vision_end_token_id):
            index += 1
        return index
    if placement == "after_question":
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(im_end) != 1:
            raise RuntimeError("Unexpected Qwen <|im_end|> tokenization")
        positions = ids.eq(int(im_end[0])).nonzero().flatten()
        if not len(positions):
            raise RuntimeError("Cannot locate the user-turn end before the assistant generation prompt")
        return int(positions[-1])
    raise ValueError("placement must be 'after_image' or 'after_question'")


def _splice_1d(source: torch.Tensor, addition: torch.Tensor, index: int) -> torch.Tensor:
    return torch.cat((source[:index], addition, source[index:]), dim=0)


def build_prototype_prefill(model, tokenizer, source_inputs: dict, global_features,
                            policy, placement: str, *, with_grad: bool,
                            question_tokens: torch.Tensor | None = None):
    """Build one augmented multimodal prompt and differentiable prototype memory.

    ``after_image``: image -> prototype visual block -> question -> assistant.
    ``after_question``: image -> question -> prototype visual block -> assistant.
    """
    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        global_tokens = global_features.pooler_output[0].to(policy.prototypes.device)
        if question_tokens is None:
            memory, attention = policy(global_tokens)
        else:
            memory, attention = policy(global_tokens, question_tokens.to(policy.prototypes.device))
    count = int(memory.shape[0])
    side = math.isqrt(count)
    if side * side != count:
        raise ValueError("Native visual placement currently requires a square prototype count (9, 16, 36, ...)")
    block_ids, block_types = _prototype_block(model, source_inputs, count)
    index = _insertion_index(model, tokenizer, source_inputs, placement)
    input_ids = _splice_1d(source_inputs["input_ids"][0], block_ids, index).unsqueeze(0)
    mm_types = _splice_1d(source_inputs["mm_token_type_ids"][0], block_types, index).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    merge = int(qwen_lm(model).config.vision_config.spatial_merge_size)
    prototype_grid = source_inputs["image_grid_thw"].new_tensor([[1, side * merge, side * merge]])
    image_grid_thw = torch.cat((source_inputs["image_grid_thw"], prototype_grid), dim=0)

    # Original DeepStack features remain intact.  The prototype block has no
    # separate ViT DeepStack outputs, so zeros isolate the new main-merger
    # memory while preserving Qwen's expected tensor alignment.
    def image_tuple(value):
        return (value,) if isinstance(value, torch.Tensor) else tuple(value)

    combined = BaseModelOutputWithDeepstackFeatures(
        pooler_output=tuple(global_features.pooler_output) + (memory,),
        deepstack_features=[image_tuple(level) + (torch.zeros_like(memory),)
                            for level in global_features.deepstack_features],
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_types,
        "image_grid_thw": image_grid_thw,
        "mm_encoder_outputs": {"image": combined},
    }, attention


def encode_question_context(model, tokenizer, source_inputs: dict, global_features,
                            question: str) -> torch.Tensor:
    """Cache frozen final-layer states for exactly the question text tokens."""
    prompt_ids = source_inputs["input_ids"][0].tolist()

    def occurrences(pattern: list[int]) -> list[int]:
        return [index for index in range(len(prompt_ids) - len(pattern) + 1)
                if prompt_ids[index:index + len(pattern)] == pattern]

    # Locate the span by stable prompt labels rather than by tokenizing the
    # question in isolation. Byte-level/BPE tokenization of the question's
    # first token can differ when it follows a newline in the full prompt.
    left_pattern = tokenizer.encode("Question:\n", add_special_tokens=False)
    right_pattern = tokenizer.encode("Provide your answer", add_special_tokens=False)
    left = occurrences(left_pattern)
    right = occurrences(right_pattern)
    if len(left) != 1 or len(right) != 1:
        raise RuntimeError(f"Cannot uniquely locate question boundaries: left={left}, right={right}")
    start = left[0] + len(left_pattern)
    end = right[0]
    # Drop whitespace-only boundary tokens while retaining the exact question
    # content. This also verifies that the located text is the requested one.
    while start < end and not tokenizer.decode([prompt_ids[start]], skip_special_tokens=True).strip():
        start += 1
    while end > start and not tokenizer.decode([prompt_ids[end - 1]], skip_special_tokens=True).strip():
        end -= 1
    if start >= end or question.casefold().strip() not in tokenizer.decode(
            prompt_ids[start:end], skip_special_tokens=True).casefold():
        raise RuntimeError("Located prompt span does not contain the requested question")
    kwargs = {key: value for key, value in source_inputs.items()
              if key not in ("pixel_values", "attention_mask")}
    with torch.no_grad():
        output = model(
            **kwargs,
            attention_mask=source_inputs["attention_mask"],
            mm_encoder_outputs={"image": global_features},
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
    return output.hidden_states[-1][0, start:end].detach().clone()
