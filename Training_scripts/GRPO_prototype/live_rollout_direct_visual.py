"""Nine-grid rollouts that append cached crop merger tokens without a chat turn.

This is intentionally separate from :mod:`live_rollout`, whose append method
opens a new user/assistant turn.  Here the crop is an ordinary causal segment
at the current tail of the decoder cache, so the next generated token remains
part of the assistant response already in progress.
"""
from __future__ import annotations

import torch
from PIL import Image

from live_rollout import (
    _cached_vision, _commit, _decision_hidden, _prompt, _resize_to_pixel_cap,
    _visual_module, prepare_crops,
)
from continuous_cache_qwen3vl import _content_inputs, _segment_positions


def _visual_only_inputs(model, crop_inputs: dict) -> dict:
    """Keep exactly one native Qwen image placeholder span, with no chat text.

    A raw ``pooler_output`` does not carry Qwen's image-grid M-RoPE layout.
    This keeps the corresponding image placeholder tokens and their metadata
    so ``get_rope_index`` can reconstruct that layout at the new cache tail.
    """
    image_mask = crop_inputs["input_ids"][0].eq(int(model.config.image_token_id))
    if not bool(image_mask.any()):
        raise RuntimeError("Crop processor inputs contain no Qwen image placeholders.")
    result = {key: value for key, value in crop_inputs.items()
              if key in ("pixel_values", "image_grid_thw")}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if key in crop_inputs:
            result[key] = crop_inputs[key][:, image_mask]
    if "mm_token_type_ids" not in result:
        result["mm_token_type_ids"] = result["input_ids"].new_ones(result["input_ids"].shape)
    return result


def append_prepared_native_visual_segment(model, prepared_entry, past, cursor: int, device,
                                          policy_hidden_layer: int = -1):
    """Append a cached crop as native Qwen visual tokens without a chat turn."""
    _, crop_inputs, features = prepared_entry
    inputs = _visual_only_inputs(model, crop_inputs)
    position_ids = _segment_positions(model, inputs, cursor)
    length = inputs["input_ids"].shape[1]
    attention_mask = torch.ones(
        (1, past.get_seq_length() + length), dtype=torch.long, device=device
    )
    kwargs = dict(inputs)
    kwargs.pop("attention_mask", None)
    with torch.inference_mode():
        with _cached_vision(_visual_module(model), features):
            output = model(**kwargs, attention_mask=attention_mask,
                           position_ids=position_ids, past_key_values=past,
                           use_cache=True, return_dict=True,
                           output_hidden_states=True)
    return (
        output.logits[:, -1].argmax(dim=-1), output.past_key_values,
        int(position_ids.max().item()) + 1, _decision_hidden(output, policy_hidden_layer),
    )


def rollout(model, processor, policy, item, prepared, *, rollout_id: int,
            global_max_pixels: int, max_new_tokens: int, look_every: int,
            max_looks: int, sample: bool, policy_hidden_layer: int = -1):
    """Generate one greedy language trajectory with sampled nine-grid actions."""
    device = model.get_input_embeddings().weight.device
    original = Image.open(item.image_path).convert("RGB")
    inputs = _content_inputs(
        processor, [_resize_to_pixel_cap(original, global_max_pixels)],
        _prompt(item.question), device, global_max_pixels,
    )
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True,
                       return_dict=True, output_hidden_states=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(positions.max().item()) + 1
    hidden = _decision_hidden(output, policy_hidden_layer)
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    ids, actions, diagnostics = [], [], []
    terminated = False
    for _ in range(max_new_tokens):
        token = int(pending.item())
        ids.append(token)
        if token in eos_ids:
            terminated = True
            break
        pending, past, cursor, hidden = _commit(model, pending, past, cursor, policy_hidden_layer)
        token_index = len(ids)
        if look_every > 0 and token_index % look_every == 0 and len(actions) < max_looks:
            logits, probs, selected, logp = policy(
                hidden.to(policy.nine_blocks.device).float(), sample=sample
            )
            action = int(selected.item())
            box = prepared[action][0]
            actions.append({"hidden": hidden.squeeze(0).cpu(), "action": action,
                            "old_log_prob": float(logp.detach()), "selected_log_prob": logp})
            diagnostics.append({"question_id": item.sample_id, "rollout_id": rollout_id,
                                "reasoning_token_index": token_index,
                                "logits": logits[0].detach().cpu().tolist(),
                                "probabilities": probs[0].detach().cpu().tolist(),
                                "selected_action": action, "crop_box": box,
                                "selected_log_probability": float(logp.detach()),
                                "append_mode": "direct_merger_embeddings"})
            pending, past, cursor, hidden = append_prepared_native_visual_segment(
                model, prepared[action], past, cursor, device, policy_hidden_layer
            )
    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    return {"question_id": item.sample_id, "rollout_id": rollout_id,
            "response": response, "actions": actions, "diagnostics": diagnostics,
            "crop_append_count": len(actions), "terminated": terminated,
            "truncated": not terminated and len(ids) >= max_new_tokens,
            "cache_length": int(past.get_seq_length()),
            "cache_policy": "continuous_global_reasoning_and_direct_crop_merger_kv"}
