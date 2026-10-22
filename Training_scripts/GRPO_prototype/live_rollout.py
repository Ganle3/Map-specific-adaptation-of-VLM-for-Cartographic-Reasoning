"""Live nine-grid crop rollouts on the existing Qwen3-VL cache path."""
from __future__ import annotations

import sys
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
from PIL import Image

SPARSE = Path(__file__).resolve().parents[2] / "SparseVLM_eliminateCONTEXT"
if str(SPARSE) not in sys.path:
    sys.path.insert(0, str(SPARSE))
from continuous_cache_qwen3vl import _append_turn_inputs, _content_inputs, _resize_to_pixel_cap, _segment_positions
from region_policy import nine_grid_boxes


def _prompt(question: str) -> str:
    """Training prompt copied from the interactive script, without GUI imports."""
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


@contextmanager
def _cached_vision(visual, features):
    original_forward = visual.forward
    # get_image_features splits and reassigns fields on the returned object.
    # Give each append its own shallow wrapper around the immutable tensors.
    visual.forward = lambda *args, **kwargs: copy.copy(features)
    try:
        yield
    finally:
        visual.forward = original_forward


def _visual_module(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.visual


def prepare_crops(model, processor, original: Image.Image, device, max_pixels: int, instruction: str):
    """Encode all nine crops once per question, preserving native ViT and merger outputs."""
    visual = _visual_module(model)
    prepared = []
    for box in nine_grid_boxes(*original.size):
        crop = _resize_to_pixel_cap(original.crop(box), max_pixels)
        inputs = _append_turn_inputs(processor, [crop], instruction, device, max_pixels=max_pixels)
        with torch.inference_mode():
            features = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"], return_dict=True)
        prepared.append((box, inputs, features))
    return prepared


def _append_prepared(model, prepared, past, cursor, device):
    box, inputs, features = prepared
    positions = _segment_positions(model, inputs, cursor)
    full_mask = torch.ones((1, past.get_seq_length() + inputs["input_ids"].shape[1]),
                           dtype=torch.long, device=device)
    kwargs = dict(inputs)
    kwargs.pop("attention_mask", None)
    with _cached_vision(_visual_module(model), features), torch.inference_mode():
        output = model(**kwargs, attention_mask=full_mask, position_ids=positions,
                       past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            int(positions.max().item()) + 1, output.hidden_states[-1][:, -1, :].detach().clone())


def _commit(model, token, past, cursor):
    mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=token.device)
    positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=token.device)
    with torch.inference_mode():
        output = model(input_ids=token.view(1, 1), attention_mask=mask, position_ids=positions,
                       past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            cursor + 1, output.hidden_states[-1][:, -1, :].detach().clone())


def rollout(model, processor, policy, item, prepared, *, rollout_id: int, global_max_pixels: int,
            max_new_tokens: int, look_every: int, max_looks: int, sample: bool):
    device = model.get_input_embeddings().weight.device
    original = Image.open(item.image_path).convert("RGB")
    inputs = _content_inputs(processor, [_resize_to_pixel_cap(original, global_max_pixels)],
                             _prompt(item.question), device, global_max_pixels)
    positions = _segment_positions(model, inputs, 0)
    with torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True,
                       return_dict=True, output_hidden_states=True)
    pending, past = output.logits[:, -1].argmax(dim=-1), output.past_key_values
    cursor = int(positions.max().item()) + 1
    hidden = output.hidden_states[-1][:, -1, :].detach().clone()
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
        pending, past, cursor, hidden = _commit(model, pending, past, cursor)
        token_index = len(ids)
        if look_every > 0 and token_index % look_every == 0 and len(actions) < max_looks:
            logits, probs, selected, logp = policy(hidden.to(policy.nine_blocks.device).float(), sample=sample)
            action = int(selected.item())
            box = prepared[action][0]
            actions.append({"hidden": hidden.squeeze(0).cpu(), "action": action,
                            "old_log_prob": float(logp.detach()), "selected_log_prob": logp})
            diagnostics.append({"question_id": item.sample_id, "rollout_id": rollout_id,
                                "reasoning_token_index": token_index,
                                "logits": logits[0].detach().cpu().tolist(),
                                "probabilities": probs[0].detach().cpu().tolist(),
                                "selected_action": action, "crop_box": box,
                                "selected_log_probability": float(logp.detach())})
            pending, past, cursor, hidden = _append_prepared(model, prepared[action], past, cursor, device)
    response = processor.tokenizer.decode(ids, skip_special_tokens=True)
    return {"question_id": item.sample_id, "rollout_id": rollout_id,
            "response": response, "actions": actions, "diagnostics": diagnostics,
            "crop_append_count": len(actions),
            "terminated": terminated,
            "truncated": not terminated and len(ids) >= max_new_tokens,
            "cache_length": int(past.get_seq_length()),
            "cache_policy": "continuous_global_reasoning_and_all_crop_kv"}
