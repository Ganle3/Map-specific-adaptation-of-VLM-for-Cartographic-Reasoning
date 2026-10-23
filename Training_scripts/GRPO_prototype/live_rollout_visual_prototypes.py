"""Rollouts that append sampled, prototype-read global visual tokens.

This is deliberately distinct from crop selection: global merger tokens are
encoded once, then K prototypes attend over them. No crop is encoded here.
"""
from __future__ import annotations
import copy
import sys
from contextlib import contextmanager
from pathlib import Path
import torch
from PIL import Image

SPARSE = Path(__file__).resolve().parents[2] / "SparseVLM_eliminateCONTEXT"
if str(SPARSE) not in sys.path:
    sys.path.insert(0, str(SPARSE))
from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap, _segment_positions


def _prompt(question: str) -> str:
    return ("This is a cartographic reasoning question.\n\n"
            "Use only the supplied map image to answer the question. Carefully inspect the\n"
            "map legend, labels, colors, boundaries, spatial relationships, and other\n"
            "relevant visual information.\n\n"
            f"Question:\n{question}\n\n"
            "Provide your answer on a separate line using exactly this format:\n"
            "Final answer: <answer>")


def _visual_module(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model.visual


@contextmanager
def _cached_vision(visual, features):
    original_forward = visual.forward
    visual.forward = lambda *args, **kwargs: copy.copy(features)
    try:
        yield
    finally:
        visual.forward = original_forward


def prepare_global_visual_prefill(model, processor, item, *, global_max_pixels: int):
    """Run ViT+merger once; reuse its output for Qwen prefill and the policy."""
    device = model.get_input_embeddings().weight.device
    with Image.open(item.image_path) as file:
        original = file.convert("RGB")
    inputs = _content_inputs(processor, [_resize_to_pixel_cap(original, global_max_pixels)],
                             _prompt(item.question), device, global_max_pixels)
    visual = _visual_module(model)
    with torch.inference_mode():
        features = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"], return_dict=True)
    image_embeddings = features.pooler_output.detach()
    positions = _segment_positions(model, inputs, 0)
    with _cached_vision(visual, features), torch.inference_mode():
        output = model(**inputs, position_ids=positions, use_cache=True,
                       return_dict=True, output_hidden_states=True)
    return {"pending": output.logits[:, -1].argmax(dim=-1).detach().clone(),
            "past": output.past_key_values, "cursor": int(positions.max().item()) + 1,
            "hidden": output.hidden_states[-1][:, -1, :].detach().clone(),
            "image_embeddings": image_embeddings}


def _commit(model, token, past, cursor):
    mask = torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=token.device)
    positions = torch.full((3, 1, 1), cursor, dtype=torch.long, device=token.device)
    with torch.inference_mode():
        output = model(input_ids=token.view(1, 1), attention_mask=mask, position_ids=positions,
                       past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            cursor + 1, output.hidden_states[-1][:, -1, :].detach().clone())


def _append_visual_slots(model, weighted_slots, past, cursor):
    if weighted_slots.ndim == 2:
        weighted_slots = weighted_slots.unsqueeze(0)
    device, length = weighted_slots.device, weighted_slots.shape[1]
    mask = torch.ones((1, past.get_seq_length() + length), dtype=torch.long, device=device)
    positions = torch.arange(cursor, cursor + length, device=device).view(1, 1, length).expand(3, -1, -1)
    with torch.inference_mode():
        output = model(inputs_embeds=weighted_slots, attention_mask=mask, position_ids=positions,
                       past_key_values=past, use_cache=True, return_dict=True, output_hidden_states=True)
    return (output.logits[:, -1].argmax(dim=-1), output.past_key_values,
            cursor + length, output.hidden_states[-1][:, -1, :].detach().clone())


def rollout(model, processor, policy, item, *, rollout_id: int, max_new_tokens: int,
            look_every: int, max_looks: int, sample: bool, prefill=None):
    if prefill is None:
        prefill = prepare_global_visual_prefill(model, processor, item, global_max_pixels=1_000_000)
        prefill_reused = False
    else:
        prefill_reused = True
    pending, past = prefill["pending"].clone(), copy.deepcopy(prefill["past"])
    cursor, hidden, image_embeddings = int(prefill["cursor"]), prefill["hidden"].clone(), prefill["image_embeddings"]
    device = model.get_input_embeddings().weight.device
    eos = processor.tokenizer.eos_token_id
    eos_ids = set(map(int, eos)) if isinstance(eos, (list, tuple, set)) else {int(eos)}
    ids, actions, diagnostics, terminated = [], [], [], False
    for _ in range(max_new_tokens):
        token = int(pending.item())
        ids.append(token)
        if token in eos_ids:
            terminated = True
            break
        pending, past, cursor, hidden = _commit(model, pending, past, cursor)
        token_index = len(ids)
        if look_every > 0 and token_index % look_every == 0 and len(actions) < max_looks:
            slots, logits, concentration, gate, logp = policy(
                hidden.to(policy.prototypes.device).float(), image_embeddings.to(policy.prototypes.device).float(),
                sample=sample)
            # Scale by K: a uniform gate preserves normal merger-token scale.
            weighted_slots = (gate.unsqueeze(-1) * slots * policy.num_prototypes).detach().to(
                device=device, dtype=model.get_input_embeddings().weight.dtype)
            actions.append({"hidden": hidden.squeeze(0).cpu(), "gate": gate.squeeze(0).detach().cpu(),
                            "old_log_prob": float(logp.detach())})
            diagnostics.append({"question_id": item.sample_id, "rollout_id": rollout_id,
                                "reasoning_token_index": token_index,
                                "slot_logits": logits[0].detach().cpu().tolist(),
                                "gate_concentration": concentration[0].detach().cpu().tolist(),
                                "gate_mean": (concentration[0] / concentration[0].sum()).detach().cpu().tolist(),
                                "gate_weights": gate[0].detach().cpu().tolist(),
                                "selected_log_probability": float(logp.detach())})
            pending, past, cursor, hidden = _append_visual_slots(model, weighted_slots, past, cursor)
    return {"question_id": item.sample_id, "rollout_id": rollout_id,
            "response": processor.tokenizer.decode(ids, skip_special_tokens=True), "actions": actions,
            "diagnostics": diagnostics, "image_embeddings": image_embeddings,
            "visual_slot_append_count": len(actions), "terminated": terminated,
            "truncated": not terminated and len(ids) >= max_new_tokens,
            "global_prefill_reused": prefill_reused, "cache_length": int(past.get_seq_length()),
            "cache_policy": "continuous_global_reasoning_and_visual_slot_kv"}
