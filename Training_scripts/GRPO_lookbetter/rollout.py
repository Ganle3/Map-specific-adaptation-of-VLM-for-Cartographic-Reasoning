"""Frozen-Qwen token-boundary rollout for the trainable visual controller.

This is purposefully not a second visual pipeline.  It imports the established
anchor/replay/crop/bias helpers and changes only the point at which a crop turn
is requested and the subset/size of heuristic crop proposals.
"""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_SPARSE_ROOT = Path(__file__).resolve().parents[2] / "SparseVLM_eliminateCONTEXT"
if str(_SPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPARSE_ROOT))

from attention_policy import aggregate_decoder_layers, salient_regions
from continuous_cache_qwen3vl import (  # fixed decoder/KV components
    ContinuousConfig, GlobalVisualKeyBias,
    _content_inputs,
    _patch_level_global_bias, _resize_to_pixel_cap,
    _segment_positions,
)
from dynamic_attention import MultiImageAttentionCapture
from rolling_sparse_qwen3vl import PixelRegion, _image_geometries, _project_component

from data import MapQuestion, build_cartographic_prompt, evaluate_answer
from geometry import Proposal, jsd_features, legal_size_actions, region_features
from policy import WhenWhereController


@dataclass(frozen=True)
class RolloutConfig:
    max_reasoning_tokens: int = 320
    # At every successful LOOK, retain the visible trace verbatim but rebuild
    # the full-image anchor *and* assistant trace under the new ROI bias.
    # Reusing an ungated anchor cache is not equivalent: prompt/header KVs
    # retain an earlier visual routing decision (verified by the HOW test).
    how_refresh: str = "roi_biased_full_anchor_trace"
    # Optionally rewrite the newest visible reasoning suffix after LOOK. The
    # preserved prefix is teacher-forced from the immutable anchor under the
    # selected ROI bias; only this suffix is greedily regenerated.
    revision_tokens: int = 0
    eligible_after_tokens: int = 12
    eligible_stride: int = 4
    max_looks: int = 10
    cooldown_tokens: int = 20
    when_probe_stride: int = 4
    when_attention_source: str = "main"  # main | ungated_probe
    proposal_cap: int = 6
    attention_mass: float = 0.60
    aggregation: str = "mean"
    replay_tail_tokens: int = 600
    # LOOK uses the existing full-map visual KV.  Selected original-image
    # regions are mapped back to this merged visual-token grid and receive a
    # fixed positive attention-logit bias; no crop ViT pass or new chat turn.
    roi_key_bias: float = 2.079442  # ln(8), validated by ROI-KV experiment
    global_key_bias: float = -4.0
    global_retain_attention_mass: float = 0.40
    global_bias_mode: str = "off"  # off | look_snapshot
    global_max_pixels: int = 1_000_000
    scout_tokens: int = 36


def _aggregate(layers: np.ndarray, mode: str) -> np.ndarray:
    weights = np.linspace(1.0, 2.0, layers.shape[0]).tolist() if mode == "weighted_mean" else None
    return aggregate_decoder_layers(layers, mode, weights)


def _roi_key_bias(regions: list[PixelRegion], grid: tuple[int, int], original_size: tuple[int, int],
                  device: torch.device, bias: float) -> torch.Tensor:
    """Map original-image regions onto the immutable global visual-KV grid."""
    height, width = map(int, grid)
    original_width, original_height = original_size
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * original_height
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * original_width
    selected = torch.zeros((height, width), device=device, dtype=torch.bool)
    for region in regions:
        selected |= ((xx[None, :] >= region.left) & (xx[None, :] < region.right)
                     & (yy[:, None] >= region.top) & (yy[:, None] < region.bottom))
    if not selected.any():
        raise RuntimeError("Selected ROI contains no global visual-token centers; enlarge the region.")
    weights = torch.zeros(height * width, device=device, dtype=torch.float32)
    weights[selected.flatten()] = bias
    return weights


def _next_token(model, inputs, past, cursor: int, global_positions, tracked_positions, grids,
                global_biases, crop_key_bias: float, score_mode: str):
    """One frozen decoder forward; returns a pending next token and compact maps.

    Unlike ``generate(max_new_tokens=...)`` this exposes the actual individual
    reasoning-token boundary needed by the When controller.  It materializes
    only the selected query's visual scores through the existing eager hook.
    """
    if inputs is not None:
        local = inputs["input_ids"].shape[1]
        positions = _segment_positions(model, inputs, cursor)
        prior = 0 if past is None else past.get_seq_length()
        mask = torch.ones((1, prior + local), dtype=torch.long, device=inputs["input_ids"].device)
        kwargs = dict(inputs)
        kwargs.pop("attention_mask", None)
        kwargs.update(attention_mask=mask, position_ids=positions, past_key_values=past)
        next_cursor = int(positions.max()) + 1
    else:
        raise ValueError("A token input is required after prefill; use _token_inputs for continuation.")
    enabled = bool(global_biases.abs().max().item() > 0) or bool(crop_key_bias) or score_mode != "attention"
    crop_positions = torch.cat(tracked_positions[1:]) if len(tracked_positions) > 1 else None
    with MultiImageAttentionCapture(model, tracked_positions, grids, score_mode) as capture, \
            GlobalVisualKeyBias(global_positions, global_biases, enabled, crop_positions, crop_key_bias, capture), \
            torch.inference_mode():
        output = model(**kwargs, use_cache=True, return_dict=True)
    token = output.logits[:, -1].argmax(dim=-1)
    raw, _ = capture.summaries(1)
    return token, output.past_key_values, next_cursor, raw


def _continuation_inputs(token: torch.Tensor, past, cursor: int):
    return {"input_ids": token.view(1, 1), "attention_mask": torch.ones((1, past.get_seq_length() + 1), dtype=torch.long, device=token.device),
            "position_ids": torch.full((3, 1, 1), cursor, dtype=torch.long, device=token.device), "past_key_values": past}


def _continue_one(model, token, past, cursor, global_positions, tracked_positions, grids, biases, crop_bias, score_mode):
    """Continuation counterpart which avoids re-running multimodal prefill."""
    kwargs = _continuation_inputs(token, past, cursor)
    enabled = bool(biases.abs().max().item() > 0) or bool(crop_bias) or score_mode != "attention"
    crop_positions = torch.cat(tracked_positions[1:]) if len(tracked_positions) > 1 else None
    with MultiImageAttentionCapture(model, tracked_positions, grids, score_mode) as capture, \
            GlobalVisualKeyBias(global_positions, biases, enabled, crop_positions, crop_bias, capture), \
            torch.inference_mode():
        output = model(**kwargs, use_cache=True, return_dict=True)
    next_token = output.logits[:, -1].argmax(dim=-1)
    raw, _ = capture.summaries(1)
    return next_token, output.past_key_values, cursor + 1, raw


def _refresh_trace_from_anchor(model, anchor_inputs, trace_token_ids: list[int],
                               global_positions, visual_biases: torch.Tensor):
    """Rebuild the full anchor and visible assistant trace under an ROI bias.

    The token sequence is *not* decoded, sampled, changed, or presented as a
    new user/assistant turn.  Only its internal decoder K/V states are
    recomputed, which gives a mid-trajectory LOOK the same causal conditioning
    semantics as an ROI bias active during an assistant-prefix prefill.
    """
    if not trace_token_ids:
        raise RuntimeError("Cannot refresh an empty assistant trace.")
    # This must be a *biased* full anchor prefill, rather than a clone of the
    # ungated anchor cache.  Decoder K/V for prompt/header tokens are causally
    # affected by how those tokens attended to global visual keys.  A later
    # bias on assistant replay alone cannot recover those states.
    device = global_positions.device
    anchor_positions = _segment_positions(model, anchor_inputs, 0)
    with GlobalVisualKeyBias(global_positions, visual_biases, True), torch.inference_mode():
        anchor_output = model(**anchor_inputs, position_ids=anchor_positions,
                              use_cache=True, return_dict=True)
    refresh_past = anchor_output.past_key_values
    anchor_cursor = int(anchor_positions.max().item()) + 1
    tokens = torch.tensor([trace_token_ids], device=device, dtype=torch.long)
    length = tokens.shape[1]
    attention_mask = torch.ones(
        (1, refresh_past.get_seq_length() + length), device=device, dtype=torch.long
    )
    positions = torch.arange(anchor_cursor, anchor_cursor + length, device=device, dtype=torch.long)
    position_ids = positions.view(1, 1, length).expand(3, -1, -1)
    with GlobalVisualKeyBias(global_positions, visual_biases, True), torch.inference_mode():
        output = model(input_ids=tokens, attention_mask=attention_mask,
                       position_ids=position_ids, past_key_values=refresh_past,
                       use_cache=True, return_dict=True)
    # The final forced token is now in cache; logits predict the one pending
    # next token, exactly matching ordinary incremental generation semantics.
    pending = output.logits[:, -1].argmax(dim=-1)
    return pending, output.past_key_values, anchor_cursor + length


def _proposals(projected_layers, raw_layers, grid, original, cfg: RolloutConfig, visited):
    score_map, raw_map = _aggregate(projected_layers, cfg.aggregation), _aggregate(raw_layers, cfg.aggregation)
    components = salient_regions(score_map, cfg.proposal_cap, cfg.attention_mass, min_patches=1)
    regions = [Proposal(*asdict(_project_component(PixelRegion(0, 0, original.width, original.height, 0.0), component, grid)).values(), component)
               for component in components]
    features, priors = region_features(regions, score_map, raw_map, visited)
    return regions, features, priors, score_map


def _record(actions, row):
    actions.append(row)
    return float(row.get("old_log_probability", 0.0))


def _is_terminal_token(model, token: torch.Tensor) -> bool:
    """Whether greedy decoding has reached Qwen's end-of-turn/end-of-sequence token."""
    eos = model.generation_config.eos_token_id
    eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}
    return int(token.item()) in eos_ids


def rollout(model, processor, item: MapQuestion, controller: WhenWhereController, cfg: RolloutConfig) -> dict:
    """Run one stochastic visual-action trajectory with frozen greedy language.

    All controller actions are serialized as feature/action records so the GRPO
    update can recompute their exact log probability later without retaining a
    Qwen computation graph.
    """
    device = model.get_input_embeddings().weight.device
    original = Image.open(item.image_path).convert("RGB")
    global_image = _resize_to_pixel_cap(original, cfg.global_max_pixels)
    # Match the native GRPO inference contract.  This asks Qwen itself to
    # choose its Thinking transition and emit a concise answer channel; no
    # later user turn or forced token sequence is inserted by this pipeline.
    prompt = build_cartographic_prompt(item.question)
    initial = _content_inputs(processor, [global_image], prompt, device, cfg.global_max_pixels)
    global_local, global_grids = _image_geometries(model, initial)
    global_positions, global_grid = global_local[0], global_grids[0]
    zero_gate = torch.zeros(global_positions.numel(), dtype=torch.float32, device=device)
    # Retain an immutable full-image + question anchor. At LOOK we replay the
    # same visible assistant tokens from this anchor under an ROI bias; no crop
    # image is encoded and no new chat turn is inserted.
    pending, past, cursor, maps = _next_token(model, initial, None, 0, global_positions, global_local, global_grids,
                                               zero_gate, 0.0, "attention")
    # ``initial`` remains immutable.  A successful LOOK re-prefills it under
    # the selected ROI bias before replaying the visible trace.
    tracked, grids = global_local, global_grids
    trace_token_ids = [] if _is_terminal_token(model, pending) else [int(pending.item())]
    text = processor.tokenizer.decode(trace_token_ids, skip_special_tokens=True)
    trace, actions, look_events, visited = text, [], [], []
    old_logp, previous_attention, cooldown = 0.0, None, 0
    look_attempts, probe_count, probe_seconds = 0, 0, 0.0
    latest_main_attention, latest_main_layers = None, None
    # This vector is aligned with immutable global visual-token positions.
    # It is replaced at each successful LOOK, exactly like the former
    # current-crop state, but it never changes the KV cache itself.
    active_visual_bias = zero_gate
    reasoning_ended = _is_terminal_token(model, pending)

    token_index = 1
    while token_index < cfg.max_reasoning_tokens:
        if reasoning_ended:
            break
        eligible = token_index >= cfg.eligible_after_tokens and token_index % cfg.eligible_stride == 0
        forced = cooldown > 0 or look_attempts >= cfg.max_looks
        diagnostic_attention, diagnostic_layers = None, None
        if eligible and not forced:
            if cfg.when_attention_source == "ungated_probe" and token_index % cfg.when_probe_stride == 0:
                # Disposable uncompressed diagnostic. It costs one additional
                # decoder forward but neither vision encoding nor persistent KV.
                began = time.perf_counter()
                _probe_token, _probe_past, _probe_cursor, raw_probe_maps = _continue_one(
                    model, pending, copy.deepcopy(past), cursor, global_positions, tracked, grids,
                    zero_gate, 0.0, "attention")
                probe_seconds += time.perf_counter() - began
                probe_count += 1
                diagnostic_layers = raw_probe_maps[0]
                diagnostic_attention = _aggregate(diagnostic_layers, cfg.aggregation)
            elif cfg.when_attention_source == "main" and latest_main_attention is not None:
                # Fast mode: with global-bias-mode=off, the normal forward is
                # already ungated and supplies the exact needed global map.
                diagnostic_layers = latest_main_layers
                diagnostic_attention = latest_main_attention
        if diagnostic_attention is not None and eligible and not forced:
            features = jsd_features(diagnostic_attention, previous_attention, look_attempts, cfg.max_looks)
            want_look, logp = controller.sample_when(torch.from_numpy(features).to(device))
            old_logp += _record(actions, {"kind": "when", "features": features.tolist(), "action": float(want_look), "old_log_probability": logp})
            if want_look:
                look_attempts += 1
                # The expensive projected-value probe is invoked only after a
                # sampled LOOK. It runs on a cloned current cache and is then
                # discarded, leaving the How trajectory untouched.
                _probe_token, _probe_past, _probe_cursor, probe_maps = _continue_one(
                    model, pending, copy.deepcopy(past), cursor, global_positions, tracked, grids,
                    zero_gate, 0.0, "projected_value_norm")
                candidates, where_features, heuristic_logits, _ = _proposals(
                    probe_maps[0], diagnostic_layers, global_grid, original, cfg, visited)
                if candidates:
                    feature_tensor = torch.from_numpy(where_features).to(device)
                    prior_tensor = torch.from_numpy(heuristic_logits).to(device)
                    keep, keep_logp = controller.sample_keep(feature_tensor, prior_tensor)
                    old_logp += _record(actions, {"kind": "keep", "features": where_features.tolist(),
                                                   "heuristic_logits": heuristic_logits.tolist(), "action": keep.float().cpu().tolist(),
                                                   "old_log_probability": float(keep_logp.sum().item())})
                    selected = []
                    for proposal, feature, selected_keep in zip(candidates, where_features, keep.cpu().tolist()):
                        if not selected_keep:
                            continue
                        legal = legal_size_actions(proposal, global_grid, (original.width, original.height))
                        ratios = np.asarray([ratio for _, ratio in legal], dtype=np.float32)
                        choice, size_logp = controller.sample_size(torch.from_numpy(feature).to(device), torch.from_numpy(ratios).to(device))
                        old_logp += _record(actions, {"kind": "size", "features": feature.tolist(), "log_ratios": ratios.tolist(),
                                                       "action": choice, "old_log_probability": size_logp})
                        left, top, right, bottom = legal[choice][0]
                        selected.append(PixelRegion(left, top, right, bottom, proposal.score))
                    if selected:
                        # Snapshot is computed from the same *ungated* global
                        # attention that caused LOOK. It is combined with the
                        # ROI bias only on normal reasoning forwards.
                        global_gate = (_patch_level_global_bias(diagnostic_layers, ContinuousConfig(
                            aggregation=cfg.aggregation, global_key_bias=cfg.global_key_bias,
                            global_retain_attention_mass=cfg.global_retain_attention_mass), global_positions.numel())[0]
                                       if cfg.global_bias_mode == "look_snapshot" and cfg.global_key_bias else zero_gate)
                        roi_bias = _roi_key_bias(selected, global_grid, original.size, device, cfg.roi_key_bias)
                        if roi_bias.numel() != global_positions.numel():
                            raise RuntimeError("ROI grid/token mismatch: global visual positions do not match image_grid_thw.")
                        # Suppression remains an optional independent setting;
                        # a selected ROI must always win over any negative
                        # global gate rather than receiving a net penalty.
                        active_visual_bias = torch.maximum(global_gate, roi_bias)
                        tracked, grids = [global_positions], [global_grid]
                        # Discard the old working cache before the ROI-biased
                        # anchor re-prefill. The emitted trace stays exactly
                        # unchanged; only its hidden/KV states change.
                        del past
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        revised_count = min(cfg.revision_tokens, max(0, len(trace_token_ids) - 1))
                        replay_ids = trace_token_ids[:-revised_count] if revised_count else trace_token_ids
                        pending, past, cursor = _refresh_trace_from_anchor(
                            model, initial, replay_ids,
                            global_positions, active_visual_bias,
                        )
                        replayed_tokens = len(replay_ids)
                        regenerated = 0
                        if revised_count:
                            # Keep the textual prefix verbatim, then let the
                            # new ROI-conditioned cache replace only its recent
                            # suffix. The final ``pending`` remains outside KV,
                            # preserving normal incremental decoding invariants.
                            trace_token_ids = list(replay_ids)
                            while regenerated < revised_count:
                                if _is_terminal_token(model, pending):
                                    reasoning_ended = True
                                    break
                                trace_token_ids.append(int(pending.item()))
                                regenerated += 1
                                if regenerated < revised_count:
                                    pending, past, cursor, _roi_maps = _continue_one(
                                        model, pending, past, cursor, global_positions, tracked, grids,
                                        active_visual_bias, 0.0, "attention")
                            trace = processor.tokenizer.decode(trace_token_ids, skip_special_tokens=True)
                        else:
                            if _is_terminal_token(model, pending):
                                reasoning_ended = True
                            else:
                                trace_token_ids.append(int(pending.item()))
                                trace = processor.tokenizer.decode(trace_token_ids, skip_special_tokens=True)
                        look_events.append({"token": token_index, "regions": [asdict(r) for r in selected],
                                            "roi_visual_tokens": int((roi_bias > 0).sum().item()),
                                            # Auditable mapping from original-image ROI to the
                                            # immutable global visual KV.  Local indices are in
                                            # the merged HxW grid; cache positions are the actual
                                            # decoder key indices patched by GlobalVisualKeyBias.
                                            "roi_visual_token_indices": torch.nonzero(roi_bias > 0, as_tuple=False).flatten().cpu().tolist(),
                                            "roi_cache_key_positions": global_positions[roi_bias > 0].detach().cpu().tolist(),
                                            "roi_key_bias": cfg.roi_key_bias,
                                            "how_refresh": cfg.how_refresh,
                                            "teacher_forced_replay_tokens": replayed_tokens,
                                            "revision_tokens": revised_count,
                                            "regenerated_tokens": regenerated})
                        visited.extend(selected)
                        previous_attention, latest_main_attention, latest_main_layers = None, None, None
                        cooldown = cfg.cooldown_tokens
                        if not revised_count:
                            token_index += 1
                        continue
            previous_attention = diagnostic_attention
        active_gate = active_visual_bias
        pending, past, cursor, _maps = _continue_one(model, pending, past, cursor, global_positions, tracked, grids,
                                                      active_gate, 0.0, "attention")
        latest_main_layers = _maps[0]
        latest_main_attention = _aggregate(latest_main_layers, cfg.aggregation)
        if _is_terminal_token(model, pending):
            reasoning_ended = True
        else:
            trace_token_ids.append(int(pending.item()))
            trace = processor.tokenizer.decode(trace_token_ids, skip_special_tokens=True)
        cooldown = max(0, cooldown - 1)
        token_index += 1

    # Do not manufacture a Thinking-channel transition.  Qwen naturally
    # chooses whether/when to emit </think>, its answer text, and EOS.  The
    # configured budget is therefore a hard generation cap only.
    evaluation = evaluate_answer(trace, item)
    return {"sample_id": item.sample_id, "question": item.question, "answer": item.answer, "response": trace,
            "reward": evaluation["reward"], "evaluation": evaluation, "actions": actions,
            "old_log_probability": old_logp, "look_events": look_events,
            "look_attempt_count": look_attempts, "look_count": len(look_events),
            "selected_crop_count": sum(len(row["regions"]) for row in look_events),
            "when_probe_count": probe_count, "when_probe_seconds": probe_seconds,
            "generation_terminated_by_eos": reasoning_ended,
            "generation_budget_tokens": cfg.max_reasoning_tokens,
            "effective_attention_modulation": {"global_bias_mode": cfg.global_bias_mode,
                "global_key_bias": cfg.global_key_bias if cfg.global_bias_mode != "off" else 0.0,
                "global_retain_attention_mass": cfg.global_retain_attention_mass,
                "roi_key_bias": cfg.roi_key_bias,
                "visual_update": "existing_global_visual_KV_roi_logit_bias"}}
