"""Cache-preserving, token-by-token Qwen3-VL rollouts with dynamic ROIs."""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch
from .attention_integration import DynamicVisualKeyBias
from .roi_policy import DynamicROIPolicy
from .spatial_mask import SoftROIMask, VisualTokenGrid
from .attention_prior import attention_guided_prior


@dataclass(frozen=True)
class RolloutConfig:
    max_new_tokens: int = 256
    do_sample_text: bool = False
    temperature: float = 1.0
    eos_token_id: int | None = None
    roi_prior_mode: str = "attention_guided"  # fixed_grid | attention_guided
    attention_prior_alpha: float = .5
    roi_residual_enabled: bool = True


def _pool_3x3(attention: torch.Tensor, grid: VisualTokenGrid) -> torch.Tensor:
    """Area-average arbitrary HxW visual attention into nine policy features."""
    image = attention.reshape(1, 1, grid.height, grid.width)
    return torch.nn.functional.adaptive_avg_pool2d(image, (3, 3)).flatten()


def _is_terminal_token(model, token: torch.Tensor) -> bool:
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple, set)) else {eos}
    return int(token.item()) in {int(value) for value in eos_ids}


def _freeze(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _prompt_rope_positions(model, inputs: dict) -> Tensor:
    """Use Qwen's own M-RoPE indexing; cache length is not a valid substitute.

    Visual placeholders advance the KV sequence but have two-dimensional image
    coordinates in Qwen3-VL.  Continuation text positions must therefore start
    after the model-calculated final 3D position, not after ``prompt_length``.
    """
    backbone = model.get_base_model() if hasattr(model, "get_base_model") else model
    for _ in range(4):
        if hasattr(backbone, "get_rope_index"):
            break
        backbone = getattr(backbone, "model", None)
        if backbone is None:
            raise AttributeError("Could not locate Qwen3VLModel.get_rope_index.")
    positions, _ = backbone.get_rope_index(
        inputs["input_ids"], inputs["mm_token_type_ids"], image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
    )
    return positions


def assert_frozen_backbone(model, policy: DynamicROIPolicy) -> None:
    """Fail early if a Qwen/LoRA parameter accidentally enters optimisation."""
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("The Qwen backbone or reasoning LoRA is not frozen.")
    if not any(parameter.requires_grad for parameter in policy.parameters()):
        raise AssertionError("ROI policy has no trainable parameters.")


@torch.inference_mode()
def collect_rollout(model, processor, inputs, policy: DynamicROIPolicy, mask: SoftROIMask,
                    reward_fn, cfg: RolloutConfig, *, deterministic_roi: bool = False) -> dict:
    """Generate one rollout without a decoder backward graph.

    Temporal alignment: prompt's last hidden state predicts the first language
    token with no ROI bias.  It then selects the ROI used while consuming that
    token, hence influencing the *next* token.  Subsequent states and visual
    summaries are from the preceding, already-completed decoder step.
    """
    _freeze(model)
    assert_frozen_backbone(model, policy)
    device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    image_id = model.config.image_token_id
    positions = (inputs["input_ids"][0] == image_id).nonzero().flatten()
    if positions.numel() == 0:
        raise ValueError("ROI rollout needs exactly one prompt image.")
    merge = model.config.vision_config.spatial_merge_size
    _t, raw_h, raw_w = inputs["image_grid_thw"][0].tolist()
    grid = VisualTokenGrid(raw_h // merge, raw_w // merge)
    if positions.numel() != grid.height * grid.width:
        raise RuntimeError("Image placeholder count and merged visual grid disagree.")
    prompt_length = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    # The prompt run encodes vision once. All subsequent calls carry its KV cache.
    with DynamicVisualKeyBias(model, positions) as hook:
        hook.set_step_bias(None)
        prompt_positions = _prompt_rope_positions(model, inputs)
        first_inputs = dict(inputs)
        first_inputs["position_ids"] = prompt_positions
        first = model(**first_inputs, use_cache=True, output_hidden_states=True, output_attentions=True, return_dict=True)
        past, hidden = first.past_key_values, first.hidden_states[-1][:, -1].detach()
        token = first.logits[:, -1].argmax(-1)
        generated, actions = [int(token.item())], []
        previous_summary = torch.zeros((1, 9), device=device, dtype=hidden.dtype)
        full_previous_attention = None
        ended_with_eos = _is_terminal_token(model, token)
        for step in range(cfg.max_new_tokens - 1):
            if ended_with_eos:
                break
            if cfg.roi_prior_mode == "fixed_grid":
                prior = policy.base_geometry
            elif cfg.roi_prior_mode == "attention_guided":
                # Previous summary is only 3x3; retain full previous global-key
                # attention only for the immediately following prior calculation.
                prior = attention_guided_prior(full_previous_attention, grid, roi_count=policy.cfg.roi_count,
                                               alpha=cfg.attention_prior_alpha)
            else:
                raise ValueError("roi_prior_mode must be fixed_grid or attention_guided.")
            sampled = policy.sample(hidden.float(), previous_summary.float(), prior_geometry=prior,
                                    deterministic=deterministic_roi or not cfg.roi_residual_enabled)
            geometry = sampled["geometry"]
            bias = mask.bias(geometry).reshape(-1)
            hook.set_step_bias(bias)
            cache_length = prompt_length + len(generated)
            # Qwen's text M-RoPE advances identically in each axis after image
            # placeholders; this is equivalent to its generation preparation.
            position = prompt_positions[:, :, -1:] + len(generated)
            attention_mask = torch.ones((1, cache_length), device=device, dtype=torch.long)
            output = model(input_ids=token[:, None], attention_mask=attention_mask, position_ids=position,
                           past_key_values=past, cache_position=torch.tensor([cache_length - 1], device=device),
                           use_cache=True, output_hidden_states=True, output_attentions=True, return_dict=True)
            summary = hook.previous_attention_summary()
            if summary is None:
                raise RuntimeError("ROI attention hook was not executed.")
            past, hidden = output.past_key_values, output.hidden_states[-1][:, -1].detach()
            full_previous_attention = summary
            previous_summary = _pool_3x3(summary, grid).unsqueeze(0)
            actions.append({"features": sampled["features"].cpu(), "latent": sampled["latent"].cpu(),
                            "prior_geometry": prior.cpu(), "geometry": geometry.cpu(),
                            "strengths": geometry[..., 4].cpu(), "attention_summary_after": previous_summary.cpu(),
                            "log_prob": float(sampled["log_prob"].item()), "bias_mean": float(bias.mean()),
                            "bias_max": float(bias.max())})
            logits = output.logits[:, -1] / cfg.temperature
            token = torch.multinomial(logits.softmax(-1), 1).squeeze(-1) if cfg.do_sample_text else logits.argmax(-1)
            generated.append(int(token.item()))
            ended_with_eos = _is_terminal_token(model, token)

        # Do not force </think>, Final answer:, or a second assistant turn.
        # They are part of Qwen3-VL-Thinking's native response distribution;
        # ROI experimentation must change only visual attention, not language
        # generation semantics. max_new_tokens remains a safety cap only.
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    reward = float(reward_fn(text))
    return {"text": text, "reward": reward, "actions": actions,
            "old_log_probability": sum(x["log_prob"] for x in actions), "tokens": generated,
            "elapsed_seconds": time.perf_counter() - started, "ended_with_eos": ended_with_eos}
