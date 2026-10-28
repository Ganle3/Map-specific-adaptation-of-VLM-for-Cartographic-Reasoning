"""Language-token PPO/GRPO loss whose only trainable source is prototype memory."""
from __future__ import annotations

import torch


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
    prototype_memory = visual.pooler_output[-1]
    if not prototype_memory.requires_grad:
        raise RuntimeError(
            "Prototype visual memory is detached before the Qwen current-policy forward"
        )
    # Standard LoRA modules are already injected into the underlying Qwen
    # layers. Calling that model directly avoids version-specific PEFT wrapper
    # inference behaviour while keeping the frozen adapter active.
    forward_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.enable_grad():
        output = forward_model(
            input_ids=full_ids,
            attention_mask=full_mask,
            mm_token_type_ids=mm_types,
            image_grid_thw=augmented_prompt["image_grid_thw"],
            mm_encoder_outputs=augmented_prompt["mm_encoder_outputs"],
            use_cache=False,
            return_dict=True,
        )
    if not output.logits.requires_grad:
        raise RuntimeError(
            "Qwen current-policy logits are detached from trainable prototype memory"
        )
    prefix_length = prefix_ids.shape[1]
    # Logit at prefix_length-1 predicts completion token 0.
    logits = output.logits[:, prefix_length - 1:prefix_length + completion_ids.shape[1] - 1, :].float()
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
