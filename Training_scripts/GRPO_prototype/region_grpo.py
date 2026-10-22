"""Clipped, region-action-only GRPO objective; language length never enters."""
from __future__ import annotations

import torch


def region_grpo_loss(policy, group: list[dict], clip_ratio: float = 0.2,
                     off_policy_mask_threshold: float | None = None):
    """GRPO loss for one question's region-action trajectories.

    ``old_log_prob`` belongs to the policy that generated the rollout.  Calling
    this function again after an optimizer step therefore performs a PPO-style
    off-policy update on the same generated trajectories.  The optional mask
    discards individual actions whose importance ratio has drifted too far.
    """
    if len(group) < 2:
        raise ValueError("At least two trajectories of the same question are required")
    if off_policy_mask_threshold is not None and off_policy_mask_threshold <= 0:
        raise ValueError("off_policy_mask_threshold must be positive when set")
    device = policy.nine_blocks.device
    rewards = torch.tensor([row["reward"] for row in group], device=device, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    terms, entropies = [], []
    action_count = active_action_count = masked_action_count = 0
    ratio_drifts = []
    for row, advantage in zip(group, advantages):
        actions = row["actions"]
        if not actions:
            continue
        h = torch.stack([entry["hidden"] for entry in actions]).to(device=device, dtype=torch.float32).detach()
        indices = torch.tensor([entry["action"] for entry in actions], device=device, dtype=torch.long)
        old = torch.tensor([entry["old_log_prob"] for entry in actions], device=device, dtype=torch.float32)
        distribution = policy.distribution(h)
        ratio = torch.exp(distribution.log_prob(indices) - old)
        action_count += ratio.numel()
        ratio_drifts.append((ratio - 1).abs().mean().detach())
        if off_policy_mask_threshold is None:
            keep = torch.ones_like(ratio, dtype=torch.bool)
        else:
            keep = (ratio - 1).abs() <= off_policy_mask_threshold
        active_action_count += int(keep.sum())
        masked_action_count += int((~keep).sum())
        if not keep.any():
            continue
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
        terms.append(-surrogate[keep].mean())  # each trajectory has equal weight, regardless of look count
        entropies.append(distribution.entropy().mean().detach())
    if not terms:
        return None, {"active_rollouts": 0, "action_entropy": 0.0,
                      "reward_std": float(rewards.std(unbiased=False)),
                      "action_count": action_count, "active_action_count": active_action_count,
                      "masked_action_count": masked_action_count,
                      "off_policy_mask_fraction": (masked_action_count / action_count if action_count else 0.0),
                      "mean_abs_ratio_drift": (float(torch.stack(ratio_drifts).mean()) if ratio_drifts else 0.0)}
    return torch.stack(terms).mean(), {
        "active_rollouts": len(terms),
        "action_entropy": float(torch.stack(entropies).mean()),
        "reward_std": float(rewards.std(unbiased=False)),
        "action_count": action_count,
        "active_action_count": active_action_count,
        "masked_action_count": masked_action_count,
        "off_policy_mask_fraction": masked_action_count / action_count if action_count else 0.0,
        "mean_abs_ratio_drift": float(torch.stack(ratio_drifts).mean()) if ratio_drifts else 0.0,
    }
