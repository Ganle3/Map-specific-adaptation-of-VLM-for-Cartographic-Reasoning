"""GRPO objective for sampled continuous visual-slot gates."""
from __future__ import annotations

import torch


def visual_slot_grpo_loss(policy, group: list[dict], clip_ratio: float = 0.2,
                          off_policy_mask_threshold: float | None = None):
    """PPO/GRPO only over Dirichlet gates; language model remains frozen."""
    device = policy.prototypes.device
    rewards = torch.tensor([row["reward"] for row in group], device=device, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    terms, entropies, ratio_drifts = [], [], []
    action_count = active_action_count = masked_action_count = 0
    for row, advantage in zip(group, advantages):
        actions = row["actions"]
        if not actions:
            continue
        image_embeddings = row["image_embeddings"].to(device=device, dtype=torch.float32).detach()
        h = torch.stack([entry["hidden"] for entry in actions]).to(device=device, dtype=torch.float32).detach()
        gates = torch.stack([entry["gate"] for entry in actions]).to(device=device, dtype=torch.float32)
        old = torch.tensor([entry["old_log_prob"] for entry in actions], device=device, dtype=torch.float32)
        distribution, _, _, _ = policy.distribution(h, image_embeddings)
        ratio = torch.exp(distribution.log_prob(gates) - old)
        action_count += ratio.numel()
        ratio_drifts.append((ratio - 1).abs().mean().detach())
        keep = (torch.ones_like(ratio, dtype=torch.bool) if off_policy_mask_threshold is None
                else (ratio - 1).abs() <= off_policy_mask_threshold)
        active_action_count += int(keep.sum())
        masked_action_count += int((~keep).sum())
        if not keep.any():
            continue
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
        terms.append(-surrogate[keep].mean())
        entropies.append(distribution.entropy().mean().detach())
    if not terms:
        return None, {"active_rollouts": 0, "action_entropy": 0.0,
                      "reward_std": float(rewards.std(unbiased=False)), "action_count": action_count,
                      "active_action_count": active_action_count, "masked_action_count": masked_action_count,
                      "off_policy_mask_fraction": masked_action_count / action_count if action_count else 0.0,
                      "mean_abs_ratio_drift": float(torch.stack(ratio_drifts).mean()) if ratio_drifts else 0.0}
    return torch.stack(terms).mean(), {
        "active_rollouts": len(terms), "action_entropy": float(torch.stack(entropies).mean()),
        "reward_std": float(rewards.std(unbiased=False)), "action_count": action_count,
        "active_action_count": active_action_count, "masked_action_count": masked_action_count,
        "off_policy_mask_fraction": masked_action_count / action_count if action_count else 0.0,
        "mean_abs_ratio_drift": float(torch.stack(ratio_drifts).mean()) if ratio_drifts else 0.0,
    }
