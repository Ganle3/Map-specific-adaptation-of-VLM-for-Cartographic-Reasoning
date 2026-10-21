"""Clipped, region-action-only GRPO objective; language length never enters."""
from __future__ import annotations

import torch


def region_grpo_loss(policy, group: list[dict], clip_ratio: float = 0.2):
    if len(group) < 2:
        raise ValueError("At least two trajectories of the same question are required")
    device = policy.nine_blocks.device
    rewards = torch.tensor([row["reward"] for row in group], device=device, dtype=torch.float32)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    terms, entropies = [], []
    for row, advantage in zip(group, advantages):
        actions = row["actions"]
        if not actions:
            continue
        h = torch.stack([entry["hidden"] for entry in actions]).to(device=device, dtype=torch.float32).detach()
        indices = torch.tensor([entry["action"] for entry in actions], device=device, dtype=torch.long)
        old = torch.tensor([entry["old_log_prob"] for entry in actions], device=device, dtype=torch.float32)
        distribution = policy.distribution(h)
        ratio = torch.exp(distribution.log_prob(indices) - old)
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
        terms.append(-surrogate.mean())  # each trajectory has equal weight, regardless of look count
        entropies.append(distribution.entropy().mean().detach())
    if not terms:
        return None, {"active_rollouts": 0, "action_entropy": 0.0, "reward_std": float(rewards.std(unbiased=False))}
    return torch.stack(terms).mean(), {
        "active_rollouts": len(terms),
        "action_entropy": float(torch.stack(entropies).mean()),
        "reward_std": float(rewards.std(unbiased=False)),
    }
