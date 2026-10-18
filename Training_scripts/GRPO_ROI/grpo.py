"""Trajectory-level PPO/GRPO update for external ROI actions only."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .roi_policy import DynamicROIPolicy


@dataclass(frozen=True)
class GRPOConfig:
    learning_rate: float = 2e-4
    clip_ratio: float = .2
    ppo_epochs: int = 1
    max_grad_norm: float = 1.0
    kl_coefficient: float = 0.0


def group_advantages(rewards: Tensor) -> Tensor:
    std = rewards.std(unbiased=False)
    return torch.zeros_like(rewards) if std < 1e-8 else (rewards - rewards.mean()) / (std + 1e-6)


class ROIGrpoUpdater:
    def __init__(self, policy: DynamicROIPolicy, cfg: GRPOConfig):
        self.policy, self.cfg = policy, cfg
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)

    def _trajectory_logp(self, trajectory: dict) -> Tensor:
        device = next(self.policy.parameters()).device
        total = next(self.policy.parameters()).sum() * 0.0
        for action in trajectory["actions"]:
            feature = torch.as_tensor(action["features"], device=device, dtype=torch.float32)
            latent = torch.as_tensor(action["latent"], device=device, dtype=torch.float32)
            total = total + self.policy.log_probability(feature, latent).sum()
        return total

    def update_group(self, trajectories: list[dict]) -> dict:
        if len(trajectories) < 2:
            raise ValueError("A GRPO group needs at least two rollouts.")
        device = next(self.policy.parameters()).device
        rewards = torch.tensor([x["reward"] for x in trajectories], device=device)
        advantages = group_advantages(rewards)
        old = torch.tensor([x["old_log_probability"] for x in trajectories], device=device)
        stats = {}
        for _ in range(self.cfg.ppo_epochs):
            current = torch.stack([self._trajectory_logp(x) for x in trajectories])
            log_ratio = current - old
            ratio = log_ratio.exp()
            unclipped, clipped = ratio * advantages, ratio.clamp(1-self.cfg.clip_ratio, 1+self.cfg.clip_ratio) * advantages
            loss = -torch.minimum(unclipped, clipped).mean()
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            stats = {"policy_loss": float(loss.detach()), "approx_kl": float((-log_ratio).mean().detach()),
                     "clip_fraction": float((ratio.sub(1).abs() > self.cfg.clip_ratio).float().mean().detach()),
                     "grad_norm": float(grad)}
        stats.update(reward_mean=float(rewards.mean()), reward_std=float(rewards.std(unbiased=False)),
                     group_has_signal=float(rewards.std(unbiased=False) >= 1e-8))
        return stats
