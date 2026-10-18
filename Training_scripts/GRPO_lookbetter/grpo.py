"""Small trajectory-level GRPO updater; it never touches Qwen parameters."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from policy import WhenWhereController, recompute_action_log_probability


@dataclass(frozen=True)
class GRPOConfig:
    learning_rate: float = 2e-3
    clip_ratio: float = 0.2
    max_grad_norm: float = 1.0


class ControllerGRPO:
    def __init__(self, controller: WhenWhereController, cfg: GRPOConfig):
        self.controller, self.cfg = controller, cfg
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=cfg.learning_rate)

    def _group_terms(self, trajectories: list[dict]) -> tuple[list[torch.Tensor], dict]:
        """Build PPO-surrogate terms with advantages normalized *within one QA*.

        A caller may combine terms from multiple QA groups into one optimizer
        step, but rewards from distinct questions must never share a baseline.
        """
        if len(trajectories) < 2:
            raise ValueError("GRPO needs at least two rollouts per question.")
        device = next(self.controller.parameters()).device
        rewards = torch.tensor([row["reward"] for row in trajectories], dtype=torch.float32, device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        losses, ratios = [], []
        for advantage, row in zip(advantages, trajectories):
            new_logp = recompute_action_log_probability(self.controller, row["actions"], device)
            old_logp = torch.tensor(row["old_log_probability"], dtype=torch.float32, device=device)
            ratio = torch.exp(new_logp - old_logp)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * advantage
            losses.append(-torch.minimum(unclipped, clipped))
            ratios.append(ratio.detach())
        return losses, {"reward_mean": float(rewards.mean()),
                        "accuracy": float(rewards.mean()),
                        "correct_rollouts": int(rewards.sum().item()),
                        "total_rollouts": len(trajectories),
                        "reward_std": float(rewards.std(unbiased=False)),
                        "ratio_mean": float(torch.stack(ratios).mean()),
                        "mixed_rewards": bool((rewards != rewards[0]).any())}

    def update_groups(self, groups: list[list[dict]]) -> tuple[dict, list[dict]]:
        """Make one controller update from several independently normalized QA groups."""
        if not groups:
            raise ValueError("At least one QA rollout group is required.")
        all_losses, group_metrics = [], []
        for trajectories in groups:
            losses, metrics = self._group_terms(trajectories)
            all_losses.extend(losses)
            group_metrics.append(metrics)
        loss = torch.stack(all_losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        update = {"loss": float(loss.detach()), "grad_norm": float(grad_norm),
                  "question_groups": len(groups), "rollouts": len(all_losses),
                  "reward_mean": float(sum(row["reward_mean"] for row in group_metrics) / len(group_metrics)),
                  "accuracy": float(sum(row["accuracy"] for row in group_metrics) / len(group_metrics)),
                  "correct_rollouts": int(sum(row["correct_rollouts"] for row in group_metrics)),
                  "total_rollouts": int(sum(row["total_rollouts"] for row in group_metrics))}
        return update, group_metrics

    def update_group(self, trajectories: list[dict]) -> dict:
        """Backward-compatible single-QA update."""
        update, metrics = self.update_groups([trajectories])
        return {**update, **metrics[0]}
