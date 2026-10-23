"""Clipped, region-action-only GRPO objective; language length never enters."""
from __future__ import annotations

import torch


@torch.no_grad()
def capture_policy_update_diagnostics(policy, batch: list[tuple[int, object, list[dict]]]) -> dict:
    """Snapshot the exact sampled actions before one PPO/GRPO update.

    This is intentionally diagnostic-only.  It uses the same per-question
    normalized advantage definition as :func:`region_grpo_loss`, but does not
    create a graph or change the loss.
    """
    hiddens, actions, advantages = [], [], []
    group_summaries = []
    for dataset_index, item, group in batch:
        rewards = torch.tensor([row["reward"] for row in group], dtype=torch.float32)
        group_advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
        group_summaries.append({"dataset_index": dataset_index, "qa_id": item.sample_id,
                                "rewards": [float(value) for value in rewards],
                                "advantages": [float(value) for value in group_advantages]})
        for row, advantage in zip(group, group_advantages):
            for entry in row["actions"]:
                hiddens.append(entry["hidden"].detach().cpu())
                actions.append(int(entry["action"]))
                advantages.append(float(advantage))
    if not hiddens:
        return {"groups": group_summaries, "count": 0}
    device = policy.nine_blocks.device
    hidden_tensor = torch.stack(hiddens).to(device=device, dtype=torch.float32)
    action_tensor = torch.tensor(actions, device=device, dtype=torch.long)
    before_log_probs = policy.distribution(hidden_tensor).log_prob(action_tensor).cpu()
    return {"groups": group_summaries, "count": len(actions),
            "hidden": hidden_tensor.cpu(), "actions": action_tensor.cpu(),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
            "before_log_probs": before_log_probs}


@torch.no_grad()
def finalize_policy_update_diagnostics(policy, snapshot: dict, prototype_gradient_norms: torch.Tensor,
                                       prototype_delta_norms: torch.Tensor) -> dict:
    """Turn a pre-update snapshot into JSON-safe evidence of a real update."""
    result = {"groups": snapshot["groups"], "action_count": snapshot["count"],
              "prototype_gradient_norms": [float(x) for x in prototype_gradient_norms.cpu()],
              "prototype_delta_norms": [float(x) for x in prototype_delta_norms.cpu()]}
    if not snapshot["count"]:
        result.update({"selected_action_counts": [0] * 9,
                       "selected_action_advantage_sum": [0.0] * 9,
                       "positive_advantage_logprob_delta_mean": None,
                       "negative_advantage_logprob_delta_mean": None})
        return result
    device = policy.nine_blocks.device
    hidden = snapshot["hidden"].to(device=device, dtype=torch.float32)
    actions = snapshot["actions"].to(device=device, dtype=torch.long)
    after_log_probs = policy.distribution(hidden).log_prob(actions).cpu()
    deltas = after_log_probs - snapshot["before_log_probs"]
    advantages = snapshot["advantages"]
    counts = torch.bincount(snapshot["actions"], minlength=9)
    advantage_sums = torch.zeros(9, dtype=torch.float32)
    advantage_sums.scatter_add_(0, snapshot["actions"], advantages)
    positive = deltas[advantages > 0]
    negative = deltas[advantages < 0]
    result.update({
        "selected_action_counts": [int(x) for x in counts],
        "selected_action_advantage_sum": [float(x) for x in advantage_sums],
        "positive_advantage_logprob_delta_mean": float(positive.mean()) if positive.numel() else None,
        "negative_advantage_logprob_delta_mean": float(negative.mean()) if negative.numel() else None,
        "zero_advantage_action_count": int((advantages == 0).sum()),
    })
    return result


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
