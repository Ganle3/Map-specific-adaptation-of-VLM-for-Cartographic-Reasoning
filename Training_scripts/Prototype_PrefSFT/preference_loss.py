"""Question-level preference objective; intentionally independent of model I/O."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class PreferenceOutput:
    loss: torch.Tensor
    mean_positive_gain: torch.Tensor
    mean_negative_gain: torch.Tensor
    mean_margin: torch.Tensor
    positive_score_proto: torch.Tensor
    negative_score_proto: torch.Tensor
    positive_score_baseline: torch.Tensor
    negative_score_baseline: torch.Tensor


def prototype_preference_loss(proto_scores: torch.Tensor, baseline_scores: torch.Tensor,
                              correct: torch.Tensor, beta: float) -> PreferenceOutput:
    """Mean -log sigmoid(beta * (gain+ - gain-)) over pairs in one question."""
    positive = correct.to(
        device=proto_scores.device,
        dtype=torch.bool,
    )
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("A preference question must contain both correct and incorrect rollouts.")
    gains = proto_scores - baseline_scores.detach()  # detach baseline to avoid double-counting its gradient
    margins = gains[positive][:, None] - gains[negative][None, :]
    return PreferenceOutput(
        loss=-F.logsigmoid(beta * margins).mean(),
        mean_positive_gain=gains[positive].mean(), mean_negative_gain=gains[negative].mean(),
        mean_margin=margins.mean(), positive_score_proto=proto_scores[positive].mean(),
        negative_score_proto=proto_scores[negative].mean(),
        positive_score_baseline=baseline_scores[positive].mean(),
        negative_score_baseline=baseline_scores[negative].mean(),
    )
