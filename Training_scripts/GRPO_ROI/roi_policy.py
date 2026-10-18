"""Small stochastic, state-conditioned ROI policy.

Actions live in unconstrained latent space.  Their Normal log probability is
therefore exact; geometry is only a deterministic rendering of that action.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ROIPolicyConfig:
    hidden_size: int
    roi_count: int = 9
    summary_size: int = 3
    max_bias: float = 4.0
    max_offset: float = 0.30
    min_size: float = 0.04
    max_size: float = 1.0
    center_exploration_std: float = 0.22
    size_exploration_std: float = 0.25
    strength_exploration_std: float = 2.0
    initial_attention_strength: float = 0.5
    roi_temperature: float = 1.0


def regular_grid(roi_count: int = 9, *, device=None, dtype=None) -> Tensor:
    """Return [R, 4] ``cx, cy, width, height`` for a square regular grid."""
    side = int(math.isqrt(roi_count))
    if side * side != roi_count:
        raise ValueError("roi_count must be a perfect square.")
    axis = (torch.arange(side, device=device, dtype=dtype) + .5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), 1 / side),
                        torch.full_like(xx.flatten(), 1 / side)), dim=-1)


class DynamicROIPolicy(nn.Module):
    """One linear head predicts 9x5 Normal means and standard deviations.

    Zero initial weights make the mean latent action zero.  Decoding zero then
    exactly gives the regular 3x3 geometry and zero enhancement strengths.
    """
    action_dim_per_roi = 5

    def __init__(self, cfg: ROIPolicyConfig):
        super().__init__()
        if cfg.roi_temperature <= 0 or min(cfg.center_exploration_std, cfg.size_exploration_std, cfg.strength_exploration_std) <= 0:
            raise ValueError("roi_temperature and all exploration standard deviations must be positive.")
        if not 0 <= cfg.initial_attention_strength < cfg.max_bias:
            raise ValueError("initial_attention_strength must be in [0, max_bias).")
        self.cfg = cfg
        feature_dim = cfg.hidden_size + cfg.summary_size ** 2
        self.mean_head = nn.Linear(feature_dim, cfg.roi_count * self.action_dim_per_roi)
        self.log_std_head = nn.Linear(feature_dim, cfg.roi_count * self.action_dim_per_roi)
        nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        stds = torch.tensor([cfg.center_exploration_std, cfg.center_exploration_std,
                             cfg.size_exploration_std, cfg.size_exploration_std,
                             cfg.strength_exploration_std]).repeat(cfg.roi_count)
        # inverse softplus: the initial conditional std is exactly configured.
        self.log_std_head.bias.data.copy_(torch.log(torch.expm1(stds)))
        self.register_buffer("base_geometry", regular_grid(cfg.roi_count), persistent=True)

    def features(self, hidden: Tensor, attention_summary: Tensor) -> Tensor:
        if hidden.shape[-1] != self.cfg.hidden_size:
            raise ValueError(f"Expected hidden size {self.cfg.hidden_size}, got {hidden.shape[-1]}.")
        expected = self.cfg.summary_size ** 2
        if attention_summary.shape[-1] != expected:
            raise ValueError(f"Expected {expected} pooled attention values.")
        return torch.cat((hidden, attention_summary), dim=-1)

    def distribution_parameters(self, features: Tensor) -> tuple[Tensor, Tensor]:
        shape = (*features.shape[:-1], self.cfg.roi_count, self.action_dim_per_roi)
        mean = self.mean_head(features).view(shape)
        std = torch.nn.functional.softplus(self.log_std_head(features)).view(shape) + 1e-5
        return mean, std

    def sample(self, hidden: Tensor, attention_summary: Tensor, *, prior_geometry: Tensor | None = None,
               deterministic: bool = False) -> dict[str, Tensor]:
        features = self.features(hidden, attention_summary)
        mean, std = self.distribution_parameters(features)
        latent = mean if deterministic else mean + self.cfg.roi_temperature * std * torch.randn_like(std)
        log_prob = self.log_probability(features, latent)
        return {"features": features, "latent": latent, "log_prob": log_prob,
                "mean": mean, "std": std, "geometry": self.decode(latent, prior_geometry)}

    def log_probability(self, features: Tensor, latent: Tensor) -> Tensor:
        mean, std = self.distribution_parameters(features)
        scaled_std = std * self.cfg.roi_temperature
        logp = -0.5 * (((latent - mean) / scaled_std).square() + 2 * scaled_std.log() + math.log(2 * math.pi))
        return logp.flatten(start_dim=-2).sum(-1)

    def decode(self, latent: Tensor, prior_geometry: Tensor | None = None) -> Tensor:
        """Map ``[..., R, 5]`` to valid ``cx,cy,w,h,lambda`` geometry."""
        base = self.base_geometry.to(device=latent.device, dtype=latent.dtype)
        if prior_geometry is not None:
            if prior_geometry.shape[-2:] != (self.cfg.roi_count, 4):
                raise ValueError("prior_geometry must end in [roi_count, 4].")
            base = prior_geometry.to(device=latent.device, dtype=latent.dtype)
        dxdy = torch.tanh(latent[..., :2]) * self.cfg.max_offset
        # Clamp after offset: valid image centres while retaining non-saturated zero.
        centre = (base[..., :2] + dxdy).clamp(0.0, 1.0)
        # exp(0)=1 makes the cold start exactly one regular-grid cell.
        size = (base[..., 2:] * torch.exp(latent[..., 2:4])).clamp(self.cfg.min_size, self.cfg.max_size)
        # A negative offset yields a near-zero cold start but retains a useful
        # derivative (unlike clamp(sigmoid(x)-.5), whose gradient is zero).
        prior_logit = math.log(self.cfg.initial_attention_strength / (self.cfg.max_bias - self.cfg.initial_attention_strength)) if self.cfg.initial_attention_strength else -20.0
        strength = self.cfg.max_bias * torch.sigmoid(latent[..., 4:5] + prior_logit)
        return torch.cat((centre, size, strength), dim=-1)

    @staticmethod
    def corners(geometry: Tensor) -> Tensor:
        """Return ``[..., R, 2, 2]`` as ``[[left, top], [right, bottom]]``."""
        centre, size = geometry[..., :2], geometry[..., 2:4]
        return torch.stack(((centre - size / 2).clamp(0, 1), (centre + size / 2).clamp(0, 1)), dim=-2)
