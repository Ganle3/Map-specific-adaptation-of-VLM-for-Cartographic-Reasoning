"""The only trainable parameters in the first When--Where experiment."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical


WHEN_FEATURES = 3
WHERE_FEATURES = 6


@dataclass(frozen=True)
class ControllerConfig:
    tau0: float
    temperature: float
    size_sigma: float = 0.35
    architecture: str = "linear"
    mlp_hidden_dim: int = 16
    # Multiplicative cold-start centre for legal ROI width/height actions.
    # It is represented by trainable log-scale biases, so GRPO can later
    # learn either expansion or contraction from this prior.
    initial_size_scale: float = 1.0


class ZeroInitialisedResidual(nn.Module):
    """Small nonlinear residual that is exactly zero at cold start.

    The first layer is normally trainable from the second optimizer step; the
    final layer starts at zero so selecting this module cannot change the
    validated linear policy before GRPO updates it.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.output(torch.tanh(self.input(features))).squeeze(-1)


class WhenWhereController(nn.Module):
    """Four scalar When parameters and shared linear Where residual heads.

    All learned residuals initialise at zero.  The supplied heuristic keep
    logits and legal heuristic size candidates therefore define the exact
    cold-start policy, rather than an accidental learned preference.
    """
    def __init__(self, cfg: ControllerConfig):
        super().__init__()
        if cfg.temperature <= 0 or cfg.size_sigma <= 0 or cfg.initial_size_scale <= 0 or cfg.mlp_hidden_dim < 1:
            raise ValueError("temperature, size_sigma, initial_size_scale, and mlp_hidden_dim must be positive.")
        if cfg.architecture not in {"linear", "residual_mlp"}:
            raise ValueError("architecture must be 'linear' or 'residual_mlp'.")
        self.cfg = cfg
        self.delta_tau = nn.Parameter(torch.zeros(()))
        self.when_linear = nn.Parameter(torch.zeros(WHEN_FEATURES))  # c, g, budget
        self.keep_weight = nn.Parameter(torch.zeros(WHERE_FEATURES))
        self.keep_bias = nn.Parameter(torch.zeros(()))
        self.width_weight = nn.Parameter(torch.zeros(WHERE_FEATURES))
        self.width_bias = nn.Parameter(torch.tensor(math.log(cfg.initial_size_scale), dtype=torch.float32))
        self.height_weight = nn.Parameter(torch.zeros(WHERE_FEATURES))
        self.height_bias = nn.Parameter(torch.tensor(math.log(cfg.initial_size_scale), dtype=torch.float32))
        if cfg.architecture == "residual_mlp":
            self.when_residual = ZeroInitialisedResidual(4, cfg.mlp_hidden_dim)
            self.keep_residual = ZeroInitialisedResidual(WHERE_FEATURES, cfg.mlp_hidden_dim)
            self.width_residual = ZeroInitialisedResidual(WHERE_FEATURES, cfg.mlp_hidden_dim)
            self.height_residual = ZeroInitialisedResidual(WHERE_FEATURES, cfg.mlp_hidden_dim)
        else:
            self.when_residual = None
            self.keep_residual = None
            self.width_residual = None
            self.height_residual = None

    def when_logits(self, features: Tensor) -> Tensor:
        """features is ``[..., jsd, concentration, growth, budget_fraction]``."""
        if features.shape[-1] != 4:
            raise ValueError("When features must be [jsd, concentration, growth, budget].")
        threshold = self.cfg.tau0 + self.delta_tau + (features[..., 1:] * self.when_linear).sum(-1)
        base_logits = (features[..., 0] - threshold) / self.cfg.temperature
        return base_logits if self.when_residual is None else base_logits + self.when_residual(features)

    def keep_logits(self, features: Tensor, heuristic_logits: Tensor) -> Tensor:
        base_logits = heuristic_logits + features @ self.keep_weight + self.keep_bias
        return base_logits if self.keep_residual is None else base_logits + self.keep_residual(features)

    def size_logits(self, features: Tensor, candidates_log_ratio: Tensor) -> Tensor:
        """Score valid [width,height] log-ratios about each heuristic bbox.

        candidates_log_ratio has final dimension 2 and can have any number of
        preceding candidate-size dimensions.
        """
        mu_w = features @ self.width_weight + self.width_bias
        mu_h = features @ self.height_weight + self.height_bias
        if self.width_residual is not None:
            mu_w = mu_w + self.width_residual(features)
            mu_h = mu_h + self.height_residual(features)
        delta_w = candidates_log_ratio[..., 0] - mu_w.unsqueeze(-1)
        delta_h = candidates_log_ratio[..., 1] - mu_h.unsqueeze(-1)
        return -(delta_w.square() + delta_h.square()) / (2 * self.cfg.size_sigma ** 2)

    @torch.no_grad()
    def sample_when(self, features: Tensor) -> tuple[bool, float]:
        dist = Bernoulli(logits=self.when_logits(features))
        action = dist.sample()
        return bool(action.item()), float(dist.log_prob(action).item())

    @torch.no_grad()
    def sample_keep(self, features: Tensor, heuristic_logits: Tensor) -> tuple[Tensor, Tensor]:
        dist = Bernoulli(logits=self.keep_logits(features, heuristic_logits))
        actions = dist.sample()
        return actions.bool(), dist.log_prob(actions)

    @torch.no_grad()
    def sample_size(self, features: Tensor, log_ratios: Tensor) -> tuple[int, float]:
        logits = self.size_logits(features.unsqueeze(0), log_ratios.unsqueeze(0)).squeeze(0)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item())


def recompute_action_log_probability(controller: WhenWhereController, actions: list[dict], device: torch.device) -> Tensor:
    """Exactly re-evaluate stored stochastic actions under the current policy."""
    # Keep a valid zero-gradient graph for legal zero-LOOK trajectories.
    # Such trajectories must be supported; they simply contribute no policy
    # gradient rather than making ``loss.backward()`` fail for a whole group.
    total = next(controller.parameters()).sum() * 0.0
    for row in actions:
        kind = row["kind"]
        if kind == "when":
            features = torch.tensor(row["features"], dtype=torch.float32, device=device)
            total = total + Bernoulli(logits=controller.when_logits(features)).log_prob(torch.tensor(row["action"], device=device))
        elif kind == "keep":
            features = torch.tensor(row["features"], dtype=torch.float32, device=device)
            prior = torch.tensor(row["heuristic_logits"], dtype=torch.float32, device=device)
            action = torch.tensor(row["action"], dtype=torch.float32, device=device)
            total = total + Bernoulli(logits=controller.keep_logits(features, prior)).log_prob(action).sum()
        elif kind == "size":
            features = torch.tensor(row["features"], dtype=torch.float32, device=device)
            ratios = torch.tensor(row["log_ratios"], dtype=torch.float32, device=device)
            logits = controller.size_logits(features.unsqueeze(0), ratios.unsqueeze(0)).squeeze(0)
            total = total + Categorical(logits=logits).log_prob(torch.tensor(row["action"], device=device))
        else:
            raise ValueError(f"Unknown action kind: {kind}")
    return total
