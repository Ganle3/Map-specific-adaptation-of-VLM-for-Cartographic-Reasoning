"""Image-conditioned continuous visual-slot policy for the prototype experiment.

The prototypes first read the frozen global merger tokens.  A reasoning state
then scores the resulting slots.  Training samples a simplex-valued gate from
a Dirichlet policy, so the appended visual tokens are a real GRPO action.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Dirichlet
import torch.nn.functional as F


class VisualPrototypeAppendPolicy(nn.Module):
    def __init__(self, hidden_size: int, num_prototypes: int = 9,
                 temperature: float = 1.0, min_concentration: float = 0.1):
        super().__init__()
        if hidden_size <= 0 or num_prototypes <= 0 or temperature <= 0:
            raise ValueError("hidden_size, num_prototypes and temperature must be positive")
        self.hidden_size = hidden_size
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.min_concentration = min_concentration
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)

    def slot_representations(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Return R=[r_1,...,r_K] by prototype-to-global-merger-token attention."""
        if image_embeddings.ndim == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
        if image_embeddings.ndim != 3 or image_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("image_embeddings must have shape [N,H] or [B,N,H]")
        scores = torch.matmul(self.prototypes.unsqueeze(0), image_embeddings.transpose(-1, -2))
        attention = torch.softmax(scores / math.sqrt(self.hidden_size), dim=-1)
        return torch.matmul(attention, image_embeddings)

    def distribution(self, h_t: torch.Tensor, image_embeddings: torch.Tensor):
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        slots = self.slot_representations(image_embeddings)
        if slots.shape[0] == 1 and h_t.shape[0] != 1:
            slots = slots.expand(h_t.shape[0], -1, -1)
        logits = torch.einsum("bkh,bh->bk", slots, h_t) / (math.sqrt(self.hidden_size) * self.temperature)
        # Dirichlet concentration must be positive.  This adds no MLP: it is
        # only the valid-domain map for the distribution parameters.
        concentration = F.softplus(logits) + self.min_concentration
        return Dirichlet(concentration), slots, logits, concentration

    def forward(self, h_t: torch.Tensor, image_embeddings: torch.Tensor, *, sample: bool = True):
        distribution, slots, logits, concentration = self.distribution(h_t, image_embeddings)
        weights = distribution.sample() if sample else distribution.mean
        return slots, logits, concentration, weights, distribution.log_prob(weights)


def decoder_hidden_size(model) -> int:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return int(base.config.text_config.hidden_size)
