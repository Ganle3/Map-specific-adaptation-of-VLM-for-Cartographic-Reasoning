"""State-conditioned prototype queries over frozen global merger tokens."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.distributions import Dirichlet
import torch.nn.functional as F


class DynamicVisualPrototypeAppendPolicy(nn.Module):
    """K shared prototypes whose image queries are conditioned by ``h_t``.

    ``p_i`` is a learned behaviour template.  At every LOOK, the elementwise
    interaction ``p_i * h_t`` produces a distinct query for each template,
    and that query attends over the frozen full-image merger embeddings.
    """
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

    def slot_representations(self, h_t: torch.Tensor, image_embeddings: torch.Tensor):
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        if image_embeddings.ndim == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
        if image_embeddings.ndim != 3 or image_embeddings.shape[-1] != self.hidden_size:
            raise ValueError("image_embeddings must have shape [N,H] or [B,N,H]")
        if image_embeddings.shape[0] == 1 and h_t.shape[0] != 1:
            image_embeddings = image_embeddings.expand(h_t.shape[0], -1, -1)
        if image_embeddings.shape[0] != h_t.shape[0]:
            raise ValueError("batch sizes of h_t and image_embeddings must match")
        # Each current reasoning state modulates every shared prototype in a
        # different way.  Normalization prevents the frozen decoder-state norm
        # from turning attention sharpness into an accidental policy signal.
        state = F.normalize(h_t, dim=-1).unsqueeze(1)
        templates = F.normalize(self.prototypes, dim=-1).unsqueeze(0)
        queries = F.normalize(templates * state, dim=-1)
        attention_logits = torch.matmul(queries, image_embeddings.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        attention = torch.softmax(attention_logits, dim=-1)
        return torch.matmul(attention, image_embeddings), attention

    def distribution(self, h_t: torch.Tensor, image_embeddings: torch.Tensor):
        slots, _ = self.slot_representations(h_t, image_embeddings)
        if h_t.ndim == 1:
            h_t = h_t.unsqueeze(0)
        logits = torch.einsum("bkh,bh->bk", slots, h_t) / (math.sqrt(self.hidden_size) * self.temperature)
        concentration = F.softplus(logits) + self.min_concentration
        return Dirichlet(concentration), slots, logits, concentration

    def forward(self, h_t: torch.Tensor, image_embeddings: torch.Tensor, *, sample: bool = True):
        distribution, slots, logits, concentration = self.distribution(h_t, image_embeddings)
        weights = distribution.sample() if sample else distribution.mean
        return slots, logits, concentration, weights, distribution.log_prob(weights)


def decoder_hidden_size(model) -> int:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return int(base.config.text_config.hidden_size)
