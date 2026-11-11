"""Shared learnable relay parameter bank for all relay placements."""
from __future__ import annotations

import torch
from torch import nn


class RelayModule(nn.Module):
    """Own the single relay ``nn.Parameter`` reused by a scheme.

    Placement and attention behavior intentionally live in each scheme's
    directory.  This module only defines the trainable ``[R, H]`` tensor.
    """

    def __init__(self, hidden_size: int, *, num_relays: int = 2,
                 init_std: float = 0.02,
                 initialization: str = "normal_0.02") -> None:
        super().__init__()
        if hidden_size <= 0 or num_relays <= 0 or init_std <= 0:
            raise ValueError("hidden_size, num_relays and init_std must be positive")
        if initialization not in {"normal_0.02", "xavier_normal"}:
            raise ValueError(f"Unsupported relay initialization: {initialization!r}")
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_relays)
        self.initialization = initialization
        self.prototypes = nn.Parameter(torch.empty(
            self.num_prototypes, self.hidden_size
        ))
        if initialization == "normal_0.02":
            nn.init.normal_(self.prototypes, mean=0.0, std=float(init_std))
        else:
            nn.init.xavier_normal_(self.prototypes)

    @property
    def relay_tokens(self) -> nn.Parameter:
        return self.prototypes

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.prototypes]
