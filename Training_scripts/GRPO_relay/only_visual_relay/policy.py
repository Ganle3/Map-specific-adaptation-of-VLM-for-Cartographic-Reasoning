"""The trainable relay parameter bank used after ``<vision_end>``."""
from __future__ import annotations

import torch
from torch import nn


class OnlyVisualRelayPolicy(nn.Module):
    def __init__(self, hidden_size: int, *, num_relays: int = 2,
                 init_std: float = 0.02) -> None:
        super().__init__()
        if hidden_size <= 0 or num_relays <= 0 or init_std <= 0:
            raise ValueError("hidden_size, num_relays and init_std must be positive")
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_relays)
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.hidden_size) * float(init_std)
        )

    @property
    def relay_tokens(self) -> nn.Parameter:
        return self.prototypes

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.prototypes]
