"""Nine shared action prototypes for fixed map regions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


class NineGridRegionPolicy(nn.Module):
    def __init__(self, hidden_size: int, temperature: float = 1.0):
        super().__init__()
        if hidden_size <= 0 or temperature <= 0:
            raise ValueError("hidden_size and temperature must be positive")
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.nine_blocks = nn.Parameter(torch.randn(9, hidden_size) * 0.02)

    def distribution(self, h_t: torch.Tensor) -> Categorical:
        if h_t.ndim != 2 or h_t.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected [batch, {self.hidden_size}] hidden states")
        return Categorical(logits=self.logits(h_t))

    def logits(self, h_t: torch.Tensor) -> torch.Tensor:
        if h_t.ndim != 2 or h_t.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected [batch, {self.hidden_size}] hidden states")
        return (h_t.float() @ self.nine_blocks.float().T) / (math.sqrt(self.hidden_size) * self.temperature)

    def forward(self, h_t: torch.Tensor, *, sample: bool = True):
        logits = self.logits(h_t)
        dist = Categorical(logits=logits)
        action = dist.sample() if sample else dist.logits.argmax(dim=-1)
        return logits, dist.probs, action, dist.log_prob(action)


def nine_grid_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    if width < 3 or height < 3:
        raise ValueError("Nine regions require an image at least 3 x 3 pixels")
    xs = [i * width // 3 for i in range(4)]
    ys = [i * height // 3 for i in range(4)]
    return [(xs[col], ys[row], xs[col + 1], ys[row + 1])
            for row in range(3) for col in range(3)]


def decoder_hidden_size(model) -> int:
    config = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    text_config = getattr(config, "text_config", None)
    if text_config is None and hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
    if text_config is None or not getattr(text_config, "hidden_size", None):
        raise RuntimeError("Qwen3-VL text_config.hidden_size unavailable")
    return int(text_config.hidden_size)
