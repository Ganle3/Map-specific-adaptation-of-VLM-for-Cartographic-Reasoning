"""Nine learnable queries that extract nine visual-memory tokens by cross-attention.

Computational graph (the default experiment trains only ``prototypes``)::

    prototypes P [9,H] --Query--+
                               |  frozen multi-head cross-attention
    global merger Z [N,H] --Key/Value
                               |
                               v
                   visual memory R [9,H]

The Q/K/V/O projections are identity-initialized and frozen by default.  This
keeps the only optimized variable equal to the requested nine nn.Parameter
tokens while still using genuine multi-head attention (head-wise scaled QK,
softmax over N image tokens, weighted V, concatenate heads).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeCrossAttentionPolicy(nn.Module):
    def __init__(self, hidden_size: int, *, num_prototypes: int = 9,
                 num_heads: int = 8, train_projections: bool = False):
        super().__init__()
        if hidden_size <= 0 or num_prototypes <= 0 or num_heads <= 0:
            raise ValueError("hidden_size, num_prototypes and num_heads must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_prototypes)
        self.num_heads = int(num_heads)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, bias=False, batch_first=True
        )
        self._initialize_identity_projections()
        for parameter in self.cross_attention.parameters():
            parameter.requires_grad_(train_projections)

    @torch.no_grad()
    def _initialize_identity_projections(self) -> None:
        eye = torch.eye(self.hidden_size)
        self.cross_attention.in_proj_weight.copy_(torch.cat((eye, eye, eye), dim=0))
        self.cross_attention.out_proj.weight.copy_(eye)

    def forward(self, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(memory_tokens, attention_weights)``.

        Args:
            image_tokens: global Qwen ViT+merger output, ``[N,H]`` or ``[B,N,H]``.
        Returns:
            memory_tokens: ``[B,9,H]`` (or ``[9,H]`` for an unbatched input).
            attention_weights: per-head maps ``[B,heads,9,N]``.
        """
        unbatched = image_tokens.ndim == 2
        if unbatched:
            image_tokens = image_tokens.unsqueeze(0)
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected image tokens [N,{self.hidden_size}] or [B,N,{self.hidden_size}]")
        batch = image_tokens.shape[0]
        # Layer-normalization has no trainable affine parameters.  Keys are normalized for stable matching; Values retain the native visual feature scale and content.
        query = F.layer_norm(self.prototypes, (self.hidden_size,)).unsqueeze(0).expand(batch, -1, -1)
        key = F.layer_norm(image_tokens.float(), (self.hidden_size,))
        value = image_tokens.float()
        memory, weights = self.cross_attention(
            query, key, value, need_weights=True, average_attn_weights=False
        )
        return (memory[0], weights) if unbatched else (memory, weights)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Explicit optimizer contract for the clean prototype-only experiment."""
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if len(parameters) != 1 or parameters[0] is not self.prototypes:
            raise RuntimeError("Default experiment must optimize only policy.prototypes")
        return parameters


def decoder_hidden_size(model) -> int:
    config = model.get_base_model().config if hasattr(model, "get_base_model") else model.config
    text_config = getattr(config, "text_config", None)
    if text_config is None and hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
    if text_config is None or not getattr(text_config, "hidden_size", None):
        raise RuntimeError("Qwen3-VL text hidden size is unavailable")
    return int(text_config.hidden_size)
