"""Post-norm Transformer cross-attention block for prototype visual memory::

    A = CrossMHA(Q=P, K=Z_image, V=Z_image)
    X = LayerNorm(P + Dropout(A))
    R = LayerNorm(X + Dropout(MLP(X)))

All policy parameters are trainable; the Qwen3-VL backbone and its loaded
adapter remain frozen by the trainer.
"""
from __future__ import annotations

import torch
from torch import nn


class PrototypeCrossAttentionBlockPolicy(nn.Module):
    def __init__(self, hidden_size: int, *, num_prototypes: int = 9,
                 num_heads: int = 8, ffn_multiplier: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        if hidden_size <= 0 or num_prototypes <= 0 or num_heads <= 0:
            raise ValueError("hidden_size, num_prototypes and num_heads must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.hidden_size = int(hidden_size)
        self.num_prototypes = int(num_prototypes)
        self.num_heads = int(num_heads)
        self.ffn_multiplier = int(ffn_multiplier)

        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, hidden_size) * 0.02
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, bias=False, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_multiplier * hidden_size),
            nn.GELU(),
            nn.Linear(ffn_multiplier * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Transformer-style initialization for stable initial variance."""
        # bias=False and equal Q/K/V dimensions make MHA store the three input
        # projections together as [3H,H].
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for norm in (self.norm1, self.norm2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def forward(self, image_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return nine image-conditioned tokens and per-head cross-attention."""
        unbatched = image_tokens.ndim == 2
        if unbatched:
            image_tokens = image_tokens.unsqueeze(0)
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected image tokens [N,{self.hidden_size}] or [B,N,{self.hidden_size}]"
            )
        image_tokens = image_tokens.to(dtype=self.prototypes.dtype)
        batch = image_tokens.shape[0]
        query = self.prototypes.unsqueeze(0).expand(batch, -1, -1)
        attended, weights = self.cross_attention(
            query=query,
            key=image_tokens,
            value=image_tokens,
            need_weights=True, average_attn_weights=False,
        )
        hidden = self.norm1(query + self.dropout(attended))
        memory = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return (memory[0], weights) if unbatched else (memory, weights)

    def trainable_parameters(self) -> list[nn.Parameter]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters or not any(parameter is self.prototypes for parameter in parameters):
            raise RuntimeError("Transformer block has no trainable prototype path")
        return parameters
