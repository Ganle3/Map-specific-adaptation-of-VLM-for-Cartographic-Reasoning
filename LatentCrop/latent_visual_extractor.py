"""Parallel latent crop representation model; the frozen VLM is not part of this module.

The model consumes the final reasoning hidden state and *all* layers of the
original image visual KV.  Its output is in the Qwen3-VL main-merger embedding
space, not in decoder-KV space.  The learned STOP vector is never an LM token.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ExtractorConfig:
    reasoning_dim: int
    kv_dim: int
    merger_dim: int
    decoder_layers: int
    latent_dim: int
    attention_heads: int = 8
    k_max: int = 384
    query_grid_width: int = 16
    mlp_ratio: int = 2
    stop_threshold: float = 0.0
    stop_logit_scale: float = 10.0

    def validate(self) -> None:
        if min(self.reasoning_dim, self.kv_dim, self.merger_dim, self.decoder_layers,
               self.latent_dim, self.attention_heads, self.k_max, self.query_grid_width) < 1:
            raise ValueError("All dimensions, layer counts and query limits must be positive.")
        if self.latent_dim % self.attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads.")
        if self.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be at least 1.")


class LatentVisualExtractor(nn.Module):
    """One-block, layer-aware KV fusion and parallel crop-like token extractor.

    ``global_keys`` and ``global_values`` have shape [B, L, N, H_kv*D_head].
    Cached keys already have decoder RoPE applied; they are treated as
    layer-specific features, not as interchangeable unrotated vision tokens.
    The original image's 2-D grid can optionally be supplied as ``global_grid``
    so its coordinates remain visible after the layer reduction.

    A canonical 2-D query grid is used for the representation-only first stage.
    It is *not* a claim that the generated sequence has a Qwen-valid crop grid;
    decoder insertion/layout is deliberately outside this training module.
    """

    def __init__(self, config: ExtractorConfig):
        super().__init__()
        config.validate()
        self.config = config
        d = config.latent_dim
        self.key_layer_logits = nn.Parameter(torch.zeros(config.decoder_layers))
        self.value_layer_logits = nn.Parameter(torch.zeros(config.decoder_layers))
        self.key_projection = nn.Linear(config.kv_dim, d)
        self.value_projection = nn.Linear(config.kv_dim, d)
        self.reasoning_projection = nn.Sequential(nn.LayerNorm(config.reasoning_dim), nn.Linear(config.reasoning_dim, d))
        self.global_position_projection = nn.Linear(2, d, bias=False)
        self.queries = nn.Parameter(torch.randn(config.k_max + 1, d) * 0.02)
        self.query_row = nn.Embedding(math.ceil((config.k_max + 1) / config.query_grid_width), d)
        self.query_col = nn.Embedding(config.query_grid_width, d)
        self.cross_attention = nn.MultiheadAttention(d, config.attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(d)
        self.feedforward = nn.Sequential(nn.Linear(d, config.mlp_ratio * d), nn.GELU(),
                                         nn.Linear(config.mlp_ratio * d, d))
        self.output_norm = nn.LayerNorm(d)
        self.merger_output = nn.Linear(d, config.merger_dim)
        self.stop_token = nn.Parameter(torch.randn(d) * 0.02)

    def _fuse_visual_memory(self, keys: torch.Tensor, values: torch.Tensor,
                            memory_mask: torch.Tensor, global_grid: torch.Tensor | None):
        cfg = self.config
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Global K/V must share shape [batch, layers, image_tokens, kv_dim].")
        batch, layers, tokens, width = keys.shape
        if layers != cfg.decoder_layers or width != cfg.kv_dim or memory_mask.shape != (batch, tokens):
            raise ValueError("Global K/V or mask dimensions do not match the extractor configuration.")
        if not memory_mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one original-image visual token.")

        # Different layers and K/V streams get separate learnable contributions.
        # Reduce before the large projections to avoid materialising [B,L,N,d].
        wk = self.key_layer_logits.softmax(dim=0)
        wv = self.value_layer_logits.softmax(dim=0)
        fused_k = torch.zeros((batch, tokens, width), device=keys.device, dtype=torch.float32)
        fused_v = torch.zeros_like(fused_k)
        for layer in range(layers):
            fused_k = fused_k + wk[layer] * F.layer_norm(keys[:, layer].float(), (width,))
            fused_v = fused_v + wv[layer] * F.layer_norm(values[:, layer].float(), (width,))
        memory_k = self.key_projection(fused_k)
        memory_v = self.value_projection(fused_v)

        if global_grid is not None:
            if global_grid.shape != (batch, 2):
                raise ValueError("global_grid must be [batch, merged_height, merged_width].")
            coordinates = torch.arange(tokens, device=keys.device)[None, :].expand(batch, -1)
            grid_width = global_grid[:, 1].clamp_min(1).to(keys.device)[:, None]
            y = torch.div(coordinates, grid_width, rounding_mode="floor")
            x = coordinates.remainder(grid_width)
            xy = torch.stack((y / global_grid[:, 0].clamp_min(1).to(keys.device)[:, None],
                              x / grid_width), dim=-1).to(memory_k.dtype)
            position = self.global_position_projection(xy)
            memory_k = memory_k + position
            memory_v = memory_v + position
        return memory_k, memory_v

    def forward(self, reasoning_hidden: torch.Tensor, global_keys: torch.Tensor,
                global_values: torch.Tensor, memory_mask: torch.Tensor,
                global_grid: torch.Tensor | None = None, num_queries: int | None = None) -> dict[str, torch.Tensor]:
        cfg = self.config
        if reasoning_hidden.ndim != 2 or reasoning_hidden.shape[-1] != cfg.reasoning_dim:
            raise ValueError("reasoning_hidden must have shape [batch, reasoning_dim].")
        count = cfg.k_max + 1 if num_queries is None else num_queries
        if not 1 <= count <= cfg.k_max + 1:
            raise ValueError("num_queries must be in [1, k_max + 1].")
        memory_k, memory_v = self._fuse_visual_memory(global_keys, global_values, memory_mask, global_grid)
        indices = torch.arange(count, device=reasoning_hidden.device)
        query = (self.queries[:count] + self.query_row(indices // cfg.query_grid_width)
                 + self.query_col(indices % cfg.query_grid_width))
        query = query[None] + self.reasoning_projection(reasoning_hidden.float())[:, None]
        attention, _ = self.cross_attention(query, memory_k, memory_v,
                                            key_padding_mask=~memory_mask, need_weights=False)
        states = self.attention_norm(query + attention)
        states = self.output_norm(states + self.feedforward(states))
        similarity = F.cosine_similarity(states.float(), self.stop_token.float()[None, None, :], dim=-1)
        return {
            "embeddings": self.merger_output(states),
            "stop_similarity": similarity,
            "stop_logits": (similarity - cfg.stop_threshold) * cfg.stop_logit_scale,
        }

    def checkpoint_config(self) -> dict:
        return asdict(self.config)
