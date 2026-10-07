"""The only trainable vision-side addition used by Prototype PrefSFT.

The merger pre-hook is deliberately installed on ``model.visual.merger``: it
therefore receives the final ViT states, and prototype tokens can never reach
Qwen's spatial merger.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from torch import nn


class PrototypeTransformer(nn.Module):
    """One pre-norm Transformer block operating on [vision, prototypes]."""

    def __init__(self, hidden_size: int, num_prototypes: int, num_heads: int = 16,
                 mlp_ratio: float = 4.0, alpha_init: float = 0.01):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by heads={num_heads}")
        self.prototypes = nn.Parameter(torch.randn(1, num_prototypes, hidden_size) * 0.02)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)), nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.num_prototypes = num_prototypes
        self.last_attention: Optional[torch.Tensor] = None
        self.last_residual_relative_norm: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor, *, retain_attention: bool = False) -> torch.Tensor:
        # z is [images, patches, D]; P exists only within this method.
        p = self.prototypes.expand(z.shape[0], -1, -1).to(dtype=z.dtype)
        x = torch.cat((z, p), dim=1)
        h = self.norm1(x)
        attended, weights = self.attn(h, h, h, need_weights=retain_attention,
                                      average_attn_weights=False)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        z_prime = x[:, :z.shape[1]]
        z_out = z + self.alpha.to(z.dtype) * (z_prime - z)
        with torch.no_grad():
            self.last_residual_relative_norm = (
                (z_out - z).float().norm(dim=-1).mean() /
                z.float().norm(dim=-1).mean().clamp_min(1e-8)
            ).detach()
            self.last_attention = weights.detach() if retain_attention else None
        return z_out


class VisionPrototypeHook:
    """Injects the module exactly before Qwen's final vision-language merger."""

    def __init__(self, visual: nn.Module, prototype: PrototypeTransformer):
        self.visual, self.prototype = visual, prototype
        if not hasattr(visual, "merger"):
            raise AttributeError("Expected Qwen visual.merger; unsupported Qwen architecture.")
        self.grid_thw: Optional[torch.Tensor] = None
        self.capture_attention = False
        self.handle = visual.merger.register_forward_pre_hook(self._before_merger)

    def _before_merger(self, module: nn.Module, args: tuple[torch.Tensor, ...]):
        z = args[0]
        if self.grid_thw is None:
            raise RuntimeError("Prototype merger hook received no image_grid_thw context.")
        lengths = self.grid_thw.to("cpu", torch.long).prod(dim=-1).tolist()
        if sum(lengths) != z.shape[0]:
            raise RuntimeError(
                f"ViT token/grid mismatch: merger received {z.shape[0]}, grid implies {sum(lengths)}. "
                "This transformers/Qwen version requires an architecture-specific adapter."
            )
        pieces, start = [], 0
        for length in lengths:
            pieces.append(self.prototype(z[start:start + length].unsqueeze(0),
                                         retain_attention=self.capture_attention).squeeze(0))
            start += length
        return (torch.cat(pieces, dim=0), *args[1:])

    @contextmanager
    def context(self, image_grid_thw: torch.Tensor, capture_attention: bool = False):
        old_grid, old_capture = self.grid_thw, self.capture_attention
        self.grid_thw, self.capture_attention = image_grid_thw, capture_attention
        try:
            yield
        finally:
            self.grid_thw, self.capture_attention = old_grid, old_capture

    def remove(self):
        self.handle.remove()


class CachedVisionForward:
    """Persistent cache for frozen final-ViT states, before the final merger.

    On a cache hit this replaces ``visual.forward`` only for the current model
    call. It still calls ``visual.merger`` so the prototype pre-hook remains in
    exactly the intended location. Qwen3-VL variants returning deepstack
    features are supported by caching/replaying their second return value.
    """
    def __init__(self, visual: nn.Module, cache_dir: str | Path):
        self.visual, self.root = visual, Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.original_forward = visual.forward
        self.keys: list[str] | None = None
        self.capture_key: str | None = None
        self._captured_z: torch.Tensor | None = None
        self.capture_handle = visual.merger.register_forward_pre_hook(self._capture_before_merger)
        visual.forward = self._forward  # Module.__call__ invokes this closure directly.

    def _path(self, key: str) -> Path:
        import hashlib
        return self.root / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".pt")

    def has(self, key: str) -> bool:
        return self._path(key).is_file()

    def _capture_before_merger(self, module, args):
        if self.capture_key is not None:
            self._captured_z = args[0].detach().cpu()

    @staticmethod
    def _cpu(value):
        if torch.is_tensor(value): return value.detach().cpu()
        if isinstance(value, tuple): return tuple(CachedVisionForward._cpu(x) for x in value)
        if isinstance(value, list): return [CachedVisionForward._cpu(x) for x in value]
        return value

    @staticmethod
    def _cat(values, device, dtype):
        first = values[0]
        if torch.is_tensor(first):
            return torch.cat([x.to(device=device, dtype=dtype) for x in values], dim=0)
        if isinstance(first, tuple):
            return tuple(CachedVisionForward._cat([x[i] for x in values], device, dtype) for i in range(len(first)))
        if isinstance(first, list):
            return [CachedVisionForward._cat([x[i] for x in values], device, dtype) for i in range(len(first))]
        return first

    def _forward(self, *args, **kwargs):
        if self.keys is not None:
            entries = [torch.load(self._path(key), map_location="cpu") for key in self.keys]
            z = self._cat([x["z"] for x in entries], next(self.visual.merger.parameters()).device,
                          next(self.visual.merger.parameters()).dtype)
            merged = self.visual.merger(z)
            auxiliary = self._cat([x["auxiliary"] for x in entries], z.device, z.dtype)
            return merged if auxiliary is None else (merged, *auxiliary)
        output = self.original_forward(*args, **kwargs)
        if self.capture_key is not None:
            if self._captured_z is None:
                raise RuntimeError("Did not observe final ViT states at visual.merger while building cache.")
            auxiliary = output[1:] if isinstance(output, tuple) else None
            temporary = self._path(self.capture_key).with_suffix(".tmp")
            torch.save({"z": self._captured_z, "auxiliary": self._cpu(auxiliary)}, temporary)
            temporary.replace(self._path(self.capture_key))
            self._captured_z = None
        return output

    @contextmanager
    def use(self, keys: list[str]):
        missing = [key for key in keys if not self.has(key)]
        if missing: raise FileNotFoundError(f"Vision cache miss for {missing[0]!r}; run precompute_vision_embeddings.py first.")
        old = self.keys; self.keys = keys
        try: yield
        finally: self.keys = old

    @contextmanager
    def capture(self, key: str):
        old = self.capture_key; self.capture_key = key
        try: yield
        finally: self.capture_key = old

    def remove(self):
        self.visual.forward = self.original_forward
        self.capture_handle.remove()
