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


class PrototypeRoutingModule(nn.Module):
    """Parameter-light image-conditioned prototype routing before Qwen merger.

    P reads image patches (P -> Z), producing P_img, then patches read only
    image-conditioned prototypes (Z -> P_img). There are deliberately no Q/K/V
    projections: the only learned visual-routing parameters are P and alpha.
    """

    def __init__(self, hidden_size: int, num_prototypes: int, alpha_init: float = 0.01,
                 tau1: float = 0.1, tau2: float = 0.1):
        super().__init__()
        if tau1 <= 0 or tau2 <= 0:
            raise ValueError("tau1 and tau2 must be positive")
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, hidden_size) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.num_prototypes = num_prototypes
        self.tau1, self.tau2 = float(tau1), float(tau2)
        self.last_collect_attention: Optional[torch.Tensor] = None
        self.last_distribute_attention: Optional[torch.Tensor] = None
        self.last_image_prototypes: Optional[torch.Tensor] = None
        self.last_residual_relative_norm: Optional[torch.Tensor] = None

    def forward(self, z: torch.Tensor, *, return_maps: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        z_norm = torch.nn.functional.normalize(z.float(), dim=-1).to(z.dtype)
        p_norm = torch.nn.functional.normalize(self.prototypes.float(), dim=-1).to(z.dtype)
        collect_logits = torch.einsum("kd,bnd->bkn", p_norm, z_norm) / self.tau1
        collect_attn = torch.softmax(collect_logits, dim=-1)
        p_img = torch.einsum("bkn,bnd->bkd", collect_attn, z)
        p_img_norm = torch.nn.functional.normalize(p_img.float(), dim=-1).to(z.dtype)
        distribute_logits = torch.einsum("bnd,bkd->bnk", z_norm, p_img_norm) / self.tau2
        distribute_attn = torch.softmax(distribute_logits, dim=-1)
        delta_z = torch.einsum("bnk,bkd->bnd", distribute_attn, p_img)
        z_out = z + self.alpha.to(z.dtype) * delta_z
        with torch.no_grad():
            self.last_residual_relative_norm = (
                (z_out - z).float().norm(dim=-1).mean() /
                z.float().norm(dim=-1).mean().clamp_min(1e-8)
            ).detach()
            self.last_collect_attention = collect_attn.detach() if return_maps else None
            self.last_distribute_attention = distribute_attn.detach() if return_maps else None
            self.last_image_prototypes = p_img.detach() if return_maps else None
        if return_maps:
            return {"z_out": z_out, "collect_attn": collect_attn,
                    "distribute_attn": distribute_attn, "image_prototypes": p_img}
        return z_out


class VisionPrototypeHook:
    """Injects the module exactly before Qwen's final vision-language merger."""

    def __init__(self, visual: nn.Module, prototype: PrototypeRoutingModule):
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
            routed = self.prototype(z[start:start + length].unsqueeze(0),
                                     return_maps=self.capture_attention)
            pieces.append((routed["z_out"] if self.capture_attention else routed).squeeze(0))
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
