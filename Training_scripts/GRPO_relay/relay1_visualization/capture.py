"""Capture each fixed Relay1 token's prefill attention to image patches."""
from __future__ import annotations

import copy

import numpy as np


class RelayPrefillAttentionCapture:
    """Capture ``[relay, layer, patch_y, patch_x]`` post-softmax attention.

    Relay1 positions occur inside the full prefill sequence.  The hooks select
    those exact query rows and retain only image-token key columns.  Heads are
    averaged, while relay tokens and decoder layers remain separate.
    """

    def __init__(self, model, image_positions, relay_positions,
                 grid: tuple[int, int]):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        self.image_positions = image_positions.flatten()
        self.relay_positions = relay_positions.flatten()
        self.grid = tuple(map(int, grid))
        self.modules = {
            int(module.layer_idx): module
            for module in model.modules()
            if isinstance(module, Qwen3VLTextAttention)
        }
        self.layers = tuple(sorted(self.modules))
        if not self.layers or self.layers != tuple(range(len(self.layers))):
            raise RuntimeError("Expected contiguous decoder attention layers from zero")
        if self.relay_positions.numel() < 1:
            raise RuntimeError("No relay query positions were supplied")
        self._maps: list[np.ndarray | None] = [None] * len(self.layers)
        self._handles = []
        self._configs = {}

    def __enter__(self):
        # SDPA does not expose post-softmax attention probabilities.
        for layer, module in self.modules.items():
            self._configs[layer] = module.config
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "eager"
            self._handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer: int):
        def capture(_module, _args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("Expected eager weights [1, heads, queries, keys]")
            # generate_from_relays performs one incremental single-token
            # decoder call after sampling the final requested token.  That
            # call contains no Relay1 query row and its logits are unused.
            if weights.shape[-2] <= int(self.relay_positions.max()):
                if weights.shape[-2] == 1:
                    return
                raise RuntimeError("A non-prefill call cannot address the relay query positions")
            if self._maps[layer] is not None:
                # There must be exactly one full-sequence prefill.
                raise RuntimeError(f"Decoder layer {layer} was captured more than once")
            image_positions = self.image_positions.to(weights.device)
            relay_positions = self.relay_positions.to(weights.device)
            if int(image_positions.max()) >= weights.shape[-1]:
                raise RuntimeError("Image position is outside the attention key axis")
            if int(relay_positions.max()) >= weights.shape[-2]:
                raise RuntimeError("Relay position is outside the attention query axis")
            # [heads, relays, keys] -> [relays, image keys], preserving R1/R2.
            selected = weights[0].index_select(1, relay_positions)
            selected = selected.index_select(2, image_positions).mean(dim=0)
            self._maps[layer] = (
                selected.detach().float().cpu().numpy().reshape(
                    self.relay_positions.numel(), *self.grid
                ).astype(np.float32)
            )
        return capture

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        for layer, config in self._configs.items():
            self.modules[layer].config = config

    def arrays(self) -> np.ndarray:
        missing = [index for index, value in enumerate(self._maps) if value is None]
        if missing:
            raise RuntimeError(f"Missing decoder attention layers: {missing}")
        # stack layers [L,R,H,W], then expose requested [R,L,H,W].
        return np.stack(self._maps, axis=0).transpose(1, 0, 2, 3).astype(np.float32)
