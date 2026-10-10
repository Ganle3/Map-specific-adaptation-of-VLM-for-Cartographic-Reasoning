"""Native eager-attention capture for one or more Qwen3-VL image inputs."""
from __future__ import annotations

import copy

import numpy as np


class MultiImageAttentionCapture:
    """Collect actual decoder attention to each supplied image independently.

    The output is deliberately per image and per decoder layer.  It supports
    rolling focus-only inputs, where a window may contain only crops and no
    full image.
    """

    def __init__(self, model, image_positions: list, grids: list[tuple[int, int]]):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        self.modules = {m.layer_idx: m for m in model.modules() if isinstance(m, Qwen3VLTextAttention)}
        if not self.modules or len(image_positions) != len(grids):
            raise ValueError("Need decoder layers and matching image position/grid metadata.")
        self.positions = image_positions
        self.grids = grids
        self.layers = tuple(sorted(self.modules))
        self.raw = {layer: [np.zeros(grid, np.float64) for grid in grids] for layer in self.layers}
        self.conditional = {layer: [np.zeros(grid, np.float64) for grid in grids] for layer in self.layers}
        self.counts = {layer: 0 for layer in self.layers}
        self.handles = []
        self.configs = {}

    def __enter__(self):
        for layer, module in self.modules.items():
            self.configs[layer] = module.config
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "eager"
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        def capture(module, args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("Expected eager attention [1, heads, queries, keys].")
            # Last query is the token actually used to predict the next token.
            for index, (positions, grid) in enumerate(zip(self.positions, self.grids)):
                values = weights[0, :, -1, :].index_select(-1, positions.to(weights.device))
                spatial = values.mean(dim=0).detach().float().cpu().numpy().reshape(grid)
                self.raw[layer][index] += spatial
                self.conditional[layer][index] += spatial / max(float(spatial.sum()), np.finfo(np.float32).tiny)
            self.counts[layer] += 1
        return capture

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for layer, config in self.configs.items():
            self.modules[layer].config = config

    def summaries(self, steps: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if steps < 1 or any(count != steps for count in self.counts.values()):
            raise RuntimeError("Attention hook and generation steps are not aligned.")
        raw = [np.stack([self.raw[layer][image] / steps for layer in self.layers]).astype(np.float32)
               for image in range(len(self.grids))]
        conditional = [np.stack([self.conditional[layer][image] / steps for layer in self.layers]).astype(np.float32)
                       for image in range(len(self.grids))]
        return raw, conditional
