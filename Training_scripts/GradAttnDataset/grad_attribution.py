"""Gradient-weighted decoder-to-vision attribution for Qwen3-VL.

This is the decoder-attention adaptation of Grad-Att from ViCrop:

    A_grad = A * ReLU(d log p(next_token) / d A)

``A`` is the post-softmax attention of an individual Qwen text-decoder layer,
and only the columns that correspond to the immutable full-image visual tokens
are retained.  The class deliberately does not mutate model weights.
"""
from __future__ import annotations

import copy

import numpy as np


class DecoderGradAttentionCapture:
    """Retain eager-attention tensors and expose Grad-Att maps after backward.

    The caller must execute one forward with gradients enabled, backpropagate
    the log probability of its selected next token, then call ``maps`` before
    leaving this context.  One instance represents one decoder query step.
    """

    def __init__(self, model, visual_positions, grid: tuple[int, int], *, layer_start: int = 0):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        modules = {
            module.layer_idx: module
            for module in model.modules()
            if isinstance(module, Qwen3VLTextAttention) and module.layer_idx >= layer_start
        }
        if not modules:
            raise ValueError("No Qwen3VLTextAttention modules selected for Grad-Att capture.")
        self.modules = modules
        self.visual_positions = visual_positions
        self.grid = tuple(map(int, grid))
        self.captured: dict[int, object] = {}
        self.configs = {}
        self.handles = []

    def __enter__(self):
        for layer, module in self.modules.items():
            # Qwen's eager implementation exposes post-softmax weights.  Do
            # this per module so another caller's global config is restored.
            self.configs[layer] = module.config
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "eager"
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer: int):
        def capture(_module, _args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("Expected eager attention with shape [1, heads, queries, keys].")
            # Frozen NF4 parameters alone do not make intermediates require
            # grad; the collector enables input gradients before each probe.
            if not weights.requires_grad:
                raise RuntimeError(
                    "Attention weights have no gradient graph. Enable input gradients before the Grad-Att forward."
                )
            weights.retain_grad()
            self.captured[layer] = weights
        return capture

    def maps(self, aggregation: str) -> tuple[np.ndarray, np.ndarray]:
        """Return [layers,H,W] and a normalized pooled [H,W] saliency map."""
        per_layer = []
        for layer in sorted(self.modules):
            weights = self.captured.get(layer)
            if weights is None or weights.grad is None:
                raise RuntimeError(f"Layer {layer} did not retain an attention gradient.")
            positions = self.visual_positions.to(weights.device)
            attention = weights[0, :, -1, :].index_select(-1, positions)
            gradient = weights.grad[0, :, -1, :].index_select(-1, positions)
            # Average heads only after each head has supplied its signed causal
            # contribution; negative gradients are deliberately excluded.
            score = (attention.float() * gradient.float().relu()).mean(dim=0)
            per_layer.append(score.detach().cpu().numpy().reshape(self.grid))
        layers = np.stack(per_layer).astype(np.float32, copy=False)
        if aggregation == "last":
            pooled = layers[-1]
        elif aggregation == "mean":
            pooled = layers.mean(axis=0)
        elif aggregation == "max":
            pooled = layers.max(axis=0)
        else:
            raise ValueError(f"Unsupported Grad-Att aggregation: {aggregation!r}")
        pooled = np.maximum(pooled, 0.0)
        total = float(pooled.sum())
        if total > np.finfo(np.float32).tiny:
            pooled = pooled / total
        return layers, pooled.astype(np.float32, copy=False)

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for layer, config in self.configs.items():
            self.modules[layer].config = config
        self.captured.clear()
