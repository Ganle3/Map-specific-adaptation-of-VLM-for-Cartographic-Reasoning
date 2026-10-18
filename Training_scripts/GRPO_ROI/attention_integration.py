"""Qwen eager-attention bridge for a per-decoding-step visual-key bias.

This deliberately patches only ``eager_attention_forward``.  It raises if
eager Qwen text attention is unavailable, rather than silently using a backend
which cannot observe the supplied additive logits.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import copy
import torch


class DynamicVisualKeyBias(AbstractContextManager):
    """Apply one [visual_tokens] bias vector to every Qwen decoder layer."""
    def __init__(self, model, visual_positions: torch.Tensor):
        self.model, self.visual_positions = model, visual_positions.long()
        self.current_bias: torch.Tensor | None = None
        self._module = self._original = None
        self._configs: list[tuple[object, object]] = []
        self._step_attention: list[torch.Tensor] = []

    def set_step_bias(self, bias: torch.Tensor | None) -> None:
        if bias is not None and bias.numel() != self.visual_positions.numel():
            raise ValueError("ROI bias length must equal the global visual-token count.")
        self.current_bias = bias
        self._step_attention.clear()

    def previous_attention_summary(self) -> torch.Tensor | None:
        """Mean layers/heads visual attention from the last executed token."""
        if not self._step_attention:
            return None
        return torch.stack(self._step_attention).mean(0)

    def __enter__(self):
        import transformers.models.qwen3_vl.modeling_qwen3_vl as qwen
        if not hasattr(qwen, "eager_attention_forward"):
            raise RuntimeError("Installed Transformers has no Qwen eager-attention hook.")
        for module in self.model.modules():
            if isinstance(module, qwen.Qwen3VLTextAttention):
                self._configs.append((module, module.config))
                module.config = copy.copy(module.config)
                module.config._attn_implementation = "eager"
        if not self._configs:
            raise RuntimeError("No Qwen3VLTextAttention modules found.")
        self._module, self._original = qwen, qwen.eager_attention_forward
        original, positions = self._original, self.visual_positions

        def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            if isinstance(module, qwen.Qwen3VLTextAttention) and self.current_bias is not None:
                live = positions.to(key.device)
                if int(live.max()) >= key.shape[-2]:
                    raise RuntimeError("Visual key positions do not fit the current KV cache.")
                additive = query.new_zeros((1, 1, 1, key.shape[-2]))
                additive[..., live] = self.current_bias.to(key.device, query.dtype)
                attention_mask = additive if attention_mask is None else attention_mask + additive
            result = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if isinstance(module, qwen.Qwen3VLTextAttention) and self.current_bias is not None:
                weights = result[1]
                if weights is None:
                    raise RuntimeError("Eager attention returned no weights; ROI summary cannot be formed.")
                self._step_attention.append(weights[0, :, -1, :].index_select(-1, positions.to(weights.device)).mean(0).detach())
            return result
        qwen.eager_attention_forward = patched
        return self

    def __exit__(self, *exc):
        if self._module is not None:
            self._module.eager_attention_forward = self._original
        for module, config in self._configs:
            module.config = config
