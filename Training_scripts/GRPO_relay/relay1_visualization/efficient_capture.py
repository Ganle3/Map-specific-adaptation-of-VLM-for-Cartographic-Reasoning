"""Exact Relay1 query attention reconstruction while the model keeps SDPA."""
from __future__ import annotations

import numpy as np
import torch


class EfficientRelayPrefillCapture:
    """Capture relay-query attention without materializing full SxS attention.

    Native q_proj/k_proj outputs and the exact MRoPE tensors are observed by
    hooks.  We reconstruct only two relay query rows against all prompt keys.
    This is algebraically the same score/softmax used by eager attention, while
    the actual decoder forward remains on memory-efficient SDPA.
    """

    def __init__(self, model, image_positions, relay_positions, grid):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        self.image_positions = image_positions.flatten()
        self.relay_positions = relay_positions.flatten()
        self.grid = tuple(map(int, grid))
        self.modules = {
            int(module.layer_idx): module for module in model.modules()
            if isinstance(module, Qwen3VLTextAttention)
        }
        self.layers = tuple(sorted(self.modules))
        if self.layers != tuple(range(len(self.layers))):
            raise RuntimeError("Expected contiguous Qwen decoder layers")
        self._states = {layer: {} for layer in self.layers}
        self._maps = [None] * len(self.layers)
        self._handles = []

    def __enter__(self):
        for layer, module in self.modules.items():
            self._handles.extend([
                module.register_forward_pre_hook(self._pre(layer), with_kwargs=True),
                module.q_proj.register_forward_hook(self._projection(layer, "q")),
                module.k_proj.register_forward_hook(self._projection(layer, "k")),
                module.register_forward_hook(self._post(layer)),
            ])
        return self

    def _pre(self, layer):
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            active = hidden is not None and hidden.shape[1] > int(self.relay_positions.max())
            self._states[layer] = {"active": active}
            if active:
                self._states[layer]["position_embeddings"] = kwargs["position_embeddings"]
                self._states[layer]["attention_mask"] = kwargs.get("attention_mask")
        return hook

    def _projection(self, layer, name):
        def hook(_module, _args, output):
            if self._states[layer].get("active"):
                self._states[layer][name] = output
        return hook

    def _post(self, layer):
        def hook(module, _args, output):
            state = self._states[layer]
            if not state.get("active"):
                return
            if self._maps[layer] is not None:
                raise RuntimeError(f"Layer {layer} encountered two full prefills")
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                apply_rotary_pos_emb, repeat_kv,
            )
            q_raw, k_raw = state["q"], state["k"]
            shape = (*q_raw.shape[:-1], -1, module.head_dim)
            query = module.q_norm(q_raw.view(shape)).transpose(1, 2)
            shape = (*k_raw.shape[:-1], -1, module.head_dim)
            key = module.k_norm(k_raw.view(shape)).transpose(1, 2)
            cos, sin = state["position_embeddings"]
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            key = repeat_kv(key, module.num_key_value_groups)
            relay = self.relay_positions.to(query.device)
            query = query.index_select(2, relay)
            scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            mask = state["attention_mask"]
            if mask is not None:
                mask = mask[:, :, relay, :key.shape[-2]]
                scores = scores + mask
            else:
                key_index = torch.arange(key.shape[-2], device=query.device)
                allowed = key_index.view(1, 1, 1, -1) <= relay.view(1, 1, -1, 1)
                scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1)
            image = self.image_positions.to(weights.device)
            selected = weights.index_select(-1, image).mean(dim=1)[0]
            self._maps[layer] = selected.detach().cpu().numpy().reshape(
                self.relay_positions.numel(), *self.grid
            ).astype(np.float32)
            self._states[layer] = {}
        return hook

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()

    def arrays(self):
        missing = [i for i, value in enumerate(self._maps) if value is None]
        if missing:
            raise RuntimeError(f"Missing reconstructed layers: {missing}")
        return np.stack(self._maps).transpose(1, 0, 2, 3).astype(np.float32)

