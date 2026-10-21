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

    def __init__(self, model, image_positions: list, grids: list[tuple[int, int]],
                 score_mode: str = "attention"):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        self.modules = {m.layer_idx: m for m in model.modules() if isinstance(m, Qwen3VLTextAttention)}
        if not self.modules or len(image_positions) != len(grids):
            raise ValueError("Need decoder layers and matching image position/grid metadata.")
        self.positions = image_positions
        self.grids = grids
        if score_mode not in {"attention", "value_norm", "projected_value_norm"}:
            raise ValueError(f"Unknown visual score mode: {score_mode}")
        self.score_mode = score_mode
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
            # Value vectors are available only inside eager_attention_forward;
            # that path calls record_eager below.  Raw attention can retain the
            # lighter forward-hook implementation.
            if self.score_mode == "attention":
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

    def record_eager(self, layer, module, weights, value):
        """Record a value-aware patch score from Qwen eager-attention inputs.

        ``value`` is the cached KV-head tensor; repeat_kv expands it to query
        heads exactly as Qwen's eager attention implementation does.
        """
        if self.score_mode == "attention":
            return
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        value_states = repeat_kv(value, module.num_key_value_groups)[0]
        for index, (positions, grid) in enumerate(zip(self.positions, self.grids)):
            patch_values = value_states.index_select(1, positions.to(value_states.device))
            patch_attention = weights[0, :, -1, :].index_select(-1, positions.to(weights.device))
            if self.score_mode == "value_norm":
                score = patch_attention * patch_values.float().norm(dim=-1)
            else:
                # o_proj.weight has [hidden_size, num_heads * head_dim].  The
                # norm is computed in chunks to avoid materialising
                # [heads, patches, hidden_size] for a whole image at once.
                heads, _, dim = patch_values.shape
                out_dim = module.o_proj.out_features
                in_dim = module.o_proj.in_features
                weight = module.o_proj.weight
                # bitsandbytes Linear4bit stores packed nibbles in ``weight``.
                # Its storage shape/numel is therefore not the logical
                # [out_features, in_features] matrix.  Treating it as BF16
                # caused the NF4-only reshape failure during a WHERE probe.
                # Dequantize exactly this frozen output projection for the
                # one probe forward, then retain the original A * ||V W_O||
                # attribution definition.  Normal BF16/FP weights take the
                # inexpensive direct branch.
                if weight.numel() != out_dim * in_dim:
                    try:
                        import bitsandbytes.functional as bnb_functional
                        matrix = bnb_functional.dequantize_4bit(
                            weight.data, quant_state=weight.quant_state
                        )
                    except (AttributeError, ImportError) as exc:
                        raise RuntimeError(
                            "projected_value_norm requires access to the NF4 quantization state "
                            "of decoder o_proj."
                        ) from exc
                else:
                    matrix = weight
                if matrix.numel() != out_dim * in_dim:
                    raise RuntimeError(
                        f"Unexpected o_proj matrix size after dequantization: got {matrix.numel()}, "
                        f"expected {out_dim * in_dim}."
                    )
                projection = matrix.float().view(out_dim, heads, dim).permute(1, 2, 0)
                norms = []
                for chunk in patch_values.float().split(64, dim=1):
                    projected = torch.einsum("hpd,hdo->hpo", chunk, projection)
                    norms.append(projected.norm(dim=-1))
                score = patch_attention * torch.cat(norms, dim=1)
            spatial = score.mean(dim=0).detach().float().cpu().numpy().reshape(grid)
            self.raw[layer][index] += spatial
            self.conditional[layer][index] += spatial / max(float(spatial.sum()), np.finfo(np.float32).tiny)
        self.counts[layer] += 1

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
