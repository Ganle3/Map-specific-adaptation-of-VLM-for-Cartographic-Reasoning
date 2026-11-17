"""Compact exact-Q/K statistics for native greedy decoding.

The capture never asks attention to return its full weight matrix.  It rebuilds
the model's post-RoPE language query and reads the actual cached image keys.
This keeps SDPA enabled and makes population-scale capture practical.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


METRIC_NAMES = (
    "nearest_key_distance",
    "top5pct_key_distance",
    "mean_key_distance",
    "attention_weighted_key_distance",
    "image_attention_mass",
)


class CompactQKCapture:
    """Capture hidden-query/image-key alignment at every decoding step.

    Distances are one minus mean-head cosine.  The attention-weighted distance
    uses the exact full-cache softmax for the language row and then conditions
    those weights on image keys.  Fixed-step projected Q and sampled K vectors
    are retained for population cluster plots; full patch maps are not stored.
    """

    def __init__(self, model, image_positions: torch.Tensor, *, query_kind: str,
                 projection_dim: int = 64, key_samples: int = 32,
                 projection_seed: int = 20261117,
                 projection_steps: tuple[int, ...] = (0, 4, 9, 19, 39, 79, 159, 319)):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        if query_kind not in {"adapter", "relay"}:
            raise ValueError(query_kind)
        if projection_dim < 2 or key_samples < 1:
            raise ValueError("projection_dim and key_samples must be positive")
        self.query_kind = query_kind
        self.image_positions = image_positions.detach().long()
        self.projection_dim = int(projection_dim)
        self.key_samples = int(key_samples)
        self.projection_seed = int(projection_seed)
        self.projection_steps = tuple(sorted(set(map(int, projection_steps))))
        self.modules = {
            int(module.layer_idx): module for module in model.modules()
            if isinstance(module, Qwen3VLTextAttention)
        }
        self.layers = tuple(sorted(self.modules))
        if self.layers != tuple(range(len(self.layers))):
            raise RuntimeError("Expected contiguous decoder layers starting at zero")
        self._handles = []
        self._query: dict[int, torch.Tensor] = {}
        self._cache: dict[int, object] = {}
        self._current: list[np.ndarray] | None = None
        self._step = 0
        self.metric_steps: list[np.ndarray] = []
        self.projected_queries: dict[int, list[np.ndarray]] = {}
        self.projected_keys: dict[int, np.ndarray] = {}
        self.sampled_image_offsets: np.ndarray | None = None
        self._projection_cpu: torch.Tensor | None = None

    def __enter__(self):
        for layer, module in self.modules.items():
            self._handles.append(module.register_forward_pre_hook(
                self._pre_hook(layer), with_kwargs=True
            ))
            self._handles.append(module.register_forward_hook(self._post_hook(layer)))
        return self

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _pre_hook(self, layer: int):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get(
                "position_embeddings", args[1] if len(args) > 1 else None
            )
            past = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or position_embeddings is None or past is None:
                raise RuntimeError("Could not recover Qwen attention inputs")
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            cos, sin = position_embeddings
            query, _ = apply_rotary_pos_emb(query, key, cos, sin)
            self._query[layer] = query.detach()
            self._cache[layer] = past
        return capture

    def _projection(self, hidden_size: int, device, dtype) -> torch.Tensor:
        if self._projection_cpu is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.projection_seed)
            signs = torch.randint(0, 2, (hidden_size, self.projection_dim),
                                  generator=generator, dtype=torch.int8)
            self._projection_cpu = (signs.float().mul_(2).sub_(1)
                                    / math.sqrt(self.projection_dim))
        if self._projection_cpu.shape[0] != hidden_size:
            raise RuntimeError("Hidden size changed during capture")
        return self._projection_cpu.to(device=device, dtype=dtype)

    @staticmethod
    def _flatten_head_unit(x: torch.Tensor) -> torch.Tensor:
        # Unit-normalize each head and scale so a flattened dot product equals
        # the mean cosine across attention heads.
        x = F.normalize(x.float(), dim=-1)
        return x.flatten(-2) / math.sqrt(x.shape[-2])

    def _sample_offsets(self, patch_count: int, device) -> torch.Tensor:
        count = min(self.key_samples, patch_count)
        offsets = torch.linspace(0, patch_count - 1, count, device=device).round().long()
        if self.sampled_image_offsets is None:
            self.sampled_image_offsets = offsets.cpu().numpy().astype(np.int32)
        return offsets

    def _metrics(self, module, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        query = self._query.pop(layer)
        cache = self._cache.pop(layer)
        cache_layer = cache.layers[layer]
        if not cache_layer.is_initialized:
            raise RuntimeError(f"Cache layer {layer} was not initialized")
        # Adapter decoding uses the last prompt/query row; Relay3 places x_t at row 0.
        row = 0 if self.query_kind == "relay" else -1
        q = query[0, :, row, :]
        all_keys = repeat_kv(cache_layer.keys, module.num_key_value_groups)[0]
        image_positions = self.image_positions.to(all_keys.device)
        image_keys = all_keys.index_select(-2, image_positions)
        cosine = F.cosine_similarity(q[:, None, :].float(), image_keys.float(), dim=-1)
        nearest_similarity = cosine.max(dim=-1).values.mean()
        top_count = max(1, math.ceil(cosine.shape[-1] * 0.05))
        top_similarity = cosine.topk(top_count, dim=-1).values.mean()
        mean_similarity = cosine.mean()

        logits = torch.matmul(q.float().unsqueeze(1),
                              all_keys.float().transpose(-2, -1)).squeeze(1)
        logits.mul_(float(module.scaling))
        full_weights = logits.softmax(dim=-1)
        image_weights = full_weights.index_select(-1, image_positions)
        image_mass = image_weights.sum(dim=-1).mean()
        conditional = image_weights / image_weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        weighted_similarity = (conditional * cosine).sum(dim=-1).mean()
        metrics = torch.stack((
            1 - nearest_similarity,
            1 - top_similarity,
            1 - mean_similarity,
            1 - weighted_similarity,
            image_mass,
        )).detach().cpu().numpy().astype(np.float32)

        projection = self._projection(q.numel(), q.device, torch.float32)
        q_projected = (self._flatten_head_unit(q).float() @ projection).detach()
        offsets = self._sample_offsets(image_keys.shape[-2], image_keys.device)
        sampled = image_keys.index_select(-2, offsets).transpose(0, 1)
        k_projected = (self._flatten_head_unit(sampled).float() @ projection).detach()
        return metrics, q_projected.cpu().numpy(), k_projected.cpu().numpy()

    def _post_hook(self, layer: int):
        def capture(module, _args, _output):
            if layer == 0:
                if self._current is not None:
                    raise RuntimeError("Previous decoder step was incomplete")
                self._current = []
            if self._current is None:
                raise RuntimeError("Decoder hooks did not start at layer zero")
            metrics, projected_q, projected_k = self._metrics(module, layer)
            self._current.append(metrics)
            if self._step in self.projection_steps:
                self.projected_queries.setdefault(self._step, []).append(projected_q)
            if layer not in self.projected_keys:
                self.projected_keys[layer] = projected_k.astype(np.float16)
            if layer == self.layers[-1]:
                if len(self._current) != len(self.layers):
                    raise RuntimeError("Decoder step skipped a layer")
                self.metric_steps.append(np.stack(self._current).astype(np.float32))
                self._current = None
                self._step += 1
        return capture

    def arrays(self, expected_steps: int, *, allow_one_unused_tail: bool = False) -> dict[str, np.ndarray]:
        if self._current is not None:
            raise RuntimeError("Capture ended inside a decoder step")
        if allow_one_unused_tail and len(self.metric_steps) == expected_steps + 1:
            self.metric_steps.pop()
            unused = max(self.projected_queries, default=-1)
            if unused >= expected_steps:
                self.projected_queries.pop(unused, None)
        if len(self.metric_steps) != expected_steps:
            raise RuntimeError(f"Captured {len(self.metric_steps)} steps, expected {expected_steps}")
        captured_steps = sorted(step for step in self.projected_queries if step < expected_steps)
        query_rows = []
        for step in captured_steps:
            rows = self.projected_queries[step]
            if len(rows) != len(self.layers):
                raise RuntimeError(f"Projected query step {step} is incomplete")
            query_rows.append(np.stack(rows))
        key_rows = [self.projected_keys[layer] for layer in self.layers]
        return {
            "metrics": np.stack(self.metric_steps).astype(np.float32),
            "metric_names": np.asarray(METRIC_NAMES),
            "projection_steps": np.asarray(captured_steps, dtype=np.int32),
            "query_projection": (np.stack(query_rows).astype(np.float16)
                                 if query_rows else np.empty((0, len(self.layers), self.projection_dim), np.float16)),
            "key_projection": np.stack(key_rows).astype(np.float16),
            "sampled_image_offsets": self.sampled_image_offsets,
        }

