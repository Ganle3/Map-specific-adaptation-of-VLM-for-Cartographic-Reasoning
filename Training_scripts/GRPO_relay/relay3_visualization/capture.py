"""Capture real decoder query-to-image attention during greedy inference."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


class ImageAttentionCapture:
    """Capture head-mean image attention for every step and decoder layer.

    ``query_kind='adapter'`` captures the last/only language query row.
    ``query_kind='relay'`` captures row 0 (x_t) and averages rows 1..R (r_t).
    Hooks observe the exact post-softmax attention weights used to mix values.
    """

    def __init__(self, model, image_positions, grid: tuple[int, int], *, query_kind: str):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        if query_kind not in {"adapter", "relay"}:
            raise ValueError(f"Unknown query_kind={query_kind!r}")
        self.query_kind = query_kind
        self.grid = tuple(map(int, grid))
        self.positions = image_positions
        self.modules = {
            int(module.layer_idx): module
            for module in model.modules()
            if isinstance(module, Qwen3VLTextAttention)
        }
        self.layers = tuple(sorted(self.modules))
        if not self.layers or self.layers != tuple(range(len(self.layers))):
            raise RuntimeError("Expected contiguous decoder attention layers starting at zero")
        self.hidden_steps: list[np.ndarray] = []
        self.relay_steps: list[np.ndarray] = []
        self.hidden_qk_cosine_steps: list[np.ndarray] = []
        self.hidden_qk_logit_steps: list[np.ndarray] = []
        self.hidden_v_contribution_steps: list[np.ndarray] = []
        self.relay_qk_cosine_steps: list[np.ndarray] = []
        self.relay_qk_logit_steps: list[np.ndarray] = []
        self.relay_v_contribution_steps: list[np.ndarray] = []
        self._hidden_current: list[np.ndarray] | None = None
        self._relay_current: list[np.ndarray] | None = None
        self._hidden_qk_cosine_current: list[np.ndarray] | None = None
        self._hidden_qk_logit_current: list[np.ndarray] | None = None
        self._hidden_v_contribution_current: list[np.ndarray] | None = None
        self._relay_qk_cosine_current: list[np.ndarray] | None = None
        self._relay_qk_logit_current: list[np.ndarray] | None = None
        self._relay_v_contribution_current: list[np.ndarray] | None = None
        self._query_by_layer: dict[int, torch.Tensor] = {}
        self._cache_by_layer: dict[int, object] = {}
        self._handles = []
        self._configs = {}

    def __enter__(self):
        # Eager is required because SDPA does not return attention weights.
        for layer, module in self.modules.items():
            self._configs[layer] = module.config
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "eager"
            self._handles.append(module.register_forward_pre_hook(
                self._pre_hook(layer), with_kwargs=True
            ))
            self._handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _pre_hook(self, layer: int):
        """Rebuild the exact post-RoPE query used by Qwen text attention.

        Image K/V are read from the cache after the attention forward. That is
        the same K/V tensor consumed by eager attention, including RoPE on K.
        """
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

        def capture(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get(
                "position_embeddings", args[1] if len(args) > 1 else None
            )
            past = kwargs.get("past_key_values", args[3] if len(args) > 3 else None)
            if hidden is None or position_embeddings is None or past is None:
                raise RuntimeError("Could not recover attention inputs for post-RoPE Q/K capture")
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
            # apply_rotary_pos_emb rotates Q and K together. Recomputing the
            # current K here avoids duplicating its private rotary formula; the
            # cached K used for metrics is still the model's actual cache tensor.
            key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
            cos, sin = position_embeddings
            query, _ = apply_rotary_pos_emb(query, key, cos, sin)
            self._query_by_layer[layer] = query.detach()
            self._cache_by_layer[layer] = past
        return capture

    def _qkv_maps(self, module, weights: torch.Tensor, layer: int,
                  rows: slice | int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return post-RoPE QK cosine/logit and pre-O image-V contribution."""
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

        query = self._query_by_layer[layer]
        cache = self._cache_by_layer[layer]
        cache_layer = cache.layers[layer]
        if not cache_layer.is_initialized:
            raise RuntimeError(f"Decoder cache layer {layer} is not initialized")
        positions = self.positions.to(cache_layer.keys.device)
        keys = repeat_kv(cache_layer.keys.index_select(-2, positions),
                         module.num_key_value_groups)
        values = repeat_kv(cache_layer.values.index_select(-2, positions),
                           module.num_key_value_groups)
        selected_q = query[:, :, rows, :]
        if selected_q.ndim == 3:
            selected_q = selected_q.unsqueeze(2)
        # [batch, heads, queries, image_patches]
        cosine = F.cosine_similarity(
            selected_q.unsqueeze(-2), keys.unsqueeze(-3), dim=-1
        )
        logits = torch.matmul(selected_q, keys.transpose(-2, -1)) * module.scaling
        selected_weights = weights[:, :, rows, :].index_select(-1, positions)
        if selected_weights.ndim == 3:
            selected_weights = selected_weights.unsqueeze(2)
        # ||concat_h(a_hi * v_hi)||_2 before W_o. This is an additive,
        # patch-resolved magnitude in the exact value stream used by attention.
        value_norm = torch.linalg.vector_norm(values.float(), dim=-1).unsqueeze(2)
        contribution = torch.linalg.vector_norm(
            selected_weights.float() * value_norm, dim=1
        )
        reduce_dims = (0, 1, 2)
        cosine_map = cosine.float().mean(dim=reduce_dims).detach().cpu().numpy().reshape(self.grid)
        logit_map = logits.float().mean(dim=reduce_dims).detach().cpu().numpy().reshape(self.grid)
        contribution_map = contribution.mean(dim=(0, 1)).detach().cpu().numpy().reshape(self.grid)
        return cosine_map, logit_map, contribution_map

    def _hook(self, layer: int):
        def capture(_module, _args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("Expected eager attention weights [1,heads,queries,keys]")
            if layer == 0:
                if self._hidden_current is not None:
                    raise RuntimeError("A new decoder step began before the previous step ended")
                self._hidden_current = []
                self._relay_current = [] if self.query_kind == "relay" else None
                self._hidden_qk_cosine_current = []
                self._hidden_qk_logit_current = []
                self._hidden_v_contribution_current = []
                self._relay_qk_cosine_current = [] if self.query_kind == "relay" else None
                self._relay_qk_logit_current = [] if self.query_kind == "relay" else None
                self._relay_v_contribution_current = [] if self.query_kind == "relay" else None
            if self._hidden_current is None:
                raise RuntimeError("Decoder attention hooks did not begin at layer zero")
            positions = self.positions.to(weights.device)
            if int(positions.max()) >= weights.shape[-1]:
                raise RuntimeError("Image position lies outside the current attention key axis")

            hidden_row = 0 if self.query_kind == "relay" else -1
            hidden = weights[0, :, hidden_row, :].index_select(-1, positions)
            hidden = hidden.mean(dim=0).detach().float().cpu().numpy().reshape(self.grid)
            self._hidden_current.append(hidden)
            hidden_metrics = self._qkv_maps(
                _module, weights, layer, 0 if self.query_kind == "relay" else -1
            )
            assert self._hidden_qk_cosine_current is not None
            assert self._hidden_qk_logit_current is not None
            assert self._hidden_v_contribution_current is not None
            self._hidden_qk_cosine_current.append(hidden_metrics[0])
            self._hidden_qk_logit_current.append(hidden_metrics[1])
            self._hidden_v_contribution_current.append(hidden_metrics[2])

            if self.query_kind == "relay":
                if weights.shape[2] < 2:
                    raise RuntimeError("Relay step must contain x_t and at least one r_t query")
                relay = weights[0, :, 1:, :].index_select(-1, positions)
                relay = relay.mean(dim=(0, 1)).detach().float().cpu().numpy().reshape(self.grid)
                assert self._relay_current is not None
                self._relay_current.append(relay)
                relay_metrics = self._qkv_maps(_module, weights, layer, slice(1, None))
                assert self._relay_qk_cosine_current is not None
                assert self._relay_qk_logit_current is not None
                assert self._relay_v_contribution_current is not None
                self._relay_qk_cosine_current.append(relay_metrics[0])
                self._relay_qk_logit_current.append(relay_metrics[1])
                self._relay_v_contribution_current.append(relay_metrics[2])

            del self._query_by_layer[layer]
            del self._cache_by_layer[layer]

            if layer == self.layers[-1]:
                if len(self._hidden_current) != len(self.layers):
                    raise RuntimeError("Decoder step did not visit every layer")
                self.hidden_steps.append(np.stack(self._hidden_current).astype(np.float32))
                self.hidden_qk_cosine_steps.append(
                    np.stack(self._hidden_qk_cosine_current).astype(np.float32)
                )
                self.hidden_qk_logit_steps.append(
                    np.stack(self._hidden_qk_logit_current).astype(np.float32)
                )
                self.hidden_v_contribution_steps.append(
                    np.stack(self._hidden_v_contribution_current).astype(np.float32)
                )
                if self.query_kind == "relay":
                    assert self._relay_current is not None
                    self.relay_steps.append(np.stack(self._relay_current).astype(np.float32))
                    assert self._relay_qk_cosine_current is not None
                    assert self._relay_qk_logit_current is not None
                    assert self._relay_v_contribution_current is not None
                    self.relay_qk_cosine_steps.append(
                        np.stack(self._relay_qk_cosine_current).astype(np.float32)
                    )
                    self.relay_qk_logit_steps.append(
                        np.stack(self._relay_qk_logit_current).astype(np.float32)
                    )
                    self.relay_v_contribution_steps.append(
                        np.stack(self._relay_v_contribution_current).astype(np.float32)
                    )
                self._hidden_current = None
                self._relay_current = None
                self._hidden_qk_cosine_current = None
                self._hidden_qk_logit_current = None
                self._hidden_v_contribution_current = None
                self._relay_qk_cosine_current = None
                self._relay_qk_logit_current = None
                self._relay_v_contribution_current = None
        return capture

    def __exit__(self, *_exc):
        for handle in self._handles:
            handle.remove()
        for layer, config in self._configs.items():
            self.modules[layer].config = config

    def arrays(self, expected_steps: int, *, allow_one_unused_tail: bool = False
               ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._hidden_current is not None:
            raise RuntimeError("Capture ended inside a decoder step")
        # generate_from_relays performs one final forward after sampling the
        # last token at a hard token limit. Its logits are never sampled, so
        # that predictor frame must not be presented as an inference step.
        if allow_one_unused_tail and len(self.hidden_steps) == expected_steps + 1:
            self.hidden_steps.pop()
            self.hidden_qk_cosine_steps.pop()
            self.hidden_qk_logit_steps.pop()
            self.hidden_v_contribution_steps.pop()
            if self.query_kind == "relay":
                self.relay_steps.pop()
                self.relay_qk_cosine_steps.pop()
                self.relay_qk_logit_steps.pop()
                self.relay_v_contribution_steps.pop()
        if len(self.hidden_steps) != expected_steps:
            raise RuntimeError(
                f"Captured {len(self.hidden_steps)} hidden steps for {expected_steps} tokens"
            )
        hidden = np.stack(self.hidden_steps).astype(np.float32)
        relay = None
        if self.query_kind == "relay":
            if len(self.relay_steps) != expected_steps:
                raise RuntimeError("Relay and language capture lengths differ")
            relay = np.stack(self.relay_steps).astype(np.float32)
        return hidden, relay

    def qkv_arrays(self, expected_steps: int) -> dict[str, np.ndarray]:
        """Return raw patch maps aligned with ``arrays`` inference frames."""
        groups = {
            "hidden_qk_cosine": self.hidden_qk_cosine_steps,
            "hidden_qk_logit": self.hidden_qk_logit_steps,
            "hidden_v_contribution": self.hidden_v_contribution_steps,
        }
        if self.query_kind == "relay":
            groups.update({
                "relay_qk_cosine": self.relay_qk_cosine_steps,
                "relay_qk_logit": self.relay_qk_logit_steps,
                "relay_v_contribution": self.relay_v_contribution_steps,
            })
        result = {}
        for name, rows in groups.items():
            if len(rows) != expected_steps:
                raise RuntimeError(
                    f"Captured {len(rows)} {name} steps for {expected_steps} tokens"
                )
            result[name] = np.stack(rows).astype(np.float32)
        return result


def set_decoder_attention_implementation(model, implementation: str) -> None:
    """Set every Qwen text-attention module to eager or SDPA."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    if implementation not in {"eager", "sdpa"}:
        raise ValueError(implementation)
    for module in model.modules():
        if isinstance(module, Qwen3VLTextAttention):
            module.config = copy.copy(module.config)
            module.config._attn_implementation = implementation
