"""Capture real decoder query-to-image attention during greedy inference."""
from __future__ import annotations

import copy

import numpy as np


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
        self._hidden_current: list[np.ndarray] | None = None
        self._relay_current: list[np.ndarray] | None = None
        self._handles = []
        self._configs = {}

    def __enter__(self):
        # Eager is required because SDPA does not return attention weights.
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
                raise RuntimeError("Expected eager attention weights [1,heads,queries,keys]")
            if layer == 0:
                if self._hidden_current is not None:
                    raise RuntimeError("A new decoder step began before the previous step ended")
                self._hidden_current = []
                self._relay_current = [] if self.query_kind == "relay" else None
            if self._hidden_current is None:
                raise RuntimeError("Decoder attention hooks did not begin at layer zero")
            positions = self.positions.to(weights.device)
            if int(positions.max()) >= weights.shape[-1]:
                raise RuntimeError("Image position lies outside the current attention key axis")

            hidden_row = 0 if self.query_kind == "relay" else -1
            hidden = weights[0, :, hidden_row, :].index_select(-1, positions)
            hidden = hidden.mean(dim=0).detach().float().cpu().numpy().reshape(self.grid)
            self._hidden_current.append(hidden)

            if self.query_kind == "relay":
                if weights.shape[2] < 2:
                    raise RuntimeError("Relay step must contain x_t and at least one r_t query")
                relay = weights[0, :, 1:, :].index_select(-1, positions)
                relay = relay.mean(dim=(0, 1)).detach().float().cpu().numpy().reshape(self.grid)
                assert self._relay_current is not None
                self._relay_current.append(relay)

            if layer == self.layers[-1]:
                if len(self._hidden_current) != len(self.layers):
                    raise RuntimeError("Decoder step did not visit every layer")
                self.hidden_steps.append(np.stack(self._hidden_current).astype(np.float32))
                if self.query_kind == "relay":
                    assert self._relay_current is not None
                    self.relay_steps.append(np.stack(self._relay_current).astype(np.float32))
                self._hidden_current = None
                self._relay_current = None
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
            if self.query_kind == "relay":
                self.relay_steps.pop()
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


def set_decoder_attention_implementation(model, implementation: str) -> None:
    """Set every Qwen text-attention module to eager or SDPA."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    if implementation not in {"eager", "sdpa"}:
        raise ValueError(implementation)
    for module in model.modules():
        if isinstance(module, Qwen3VLTextAttention):
            module.config = copy.copy(module.config)
            module.config._attn_implementation = implementation
