"""Training-free fixed and rolling ILVAD for eager Qwen3-VL text attention."""
from __future__ import annotations

import copy
import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ILVADConfig:
    alpha: float = 1.5
    beta: float = 0.0
    tau: float = 5.0
    layers: tuple[int, ...] = tuple(range(8, 36))
    rolling: bool = False
    window: int = 8
    update_interval: int = 4
    warmup: int = 12


def text_attention_modules(model):
    """Return Qwen decoder attention modules, also when wrapped by PEFT."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
    found = {m.layer_idx: m for m in model.modules() if isinstance(m, Qwen3VLTextAttention)}
    if not found:
        raise RuntimeError("No Qwen3VLTextAttention modules found; check transformers/model version.")
    return dict(sorted(found.items()))


def _clipped_exp(values: torch.Tensor, scale: float) -> torch.Tensor:
    values = values.float()
    low, high = values.mean() - 3 * values.std(), values.mean() + 3 * values.std()
    values = values.clamp(low.item(), high.item())
    spread = values.max() - values.min()
    if spread <= torch.finfo(values.dtype).eps:
        return torch.ones_like(values)
    return torch.exp((values - values.min()) / spread * scale)


def saliency_from_records(records: list[dict[int, torch.Tensor]], tau: float, alpha: float,
                          *, skip_first: bool = True) -> torch.Tensor:
    """Paper Eq.-equivalent map from records[layer] = [heads, image_tokens]."""
    if len(records) < 1:
        raise ValueError("No decoder attention records captured.")
    # The upstream code skips generation's prefill record when a later step exists.
    records = records[1:] if skip_first and len(records) > 1 else records
    layer_ids = sorted(records[0])
    per_layer = []
    for layer in layer_ids:
        steps = []
        for record in records:
            value = record[layer]
            steps.append(value[value.sum(-1).topk(max(1, value.shape[0] // 2)).indices].mean(0))
        average = torch.stack(steps).mean(0)
        per_layer.append((average > tau * average.mean()).float())
    binary = torch.stack(per_layer)
    gains = (binary[1:] - binary[:-1]).clamp_min(0).sum(0)
    return _clipped_exp(gains, alpha)


class AttentionRecorder:
    """Records final-query image attention for every decoder layer and step."""
    def __init__(self, model, image_positions: torch.Tensor):
        self.modules = text_attention_modules(model)
        self.positions = image_positions
        self.records: list[dict[int, torch.Tensor]] = []
        self._active: dict[int, torch.Tensor] = {}
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
        def capture(_module, _args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4:
                raise RuntimeError("ILVAD requires eager [batch, heads, query, key] attention weights.")
            self._active[layer] = weights[0, :, -1].index_select(-1, self.positions.to(weights.device)).detach().float().cpu()
            if layer == max(self.modules):
                if len(self._active) != len(self.modules):
                    raise RuntimeError("Incomplete decoder attention pass captured.")
                self.records.append(self._active)
                self._active = {}
        return capture

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        for layer, config in self.configs.items():
            self.modules[layer].config = config


class ILVADContext:
    """Qwen3-VL ILVAD; optionally replace map by a fresh rolling map.

    In rolling mode, each map is computed from pre-enhancement attention in the
    most recent window, then applies only from the *next* decoding step. This
    prevents enhanced attention from feeding itself back into later maps.
    """
    def __init__(self, model, image_start: int, image_end: int,
                 enhanced_map: torch.Tensor | None, config: ILVADConfig):
        self.modules = text_attention_modules(model)
        self.s, self.e = image_start, image_end
        self.enhanced_map = enhanced_map
        self.config = config
        self.original = {}
        self.text_scores = None
        self.raw_window: list[dict[int, torch.Tensor]] = []
        self.raw_step: dict[int, torch.Tensor] = {}
        self.step = 0
        self.map_history: list[tuple[int, torch.Tensor]] = []

    def __enter__(self):
        for layer, module in self.modules.items():
            self.original[layer] = module.forward
            module.forward = functools.partial(self._forward, module)
        return self

    def __exit__(self, *_exc):
        for layer, module in self.modules.items():
            module.forward = self.original[layer]

    def _forward(self, module, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        # Mirrors Qwen3VLTextAttention.forward but intervenes after softmax.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden_states).view(head_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, module.layer_idx)
        key, value = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
        weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            weights = weights + attention_mask[..., : key.shape[-2]]
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # Only generated-token rows are modified. LoRA remains active: projections
        # above are the PEFT-wrapped q/k/v/o projection modules.
        if weights.shape[-2] == 1 and self.e <= weights.shape[-1]:
            vision = weights[0, :, -1, self.s:self.e]
            if self.config.rolling:
                # Store native attention before any ILVAD multiplication.
                self.raw_step[module.layer_idx] = vision.detach()
            if self.enhanced_map is not None and module.layer_idx in self.config.layers:
                evidence = self.enhanced_map.to(weights.device, weights.dtype)
                score = (vision * evidence.log()).sum(-1) / vision.sum(-1).clamp_min(1e-8)
                selected = score.topk(max(1, vision.shape[0] // 2)).indices
                vision[selected] *= evidence
                # Qwen's text keys start at image_end. Retain the paper's
                # grounded-text reinforcement while avoiding NaNs for one token.
                query_part = weights[0, :, -1, self.e:]
                # beta=0 is explicitly vision-only; do not silently change text
                # attention in that ablation.
                if self.config.beta != 0 and query_part.shape[-1]:
                    grounded = (vision[selected] * evidence.log()).mean()
                    self.text_scores = grounded[None] if self.text_scores is None else torch.cat((self.text_scores, grounded[None]))
                    z = self.text_scores
                    z = (z - z.min()) / (z.max() - z.min()).clamp_min(1e-8) + self.config.beta
                    heads = query_part.sum(-1).topk(max(1, query_part.shape[0] // 2)).indices
                    query_part[heads, : min(len(z), query_part.shape[-1])] *= z[:query_part.shape[-1]]
                weights[0, :, -1] /= weights[0, :, -1].sum(-1, keepdim=True).clamp_min(1e-8)
            if self.config.rolling and module.layer_idx == max(self.modules):
                self._finish_rolling_step()
        weights = F.dropout(weights, p=module.attention_dropout, training=module.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous().reshape(*shape, -1)
        return module.o_proj(output), weights

    def _finish_rolling_step(self):
        """Refresh after this pass, so every layer used one consistent old map."""
        if len(self.raw_step) != len(self.modules):
            raise RuntimeError("Incomplete raw-attention step in rolling ILVAD.")
        self.step += 1
        self.raw_window.append(self.raw_step)
        self.raw_window = self.raw_window[-self.config.window:]
        self.raw_step = {}
        due = (self.step >= self.config.warmup and
               (self.step - self.config.warmup) % self.config.update_interval == 0)
        if due and len(self.raw_window) == self.config.window:
            # No prefill exists in this window, so do not drop its first record.
            self.enhanced_map = saliency_from_records(
                self.raw_window, self.config.tau, self.config.alpha, skip_first=False)
            self.text_scores = None
            self.map_history.append((self.step, self.enhanced_map.detach().float().cpu()))
