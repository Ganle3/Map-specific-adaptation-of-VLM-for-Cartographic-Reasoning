from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    repeat_kv,
)

from .capture import ImageAttentionCapture


class _OneLayerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Qwen3VLTextAttention(config, layer_idx=0)


class ExactQKVCaptureTest(unittest.TestCase):
    def test_post_rope_qk_and_value_maps_match_direct_calculation(self):
        torch.manual_seed(7)
        config = Qwen3VLTextConfig(
            hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        )
        model = _OneLayerModel(config).eval()
        hidden = torch.randn(1, 3, 8)  # x_t and two temporary relay rows
        cos = torch.ones(1, 3, 4)
        sin = torch.zeros(1, 3, 4)  # identity RoPE, but through the real function
        mask = torch.zeros(1, 1, 3, 3)
        cache = DynamicCache(config=config)

        with ImageAttentionCapture(
            model, torch.tensor([0]), (1, 1), query_kind="relay"
        ) as capture:
            model.attn(
                hidden_states=hidden,
                position_embeddings=(cos, sin),
                attention_mask=mask,
                past_key_values=cache,
            )

        attention, relay_attention = capture.arrays(1)
        metrics = capture.qkv_arrays(1)
        self.assertEqual(attention.shape, (1, 1, 1, 1))
        self.assertEqual(relay_attention.shape, (1, 1, 1, 1))
        for value in metrics.values():
            self.assertEqual(value.shape, (1, 1, 1, 1))
            self.assertTrue(np.isfinite(value).all())

        module = model.attn
        shape = (1, 3, -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden).view(shape)).transpose(1, 2)
        keys = repeat_kv(cache.layers[0].keys[..., :1, :], module.num_key_value_groups)
        expected_cosine = torch.nn.functional.cosine_similarity(
            query[:, :, :1, :].unsqueeze(-2), keys.unsqueeze(-3), dim=-1
        ).mean().item()
        expected_logit = (
            torch.matmul(query[:, :, :1, :], keys.transpose(-2, -1)) * module.scaling
        ).mean().item()
        self.assertAlmostEqual(
            float(metrics["hidden_qk_cosine"][0, 0, 0, 0]), expected_cosine, places=6
        )
        self.assertAlmostEqual(
            float(metrics["hidden_qk_logit"][0, 0, 0, 0]), expected_logit, places=6
        )


if __name__ == "__main__":
    unittest.main()
