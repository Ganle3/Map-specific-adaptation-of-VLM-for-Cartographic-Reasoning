"""Small CPU tests of the real Qwen attention hook and token/grid alignment."""
import unittest

import numpy as np
import torch
from transformers import Qwen3VLTextConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
from transformers.vision_utils import get_vision_position_ids

from attention_runner import AttentionCapture, Settings


class CaptureTests(unittest.TestCase):
    def test_merged_image_tokens_are_row_major_spatial_blocks(self):
        # Qwen patchifies in (block_y, block_x, inner_y, inner_x) order. The
        # merger consumes each consecutive 2x2 group, leaving block_y/x in
        # raster order. This is exactly the order reshaped by AttentionCapture.
        positions = get_vision_position_ids(torch.tensor([[1, 6, 8]]), spatial_merge_size=2)
        merged_top_left = positions.reshape(-1, 4, 2)[:, 0]
        expected = torch.tensor([[y, x] for y in range(0, 6, 2) for x in range(0, 8, 2)])
        torch.testing.assert_close(merged_top_left, expected)

    def test_prefill_and_cached_decode_match_native_weights(self):
        torch.manual_seed(7)
        config = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        config._attn_implementation = "eager"
        attention = Qwen3VLTextAttention(config, layer_idx=0).eval()
        model = torch.nn.ModuleList([attention])
        hidden = torch.randn(1, 5, 32)
        mask = torch.zeros(1, 1, 5, 5).masked_fill(
            torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf"))
        embeddings = (torch.ones(1, 5, 8), torch.zeros(1, 5, 8))
        with torch.no_grad():
            reference, weights = attention(hidden, embeddings, mask)
        cache = DynamicCache(config=config)
        settings = Settings(layers=(-1,), heads=(3, 1))
        with AttentionCapture(model, settings, torch.tensor([0, 1, 2, 3]), (2, 2)) as capture, torch.no_grad():
            actual, _ = attention(hidden, embeddings, mask, past_key_values=cache)
            torch.testing.assert_close(actual, reference)
            _, decode_weights = attention(torch.randn(1, 1, 32),
                                          (torch.ones(1, 1, 8), torch.zeros(1, 1, 8)),
                                          torch.zeros(1, 1, 1, 6), past_key_values=cache)
        array = capture.array(2)
        self.assertEqual(array.shape, (2, 1, 2, 2, 2))
        np.testing.assert_allclose(array[0, 0].reshape(2, 4), weights[0, [3, 1], -1, :4].numpy())
        np.testing.assert_allclose(array[1, 0].reshape(2, 4), decode_weights[0, [3, 1], -1, :4].numpy())
        self.assertIs(attention.config, config)
        self.assertEqual(len(attention._forward_hooks), 0)
        with self.assertRaises(RuntimeError):
            capture.array(3)

    def test_invalid_layer_rejected(self):
        config = Qwen3VLTextConfig()
        model = torch.nn.ModuleList([Qwen3VLTextAttention(config, layer_idx=0)])
        with self.assertRaises(ValueError):
            AttentionCapture(model, Settings(layers=(1,)), torch.tensor([0]), (1, 1))

    def test_uncaptured_layers_remain_eager_during_capture(self):
        config = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=8)
        config._attn_implementation = "sdpa"
        layers = torch.nn.ModuleList([Qwen3VLTextAttention(config, layer_idx=i) for i in range(2)])
        with AttentionCapture(layers, Settings(layers=(1,), heads=(0,)), torch.tensor([0]), (1, 1)):
            self.assertEqual(layers[0].config._attn_implementation, "eager")
            self.assertEqual(layers[1].config._attn_implementation, "eager")
        self.assertIs(layers[0].config, config)
        self.assertIs(layers[1].config, config)


if __name__ == "__main__":
    unittest.main()
