"""Small CPU tests; they do not load Qwen or allocate a GPU."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import Example, LatentCropDataset, collate_steps, split_by_map
from latent_visual_extractor import ExtractorConfig, LatentVisualExtractor
from train import reconstruction_and_stop_loss


class LatentCropTests(unittest.TestCase):
    def test_all_layers_and_stop_receive_gradients(self):
        cfg = ExtractorConfig(reasoning_dim=8, kv_dim=4, merger_dim=8, decoder_layers=3,
                              latent_dim=8, attention_heads=2, k_max=7, query_grid_width=4)
        model = LatentVisualExtractor(cfg)
        keys = torch.randn(2, 3, 5, 4)
        values = torch.randn(2, 3, 5, 4)
        memory_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
        output = model(torch.randn(2, 8), keys, values, memory_mask,
                       torch.tensor([[1, 5], [1, 3]]), num_queries=6)
        self.assertEqual(tuple(output["embeddings"].shape), (2, 6, 8))
        losses = reconstruction_and_stop_loss(output, torch.randn(2, 5, 8),
                                              torch.tensor([0, 5]), 0.1, 1.0)
        losses["total"].backward()
        for parameter in (model.key_layer_logits, model.value_layer_logits,
                          model.reasoning_projection[1].weight, model.merger_output.weight,
                          model.stop_token):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_map_split_keeps_all_steps_together(self):
        examples = [Example(f"map-{m}", f"q-{m}-{step}", Path("hidden"), step, step,
                            Path("kv"), (), 0)
                    for m in range(10) for step in range(3)]
        splits = split_by_map(examples, seed=42, validation_fraction=0.2, test_fraction=0.2)
        assignments = {example.map_id: split for split, part in splits.items() for example in part}
        self.assertEqual(len(assignments), 10)
        self.assertEqual({name: len({example.map_id for example in part}) for name, part in splits.items()},
                         {"train": 6, "validation": 2, "test": 2})
        self.assertEqual(len(splits[assignments["map-0"]]), 3 * len(
            {example.map_id for example in splits[assignments["map-0"]]}))

    def test_saved_gqa_visual_kv_order_and_empty_target(self):
        kv, hidden = Path("kv.pt"), Path("hidden.pt")
        # Two KV heads, two visual positions, two head features.
        raw = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
        payloads = {
            kv: {"metadata": {"image_grid": [1, 2]},
                 "layer_00_key": raw, "layer_00_value": raw + 10},
            hidden: {"hidden_states_bf16": torch.ones(1, 8, dtype=torch.bfloat16),
                     "token_indices": torch.tensor([4])},
        }
        with patch("data.torch.load", side_effect=lambda path, **kwargs: payloads[path]):
            dataset = LatentCropDataset([Example("map", "question", hidden, 0, 4, kv, (), 0)], merger_dim=8)
            sample = dataset[0]
            self.assertEqual(sample["keys"][0, 0].tolist(), [1., 2., 5., 6.])
            self.assertEqual(sample["keys"][0, 1].tolist(), [3., 4., 7., 8.])
            self.assertEqual(tuple(sample["target"].shape), (0, 8))
            batch = collate_steps([sample])
            self.assertEqual(tuple(batch["target"].shape), (1, 0, 8))
            self.assertEqual(batch["lengths"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
