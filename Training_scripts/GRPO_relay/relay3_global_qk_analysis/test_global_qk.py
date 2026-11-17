from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compact_capture import CompactQKCapture
from analyze_population import binned_matrix


class GlobalQKTest(unittest.TestCase):
    def test_flattened_head_dot_equals_mean_head_cosine(self):
        torch.manual_seed(4)
        q = torch.randn(8, 16)
        keys = torch.randn(11, 8, 16)
        q_flat = CompactQKCapture._flatten_head_unit(q)
        k_flat = CompactQKCapture._flatten_head_unit(keys)
        observed = k_flat @ q_flat
        expected = torch.nn.functional.cosine_similarity(
            keys, q.unsqueeze(0), dim=-1
        ).mean(-1)
        torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)

    def test_step_bins_do_not_mix_missing_tail(self):
        metrics = np.zeros((7, 3, 5), dtype=np.float32)
        metrics[:, :, 3] = np.arange(7)[:, None]
        record = {"arrays": {
            "metrics": metrics,
            "metric_names": np.asarray([
                "nearest_key_distance", "top5pct_key_distance", "mean_key_distance",
                "attention_weighted_key_distance", "image_attention_mass",
            ]),
        }}
        result = binned_matrix(record, "attention_weighted_key_distance")
        np.testing.assert_allclose(result[0], 2.0)
        np.testing.assert_allclose(result[1], 5.5)
        self.assertTrue(math.isnan(result[2, 0]))


if __name__ == "__main__":
    unittest.main()
