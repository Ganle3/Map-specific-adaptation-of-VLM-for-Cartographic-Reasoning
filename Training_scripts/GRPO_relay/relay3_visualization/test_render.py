from __future__ import annotations

import unittest

import numpy as np

try:
    from .render import _normalized_u8, _signed_extreme
except ImportError:  # direct execution from this directory
    from render import _normalized_u8, _signed_extreme


class AggregationTest(unittest.TestCase):
    def test_each_map_is_normalized_after_its_patch_max(self):
        raw = np.array([[[[1.0, 2.0], [0.0, 1.0]]]], dtype=np.float32)
        actual = _normalized_u8(raw)
        np.testing.assert_array_equal(
            actual, np.array([[[[128, 255], [0, 128]]]], dtype=np.uint8)
        )

    def test_p99_is_applied_after_aggregation(self):
        # One extreme patch should be clipped while the remaining spatial
        # pattern keeps useful contrast.
        raw = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
        raw[0, 0, 0, 0] = 10000.0
        literal = _normalized_u8(raw)
        robust = _normalized_u8(raw, quantile=0.99)
        self.assertLess(int(literal[0, 0, 9, 9]), 10)
        self.assertGreater(int(robust[0, 0, 9, 9]), 10 * int(literal[0, 0, 9, 9]))

    def test_signed_qk_maps_preserve_zero_and_both_directions(self):
        raw = np.array([[[[-2.0, 0.0, 1.0]]]], dtype=np.float32)
        actual = _normalized_u8(raw, signed=True)
        np.testing.assert_array_equal(
            actual, np.array([[[[0, 128, 191]]]], dtype=np.uint8)
        )

    def test_signed_all_layer_aggregation_keeps_largest_absolute_sign(self):
        raw = np.array([[[[0.2, -0.4]], [[-0.8, 0.3]]]], dtype=np.float32)
        actual = _signed_extreme(raw, axis=1)
        np.testing.assert_allclose(actual, np.array([[[-0.8, -0.4]]], dtype=np.float32))

if __name__ == "__main__":
    unittest.main()
