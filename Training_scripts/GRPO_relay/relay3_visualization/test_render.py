from __future__ import annotations

import unittest

import numpy as np

try:
    from .render import _normalized_u8
except ImportError:  # direct execution from this directory
    from render import _normalized_u8


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

if __name__ == "__main__":
    unittest.main()
