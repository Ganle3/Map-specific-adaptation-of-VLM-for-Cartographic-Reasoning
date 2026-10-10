import unittest

import numpy as np

from attention_policy import aggregate_decoder_layers, salient_regions, sparsevlm_layer_scores, top_regions
from rolling_sparse_qwen3vl import PixelRegion, _extract_final_answer, _project_component


class AttentionPolicyTests(unittest.TestCase):
    def test_aggregation_is_explicit(self):
        maps = np.array([[[1., 3.]], [[5., 2.]]])
        np.testing.assert_allclose(aggregate_decoder_layers(maps, "last"), [[5., 2.]])
        np.testing.assert_allclose(aggregate_decoder_layers(maps, "mean"), [[3., 2.5]])
        np.testing.assert_allclose(aggregate_decoder_layers(maps, "max"), [[5., 3.]])
        np.testing.assert_allclose(aggregate_decoder_layers(maps, "weighted_mean", [1, 3]), [[4., 2.25]])

    def test_sparsevlm_is_head_then_text_mean_for_one_layer(self):
        # heads x queries x keys; queries 1,2 score visual keys 0,1.
        a = np.zeros((2, 3, 4))
        a[0, 1:3, :2] = [[2, 4], [6, 8]]
        a[1, 1:3, :2] = [[4, 6], [8, 10]]
        np.testing.assert_allclose(sparsevlm_layer_scores(a, [1, 2], [0, 1]), [5., 7.])

    def test_regions_are_ranked_and_non_overlapping(self):
        scores = np.zeros((6, 6))
        scores[:2, :2] = 10
        scores[4:, 4:] = 9
        regions = top_regions(scores, count=2, side=2, iou_limit=0)
        self.assertEqual((regions[0].top, regions[0].left), (0, 0))
        self.assertEqual((regions[1].top, regions[1].left), (4, 4))

    def test_salient_regions_follow_attention_shape_not_fixed_square(self):
        scores = np.zeros((6, 8))
        scores[1, 1:5] = 4  # a one-patch-high, four-patch-wide salient road
        scores[4:6, 6:8] = 1
        regions = salient_regions(scores, count=2, attention_mass=0.80)
        self.assertEqual((regions[0].top, regions[0].left, regions[0].bottom, regions[0].right), (1, 1, 2, 5))

    def test_projection_of_dense_patch_grid_never_creates_empty_crop(self):
        parent = PixelRegion(10, 20, 11, 21, 1.0)  # a one-pixel source crop
        local = top_regions(np.array([[1., 0.], [0., 0.]]), count=1, side=1)[0]
        projected = _project_component(parent, local, (32, 32))
        self.assertGreater(projected.right, projected.left)
        self.assertGreater(projected.bottom, projected.top)

    def test_final_answer_parser_prefers_the_final_labeled_line(self):
        self.assertEqual(_extract_final_answer("analysis\nFinal answer: 2\n"), "2")


if __name__ == "__main__":
    unittest.main()
