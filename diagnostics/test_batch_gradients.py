"""CPU regression checks; model integration must run in the training container."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.configs.fixed20_batches import FIXED_BATCHES, index_fixed20
from diagnostics.gradient.analyze_batch_gradients import pairwise_geometry, write_results
from diagnostics.visualization.plot_batch_gradient_matrix import plot_results


def test_geometry_and_zero_mask():
    vectors = [[1., 0], [-1., 0], [0., 2], [0., 0], [1e-14, 0]]
    matrices, norms, valid = pairwise_geometry(vectors)
    assert valid.tolist() == [True, True, True, False, False]
    assert matrices['cosine'][0, 1] == -1
    assert matrices['cosine'][0, 2] == 0
    assert np.isnan(matrices['cosine'][3:, :]).all()
    assert np.isnan(matrices['cosine'][:, 3:]).all()
    assert matrices['dot_product'][0, 1] == -1
    assert matrices['distance'][0, 1] == 2
    np.testing.assert_allclose(matrices['distance'][0, 2], np.sqrt(5))
    np.testing.assert_allclose(norms, [1, 1, 2, 0, 1e-14])


@pytest.mark.parametrize('vectors', [[[np.nan]], [[np.inf]], [[1], [1, 2]]])
def test_invalid_gradients_fail(vectors):
    with pytest.raises(ValueError):
        pairwise_geometry(vectors)


def test_fixed20_matches_saved_selection():
    rows = json.loads((ROOT / 'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_joint44_improved_20.json').read_text())
    ids = [f"mapwise_{r['country']}_{r['map_no']}_t{int(r['template_no'])}_src{r['source_index']:04d}" for r in rows]
    index_fixed20({'qa_id': ids})
    assert ids == [qid for batch in FIXED_BATCHES.values() for qid in batch]
    with pytest.raises(ValueError):
        index_fixed20({'qa_id': [ids[0]] * 20})


def test_csv_and_plots(tmp_path):
    labels = list(FIXED_BATCHES)
    write_results(tmp_path, labels, np.zeros((5, 3)),
                  [dict(batch_id=b) for b in labels], 1e-12)
    plot_results(tmp_path, dot_product=True)
    with (tmp_path / 'batch_metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(r['valid_gradient'] == 'False' for r in rows)
    for name in ('cosine_matrix.png', 'gradient_norms.png', 'dot_product_matrix.png'):
        assert (tmp_path / name).stat().st_size > 1000
