"""CPU regression tests for scheme-3 masks and temporary cache removal."""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from Image_question_CoT_shared_relay.dynamic_decode import (
    _full_attention_mask, _step_attention_mask,
)


def allowed(mask: torch.Tensor) -> torch.Tensor:
    return mask[0, 0].eq(0)


def main() -> None:
    visible = torch.tensor([False, True, True, False, False])
    step = allowed(_step_attention_mask(5, visible, torch.float32, "cpu"))
    assert step.shape == (2, 7)
    assert step[0].all()
    assert step[1].tolist() == [False, True, True, False, False, True, True]

    step_r4 = allowed(_step_attention_mask(
        5, visible, torch.float32, "cpu", num_relays=4
    ))
    assert step_r4.shape == (5, 10)
    assert step_r4[0].all()
    expected_relay = [False, True, True, False, False] + [True] * 5
    assert all(row.tolist() == expected_relay for row in step_r4[1:])

    full = allowed(_full_attention_mask(5, 2, visible, torch.float32, "cpu"))
    # First pair positions are x=5,r=6; second are x=7,r=8.
    assert full[5].tolist() == [True] * 7 + [False, False]
    assert full[6].tolist() == [False, True, True, False, False, True, True, False, False]
    assert full[7].tolist() == [True] * 6 + [False, True, True]
    assert full[8].tolist() == [False, True, True, False, False, False, False, True, True]

    full_r2 = allowed(_full_attention_mask(
        5, 2, visible, torch.float32, "cpu", num_relays=2
    ))
    assert full_r2.shape == (11, 11)
    # Prefix=0..4; groups are [x0,r00,r01] and [x1,r10,r11].
    assert full_r2[5].tolist() == [True] * 8 + [False] * 3
    assert full_r2[6].tolist() == [False, True, True, False, False, True, True, True, False, False, False]
    assert full_r2[7].tolist() == full_r2[6].tolist()
    assert full_r2[8].tolist() == [True] * 6 + [False, False] + [True] * 3
    assert full_r2[9].tolist() == [False, True, True, False, False, False, False, False, True, True, True]
    assert full_r2[10].tolist() == full_r2[9].tolist()

    cache = DynamicCache()
    keys = torch.randn(1, 2, 5, 4)
    values = torch.randn(1, 2, 5, 4)
    cache.update(keys, values, 0)
    pair_keys = torch.randn(1, 2, 2, 4)
    pair_values = torch.randn(1, 2, 2, 4)
    cache.update(pair_keys, pair_values, 0)
    assert cache.get_seq_length() == 7
    retained_language_key = cache.layers[0].keys[..., -2, :].clone()
    cache.crop(-1)
    assert cache.get_seq_length() == 6
    assert torch.equal(cache.layers[0].keys[..., -1, :], retained_language_key)

    multi_cache = DynamicCache()
    multi_cache.update(keys, values, 0)
    group_keys = torch.randn(1, 2, 5, 4)  # x_t plus four temporary relays
    group_values = torch.randn(1, 2, 5, 4)
    multi_cache.update(group_keys, group_values, 0)
    retained_multi_language_key = multi_cache.layers[0].keys[..., -5, :].clone()
    multi_cache.crop(-4)
    assert multi_cache.get_seq_length() == 6
    assert torch.equal(multi_cache.layers[0].keys[..., -1, :], retained_multi_language_key)
    print({
        "status": "PASS",
        "step_mask_shape": list(step.shape),
        "full_mask_shape": list(full.shape),
        "cache_after_pair": 7,
        "cache_after_crop": cache.get_seq_length(),
        "retained_position": "x_t",
        "removed_position": "r_t",
    })


if __name__ == "__main__":
    main()
