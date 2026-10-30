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

    full = allowed(_full_attention_mask(5, 2, visible, torch.float32, "cpu"))
    # First pair positions are x=5,r=6; second are x=7,r=8.
    assert full[5].tolist() == [True] * 7 + [False, False]
    assert full[6].tolist() == [False, True, True, False, False, True, True, False, False]
    assert full[7].tolist() == [True] * 6 + [False, True, True]
    assert full[8].tolist() == [False, True, True, False, False, False, False, True, True]

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
