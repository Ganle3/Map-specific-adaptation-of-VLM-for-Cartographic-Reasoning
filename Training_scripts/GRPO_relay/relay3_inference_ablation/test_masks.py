"""CPU-only regression tests for the two path-specific masks."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablation_decode import step_attention_mask


def allowed(mask):
    return mask[0, 0].eq(0)


def test_current_x_path_can_be_removed_without_changing_x_row():
    visible = torch.tensor([False, True, True, False])
    full = allowed(step_attention_mask(4, visible, torch.float32, "cpu", 1,
                                       relay_reads_current_x=True))
    cut = allowed(step_attention_mask(4, visible, torch.float32, "cpu", 1,
                                      relay_reads_current_x=False))
    assert torch.equal(full[0], cut[0])
    assert bool(full[1, 4]) and not bool(cut[1, 4])
    assert torch.equal(full[1, :4], cut[1, :4])
    assert bool(cut[1, 5])


def test_multiple_relays_keep_relay_bank_visible():
    visible = torch.tensor([True, False])
    cut = allowed(step_attention_mask(2, visible, torch.float32, "cpu", 2,
                                      relay_reads_current_x=False))
    assert cut.shape == (3, 5)
    assert cut[1:, 2].sum() == 0
    assert cut[1:, 3:].all()


if __name__ == "__main__":
    test_current_x_path_can_be_removed_without_changing_x_row()
    test_multiple_relays_keep_relay_bank_visible()
    print("PASS: Relay3 ablation masks")
