#!/usr/bin/env python3
"""Fast structural and gradient regression for only_visual_relay."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from only_visual_relay_policy import OnlyVisualRelayPolicy
from only_visual_relay_prefill import (build_only_visual_relay_prefill,
                                       explicit_relay_decoder_inputs)


class FakeQwen(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=10, video_token_id=11,
            vision_start_token_id=12, vision_end_token_id=13,
            pad_token_id=0,
        )
        self.embedding = nn.Embedding(64, hidden_size)

    def get_input_embeddings(self):
        return self.embedding


def main() -> None:
    torch.manual_seed(7)
    model = FakeQwen()
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    # text, <vision_start>, two image holders, <vision_end>, question tokens
    source = {
        "input_ids": torch.tensor([[1, 12, 10, 10, 13, 20, 21]]),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 0, 0, 0]]),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    features = SimpleNamespace(
        pooler_output=(torch.randn(2, 16),),
        deepstack_features=[(torch.randn(2, 16),) for _ in range(3)],
    )
    policy = OnlyVisualRelayPolicy(16, num_relays=2)
    augmented, attention = build_only_visual_relay_prefill(
        model, tokenizer, source, features, policy, with_grad=True,
    )

    assert attention is None
    assert augmented["input_ids"].shape[1] == source["input_ids"].shape[1] + 2
    assert augmented["input_ids"][0, 4].item() == 13  # <vision_end>
    assert augmented["relay_positions"].tolist() == [5, 6]
    assert augmented["mm_token_type_ids"][0, 5:7].tolist() == [0, 0]
    assert torch.equal(augmented["image_grid_thw"], source["image_grid_thw"])
    assert int(augmented["input_ids"].eq(10).sum()) == 2

    embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, augmented["input_ids"], features,
        augmented["relay_embeddings"], augmented["relay_positions"],
    )
    assert visual_mask.nonzero().tolist() == [[0, 2], [0, 3]]
    assert all(level.shape == (2, 16) for level in deepstack)
    # Only the two injected positions contribute to this test objective.
    embeds[:, 5:7].square().sum().backward()
    assert policy.prototypes.grad is not None
    assert float(policy.prototypes.grad.norm()) > 0.0
    assert model.embedding.weight.grad is not None  # fake backbone is intentionally unfrozen
    print({
        "status": "PASS",
        "layout": "vision_end -> R1 R2 -> question",
        "relay_mm_types": augmented["mm_token_type_ids"][0, 5:7].tolist(),
        "image_placeholder_count": int(visual_mask.sum()),
        "trainable_relay_parameters": sum(p.numel() for p in policy.parameters()),
        "relay_gradient_norm": float(policy.prototypes.grad.norm()),
    })


if __name__ == "__main__":
    main()
