"""Tiny Qwen3-VL decoder compatibility test for the scheme-3 step forward."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from Image_question_CoT_shared_relay.dynamic_decode import _decode_pair


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        config = Qwen3VLTextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        )
        self.language_model = Qwen3VLTextModel(config)
        self.model = SimpleNamespace(language_model=self.language_model)
        self.lm_head = nn.Linear(32, 64, bias=False)

    def get_input_embeddings(self):
        return self.language_model.embed_tokens


def main() -> None:
    torch.manual_seed(7)
    model = TinyQwen().eval()
    prefix = torch.tensor([[3, 4, 5]])
    positions = torch.arange(3).view(1, 1, 3).expand(3, 1, 3)
    prefill = model.language_model(
        input_ids=prefix, attention_mask=torch.ones_like(prefix),
        position_ids=positions, use_cache=True, return_dict=True,
    )
    relay = nn.Parameter(torch.randn(32) * 0.02)
    visible = torch.tensor([False, True, True])
    step_position = torch.full((3, 1, 1), 3, dtype=torch.long)
    logits, cache = _decode_pair(
        model, token=torch.tensor(6), relay=relay, position=step_position,
        past=prefill.past_key_values, relay_visible_past=visible,
    )
    assert logits.shape == (1, 64)
    assert cache.get_seq_length() == 4
    logits[0, 9].backward()
    assert relay.grad is not None and bool(relay.grad.count_nonzero())
    print({
        "status": "PASS", "decoder_layers": 3,
        "cache_before_pair": 3, "cache_after_crop": cache.get_seq_length(),
        "relay_gradient_norm": float(relay.grad.norm()),
    })


if __name__ == "__main__":
    main()
