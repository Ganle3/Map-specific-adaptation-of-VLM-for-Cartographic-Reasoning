"""Lightweight regression test for generation without GenerationMixin."""
from types import SimpleNamespace

import torch

from prototype_prefill import generate_from_preencoded_vision


class _Cache:
    def __init__(self, length: int):
        self.length = length

    def get_seq_length(self):
        return self.length


class _VLBackbone:
    def get_rope_index(self, input_ids, _types, **_kwargs):
        positions = torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long)
        return positions, None


class _ConditionalModel:
    def __init__(self):
        self.model = _VLBackbone()
        self.generation_config = SimpleNamespace(eos_token_id=3)


class _PeftLikeWrapper:
    def __init__(self):
        self.base = _ConditionalModel()
        self.calls = 0

    def get_base_model(self):
        return self.base

    def __call__(self, input_ids, past_key_values=None, **_kwargs):
        self.calls += 1
        length = (input_ids.shape[1] if past_key_values is None
                  else past_key_values.get_seq_length() + 1)
        logits = torch.full((1, input_ids.shape[1], 5), -10.0)
        logits[:, -1, 2 if self.calls == 1 else 3] = 10.0
        return SimpleNamespace(logits=logits, past_key_values=_Cache(length))


def test_prefill_kv_and_eos():
    model = _PeftLikeWrapper()
    input_ids = torch.tensor([[7, 8]])
    output = generate_from_preencoded_vision(
        model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=torch.zeros_like(input_ids),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_encoder_outputs={"image": "cached"},
        do_sample=False,
        max_new_tokens=4,
        return_dict_in_generate=True,
        output_scores=True,
    )
    assert output.sequences.tolist() == [[7, 8, 2, 3]]
    assert len(output.scores) == 2
    assert model.calls == 2


if __name__ == "__main__":
    test_prefill_kv_and_eos()
    print("PASS: visual prefill -> cached token -> EOS")
