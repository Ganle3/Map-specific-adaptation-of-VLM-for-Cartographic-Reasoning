"""CPU-only relay placement regression test; no Qwen checkpoint required."""
from types import SimpleNamespace

import torch

from utils import build_relay_prefill, explicit_relay_decoder_inputs
from only_visual_relay.policy import OnlyVisualRelayPolicy
from Image_question_shared_relay.policy import ImageQuestionSharedRelayPolicy


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=10, video_token_id=11,
            vision_start_token_id=12, vision_end_token_id=13,
            pad_token_id=0,
        )
        self.embedding = torch.nn.Embedding(128, 8)

    def get_input_embeddings(self):
        return self.embedding


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        assert text == "<|im_end|>"
        return [99]


def source():
    # vision_start, two image placeholders, vision_end, question x2,
    # user im_end, assistant prefix
    ids = torch.tensor([[12, 10, 10, 13, 20, 21, 99, 30]])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0]]),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }


def main():
    model, tokenizer = FakeModel(), FakeTokenizer()
    features = SimpleNamespace(
        pooler_output=(torch.zeros(2, 8),), deepstack_features=[],
    )
    only = OnlyVisualRelayPolicy(8, num_relays=2)
    dual = ImageQuestionSharedRelayPolicy(8, num_relays=2)
    one, _ = build_relay_prefill(
        model, tokenizer, source(), features, only,
        scheme="only_visual", with_grad=True,
    )
    two, _ = build_relay_prefill(
        model, tokenizer, source(), features, dual,
        scheme="image_question_shared", with_grad=True,
    )
    assert one["relay_positions"].tolist() == [[4, 5]]
    assert two["relay_positions"].tolist() == [[4, 5], [8, 9]]
    assert one["input_ids"].shape[1] == 10
    assert two["input_ids"].shape[1] == 12
    assert two["relay_embeddings"] is dual.prototypes
    assert two["relay_embeddings"].shape == (2, 8)
    assert two["mm_token_type_ids"][0, two["relay_positions"].flatten()].tolist() == [0, 0, 0, 0]
    embeds, _, _ = explicit_relay_decoder_inputs(
        model, two["input_ids"], features, two["relay_embeddings"],
        two["relay_positions"],
    )
    embeds[0, two["relay_positions"].flatten()].sum().backward()
    # Both placements contribute one unit to the same parameter bank.
    assert torch.equal(dual.prototypes.grad, torch.full_like(dual.prototypes, 2.0))
    print({
        "status": "PASS",
        "only_visual_positions": one["relay_positions"].tolist(),
        "shared_dual_positions": two["relay_positions"].tolist(),
        "parameter_shape": list(dual.prototypes.shape),
        "parameter_object_count": 1,
        "shared_gradient_per_element": float(dual.prototypes.grad[0, 0]),
    })


if __name__ == "__main__":
    main()
