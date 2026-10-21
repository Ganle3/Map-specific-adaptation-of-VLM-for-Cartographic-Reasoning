"""Small CPU checks for action geometry and GRPO gradient; run when torch is installed."""
import torch

from region_grpo import region_grpo_loss
from region_policy import NineGridRegionPolicy, nine_grid_boxes


def main():
    for width, height in ((3, 3), (10, 11), (101, 77)):
        boxes = nine_grid_boxes(width, height)
        pixels = set()
        for left, top, right, bottom in boxes:
            assert left < right and top < bottom
            for x in range(left, right):
                for y in range(top, bottom):
                    assert (x, y) not in pixels
                    pixels.add((x, y))
        assert len(pixels) == width * height
    torch.manual_seed(3)
    policy = NineGridRegionPolicy(16)
    h1, h2 = torch.randn(16), torch.randn(16)
    assert not torch.allclose(policy(h1[None], sample=False)[1], policy(h2[None], sample=False)[1])
    old1 = float(policy.distribution(h1[None]).log_prob(torch.tensor([0]))[0].detach())
    old2 = float(policy.distribution(h2[None]).log_prob(torch.tensor([1]))[0].detach())
    group = [{"reward": 1.0, "actions": [{"hidden": h1, "action": 0, "old_log_prob": old1}]},
             {"reward": 0.0, "actions": [{"hidden": h2, "action": 1, "old_log_prob": old2}]}]
    before = policy.nine_blocks.detach().clone()
    optimizer = torch.optim.AdamW([policy.nine_blocks], lr=0.01)
    loss, _ = region_grpo_loss(policy, group)
    loss.backward()
    assert policy.nine_blocks.grad is not None and policy.nine_blocks.grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, policy.nine_blocks)
    assert region_grpo_loss(policy, [{"reward": 0, "actions": []}, {"reward": 1, "actions": []}])[0] is None
    print("grid coverage, state dependence, region GRPO gradient and update: PASS")


if __name__ == "__main__":
    main()
