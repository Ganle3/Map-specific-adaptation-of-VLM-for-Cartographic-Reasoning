"""CPU-only checks for policy geometry, mask changes, and PPO gradients."""
import torch
from .roi_policy import DynamicROIPolicy, ROIPolicyConfig
from .spatial_mask import SoftROIMask, VisualTokenGrid
from .grpo import ROIGrpoUpdater, GRPOConfig
from .attention_prior import attention_guided_prior


def test_geometry_mask_and_update():
    torch.manual_seed(7); policy = DynamicROIPolicy(ROIPolicyConfig(hidden_size=8))
    assert any(p.requires_grad for p in policy.parameters())
    hidden, summary = torch.randn(1, 8), torch.zeros(1, 9)
    first, second = policy.sample(hidden, summary), policy.sample(hidden, summary)
    geometry = first["geometry"]
    assert torch.isfinite(first["log_prob"]).all() and geometry[..., :4].ge(0).all() and geometry[..., :4].le(1).all()
    assert geometry[..., 2:4].gt(0).all()
    mask = SoftROIMask(VisualTokenGrid(6, 8)); one, two = mask.bias(first["geometry"]), mask.bias(second["geometry"])
    assert one.shape[-1] == 48 and not torch.allclose(one, two)
    trajectories = []
    for sample, reward in ((first, 1.0), (second, 0.0)):
        trajectories.append({"reward": reward, "old_log_probability": float(sample["log_prob"]),
                             "actions": [{"features": sample["features"].detach(), "latent": sample["latent"].detach()}]})
    before = policy.mean_head.bias.detach().clone(); metrics = ROIGrpoUpdater(policy, GRPOConfig()).update_group(trajectories)
    assert metrics["grad_norm"] > 0 and not torch.equal(before, policy.mean_head.bias)


def test_attention_guided_prior_and_zero_residual():
    grid = VisualTokenGrid(6, 6)
    left = torch.zeros(36); left[0:2] = 1
    right = torch.zeros(36); right[4:6] = 1
    prior_left = attention_guided_prior(left, grid, alpha=.5)
    prior_right = attention_guided_prior(right, grid, alpha=.5)
    assert not torch.allclose(prior_left, prior_right)
    fixed = attention_guided_prior(left, grid, alpha=0)
    assert torch.allclose(fixed, DynamicROIPolicy(ROIPolicyConfig(hidden_size=2)).base_geometry)
    policy = DynamicROIPolicy(ROIPolicyConfig(hidden_size=2))
    out = policy.sample(torch.zeros(1, 2), torch.zeros(1, 9), prior_geometry=prior_left, deterministic=True)
    # Zero residual latent leaves the supplied attention-guided centres intact.
    assert torch.allclose(out["geometry"][..., :2], prior_left[..., :2])
