#!/usr/bin/env python3
"""Fast CPU smoke test; use train_roi_grpo with --limit 2 for GPU integration."""
from __future__ import annotations
import torch
from .roi_policy import DynamicROIPolicy, ROIPolicyConfig
from .spatial_mask import SoftROIMask, VisualTokenGrid
from .grpo import ROIGrpoUpdater, GRPOConfig


def main() -> None:
    torch.manual_seed(0); policy = DynamicROIPolicy(ROIPolicyConfig(hidden_size=32))
    state, prior = torch.randn(1, 32), torch.zeros(1, 9); mask = SoftROIMask(VisualTokenGrid(12, 16))
    actions = [policy.sample(state, prior), policy.sample(state, prior)]
    assert not torch.allclose(mask.bias(actions[0]["geometry"]), mask.bias(actions[1]["geometry"]))
    rows = [{"reward": float(i), "old_log_probability": float(a["log_prob"]),
             "actions": [{"features": a["features"].detach(), "latent": a["latent"].detach()}]} for i, a in enumerate(actions)]
    report = ROIGrpoUpdater(policy, GRPOConfig()).update_group(rows)
    assert report["grad_norm"] > 0
    print("CPU ROI smoke test passed:", report)


if __name__ == "__main__": main()
