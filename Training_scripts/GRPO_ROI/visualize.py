"""Small diagnostics for soft masks and token-level ROI trajectories."""
from __future__ import annotations
from pathlib import Path
import torch
from .roi_policy import DynamicROIPolicy
from .spatial_mask import SoftROIMask


def save_roi_diagnostic(image, geometry: torch.Tensor, mask: SoftROIMask, output: str | Path) -> None:
    """Render normalized ROI boxes and the resulting merged-token soft bias."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    geometry = geometry.detach().cpu()
    if geometry.ndim == 3: geometry = geometry[0]
    bias = mask.bias(geometry).detach().cpu().reshape(mask.grid.height, mask.grid.width)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    width, height = image.size
    for left_top, right_bottom in DynamicROIPolicy.corners(geometry):
        left, top = left_top.tolist(); right, bottom = right_bottom.tolist()
        axes[0].add_patch(Rectangle((left*width, top*height), (right-left)*width, (bottom-top)*height,
                                    fill=False, edgecolor="lime", linewidth=1.5))
    axes[0].set_axis_off(); axes[1].imshow(bias, cmap="magma", vmin=0, vmax=4); axes[1].set_title("ROI attention-logit bias")
    figure.tight_layout(); figure.savefig(output, dpi=160); plt.close(figure)
