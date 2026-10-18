"""Causally available attention-guided 3x3 ROI spatial prior."""
from __future__ import annotations
import torch
from torch import Tensor
from .roi_policy import regular_grid
from .spatial_mask import VisualTokenGrid


def attention_guided_prior(attention: Tensor | None, grid: VisualTokenGrid, *, roi_count: int = 9,
                           alpha: float = .5, minimum_mass: float = 1e-8) -> Tensor:
    """Return [R,4] grid geometry with centres moved to local attention centroids.

    ``attention`` is the preceding token's already head/layer-averaged vector
    over *only global image keys*.  Low-mass cells fall back exactly to their
    regular-grid centres; widths/heights intentionally remain grid-cell size.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("attention prior alpha must be in [0, 1].")
    base = regular_grid(roi_count, device=attention.device if attention is not None else None,
                        dtype=attention.dtype if attention is not None else None)
    if attention is None:
        return base
    if attention.numel() != grid.height * grid.width or not torch.isfinite(attention).all():
        raise ValueError("Attention must be finite and match the global visual grid.")
    side = int(roi_count ** .5)
    if side * side != roi_count:
        raise ValueError("roi_count must form a square grid.")
    coords = grid.coordinates(device=attention.device, dtype=attention.dtype).reshape(grid.height, grid.width, 2)
    weights = attention.clamp_min(0).reshape(grid.height, grid.width)
    centres = base[:, :2].clone()
    for row in range(side):
        for col in range(side):
            index = row * side + col
            y0, y1 = row * grid.height // side, (row + 1) * grid.height // side
            x0, x1 = col * grid.width // side, (col + 1) * grid.width // side
            local_weight = weights[y0:y1, x0:x1]
            mass = local_weight.sum()
            if mass > minimum_mass:
                centroid = (coords[y0:y1, x0:x1] * local_weight.unsqueeze(-1)).sum((0, 1)) / mass
                centres[index] = base[index, :2] + alpha * (centroid - base[index, :2])
    return torch.cat((centres.clamp(0, 1), base[:, 2:]), dim=-1)
