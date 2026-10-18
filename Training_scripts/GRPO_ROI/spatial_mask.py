"""Differentiable mapping from normalized ROIs to Qwen merged visual tokens."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class VisualTokenGrid:
    height: int
    width: int

    def coordinates(self, *, device=None, dtype=None) -> Tensor:
        y = (torch.arange(self.height, device=device, dtype=dtype) + .5) / self.height
        x = (torch.arange(self.width, device=device, dtype=dtype) + .5) / self.width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


class SoftROIMask:
    def __init__(self, grid: VisualTokenGrid, boundary_softness: float = .025, max_bias: float = 4.0):
        if boundary_softness <= 0 or max_bias <= 0:
            raise ValueError("boundary_softness and max_bias must be positive.")
        self.grid, self.boundary_softness, self.max_bias = grid, boundary_softness, max_bias

    def memberships(self, geometry: Tensor) -> Tensor:
        """Return ``[..., R, V]`` smooth rectangle memberships."""
        coords = self.grid.coordinates(device=geometry.device, dtype=geometry.dtype)
        centre, size = geometry[..., :2], geometry[..., 2:4]
        left_top, right_bottom = centre - size / 2, centre + size / 2
        xy = coords.view(*([1] * (geometry.ndim - 2)), 1, -1, 2)
        lo, hi = left_top.unsqueeze(-2), right_bottom.unsqueeze(-2)
        scale = self.boundary_softness
        return (torch.sigmoid((xy[..., 0] - lo[..., 0]) / scale) * torch.sigmoid((hi[..., 0] - xy[..., 0]) / scale)
                * torch.sigmoid((xy[..., 1] - lo[..., 1]) / scale) * torch.sigmoid((hi[..., 1] - xy[..., 1]) / scale))

    def bias(self, geometry: Tensor) -> Tensor:
        """Bounded noisy-OR combination; overlapping ROIs never exceed max_bias."""
        weighted = geometry[..., 4].unsqueeze(-1) * self.memberships(geometry)
        max_bias = geometry.new_tensor(self.max_bias)
        return max_bias * (1 - torch.exp((-weighted / max_bias).sum(dim=-2)))
