"""Original-coordinate tile geometry and lazy CPU feature cache."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image


@dataclass(frozen=True)
class TileGeometry:
    action: int
    row: int
    col: int
    merged_grid_height: int
    merged_grid_width: int
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["bbox"] = list(self.bbox)
        return value


class CoarseImageGrid:
    """Map merger-token indices to windows in original-image coordinates."""

    def __init__(self, *, original_size: tuple[int, int], image_grid_thw: torch.Tensor,
                 spatial_merge_size: int, expected_tokens: int,
                 window_side: int = 5, margin_fraction: float = 0.10) -> None:
        if image_grid_thw.numel() != 3:
            raise ValueError("The demo expects exactly one image_grid_thw row")
        temporal, patch_h, patch_w = map(int, image_grid_thw.reshape(-1).tolist())
        if temporal != 1:
            raise ValueError("Static-image demo expected temporal grid size 1")
        if patch_h % spatial_merge_size or patch_w % spatial_merge_size:
            raise ValueError("Patch grid is not divisible by spatial merge size")
        self.height = patch_h // spatial_merge_size
        self.width = patch_w // spatial_merge_size
        if self.height * self.width != int(expected_tokens):
            raise RuntimeError(
                f"Merged grid {self.height}x{self.width} does not match "
                f"{expected_tokens} global visual tokens"
            )
        if window_side < 1 or window_side % 2 != 1:
            raise ValueError("window_side must be a positive odd integer")
        if margin_fraction < 0:
            raise ValueError("margin_fraction must be non-negative")
        self.original_width, self.original_height = map(int, original_size)
        self.window_side = int(window_side)
        self.margin_fraction = float(margin_fraction)

    def geometry(self, action: int) -> TileGeometry:
        if not 0 <= action < self.height * self.width:
            raise IndexError(f"Coarse action {action} is outside the global grid")
        row, col = divmod(int(action), self.width)
        radius = self.window_side // 2
        row0, row1 = max(0, row - radius), min(self.height, row + radius + 1)
        col0, col1 = max(0, col - radius), min(self.width, col + radius + 1)

        x0 = col0 * self.original_width / self.width
        x1 = col1 * self.original_width / self.width
        y0 = row0 * self.original_height / self.height
        y1 = row1 * self.original_height / self.height
        margin_x = (x1 - x0) * self.margin_fraction
        margin_y = (y1 - y0) * self.margin_fraction
        bbox = (
            max(0, int(x0 - margin_x)),
            max(0, int(y0 - margin_y)),
            min(self.original_width, int(x1 + margin_x + 0.999999)),
            min(self.original_height, int(y1 + margin_y + 0.999999)),
        )
        return TileGeometry(
            action=int(action), row=row, col=col,
            merged_grid_height=self.height, merged_grid_width=self.width,
            bbox=bbox,
        )


class OnDemandTileFeatureCache:
    """Encode a selected original-image tile once and retain it on CPU."""

    def __init__(self, *, model, processor, original_image: Image.Image,
                 device, max_pixels: int, output_dir: Path) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.model = model
        self.processor = processor
        self.original_image = original_image.copy().convert("RGB")
        self.device = device
        self.max_pixels = int(max_pixels)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cpu_features: dict[tuple[int, int, int, int], torch.Tensor] = {}
        self._metadata: dict[tuple[int, int, int, int], dict] = {}
        self.encode_count = 0
        self.hit_count = 0

    def get(self, geometry: TileGeometry):
        key = tuple(geometry.bbox)
        if key in self._cpu_features:
            self.hit_count += 1
            return self._cpu_features[key].to(self.device, non_blocking=True), True, self._metadata[key]

        # Imports remain local so geometry unit tests do not require the full
        # Transformers/Qwen environment.
        import sys
        relay_root = Path(__file__).resolve().parents[1]
        vlm_root = relay_root.parents[1]
        sparse_root = vlm_root / "SparseVLM_eliminateCONTEXT"
        for path in (relay_root, sparse_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
        from utils import encode_global_image

        crop = self.original_image.crop(key)
        submitted = _resize_to_pixel_cap(crop, self.max_pixels)
        crop_path = self.output_dir / f"tile_action_{geometry.action:04d}_{key[0]}_{key[1]}_{key[2]}_{key[3]}.png"
        crop.save(crop_path)
        inputs = _content_inputs(
            self.processor, [submitted],
            "Inspect this local map region for labels, legend colors, and spatial evidence.",
            self.device, self.max_pixels,
        )
        features = encode_global_image(self.model, inputs)
        pooled = features.pooler_output
        pooled = pooled if isinstance(pooled, torch.Tensor) else torch.cat(tuple(pooled), dim=0)
        cpu = pooled.detach().to("cpu", copy=True).contiguous()
        metadata = {
            **geometry.to_dict(),
            "crop_path": str(crop_path),
            "original_crop_size": [crop.width, crop.height],
            "submitted_size": [submitted.width, submitted.height],
            "local_visual_tokens": int(cpu.shape[0]),
            "local_max_pixels": self.max_pixels,
        }
        self._cpu_features[key] = cpu
        self._metadata[key] = metadata
        self.encode_count += 1
        return cpu.to(self.device, non_blocking=True), False, metadata

    def summary(self) -> dict:
        return {
            "unique_tiles_encoded": self.encode_count,
            "cpu_cache_hits": self.hit_count,
            "cached_bboxes": [list(key) for key in self._cpu_features],
        }

