"""Attention aggregation and spatial re-input policies.

This module deliberately separates SparseVLM's *token-pruning* rule from a
diagnostic / rolling-window region-ranking rule.  SparseVLM ranks each pruning
layer separately; it does not pool decoder layers before pruning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Aggregation = Literal["last", "mean", "max", "weighted_mean"]


def aggregate_decoder_layers(
    layer_maps: np.ndarray,
    method: Aggregation = "last",
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Pool ``[layer, patch_y, patch_x]`` attention maps for region ordering.

    ``last`` is the default because later decoder layers are closest to the
    next-token decision.  ``max`` is intentionally available only as an
    ablation/visualisation choice: it can over-rank a patch due to one noisy
    layer and is not the SparseVLM selection algorithm.
    """
    maps = np.asarray(layer_maps, dtype=np.float64)
    if maps.ndim != 3 or not all(maps.shape):
        raise ValueError("layer_maps must have shape [layers, patch_y, patch_x].")
    if method == "last":
        return maps[-1]
    if method == "mean":
        return maps.mean(axis=0)
    if method == "max":
        return maps.max(axis=0)
    if method == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires one weight per decoder layer.")
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (maps.shape[0],) or np.any(w < 0) or w.sum() == 0:
            raise ValueError("weights must be non-negative and match the layer count.")
        return np.average(maps, axis=0, weights=w)
    raise ValueError(f"Unknown aggregation method: {method}")


def sparsevlm_layer_scores(
    attention: np.ndarray,
    text_query_indices: Sequence[int],
    visual_key_indices: Sequence[int],
    *,
    priority_heads: Sequence[int] | None = None,
) -> np.ndarray:
    """Reproduce SparseVLM's scoring for exactly one pruning layer.

    ``attention`` is post-softmax ``[heads, query_tokens, key_tokens]``.
    The result is one score per visual token.  It intentionally accepts only a
    single layer, making accidental cross-layer aggregation impossible.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("attention must have shape [heads, queries, keys].")
    if priority_heads is not None:
        a = a[np.asarray(priority_heads, dtype=np.int64)]
    queries = np.asarray(text_query_indices, dtype=np.int64)
    keys = np.asarray(visual_key_indices, dtype=np.int64)
    if not len(queries) or not len(keys):
        raise ValueError("At least one text query and visual key is required.")
    # SparseVLM score.py: mean(head) -> select text queries -> mean(text).
    return a.mean(axis=0)[np.ix_(queries, keys)].mean(axis=0)


@dataclass(frozen=True)
class Region:
    """A merged-grid rectangle, half-open: [top:bottom, left:right]."""
    top: int
    left: int
    bottom: int
    right: int
    score: float


def top_regions(
    patch_scores: np.ndarray,
    count: int = 4,
    side: int = 4,
    iou_limit: float = 0.25,
) -> list[Region]:
    """Return high-score, low-overlap square regions in visual-priority order."""
    scores = np.asarray(patch_scores, dtype=np.float64)
    if scores.ndim != 2 or not all(scores.shape) or count < 1 or side < 1:
        raise ValueError("Invalid patch map, count, or region side.")
    h, w = scores.shape
    side = min(side, h, w)
    candidates: list[Region] = []
    for top in range(h - side + 1):
        for left in range(w - side + 1):
            candidates.append(Region(top, left, top + side, left + side,
                                     float(scores[top:top + side, left:left + side].mean())))
    candidates.sort(key=lambda r: r.score, reverse=True)
    selected: list[Region] = []
    for candidate in candidates:
        if all(_iou(candidate, prior) <= iou_limit for prior in selected):
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def salient_regions(
    patch_scores: np.ndarray,
    count: int = 4,
    attention_mass: float = 0.60,
    min_patches: int = 1,
) -> list[Region]:
    """Extract variable-sized connected high-attention patch components.

    Rather than sliding a fixed square across the map, retain the smallest set
    of patches whose scores account for ``attention_mass`` of all positive
    attention.  Four-connected retained patches form one crop each.  This is
    suitable for re-input: crop boundaries follow the attention support rather
    than a manually chosen square size.
    """
    scores = np.asarray(patch_scores, dtype=np.float64)
    if scores.ndim != 2 or not all(scores.shape) or count < 1:
        raise ValueError("patch_scores must be a non-empty 2D map and count must be positive.")
    if not 0 < attention_mass <= 1 or min_patches < 1:
        raise ValueError("attention_mass must be in (0, 1] and min_patches must be positive.")
    positive = np.maximum(scores, 0)
    total = float(positive.sum())
    if total <= np.finfo(np.float64).tiny:
        return []
    order = np.argsort(positive.ravel())[::-1]
    cumulative = np.cumsum(positive.ravel()[order])
    keep_count = max(1, int(np.searchsorted(cumulative, total * attention_mass, side="left")) + 1)
    mask = np.zeros(scores.shape, dtype=bool)
    mask.ravel()[order[:keep_count]] = True

    components: list[Region] = []
    visited = np.zeros_like(mask)
    height, width = mask.shape
    for row, col in zip(*np.nonzero(mask)):
        if visited[row, col]:
            continue
        stack, cells = [(int(row), int(col))], []
        visited[row, col] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(cells) >= min_patches:
            ys, xs = zip(*cells)
            components.append(Region(min(ys), min(xs), max(ys) + 1, max(xs) + 1,
                                     float(sum(positive[y, x] for y, x in cells))))
    return sorted(components, key=lambda region: region.score, reverse=True)[:count]


def _iou(a: Region, b: Region) -> float:
    height = max(0, min(a.bottom, b.bottom) - max(a.top, b.top))
    width = max(0, min(a.right, b.right) - max(a.left, b.left))
    intersection = height * width
    union = (a.bottom - a.top) * (a.right - a.left) + (b.bottom - b.top) * (b.right - b.left) - intersection
    return intersection / union if union else 0.0
