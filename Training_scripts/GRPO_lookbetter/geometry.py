"""Frozen proposal features and legal discrete crop-size actions."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Proposal:
    """Original-pixel bbox paired with the component that produced it."""
    left: int
    top: int
    right: int
    bottom: int
    score: float
    grid_region: object


def jsd_features(current: np.ndarray, previous: np.ndarray | None, looks_used: int, max_looks: int) -> np.ndarray:
    """[JSD, concentration, positive redistribution growth, remaining budget]."""
    p = np.maximum(np.asarray(current, dtype=np.float64).reshape(-1), 0)
    p /= max(p.sum(), np.finfo(np.float64).tiny)
    entropy = -(p * np.log(np.maximum(p, np.finfo(np.float64).tiny))).sum()
    concentration = 1.0 - entropy / math.log(max(p.size, 2))
    if previous is None:
        jsd = growth = 0.0
    else:
        q = np.maximum(np.asarray(previous, dtype=np.float64).reshape(-1), 0)
        q /= max(q.sum(), np.finfo(np.float64).tiny)
        midpoint = 0.5 * (p + q)
        jsd = 0.5 * ((p * np.log(np.maximum(p, 1e-12) / np.maximum(midpoint, 1e-12))).sum() +
                     (q * np.log(np.maximum(q, 1e-12) / np.maximum(midpoint, 1e-12))).sum())
        # Positive redistribution: mass newly assigned to patches, normalized
        # to [0,1]. It is deterministic and insensitive to total attention.
        growth = float(np.maximum(p - q, 0).sum())
    budget = max(0.0, 1.0 - looks_used / max(max_looks, 1))
    return np.asarray([jsd, concentration, growth, budget], dtype=np.float32)


def region_features(regions, score_map: np.ndarray, raw_attention_map: np.ndarray, visited) -> tuple[np.ndarray, np.ndarray]:
    """Build [score, mass, area, dispersion, rank, novelty] for PixelRegions.

    Region bounds here are in global-grid coordinates supplied separately by
    the caller through the temporary ``grid_region`` members.  The production
    rollout attaches these before calling this helper.
    """
    score = np.maximum(np.asarray(score_map, dtype=np.float64), 0)
    raw = np.maximum(np.asarray(raw_attention_map, dtype=np.float64), 0)
    score_scale = max(float(score.max()), 1e-12)
    raw_total = max(float(raw.sum()), 1e-12)
    height, width = score.shape
    rows, priors = [], []
    for rank, region in enumerate(regions):
        local = region.grid_region
        values = score[local.top:local.bottom, local.left:local.right]
        attention = raw[local.top:local.bottom, local.left:local.right]
        mass = float(attention.sum() / raw_total)
        normalized = attention / max(float(attention.sum()), 1e-12)
        dispersion = float(-(normalized * np.log(np.maximum(normalized, 1e-12))).sum() / math.log(max(values.size, 2)))
        area = values.size / (height * width)
        novelty = min(1.0, min((_iou_pixels(region, old) for old in visited), default=0.0))
        rows.append([float(values.sum() / score_scale), mass, area, dispersion, rank / max(len(regions) - 1, 1), 1.0 - novelty])
        # Monotone calibrated prior: top candidates begin likely but no action
        # is deterministic. Values are logits, not trainable parameters.
        priors.append(1.4 - 2.8 * rank / max(len(regions) - 1, 1))
    return np.asarray(rows, np.float32), np.asarray(priors, np.float32)


def legal_size_actions(region, grid_shape: tuple[int, int], image_size: tuple[int, int]):
    """Return valid pixel boxes centered on a proposal and their log-ratios.

    Candidate dimensions are in *global visual patch* units. Invalid pairs are
    rejected before sampling; no post-sample clamping is performed.
    """
    gh, gw = grid_shape
    image_w, image_h = image_size
    base_w = max(1, region.grid_region.right - region.grid_region.left)
    base_h = max(1, region.grid_region.bottom - region.grid_region.top)
    max_area = max(1, (gh * gw) // 4)
    cx = (region.grid_region.left + region.grid_region.right) / 2
    cy = (region.grid_region.top + region.grid_region.bottom) / 2
    choices = []
    for w in range(1, gw + 1):
        for h in range(1, gh + 1):
            left, right = int(math.floor(cx - w / 2)), int(math.ceil(cx + w / 2))
            top, bottom = int(math.floor(cy - h / 2)), int(math.ceil(cy + h / 2))
            if w * h > max_area or left < 0 or top < 0 or right > gw or bottom > gh or right <= left or bottom <= top:
                continue
            px_left, px_right = left * image_w // gw, math.ceil(right * image_w / gw)
            px_top, px_bottom = top * image_h // gh, math.ceil(bottom * image_h / gh)
            choices.append(((px_left, px_top, max(px_left + 1, px_right), max(px_top + 1, px_bottom)),
                            (math.log(w / base_w), math.log(h / base_h))))
    if not choices:
        raise RuntimeError("No legal crop size actions for proposal.")
    return choices


def _iou_pixels(a, b) -> float:
    left, top = max(a.left, b.left), max(a.top, b.top)
    right, bottom = min(a.right, b.right), min(a.bottom, b.bottom)
    inter = max(0, right - left) * max(0, bottom - top)
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union else 0.0
