"""Render standalone Relay1 R1/R2 prefill attention viewers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _normalize(array: np.ndarray, robust: bool = True) -> np.ndarray:
    positive = array[array > 0]
    scale = float(np.quantile(positive, 0.99)) if robust and positive.size else float(array.max())
    if scale <= 0:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip(array / scale, 0.0, 1.0).astype(np.float32)


def _overlay(image: Image.Image, patch_map: np.ndarray) -> Image.Image:
    # The source maps can exceed 6000 px.  Browser inspection does not benefit
    # from saving 74 full-resolution overlays per question, and PNG encoding
    # becomes needlessly slow.  Preserve aspect ratio at a practical viewer size.
    if max(image.size) > 1800:
        image = image.copy()
        image.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = Image.fromarray(np.rint(_normalize(patch_map) * 255).astype(np.uint8))
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    value = np.asarray(heat, dtype=np.float32) / 255.0
    # Transparent yellow -> orange -> red, with zero attention fully transparent.
    color = np.stack((np.ones_like(value), 0.92 - 0.72 * value,
                      0.10 * (1.0 - value)), axis=-1) * 255.0
    alpha = (0.78 * value)[..., None]
    result = base * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.rint(np.clip(result, 0, 255)).astype(np.uint8))


def render_question(question_dir: Path) -> Path:
    record = json.loads((question_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(question_dir / "attention.npz") as archive:
        maps = archive["relay_attention"].astype(np.float32)
    if maps.ndim != 4 or maps.shape[0] != 2:
        raise ValueError(f"Expected [2,layers,y,x], got {maps.shape}")
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")
    images_dir = question_dir / "heatmaps"
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for relay in range(2):
        relay_entries = []
        layer_maps = [maps[relay, layer] for layer in range(maps.shape[1])]
        thresholds = np.quantile(maps[relay], 0.95, axis=(1, 2), keepdims=True)
        summaries = [
            ("all_layers_mean", maps[relay].mean(axis=0),
             "mean raw attention across all layers"),
            ("all_layers_max", maps[relay].max(axis=0),
             "sum of patch-wise layer maxima (not probability mass)"),
            ("top5_frequency", (maps[relay] >= thresholds).mean(axis=0),
             "fraction of layers where patch is in that layer's top 5%"),
        ]
        layer_maps.extend(item[1] for item in summaries)
        for layer, patch_map in enumerate(layer_maps):
            if layer < maps.shape[1]:
                name = f"relay_{relay + 1}_layer_{layer:02d}.png"
                label = "image attention mass"
            else:
                summary_name, _, label = summaries[layer - maps.shape[1]]
                name = f"relay_{relay + 1}_{summary_name}.png"
            _overlay(image, patch_map).save(images_dir / name)
            relay_entries.append({
                "src": f"heatmaps/{name}",
                "raw_max": float(patch_map.max()),
                "aggregate_value": float(patch_map.sum()),
                "aggregate_label": label,
            })
        entries.append(relay_entries)

    payload = {
        "record": record,
        "layers": int(maps.shape[1]),
        "entries": entries,
    }
    html = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    destination = question_dir / "relay1_attention.html"
    destination.write_text(
        html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")),
        encoding="utf-8",
    )
    return destination
