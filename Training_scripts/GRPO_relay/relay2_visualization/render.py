"""Render the four Relay2 placement/token attention streams."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from relay1_visualization.render import _overlay


STREAMS = ("image-side R1", "image-side R2", "question-side R1", "question-side R2")


def render_question(directory: Path):
    record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    with np.load(directory / "attention.npz") as archive:
        maps = archive["relay_attention"].astype(np.float32)
    if maps.ndim != 5 or maps.shape[:2] != (2, 2):
        raise ValueError(f"Expected [2 placements,2 relays,layers,y,x], got {maps.shape}")
    flat = maps.reshape(4, *maps.shape[2:])
    with Image.open(record["image"]) as handle:
        image = handle.convert("RGB")
    heatmaps = directory / "heatmaps"; heatmaps.mkdir(exist_ok=True)
    entries = []
    for stream, array in enumerate(flat):
        thresholds = np.quantile(array, .95, axis=(1, 2), keepdims=True)
        summaries = [
            ("mean", array.mean(0), "mean raw attention across layers"),
            ("max", array.max(0), "sum of patch-wise layer maxima"),
            ("top5_frequency", (array >= thresholds).mean(0), "sum of top-5% layer frequencies"),
        ]
        frames = [(f"layer_{layer:02d}", array[layer], "image attention mass")
                  for layer in range(array.shape[0])] + summaries
        stream_entries = []
        for name, patch_map, label in frames:
            filename = f"stream_{stream}_{name}.png"
            _overlay(image, patch_map).save(heatmaps / filename)
            stream_entries.append({"src": f"heatmaps/{filename}", "value": float(patch_map.sum()),
                                   "label": label, "raw_max": float(patch_map.max())})
        entries.append(stream_entries)
    payload = {"record": record, "layers": int(flat.shape[1]), "entries": entries,
               "stream_names": STREAMS}
    html = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    destination = directory / "relay2_attention.html"
    destination.write_text(html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")), encoding="utf-8")
    return destination

