"""Render all-decoder-layer attention maps by generation step and full inference."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _encode(array: np.ndarray) -> str:
    return base64.b64encode(array.astype("<f4").tobytes()).decode("ascii")


def render_comparison(directory: Path) -> Path:
    records, steps, layer_summaries, step_layer_maxes, captured_layer_maxes = [], [], [], [], []
    for variant in ("baseline", "adapted"):
        variant_dir = directory / variant
        records.append(json.loads((variant_dir / "result.json").read_text(encoding="utf-8")))
        with np.load(variant_dir / "all_layer_steps.npz") as archive:
            steps.append(archive["direct_layer_sum"])
            step_layer_maxes.append(archive["direct_layer_max"] if "direct_layer_max" in archive else None)
        with np.load(variant_dir / "full_layer_summary.npz") as archive:
            layer_summaries.append(archive["raw"])
        # attention.npz retains every step for the user-selected diagnostic
        # layers (normally 8, 16, 24, 35), unlike the all-36-layer summary.
        with np.load(variant_dir / "attention.npz") as archive:
            detailed = archive["attention"]  # [step, captured_layer, query_head, y, x]
            captured_layer_maxes.append(detailed.mean(axis=2).max(axis=1).astype(np.float32))
    for key in ("merged_grid_hw", "prompt", "image"):
        if records[0][key] != records[1][key]:
            raise ValueError(f"Cannot compare incompatible {key} values.")
    mean_maps = [a.mean(axis=0) for a in steps]
    literal_maps = [a.sum(axis=0) for a in steps]
    max_maps = [a.max(axis=0) for a in steps]
    p95_maps = [np.quantile(a, 0.95, axis=0).astype(np.float32) for a in steps]
    # For every image patch, retain the strongest complete-inference signal
    # from the 36 decoder layers.  Normalization happens only after this max.
    layer_max_maps = [a.max(axis=0).astype(np.float32) for a in layer_summaries]
    has_step_layermax = all(a is not None for a in step_layer_maxes)
    has_captured_layermax = bool(captured_layer_maxes)
    step_layer_max_mean_maps = [a.mean(axis=0).astype(np.float32) for a in step_layer_maxes] if has_step_layermax else None
    # Each inference step is put on its own robust [0, 1] visual scale before
    # aggregation.  This intentionally measures repeated *relative* spatial
    # focus, rather than raw attention mass.
    step_p99_normalized_sum_maps = []
    above_step_mean_rate_maps = []
    above_step_median_rate_maps = []
    for array in steps:
        step_p99 = np.quantile(array, 0.99, axis=(1, 2), keepdims=True)
        normalized = np.minimum(array / np.maximum(step_p99, 1e-30), 1.0)
        step_p99_normalized_sum_maps.append(normalized.sum(axis=0).astype(np.float32))
        # A patch receives one vote when it is above the spatial baseline for
        # that particular generation step. Dividing by step count removes the
        # baseline/adapted response-length advantage.
        step_mean = array.mean(axis=(1, 2), keepdims=True)
        step_median = np.median(array, axis=(1, 2), keepdims=True)
        above_step_mean_rate_maps.append((array > step_mean).mean(axis=0).astype(np.float32))
        above_step_median_rate_maps.append((array > step_median).mean(axis=0).astype(np.float32))
    scales = {
        "step": float(max(a.max() for a in steps)),
        "mean": float(max(a.max() for a in mean_maps)),
        "literal": float(max(a.max() for a in literal_maps)),
        "max": float(max(a.max() for a in max_maps)),
        "p95": float(max(a.max() for a in p95_maps)),
        "stepnorm": float(max(a.max() for a in step_p99_normalized_sum_maps)),
        "above_mean_rate": 1.0,
        "above_median_rate": 1.0,
        "layermax": float(max(a.max() for a in layer_max_maps)),
    }
    p99_scales = {
        "step": float(max(np.quantile(a, 0.99) for a in steps)),
        "mean": float(max(np.quantile(a, 0.99) for a in mean_maps)),
        "literal": float(max(np.quantile(a, 0.99) for a in literal_maps)),
        "max": float(max(np.quantile(a, 0.99) for a in max_maps)),
        "p95": float(max(np.quantile(a, 0.99) for a in p95_maps)),
        "stepnorm": float(max(np.quantile(a, 0.99) for a in step_p99_normalized_sum_maps)),
        "above_mean_rate": float(max(np.quantile(a, 0.99) for a in above_step_mean_rate_maps)),
        "above_median_rate": float(max(np.quantile(a, 0.99) for a in above_step_median_rate_maps)),
        "layermax": float(max(np.quantile(a, 0.99) for a in layer_max_maps)),
    }
    if has_step_layermax:
        scales["layermax_step"] = float(max(a.max() for a in step_layer_maxes))
        p99_scales["layermax_step"] = float(max(np.quantile(a, 0.99) for a in step_layer_maxes))
    if has_captured_layermax:
        scales["layermax_captured"] = float(max(a.max() for a in captured_layer_maxes))
        p99_scales["layermax_captured"] = float(max(np.quantile(a, 0.99) for a in captured_layer_maxes))
    for index, (record, array, stepnorm_map, mean_rate_map, median_rate_map, layermax_map) in enumerate(zip(
        records, steps, step_p99_normalized_sum_maps,
        above_step_mean_rate_maps, above_step_median_rate_maps, layer_max_maps,
    )):
        record["all_layer_steps_f32"] = _encode(array)
        record["all_layer_steps_shape"] = list(array.shape)
        record["step_p99_normalized_sum_f32"] = _encode(stepnorm_map)
        record["above_step_mean_rate_f32"] = _encode(mean_rate_map)
        record["above_step_median_rate_f32"] = _encode(median_rate_map)
        record["layer_max_f32"] = _encode(layermax_map)
        record["captured_layer_step_max_f32"] = _encode(captured_layer_maxes[index])
        record["captured_layer_step_max_shape"] = list(captured_layer_maxes[index].shape)
        if has_step_layermax:
            record["all_layer_step_max_f32"] = _encode(step_layer_maxes[index])
    with Image.open(records[0]["image"]) as source:
        image = source.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {
        "records": records,
        "vmax": scales,
        "p99_vmax": p99_scales,
        "has_step_layermax": has_step_layermax,
        "has_captured_layermax": has_captured_layermax,
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    output = directory / "comparison.html"
    output.write_text(
        template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")),
        encoding="utf-8",
    )
    render_map(directory, records, mean_maps, scales["mean"], image,
               "all_layers_step_mean.png", "Mean of direct all-layer maps across all generated steps")
    render_map(directory, records, literal_maps, scales["literal"], image,
               "all_layers_step_literal_sum.png", "Literal sum of direct all-layer maps across all generated steps")
    p99 = p99_scales["mean"]
    render_map(directory, records, mean_maps, p99, image,
               "all_layers_step_mean_p99_contrast.png", "All-layer step mean; display clipped at shared 99th percentile")
    # Keep the earlier filename: summing per-layer step means equals taking the
    # mean of each step's direct all-layer sum, so the underlying map is equal.
    render_map(directory, records, mean_maps, p99, image,
               "all_layers_raw_sum_p99_contrast.png", "All-layer step mean; display clipped at shared 99th percentile")
    render_map(directory, records, max_maps, p99_scales["max"], image,
               "all_layers_step_max_p99_contrast.png", "All-layer maximum over generated steps; display clipped at shared 99th percentile")
    render_map(directory, records, p95_maps, p99_scales["p95"], image,
               "all_layers_step_p95_p99_contrast.png", "All-layer 95th percentile over generated steps; display clipped at shared 99th percentile")
    render_map(directory, records, step_p99_normalized_sum_maps, p99_scales["stepnorm"], image,
               "all_layers_step_p99_normalized_sum_p99_contrast.png",
               "Each step p99-normalized then summed; display clipped at shared 99th percentile")
    render_map(directory, records, above_step_mean_rate_maps, p99_scales["above_mean_rate"], image,
               "all_layers_above_step_mean_rate_p99_contrast.png",
               "Fraction of generated steps where each patch exceeds that step's spatial mean; display clipped at shared 99th percentile")
    render_map(directory, records, above_step_median_rate_maps, p99_scales["above_median_rate"], image,
               "all_layers_above_step_median_rate_p99_contrast.png",
               "Fraction of generated steps where each patch exceeds that step's spatial median; display clipped at shared 99th percentile")
    render_map(directory, records, layer_max_maps, p99_scales["layermax"], image,
               "all_layers_patch_max_p99_mask.png",
               "For each patch: maximum complete-inference attention across all decoder layers; normalized after layer max and clipped at shared 99th percentile")
    if has_step_layermax:
        render_map(directory, records, step_layer_max_mean_maps, p99_scales["layermax_step"], image,
                   "all_steps_layer_max_mean_p99_mask.png",
                   "At each step take the patch-wise maximum across decoder layers, then mean across generated steps; display clipped at shared 99th percentile")
    captured_layer_names = ", ".join(str(i) for i in records[0]["settings"]["layers"])
    render_map(directory, records, [a.mean(axis=0) for a in captured_layer_maxes], p99_scales["layermax_captured"], image,
               "captured_layers_step_max_mean_p99_mask.png",
               f"At each step take the patch-wise maximum across captured layers {captured_layer_names}, then mean across generated steps; display clipped at shared 99th percentile")
    return output


def render_map(directory, records, maps, maximum, image, filename, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
    for ax, record, heat in zip(axes, records, maps):
        ax.imshow(image)
        ax.imshow(np.zeros_like(heat), extent=(-0.5, image.width - 0.5, image.height - 0.5, -0.5),
                  interpolation="nearest", cmap="viridis", vmin=0, vmax=1, alpha=0.20)
        ax.imshow(heat, extent=(-0.5, image.width - 0.5, image.height - 0.5, -0.5),
                  interpolation="nearest", cmap="viridis", vmin=0, vmax=maximum,
                  alpha=0.15 + 0.65 * np.minimum(heat / maximum, 1.0))
        ax.set_title(f"{record['variant']}")
        ax.axis("off")
    fig.suptitle(f"{records[0]['question']}\n{subtitle}")
    fig.savefig(directory / filename, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(render_comparison(parser.parse_args().directory))
