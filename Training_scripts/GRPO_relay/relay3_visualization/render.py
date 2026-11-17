"""Build a standalone three-way relay3/adapter attention viewer."""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _gzip_b64(array: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(array.tobytes(), compresslevel=9)).decode("ascii")


def _normalized_u8(array: np.ndarray, *, quantile: float | None = None,
                   signed: bool = False) -> np.ndarray:
    """Scale each spatial map independently, optionally clipping at a quantile.

    Aggregation is deliberately performed by the caller before this function.
    Thus robust all-layer display still means patch-wise maximum over raw decoder
    attention first, followed by p99 display scaling.
    """
    magnitude = np.abs(array) if signed else array
    if quantile is None:
        scale = magnitude.max(axis=(-2, -1), keepdims=True)
    else:
        if not 0.0 < quantile <= 1.0:
            raise ValueError(f"quantile must be in (0,1], got {quantile}")
        scale = np.quantile(magnitude, quantile, axis=(-2, -1), keepdims=True)
    scaled = np.divide(array, scale, out=np.zeros_like(array), where=scale > 0)
    if signed:
        return np.rint((np.clip(scaled, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _signed_extreme(array: np.ndarray, axis) -> np.ndarray:
    """Select the largest-absolute entry while preserving its sign."""
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a if a >= 0 else array.ndim + a for a in axes)
    keep = tuple(i for i in range(array.ndim) if i not in axes)
    transposed = np.transpose(array, keep + axes)
    flat = transposed.reshape(*transposed.shape[:len(keep)], -1)
    index = np.abs(flat).argmax(axis=-1, keepdims=True)
    return np.take_along_axis(flat, index, axis=-1)[..., 0]


def _stream_payload(raw: np.ndarray, *, signed: bool = False) -> dict:
    if raw.ndim != 4:
        raise ValueError(f"Expected [step,layer,y,x], got {raw.shape}")
    # Preserve the requested aggregation exactly: take the raw patch-wise
    # maximum across layers/steps first. Robust contrast only changes the
    # subsequent display scale; it never changes which values are aggregated.
    aggregate = _signed_extreme if signed else lambda value, axis: value.max(axis=axis)
    raw_step_all = aggregate(raw, 1)
    raw_full_layer = aggregate(raw, 0)
    raw_full_all = aggregate(raw, (0, 1))
    step_layer = _normalized_u8(raw, signed=signed)
    step_layer_p99 = _normalized_u8(raw, quantile=0.99, signed=signed)
    step_all = _normalized_u8(raw_step_all, signed=signed)
    step_all_p99 = _normalized_u8(raw_step_all, quantile=0.99, signed=signed)
    full_layer = _normalized_u8(raw_full_layer, signed=signed)
    full_layer_p99 = _normalized_u8(raw_full_layer, quantile=0.99, signed=signed)
    full_all = _normalized_u8(raw_full_all, signed=signed)
    full_all_p99 = _normalized_u8(raw_full_all, quantile=0.99, signed=signed)
    return {
        "shape": list(raw.shape),
        "signed": signed,
        "step_layer_gzip_b64": _gzip_b64(step_layer),
        "step_layer_p99_gzip_b64": _gzip_b64(step_layer_p99),
        "step_all_gzip_b64": _gzip_b64(step_all),
        "step_all_p99_gzip_b64": _gzip_b64(step_all_p99),
        "full_layer_gzip_b64": _gzip_b64(full_layer),
        "full_layer_p99_gzip_b64": _gzip_b64(full_layer_p99),
        "full_all_gzip_b64": _gzip_b64(full_all),
        "full_all_p99_gzip_b64": _gzip_b64(full_all_p99),
        "raw_step_layer_max": np.abs(raw).max(axis=(-2, -1)).astype(float).tolist(),
        "raw_step_all_max": np.abs(raw).max(axis=(1, 2, 3)).astype(float).tolist(),
        "raw_step_all_p99": np.quantile(np.abs(raw_step_all), 0.99, axis=(-2, -1)).astype(float).tolist(),
    }


def render_question(question_dir: Path) -> Path:
    adapter_dir = question_dir / "adapter_only"
    relay_dir = question_dir / "relay3"
    adapter_record = json.loads((adapter_dir / "result.json").read_text(encoding="utf-8"))
    relay_record = json.loads((relay_dir / "result.json").read_text(encoding="utf-8"))
    extra_streams = {}
    with np.load(adapter_dir / "attention.npz") as archive:
        adapter_hidden = archive["hidden_attention"].astype(np.float32)
        for name in ("hidden_qk_cosine", "hidden_qk_logit", "hidden_v_contribution"):
            if name in archive:
                extra_streams[f"adapter_{name}"] = archive[name].astype(np.float32)
    with np.load(relay_dir / "attention.npz") as archive:
        relay_hidden = archive["hidden_attention"].astype(np.float32)
        relay_query = archive["relay_attention"].astype(np.float32)
        for name in (
            "hidden_qk_cosine", "hidden_qk_logit", "hidden_v_contribution",
            "relay_qk_cosine", "relay_qk_logit", "relay_v_contribution",
        ):
            if name in archive:
                extra_streams[f"relay3_{name}"] = archive[name].astype(np.float32)
    if adapter_hidden.shape[1:] != relay_hidden.shape[1:] or relay_hidden.shape != relay_query.shape:
        raise ValueError("Adapter and relay image grids/layer counts are incompatible")

    image_path = Path(adapter_record["image"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    payload = {
        "question": adapter_record["question"],
        "ground_truth": adapter_record["ground_truth"],
        "image_path": str(image_path),
        "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
        "grid": adapter_record["merged_grid_hw"],
        "layers": adapter_hidden.shape[1],
        "adapter": adapter_record,
        "relay": relay_record,
        "streams": {
            "adapter_hidden": _stream_payload(adapter_hidden),
            "relay_hidden": _stream_payload(relay_hidden),
            "relay_query": _stream_payload(relay_query),
            **{
                name: _stream_payload(value, signed=("qk_" in name))
                for name, value in extra_streams.items()
            },
        },
        "qkv_available": bool(extra_streams),
    }
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    destination = question_dir / "comparison.html"
    destination.write_text(
        template.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")),
        encoding="utf-8",
    )
    if extra_streams:
        qkv_template = Path(__file__).with_name("qkv_viewer.html").read_text(encoding="utf-8")
        qkv_destination = question_dir / "qkv_comparison.html"
        qkv_destination.write_text(
            qkv_template.replace(
                "__PAYLOAD__", json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
            ),
            encoding="utf-8",
        )
    return destination
