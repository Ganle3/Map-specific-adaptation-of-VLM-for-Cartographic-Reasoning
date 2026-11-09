"""Export PowerPoint-compatible relay-query attention GIFs from raw NPZ maps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_LAYERS = (6, 9, 12, 15, 18, 25, 30)
DEFAULT_QUESTIONS = ("mapverse_qa273", "mapwise_qa294")


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _heat_rgba(raw: np.ndarray, width: int, height: int) -> tuple[Image.Image, float]:
    """Apply per-frame p99 contrast after retaining the raw spatial map."""
    scale = float(np.quantile(raw, 0.99))
    values = np.clip(raw / max(scale, 1e-30), 0.0, 1.0)
    red = np.clip(1.5 * values, 0.0, 1.0)
    green = np.clip(1.5 - 2.0 * np.abs(values - 0.65), 0.0, 1.0)
    blue = np.clip(1.2 - 2.0 * values, 0.0, 1.0)
    alpha = 0.88 * np.power(values, 0.65)
    rgba = np.stack((red, green, blue, alpha), axis=-1)
    grid = Image.fromarray(np.rint(rgba * 255).astype(np.uint8), mode="RGBA")
    return grid.resize((width, height), Image.Resampling.NEAREST), scale


def _base_image(path: Path, max_width: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.width > max_width:
        height = round(image.height * max_width / image.width)
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    return image


def _overlay(base: Image.Image, raw: np.ndarray) -> tuple[Image.Image, float]:
    heat, scale = _heat_rgba(raw, base.width, base.height)
    return Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB"), scale


def _token_text(token: str) -> str:
    return token.replace("\n", "\\n").replace("\r", "\\r")


def _individual_frame(base: Image.Image, raw: np.ndarray, *, layer: int, step: int,
                      steps: int, token: str, question: str) -> Image.Image:
    overlay, scale = _overlay(base, raw)
    header = 78
    frame = Image.new("RGB", (base.width, base.height + header), "white")
    frame.paste(overlay, (0, header))
    draw = ImageDraw.Draw(frame)
    draw.text((12, 8), f"Relay3 query -> image | decoder layer {layer}",
              fill="#172033", font=_font(20, bold=True))
    draw.text((12, 36), f"step {step + 1}/{steps} | predicting {_token_text(token)!r} | "
              f"per-frame p99={scale:.3e}", fill="#334155", font=_font(15))
    question_short = question if len(question) <= 105 else question[:102] + "..."
    draw.text((12, 58), question_short, fill="#475569", font=_font(12))
    return frame


def _panel(base: Image.Image, raw: np.ndarray, layer: int, width: int) -> Image.Image:
    height = round(base.height * width / base.width)
    resized = base.resize((width, height), Image.Resampling.LANCZOS)
    overlay, _ = _overlay(resized, raw)
    title = 34
    panel = Image.new("RGB", (width, height + title), "white")
    panel.paste(overlay, (0, title))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), f"Layer {layer}", fill="#172033", font=_font(17, bold=True))
    return panel


def _overview_frame(base: Image.Image, maps: np.ndarray, *, layers: tuple[int, ...],
                    step: int, steps: int, token: str, question: str,
                    panel_width: int) -> Image.Image:
    panels = [_panel(base, maps[layer], layer, panel_width) for layer in layers]
    cols, rows, gap, header = 4, 2, 10, 82
    panel_h = max(panel.height for panel in panels)
    width = cols * panel_width + (cols - 1) * gap
    height = header + rows * panel_h + (rows - 1) * gap
    frame = Image.new("RGB", (width, height), "#eef2f7")
    draw = ImageDraw.Draw(frame)
    draw.text((12, 8), "Relay3 query-to-image attention across decoder layers",
              fill="#172033", font=_font(22, bold=True))
    draw.text((12, 38), f"step {step + 1}/{steps} | predicting {_token_text(token)!r}",
              fill="#334155", font=_font(16))
    question_short = question if len(question) <= 150 else question[:147] + "..."
    draw.text((12, 61), question_short, fill="#475569", font=_font(12))
    for index, panel in enumerate(panels):
        row, col = divmod(index, cols)
        frame.paste(panel, (col * (panel_width + gap), header + row * (panel_h + gap)))
    return frame


def _gif_frames(frames: list[Image.Image]) -> list[Image.Image]:
    return [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128,
                          dither=Image.Dither.NONE) for frame in frames]


def _save_gif(frames: list[Image.Image], destination: Path, duration_ms: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    paletted = _gif_frames(frames)
    durations = [duration_ms] * len(paletted)
    durations[-1] = max(1000, duration_ms)
    paletted[0].save(destination, save_all=True, append_images=paletted[1:],
                     duration=durations, loop=0, disposal=2, optimize=False)


def _trajectory_rows(attention: np.ndarray, layers: tuple[int, ...], tokens: list[str]) -> list[dict]:
    rows = []
    for step in range(attention.shape[0]):
        for layer in layers:
            raw = attention[step, layer].astype(np.float64)
            probability = raw / max(float(raw.sum()), 1e-30)
            height, width = raw.shape
            row_mass = probability.sum(axis=1)
            col_mass = probability.sum(axis=0)
            entropy = -float((probability * np.log(np.maximum(probability, 1e-30))).sum())
            rows.append({
                "step_0based": step, "layer": layer, "token": tokens[step],
                "image_attention_mass": float(raw.sum()),
                "normalized_entropy": entropy / np.log(raw.size),
                "centroid_y_0to1": float(row_mass @ np.linspace(0.0, 1.0, height)),
                "centroid_x_0to1": float(col_mass @ np.linspace(0.0, 1.0, width)),
                "peak_row_0based": int(row_mass.argmax()),
                "peak_col_0based": int(col_mass.argmax()),
            })
    return rows


def export_question(question_dir: Path, layers: tuple[int, ...], duration_ms: int,
                    individual_width: int, panel_width: int) -> list[Path]:
    relay_dir = question_dir / "relay3"
    record = json.loads((relay_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(relay_dir / "attention.npz") as archive:
        attention = archive["relay_attention"].astype(np.float32)
    if attention.shape[0] != len(record["tokens"]):
        raise ValueError(f"Step/token mismatch in {question_dir}: {attention.shape[0]} vs {len(record['tokens'])}")
    invalid = [layer for layer in layers if not 0 <= layer < attention.shape[1]]
    if invalid:
        raise ValueError(f"Invalid layers {invalid}; checkpoint has {attention.shape[1]} layers")

    base = _base_image(Path(record["image"]), individual_width)
    output = relay_dir / "gifs"
    output.mkdir(parents=True, exist_ok=True)
    destinations = []
    for layer in layers:
        frames = [
            _individual_frame(base, attention[step, layer], layer=layer, step=step,
                              steps=attention.shape[0], token=record["tokens"][step],
                              question=record["question"])
            for step in range(attention.shape[0])
        ]
        destination = output / f"relay_query_layer_{layer:02d}.gif"
        _save_gif(frames, destination, duration_ms)
        destinations.append(destination)

    overview = [
        _overview_frame(base, attention[step], layers=layers, step=step,
                        steps=attention.shape[0], token=record["tokens"][step],
                        question=record["question"], panel_width=panel_width)
        for step in range(attention.shape[0])
    ]
    overview_path = output / "relay_query_layers_06_09_12_15_18_25_30.gif"
    _save_gif(overview, overview_path, duration_ms)
    destinations.append(overview_path)

    rows = _trajectory_rows(attention, layers, record["tokens"])
    csv_path = output / "relay_query_spatial_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    destinations.append(csv_path)
    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "VLM_adaptation/Training_outputs/Prototype/relay3_visualization_step4000"))
    parser.add_argument("--questions", nargs="+", default=list(DEFAULT_QUESTIONS))
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--duration-ms", type=int, default=250)
    parser.add_argument("--individual-width", type=int, default=720)
    parser.add_argument("--panel-width", type=int, default=350)
    args = parser.parse_args()
    layers = tuple(args.layers)
    for name in args.questions:
        question_dir = args.root / name
        for path in export_question(question_dir, layers, args.duration_ms,
                                    args.individual_width, args.panel_width):
            print(path, flush=True)


if __name__ == "__main__":
    main()
