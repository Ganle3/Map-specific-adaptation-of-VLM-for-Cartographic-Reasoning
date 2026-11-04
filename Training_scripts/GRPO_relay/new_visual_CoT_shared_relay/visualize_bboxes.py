#!/usr/bin/env python3
"""Visualize Relay-4 bbox selections and their reuse frequency."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = (
    (230, 57, 70), (31, 119, 180), (46, 160, 67),
    (255, 127, 14), (148, 103, 189), (23, 190, 207),
    (227, 119, 194), (188, 189, 34), (127, 127, 127),
)


def _font(size: int, *, bold: bool = False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _selections(result: dict) -> list[dict]:
    selections = []
    for event_index, event in enumerate(result.get("lookup_trace", []), start=1):
        token = int(event["generated_token_count"])
        for selection in event.get("relay_selections", []):
            selections.append({
                "event": event_index,
                "token": token,
                "relay": int(selection["relay_index"]),
                "action": int(selection["action"]),
                "bbox": tuple(map(int, selection["bbox"])),
                "cache_hit": bool(selection["cache_hit"]),
            })
    return selections


def _group(selections: list[dict]):
    grouped: OrderedDict[tuple[int, int, int, int], dict] = OrderedDict()
    for selection in selections:
        bbox = selection["bbox"]
        if bbox not in grouped:
            grouped[bbox] = {
                "bbox": bbox, "count": 0, "actions": set(),
                "events": [], "tokens": [], "relays": [],
            }
        item = grouped[bbox]
        item["count"] += 1
        item["actions"].add(selection["action"])
        item["events"].append(selection["event"])
        item["tokens"].append(selection["token"])
        item["relays"].append(selection["relay"])
    return list(grouped.values())


def draw_overlay(image: Image.Image, grouped: list[dict], output: Path) -> None:
    map_width, map_height = image.size
    panel_width = max(480, int(map_width * 0.32))
    canvas = Image.new("RGB", (map_width + panel_width, map_height), "white")
    canvas.paste(image.convert("RGB"), (0, 0))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    title_font = _font(max(22, map_width // 55), bold=True)
    text_font = _font(max(16, map_width // 80))
    label_font = _font(max(18, map_width // 70), bold=True)

    for index, item in enumerate(grouped, start=1):
        color = PALETTE[(index - 1) % len(PALETTE)]
        x0, y0, x1, y1 = item["bbox"]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(*color, 48),
                       outline=(*color, 255), width=max(3, map_width // 400))
        label = f"B{index} x{item['count']}"
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_w = label_box[2] - label_box[0] + 12
        label_h = label_box[3] - label_box[1] + 8
        label_x = min(max(0, x0 + 4), map_width - label_w)
        label_y = min(max(0, y0 + 4), map_height - label_h)
        draw.rectangle((label_x, label_y, label_x + label_w, label_y + label_h),
                       fill=(*color, 235))
        draw.text((label_x + 6, label_y + 3), label, font=label_font, fill="white")

    panel_x = map_width + 22
    draw.rectangle((map_width, 0, canvas.width, canvas.height), fill=(248, 250, 253, 255))
    draw.text((panel_x, 22), "Relay-4 selected regions", font=title_font, fill=(25, 35, 50, 255))
    draw.text((panel_x, 62), f"Unique bboxes: {len(grouped)}", font=text_font, fill=(65, 75, 90, 255))
    cursor_y = 104
    for index, item in enumerate(grouped, start=1):
        color = PALETTE[(index - 1) % len(PALETTE)]
        draw.rectangle((panel_x, cursor_y + 3, panel_x + 22, cursor_y + 25), fill=(*color, 255))
        bbox = ", ".join(map(str, item["bbox"]))
        tokens = ",".join(map(str, item["tokens"]))
        relays = ",".join(map(str, item["relays"]))
        lines = [
            f"B{index}: selected {item['count']} times",
            f"bbox [{bbox}]",
            f"tokens: {tokens}",
            f"relays: {relays}",
        ]
        for line_index, line in enumerate(lines):
            draw.text((panel_x + 32, cursor_y + line_index * 25), line,
                      font=text_font, fill=(32, 42, 58, 255))
        cursor_y += 118
        if cursor_y + 100 > map_height:
            break

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def draw_montage(image: Image.Image, grouped: list[dict], output: Path) -> None:
    thumb_w, thumb_h = 360, 280
    header_h = 64
    columns = 3
    rows = (len(grouped) + columns - 1) // columns
    montage = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + header_h)), "white")
    draw = ImageDraw.Draw(montage)
    header_font = _font(20, bold=True)
    detail_font = _font(15)
    for offset, item in enumerate(grouped):
        index = offset + 1
        column, row = offset % columns, offset // columns
        left, top = column * thumb_w, row * (thumb_h + header_h)
        color = PALETTE[offset % len(PALETTE)]
        crop = image.crop(item["bbox"]).convert("RGB")
        crop.thumbnail((thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
        paste_x = left + (thumb_w - crop.width) // 2
        paste_y = top + header_h + (thumb_h - crop.height) // 2
        montage.paste(crop, (paste_x, paste_y))
        draw.rectangle((left, top, left + thumb_w - 1, top + header_h + thumb_h - 1),
                       outline=color, width=4)
        draw.text((left + 10, top + 7), f"B{index} | selected {item['count']} times",
                  font=header_font, fill=color)
        draw.text((left + 10, top + 35), f"bbox={list(item['bbox'])}",
                  font=detail_font, fill=(45, 45, 45))
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    image_path = args.image or Path(result["image"])
    output_dir = args.output_dir or args.result.parent / "bbox_visualization"
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    selections = _selections(result)
    grouped = _group(selections)
    if not grouped:
        raise RuntimeError("Result contains no relay bbox selections")
    overlay = output_dir / "qa318_bbox_overlay.png"
    montage = output_dir / "qa318_bbox_montage.png"
    draw_overlay(image, grouped, overlay)
    draw_montage(image, grouped, montage)
    summary = {
        "selection_count": len(selections),
        "unique_bbox_count": len(grouped),
        "overlay": str(overlay),
        "montage": str(montage),
    }
    (output_dir / "bbox_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

