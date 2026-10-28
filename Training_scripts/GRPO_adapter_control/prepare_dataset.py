#!/usr/bin/env python3
"""Normalize mixed/standard MapQA JSON into the established adapter trainer schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def existing_image(base: Path) -> Path:
    candidates = [base] if base.suffix else [base.with_suffix(s) for s in SUFFIXES]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"No image found for {base}")


def normalize(row: dict, index: int, mapverse_root: Path, mapwise_root: Path) -> dict:
    out = dict(row)
    dataset = str(out.get("dataset", "")).casefold()
    if not dataset:
        dataset = "mapwise" if "country" in out and "map_no" in out else "mapverse"
    if dataset == "mapwise":
        if out.get("country") and out.get("map_no"):
            country = str(out["country"]).casefold()
            variant = str(out.get("map_type") or "with_annotations")
            base = mapwise_root / country / "images" / variant / str(out["map_no"])
        else:
            parts = Path(str(out["image_name"])).parts
            if len(parts) != 2:
                raise ValueError(f"MapWise row {index}: expected <country>/<stem>")
            country, stem = parts
            variant = str(out.get("map_type") or "with_annotations")
            base = mapwise_root / country / "images" / variant / stem
        image = existing_image(base)
    elif dataset == "mapverse":
        country = "mapverse"
        name = out.get("image_name", out.get("map_no"))
        image = existing_image(mapverse_root / str(name))
    else:
        raise ValueError(f"Row {index}: unsupported dataset={dataset!r}")

    out.update(
        dataset=dataset,
        country=country,
        image_name=str(image),
        map_no=str(out.get("map_no", out.get("image_name", index))),
        template_no=int(out.get("template_no", out.get("source_id", index))),
        # Preserve the same 1-based dataset index / 0-based qa_id convention
        # used by the prototype trainer's easier1000 logs.
        source_index=index + 1,
        qa_id=str(out.get("qa_id", index)),
        ground_truth=str(out.get("ground_truth", out.get("correct_answer", ""))),
        ground_truth_type=str(out.get("ground_truth_type", out.get("answer_type", ""))),
    )
    if not out.get("question") or not out["ground_truth"]:
        raise ValueError(f"Row {index}: missing question or answer")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mapverse-root", type=Path, required=True)
    p.add_argument("--mapwise-root", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    normalized = [normalize(row, i, args.mapverse_root, args.mapwise_root)
                  for i, row in enumerate(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(args.input), "output": str(args.output),
                      "rows": len(normalized)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
