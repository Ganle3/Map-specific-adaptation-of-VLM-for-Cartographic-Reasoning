#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare FRIEDA + MapWise data for Unsloth Qwen3-VL SFT.

Expected project structure
--------------------------
Train_Val_data/
├── FRIEDA/
│   ├── image/
│   ├── frieda_train.json
│   └── frieda_validation.json
└── MAPWise/
    ├── image/
    ├── china_train_175_balanced.json
    ├── china_validation_37_balanced.json
    ├── usa_train_175_balanced.json
    └── usa_validation_38_balanced.json

Outputs
-------
Train_Val_data/Prepared_SFT/
├── sft_train.jsonl
├── sft_validation.jsonl
├── unified_train.json
├── unified_validation.json
├── preparation_report.json
├── missing_images.json
└── preview_samples.json

Important
---------
The supplied JSON files contain questions and final answers, but no CoT/reasoning
field. Therefore, by default the assistant target is:

    Final answer: <answer>

If a future JSON contains one of these fields:
    reasoning, cot, rationale, chain_of_thought
the script automatically formats:
    Reasoning: ...
    Final answer: ...

Usage
-----
python prepare_sft_data.py --data-root "C:\\Users\\junyhuang\\Thesis\\VLM_adaptation\\Train_Val_data"

For a validation-only dry run when some source JSON files are not present:
python prepare_sft_data.py --data-root "..." --allow-missing-source-files

By default, unresolved image paths cause the script to stop. To inspect the
report without stopping:
python prepare_sft_data.py --data-root "..." --no-strict-images
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REASONING_FIELDS = ("reasoning", "cot", "rationale", "chain_of_thought")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def read_json(path: Path) -> list[dict[str, Any]]:
    """Read a top-level JSON list and validate its basic structure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a top-level JSON list.")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError(f"Every item in {path} must be a JSON object.")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_relpath(value: str) -> str:
    """Normalize slash style for matching paths from JSON."""
    return value.replace("\\", "/").lstrip("./")


class ImageResolver:
    """Resolve image references robustly without silently choosing ambiguity."""

    def __init__(self, image_root: Path):
        self.image_root = image_root.resolve()
        self.by_relative: dict[str, Path] = {}
        self.by_name: dict[str, list[Path]] = defaultdict(list)
        self.by_stem: dict[str, list[Path]] = defaultdict(list)

        if not self.image_root.exists():
            return

        for path in self.image_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            resolved = path.resolve()
            rel = normalize_relpath(str(path.relative_to(self.image_root)))
            self.by_relative[rel.lower()] = resolved
            self.by_name[path.name.lower()].append(resolved)
            self.by_stem[path.stem.lower()].append(resolved)

    def resolve_relative(self, reference: str) -> tuple[Path | None, str | None]:
        """
        Resolve a FRIEDA-style relative reference.

        Matching order:
        1. image_root / full relative reference
        2. indexed full relative path
        3. unique basename match
        """
        ref = normalize_relpath(reference)
        direct = (self.image_root / Path(ref)).resolve()
        if direct.is_file():
            return direct, None

        indexed = self.by_relative.get(ref.lower())
        if indexed is not None:
            return indexed, None

        matches = self.by_name.get(Path(ref).name.lower(), [])
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"Ambiguous basename {Path(ref).name!r}: {len(matches)} matches"
        return None, "No matching image found"

    def resolve_mapwise(self, map_no: str) -> tuple[Path | None, str | None]:
        """
        Resolve a MapWise image from map_no.

        It first searches for an exact filename stem. This supports files such
        as map12_2D.png and map_12788.jpg, regardless of nested folders.
        """
        key = str(map_no).strip().lower()
        matches = self.by_stem.get(key, [])
        if len(matches) == 1:
            return matches[0], None
        if len(matches) > 1:
            return None, f"Ambiguous map_no {map_no!r}: {len(matches)} image matches"

        # Conservative fallback for filenames containing the exact map_no.
        contains = [
            p for stem, paths in self.by_stem.items()
            if key == stem or stem.startswith(key + "_") or stem.endswith("_" + key)
            for p in paths
        ]
        unique = sorted(set(contains))
        if len(unique) == 1:
            return unique[0], None
        if len(unique) > 1:
            return None, f"Ambiguous fallback match for map_no {map_no!r}: {len(unique)} matches"
        return None, "No image with a filename stem matching map_no"


def first_nonempty(item: dict[str, Any], fields: Iterable[str]) -> str | None:
    for field in fields:
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def build_assistant_target(item: dict[str, Any], answer: str) -> tuple[str, bool]:
    reasoning = first_nonempty(item, REASONING_FIELDS)
    if reasoning:
        return f"Reasoning: {reasoning}\nFinal answer: {answer}", True
    return f"Final answer: {answer}", False


def to_unsloth_messages(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one unified sample to multimodal chat messages."""
    user_content = [
        {"type": "image", "image": image_path}
        for image_path in sample["images"]
    ]
    user_content.append({"type": "text", "text": sample["question"]})

    row = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["target_text"]}],
            },
        ],
        # Metadata is retained for stratified evaluation/debugging.
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "split": sample["split"],
        "answer": sample["answer"],
        "answer_type": sample.get("answer_type"),
        "task_category": sample.get("task_category"),
        "map_id": sample.get("map_id"),
        "image_count": len(sample["images"]),
        "has_reasoning": sample["has_reasoning"],
    }
    return row


def convert_frieda(
    items: list[dict[str, Any]],
    resolver: ImageResolver,
    source_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        sample_id = str(item.get("question_ref") or f"frieda_{index:04d}")
        question = first_nonempty(item, ("question_text", "question"))
        answer = first_nonempty(item, ("expected_answer", "answer", "ground_truth"))
        image_refs = item.get("image_urls")

        errors: list[str] = []
        if not question:
            errors.append("Missing question_text")
        if not answer:
            errors.append("Missing expected_answer")
        if not isinstance(image_refs, list) or not image_refs:
            errors.append("image_urls must be a non-empty list")
            image_refs = []

        image_paths: list[str] = []
        for ref in image_refs:
            path, error = resolver.resolve_relative(str(ref))
            if path is None:
                errors.append(f"{ref}: {error}")
            else:
                image_paths.append(str(path))

        if errors:
            missing.append({
                "dataset": "FRIEDA",
                "sample_id": sample_id,
                "source_file": str(source_file),
                "errors": errors,
                "raw_image_refs": image_refs,
            })
            continue

        target_text, has_reasoning = build_assistant_target(item, answer)
        converted.append({
            "sample_id": f"frieda_{sample_id}",
            "dataset": "FRIEDA",
            "split": str(item.get("split") or "").lower(),
            "images": image_paths,
            "question": question,
            "answer": answer,
            "target_text": target_text,
            "has_reasoning": has_reasoning,
            "answer_type": item.get("answer_type"),
            "task_category": item.get("spatial_relationship"),
            "map_id": item.get("source_document_id"),
            "metadata": {
                "domain": item.get("domain"),
                "map_elements": item.get("map_elements"),
                "map_count": item.get("map_count"),
                "source_document_ids": item.get("source_document_ids"),
            },
        })

    return converted, missing


def convert_mapwise(
    items: list[dict[str, Any]],
    resolver: ImageResolver,
    source_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for index, item in enumerate(items):
        sample_id = str(item.get("qa_id") or f"mapwise_{index:04d}")
        question = first_nonempty(item, ("question", "question_text"))
        answer = first_nonempty(item, ("ground_truth", "expected_answer", "answer"))
        map_no = first_nonempty(item, ("map_no",))

        errors: list[str] = []
        if not question:
            errors.append("Missing question")
        if not answer:
            errors.append("Missing ground_truth")
        if not map_no:
            errors.append("Missing map_no")

        image_paths: list[str] = []
        if map_no:
            path, error = resolver.resolve_mapwise(map_no)
            if path is None:
                errors.append(f"{map_no}: {error}")
            else:
                image_paths.append(str(path))

        if errors:
            missing.append({
                "dataset": "MapWise",
                "sample_id": sample_id,
                "source_file": str(source_file),
                "errors": errors,
                "map_no": map_no,
            })
            continue

        target_text, has_reasoning = build_assistant_target(item, answer)
        converted.append({
            "sample_id": sample_id,
            "dataset": "MapWise",
            "split": str(item.get("split") or "").lower(),
            "images": image_paths,
            "question": question,
            "answer": answer,
            "target_text": target_text,
            "has_reasoning": has_reasoning,
            "answer_type": item.get("ground_truth_type"),
            "task_category": f"template_{item.get('template_no')}",
            "map_id": map_no,
            "metadata": {
                "country": item.get("country"),
                "template_no": item.get("template_no"),
                "relative_region": item.get("relative_region"),
                "data_group_id": item.get("data_group_id"),
                "legend_style": item.get("legend_style"),
                "c_or_d": item.get("c_or_d"),
                "source_index": item.get("source_index"),
            },
        })

    return converted, missing


def duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def create_report(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    skipped_sources: list[str],
) -> dict[str, Any]:
    train_ids = {x["sample_id"] for x in train}
    val_ids = {x["sample_id"] for x in validation}
    train_maps = {f'{x["dataset"]}:{x["map_id"]}' for x in train if x.get("map_id")}
    val_maps = {f'{x["dataset"]}:{x["map_id"]}' for x in validation if x.get("map_id")}

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total": len(rows),
            "by_dataset": dict(Counter(x["dataset"] for x in rows)),
            "by_answer_type": dict(Counter(str(x.get("answer_type")) for x in rows)),
            "by_task_category": dict(Counter(str(x.get("task_category")) for x in rows)),
            "by_image_count": dict(Counter(len(x["images"]) for x in rows)),
            "with_reasoning": sum(bool(x["has_reasoning"]) for x in rows),
            "without_reasoning": sum(not bool(x["has_reasoning"]) for x in rows),
            "duplicate_sample_ids": duplicates(x["sample_id"] for x in rows),
        }

    return {
        "train": summarize(train),
        "validation": summarize(validation),
        "cross_split_sample_id_overlap": sorted(train_ids & val_ids),
        # This is diagnostic only. FRIEDA may intentionally share source documents
        # while using different images/questions; MapWise should normally be checked
        # at the data_group/map level according to the experimental split design.
        "cross_split_map_id_overlap": sorted(train_maps & val_maps),
        "unresolved_or_invalid_samples": len(missing),
        "skipped_source_files": skipped_sources,
        "note": (
            "No supplied sample contains a reasoning/CoT field unless "
            "'with_reasoning' is greater than zero. Answer-only targets do not "
            "constitute CoT SFT."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to Train_Val_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <data-root>/Prepared_SFT",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Deterministic shuffle seed.",
    )
    parser.add_argument(
        "--no-strict-images",
        action="store_true",
        help="Write outputs even when some image paths cannot be resolved.",
    )
    parser.add_argument(
        "--allow-missing-source-files",
        action="store_true",
        help="Skip absent JSON source files instead of stopping.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir or data_root / "Prepared_SFT").resolve()

    frieda_dir = data_root / "FRIEDA"
    mapwise_dir = data_root / "MAPWise"

    source_specs = [
        ("FRIEDA", "train", frieda_dir / "frieda_train.json"),
        ("FRIEDA", "validation", frieda_dir / "frieda_validation.json"),
        ("MapWise", "train", mapwise_dir / "china_train_175_balanced.json"),
        ("MapWise", "train", mapwise_dir / "usa_train_175_balanced.json"),
        ("MapWise", "validation", mapwise_dir / "china_validation_37_balanced.json"),
        ("MapWise", "validation", mapwise_dir / "usa_validation_38_balanced.json"),
    ]

    frieda_resolver = ImageResolver(frieda_dir / "image")
    mapwise_resolver = ImageResolver(mapwise_dir / "image")

    converted_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    all_missing: list[dict[str, Any]] = []
    skipped_sources: list[str] = []

    for dataset_name, expected_split, source_path in source_specs:
        if not source_path.exists():
            message = f"Missing source JSON: {source_path}"
            if args.allow_missing_source_files:
                print(f"[WARNING] {message}", file=sys.stderr)
                skipped_sources.append(str(source_path))
                continue
            raise FileNotFoundError(message)

        items = read_json(source_path)
        if dataset_name == "FRIEDA":
            converted, missing = convert_frieda(items, frieda_resolver, source_path)
        else:
            converted, missing = convert_mapwise(items, mapwise_resolver, source_path)

        # The filename-level expected split is authoritative, while mismatch is reported.
        for row in converted:
            if row["split"] and row["split"] != expected_split:
                all_missing.append({
                    "dataset": dataset_name,
                    "sample_id": row["sample_id"],
                    "source_file": str(source_path),
                    "errors": [
                        f"JSON split={row['split']!r} but source file is assigned to "
                        f"{expected_split!r}"
                    ],
                })
                continue
            row["split"] = expected_split
            converted_by_split[expected_split].append(row)

        all_missing.extend(missing)

    rng = random.Random(args.seed)
    rng.shuffle(converted_by_split["train"])
    rng.shuffle(converted_by_split["validation"])

    report = create_report(
        converted_by_split["train"],
        converted_by_split["validation"],
        all_missing,
        skipped_sources,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "missing_images.json", all_missing)
    write_json(output_dir / "preparation_report.json", report)

    if all_missing and not args.no_strict_images:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise RuntimeError(
            f"Found {len(all_missing)} invalid/unresolved samples. "
            f"Inspect {output_dir / 'missing_images.json'}. "
            "Fix the image paths or rerun with --no-strict-images for diagnostics."
        )

    train_unified = converted_by_split["train"]
    val_unified = converted_by_split["validation"]
    train_messages = [to_unsloth_messages(x) for x in train_unified]
    val_messages = [to_unsloth_messages(x) for x in val_unified]

    write_json(output_dir / "unified_train.json", train_unified)
    write_json(output_dir / "unified_validation.json", val_unified)
    write_jsonl(output_dir / "sft_train.jsonl", train_messages)
    write_jsonl(output_dir / "sft_validation.jsonl", val_messages)

    previews = {
        "train": train_messages[:4],
        "validation": val_messages[:4],
    }
    write_json(output_dir / "preview_samples.json", previews)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPrepared data written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())