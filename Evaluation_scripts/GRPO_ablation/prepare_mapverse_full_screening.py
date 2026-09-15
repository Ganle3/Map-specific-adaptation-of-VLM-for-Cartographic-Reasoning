#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


EXACT_TYPES = {"Boolean", "Counting", "Single Entity"}


def main():

    parser = argparse.ArgumentParser(
        description="Prepare MapVerse exact-match-compatible screening shards."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--image-base",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(args.input)

    # Preserve original row position in typed_questions.csv
    df["source_row"] = range(len(df))

    required_columns = {
        "image_name",
        "question",
        "correct_answer",
        "question_type",
        "answer_type",
        "map_type",
        "image_size_bucket",
        "geographic_level",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    print("=" * 72)
    print("MAPVERSE FULL SCREENING PREPARATION")
    print("=" * 72)

    print(f"Original rows: {len(df)}")

    # ---------------------------------------------------------
    # Keep only exact-reward-compatible tasks
    # ---------------------------------------------------------

    pool = df[
        df["answer_type"].isin(EXACT_TYPES)
    ].copy()

    print(f"Exact-compatible before cleaning: {len(pool)}")

    # ---------------------------------------------------------
    # Remove malformed rows
    # ---------------------------------------------------------

    required_nonempty = [
        "image_name",
        "question",
        "correct_answer",
        "answer_type",
    ]

    for column in required_nonempty:

        pool = pool[
            pool[column].notna()
        ]

        pool = pool[
            pool[column].astype(str).str.strip() != ""
        ]

    pool = pool.reset_index(drop=True)

    # ---------------------------------------------------------
    # Prepare output directory early
    # ---------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Check image availability
    # ---------------------------------------------------------

    missing_image_count = 0

    if args.image_base is not None:

        image_base = (
            args.image_base
            .expanduser()
            .resolve()
        )

        image_exists = pool[
            "image_name"
        ].astype(str).apply(
            lambda name: (
                image_base / name
            ).is_file()
        )

        missing_rows = pool.loc[
            ~image_exists
        ].copy()

        missing_image_count = len(
            missing_rows
        )

        if missing_image_count > 0:

            print()
            print(
                f"WARNING: {missing_image_count} QA rows "
                "reference missing images."
            )

            missing_path = (
                args.output_dir
                / "mapverse_missing_image_rows.csv"
            )

            missing_rows.to_csv(
                missing_path,
                index=False,
            )

            print(
                f"Missing-image rows saved to:\n"
                f"{missing_path}"
            )

            # Remove unavailable samples
            pool = pool.loc[
                image_exists
            ].copy()

            pool = pool.reset_index(
                drop=True
            )

    # ---------------------------------------------------------
    # Assign screening IDs AFTER filtering
    # ---------------------------------------------------------

    pool["screening_id"] = range(
        1,
        len(pool) + 1,
    )

    # ---------------------------------------------------------
    # Save full usable pool
    # ---------------------------------------------------------

    pool_path = (
        args.output_dir
        / "mapverse_exact_screening_pool.csv"
    )

    pool.to_csv(
        pool_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Split into deterministic contiguous shards
    # ---------------------------------------------------------

    shard_size = math.ceil(
        len(pool) / args.num_shards
    )

    manifest_rows = []

    for shard_id in range(
        args.num_shards
    ):

        start = (
            shard_id * shard_size
        )

        end = min(
            start + shard_size,
            len(pool),
        )

        shard = pool.iloc[
            start:end
        ].copy()

        if shard.empty:
            continue

        shard_path = (
            args.output_dir
            / f"mapverse_screening_shard_{shard_id:02d}.csv"
        )

        shard.to_csv(
            shard_path,
            index=False,
        )

        manifest_rows.append(
            {
                "shard_id": shard_id,
                "start_screening_id": int(
                    shard["screening_id"].iloc[0]
                ),
                "end_screening_id": int(
                    shard["screening_id"].iloc[-1]
                ),
                "num_rows": len(shard),
                "csv": shard_path.name,
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        args.output_dir
        / "mapverse_screening_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("SCREENING POOL SUMMARY")
    print("=" * 72)

    print(f"Original dataset rows:  {len(df)}")
    print(f"Missing-image QA rows:  {missing_image_count}")
    print(f"Usable screening rows:  {len(pool)}")

    print()

    print(
        "Boolean:       ",
        int(
            (
                pool["answer_type"]
                == "Boolean"
            ).sum()
        ),
    )

    print(
        "Counting:      ",
        int(
            (
                pool["answer_type"]
                == "Counting"
            ).sum()
        ),
    )

    print(
        "Single Entity: ",
        int(
            (
                pool["answer_type"]
                == "Single Entity"
            ).sum()
        ),
    )

    print()

    print(
        f"Number of shards:      "
        f"{len(manifest_rows)}"
    )

    print(
        f"Full pool saved to:\n"
        f"{pool_path}"
    )

    print(
        f"Manifest saved to:\n"
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()