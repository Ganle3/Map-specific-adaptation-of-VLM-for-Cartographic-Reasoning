"""Regenerate standalone HTML files from existing result JSON and raw NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

from render import render_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    question_dirs = sorted(
        path.parent.parent for path in args.output.glob("*/adapter_only/attention.npz")
        if (path.parent.parent / "relay3" / "attention.npz").is_file()
    )
    if not question_dirs:
        raise FileNotFoundError(f"No completed question outputs under {args.output}")
    for question_dir in question_dirs:
        print(render_question(question_dir), flush=True)


if __name__ == "__main__":
    main()
