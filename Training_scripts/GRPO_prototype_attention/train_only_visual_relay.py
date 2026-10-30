#!/usr/bin/env python3
"""Dedicated training entry point for the native only_visual_relay policy."""
from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    forbidden = {"--policy-kind", "--placement"}.intersection(sys.argv[1:])
    if forbidden:
        raise SystemExit(
            "train_only_visual_relay.py fixes policy and placement; remove "
            + ", ".join(sorted(forbidden))
        )
    sys.argv[1:1] = [
        "--policy-kind", "only_visual_relay",
        "--placement", "after_vision_end",
    ]
    main()
