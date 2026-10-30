#!/usr/bin/env python3
"""Joint GRPO entry point: loaded LoRA adapter plus native visual relays."""
from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    forbidden = {"--policy-kind", "--placement", "--train-adapter"}.intersection(sys.argv[1:])
    if forbidden:
        raise SystemExit(
            "train_only_visual_relay_joint.py fixes policy, placement and adapter training; remove "
            + ", ".join(sorted(forbidden))
        )
    sys.argv[1:1] = [
        "--policy-kind", "only_visual_relay",
        "--placement", "after_vision_end",
        "--train-adapter",
        "--max-grad-norm", "0.1",
    ]
    main()
