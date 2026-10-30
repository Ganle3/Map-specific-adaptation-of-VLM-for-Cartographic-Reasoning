#!/usr/bin/env python3
"""Train scheme 3: one temporary shared relay at every reasoning step."""
from __future__ import annotations

import sys
from pathlib import Path

RELAY_ROOT = Path(__file__).resolve().parents[1]
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))

from train_core import main


if __name__ == "__main__":
    forbidden = {"--policy-kind", "--placement", "--num-relay-tokens"}.intersection(
        sys.argv[1:]
    )
    if forbidden:
        raise SystemExit("This entry point fixes " + ", ".join(sorted(forbidden)))
    sys.argv[1:1] = [
        "--policy-kind", "image_question_cot_shared",
        "--placement", "dynamic_cot_relay",
        "--num-relay-tokens", "1",
        "--train-adapter",
        "--max-grad-norm", "0.1",
    ]
    main()

