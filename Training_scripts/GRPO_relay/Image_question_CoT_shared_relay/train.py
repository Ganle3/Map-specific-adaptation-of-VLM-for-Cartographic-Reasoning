#!/usr/bin/env python3
"""Train scheme 3: temporary shared relays at every reasoning step."""
from __future__ import annotations

import sys
from pathlib import Path

RELAY_ROOT = Path(__file__).resolve().parents[1]
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))

from train_core import main


if __name__ == "__main__":
    forbidden = {"--policy-kind", "--placement"}.intersection(
        sys.argv[1:]
    )
    if forbidden:
        raise SystemExit("This entry point fixes " + ", ".join(sorted(forbidden)))
    fixed = [
        "--policy-kind", "image_question_cot_shared",
        "--placement", "dynamic_cot_relay",
        "--train-adapter",
        "--max-grad-norm", "0.1",
    ]
    if "--num-relay-tokens" not in sys.argv[1:]:
        fixed += ["--num-relay-tokens", "1"]
    sys.argv[1:1] = fixed
    main()
