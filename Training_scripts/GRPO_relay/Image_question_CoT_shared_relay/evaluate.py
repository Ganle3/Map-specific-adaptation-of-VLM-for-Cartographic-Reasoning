#!/usr/bin/env python3
"""Evaluate scheme 3 checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_core import main

if __name__ == "__main__":
    forbidden = {"--policy-kind", "--placement"}.intersection(
        sys.argv[1:]
    )
    if forbidden:
        raise SystemExit("This entry point fixes " + ", ".join(sorted(forbidden)))
    fixed = [
        "--policy-kind", "image_question_cot_shared",
        "--placement", "dynamic_cot_relay",
    ]
    if "--num-relay-tokens" not in sys.argv[1:]:
        fixed += ["--num-relay-tokens", "1"]
    sys.argv[1:1] = fixed
    main()
