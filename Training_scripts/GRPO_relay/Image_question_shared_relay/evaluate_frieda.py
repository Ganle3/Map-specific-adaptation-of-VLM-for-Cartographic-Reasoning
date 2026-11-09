#!/usr/bin/env python3
"""Evaluate a trained scheme-2 shared dual-placement relay on FRIEDA."""
from __future__ import annotations

import sys
from pathlib import Path

RELAY_ROOT = Path(__file__).resolve().parents[1]
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))

from evaluate_frieda_relay import main


if __name__ == "__main__":
    sys.argv[1:1] = ["--policy-kind", "image_question_shared"]
    main()
