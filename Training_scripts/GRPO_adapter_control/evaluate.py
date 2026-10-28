#!/usr/bin/env python3
"""Thin generic-prompt wrapper around Debug_GRPO/evaluate_grpo.py."""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
VLM_ROOT = THIS_DIR.parents[1]
for path in (THIS_DIR.parent / "Debug_GRPO", THIS_DIR.parent / "GRPO_lookbetter",
             VLM_ROOT / "Evaluation_scripts" / "GRPO_ablation"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_mapwise_trl
from data import build_cartographic_prompt

inference_mapwise_trl.build_mapwise_prompt = build_cartographic_prompt

from evaluate_grpo import main


if __name__ == "__main__":
    main()
