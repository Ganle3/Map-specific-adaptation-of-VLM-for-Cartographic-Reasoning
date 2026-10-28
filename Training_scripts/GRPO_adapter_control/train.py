#!/usr/bin/env python3
"""Thin mixed-dataset wrapper around the established joint-LoRA GRPO trainer."""
from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DEBUG_DIR = THIS_DIR.parent / "Debug_GRPO"
LOOKBETTER_DIR = THIS_DIR.parent / "GRPO_lookbetter"
for path in (DEBUG_DIR, LOOKBETTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _vislora_grpo_baseline_snapshot as baseline
from data import build_cartographic_prompt

# Patch only the prompt text. Model loading, NF4, PEFT initialization, TRL
# rollout generation, GRPO loss, optimizer and checkpointing remain in the
# established Debug_GRPO implementation.
baseline.build_mapwise_prompt = build_cartographic_prompt

from grpo_joint_debug import main


if __name__ == "__main__":
    main()
