#!/usr/bin/env python3
"""Train the nine-grid policy with direct crop-merger embedding appends.

This launcher leaves the legacy ``train.py`` defaults unchanged while selecting
the continuity-preserving rollout implementation.
"""
from __future__ import annotations

import sys

if "--rollout-module" not in sys.argv:
    sys.argv.extend(("--rollout-module", "live_rollout_direct_visual"))

from train import main

if __name__ == "__main__":
    main()
