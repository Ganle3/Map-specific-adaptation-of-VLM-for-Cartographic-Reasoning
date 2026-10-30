#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke_gradient_core import main

if __name__ == "__main__":
    if "--num-relay-tokens" not in sys.argv:
        sys.argv[1:1] = ["--num-relay-tokens", "1"]
    main(scheme="image_question_cot_shared")

