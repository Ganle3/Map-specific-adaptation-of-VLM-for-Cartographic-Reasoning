"""Dataset adapter reusing the established MapVerse/MapWise exact scorer."""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "GRPO_lookbetter"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from data import MapQuestion, evaluate_answer, extract_final_answer, load_mapverse_questions  # noqa: E402

__all__ = ["MapQuestion", "evaluate_answer", "extract_final_answer", "load_mapverse_questions"]
