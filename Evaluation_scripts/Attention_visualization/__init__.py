"""Actual-generation attention capture for Qwen3-VL."""
from .attention_runner import Question, Settings, load_questions, run_comparison

__all__ = ["Question", "Settings", "load_questions", "run_comparison"]
