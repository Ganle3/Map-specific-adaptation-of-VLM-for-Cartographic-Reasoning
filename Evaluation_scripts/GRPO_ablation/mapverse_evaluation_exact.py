"""
mapverse_evaluation_exact.py

Deterministic correctness evaluator for the first MapVerse GRPO diagnostic.

Supported answer types:
  - Boolean
  - Counting
  - Single Entity

Design principle:
  Keep reward binary and verifiable. Normalize formatting, but do NOT use
  fuzzy similarity, semantic judges, substring matching for Single Entity,
  or partial credit.

The implementation is intentionally stricter than some exploratory cells in
MapVerse's Eval_Analysis notebooks. Those notebooks also experiment with
substring/fuzzy matching for Single Entity; that is useful for benchmark
analysis but is undesirable for a binary GRPO correctness reward because it
can create false positives (e.g. "Virginia" vs "West Virginia").
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


SUPPORTED_TYPES = {"Boolean", "Counting", "Single Entity"}


def extract_final_answer(text: Any) -> str:
    """Extract the final answer using the same convention as MapWise GRPO."""
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # Preferred formal format shared with MapWise:
    #   Final answer: <answer>
    matches = re.findall(
        r"Final\s+answer\s*:\s*(.+)",
        s,
        flags=re.I,
    )
    if matches:
        return matches[-1].strip()

    # If Qwen exposes a textual thinking block, score only the answer after it.
    if "</think>" in s.lower():
        s = re.split(r"</think>", s, flags=re.I)[-1].strip()

        matches = re.findall(
            r"Final\s+answer\s*:\s*(.+)",
            s,
            flags=re.I,
        )
        if matches:
            return matches[-1].strip()

    # Backward-compatible fallbacks for older diagnostic outputs.
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", s, flags=re.I | re.S)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"<final>\s*(.*?)\s*</final>", s, flags=re.I | re.S)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r"^\s*Answer\s*:\s*(.+)$", s, flags=re.I | re.M)
    if matches:
        return matches[-1].strip()

    # Last-resort raw output if the model ignores the requested format.
    return s.strip()


def normalize_text(text: Any) -> str:
    """Case/punctuation/whitespace normalization without fuzzy semantics."""
    s = extract_final_answer(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold().strip()

    # Normalize curly punctuation/dashes and remove surrounding quotes.
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip("\"'`“”‘’ ")

    # Remove punctuation but preserve alphanumeric tokens.
    # This mirrors the spirit of the MapVerse notebook clean() helper.
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def canonical_boolean(text: Any) -> Optional[str]:
    s = normalize_text(text)
    if not s:
        return None

    tokens = s.split()
    if not tokens:
        return None

    # Permit concise natural-language final answers ("yes it is", "no").
    first = tokens[0]
    if first in {"yes", "true"}:
        return "yes"
    if first in {"no", "false"}:
        return "no"

    # If the whole normalized answer is exactly a Boolean class.
    if s in {"yes", "true"}:
        return "yes"
    if s in {"no", "false"}:
        return "no"
    return None


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


def _parse_simple_number_words(s: str) -> Optional[int]:
    """Parse simple integer words up to 99; return None when ambiguous."""
    words = s.split()
    if not words:
        return None

    if len(words) == 1 and words[0] in _NUMBER_WORDS:
        return _NUMBER_WORDS[words[0]]

    # e.g. "twenty six"
    if len(words) == 2 and words[0] in {
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
    } and words[1] in {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    }:
        return _NUMBER_WORDS[words[0]] + _NUMBER_WORDS[words[1]]

    return None


def canonical_count(text: Any) -> Optional[int]:
    raw = extract_final_answer(text)
    s = unicodedata.normalize("NFKC", raw).casefold().strip()
    if not s:
        return None

    # Exact integer.
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)

    # Short sentence with exactly one integer ("There are 4.").
    nums = re.findall(r"(?<!\d)[+-]?\d+(?!\d)", s)
    if len(nums) == 1:
        return int(nums[0])

    # Simple number words, after punctuation removal.
    word_form = re.sub(r"[^\w\s-]", " ", s)
    word_form = word_form.replace("-", " ")
    word_form = re.sub(r"\s+", " ", word_form).strip()

    # Strip common concise wrappers before number-word parsing.
    word_form = re.sub(
        r"^(?:there\s+(?:are|is)\s+|the\s+answer\s+is\s+|answer\s+is\s+)",
        "",
        word_form,
    ).strip()

    return _parse_simple_number_words(word_form)


def canonical_single_entity(text: Any) -> str:
    s = extract_final_answer(text)

    # Strip only generic answer wrappers.
    s = re.sub(
        r"^\s*(?:the\s+answer\s+is|it\s+is|this\s+is)\s+",
        "",
        s,
        flags=re.I,
    ).strip()

    return normalize_text(s)


def evaluate_exact(
    prediction: Any,
    ground_truth: Any,
    answer_type: str,
) -> Dict[str, Any]:
    """
    Return binary correctness and canonical forms.

    reward is always 0.0 or 1.0.
    """
    if answer_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported answer_type={answer_type!r}. "
            f"Supported types: {sorted(SUPPORTED_TYPES)}"
        )

    if answer_type == "Boolean":
        pred_c = canonical_boolean(prediction)
        gold_c = canonical_boolean(ground_truth)
    elif answer_type == "Counting":
        pred_c = canonical_count(prediction)
        gold_c = canonical_count(ground_truth)
    else:
        pred_c = canonical_single_entity(prediction)
        gold_c = canonical_single_entity(ground_truth)

    correct = pred_c is not None and gold_c is not None and pred_c == gold_c

    return {
        "correct": bool(correct),
        "reward": 1.0 if correct else 0.0,
        "answer_type": answer_type,
        "prediction_raw": "" if prediction is None else str(prediction),
        "ground_truth_raw": "" if ground_truth is None else str(ground_truth),
        "prediction_canonical": pred_c,
        "ground_truth_canonical": gold_c,
    }


def correctness_reward(
    prediction: Any,
    ground_truth: Any,
    answer_type: str,
) -> float:
    """Convenience function for diagnostic / GRPO code."""
    return evaluate_exact(prediction, ground_truth, answer_type)["reward"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV containing predictions.")
    parser.add_argument("--prediction-col", default="prediction")
    parser.add_argument("--ground-truth-col", default="correct_answer")
    parser.add_argument("--answer-type-col", default="answer_type")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = []
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = evaluate_exact(
                row[args.prediction_col],
                row[args.ground_truth_col],
                row[args.answer_type_col],
            )
            rows.append({**row, **result})

    accuracy = sum(x["reward"] for x in rows) / len(rows) if rows else 0.0
    print(f"N={len(rows)}")
    print(f"Exact correctness accuracy={accuracy:.4f}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
