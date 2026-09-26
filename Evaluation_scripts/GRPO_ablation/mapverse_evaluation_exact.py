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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


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


def _canonical_decimal(value: str) -> Optional[str]:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    rendered = format(number.normalize(), "f")
    return "0" if rendered in {"-0", "+0"} else rendered


def canonical_count(text: Any) -> Optional[str]:
    raw = extract_final_answer(text)
    s = unicodedata.normalize("NFKC", raw).casefold().strip()
    if not s:
        return None

    # Normalize spacing, dash variants, and currency placement without
    # discarding semantic units such as percent.
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    numeric = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"

    # Exact integer or decimal.
    if re.fullmatch(numeric, s):
        return _canonical_decimal(s)

    # A numeric range, optionally carrying percent on one or both endpoints.
    range_match = re.fullmatch(
        rf"({numeric})\s*%?\s*-\s*({numeric})\s*(%)?",
        s,
    )
    if range_match:
        left = _canonical_decimal(range_match.group(1))
        right = _canonical_decimal(range_match.group(2))
        suffix = "%" if range_match.group(3) else ""
        return f"{left}-{right}{suffix}"

    # Currency symbols may appear before or after the numeric value.
    currency_match = re.fullmatch(rf"(?:[$€£]\s*)?({numeric})(?:\s*[$€£])?", s)
    if currency_match and any(symbol in s for symbol in "$€£"):
        return _canonical_decimal(currency_match.group(1))

    # Short sentence with exactly one integer ("There are 4.").
    nums = re.findall(r"(?<!\d)[+-]?\d+(?!\d)", s)
    if len(nums) == 1:
        return _canonical_decimal(nums[0])

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

    word_number = _parse_simple_number_words(word_form)
    if word_number is not None:
        return str(word_number)

    # A small closed set covers explicitly qualitative gold answers present
    # in MapVerse Counting rows.  Arbitrary prose is still rejected.
    qualitative = {
        "all": "all",
        "many": "many",
        "infinity": "infinity",
        "not possible": "not possible",
    }
    return qualitative.get(word_form)


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


# Conservative, explicit equivalences used by MapVerse answers.  These avoid
# the notebook evaluator's substring/fuzzy matching (which, for example, can
# confuse "Virginia" with "West Virginia") while accepting forms that denote
# exactly the same answer.
_US_STATE_ABBREVIATIONS = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota",
    "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin",
    "wy": "wyoming", "dc": "district of columbia",
}
_STATE_NAMES = set(_US_STATE_ABBREVIATIONS.values())
_SAFE_ENTITY_SUFFIXES = {"line", "route"}
_SAFE_UNITS = {
    "mi", "mile", "miles", "km", "kilometer", "kilometers",
    "m", "meter", "meters", "ft", "foot", "feet", "percent",
}


def canonical_single_entity_variants(text: Any, *, is_ground_truth: bool = False) -> Set[str]:
    """Return only explicitly equivalent Single Entity answer forms.

    Ground truths may encode alternatives as ``A or B``.  Predictions are not
    split this way, so a hedged answer such as ``Virginia or West Virginia``
    does not receive credit for merely containing the gold answer.
    """
    canonical = canonical_single_entity(text)
    if not canonical:
        return set()

    seeds = {canonical}
    if is_ground_truth:
        alternatives = {
            part.strip() for part in re.split(r"\s+or\s+", canonical)
            if part.strip()
        }
        if len(alternatives) > 1:
            seeds.update(alternatives)

    variants: Set[str] = set()
    for value in seeds:
        variants.add(value)

        # State name <-> postal abbreviation, but only for an entire answer.
        if value in _US_STATE_ABBREVIATIONS:
            variants.add(_US_STATE_ABBREVIATIONS[value])
        elif value in _STATE_NAMES:
            variants.update(
                abbr for abbr, name in _US_STATE_ABBREVIATIONS.items()
                if name == value
            )

    return variants


def _single_entity_equivalent(prediction: Any, ground_truth: Any) -> bool:
    pred_variants = canonical_single_entity_variants(prediction)
    gold_variants = canonical_single_entity_variants(
        ground_truth, is_ground_truth=True
    )
    if pred_variants & gold_variants:
        return True

    number = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
    for pred in pred_variants:
        for gold in gold_variants:
            pred_tokens, gold_tokens = pred.split(), gold.split()

            # A bare value may omit the gold unit (or vice versa), but two
            # conflicting explicit units, such as 800 km vs 800 mi, remain
            # incorrect.
            if re.fullmatch(number, pred) and len(gold_tokens) == 2:
                if gold_tokens[1] in _SAFE_UNITS and pred == gold_tokens[0]:
                    return True
            if re.fullmatch(number, gold) and len(pred_tokens) == 2:
                if pred_tokens[1] in _SAFE_UNITS and gold == pred_tokens[0]:
                    return True

            # Accept one omitted map-network suffix.  Do not collapse two
            # different explicit suffixes.
            if len(gold_tokens) >= 2 and gold_tokens[-1] in _SAFE_ENTITY_SUFFIXES:
                if pred == " ".join(gold_tokens[:-1]):
                    return True
            if len(pred_tokens) >= 2 and pred_tokens[-1] in _SAFE_ENTITY_SUFFIXES:
                if gold == " ".join(pred_tokens[:-1]):
                    return True

    return False


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

    if answer_type == "Single Entity":
        correct = _single_entity_equivalent(prediction, ground_truth)
    else:
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
