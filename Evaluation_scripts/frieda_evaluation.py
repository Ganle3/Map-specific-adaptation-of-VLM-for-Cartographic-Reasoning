# python3
# -*- coding: utf-8 -*-
"""
FRIEDA-style evaluation for Qwen3-VL outputs.

Evaluation protocol:
1. Extract the text after the last "Final answer:" marker.
2. Route by answer_type:
   - textual: deterministic normalized match, then optional LLM-as-Judge
   - distance: unit-aware parsing + MAPE; correct when relative error <= 20%
   - cardinal: accept the gold direction and its two adjacent directions
3. Save per-question scores and aggregate accuracy by category.

The script uses only the Python standard library unless an optional
TransformersJudge is instantiated.

Expected prediction JSON formats
--------------------------------
A. List of result objects:
[
  {
    "question_ref": "q_0001",
    "raw_response": "... Final answer: ..."
  }
]

B. Dictionary keyed by question_ref:
{
  "q_0001": "... Final answer: ..."
}

Recognized response fields:
raw_response, model_output, response, prediction, generated_text, output
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import pickle
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol


SCRIPT_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger("frieda_evaluation")


RESPONSE_FIELDS = (
    "raw_response",
    "model_output",
    "response",
    "prediction",
    "generated_text",
    "output",
)

DEFAULT_ORIENTATION_MAP = {
    "North": ["North", "North West", "North East"],
    "North East": ["North East", "North", "East"],
    "East": ["East", "North East", "South East"],
    "South East": ["South East", "East", "South"],
    "South": ["South", "South East", "South West"],
    "South West": ["South West", "South", "West"],
    "West": ["West", "South West", "North West"],
    "North West": ["North West", "West", "North"],
}

UNIT_TO_METERS = {
    # metric
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "kilometre": 1000.0,
    "kilometres": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    # imperial / US customary
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "'": 0.3048,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
}


class JudgeProtocol(Protocol):
    def __call__(self, question: str, expected: str, response: str) -> int:
        """Return 1 for correct and 0 for incorrect."""


@dataclass
class EvaluationResult:
    correct: int
    evaluator: str
    extracted_answer: str
    normalized_expected: str = ""
    normalized_prediction: str = ""
    distance_expected_m: Optional[float] = None
    distance_prediction_m: Optional[float] = None
    absolute_percentage_error: Optional[float] = None
    note: str = ""


def extract_final_answer(raw_response: Any) -> str:
    """
    Extract the answer after the LAST 'Final answer:' marker.

    Qwen3-VL-Thinking normally emits:
        ... reasoning ...
        </think>

        Final answer: ...

    Using the last marker is safer if the phrase appears inside the reasoning.
    If no marker is found, text after </think> is used. As a final fallback,
    the whole response is returned.
    """
    if raw_response is None:
        return ""

    text = str(raw_response).strip()
    if not text:
        return ""

    matches = list(
        re.finditer(r"final\s+answer\s*:\s*", text, flags=re.IGNORECASE)
    )
    if matches:
        answer = text[matches[-1].end():].strip()
    elif "</think>" in text.lower():
        # Preserve original case while locating the case-insensitive marker.
        idx = text.lower().rfind("</think>")
        answer = text[idx + len("</think>"):].strip()
    else:
        answer = text

    # Remove common chat-template artefacts that may follow the answer.
    answer = re.split(
        r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Remove accidental wrapping quotes but preserve meaningful apostrophes.
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1].strip()

    return answer


def _ascii_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = (
        value.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )
    return value


def normalize_text(text: Any) -> str:
    """
    FRIEDA-inspired text normalization.

    This is intentionally conservative: it normalizes case, Unicode,
    punctuation, and whitespace without deleting digits or letters.
    """
    value = _ascii_text(text).casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compact_normalize(text: Any) -> str:
    """Normalization equivalent in spirit to FRIEDA's public helper."""
    return re.sub(r"[^a-z0-9]+", "", _ascii_text(text).casefold())


def split_answer_items(text: Any) -> list[str]:
    """
    Split multi-item answers.

    Gold FRIEDA answers use semicolons. Model answers often use commas,
    newlines, bullets, or 'and'. Semicolon splitting is preferred. Commas
    and conjunctions are used only when the answer visibly looks like a list.
    """
    value = _ascii_text(text).strip()
    if not value:
        return []

    # Remove a leading answer label if one survived extraction.
    value = re.sub(r"^\s*(?:answer|final answer)\s*:\s*", "", value, flags=re.I)

    if ";" in value:
        parts = re.split(r"\s*;\s*", value)
    elif "\n" in value:
        parts = re.split(r"\s*(?:\n+|•|\u2022)\s*", value)
    elif "," in value:
        parts = re.split(r"\s*,\s*(?:and\s+)?", value)
    elif re.search(r"\s+\band\b\s+", value, flags=re.I):
        parts = re.split(r"\s+\band\b\s+", value, flags=re.I)
    else:
        parts = [value]

    cleaned = []
    for part in parts:
        part = re.sub(r"^\s*[-*•]\s*", "", part).strip(" \t\r\n.;")
        if part:
            cleaned.append(part)
    return cleaned


def normalized_item_multiset(text: Any) -> list[str]:
    """
    Return sorted normalized items.

    A list rather than a set is used so duplicate predictions do not become
    silently correct.
    """
    return sorted(compact_normalize(x) for x in split_answer_items(text) if compact_normalize(x))


def deterministic_text_match(expected: Any, prediction: Any) -> tuple[bool, str]:
    """
    Apply deterministic matches before invoking an LLM judge.

    Checks:
    1. full normalized string equality;
    2. exact order-insensitive equality of multi-answer items.
    """
    exp_compact = compact_normalize(expected)
    pred_compact = compact_normalize(prediction)

    if exp_compact and exp_compact == pred_compact:
        return True, "normalized_exact"

    exp_items = normalized_item_multiset(expected)
    pred_items = normalized_item_multiset(prediction)
    if len(exp_items) > 1 and exp_items == pred_items:
        return True, "normalized_item_match"

    return False, "mismatch"


def build_frieda_judge_prompt(
    question: str,
    expected: str,
    response: str,
) -> str:
    """
    Reproduce the logic of FRIEDA Appendix E.1.

    The judge must evaluate all required items, regardless of order, and reject
    missing or additional items.
    """
    return f"""You will be given a triple consisting of a question, an expected answer, and a given response. Your task is to output either 'yes' or 'no'.

Given the question and response, extract only the exact portion of the text that serves as the answer from the given response. Then output 'yes' if the user response conveys the same meaning as the expected answer in relation to the question. Output 'no' if it does not.

For questions with multiple correct answers, the expected answers are separated by semicolons. The user response is correct if it matches all required answers, regardless of order. When the user provides more items than required, the response is incorrect. If the user lists fewer items than expected, mark the response as incorrect.

Differences in plurality, extra details such as acronyms or counts, minor typographical errors, and differences in wording style do not affect correctness. Focus only on whether the meaning matches.

Question: {question}
Expected answer: {expected}
Given response: {response}

Does the response correctly answer the question based on the expected answer? Answer strictly 'yes' or 'no'."""


class TransformersJudge:
    """
    Optional local Hugging Face judge.

    Run this in a separate evaluation process after Qwen inference so that the
    Qwen model and judge do not compete for GPU memory.

    Example model choices:
      - paper protocol: mistralai/Mistral-Small-3.1-24B-Instruct-2503
      - public FRIEDA code: mistralai/Ministral-8B-Instruct-2410

    The 24B judge will not fit comfortably on a 16 GB GPU without aggressive
    quantization/offloading. The 8B model is much more practical but is not
    identical to the paper's judge.
    """

    def __init__(
        self,
        model_name: str,
        *,
        load_in_4bit: bool = True,
        max_new_tokens: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "TransformersJudge requires torch and transformers."
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        kwargs: dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": "auto",
        }
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "4-bit loading requires bitsandbytes and a compatible "
                    "transformers installation."
                ) from exc

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def __call__(self, question: str, expected: str, response: str) -> int:
        prompt = build_frieda_judge_prompt(question, expected, response)
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip().casefold()

        return 1 if re.match(r"^yes\b", answer) else 0


def evaluate_textual(
    question: str,
    expected: str,
    extracted_answer: str,
    judge: Optional[JudgeProtocol] = None,
) -> EvaluationResult:
    matched, method = deterministic_text_match(expected, extracted_answer)
    if matched:
        return EvaluationResult(
            correct=1,
            evaluator=method,
            extracted_answer=extracted_answer,
            normalized_expected=normalize_text(expected),
            normalized_prediction=normalize_text(extracted_answer),
        )

    if judge is None:
        return EvaluationResult(
            correct=0,
            evaluator="text_mismatch_no_judge",
            extracted_answer=extracted_answer,
            normalized_expected=normalize_text(expected),
            normalized_prediction=normalize_text(extracted_answer),
            note="Deterministic match failed and no LLM judge was configured.",
        )

    try:
        correct = int(bool(judge(question, expected, extracted_answer)))
        return EvaluationResult(
            correct=correct,
            evaluator="llm_judge",
            extracted_answer=extracted_answer,
            normalized_expected=normalize_text(expected),
            normalized_prediction=normalize_text(extracted_answer),
        )
    except Exception as exc:
        return EvaluationResult(
            correct=0,
            evaluator="llm_judge_error",
            extracted_answer=extracted_answer,
            normalized_expected=normalize_text(expected),
            normalized_prediction=normalize_text(extracted_answer),
            note=f"{type(exc).__name__}: {exc}",
        )


@dataclass(frozen=True)
class ParsedDistance:
    value: float
    unit: Optional[str]
    value_meters: Optional[float]


def _canonical_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    value = _ascii_text(unit).casefold().strip().rstrip(".")
    aliases = {
        "kms": "km",
        "ms": "m",
        "fts": "ft",
        "mis": "mi",
    }
    return aliases.get(value, value)


def parse_distance(text: Any) -> Optional[ParsedDistance]:
    """
    Parse the first plausible number and an optional distance unit.

    Examples:
      40 m
      0.62 Miles
      7,500 feet
      approximately 525 metres
      25000
    """
    value = _ascii_text(text).casefold()
    value = value.replace(",", "")

    unit_pattern = (
        r"kilometers?|kilometres?|km|"
        r"meters?|metres?|m|"
        r"centimeters?|centimetres?|cm|"
        r"miles?|mi|"
        r"feet|foot|ft|"
        r"yards?|yd|"
        r"'"
    )
    match = re.search(
        rf"(?<![\w.])([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*({unit_pattern})?\b",
        value,
        flags=re.I,
    )
    if not match:
        # Handle a feet apostrophe, where \b after apostrophe is unreliable.
        match = re.search(r"(?<![\w.])([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*'", value)
    if not match:
        return None

    number = float(match.group(1))
    unit = _canonical_unit(match.group(2) if match.lastindex and match.lastindex >= 2 else None)

    if unit in UNIT_TO_METERS:
        meters = number * UNIT_TO_METERS[unit]
    else:
        meters = None

    return ParsedDistance(number, unit, meters)


def evaluate_distance(
    expected: str,
    extracted_answer: str,
    tolerance: float = 0.20,
) -> EvaluationResult:
    gold = parse_distance(expected)
    pred = parse_distance(extracted_answer)

    if gold is None or pred is None:
        return EvaluationResult(
            correct=0,
            evaluator="distance_parse_error",
            extracted_answer=extracted_answer,
            note=f"Parsed expected={gold!r}; parsed prediction={pred!r}",
        )

    # Unit-aware comparison when both units are known. If the reference has no
    # unit, compare numeric values directly because the question may explicitly
    # state that a unit is not required.
    if gold.value_meters is not None and pred.value_meters is not None:
        gold_value = gold.value_meters
        pred_value = pred.value_meters
        expected_m = gold.value_meters
        prediction_m = pred.value_meters
    elif gold.unit is None:
        gold_value = gold.value
        pred_value = pred.value
        expected_m = None
        prediction_m = None
    elif pred.unit is None:
        # FRIEDA questions often make the expected unit explicit. Treat a
        # unitless prediction as using the gold unit.
        gold_value = gold.value
        pred_value = pred.value
        expected_m = gold.value_meters
        prediction_m = (
            pred.value * UNIT_TO_METERS[gold.unit]
            if gold.unit in UNIT_TO_METERS
            else None
        )
    else:
        return EvaluationResult(
            correct=0,
            evaluator="distance_unit_error",
            extracted_answer=extracted_answer,
            note=f"Incompatible/unknown units: expected={gold.unit}, prediction={pred.unit}",
        )

    if math.isclose(gold_value, 0.0, abs_tol=1e-12):
        ape = 0.0 if math.isclose(pred_value, 0.0, abs_tol=1e-12) else math.inf
    else:
        ape = abs(pred_value - gold_value) / abs(gold_value)

    return EvaluationResult(
        correct=int(ape <= tolerance),
        evaluator="distance_mape",
        extracted_answer=extracted_answer,
        distance_expected_m=expected_m,
        distance_prediction_m=prediction_m,
        absolute_percentage_error=ape,
        note=f"Tolerance={tolerance:.0%}",
    )


def canonicalize_direction(text: Any) -> Optional[str]:
    """
    Canonicalize a response to one of FRIEDA's eight directions.

    The function prioritizes compound directions so 'North West' is not
    accidentally reduced to 'North'.
    """
    value = normalize_text(text)
    if not value:
        return None

    replacements = {
        "northwestern": "north west",
        "northwest": "north west",
        "north western": "north west",
        "northeastern": "north east",
        "northeast": "north east",
        "north eastern": "north east",
        "southwestern": "south west",
        "southwest": "south west",
        "south western": "south west",
        "southeastern": "south east",
        "southeast": "south east",
        "south eastern": "south east",
    }
    for source, target in replacements.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)

    patterns = [
        ("North West", r"\b(?:north west|nw)\b"),
        ("North East", r"\b(?:north east|ne)\b"),
        ("South West", r"\b(?:south west|sw)\b"),
        ("South East", r"\b(?:south east|se)\b"),
        ("North", r"\b(?:north|northern|n)\b"),
        ("South", r"\b(?:south|southern|s)\b"),
        ("East", r"\b(?:east|eastern|e)\b"),
        ("West", r"\b(?:west|western|w)\b"),
    ]
    for canonical, pattern in patterns:
        if re.search(pattern, value):
            return canonical
    return None


def load_orientation_map(path: Optional[str | Path] = None) -> dict[str, list[str]]:
    if path is None:
        return {k: list(v) for k, v in DEFAULT_ORIENTATION_MAP.items()}

    with Path(path).open("rb") as handle:
        loaded = pickle.load(handle)

    if not isinstance(loaded, dict):
        raise TypeError("orientation.pkl must contain a dictionary.")

    normalized: dict[str, list[str]] = {}
    for key, values in loaded.items():
        canonical_key = canonicalize_direction(key)
        if canonical_key is None:
            continue
        if isinstance(values, str):
            values = [values]
        valid = [canonicalize_direction(x) for x in values]
        normalized[canonical_key] = [x for x in valid if x is not None]

    if not normalized:
        raise ValueError("No valid direction mappings found in orientation.pkl.")
    return normalized


def evaluate_cardinal(
    expected: str,
    extracted_answer: str,
    orientation_map: Mapping[str, Iterable[str]],
) -> EvaluationResult:
    gold = canonicalize_direction(expected)
    pred = canonicalize_direction(extracted_answer)

    if gold is None or pred is None:
        return EvaluationResult(
            correct=0,
            evaluator="cardinal_parse_error",
            extracted_answer=extracted_answer,
            normalized_expected=str(gold or ""),
            normalized_prediction=str(pred or ""),
        )

    accepted = {
        canonicalize_direction(x)
        for x in orientation_map.get(gold, [gold])
    }
    accepted.discard(None)

    return EvaluationResult(
        correct=int(pred in accepted),
        evaluator="cardinal_adjacent",
        extracted_answer=extracted_answer,
        normalized_expected=gold,
        normalized_prediction=pred,
        note="Accepted: " + "; ".join(sorted(accepted)),
    )


def evaluate_sample(
    sample: Mapping[str, Any],
    raw_response: Any,
    *,
    judge: Optional[JudgeProtocol] = None,
    orientation_map: Optional[Mapping[str, Iterable[str]]] = None,
    distance_tolerance: float = 0.20,
) -> dict[str, Any]:
    extracted = extract_final_answer(raw_response)
    answer_type = str(sample.get("answer_type", "textual")).casefold().strip()
    expected = str(sample.get("expected_answer", ""))
    question = str(sample.get("question_text", ""))

    if answer_type in {"cardinal", "direction", "orientation"}:
        result = evaluate_cardinal(
            expected,
            extracted,
            orientation_map or DEFAULT_ORIENTATION_MAP,
        )
    elif answer_type in {"distance", "metric"}:
        result = evaluate_distance(
            expected,
            extracted,
            tolerance=distance_tolerance,
        )
    else:
        result = evaluate_textual(
            question,
            expected,
            extracted,
            judge=judge,
        )

    row = dict(sample)
    row.update(
        {
            "raw_response": "" if raw_response is None else str(raw_response),
            "extracted_answer": result.extracted_answer,
            "correct": result.correct,
            "evaluator": result.evaluator,
            "normalized_expected": result.normalized_expected,
            "normalized_prediction": result.normalized_prediction,
            "distance_expected_m": result.distance_expected_m,
            "distance_prediction_m": result.distance_prediction_m,
            "absolute_percentage_error": result.absolute_percentage_error,
            "evaluation_note": result.note,
        }
    )
    return row


def _extract_response_from_record(record: Mapping[str, Any]) -> str:
    for field in RESPONSE_FIELDS:
        if field in record and record[field] is not None:
            return str(record[field])
    return ""


def load_predictions(path: str | Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    predictions: dict[str, str] = {}

    if isinstance(obj, dict):
        # Common wrapper structures.
        for wrapper_key in ("results", "predictions", "data"):
            if wrapper_key in obj and isinstance(obj[wrapper_key], list):
                obj = obj[wrapper_key]
                break

    if isinstance(obj, list):
        for record in obj:
            if not isinstance(record, dict):
                continue
            qid = (
                record.get("question_ref")
                or record.get("qa_id")
                or record.get("id")
            )
            if qid is None:
                continue
            predictions[str(qid)] = _extract_response_from_record(record)
        return predictions

    if isinstance(obj, dict):
        for qid, value in obj.items():
            if isinstance(value, dict):
                predictions[str(qid)] = _extract_response_from_record(value)
            else:
                predictions[str(qid)] = "" if value is None else str(value)
        return predictions

    raise ValueError("Unsupported prediction JSON structure.")


def summarize_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize_group(items: list[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(items)
        correct = sum(int(x.get("correct", 0)) for x in items)
        return {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else None,
        }

    summary: dict[str, Any] = {
        "overall": summarize_group(rows),
        "by_answer_type": {},
        "by_spatial_relationship": {},
        "by_map_count": {},
        "by_domain": {},
        "by_evaluator": {},
    }

    dimensions = {
        "by_answer_type": "answer_type",
        "by_spatial_relationship": "spatial_relationship",
        "by_map_count": "map_count",
        "by_domain": "domain",
        "by_evaluator": "evaluator",
    }

    for output_key, field in dimensions.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "Unknown"))].append(row)
        summary[output_key] = {
            key: summarize_group(group_rows)
            for key, group_rows in sorted(groups.items())
        }

    distance_rows = [
        row for row in rows
        if row.get("absolute_percentage_error") is not None
        and math.isfinite(float(row["absolute_percentage_error"]))
    ]
    if distance_rows:
        summary["distance_mape"] = sum(
            float(row["absolute_percentage_error"]) for row in distance_rows
        ) / len(distance_rows)
    else:
        summary["distance_mape"] = None

    summary["missing_or_empty_predictions"] = sum(
        not str(row.get("raw_response", "")).strip() for row in rows
    )
    return summary


def evaluate_dataset(
    qa_data: list[Mapping[str, Any]],
    predictions: Mapping[str, str],
    *,
    judge: Optional[JudgeProtocol] = None,
    orientation_map: Optional[Mapping[str, Iterable[str]]] = None,
    distance_tolerance: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for sample in qa_data:
        qid = str(sample.get("question_ref", sample.get("qa_id", "")))
        raw_response = predictions.get(qid, "")
        rows.append(
            evaluate_sample(
                sample,
                raw_response,
                judge=judge,
                orientation_map=orientation_map,
                distance_tolerance=distance_tolerance,
            )
        )
    return rows, summarize_results(rows)


def save_results(
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save detailed and aggregate evaluation outputs.

    Returns the generated paths so orchestration scripts can record them.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    details_json = output / "evaluation_details.json"
    summary_json = output / "evaluation_summary.json"
    details_csv = output / "evaluation_details.csv"

    with details_json.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    csv_fields = [
        "question_ref",
        "answer_type",
        "spatial_relationship",
        "map_count",
        "domain",
        "question_text",
        "expected_answer",
        "extracted_answer",
        "correct",
        "evaluator",
        "normalized_expected",
        "normalized_prediction",
        "distance_expected_m",
        "distance_prediction_m",
        "absolute_percentage_error",
        "evaluation_note",
    ]
    with details_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "details_json": details_json,
        "summary_json": summary_json,
        "details_csv": details_csv,
    }


def load_qa_data(path: str | Path) -> list[Mapping[str, Any]]:
    """Load a FRIEDA QA list, including common wrapper structures."""
    qa_path = Path(path)
    with qa_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, dict):
        for key in ("data", "questions", "results", "items"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError(
            f"QA JSON must contain a list of question objects: {qa_path}"
        )

    rows = [x for x in obj if isinstance(x, Mapping)]
    if len(rows) != len(obj):
        raise ValueError("Every QA entry must be a JSON object.")
    return rows


def load_prediction_records(path: str | Path) -> list[Mapping[str, Any]]:
    """Load raw prediction records when they already contain QA metadata."""
    with Path(path).open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        return []
    return [x for x in obj if isinstance(x, Mapping)]


def _qa_from_prediction_records(
    records: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Use prediction records as QA metadata when inference saved gold fields."""
    required = {"question_ref", "expected_answer"}
    if not records or any(not required.issubset(record) for record in records):
        return []

    qa_fields = {
        "question_ref",
        "qa_id",
        "id",
        "question_text",
        "expected_answer",
        "answer_type",
        "spatial_relationship",
        "map_count",
        "domain",
        "image_urls",
    }
    return [
        {key: value for key, value in record.items() if key in qa_fields}
        for record in records
    ]


def _flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Add stable convenience keys for run_frieda.py and later experiments."""
    overall = summary.get("overall", {})
    summary["total"] = overall.get("total", 0)
    summary["correct"] = overall.get("correct", 0)
    summary["overall_accuracy"] = overall.get("accuracy")

    answer_groups = summary.get("by_answer_type", {})

    def group_accuracy(*names: str) -> Optional[float]:
        for name in names:
            for key, value in answer_groups.items():
                if str(key).casefold() == name.casefold():
                    return value.get("accuracy")
        return None

    summary["textual_accuracy"] = group_accuracy("textual", "text")
    summary["distance_accuracy"] = group_accuracy("distance", "metric")
    summary["cardinal_accuracy"] = group_accuracy(
        "cardinal", "direction", "orientation"
    )
    return summary


def evaluate_frieda(
    prediction_json: str | Path,
    qa_json: Optional[str | Path] = None,
    *,
    output_dir: Optional[str | Path] = None,
    orientation_pkl: Optional[str | Path] = None,
    distance_tolerance: float = 0.20,
    judge: Optional[JudgeProtocol] = None,
    judge_model: Optional[str] = None,
    judge_load_in_4bit: bool = True,
) -> dict[str, Any]:
    """Evaluate a FRIEDA prediction file and save all result artifacts.

    Parameters
    ----------
    prediction_json:
        Path to ``frieda_predictions.json``.
    qa_json:
        Path to the held-out FRIEDA QA JSON. This may be omitted only when
        prediction records already contain ``question_ref``,
        ``expected_answer``, ``question_text`` and ``answer_type`` metadata.
    output_dir:
        Destination for evaluation outputs. Defaults to the prediction file's
        parent directory.
    judge / judge_model:
        Provide either a callable judge or a Hugging Face judge model name.
        Deterministic matching is always attempted first.

    Returns
    -------
    dict
        Aggregate summary with convenience keys such as
        ``overall_accuracy``, ``textual_accuracy``, ``distance_accuracy`` and
        ``cardinal_accuracy``.
    """
    prediction_path = Path(prediction_json).resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {prediction_path}")

    if qa_json is not None:
        qa_path = Path(qa_json).resolve()
        if not qa_path.exists():
            raise FileNotFoundError(f"QA JSON not found: {qa_path}")
        qa_data = load_qa_data(qa_path)
    else:
        records = load_prediction_records(prediction_path)
        qa_data = _qa_from_prediction_records(records)
        if not qa_data:
            raise ValueError(
                "qa_json was omitted, but the prediction records do not "
                "contain sufficient ground-truth metadata. Pass qa_json."
            )

    predictions = load_predictions(prediction_path)
    orientation_map = load_orientation_map(orientation_pkl)

    if judge is not None and judge_model is not None:
        raise ValueError("Provide either judge or judge_model, not both.")
    if judge_model is not None:
        judge = TransformersJudge(
            judge_model,
            load_in_4bit=judge_load_in_4bit,
        )

    rows, summary = evaluate_dataset(
        qa_data,
        predictions,
        judge=judge,
        orientation_map=orientation_map,
        distance_tolerance=distance_tolerance,
    )
    summary = _flatten_summary(summary)

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else prediction_path.parent
    )
    paths = save_results(rows, summary, destination)
    summary["artifacts"] = {key: str(path.resolve()) for key, path in paths.items()}

    # Re-save after adding artifact paths.
    with paths["summary_json"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def _format_accuracy(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f} ({value:.2%})"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL outputs using the FRIEDA protocol."
    )
    parser.add_argument(
        "--qa-json",
        default=None,
        help=(
            "FRIEDA test QA JSON. Optional only when predictions already "
            "contain all required gold metadata."
        ),
    )
    parser.add_argument(
        "--predictions-json",
        required=True,
        help="Path to frieda_predictions.json.",
    )
    parser.add_argument(
        "--orientation-pkl",
        default=None,
        help="Optional FRIEDA orientation.pkl; otherwise built-in mapping is used.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to the prediction file's directory.",
    )
    parser.add_argument(
        "--distance-tolerance",
        type=float,
        default=0.20,
        help="Maximum absolute percentage error counted as correct.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Optional Hugging Face textual-answer judge. When omitted, "
            "deterministic mismatches are marked incorrect."
        ),
    )
    parser.add_argument(
        "--judge-no-4bit",
        action="store_true",
        help="Load the optional judge without 4-bit quantization.",
    )
    return parser.parse_args()


def print_summary(summary: Mapping[str, Any]) -> None:
    print("=" * 72)
    print("FRIEDA EVALUATION")
    print("=" * 72)
    print(f"Questions:             {summary.get('total', 0)}")
    print(f"Correct:               {summary.get('correct', 0)}")
    print(f"Overall accuracy:      {_format_accuracy(summary.get('overall_accuracy'))}")
    print(f"Textual accuracy:      {_format_accuracy(summary.get('textual_accuracy'))}")
    print(f"Distance accuracy:     {_format_accuracy(summary.get('distance_accuracy'))}")
    print(f"Cardinal accuracy:     {_format_accuracy(summary.get('cardinal_accuracy'))}")

    distance_mape = summary.get("distance_mape")
    if distance_mape is not None:
        print(f"Distance MAPE:         {distance_mape:.4f} ({distance_mape:.2%})")

    print(
        "Missing predictions:   "
        f"{summary.get('missing_or_empty_predictions', 0)}"
    )
    artifacts = summary.get("artifacts", {})
    if artifacts:
        print(f"Summary JSON:          {artifacts.get('summary_json', '')}")
        print(f"Details JSON:          {artifacts.get('details_json', '')}")
        print(f"Details CSV:           {artifacts.get('details_csv', '')}")


def main() -> None:
    args = parse_args()
    summary = evaluate_frieda(
        prediction_json=args.predictions_json,
        qa_json=args.qa_json,
        output_dir=args.output_dir,
        orientation_pkl=args.orientation_pkl,
        distance_tolerance=args.distance_tolerance,
        judge_model=args.judge_model,
        judge_load_in_4bit=not args.judge_no_4bit,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()