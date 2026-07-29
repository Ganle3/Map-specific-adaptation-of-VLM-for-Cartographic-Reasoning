# python3
# -*- coding: utf-8 -*-
"""Deterministic MapWise evaluation for Qwen3-VL predictions."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


def _ascii_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return (text.replace("–", "-").replace("—", "-").replace("−", "-")
            .replace("’", "'").replace("‘", "'").replace("“", '"')
            .replace("”", '"').replace("&", " and "))


def normalize_text(value: Any) -> str:
    text = _ascii_text(value).casefold().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii_text(value).casefold())


def extract_final_answer(raw_response: Any) -> str:
    if raw_response is None:
        return ""
    text = str(raw_response).strip()
    if not text:
        return ""
    matches = list(re.finditer(r"final\s+answer\s*:\s*", text, flags=re.I))
    if matches:
        answer = text[matches[-1].end():].strip()
    elif "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        answer = text[idx + len("</think>"):].strip()
    else:
        return ""
    answer = re.split(r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)", answer,
                      maxsplit=1, flags=re.I)[0].strip()
    if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in {'"', "'"}:
        answer = answer[1:-1].strip()
    return answer


def resolve_prediction_answer(record: Mapping[str, Any]) -> tuple[str, str]:
    saved = str(record.get("final_answer", "") or "").strip()
    if saved:
        return saved, "saved_final_answer"
    extracted = extract_final_answer(record.get("raw_response", ""))
    if extracted:
        return extracted, "raw_response_reextract"
    return "", "not_extractable"


# ============================================================
# Canonical Indian state / union-territory names and aliases
# ============================================================

CANONICAL_PLACE_ALIASES: dict[str, set[str]] = {
    "andhra pradesh": {"andhra pradesh", "ap"},
    "arunachal pradesh": {"arunachal pradesh", "ar"},
    "assam": {"assam", "as"},
    "bihar": {"bihar", "br"},
    "chhattisgarh": {"chhattisgarh", "chattisgarh", "cg", "ct"},
    "goa": {"goa", "ga"},
    "gujarat": {"gujarat", "gj"},
    "haryana": {"haryana", "hr"},
    "himachal pradesh": {"himachal pradesh", "hp"},
    "jharkhand": {"jharkhand", "jh"},
    "karnataka": {"karnataka", "ka"},
    "kerala": {"kerala", "kl"},
    "madhya pradesh": {"madhya pradesh", "mp"},
    "maharashtra": {"maharashtra", "mh"},
    "manipur": {"manipur", "mn"},
    "meghalaya": {"meghalaya", "ml"},
    "mizoram": {"mizoram", "mz"},
    "nagaland": {"nagaland", "nl"},
    "odisha": {"odisha", "orissa", "od", "or"},
    "punjab": {"punjab", "pb"},
    "rajasthan": {"rajasthan", "rajasathan", "rajastan", "rj"},
    "sikkim": {"sikkim", "sk"},
    "tamil nadu": {"tamil nadu", "tamilnadu", "tn"},
    "telangana": {"telangana", "tg", "ts"},
    "tripura": {"tripura", "tr"},
    "uttar pradesh": {"uttar pradesh", "up"},
    "uttarakhand": {"uttarakhand", "uttaranchal", "uk", "ut"},
    "west bengal": {"west bengal", "westbengal", "wb"},
    "andaman and nicobar islands": {
        "andaman and nicobar islands",
        "andaman nicobar islands",
        "andaman and nicobar",
        "a and n islands",
        "an",
    },
    "chandigarh": {"chandigarh", "ch"},
    "dadra and nagar haveli and daman and diu": {
        "dadra and nagar haveli and daman and diu",
        "dadra nagar haveli daman diu",
        "dadra and nagar haveli",
        "daman and diu",
        "dn",
        "dd",
        "dnhdd",
    },
    "delhi": {
        "delhi",
        "new delhi",
        "nct of delhi",
        "national capital territory of delhi",
        "dl",
    },
    "jammu and kashmir": {
        "jammu and kashmir",
        "jammu kashmir",
        "jammu and kashmir state",
        "j and k",
        "jk",
    },
    "ladakh": {"ladakh", "la"},
    "lakshadweep": {"lakshadweep", "ld"},
    "puducherry": {"puducherry", "pondicherry", "py"},
}


def _build_place_alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}

    for canonical_name, aliases in CANONICAL_PLACE_ALIASES.items():
        normalized_canonical = normalize_text(canonical_name)

        for alias in aliases | {canonical_name}:
            normalized_alias = normalize_text(alias)
            existing = lookup.get(normalized_alias)

            if existing is not None and existing != normalized_canonical:
                raise ValueError(
                    f"Ambiguous place alias {alias!r}: "
                    f"{existing!r} and {normalized_canonical!r}"
                )

            lookup[normalized_alias] = normalized_canonical

    return lookup


PLACE_ALIAS_LOOKUP = _build_place_alias_lookup()


def canonicalize_item(value: Any) -> str:
    """
    Convert a full state/UT name, abbreviation, or spelling variant into one
    canonical lowercase name. Unknown text remains normalized rather than
    being silently discarded.
    """
    text = normalize_text(value)

    text = re.sub(
        r"^(?:the\s+)?"
        r"(?:(?:state|states|union territory|union territories|"
        r"territory|territories)\s+(?:of\s+)?)",
        "",
        text,
    ).strip()

    text = re.sub(
        r"\s+(?:state|states|union territory|union territories|"
        r"territory|territories)$",
        "",
        text,
    ).strip()

    return PLACE_ALIAS_LOOKUP.get(text, text)


def _protect_internal_place_conjunctions(text: str) -> str:
    """
    Protect official names containing 'and' before generic list splitting.
    """
    replacements = {
        r"andaman\s+(?:and|&)\s+nicobar(?:\s+islands)?":
            "andaman__place_and__nicobar islands",
        r"jammu\s+(?:and|&)\s+kashmir":
            "jammu__place_and__kashmir",
        r"dadra\s+(?:and|&)\s+nagar\s+haveli\s+(?:and|&)\s+"
        r"daman\s+(?:and|&)\s+diu":
            "dadra__place_and__nagar haveli"
            "__place_and__daman__place_and__diu",
        r"a\s+(?:and|&)\s+n\s+islands":
            "a__place_and__n islands",
    }

    protected = text
    for pattern, replacement in replacements.items():
        protected = re.sub(pattern, replacement, protected, flags=re.I)

    return protected


def split_answer_items(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []

    text = re.sub(
        r"^\s*(?:answer|final answer)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = _protect_internal_place_conjunctions(text)

    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = re.sub(r"[\r\n•]+", ";", text)
    text = re.sub(r"\s*/\s*", ";", text)
    text = re.sub(r"\s*;\s*", ";", text)
    text = re.sub(r"\s*,\s*", ";", text)
    text = re.sub(r"\s+(?:and|&)\s+", ";", text, flags=re.I)

    parts = [part.strip(" \t.;:-") for part in text.split(";")]

    output: list[str] = []
    for part in parts:
        restored = part.replace("__place_and__", " and ")
        canonical = canonicalize_item(restored)
        if canonical:
            output.append(canonical)

    return output


def parse_binary(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    tokens = re.findall(r"\b(?:yes|no)\b", text)
    return tokens[0] if tokens and len(set(tokens)) == 1 else None


NUMBER_WORD_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
NUMBER_WORD_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def parse_number_words(text: str) -> Optional[int]:
    tokens = normalize_text(text).replace("-", " ").split()
    for i, token in enumerate(tokens):
        if token in NUMBER_WORD_UNITS:
            return NUMBER_WORD_UNITS[token]
        if token in NUMBER_WORD_TENS:
            value = NUMBER_WORD_TENS[token]
            if i + 1 < len(tokens) and tokens[i + 1] in NUMBER_WORD_UNITS:
                value += NUMBER_WORD_UNITS[tokens[i + 1]]
            return value
    return None


def parse_count(value: Any) -> Optional[int]:
    text = _ascii_text(value).casefold().replace(",", "")
    match = re.search(r"(?<![\w.])-?\d+(?![\w.])", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            pass
    return parse_number_words(text)


NUMBER_PATTERN = r"[+-]?\s*(?:\d[\d,]*\.?\d*|\.\d+)\s*[kKmM]?"


def parse_scaled_number(value: str) -> float:
    text = value.strip().replace(" ", "").replace(",", "")
    multiplier = 1.0
    if text[-1:].casefold() == "k":
        multiplier, text = 1_000.0, text[:-1]
    elif text[-1:].casefold() == "m":
        multiplier, text = 1_000_000.0, text[:-1]
    return float(text) * multiplier


def parse_range(value: Any) -> Optional[tuple[float, float]]:
    text = _ascii_text(value)
    match = re.search(rf"({NUMBER_PATTERN})\s*(?:-|to)\s*({NUMBER_PATTERN})", text, flags=re.I)
    if not match:
        return None
    try:
        first, second = parse_scaled_number(match.group(1)), parse_scaled_number(match.group(2))
    except ValueError:
        return None
    return min(first, second), max(first, second)


def numbers_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def ranges_exact(gold: tuple[float, float], pred: tuple[float, float]) -> bool:
    return numbers_close(gold[0], pred[0]) and numbers_close(gold[1], pred[1])


def ranges_strictly_overlap(gold: tuple[float, float], pred: tuple[float, float]) -> bool:
    return max(gold[0], pred[0]) < min(gold[1], pred[1])


RANK_OPERATOR_PATTERN = re.compile(r"\s*(<=|>=|<|>|=)\s*")


def parse_ranking(value: Any) -> Optional[list[set[str]]]:
    text = _ascii_text(value).strip()
    parts = RANK_OPERATOR_PATTERN.split(text)
    if len(parts) < 3 or len(parts) % 2 == 0:
        return None
    items = [canonicalize_item(parts[i]) for i in range(0, len(parts), 2)]
    operators = [parts[i] for i in range(1, len(parts), 2)]
    if any(not item for item in items):
        return None
    directional = {op for op in operators if op != "="}
    if directional.issubset({"<", "<="}):
        descending = False
    elif directional.issubset({">", ">="}):
        descending = True
    elif not directional:
        descending = False
    else:
        return None
    groups: list[set[str]] = [{items[0]}]
    for op, item in zip(operators, items[1:]):
        if op == "=":
            groups[-1].add(item)
        else:
            groups.append({item})
    if descending:
        groups.reverse()
    return groups


def exact_rank_match(gold: list[set[str]], pred: list[set[str]]) -> bool:
    return len(gold) == len(pred) and all(g == p for g, p in zip(gold, pred))


def rankwise_precision(gold: list[set[str]], pred: list[set[str]]) -> tuple[float, list[float]]:
    n = max(len(gold), len(pred))
    if n == 0:
        return 0.0, []
    scores = []
    for i in range(n):
        if i >= len(gold) or i >= len(pred) or not pred[i]:
            scores.append(0.0)
        else:
            scores.append(len(gold[i] & pred[i]) / len(pred[i]))
    return sum(scores) / n, scores


def rankwise_mrr_and_map(gold: list[set[str]], pred: list[set[str]]) -> tuple[float, float, list[int]]:
    n = max(len(gold), len(pred))
    if n == 0:
        return 0.0, 0.0, []
    hits = [int(i < len(gold) and i < len(pred) and bool(gold[i] & pred[i])) for i in range(n)]
    first = next((i for i, hit in enumerate(hits) if hit), None)
    mrr = 0.0 if first is None else 1.0 / (first + 1)
    cumulative = 0
    precisions = []
    for i, hit in enumerate(hits, start=1):
        if hit:
            cumulative += 1
            precisions.append(cumulative / i)
    map_score = sum(precisions) / max(len(gold), 1)
    return mrr, map_score, hits


def evaluate_binary(gold: str, pred: str) -> dict[str, Any]:
    g, p = parse_binary(gold), parse_binary(pred)
    exact = g is not None and p is not None and g == p
    return {"primary_score": float(exact), "strict_exact_match": int(exact),
            "metric": "binary_exact_match", "normalized_ground_truth": g or "",
            "normalized_prediction": p or "", "evaluation_note": ""}


def evaluate_count(gold: str, pred: str) -> dict[str, Any]:
    g, p = parse_count(gold), parse_count(pred)
    exact = g is not None and p is not None and g == p
    return {"primary_score": float(exact), "strict_exact_match": int(exact),
            "metric": "count_exact_match", "normalized_ground_truth": g,
            "normalized_prediction": p, "evaluation_note": ""}


def evaluate_single(gold: str, pred: str) -> dict[str, Any]:
    g, p = set(split_answer_items(gold)), set(split_answer_items(pred))
    inter = g & p
    precision = len(inter) / len(p) if p else 0.0
    recall = len(inter) / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"primary_score": recall, "strict_exact_match": int(bool(g) and g == p),
            "metric": "single_recall", "precision": precision, "recall": recall,
            "f1": f1, "normalized_ground_truth": sorted(g),
            "normalized_prediction": sorted(p), "evaluation_note": ""}


def evaluate_list(gold: str, pred: str) -> dict[str, Any]:
    g, p = set(split_answer_items(gold)), set(split_answer_items(pred))
    inter = g & p
    precision = len(inter) / len(p) if p else 0.0
    recall = len(inter) / len(g) if g else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"primary_score": f1, "strict_exact_match": int(bool(g) and g == p),
            "metric": "list_f1", "precision": precision, "recall": recall,
            "f1": f1, "normalized_ground_truth": sorted(g),
            "normalized_prediction": sorted(p), "evaluation_note": ""}


def evaluate_range(gold: str, pred: str, legend_style: str) -> dict[str, Any]:
    g, p = parse_range(gold), parse_range(pred)
    if g is None or p is None:
        return {"primary_score": 0.0, "strict_exact_match": 0,
                "metric": "range_parse_error", "normalized_ground_truth": g,
                "normalized_prediction": p, "range_relation": "parse_error",
                "evaluation_note": f"Parsed ground_truth={g!r}; prediction={p!r}"}
    exact = ranges_exact(g, p)
    style = str(legend_style or "").casefold().strip()
    if exact:
        score, relation = 1.0, "exact"
    elif style == "c" and ranges_strictly_overlap(g, p):
        score, relation = 0.5, "overlap"
    else:
        score, relation = 0.0, "no_match"
    return {"primary_score": score, "strict_exact_match": int(exact),
            "metric": "continuous_range_score" if style == "c" else "discrete_range_exact_match",
            "normalized_ground_truth": list(g), "normalized_prediction": list(p),
            "range_relation": relation,
            "evaluation_note": "Continuous: exact=1, strict overlap=0.5, touching=0." if style == "c" else "Discrete: exact normalized range match only."}


def evaluate_rank(gold: str, pred: str) -> dict[str, Any]:
    g, p = parse_ranking(gold), parse_ranking(pred)
    if g is None or p is None:
        return {"primary_score": 0.0, "strict_exact_match": 0,
                "metric": "rank_parse_error", "rankwise_precision": 0.0,
                "mrr": 0.0, "map": 0.0, "rank_precisions": [], "rank_hits": [],
                "normalized_ground_truth": [sorted(x) for x in g] if g else None,
                "normalized_prediction": [sorted(x) for x in p] if p else None,
                "evaluation_note": "Could not parse one or both ranking expressions."}
    rwp, per_rank = rankwise_precision(g, p)
    mrr, map_score, hits = rankwise_mrr_and_map(g, p)
    return {"primary_score": rwp, "strict_exact_match": int(exact_rank_match(g, p)),
            "metric": "rankwise_precision", "rankwise_precision": rwp,
            "mrr": mrr, "map": map_score, "rank_precisions": per_rank,
            "rank_hits": hits, "normalized_ground_truth": [sorted(x) for x in g],
            "normalized_prediction": [sorted(x) for x in p], "evaluation_note": ""}


def evaluate_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    pred, extraction_method = resolve_prediction_answer(record)
    gold = str(record.get("ground_truth", "") or "").strip()
    answer_type = str(record.get("ground_truth_type", "")).casefold().strip()
    template_no = int(record.get("template_no", -1))
    legend_style = str(record.get("legend_style", record.get("c_or_d", ""))).strip()
    if not pred:
        result = {"primary_score": 0.0, "strict_exact_match": 0,
                  "metric": "not_extractable", "normalized_ground_truth": "",
                  "normalized_prediction": "", "evaluation_note": "No extractable final answer."}
    elif template_no == 43:
        result = evaluate_rank(gold, pred)
    elif answer_type == "binary":
        result = evaluate_binary(gold, pred)
    elif answer_type == "count":
        result = evaluate_count(gold, pred)
    elif answer_type == "range":
        result = evaluate_range(gold, pred, legend_style)
    elif answer_type == "list":
        result = evaluate_list(gold, pred)
    elif answer_type == "single":
        result = evaluate_single(gold, pred)
    else:
        exact = bool(compact_normalize(gold)) and compact_normalize(gold) == compact_normalize(pred)
        result = {"primary_score": float(exact), "strict_exact_match": int(exact),
                  "metric": "fallback_text_exact_match",
                  "normalized_ground_truth": normalize_text(gold),
                  "normalized_prediction": normalize_text(pred),
                  "evaluation_note": f"Unexpected ground_truth_type={answer_type!r}."}
    row = dict(record)
    row.update({"evaluated_answer": pred, "answer_extraction_method": extraction_method, **result})
    return row


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError("Prediction JSON must contain a list of records.")
    return [dict(x) for x in obj if isinstance(x, Mapping)]


def load_qa_records(path: str | Path) -> list[dict[str, Any]]:
    obj = load_json(path)
    if isinstance(obj, dict):
        for key in ("data", "questions", "results", "items"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise ValueError("QA JSON must contain a list of records.")
    rows = [dict(x) for x in obj if isinstance(x, Mapping)]
    if any("qa_id" not in row for row in rows):
        raise ValueError("Every QA record must contain qa_id.")
    return rows


def merge_predictions_with_qa(predictions: list[dict[str, Any]], qa_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_map = {str(x.get("qa_id")): x for x in predictions if x.get("qa_id") is not None}
    merged = []
    for qa in qa_records:
        qa_id = str(qa["qa_id"])
        combined = dict(qa)
        combined.update(pred_map.get(qa_id, {}))
        combined["qa_id"] = qa_id
        merged.append(combined)
    return merged


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(cleaned) / len(cleaned) if cleaned else None


def summarize_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(int(x.get("strict_exact_match", 0)) for x in rows)
    out = {"total": total, "exact_correct": exact,
           "exact_match_accuracy": exact / total if total else None,
           "mean_primary_score": safe_mean(x.get("primary_score") for x in rows)}
    for output_key, row_key in {
        "mean_precision": "precision", "mean_recall": "recall", "mean_f1": "f1",
        "mean_rankwise_precision": "rankwise_precision", "mean_mrr": "mrr", "mean_map": "map",
    }.items():
        vals = [x.get(row_key) for x in rows if x.get(row_key) is not None]
        out[output_key] = safe_mean(vals) if vals else None
    return out


def summarize_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {"overall": summarize_group(rows)}
    dims = {
        "by_ground_truth_type": "ground_truth_type", "by_template_no": "template_no",
        "by_legend_style": "legend_style", "by_relative_region": "relative_region",
        "by_map_no": "map_no", "by_generation_status": "generation_status", "by_metric": "metric",
    }
    for output_key, field in dims.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "Unknown"))].append(row)
        summary[output_key] = {k: summarize_group(v) for k, v in sorted(groups.items())}
    extractable = sum(bool(str(x.get("evaluated_answer", "")).strip()) for x in rows)
    complete = sum(str(x.get("generation_status", "")).casefold() == "complete" for x in rows)
    summary["generation"] = {
        "total": len(rows), "extractable_answers": extractable,
        "answer_extraction_rate": extractable / len(rows) if rows else None,
        "complete_generations": complete, "completion_rate": complete / len(rows) if rows else None,
        "average_generated_tokens": safe_mean(x.get("generated_tokens") for x in rows),
        "average_inference_seconds": safe_mean(x.get("inference_seconds") for x in rows),
    }
    summary["total"] = summary["overall"]["total"]
    summary["overall_exact_match_accuracy"] = summary["overall"]["exact_match_accuracy"]
    summary["overall_mean_primary_score"] = summary["overall"]["mean_primary_score"]
    return summary


def save_results(rows: list[Mapping[str, Any]], summary: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    details_json, summary_json, details_csv = output / "evaluation_details.json", output / "evaluation_summary.json", output / "evaluation_details.csv"
    details_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["qa_id", "sample_index", "country", "map_no", "template_no", "question",
              "ground_truth", "ground_truth_type", "legend_style", "relative_region",
              "generation_status", "generated_tokens", "inference_seconds", "final_answer",
              "evaluated_answer", "answer_extraction_method", "primary_score", "strict_exact_match",
              "metric", "precision", "recall", "f1", "rankwise_precision", "mrr", "map",
              "range_relation", "normalized_ground_truth", "normalized_prediction", "evaluation_note"]
    with details_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for key in ("normalized_ground_truth", "normalized_prediction", "rank_precisions", "rank_hits"):
                if isinstance(item.get(key), (list, dict)):
                    item[key] = json.dumps(item[key], ensure_ascii=False)
            writer.writerow(item)
    return {"details_json": details_json, "summary_json": summary_json, "details_csv": details_csv}


def evaluate_mapwise(prediction_json: str | Path, qa_json: Optional[str | Path] = None,
                     *, output_dir: Optional[str | Path] = None) -> dict[str, Any]:
    prediction_path = Path(prediction_json).resolve()
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction JSON not found: {prediction_path}")
    predictions = load_prediction_records(prediction_path)
    records = merge_predictions_with_qa(predictions, load_qa_records(qa_json)) if qa_json is not None else predictions
    required = {"qa_id", "ground_truth", "ground_truth_type", "template_no"}
    for i, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Record {i} is missing required fields: {sorted(missing)}")
    rows = [evaluate_sample(record) for record in records]
    summary = summarize_results(rows)
    destination = Path(output_dir).resolve() if output_dir is not None else prediction_path.parent
    paths = save_results(rows, summary, destination)
    summary["artifacts"] = {k: str(v.resolve()) for k, v in paths.items()}
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def format_metric(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f} ({value:.2%})"


def print_summary(summary: Mapping[str, Any]) -> None:
    print("=" * 76)
    print("MAPWISE EVALUATION")
    print("=" * 76)
    print(f"Questions:                    {summary.get('total', 0)}")
    print(f"Overall exact-match accuracy: {format_metric(summary.get('overall_exact_match_accuracy'))}")
    print(f"Overall mean primary score:    {format_metric(summary.get('overall_mean_primary_score'))}")
    for answer_type in ("Binary", "Count", "Single", "List", "Range"):
        group = next((v for k, v in summary.get("by_ground_truth_type", {}).items() if k.casefold() == answer_type.casefold()), None)
        if group:
            print(f"{answer_type:<12} primary score:       {format_metric(group.get('mean_primary_score'))}")
    rank = summary.get("by_template_no", {}).get("43")
    if rank:
        print(f"Rank RWP:                    {format_metric(rank.get('mean_rankwise_precision'))}")
        print(f"Rank MRR:                    {format_metric(rank.get('mean_mrr'))}")
        print(f"Rank MAP:                    {format_metric(rank.get('mean_map'))}")
    generation = summary.get("generation", {})
    print(f"Answer extraction rate:       {format_metric(generation.get('answer_extraction_rate'))}")
    print(f"Generation completion rate:   {format_metric(generation.get('completion_rate'))}")
    if generation.get("average_generated_tokens") is not None:
        print(f"Average generated tokens:     {generation['average_generated_tokens']:.2f}")
    if generation.get("average_inference_seconds") is not None:
        print(f"Average inference seconds:    {generation['average_inference_seconds']:.2f}")
    artifacts = summary.get("artifacts", {})
    if artifacts:
        print(f"Summary JSON:                 {artifacts.get('summary_json', '')}")
        print(f"Details JSON:                 {artifacts.get('details_json', '')}")
        print(f"Details CSV:                  {artifacts.get('details_csv', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MapWise prediction JSON.")
    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_summary(evaluate_mapwise(args.predictions_json, args.qa_json, output_dir=args.output_dir))


if __name__ == "__main__":
    main()