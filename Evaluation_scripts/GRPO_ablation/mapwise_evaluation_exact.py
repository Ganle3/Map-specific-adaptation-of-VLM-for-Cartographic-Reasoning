# python3
# -*- coding: utf-8 -*-
"""
Strict-exact MapWise evaluation for GRPO validation predictions.

Key design decisions:
- The only task score is strict exact match, matching the GRPO reward.
- Administrative-unit abbreviations are canonicalized with COUNTRY-AWARE
  dictionaries, because codes such as AR, GA, OR, TN, UT, MI, MN, and HI
  are ambiguous across countries.
- List/ranking questions are expected to be absent.
- Single questions use strict exact canonical-set equality, not recall.
"""

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


# ============================================================
# 1. Basic text normalization
# ============================================================

def _ascii_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("&", " and ")
    )


def normalize_text(value: Any) -> str:
    text = _ascii_text(value).casefold().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _ascii_text(value).casefold())


def normalize_country(value: Any) -> str:
    text = normalize_text(value)
    if text in {"us", "u s", "usa", "u s a", "united states",
                "united states of america"}:
        return "usa"
    if text in {"china", "prc", "p r c", "people s republic of china"}:
        return "china"
    if text in {"india"}:
        return "india"
    return text


# ============================================================
# 2. Final-answer extraction
# ============================================================

def extract_final_answer(raw_response: Any) -> str:
    if raw_response is None:
        return ""

    text = str(raw_response).strip()
    if not text:
        return ""

    matches = list(
        re.finditer(r"final\s+answer\s*:\s*", text, flags=re.I)
    )

    if matches:
        answer = text[matches[-1].end():].strip()
    elif "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        answer = text[idx + len("</think>"):].strip()
    else:
        # Fallback keeps a direct answer from an Instruct model evaluable even
        # if it omitted the requested marker.
        answer = text

    answer = re.split(
        r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)",
        answer,
        maxsplit=1,
        flags=re.I,
    )[0].strip()

    if (
        len(answer) >= 2
        and answer[0] == answer[-1]
        and answer[0] in {'"', "'"}
    ):
        answer = answer[1:-1].strip()

    return answer


# Maximum length accepted for a final-answer field.
# If an inference pipeline accidentally copies the whole reasoning/raw response
# into final_answer, reject it instead of letting downstream parsers extract
# an incidental number or entity from the reasoning text.
MAX_FINAL_ANSWER_CHARS = 200


def resolve_prediction_answer(
    record: Mapping[str, Any],
) -> tuple[str, str]:
    saved = str(record.get("final_answer", "") or "").strip()
    raw = str(record.get("raw_response", "") or "").strip()

    if saved:
        # Guard against inference failures where final_answer is actually the
        # entire raw response / reasoning trace.
        if len(saved) > MAX_FINAL_ANSWER_CHARS:
            return "", "saved_final_answer_too_long"

        if raw and saved == raw:
            return "", "saved_final_answer_equals_raw_response"

        return saved, "saved_final_answer"

    extracted = extract_final_answer(raw)
    if extracted:
        # Apply the same sanity check to answers re-extracted from raw output.
        if len(extracted) > MAX_FINAL_ANSWER_CHARS:
            return "", "raw_response_extracted_too_long"

        return extracted, "raw_response_reextract"

    return "", "not_extractable"


# ============================================================
# 3. Country-aware place aliases
# ============================================================

# ----------------------------
# 3.1 USA: USPS abbreviations
# ----------------------------

USA_PLACE_ALIASES: dict[str, set[str]] = {
    "alabama": {"alabama", "al"},
    "alaska": {"alaska", "ak"},
    "arizona": {"arizona", "az", "arizaon"},  # observed dataset typo
    "arkansas": {"arkansas", "ar"},
    "california": {"california", "ca"},
    "colorado": {"colorado", "co"},
    "connecticut": {"connecticut", "ct"},
    "delaware": {"delaware", "de"},
    "florida": {"florida", "fl"},
    "georgia": {"georgia", "ga"},
    "hawaii": {"hawaii", "hi"},
    "idaho": {"idaho", "id"},
    "illinois": {"illinois", "il"},
    "indiana": {"indiana", "in"},
    "iowa": {"iowa", "ia"},
    "kansas": {"kansas", "ks"},
    "kentucky": {"kentucky", "ky"},
    "louisiana": {"louisiana", "la"},
    "maine": {"maine", "me"},
    "maryland": {"maryland", "md"},
    "massachusetts": {"massachusetts", "ma"},
    "michigan": {"michigan", "mi"},
    "minnesota": {"minnesota", "mn"},
    "mississippi": {"mississippi", "ms"},
    "missouri": {"missouri", "mo"},
    "montana": {"montana", "mt"},
    "nebraska": {"nebraska", "ne"},
    "nevada": {"nevada", "nv"},
    "new hampshire": {"new hampshire", "nh"},
    "new jersey": {"new jersey", "nj"},
    "new mexico": {"new mexico", "nm"},
    "new york": {"new york", "ny"},
    "north carolina": {"north carolina", "nc"},
    "north dakota": {"north dakota", "nd"},
    "ohio": {"ohio", "oh"},
    "oklahoma": {"oklahoma", "ok"},
    "oregon": {"oregon", "or"},
    "pennsylvania": {"pennsylvania", "pa"},
    "rhode island": {"rhode island", "ri"},
    "south carolina": {"south carolina", "sc"},
    "south dakota": {"south dakota", "sd"},
    "tennessee": {"tennessee", "tn"},
    "texas": {"texas", "tx"},
    "utah": {"utah", "ut"},
    "vermont": {"vermont", "vt"},
    "virginia": {"virginia", "va"},
    "washington": {"washington", "wa", "washington state"},
    "west virginia": {"west virginia", "wv"},
    "wisconsin": {"wisconsin", "wi"},
    "wyoming": {"wyoming", "wy"},
}


# ---------------------------------------------
# 3.2 China: abbreviations visible in MapWise
# ---------------------------------------------

CHINA_PLACE_ALIASES: dict[str, set[str]] = {
    "anhui": {"anhui", "ah"},
    "beijing": {"beijing", "bj"},
    "chongqing": {"chongqing", "cq"},
    "fujian": {"fujian", "fj"},
    "gansu": {"gansu", "gs"},
    "guangdong": {"guangdong", "gd"},
    "guangxi": {"guangxi", "gx", "guangxi zhuang autonomous region"},
    "guizhou": {"guizhou", "gz"},
    "hainan": {"hainan", "hi"},
    "hebei": {"hebei", "hb"},
    "heilongjiang": {"heilongjiang", "hl"},
    "henan": {"henan", "ha"},
    "hubei": {"hubei", "hub"},
    "hunan": {"hunan", "hn"},
    "inner mongolia": {
        "inner mongolia",
        "nm",
        "nei mongol",
        "inner mongolia autonomous region",
    },
    "jiangsu": {"jiangsu", "js"},
    "jiangxi": {"jiangxi", "jx"},
    "jilin": {"jilin", "jl"},
    "liaoning": {"liaoning", "ln"},
    "ningxia": {
        "ningxia",
        "nx",
        "ningxia hui autonomous region",
    },
    "qinghai": {"qinghai", "qh"},
    "shaanxi": {"shaanxi", "sn"},
    "shandong": {"shandong", "sd"},
    "shanghai": {"shanghai", "sh"},
    "shanxi": {"shanxi", "sx"},
    "sichuan": {"sichuan", "sc"},
    "tianjin": {"tianjin", "tj"},
    "tibet": {
        "tibet",
        "xz",
        "xizang",
        "tibet autonomous region",
        "xizang autonomous region",
    },
    "xinjiang": {
        "xinjiang",
        "xj",
        "xinjiang uygur autonomous region",
        "xinjiang uyghur autonomous region",
    },
    "yunnan": {"yunnan", "yn"},
    "zhejiang": {"zhejiang", "zj"},

    # Included because these labels can appear on the supplied China maps.
    "hong kong": {"hong kong", "hk"},
    "macau": {"macau", "macao", "mo"},
}


# ---------------------------------------------
# 3.3 India: abbreviations visible in MapWise
# ---------------------------------------------

INDIA_PLACE_ALIASES: dict[str, set[str]] = {
    "andhra pradesh": {"andhra pradesh", "ap"},
    "arunachal pradesh": {"arunachal pradesh", "ar"},
    "assam": {"assam", "as"},
    "bihar": {"bihar", "bh", "br"},
    "chhattisgarh": {
        "chhattisgarh",
        "chattisgarh",
        "ct",
        "cg",
    },
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
    "meghalaya": {"meghalaya", "me", "ml"},
    "mizoram": {"mizoram", "mi", "mz"},
    "nagaland": {"nagaland", "nl"},
    "odisha": {"odisha", "orissa", "or", "od"},
    "punjab": {"punjab", "pb"},
    "rajasthan": {
        "rajasthan",
        "rj",
        "rajasathan",
        "rajastan",
    },
    "sikkim": {"sikkim", "sk"},
    "tamil nadu": {"tamil nadu", "tamilnadu", "tn"},
    "telangana": {"telangana", "ts", "tg"},
    "tripura": {"tripura", "tr"},
    "uttar pradesh": {"uttar pradesh", "up"},
    "uttarakhand": {
        "uttarakhand",
        "uttaranchal",
        "ut",
        "uk",
    },
    "west bengal": {"west bengal", "westbengal", "wb"},

    "andaman and nicobar islands": {
        "andaman and nicobar islands",
        "andaman nicobar islands",
        "andaman and nicobar",
        "a and n islands",
        "an",
    },
    "chandigarh": {"chandigarh", "ch"},
    "dadra and nagar haveli": {
        "dadra and nagar haveli",
        "dadra nagar haveli",
        "dn",
    },
    "daman and diu": {
        "daman and diu",
        "daman diu",
        "dd",
    },
    "dadra and nagar haveli and daman and diu": {
        "dadra and nagar haveli and daman and diu",
        "dadra nagar haveli daman diu",
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
    "puducherry": {
        "puducherry",
        "pondicherry",
        "pd",
        "py",
    },
}


COUNTRY_PLACE_ALIASES = {
    "usa": USA_PLACE_ALIASES,
    "china": CHINA_PLACE_ALIASES,
    "india": INDIA_PLACE_ALIASES,
}


def _build_place_alias_lookup(
    aliases_by_country: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, dict[str, str]]:
    """
    Build one lookup PER COUNTRY.

    This is essential because abbreviations collide:
      AR = Arkansas / Arunachal Pradesh
      GA = Georgia / Goa
      OR = Oregon / Odisha
      TN = Tennessee / Tamil Nadu
      UT = Utah / Uttarakhand
      MI = Michigan / Mizoram
      MN = Minnesota / Manipur
      HI = Hawaii / Hainan
    """
    all_lookups: dict[str, dict[str, str]] = {}

    for country, alias_table in aliases_by_country.items():
        lookup: dict[str, str] = {}

        for canonical_name, aliases in alias_table.items():
            normalized_canonical = normalize_text(canonical_name)

            for alias in aliases | {canonical_name}:
                normalized_alias = normalize_text(alias)
                existing = lookup.get(normalized_alias)

                if (
                    existing is not None
                    and existing != normalized_canonical
                ):
                    raise ValueError(
                        f"Ambiguous alias within country={country!r}: "
                        f"{alias!r} -> {existing!r} and "
                        f"{normalized_canonical!r}"
                    )

                lookup[normalized_alias] = normalized_canonical

        all_lookups[country] = lookup

    return all_lookups


PLACE_ALIAS_LOOKUPS = _build_place_alias_lookup(COUNTRY_PLACE_ALIASES)


def canonicalize_item(value: Any, country: Any) -> str:
    """
    Canonicalize an administrative-unit answer using the record's country.

    Also removes trailing parenthetical annotations such as:
        Arunachal Pradesh (AR)
        California (CA)
        Jilin (14,384.0 - 19,505.5)

    Unknown text is kept after normalization instead of being discarded.
    """
    raw_text = str(value or "").strip()

    # Remove trailing parenthetical annotations before normalization.
    # Examples:
    #   "Arunachal Pradesh (AR)" -> "Arunachal Pradesh"
    #   "Jilin (14,384.0 - 19,505.5)" -> "Jilin"
    raw_text = re.sub(
        r"\s*\([^()]*\)\s*$",
        "",
        raw_text,
    ).strip()

    text = normalize_text(raw_text)
    country_key = normalize_country(country)

    text = re.sub(
        r"^(?:the\s+)?"
        r"(?:(?:state|states|province|provinces|"
        r"union territory|union territories|"
        r"territory|territories)\s+(?:of\s+)?)",
        "",
        text,
    ).strip()

    text = re.sub(
        r"\s+(?:state|states|province|provinces|"
        r"union territory|union territories|"
        r"territory|territories)$",
        "",
        text,
    ).strip()

    lookup = PLACE_ALIAS_LOOKUPS.get(country_key, {})
    return lookup.get(text, text)


def _protect_internal_place_conjunctions(text: str) -> str:
    """Protect official names containing 'and' before list splitting."""
    replacements = {
        r"andaman\s+(?:and|&)\s+nicobar(?:\s+islands)?":
            "andaman__place_and__nicobar islands",
        r"jammu\s+(?:and|&)\s+kashmir":
            "jammu__place_and__kashmir",
        r"dadra\s+(?:and|&)\s+nagar\s+haveli\s+(?:and|&)\s+"
        r"daman\s+(?:and|&)\s+diu":
            "dadra__place_and__nagar haveli"
            "__place_and__daman__place_and__diu",
        r"dadra\s+(?:and|&)\s+nagar\s+haveli":
            "dadra__place_and__nagar haveli",
        r"daman\s+(?:and|&)\s+diu":
            "daman__place_and__diu",
        r"a\s+(?:and|&)\s+n\s+islands":
            "a__place_and__n islands",
    }

    protected = text
    for pattern, replacement in replacements.items():
        protected = re.sub(
            pattern,
            replacement,
            protected,
            flags=re.I,
        )

    return protected


def split_answer_items(value: Any, country: Any) -> list[str]:
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
        canonical = canonicalize_item(restored, country)
        if canonical:
            output.append(canonical)

    return output


# ============================================================
# 4. Binary / count / range parsing
# ============================================================

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
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

NUMBER_WORD_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def parse_number_words(text: str) -> Optional[int]:
    tokens = normalize_text(text).replace("-", " ").split()

    for i, token in enumerate(tokens):
        if token in NUMBER_WORD_UNITS:
            return NUMBER_WORD_UNITS[token]

        if token in NUMBER_WORD_TENS:
            value = NUMBER_WORD_TENS[token]
            if (
                i + 1 < len(tokens)
                and tokens[i + 1] in NUMBER_WORD_UNITS
            ):
                value += NUMBER_WORD_UNITS[tokens[i + 1]]
            return value

    return None


def parse_count(value: Any) -> Optional[int]:
    text_norm = normalize_text(value)

    # The supplied MapWise QA includes Count answers represented as "None".
    if text_norm in {
        "none",
        "no",
        "no state",
        "no states",
        "zero",
    }:
        return 0

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


def normalize_range_surface(value: Any) -> str:
    """
    Normalize a range/category expression for exact surface-semantic matching.

    Examples:
      '8.2% - 10.4%' -> '8.2%-10.4%'
      '> 10 M'       -> '>10m'
      '1,000 - 2,000'-> '1000-2000'
    """
    text = _ascii_text(value).casefold().strip()
    text = text.replace(",", "")
    text = re.sub(r"\bto\b", "-", text)
    text = re.sub(r"\s+", "", text)
    return text


def parse_bounded_range(
    value: Any,
) -> Optional[tuple[float, float]]:
    """
    Parse only a standard two-ended numeric interval.

    Open-ended categories such as >50, <50, or a single 0% are handled by
    normalized exact matching in evaluate_range().
    """
    text = _ascii_text(value)

    match = re.fullmatch(
        rf"\s*({NUMBER_PATTERN})\s*-\s*({NUMBER_PATTERN})\s*%?\s*",
        text,
        flags=re.I,
    )

    if not match:
        match = re.fullmatch(
            rf"\s*({NUMBER_PATTERN})\s+to\s+({NUMBER_PATTERN})\s*%?\s*",
            text,
            flags=re.I,
        )

    if not match:
        return None

    try:
        first = parse_scaled_number(match.group(1))
        second = parse_scaled_number(match.group(2))
    except ValueError:
        return None

    return min(first, second), max(first, second)


def numbers_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def ranges_exact(
    gold: tuple[float, float],
    pred: tuple[float, float],
) -> bool:
    return (
        numbers_close(gold[0], pred[0])
        and numbers_close(gold[1], pred[1])
    )


def ranges_strictly_overlap(
    gold: tuple[float, float],
    pred: tuple[float, float],
) -> bool:
    return max(gold[0], pred[0]) < min(gold[1], pred[1])


# ============================================================


# ============================================================
# 5. Strict-exact evaluation only
# ============================================================

SUPPORTED_ANSWER_TYPES = {"binary", "count", "range", "single"}


def evaluate_binary_exact(gold: str, pred: str) -> dict[str, Any]:
    g = parse_binary(gold)
    p = parse_binary(pred)
    exact = g is not None and p is not None and g == p
    return {
        "primary_score": float(exact),
        "strict_exact_match": int(exact),
        "metric": "binary_exact_match",
        "normalized_ground_truth": g or "",
        "normalized_prediction": p or "",
        "evaluation_note": "",
    }


def evaluate_count_exact(gold: str, pred: str) -> dict[str, Any]:
    g = parse_count(gold)
    p = parse_count(pred)
    exact = g is not None and p is not None and g == p
    return {
        "primary_score": float(exact),
        "strict_exact_match": int(exact),
        "metric": "count_exact_match",
        "normalized_ground_truth": g,
        "normalized_prediction": p,
        "evaluation_note": "",
    }


def evaluate_range_exact(gold: str, pred: str) -> dict[str, Any]:
    g = normalize_range_surface(gold)
    p = normalize_range_surface(pred)
    exact = bool(g) and g == p
    return {
        "primary_score": float(exact),
        "strict_exact_match": int(exact),
        "metric": "range_exact_match",
        "normalized_ground_truth": g,
        "normalized_prediction": p,
        "evaluation_note": "",
    }


def evaluate_single_exact(
    gold: str,
    pred: str,
    country: Any,
) -> dict[str, Any]:
    """Single questions are strict exact after canonical surface normalization."""
    g = set(split_answer_items(gold, country))
    p = set(split_answer_items(pred, country))
    exact = bool(g) and g == p
    return {
        "primary_score": float(exact),
        "strict_exact_match": int(exact),
        "metric": "single_exact_match",
        "normalized_ground_truth": sorted(g),
        "normalized_prediction": sorted(p),
        "evaluation_note": "",
    }


def evaluate_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    """
    Reward-compatible strict exact evaluator.

    train_mapwise_grpo_visLoRA_trl.py can call:
        result = evaluate_sample(record)
        reward = float(result["strict_exact_match"])

    Validation uses the same exact criterion.
    """
    pred, extraction_method = resolve_prediction_answer(record)
    gold = str(record.get("ground_truth", "") or "").strip()
    answer_type = str(record.get("ground_truth_type", "")).casefold().strip()
    country = normalize_country(record.get("country", ""))
    template_no = int(record.get("template_no", -1))

    if template_no == 43 or answer_type in {"list", "rank", "ranking"}:
        raise ValueError(
            "List/ranking questions are not supported by this GRPO exact evaluator. "
            "They should already have been removed from the dataset. "
            f"qa_id={record.get('qa_id', '')}, type={answer_type!r}, "
            f"template_no={template_no}"
        )

    if answer_type not in SUPPORTED_ANSWER_TYPES:
        raise ValueError(
            "Unsupported ground_truth_type for strict GRPO validation: "
            f"{answer_type!r}. Expected Binary / Count / Range / Single."
        )

    if not pred:
        result = {
            "primary_score": 0.0,
            "strict_exact_match": 0,
            "metric": "not_extractable",
            "normalized_ground_truth": "",
            "normalized_prediction": "",
            "evaluation_note": "No extractable final answer.",
        }
    elif answer_type == "binary":
        result = evaluate_binary_exact(gold, pred)
    elif answer_type == "count":
        result = evaluate_count_exact(gold, pred)
    elif answer_type == "range":
        result = evaluate_range_exact(gold, pred)
    else:
        result = evaluate_single_exact(gold, pred, country)

    row = dict(record)
    row.update({
        "evaluated_answer": pred,
        "answer_extraction_method": extraction_method,
        **result,
    })
    return row


# ============================================================
# 6. Prediction loading
# ============================================================

def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prediction JSON not found:\n{path}")

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    if isinstance(obj, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError("Prediction JSON must contain a list of records.")

    return [dict(x) for x in obj if isinstance(x, Mapping)]


# ============================================================
# 7. Exact-accuracy summaries
# ============================================================

def summarize_group(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row.get("strict_exact_match", 0)) for row in rows)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
    }


def summarize_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    overall = summarize_group(rows)
    summary: dict[str, Any] = {
        "overall": overall,
        "total": overall["total"],
        "correct": overall["correct"],
        "validation_accuracy": overall["accuracy"],
    }

    for output_key, field in {
        "by_ground_truth_type": "ground_truth_type",
        "by_country": "country",
    }.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "Unknown"))].append(row)
        summary[output_key] = {
            key: summarize_group(group_rows)
            for key, group_rows in sorted(groups.items())
        }

    extractable = sum(
        bool(str(row.get("evaluated_answer", "")).strip())
        for row in rows
    )
    complete = sum(
        str(row.get("generation_status", "")).casefold() == "complete"
        for row in rows
    )
    n = len(rows)
    summary["generation"] = {
        "extractable_answers": extractable,
        "answer_extraction_rate": extractable / n if n else None,
        "complete_generations": complete,
        "completion_rate": complete / n if n else None,
    }
    return summary


# ============================================================
# 8. Save results
# ============================================================

def save_results(
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    details_json = output / "evaluation_details.json"
    details_csv = output / "evaluation_details.csv"
    summary_json = output / "evaluation_summary.json"

    details_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = [
        "qa_id",
        "sample_index",
        "country",
        "map_no",
        "template_no",
        "question",
        "ground_truth",
        "ground_truth_type",
        "generation_status",
        "generated_tokens",
        "final_answer",
        "evaluated_answer",
        "answer_extraction_method",
        "strict_exact_match",
        "normalized_ground_truth",
        "normalized_prediction",
        "inference_seconds",
        "adapter_name",
        "adapter_path",
        "evaluation_note",
    ]

    with details_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for key in ("normalized_ground_truth", "normalized_prediction"):
                if isinstance(item.get(key), (list, dict)):
                    item[key] = json.dumps(item[key], ensure_ascii=False)
            writer.writerow(item)

    return {
        "details_json": details_json,
        "details_csv": details_csv,
        "summary_json": summary_json,
    }


# ============================================================
# 9. Public evaluation API
# ============================================================

def evaluate_mapwise(
    prediction_json: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    prediction_path = Path(prediction_json).expanduser().resolve()
    predictions = load_prediction_records(prediction_path)

    required = {"qa_id", "country", "ground_truth", "ground_truth_type"}
    for index, record in enumerate(predictions):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"Prediction record {index} is missing: {sorted(missing)}"
            )

    rows = [evaluate_sample(record) for record in predictions]
    summary = summarize_results(rows)
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else prediction_path.parent
    )
    paths = save_results(rows, summary, destination)
    summary["artifacts"] = {key: str(path) for key, path in paths.items()}
    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


# ============================================================
# 10. Console / CLI
# ============================================================

def format_accuracy(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f} ({value:.2%})"


def print_summary(summary: Mapping[str, Any]) -> None:
    print("=" * 76)
    print("MAPWISE STRICT-EXACT VALIDATION")
    print("=" * 76)
    print(f"Questions:            {summary.get('total', 0)}")
    print(f"Correct:              {summary.get('correct', 0)}")
    print(
        "Validation accuracy:  "
        f"{format_accuracy(summary.get('validation_accuracy'))}"
    )

    print("\nBy answer type:")
    for name, group in summary.get("by_ground_truth_type", {}).items():
        print(
            f"  {name:<10} {group['correct']:>4}/{group['total']:<4} "
            f"{format_accuracy(group['accuracy'])}"
        )

    print("\nBy country:")
    for name, group in summary.get("by_country", {}).items():
        print(
            f"  {name:<10} {group['correct']:>4}/{group['total']:<4} "
            f"{format_accuracy(group['accuracy'])}"
        )

    generation = summary.get("generation", {})
    print(
        "\nAnswer extraction rate: "
        f"{format_accuracy(generation.get('answer_extraction_rate'))}"
    )
    print(
        "Generation completion rate: "
        f"{format_accuracy(generation.get('completion_rate'))}"
    )

    artifacts = summary.get("artifacts", {})
    if artifacts:
        print(f"Summary JSON:         {artifacts.get('summary_json', '')}")
        print(f"Details JSON:         {artifacts.get('details_json', '')}")
        print(f"Details CSV:          {artifacts.get('details_csv', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MapWise GRPO validation predictions using strict "
            "exact-match accuracy only."
        )
    )
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_mapwise(
        prediction_json=args.predictions_json,
        output_dir=args.output_dir,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
