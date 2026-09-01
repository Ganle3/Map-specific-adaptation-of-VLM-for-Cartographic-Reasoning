# python3
# -*- coding: utf-8 -*-
"""
Deterministic MapWise evaluation for mixed China / India / USA predictions.

Key design decisions:
- ground_truth_type is used only for evaluation, never for inference.
- Administrative-unit abbreviations are canonicalized with COUNTRY-AWARE
  dictionaries, because codes such as AR, GA, OR, TN, UT, MI, MN, and HI
  are ambiguous across countries.
- Alias normalization removes surface-form / abbreviation differences without
  performing task reasoning for the model.
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
# 5. Ranking parsing
# ============================================================

RANK_OPERATOR_PATTERN = re.compile(r"\s*(<=|>=|<|>|=)\s*")


def normalize_ranking_expression(value: Any) -> tuple[str, bool]:
    """
    Conservatively normalize an explicitly stated ranking expression.

    This function only normalizes surface form. It does NOT infer an ordering
    from comma-separated names, numeric legend ranges, or free-form reasoning.

    Examples accepted:
        A < B = C
        A is lower than B
        A is higher than B
        A is tied with B

    Returns:
        (normalized_text, canonical_symbol_format)

    canonical_symbol_format is True only when the original final answer already
    used an explicit ranking operator (<, >, =, <=, >=). This is useful for
    separately tracking output-format compliance.
    """
    text = str(value or "").strip()
    if not text:
        return "", False

    # Normalize Unicode/full-width comparison symbols.
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("≤", "<=")
        .replace("≥", ">=")
    )

    # Remove harmless leading labels.
    text = re.sub(
        r"^\s*(?:final\s+answer|answer|ranking|rank|order)\s*:\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    canonical_symbol_format = bool(
        re.search(r"(?:<=|>=|<|>|=)", text)
    )

    # Conservative normalization of explicit relational prose.
    # These replacements preserve stated semantics; they do not derive new
    # relations from values or list order.
    text = re.sub(
        r"\s+(?:is\s+)?(?:tied\s+with|equal\s+to|equals?)\s+",
        " = ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s+(?:is\s+)?(?:lower\s+than|less\s+than|below)\s+",
        " < ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s+(?:is\s+)?(?:higher\s+than|greater\s+than|above)\s+",
        " > ",
        text,
        flags=re.I,
    )

    return text.strip(), canonical_symbol_format


def parse_ranking_with_metadata(
    value: Any,
    country: Any,
) -> tuple[Optional[list[set[str]]], bool]:
    """
    Parse a ranking into ordered rank groups.

    The internal representation is always normalized to ascending rank order.
    For example:
        A < B = C  -> [{A}, {B, C}]
        C > B = A  -> [{A, B}, {C}]

    A plain comma-separated list such as "A, B, C" is intentionally NOT
    parsed because its ranking direction and ties are ambiguous.
    """
    text, canonical_symbol_format = normalize_ranking_expression(value)

    if not text:
        return None, canonical_symbol_format

    parts = RANK_OPERATOR_PATTERN.split(text)

    if len(parts) < 3 or len(parts) % 2 == 0:
        return None, canonical_symbol_format

    items = [
        canonicalize_item(parts[i], country)
        for i in range(0, len(parts), 2)
    ]
    operators = [
        parts[i]
        for i in range(1, len(parts), 2)
    ]

    if any(not item for item in items):
        return None, canonical_symbol_format

    directional = {op for op in operators if op != "="}

    if directional.issubset({"<", "<="}):
        descending = False
    elif directional.issubset({">", ">="}):
        descending = True
    elif not directional:
        descending = False
    else:
        # Mixed < and > directions are internally inconsistent.
        return None, canonical_symbol_format

    groups: list[set[str]] = [{items[0]}]

    for op, item in zip(operators, items[1:]):
        if op == "=":
            groups[-1].add(item)
        else:
            groups.append({item})

    if descending:
        groups.reverse()

    return groups, canonical_symbol_format


def parse_ranking(
    value: Any,
    country: Any,
) -> Optional[list[set[str]]]:
    """Backward-compatible ranking parser."""
    groups, _ = parse_ranking_with_metadata(value, country)
    return groups


def exact_rank_match(
    gold: list[set[str]],
    pred: list[set[str]],
) -> bool:
    return (
        len(gold) == len(pred)
        and all(g == p for g, p in zip(gold, pred))
    )


def rankwise_precision(
    gold: list[set[str]],
    pred: list[set[str]],
) -> tuple[float, list[float]]:
    """
    MAPWise Rank-wise Precision (RWP).

    Average precision over the GROUND-TRUTH ranks, following the official
    MAPWise definition. Missing predicted ranks receive 0 precision.
    Extra predicted ranks beyond the number of ground-truth ranks do not add
    extra denominator terms.
    """
    n = len(gold)

    if n == 0:
        return 0.0, []

    scores = []

    for i in range(n):
        if i >= len(pred) or not pred[i]:
            scores.append(0.0)
        else:
            scores.append(
                len(gold[i] & pred[i]) / len(pred[i])
            )

    return sum(scores) / n, scores


def rankwise_mrr_and_map(
    gold: list[set[str]],
    pred: list[set[str]],
) -> tuple[float, float, list[int]]:
    n = max(len(gold), len(pred))

    if n == 0:
        return 0.0, 0.0, []

    hits = [
        int(
            i < len(gold)
            and i < len(pred)
            and bool(gold[i] & pred[i])
        )
        for i in range(n)
    ]

    first = next(
        (i for i, hit in enumerate(hits) if hit),
        None,
    )

    mrr = 0.0 if first is None else 1.0 / (first + 1)

    cumulative = 0
    precisions = []

    for i, hit in enumerate(hits, start=1):
        if hit:
            cumulative += 1
            precisions.append(cumulative / i)

    map_score = sum(precisions) / max(len(gold), 1)
    return mrr, map_score, hits


# ============================================================
# 6. Per-answer-type evaluation
# ============================================================

def evaluate_binary(gold: str, pred: str) -> dict[str, Any]:
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


def evaluate_count(gold: str, pred: str) -> dict[str, Any]:
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


def evaluate_single(
    gold: str,
    pred: str,
    country: Any,
) -> dict[str, Any]:
    g = set(split_answer_items(gold, country))
    p = set(split_answer_items(pred, country))

    inter = g & p
    precision = len(inter) / len(p) if p else 0.0
    recall = len(inter) / len(g) if g else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "primary_score": recall,
        "strict_exact_match": int(bool(g) and g == p),
        "metric": "single_recall",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normalized_ground_truth": sorted(g),
        "normalized_prediction": sorted(p),
        "evaluation_note": "",
    }


def evaluate_list(
    gold: str,
    pred: str,
    country: Any,
) -> dict[str, Any]:
    g = set(split_answer_items(gold, country))
    p = set(split_answer_items(pred, country))

    inter = g & p
    precision = len(inter) / len(p) if p else 0.0
    recall = len(inter) / len(g) if g else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "primary_score": f1,
        "strict_exact_match": int(bool(g) and g == p),
        "metric": "list_f1",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normalized_ground_truth": sorted(g),
        "normalized_prediction": sorted(p),
        "evaluation_note": "",
    }


def evaluate_range(
    gold: str,
    pred: str,
    legend_style: str,
) -> dict[str, Any]:
    """
    Range scoring:
    1. First perform normalized exact matching. This supports:
       0%, >50, <50, >10M, and other legend categories.
    2. If both answers are ordinary bounded numeric intervals, also allow the
       previous continuous-range overlap score for style C.
    """
    gold_surface = normalize_range_surface(gold)
    pred_surface = normalize_range_surface(pred)

    if gold_surface and gold_surface == pred_surface:
        return {
            "primary_score": 1.0,
            "strict_exact_match": 1,
            "metric": "range_exact_match",
            "normalized_ground_truth": gold_surface,
            "normalized_prediction": pred_surface,
            "range_relation": "exact",
            "evaluation_note": "",
        }

    g = parse_bounded_range(gold)
    p = parse_bounded_range(pred)

    if g is None or p is None:
        return {
            "primary_score": 0.0,
            "strict_exact_match": 0,
            "metric": "range_no_match",
            "normalized_ground_truth": (
                list(g) if g is not None else gold_surface
            ),
            "normalized_prediction": (
                list(p) if p is not None else pred_surface
            ),
            "range_relation": "no_match",
            "evaluation_note": (
                "Open-ended/single-value legend categories require "
                "normalized exact matching; bounded intervals additionally "
                "support numeric comparison."
            ),
        }

    exact = ranges_exact(g, p)
    style = str(legend_style or "").casefold().strip()

    if exact:
        score, relation = 1.0, "exact"
    elif style == "c" and ranges_strictly_overlap(g, p):
        score, relation = 0.5, "overlap"
    else:
        score, relation = 0.0, "no_match"

    return {
        "primary_score": score,
        "strict_exact_match": int(exact),
        "metric": (
            "continuous_range_score"
            if style == "c"
            else "discrete_range_exact_match"
        ),
        "normalized_ground_truth": list(g),
        "normalized_prediction": list(p),
        "range_relation": relation,
        "evaluation_note": (
            "Continuous: exact=1, strict overlap=0.5, touching=0."
            if style == "c"
            else "Discrete: exact normalized range match only."
        ),
    }


def evaluate_rank(
    gold: str,
    pred: str,
    country: Any,
) -> dict[str, Any]:
    g, _ = parse_ranking_with_metadata(gold, country)
    p, canonical_symbol_format = parse_ranking_with_metadata(pred, country)

    if g is None or p is None:
        return {
            "primary_score": 0.0,
            "strict_exact_match": 0,
            "metric": "rank_parse_error",
            "rankwise_precision": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "rank_precisions": [],
            "rank_hits": [],
            "rank_parse_success": 0,
            "rank_format_compliant": int(canonical_symbol_format),
            "normalized_ground_truth": (
                [sorted(x) for x in g] if g else None
            ),
            "normalized_prediction": (
                [sorted(x) for x in p] if p else None
            ),
            "evaluation_note": (
                "Could not parse one or both ranking expressions. "
                "Comma-only rankings remain intentionally ambiguous."
            ),
        }

    rwp, per_rank = rankwise_precision(g, p)
    mrr, map_score, hits = rankwise_mrr_and_map(g, p)

    return {
        "primary_score": rwp,
        "strict_exact_match": int(exact_rank_match(g, p)),
        "metric": "rankwise_precision",
        "rankwise_precision": rwp,
        "mrr": mrr,
        "map": map_score,
        "rank_precisions": per_rank,
        "rank_hits": hits,
        "rank_parse_success": 1,
        "rank_format_compliant": int(canonical_symbol_format),
        "normalized_ground_truth": [sorted(x) for x in g],
        "normalized_prediction": [sorted(x) for x in p],
        "evaluation_note": (
            ""
            if canonical_symbol_format
            else "Ranking was parsed from explicit relational prose rather "
                 "than the requested <, >, = symbol format."
        ),
    }


def evaluate_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    pred, extraction_method = resolve_prediction_answer(record)

    gold = str(record.get("ground_truth", "") or "").strip()
    answer_type = str(
        record.get("ground_truth_type", "")
    ).casefold().strip()

    template_no = int(record.get("template_no", -1))
    legend_style = str(
        record.get(
            "legend_style",
            record.get("c_or_d", ""),
        )
    ).strip()

    country = normalize_country(record.get("country", ""))

    if not pred:
        result = {
            "primary_score": 0.0,
            "strict_exact_match": 0,
            "metric": "not_extractable",
            "normalized_ground_truth": "",
            "normalized_prediction": "",
            "evaluation_note": "No extractable final answer.",
        }

    elif template_no == 43:
        result = evaluate_rank(gold, pred, country)

    elif answer_type == "binary":
        result = evaluate_binary(gold, pred)

    elif answer_type == "count":
        result = evaluate_count(gold, pred)

    elif answer_type == "range":
        result = evaluate_range(gold, pred, legend_style)

    elif answer_type == "list":
        result = evaluate_list(gold, pred, country)

    elif answer_type == "single":
        result = evaluate_single(gold, pred, country)

    else:
        exact = (
            bool(compact_normalize(gold))
            and compact_normalize(gold)
            == compact_normalize(pred)
        )

        result = {
            "primary_score": float(exact),
            "strict_exact_match": int(exact),
            "metric": "fallback_text_exact_match",
            "normalized_ground_truth": normalize_text(gold),
            "normalized_prediction": normalize_text(pred),
            "evaluation_note": (
                f"Unexpected ground_truth_type={answer_type!r}."
            ),
        }

    row = dict(record)
    row.update(
        {
            "evaluated_answer": pred,
            "answer_extraction_method": extraction_method,
            **result,
        }
    )
    return row


# ============================================================
# 7. Input loading and deterministic QA IDs
# ============================================================

def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_qa_id(sample: Mapping[str, Any], index: int) -> str:
    country = str(sample.get("country", "unknown")).strip().lower()
    map_no = str(sample.get("map_no", "unknown")).strip()
    template_no = int(sample.get("template_no", -1))

    return (
        f"mapwise_{country}_{map_no}_"
        f"t{template_no}_idx{index:04d}"
    )


def load_prediction_records(
    path: str | Path,
) -> list[dict[str, Any]]:
    obj = load_json(path)

    if isinstance(obj, dict):
        for key in ("results", "predictions", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError(
            "Prediction JSON must contain a list of records."
        )

    return [
        dict(x)
        for x in obj
        if isinstance(x, Mapping)
    ]


def load_qa_records(
    path: str | Path,
) -> list[dict[str, Any]]:
    obj = load_json(path)

    if isinstance(obj, dict):
        for key in ("data", "questions", "results", "items"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break

    if not isinstance(obj, list):
        raise ValueError(
            "QA JSON must contain a list of records."
        )

    rows = [
        dict(x)
        for x in obj
        if isinstance(x, Mapping)
    ]

    for index, row in enumerate(rows):
        if not str(row.get("qa_id", "")).strip():
            row["qa_id"] = make_qa_id(row, index)

    return rows


def merge_predictions_with_qa(
    predictions: list[dict[str, Any]],
    qa_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_map = {
        str(x.get("qa_id")): x
        for x in predictions
        if x.get("qa_id") is not None
    }

    merged = []

    for qa in qa_records:
        qa_id = str(qa["qa_id"])
        combined = dict(qa)
        combined.update(pred_map.get(qa_id, {}))
        combined["qa_id"] = qa_id
        merged.append(combined)

    return merged


# ============================================================
# 8. Summaries
# ============================================================

def safe_mean(
    values: Iterable[Optional[float]],
) -> Optional[float]:
    cleaned = [
        float(v)
        for v in values
        if v is not None
        and math.isfinite(float(v))
    ]

    return (
        sum(cleaned) / len(cleaned)
        if cleaned
        else None
    )


def summarize_group(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    exact = sum(
        int(x.get("strict_exact_match", 0))
        for x in rows
    )

    out = {
        "total": total,
        "exact_correct": exact,
        "exact_match_accuracy": (
            exact / total if total else None
        ),
        "mean_primary_score": safe_mean(
            x.get("primary_score")
            for x in rows
        ),
    }

    for output_key, row_key in {
        "mean_precision": "precision",
        "mean_recall": "recall",
        "mean_f1": "f1",
        "mean_rankwise_precision": "rankwise_precision",
        "mean_mrr": "mrr",
        "mean_map": "map",
        "rank_parse_success_rate": "rank_parse_success",
        "rank_format_compliance_rate": "rank_format_compliant",
    }.items():
        vals = [
            x.get(row_key)
            for x in rows
            if x.get(row_key) is not None
        ]
        out[output_key] = (
            safe_mean(vals)
            if vals
            else None
        )

    return out


def summarize_results(
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = {
        "overall": summarize_group(rows)
    }

    dims = {
        "by_country": "country",
        "by_ground_truth_type": "ground_truth_type",
        "by_template_no": "template_no",
        "by_legend_style": "legend_style",
        "by_relative_region": "relative_region",
        "by_map_no": "map_no",
        "by_generation_status": "generation_status",
        "by_metric": "metric",
    }

    for output_key, field in dims.items():
        groups: dict[
            str,
            list[Mapping[str, Any]],
        ] = defaultdict(list)

        for row in rows:
            groups[str(row.get(field, "Unknown"))].append(row)

        summary[output_key] = {
            k: summarize_group(v)
            for k, v in sorted(groups.items())
        }

    extractable = sum(
        bool(str(x.get("evaluated_answer", "")).strip())
        for x in rows
    )

    complete = sum(
        str(x.get("generation_status", "")).casefold()
        == "complete"
        for x in rows
    )

    summary["generation"] = {
        "total": len(rows),
        "extractable_answers": extractable,
        "answer_extraction_rate": (
            extractable / len(rows)
            if rows
            else None
        ),
        "complete_generations": complete,
        "completion_rate": (
            complete / len(rows)
            if rows
            else None
        ),
        "average_generated_tokens": safe_mean(
            x.get("generated_tokens")
            for x in rows
        ),
        "average_inference_seconds": safe_mean(
            x.get("inference_seconds")
            for x in rows
        ),
    }

    summary["total"] = summary["overall"]["total"]
    summary["overall_exact_match_accuracy"] = (
        summary["overall"]["exact_match_accuracy"]
    )
    summary["overall_mean_primary_score"] = (
        summary["overall"]["mean_primary_score"]
    )

    # Expose official Rank RWP and ranking diagnostics at the top level.
    rank_group = summary.get("by_template_no", {}).get("43")
    if rank_group is not None:
        summary["rank_rwp"] = rank_group.get("mean_rankwise_precision")
        summary["rank_total"] = rank_group.get("total", 0)
        summary["rank_parse_success_rate"] = rank_group.get(
            "rank_parse_success_rate"
        )
        summary["rank_format_compliance_rate"] = rank_group.get(
            "rank_format_compliance_rate"
        )
    else:
        summary["rank_rwp"] = None
        summary["rank_total"] = 0
        summary["rank_parse_success_rate"] = None
        summary["rank_format_compliance_rate"] = None

    return summary


# ============================================================
# 9. Save results
# ============================================================

def save_results(
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    details_json = output / "evaluation_details.json"
    summary_json = output / "evaluation_summary.json"
    details_csv = output / "evaluation_details.csv"

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
        "legend_style",
        "relative_region",
        "generation_status",
        "generated_tokens",
        "inference_seconds",
        "final_answer",
        "evaluated_answer",
        "answer_extraction_method",
        "primary_score",
        "strict_exact_match",
        "metric",
        "precision",
        "recall",
        "f1",
        "rankwise_precision",
        "mrr",
        "map",
        "rank_parse_success",
        "rank_format_compliant",
        "range_relation",
        "normalized_ground_truth",
        "normalized_prediction",
        "evaluation_note",
    ]

    with details_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            item = dict(row)

            for key in (
                "normalized_ground_truth",
                "normalized_prediction",
                "rank_precisions",
                "rank_hits",
            ):
                if isinstance(item.get(key), (list, dict)):
                    item[key] = json.dumps(
                        item[key],
                        ensure_ascii=False,
                    )

            writer.writerow(item)

    return {
        "details_json": details_json,
        "summary_json": summary_json,
        "details_csv": details_csv,
    }


# ============================================================
# 10. Main evaluation API
# ============================================================

def evaluate_mapwise(
    prediction_json: str | Path,
    qa_json: Optional[str | Path] = None,
    *,
    output_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    prediction_path = Path(prediction_json).resolve()

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction JSON not found: {prediction_path}"
        )

    predictions = load_prediction_records(prediction_path)

    records = (
        merge_predictions_with_qa(
            predictions,
            load_qa_records(qa_json),
        )
        if qa_json is not None
        else predictions
    )

    required = {
        "qa_id",
        "country",
        "ground_truth",
        "ground_truth_type",
        "template_no",
    }

    for i, record in enumerate(records):
        missing = required.difference(record)

        if missing:
            raise ValueError(
                f"Record {i} is missing required fields: "
                f"{sorted(missing)}"
            )

    rows = [
        evaluate_sample(record)
        for record in records
    ]

    summary = summarize_results(rows)

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else prediction_path.parent
    )

    paths = save_results(
        rows,
        summary,
        destination,
    )

    summary["artifacts"] = {
        k: str(v.resolve())
        for k, v in paths.items()
    }

    paths["summary_json"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return summary


# ============================================================
# 11. Console output
# ============================================================

def format_metric(
    value: Optional[float],
) -> str:
    return (
        "N/A"
        if value is None
        else f"{value:.4f} ({value:.2%})"
    )


def print_summary(
    summary: Mapping[str, Any],
) -> None:
    print("=" * 76)
    print("MAPWISE EVALUATION")
    print("=" * 76)

    print(
        f"Questions:                    "
        f"{summary.get('total', 0)}"
    )
    print(
        f"Overall exact-match accuracy: "
        f"{format_metric(summary.get('overall_exact_match_accuracy'))}"
    )
    print(
        f"Overall mean primary score:    "
        f"{format_metric(summary.get('overall_mean_primary_score'))}"
    )

    print("\nBy country:")
    for country, group in summary.get("by_country", {}).items():
        print(
            f"  {country:<8} exact accuracy: "
            f"{format_metric(group.get('exact_match_accuracy'))}"
        )

    for answer_type in (
        "Binary",
        "Count",
        "Single",
        "List",
        "Range",
    ):
        group = next(
            (
                v
                for k, v in summary.get(
                    "by_ground_truth_type",
                    {},
                ).items()
                if k.casefold() == answer_type.casefold()
            ),
            None,
        )

        if group:
            print(
                f"{answer_type:<12} primary score:       "
                f"{format_metric(group.get('mean_primary_score'))}"
            )

    rank = summary.get(
        "by_template_no",
        {},
    ).get("43")

    if rank:
        print(
            f"Rank RWP:                     "
            f"{format_metric(rank.get('mean_rankwise_precision'))}"
        )
        print(
            f"Rank parse success rate:      "
            f"{format_metric(rank.get('rank_parse_success_rate'))}"
        )
        print(
            f"Rank format compliance rate:  "
            f"{format_metric(rank.get('rank_format_compliance_rate'))}"
        )
        print(
            f"Rank MRR:                     "
            f"{format_metric(rank.get('mean_mrr'))}"
        )
        print(
            f"Rank MAP:                     "
            f"{format_metric(rank.get('mean_map'))}"
        )

    generation = summary.get("generation", {})

    print(
        f"Answer extraction rate:       "
        f"{format_metric(generation.get('answer_extraction_rate'))}"
    )
    print(
        f"Generation completion rate:   "
        f"{format_metric(generation.get('completion_rate'))}"
    )

    if generation.get("average_generated_tokens") is not None:
        print(
            f"Average generated tokens:     "
            f"{generation['average_generated_tokens']:.2f}"
        )

    if generation.get("average_inference_seconds") is not None:
        print(
            f"Average inference seconds:    "
            f"{generation['average_inference_seconds']:.2f}"
        )

    artifacts = summary.get("artifacts", {})

    if artifacts:
        print(
            f"Summary JSON:                 "
            f"{artifacts.get('summary_json', '')}"
        )
        print(
            f"Details JSON:                 "
            f"{artifacts.get('details_json', '')}"
        )
        print(
            f"Details CSV:                  "
            f"{artifacts.get('details_csv', '')}"
        )


# ============================================================
# 12. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mixed-country MapWise prediction JSON."
    )

    parser.add_argument("--predictions-json", required=True)
    parser.add_argument("--qa-json", default=None)
    parser.add_argument("--output-dir", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_summary(
        evaluate_mapwise(
            args.predictions_json,
            args.qa_json,
            output_dir=args.output_dir,
        )
    )


if __name__ == "__main__":
    main()