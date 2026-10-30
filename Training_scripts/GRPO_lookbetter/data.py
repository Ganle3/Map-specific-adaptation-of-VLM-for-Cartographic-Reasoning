"""MapVerse JSON reading and answer scoring used by controller GRPO."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path


_ADAPTER_CONTROL_ROOT = Path(__file__).resolve().parents[1] / "GRPO_adapter_control"
if str(_ADAPTER_CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(_ADAPTER_CONTROL_ROOT))

# This is the exact dataset-aware scorer used by the matched adapter control,
# whose trainer enters through Debug_GRPO/grpo_joint_debug.py.
from unified_evaluation_exact import evaluate_sample as evaluate_unified_sample


def build_cartographic_prompt(question: str) -> str:
    """Native-GRPO answer contract, shared by MapWise and MapVerse rollouts."""
    return f"""
This is a cartographic reasoning question.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()


@dataclass(frozen=True)
class MapQuestion:
    sample_id: str
    image_path: Path
    question: str
    answer: str
    answer_type: str
    metadata: dict


def load_mapverse_questions(json_path: Path, image_root: Path, *, mapwise_root: Path | None = None,
                            limit: int | None = None, recoverable_only: bool = False) -> list[MapQuestion]:
    """Load MapVerse-only or unified MapVerse/MapWise JSON records.

    ``image_root`` is the MapVerse image directory. For a unified JSON,
    ``mapwise_root`` must point to the cloned ``mapwise-dataset`` repository.
    """
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON list of MapVerse records.")
    records = []
    for row in rows:
        if recoverable_only and str(row.get("diagnostic_class", "")) != "mixed":
            continue
        for key in ("image_name", "question", "correct_answer"):
            if not row.get(key):
                raise ValueError(f"Missing {key!r} in dataset record {row.get('sample_id')!r}.")
        image_path = resolve_image_path(row, image_root, mapwise_root)
        metadata = dict(row)
        # Unified When--Where JSONs identify MapWise images as
        # ``india/map22_6C`` etc., but do not always carry an explicit
        # country field.  The exact evaluator needs that country to resolve
        # colliding abbreviations such as MH -> Maharashtra (India).
        if str(metadata.get("dataset", "mapverse")).casefold() == "mapwise" and not metadata.get("country"):
            parts = Path(str(metadata["image_name"])).parts
            if parts:
                metadata["country"] = parts[0]
        records.append(MapQuestion(
            sample_id=str(row.get("sample_id", len(records))), image_path=image_path,
            question=str(row["question"]), answer=str(row["correct_answer"]),
            answer_type=str(row.get("answer_type", "Single Entity")), metadata=metadata,
        ))
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError("No usable questions after dataset filtering.")
    return records


def resolve_image_path(row: dict, mapverse_root: Path, mapwise_root: Path | None) -> Path:
    """Resolve a row's image without guessing between datasets.

    MapWise records use e.g. ``china/map3_6D`` plus ``with_annotations``;
    their images live under ``<root>/china/images/with_annotations``. MapVerse
    records keep their original file name and are resolved directly below the
    supplied MapVerse image directory.
    """
    dataset = str(row.get("dataset", "mapverse")).lower()
    image_name = Path(str(row["image_name"]))
    if dataset == "mapverse":
        path = (mapverse_root / image_name).resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"MapVerse image referenced by dataset is missing: {path}")
    if dataset != "mapwise":
        raise ValueError(f"Unsupported dataset value {dataset!r} for image {image_name!s}.")
    if mapwise_root is None:
        raise ValueError("A unified dataset with MapWise rows requires --mapwise-root.")
    if len(image_name.parts) != 2:
        raise ValueError(f"Expected MapWise image_name '<country>/<stem>', got {image_name!s}.")
    country, stem = image_name.parts
    variant = str(row.get("map_type") or "with_annotations")
    base = (mapwise_root / country / "images" / variant / stem)
    if base.suffix:
        candidates = [base]
    else:
        candidates = [base.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg", ".webp")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"MapWise image missing for image_name={image_name!s}, map_type={variant!r}; "
        f"looked below {base.parent}"
    )


def extract_final_answer(text: str) -> str:
    matches = re.findall(r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$", text)
    return matches[-1].strip() if matches else ""


def evaluate_answer(response: str, item: MapQuestion) -> dict:
    """Use the established dataset-specific exact evaluator for RL reward.

    MapWise Range answers require numeric endpoint comparison (for example
    ``14-15.9%`` equals ``14.0%-15.9%``), which a string-only reward cannot
    safely reproduce. MapVerse keeps its existing Boolean/Counting/Entity
    evaluator. Neither path uses an LLM judge or fuzzy similarity.
    """
    dataset = str(item.metadata.get("dataset", "mapverse")).casefold()
    if dataset not in {"mapwise", "mapverse"}:
        raise ValueError(f"Unsupported dataset for exact evaluation: {dataset!r}")
    record = {
        "raw_response": response,
        "final_answer": "",
        "ground_truth": item.answer,
        "ground_truth_type": item.answer_type,
        "country": ("mapverse" if dataset == "mapverse"
                    else item.metadata.get("country", "")),
        "template_no": item.metadata.get("template_no", -1),
        "legend_style": item.metadata.get("legend_style", ""),
        "c_or_d": item.metadata.get("c_or_d", ""),
        "qa_id": item.metadata.get("qa_id", item.sample_id),
    }
    result = evaluate_unified_sample(record)
    return {
        "reward": float(result["strict_exact_match"]),
        "evaluator": "unified_evaluation_exact",
        "prediction": result.get("evaluated_answer", ""),
        "metric": result.get("metric", ""),
        "note": result.get("evaluation_note", ""),
        "canonical_prediction": result.get("normalized_prediction", ""),
        "canonical_target": result.get("normalized_ground_truth", ""),
    }


def answer_reward(response: str, item: MapQuestion) -> float:
    """Compatibility wrapper used by callers that only require scalar reward."""
    return evaluate_answer(response, item)["reward"]


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()
