#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose stochastic FRIEDA rollouts using a saved adapted best_adapter.

The script does NOT train the model. It loads a saved LoRA/QLoRA adapter and:

1. selects a reproducible, approximately stratified subset of FRIEDA train;
2. samples multiple stochastic completions for each map question;
3. extracts the final answer and applies conservative deterministic scoring;
4. measures within-question reward/correctness variation;
5. writes detailed rollout records and a feasibility summary.

The key diagnostic is the fraction of questions whose sampled completions contain
both correct and incorrect answers. Such groups provide direct relative signal
for GRPO. All-correct and all-wrong groups have zero correctness advantage.

Example:
    python frieda_grpo_feasibility.py \
        --sample-size 50 \
        --num-generations 4 \
        --overwrite
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
NUM_GENERATIONS = 4
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50
SEED = 42
SAMPLE_SIZE = 50

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_QA_JSON = (
    PROJECT_ROOT / "Train_Val_data" / "FRIEDA" / "frieda_train.json"
)
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "Train_Val_data" / "FRIEDA" / "image"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Evaluation_results" / "FRIEDA_GRPO_diagnostic"
DEFAULT_ROLLOUT_JSON = DEFAULT_OUTPUT_DIR / "frieda_grpo_rollouts.json"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "frieda_grpo_summary.json"

CARDINAL_RING = [
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
]

CARDINAL_ALIASES = {
    "n": "north",
    "north": "north",
    "ne": "northeast",
    "north east": "northeast",
    "northeast": "northeast",
    "e": "east",
    "east": "east",
    "se": "southeast",
    "south east": "southeast",
    "southeast": "southeast",
    "s": "south",
    "south": "south",
    "sw": "southwest",
    "south west": "southwest",
    "southwest": "southwest",
    "w": "west",
    "west": "west",
    "nw": "northwest",
    "north west": "northwest",
    "northwest": "northwest",
}

UNIT_TO_METERS = {
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
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}


# -----------------------------------------------------------------------------
# JSON, images, and prompt
# -----------------------------------------------------------------------------


def load_json_list(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist:\n{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not data:
        raise ValueError("FRIEDA JSON must be a non-empty top-level list.")

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Sample {index} is not a dictionary.")

    return data


def validate_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist:\n{path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(f"Image cannot be opened: {path}\n{error}") from error


def resolve_frieda_images(
    sample: dict[str, Any], image_root: Path
) -> list[Path]:
    image_urls = sample.get("image_urls")
    if not isinstance(image_urls, list) or not image_urls:
        raise ValueError(
            f"{sample.get('question_ref', 'unknown')} has no valid image_urls."
        )

    paths: list[Path] = []
    for relative in image_urls:
        relative_path = Path(str(relative).replace("\\", "/"))
        path = (image_root / relative_path).resolve()
        validate_image(path)
        paths.append(path)
    return paths


def build_frieda_prompt(question: str) -> str:
    """Use the same core prompt as normal FRIEDA inference."""
    return f"""
This is a cartographic reasoning question from the FRIEDA dataset.

Use only the supplied map image or images. Carefully inspect all relevant map
labels, legends, symbols, boundaries, directions, distances, scales, and
spatial relationships. When multiple maps are supplied, use all of them and
identify the correspondence between their mapped features.

Question:
{question}

Reason through the problem carefully. Analyze the map systematically and avoid repeating earlier observations.
Use no more than 15 reasoning steps.

At the end, provide the answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()


def build_messages(image_paths: list[Path], prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(path)} for path in image_paths
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


# -----------------------------------------------------------------------------
# Final answer and reasoning diagnostics
# -----------------------------------------------------------------------------


def extract_final_answer(raw_response: Any) -> str:
    if raw_response is None:
        return ""
    text = str(raw_response).strip()
    if not text:
        return ""

    matches = list(
        re.finditer(r"final\s+answer\s*:\s*", text, flags=re.IGNORECASE)
    )
    if matches:
        answer = text[matches[-1].end() :].strip()
    elif "</think>" in text.lower():
        index = text.lower().rfind("</think>")
        answer = text[index + len("</think>") :].strip()
    else:
        return ""

    answer = re.split(
        r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return answer


def reasoning_trace_status(raw_response: str) -> tuple[str, int]:
    """Return trace status and approximate word count inside <think> tags."""
    text = str(raw_response or "")
    lower = text.lower()
    start = lower.find("<think>")
    end = lower.rfind("</think>")

    if start < 0 and end < 0:
        return "missing", 0
    if start >= 0 and end < 0:
        reasoning = text[start + len("<think>") :]
        return "truncated", len(reasoning.split())
    if start < 0 and end >= 0:
        return "malformed", 0

    reasoning = text[start + len("<think>") : end].strip()
    if not reasoning:
        return "empty", 0
    return "valid", len(reasoning.split())


def has_final_answer_marker(raw_response: str) -> bool:
    return bool(
        re.search(r"final\s+answer\s*:\s*\S", raw_response or "", re.I)
    )


def repetition_ratio(text: str, ngram_size: int = 4) -> float:
    """Approximate repeated n-gram fraction. Higher means more looping."""
    words = re.findall(r"\w+", str(text).lower())
    if len(words) < ngram_size * 2:
        return 0.0
    ngrams = [tuple(words[i : i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


# -----------------------------------------------------------------------------
# Conservative deterministic answer scoring
# -----------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("–", "-")
    text = text.lower().strip()
    text = re.sub(r"^final\s+answer\s*:\s*", "", text)
    text = re.sub(r"[\[\]{}()\"']", " ", text)
    text = re.sub(r"[^\w\s./&+-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_cardinal(value: Any) -> str:
    text = normalize_text(value)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return CARDINAL_ALIASES.get(text, text.replace(" ", ""))


def split_semicolon_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    parts = re.split(r"\s*;\s*|\n+", text)
    return [normalize_text(part) for part in parts if normalize_text(part)]


def token_f1(prediction: Any, expected: Any) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(expected).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    overlap = sum((pred_counter & gold_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_distance(value: Any) -> Optional[tuple[float, Optional[str]]]:
    text = normalize_text(value).replace(",", "")
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z]+)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    return number, unit


def distance_correct(
    prediction: Any,
    expected: Any,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    pred = parse_distance(prediction)
    gold = parse_distance(expected)
    if pred is None or gold is None:
        return False

    pred_value, pred_unit = pred
    gold_value, gold_unit = gold

    if pred_unit in UNIT_TO_METERS and gold_unit in UNIT_TO_METERS:
        pred_value *= UNIT_TO_METERS[pred_unit]
        gold_value *= UNIT_TO_METERS[gold_unit]
    elif pred_unit and gold_unit and pred_unit != gold_unit:
        return False

    allowed = max(absolute_tolerance, abs(gold_value) * relative_tolerance)
    return abs(pred_value - gold_value) <= allowed


def score_answer(
    prediction: str,
    expected: str,
    answer_type: str,
    *,
    distance_relative_tolerance: float,
    distance_absolute_tolerance: float,
) -> dict[str, Any]:
    """
    Conservative scoring for feasibility analysis.

    Textual answers receive strict normalized exact/set matching. This is a
    lower-bound estimate because legitimate paraphrases are intentionally not
    accepted without an external judge.
    """
    answer_type = str(answer_type or "textual").strip().lower()
    method = "normalized_exact"

    if not str(prediction or "").strip():
        correct = False
        method = "empty_prediction"
    elif answer_type == "cardinal":
        pred_dir = canonical_cardinal(prediction)
        gold_dir = canonical_cardinal(expected)
        if gold_dir in CARDINAL_RING:
            idx = CARDINAL_RING.index(gold_dir)
            accepted = {
                CARDINAL_RING[(idx - 1) % len(CARDINAL_RING)],
                CARDINAL_RING[idx],
                CARDINAL_RING[(idx + 1) % len(CARDINAL_RING)],
            }
        else:
            accepted = {gold_dir}
        correct = pred_dir in accepted
        method = "cardinal_gold_plus_adjacent_45deg"
    elif answer_type == "distance":
        correct = distance_correct(
            prediction,
            expected,
            relative_tolerance=distance_relative_tolerance,
            absolute_tolerance=distance_absolute_tolerance,
        )
        method = "distance_tolerance"
    elif ";" in str(expected):
        pred_items = split_semicolon_list(prediction)
        gold_items = split_semicolon_list(expected)
        correct = bool(pred_items) and Counter(pred_items) == Counter(gold_items)
        method = "semicolon_set_exact"
    else:
        correct = normalize_text(prediction) == normalize_text(expected)

    return {
        "correct": bool(correct),
        "correctness_reward": 1.0 if correct else 0.0,
        "scoring_method": method,
        "token_f1": round(token_f1(prediction, expected), 4),
        "normalized_prediction": normalize_text(prediction),
        "normalized_expected": normalize_text(expected),
    }


def calculate_auxiliary_rewards(raw_response: str) -> dict[str, float]:
    """Small diagnostics that may later become GRPO reward components."""
    format_reward = 0.1 if has_final_answer_marker(raw_response) else 0.0
    repeat_ratio = repetition_ratio(raw_response)
    repetition_penalty = -0.1 if repeat_ratio >= 0.35 else 0.0
    return {
        "format_reward": format_reward,
        "repetition_penalty": repetition_penalty,
        "repetition_ratio": round(repeat_ratio, 4),
    }


# -----------------------------------------------------------------------------
# Reproducible approximately stratified selection
# -----------------------------------------------------------------------------


def select_diagnostic_samples(
    samples: list[dict[str, Any]], sample_size: int, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    """
    Round-robin across (answer_type, map_count, spatial_relationship) buckets.
    This gives broader coverage than a single contiguous slice.
    """
    indexed = list(enumerate(samples))
    if sample_size <= 0 or sample_size >= len(indexed):
        return indexed

    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in indexed:
        _, sample = item
        key = (
            str(sample.get("answer_type", "unknown")),
            str(sample.get("map_count", "unknown")),
            str(sample.get("spatial_relationship", "unknown")),
        )
        buckets[key].append(item)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets)
    rng.shuffle(keys)
    chosen: list[tuple[int, dict[str, Any]]] = []

    while len(chosen) < sample_size:
        added = False
        for key in keys:
            if buckets[key]:
                chosen.append(buckets[key].pop())
                added = True
                if len(chosen) == sample_size:
                    break
        if not added:
            break

    return chosen


# -----------------------------------------------------------------------------
# Model and generation
# -----------------------------------------------------------------------------


def print_gpu_information() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. "
            f"torch={torch.__version__}, CUDA build={torch.version.cuda}"
        )
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")


def _infer_base_model_name(adapter_path: Path, fallback: str) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return fallback
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        base = str(config.get("base_model_name_or_path", "")).strip()
        return base or fallback
    except Exception:
        return fallback


def load_model_and_processor(
    model_name: str,
    adapter_path: Optional[Path] = None,
):
    print("\nLoading Qwen3-VL model...")
    print(f"Base model: {model_name}")
    print_gpu_information()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path is None:
        print("Adapter:    None (baseline)")
        model = base_model
    else:
        adapter_path = Path(adapter_path).expanduser().resolve()

        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Adapter directory does not exist:\n{adapter_path}"
            )

        print(f"Adapter:    {adapter_path}")

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()
    print("Model and processor loaded successfully.")

    return model, processor, model_name


def move_inputs_to_model_device(inputs: dict[str, Any], model) -> dict[str, Any]:
    device = next(model.parameters()).device
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def generate_one_response(
    model,
    processor,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[str, int]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs, model)
    input_length = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_cache=True,
        )

    generated_tokens = generated_ids[:, input_length:]
    token_count = int(generated_tokens.shape[1])
    text = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del inputs, generated_ids, generated_tokens
    return text, token_count


# -----------------------------------------------------------------------------
# Persistence and summary
# -----------------------------------------------------------------------------


def atomic_save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temp.replace(path)


def load_existing_groups(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(f"Existing rollout file is not a list: {path}")
    return [item for item in data if isinstance(item, dict)]


def group_is_complete(group: dict[str, Any], num_generations: int) -> bool:
    rollouts = group.get("rollouts", [])
    return isinstance(rollouts, list) and len(rollouts) >= num_generations


def replace_or_append_group(
    groups: list[dict[str, Any]], new_group: dict[str, Any]
) -> None:
    question_ref = str(new_group["question_ref"])
    for index, group in enumerate(groups):
        if str(group.get("question_ref", "")) == question_ref:
            groups[index] = new_group
            return
    groups.append(new_group)


def safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_pstdev(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) >= 2 else 0.0


def build_group_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    correctness = [float(item["correctness_reward"]) for item in rollouts]
    total_rewards = [float(item["total_reward"]) for item in rollouts]
    correct_count = int(sum(correctness))

    if correct_count == len(rollouts):
        group_type = "all_correct"
    elif correct_count == 0:
        group_type = "all_wrong"
    else:
        group_type = "mixed_correctness"

    return {
        "group_type": group_type,
        "correct_count": correct_count,
        "incorrect_count": len(rollouts) - correct_count,
        "correctness_mean": round(safe_mean(correctness), 4),
        "correctness_std": round(safe_pstdev(correctness), 4),
        "total_reward_mean": round(safe_mean(total_rewards), 4),
        "total_reward_std": round(safe_pstdev(total_rewards), 4),
        "unique_final_answers": len(
            {normalize_text(item.get("final_answer", "")) for item in rollouts}
        ),
    }


def aggregate_summary(
    groups: list[dict[str, Any]],
    *,
    model_name: str,
    adapter_path: Path,
    qa_json: Path,
    image_root: Path,
    sample_size: int,
    num_generations: int,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    completed = [group for group in groups if group.get("group_summary")]
    rollouts = [r for group in completed for r in group.get("rollouts", [])]

    group_types = Counter(
        group["group_summary"]["group_type"] for group in completed
    )
    trace_status = Counter(r.get("reasoning_status", "unknown") for r in rollouts)

    mixed = group_types.get("mixed_correctness", 0)
    total_groups = len(completed)
    nonzero_correctness_std = sum(
        group["group_summary"]["correctness_std"] > 0 for group in completed
    )
    nonzero_total_reward_std = sum(
        group["group_summary"]["total_reward_std"] > 0 for group in completed
    )

    overall_accuracy = safe_mean(
        [float(r.get("correctness_reward", 0.0)) for r in rollouts]
    )
    parse_rate = safe_mean(
        [1.0 if str(r.get("final_answer", "")).strip() else 0.0 for r in rollouts]
    )
    marker_rate = safe_mean(
        [1.0 if r.get("has_final_answer_marker") else 0.0 for r in rollouts]
    )

    # Heuristic interpretation, not a statistical theorem.
    mixed_ratio = mixed / total_groups if total_groups else 0.0
    if total_groups < 20:
        verdict = "insufficient_sample"
    elif mixed_ratio >= 0.25:
        verdict = "promising"
    elif mixed_ratio >= 0.10:
        verdict = "borderline"
    else:
        verdict = "weak_correctness_signal"

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": model_name,
        "qa_json": str(qa_json),
        "image_root": str(image_root),
        "requested_sample_size": sample_size,
        "completed_groups": total_groups,
        "num_generations": num_generations,
        "total_rollouts": len(rollouts),
        "generation_config": generation_config,
        "overall_strict_accuracy": round(overall_accuracy, 4),
        "final_answer_parse_rate": round(parse_rate, 4),
        "final_answer_marker_rate": round(marker_rate, 4),
        "group_counts": dict(group_types),
        "mixed_correctness_group_ratio": round(mixed_ratio, 4),
        "groups_with_nonzero_correctness_std": nonzero_correctness_std,
        "groups_with_nonzero_total_reward_std": nonzero_total_reward_std,
        "reasoning_status_counts": dict(trace_status),
        "average_generated_tokens": round(
            safe_mean([float(r.get("generated_tokens", 0)) for r in rollouts]), 2
        ),
        "average_inference_seconds": round(
            safe_mean([float(r.get("inference_seconds", 0)) for r in rollouts]), 3
        ),
        "feasibility_verdict": verdict,
        "interpretation": {
            "promising": (
                "At least 25% of sampled questions produced both correct and "
                "incorrect completions, giving direct within-group correctness signal."
            ),
            "borderline": (
                "Some useful within-group correctness variation exists, but many "
                "groups are uniformly correct or uniformly wrong."
            ),
            "weak_correctness_signal": (
                "Fewer than 10% of groups have mixed correctness. Direct GRPO with "
                "a binary correctness reward may be inefficient without denser rewards, "
                "hardness filtering, or a warm start."
            ),
            "insufficient_sample": (
                "Run at least 20 question groups before interpreting feasibility."
            ),
        }[verdict],
        "important_limitation": (
            "Textual scoring is deliberately conservative. Normalized exact/set "
            "matching may mark valid paraphrases as wrong, so strict accuracy and "
            "mixed-group ratios are lower-bound estimates."
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("FRIEDA GRPO FEASIBILITY SUMMARY")
    print("=" * 80)
    print(f"Completed question groups:      {summary['completed_groups']}")
    print(f"Rollouts per question:          {summary['num_generations']}")
    print(f"Total rollouts:                 {summary['total_rollouts']}")
    print(f"Strict rollout accuracy:        {summary['overall_strict_accuracy']:.3f}")
    print(f"Final-answer parse rate:        {summary['final_answer_parse_rate']:.3f}")
    print(f"All-correct groups:             {summary['group_counts'].get('all_correct', 0)}")
    print(f"All-wrong groups:               {summary['group_counts'].get('all_wrong', 0)}")
    print(f"Mixed-correctness groups:       {summary['group_counts'].get('mixed_correctness', 0)}")
    print(f"Mixed-correctness group ratio:  {summary['mixed_correctness_group_ratio']:.3f}")
    print(f"Nonzero correctness std groups: {summary['groups_with_nonzero_correctness_std']}")
    print(f"Verdict:                        {summary['feasibility_verdict']}")
    print(f"Interpretation: {summary['interpretation']}")
    print("Note: textual exact matching is a conservative lower bound.")


# -----------------------------------------------------------------------------
# Main diagnostic loop
# -----------------------------------------------------------------------------


def run_diagnostic(
    *,
    model_name: str,
    adapter_path: Path,
    qa_json: Path,
    image_root: Path,
    rollout_json: Path,
    summary_json: Path,
    sample_size: int,
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    distance_relative_tolerance: float,
    distance_absolute_tolerance: float,
    resume: bool,
    overwrite: bool,
) -> tuple[Path, Path]:
    if adapter_path is not None:
        adapter_path = adapter_path.expanduser().resolve()
    qa_json = qa_json.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    rollout_json = rollout_json.expanduser().resolve()
    summary_json = summary_json.expanduser().resolve()

    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for GRPO diagnosis.")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0 for stochastic rollout.")

    samples = load_json_list(qa_json)
    selected = select_diagnostic_samples(samples, sample_size, seed)

    if overwrite:
        for path in (rollout_json, summary_json):
            if path.exists():
                path.unlink()

    groups = load_existing_groups(rollout_json) if resume and not overwrite else []
    completed_ids = {
        str(group.get("question_ref"))
        for group in groups
        if group_is_complete(group, num_generations)
    }

    print("=" * 80)
    print("FRIEDA GRPO FEASIBILITY DIAGNOSTIC")
    print("=" * 80)
    print(f"QA JSON:          {qa_json}")
    print(f"Adapter path:     {adapter_path.expanduser().resolve() if adapter_path is not None else 'None (baseline)'}")
    print(f"Image root:       {image_root}")
    print(f"Dataset size:     {len(samples)}")
    print(f"Selected groups:  {len(selected)}")
    print(f"Generations/group:{num_generations}")
    print(f"Temperature:      {temperature}")
    print(f"Top-p / top-k:    {top_p} / {top_k}")
    print(f"Max new tokens:   {max_new_tokens}")
    print(f"Seed:             {seed}")
    print(f"Already complete: {len(completed_ids)}")

    model, processor, resolved_base = load_model_and_processor(model_name, adapter_path)

    try:
        for selected_index, (dataset_index, sample) in enumerate(selected, start=1):
            question_ref = str(sample.get("question_ref", f"frieda_{dataset_index}"))
            if resume and question_ref in completed_ids:
                print(f"[{selected_index}/{len(selected)}] {question_ref}: skipped")
                continue

            print(
                f"\n[{selected_index}/{len(selected)}] {question_ref} | "
                f"{sample.get('answer_type')} | {sample.get('map_count')} | "
                f"{sample.get('spatial_relationship')}",
                flush=True,
            )

            image_paths = resolve_frieda_images(sample, image_root)
            question = str(sample.get("question_text", "")).strip()
            expected = str(sample.get("expected_answer", "")).strip()
            prompt = build_frieda_prompt(question)
            messages = build_messages(image_paths, prompt)

            rollout_records: list[dict[str, Any]] = []
            for rollout_index in range(num_generations):
                rollout_seed = seed + dataset_index * 1000 + rollout_index
                start = time.perf_counter()
                raw_response, token_count = generate_one_response(
                    model,
                    processor,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=rollout_seed,
                )
                elapsed = time.perf_counter() - start

                final_answer = extract_final_answer(raw_response)
                score = score_answer(
                    final_answer,
                    expected,
                    str(sample.get("answer_type", "textual")),
                    distance_relative_tolerance=distance_relative_tolerance,
                    distance_absolute_tolerance=distance_absolute_tolerance,
                )
                auxiliary = calculate_auxiliary_rewards(raw_response)
                status, reasoning_words = reasoning_trace_status(raw_response)
                total_reward = (
                    score["correctness_reward"]
                    + auxiliary["format_reward"]
                    + auxiliary["repetition_penalty"]
                )

                record = {
                    "rollout_index": rollout_index,
                    "seed": rollout_seed,
                    "raw_response": raw_response,
                    "final_answer": final_answer,
                    **score,
                    **auxiliary,
                    "total_reward": round(total_reward, 4),
                    "has_final_answer_marker": has_final_answer_marker(raw_response),
                    "reasoning_status": status,
                    "reasoning_word_count": reasoning_words,
                    "generated_tokens": token_count,
                    "inference_seconds": round(elapsed, 3),
                }
                rollout_records.append(record)

                preview = final_answer.replace("\n", " ")[:100]
                print(
                    f"  rollout {rollout_index + 1}/{num_generations} | "
                    f"correct={score['correct']} | reward={total_reward:.2f} | "
                    f"tokens={token_count} | answer={preview!r}",
                    flush=True,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            group = {
                "question_ref": question_ref,
                "dataset_index": dataset_index,
                "question_text": question,
                "expected_answer": expected,
                "answer_type": str(sample.get("answer_type", "textual")),
                "map_count": str(sample.get("map_count", "")),
                "spatial_relationship": str(sample.get("spatial_relationship", "")),
                "domain": str(sample.get("domain", "")),
                "image_urls": list(sample.get("image_urls", [])),
                "resolved_image_paths": [str(path) for path in image_paths],
                "rollouts": rollout_records,
                "group_summary": build_group_summary(rollout_records),
                "generated_at": datetime.now().astimezone().isoformat(),
            }
            replace_or_append_group(groups, group)
            atomic_save_json(groups, rollout_json)

            gs = group["group_summary"]
            print(
                f"  GROUP: {gs['group_type']} | correct={gs['correct_count']}/"
                f"{num_generations} | correctness_std={gs['correctness_std']:.3f}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed groups...", file=sys.stderr)

    finally:
        atomic_save_json(groups, rollout_json)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "distance_relative_tolerance": distance_relative_tolerance,
            "distance_absolute_tolerance": distance_absolute_tolerance,
        }
        summary = aggregate_summary(
            groups,
            model_name=f"{resolved_base} + adapter:{adapter_path}",
            adapter_path=adapter_path,
            qa_json=qa_json,
            image_root=image_root,
            sample_size=sample_size,
            num_generations=num_generations,
            generation_config=generation_config,
        )
        atomic_save_json(summary, summary_json)
        print_summary(summary)
        print(f"\nDetailed rollouts: {rollout_json}")
        print(f"Summary:           {summary_json}")

    return rollout_json, summary_json


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample multiple FRIEDA rollouts and diagnose GRPO feasibility."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional PEFT/LoRA adapter directory. Omit for baseline.",)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rollout-json", type=Path, default=DEFAULT_ROLLOUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--distance-relative-tolerance",
        type=float,
        default=0.20,
        help="Relative tolerance for approximate distance answers (default 20%%).",
    )
    parser.add_argument(
        "--distance-absolute-tolerance",
        type=float,
        default=0.0,
        help="Minimum absolute tolerance after unit conversion.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> tuple[Path, Path]:
    args = parse_args()
    return run_diagnostic(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        qa_json=args.qa_json,
        image_root=args.image_root,
        rollout_json=args.rollout_json,
        summary_json=args.summary_json,
        sample_size=args.sample_size,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        distance_relative_tolerance=args.distance_relative_tolerance,
        distance_absolute_tolerance=args.distance_absolute_tolerance,
        resume=not args.no_resume,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nProgram failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise