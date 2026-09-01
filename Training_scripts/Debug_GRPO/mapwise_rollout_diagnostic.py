#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapWise GRPO rollout feasibility diagnostic.

This script does NOT train. It samples multiple stochastic rollouts per
question and diagnoses whether groups are all-correct, mixed-correctness, or
all-wrong. Correctness is defined by strict_exact_match from the user's
mapwise_evaluation.py, so Single recall and continuous-Range overlap do not
count as correct for this GRPO feasibility check.

Default diagnostic:
- 60 questions total, 15 each from L0/L1/L2/L3
- approximately balanced answer types within each level
- 4 stochastic generations/question
- temperature=0.8, top_p=0.95, top_k=50

The raw template_no is never rewritten. If corrected_template_no and
ability_level already exist in the split JSON, they are carried through.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

import unsloth  # noqa: F401  # must precede transformers
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
NUM_GENERATIONS = 4
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 50
SEED = 42
SAMPLE_SIZE = 60

SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
SUPPORTED_COUNTRIES = {"china", "india", "usa"}
ABILITY_LEVELS = ("L0", "L1", "L2", "L3")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_QA_JSON = PROJECT_ROOT / "Train_Val_data" / "Mapwise" / "mapwise_grpo_train.json"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "Datasets" / "mapwise-dataset"
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation\Evaluation_results\Mapwise_GRPO_diagnostic"
)
DEFAULT_ROLLOUT_JSON = DEFAULT_OUTPUT_DIR / "mapwise_grpo_rollouts.json"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "mapwise_grpo_summary.json"


def load_evaluation_module(evaluation_script: Optional[Path]):
    """Load the user's mapwise_evaluation.py and require evaluate_sample()."""
    if evaluation_script is None:
        try:
            import mapwise_evaluation as module
        except Exception as error:
            raise ImportError(
                "Could not import mapwise_evaluation.py. Put this script beside "
                "mapwise_evaluation.py or provide --evaluation-script."
            ) from error
    else:
        evaluation_script = evaluation_script.expanduser().resolve()
        if not evaluation_script.is_file():
            raise FileNotFoundError(f"Evaluation script not found:\n{evaluation_script}")
        spec = importlib.util.spec_from_file_location("mapwise_evaluation_external", evaluation_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluation script:\n{evaluation_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_sample"):
        raise AttributeError("Evaluation module must expose evaluate_sample(record).")
    return module


def load_json_list(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"QA JSON does not exist:\n{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("MapWise QA JSON must be a non-empty top-level list.")

    rows = []
    required = ("country", "map_no", "template_no", "question", "ground_truth", "ground_truth_type")
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise TypeError(f"Sample {index} is not a dictionary.")
        row = dict(item)
        missing = [k for k in required if k not in row]
        if missing:
            raise KeyError(f"Sample {index} missing fields: {missing}")
        country = str(row.get("country", "")).strip().lower()
        if country not in SUPPORTED_COUNTRIES:
            raise ValueError(f"Sample {index} has unsupported country={country!r}.")
        if not str(row.get("qa_id", "")).strip():
            source_index = int(row.get("source_index", index))
            row["qa_id"] = (
                f"mapwise_{country}_{row['map_no']}_"
                f"t{int(row['template_no'])}_src{source_index:04d}"
            )
        rows.append(row)
    return rows


def validate_image(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist:\n{path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(f"Image cannot be opened:\n{path}\n{error}") from error


def resolve_mapwise_image(sample: Mapping[str, Any], image_root: Path) -> Path:
    """Resolve <root>/<country>/images/with_annotations/<map_no>.<suffix>."""
    country = str(sample.get("country", "")).strip().lower()
    map_no = str(sample.get("map_no", "")).strip()
    qa_id = str(sample.get("qa_id", "unknown"))
    if country not in SUPPORTED_COUNTRIES:
        raise ValueError(f"Unsupported country={country!r} for {qa_id}.")
    if not map_no:
        raise ValueError(f"Sample {qa_id} has empty map_no.")

    country_root = image_root.expanduser().resolve() / country / "images" / "with_annotations"
    if not country_root.is_dir():
        raise FileNotFoundError(f"Country image directory does not exist:\n{country_root}")

    raw = Path(map_no)
    if raw.suffix:
        direct = country_root / raw
        if direct.exists():
            validate_image(direct)
            return direct.resolve()

    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = country_root / f"{map_no}{suffix}"
        if candidate.exists():
            validate_image(candidate)
            return candidate.resolve()

    target_stem = raw.stem.casefold()
    matches = [
        p for p in country_root.rglob("*")
        if p.is_file() and p.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        and p.stem.casefold() == target_stem
    ]
    if len(matches) == 1:
        validate_image(matches[0])
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple images match {qa_id}, map_no={map_no!r}:\n" +
            "\n".join(str(p) for p in matches)
        )
    raise FileNotFoundError(
        f"No image found for qa_id={qa_id}, country={country}, map_no={map_no!r} under:\n{country_root}"
    )


def build_mapwise_prompt(question: str) -> str:
    """Same semantic prompt as the mixed-country MapWise benchmark inference."""
    return f"""
This is a cartographic reasoning question from the MapWise dataset.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>

For ranking questions, express the final ranking explicitly using
"<", ">", and "=" as appropriate between items. Use "=" for ties.
Do not replace the ranking symbols with commas or prose.

Example:
Final answer: A < B = C
""".strip()


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]}]


def has_final_answer_marker(raw_response: str) -> bool:
    return bool(re.search(r"final\s+answer\s*:\s*\S", raw_response or "", flags=re.I))


def reasoning_trace_status(raw_response: str) -> tuple[str, int]:
    text = str(raw_response or "")
    lower = text.lower()
    start = lower.find("<think>")
    end = lower.rfind("</think>")
    if start < 0 and end < 0:
        return "missing", 0
    if start >= 0 and end < 0:
        return "truncated", len(text[start + len("<think>"):].split())
    if start < 0 and end >= 0:
        return "malformed", 0
    reasoning = text[start + len("<think>"):end].strip()
    if not reasoning:
        return "empty", 0
    return "valid", len(reasoning.split())


def repetition_ratio(text: str, ngram_size: int = 4) -> float:
    words = re.findall(r"\w+", str(text).lower())
    if len(words) < ngram_size * 2:
        return 0.0
    ngrams = [tuple(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def classify_generation_status(raw_response: str, evaluated_answer: str,
                               generated_tokens: int, max_new_tokens: int) -> str:
    if not str(raw_response).strip():
        return "empty"
    marker = bool(re.search(r"final\s+answer\s*:", raw_response or "", flags=re.I))
    reached_limit = generated_tokens >= max_new_tokens
    if marker and str(evaluated_answer).strip():
        return "complete"
    if reached_limit:
        return "truncated"
    if str(evaluated_answer).strip():
        return "fallback_extracted"
    return "not_extractable"


def infer_ability_level(sample: Mapping[str, Any]) -> str:
    """Prefer stored ability_level; otherwise infer using the corrected mapping."""
    stored = str(sample.get("ability_level", "")).strip().upper()
    if stored in ABILITY_LEVELS:
        return stored

    raw = int(sample.get("template_no", -1))
    if 1 <= raw <= 13:
        corrected = raw
    elif 15 <= raw <= 43:
        corrected = raw - 1
    else:
        raise ValueError(f"Cannot infer ability level from raw template_no={raw}.")

    l0 = {1, 2, 3, 4, 5, 6, 7, 8, 40}
    l1 = {9, 10, 15, 18}
    l2 = {11, 12, 13, 14, 16, *range(20, 32), 34, 37, 38, 41, 42}
    l3 = {17, 19, 32, 33, 35, 36, 39}
    if corrected in l0:
        return "L0"
    if corrected in l1:
        return "L1"
    if corrected in l2:
        return "L2"
    if corrected in l3:
        return "L3"
    raise ValueError(f"Corrected template_no={corrected} has no ability mapping.")


def _balanced_take_from_level(items, n: int, rng: random.Random):
    """Round-robin across answer types within one ability level."""
    buckets = defaultdict(list)
    for item in items:
        buckets[str(item[1].get("ground_truth_type", "unknown"))].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < n:
        added = False
        for key in keys:
            if buckets[key]:
                chosen.append(buckets[key].pop())
                added = True
                if len(chosen) >= n:
                    break
        if not added:
            break
    return chosen


def select_diagnostic_samples(samples: list[dict[str, Any]], sample_size: int, seed: int):
    """Evenly distribute sample_size across L0-L3, then balance answer types."""
    indexed = list(enumerate(samples))
    if sample_size <= 0 or sample_size >= len(indexed):
        return indexed
    rng = random.Random(seed)
    level_buckets = {level: [] for level in ABILITY_LEVELS}
    for item in indexed:
        level_buckets[infer_ability_level(item[1])].append(item)

    base = sample_size // 4
    remainder = sample_size % 4
    targets = {level: base + (1 if i < remainder else 0)
               for i, level in enumerate(ABILITY_LEVELS)}
    for level, target in targets.items():
        if len(level_buckets[level]) < target:
            raise ValueError(f"Not enough {level}: need {target}, have {len(level_buckets[level])}.")

    chosen = []
    for level in ABILITY_LEVELS:
        chosen.extend(_balanced_take_from_level(level_buckets[level], targets[level], rng))
    rng.shuffle(chosen)
    return chosen


def print_gpu_information() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. "
            f"torch={torch.__version__}, CUDA build={torch.version.cuda}"
        )
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")


def infer_base_model_name(adapter_path: Path, fallback: str) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return fallback
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return str(config.get("base_model_name_or_path", "")).strip() or fallback
    except Exception:
        return fallback


def load_model_and_processor(model_name: str, adapter_path: Optional[Path]):
    if adapter_path is not None:
        adapter_path = adapter_path.expanduser().resolve()
        model_name = infer_base_model_name(adapter_path, model_name)

    print("\nLoading model...")
    print(f"Base model: {model_name}")
    print_gpu_information()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path is None:
        print("Adapter:    None (baseline)")
        model = base_model
    else:
        if PeftModel is None:
            raise ImportError("peft is required when --adapter-path is supplied.")
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory does not exist:\n{adapter_path}")
        print(f"Adapter:    {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)

    model.eval()
    return model, processor, model_name


def move_inputs_to_model_device(inputs: Mapping[str, Any], model) -> dict[str, Any]:
    device = next(model.parameters()).device
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}


def generate_one_response(model, processor, messages,
                          *, max_new_tokens: int, temperature: float,
                          top_p: float, top_k: int, seed: int) -> tuple[str, int]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_model_device(inputs, model)
    input_length = int(inputs["input_ids"].shape[1])

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


def evaluate_rollout(evaluation_module, sample: Mapping[str, Any], raw_response: str) -> dict[str, Any]:
    """Score with the user's evaluator; strict_exact_match is the GRPO correctness reward."""
    record = dict(sample)
    record["raw_response"] = raw_response
    record["final_answer"] = ""  # force evaluator to use its own extraction logic
    result = evaluation_module.evaluate_sample(record)
    strict = int(result.get("strict_exact_match", 0))
    return {
        "correct": bool(strict),
        "correctness_reward": float(strict),
        "primary_score": float(result.get("primary_score", 0.0) or 0.0),
        "metric": str(result.get("metric", "")),
        "evaluated_answer": str(result.get("evaluated_answer", "") or ""),
        "answer_extraction_method": str(result.get("answer_extraction_method", "")),
        "normalized_ground_truth": result.get("normalized_ground_truth"),
        "normalized_prediction": result.get("normalized_prediction"),
        "evaluation_note": str(result.get("evaluation_note", "") or ""),
    }


def atomic_save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_existing_groups(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Existing rollout file is not a list: {path}")
    return [dict(x) for x in data if isinstance(x, Mapping)]


def group_is_complete(group: Mapping[str, Any], num_generations: int) -> bool:
    rollouts = group.get("rollouts", [])
    return isinstance(rollouts, list) and len(rollouts) >= num_generations


def replace_or_append_group(groups, new_group) -> None:
    qa_id = str(new_group["qa_id"])
    for i, group in enumerate(groups):
        if str(group.get("qa_id", "")) == qa_id:
            groups[i] = new_group
            return
    groups.append(new_group)


def safe_mean(values) -> float:
    vals = list(values)
    return statistics.mean(vals) if vals else 0.0


def safe_pstdev(values) -> float:
    vals = list(values)
    return statistics.pstdev(vals) if len(vals) >= 2 else 0.0


def build_group_summary(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    correctness = [float(item["correctness_reward"]) for item in rollouts]
    correct_count = int(sum(correctness))
    if correct_count == len(rollouts):
        group_type = "all_correct"
    elif correct_count == 0:
        group_type = "all_wrong"
    else:
        group_type = "mixed_correctness"

    normalized_answers = set()
    for item in rollouts:
        value = item.get("normalized_prediction")
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        normalized_answers.add(str(value))

    return {
        "group_type": group_type,
        "correct_count": correct_count,
        "incorrect_count": len(rollouts) - correct_count,
        "correctness_mean": round(safe_mean(correctness), 4),
        "correctness_std": round(safe_pstdev(correctness), 4),
        "unique_normalized_answers": len(normalized_answers),
    }


def aggregate_partition(groups: list[dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "questions": 0, "all_correct": 0, "mixed_correctness": 0,
            "all_wrong": 0, "mixed_correctness_ratio": 0.0,
            "rollout_accuracy": 0.0,
        }
    types = Counter(g["group_summary"]["group_type"] for g in groups)
    rollout_correct = [
        float(r.get("correctness_reward", 0.0))
        for g in groups for r in g.get("rollouts", [])
    ]
    return {
        "questions": len(groups),
        "all_correct": types.get("all_correct", 0),
        "mixed_correctness": types.get("mixed_correctness", 0),
        "all_wrong": types.get("all_wrong", 0),
        "mixed_correctness_ratio": round(types.get("mixed_correctness", 0) / len(groups), 4),
        "rollout_accuracy": round(safe_mean(rollout_correct), 4),
    }


def aggregate_summary(groups, *, model_name: str, adapter_path: Optional[Path],
                      qa_json: Path, image_root: Path, sample_size: int,
                      num_generations: int, generation_config: dict[str, Any]) -> dict[str, Any]:
    completed = [g for g in groups if g.get("group_summary")]
    all_rollouts = [r for g in completed for r in g.get("rollouts", [])]
    overall = aggregate_partition(completed)

    by_level = {
        level: aggregate_partition([
            g for g in completed if str(g.get("ability_level", "")).upper() == level
        ])
        for level in ABILITY_LEVELS
    }
    answer_types = sorted({str(g.get("ground_truth_type", "Unknown")) for g in completed})
    by_answer_type = {
        answer_type: aggregate_partition([
            g for g in completed if str(g.get("ground_truth_type", "Unknown")) == answer_type
        ])
        for answer_type in answer_types
    }

    trace_status = Counter(r.get("reasoning_status", "unknown") for r in all_rollouts)
    generation_status = Counter(r.get("generation_status", "unknown") for r in all_rollouts)
    mixed_ratio = float(overall["mixed_correctness_ratio"])
    if len(completed) < 20:
        verdict = "insufficient_sample"
    elif mixed_ratio >= 0.25:
        verdict = "promising"
    elif mixed_ratio >= 0.10:
        verdict = "borderline"
    else:
        verdict = "weak_correctness_signal"

    interpretations = {
        "promising": "At least 25% of sampled questions contain both correct and incorrect rollouts.",
        "borderline": "Some useful within-group exact-correctness variation exists, but many groups are uniform.",
        "weak_correctness_signal": "Fewer than 10% of groups have mixed exact correctness; binary exact-match reward may be weak.",
        "insufficient_sample": "Run at least 20 question groups before interpreting feasibility.",
    }

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "model_name": model_name,
        "adapter_path": str(adapter_path.expanduser().resolve()) if adapter_path is not None else None,
        "qa_json": str(qa_json),
        "image_root": str(image_root),
        "requested_sample_size": sample_size,
        "completed_groups": len(completed),
        "num_generations": num_generations,
        "total_rollouts": len(all_rollouts),
        "generation_config": generation_config,
        "correctness_definition": "strict_exact_match from mapwise_evaluation.evaluate_sample",
        "overall": overall,
        "by_ability_level": by_level,
        "by_ground_truth_type": by_answer_type,
        "final_answer_marker_rate": round(safe_mean([
            1.0 if r.get("has_final_answer_marker") else 0.0 for r in all_rollouts
        ]), 4),
        "reasoning_status_counts": dict(trace_status),
        "generation_status_counts": dict(generation_status),
        "average_generated_tokens": round(safe_mean([
            float(r.get("generated_tokens", 0)) for r in all_rollouts
        ]), 2),
        "average_inference_seconds": round(safe_mean([
            float(r.get("inference_seconds", 0)) for r in all_rollouts
        ]), 3),
        "feasibility_verdict": verdict,
        "interpretation": interpretations[verdict],
        "important_note": (
            "Single recall and continuous-Range overlap primary scores are not treated as correct. "
            "Only strict exact match is used for this GRPO diagnostic."
        ),
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("MAPWISE GRPO ROLLOUT FEASIBILITY SUMMARY")
    print("=" * 80)
    overall = summary["overall"]
    print(f"Completed question groups:      {summary['completed_groups']}")
    print(f"Rollouts per question:          {summary['num_generations']}")
    print(f"Total rollouts:                 {summary['total_rollouts']}")
    print(f"Strict rollout accuracy:        {overall['rollout_accuracy']:.3f}")
    print(f"All-correct groups:             {overall['all_correct']}")
    print(f"All-wrong groups:               {overall['all_wrong']}")
    print(f"Mixed-correctness groups:       {overall['mixed_correctness']}")
    print(f"Mixed-correctness group ratio:  {overall['mixed_correctness_ratio']:.3f}")

    print("\nBy ability level:")
    for level in ABILITY_LEVELS:
        row = summary["by_ability_level"][level]
        print(
            f"  {level}: n={row['questions']:>2} | all_correct={row['all_correct']:>2} | "
            f"mixed={row['mixed_correctness']:>2} | all_wrong={row['all_wrong']:>2} | "
            f"mixed_ratio={row['mixed_correctness_ratio']:.3f} | "
            f"rollout_acc={row['rollout_accuracy']:.3f}"
        )

    print("\nBy ground-truth type:")
    for answer_type, row in summary["by_ground_truth_type"].items():
        print(
            f"  {answer_type:<8}: n={row['questions']:>2} | mixed={row['mixed_correctness']:>2} | "
            f"mixed_ratio={row['mixed_correctness_ratio']:.3f} | rollout_acc={row['rollout_accuracy']:.3f}"
        )

    print(f"\nVerdict: {summary['feasibility_verdict']}")
    print(f"Interpretation: {summary['interpretation']}")


def run_diagnostic(*, model_name: str, adapter_path: Optional[Path],
                   evaluation_script: Optional[Path], qa_json: Path,
                   image_root: Path, rollout_json: Path, summary_json: Path,
                   sample_size: int, num_generations: int, max_new_tokens: int,
                   temperature: float, top_p: float, top_k: int, seed: int,
                   resume: bool, overwrite: bool) -> tuple[Path, Path]:
    qa_json = qa_json.expanduser().resolve()
    image_root = image_root.expanduser().resolve()
    rollout_json = rollout_json.expanduser().resolve()
    summary_json = summary_json.expanduser().resolve()

    if num_generations < 2:
        raise ValueError("num_generations must be at least 2.")
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for stochastic diagnosis.")

    evaluation_module = load_evaluation_module(evaluation_script)
    samples = load_json_list(qa_json)
    selected = select_diagnostic_samples(samples, sample_size, seed)

    if overwrite:
        for path in (rollout_json, summary_json):
            if path.exists():
                path.unlink()

    groups = load_existing_groups(rollout_json) if resume and not overwrite else []
    completed_ids = {
        str(g.get("qa_id", "")) for g in groups if group_is_complete(g, num_generations)
    }
    selection_counts = Counter(infer_ability_level(sample) for _, sample in selected)

    print("=" * 80)
    print("MAPWISE GRPO ROLLOUT FEASIBILITY DIAGNOSTIC")
    print("=" * 80)
    print(f"QA JSON:           {qa_json}")
    print(f"Image root:        {image_root}")
    print(f"Model:             {model_name}")
    print(f"Adapter:           {adapter_path if adapter_path is not None else 'None (baseline)'}")
    print(f"Dataset size:      {len(samples)}")
    print(f"Selected groups:   {len(selected)}")
    print(f"Selection by level:{dict(selection_counts)}")
    print(f"Generations/group: {num_generations}")
    print(f"Temperature:       {temperature}")
    print(f"Top-p / top-k:     {top_p} / {top_k}")
    print(f"Max new tokens:    {max_new_tokens}")
    print(f"Seed:              {seed}")
    print(f"Already complete:  {len(completed_ids)}")

    model, processor, resolved_base = load_model_and_processor(model_name, adapter_path)

    try:
        for selected_i, (dataset_index, sample) in enumerate(selected, start=1):
            qa_id = str(sample["qa_id"])
            if resume and qa_id in completed_ids:
                print(f"[{selected_i}/{len(selected)}] {qa_id}: skipped")
                continue

            level = infer_ability_level(sample)
            answer_type = str(sample.get("ground_truth_type", ""))
            print(
                f"\n[{selected_i}/{len(selected)}] {qa_id} | {level} | {answer_type} | "
                f"template={sample.get('template_no')}",
                flush=True,
            )

            image_path = resolve_mapwise_image(sample, image_root)
            question = str(sample.get("question", "")).strip()
            messages = build_messages(image_path, build_mapwise_prompt(question))
            rollout_records = []

            for rollout_index in range(num_generations):
                source_index = int(sample.get("source_index", dataset_index))
                rollout_seed = seed + source_index * 1000 + rollout_index
                start = time.perf_counter()
                raw_response, token_count = generate_one_response(
                    model, processor, messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=rollout_seed,
                )
                elapsed = time.perf_counter() - start

                score = evaluate_rollout(evaluation_module, sample, raw_response)
                reasoning_status, reasoning_words = reasoning_trace_status(raw_response)
                generation_status = classify_generation_status(
                    raw_response,
                    score["evaluated_answer"],
                    token_count,
                    max_new_tokens,
                )

                record = {
                    "rollout_index": rollout_index,
                    "seed": rollout_seed,
                    "raw_response": raw_response,
                    **score,
                    "has_final_answer_marker": has_final_answer_marker(raw_response),
                    "reasoning_status": reasoning_status,
                    "reasoning_word_count": reasoning_words,
                    "repetition_ratio": round(repetition_ratio(raw_response), 4),
                    "generated_tokens": token_count,
                    "generation_status": generation_status,
                    "inference_seconds": round(elapsed, 3),
                }
                rollout_records.append(record)

                preview = score["evaluated_answer"].replace("\n", " ")[:100]
                print(
                    f"  rollout {rollout_index + 1}/{num_generations} | "
                    f"correct={score['correct']} | strict_reward={score['correctness_reward']:.0f} | "
                    f"primary={score['primary_score']:.2f} | status={generation_status} | "
                    f"tokens={token_count} | answer={preview!r}",
                    flush=True,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            group = {
                "qa_id": qa_id,
                "dataset_index": dataset_index,
                "source_index": sample.get("source_index"),
                "country": str(sample.get("country", "")),
                "map_no": str(sample.get("map_no", "")),
                "map_family": str(sample.get("map_family", "")),
                "template_no": int(sample.get("template_no", -1)),
                "corrected_template_no": sample.get("corrected_template_no"),
                "ability_level": level,
                "question": question,
                "ground_truth": str(sample.get("ground_truth", "")),
                "ground_truth_type": answer_type,
                "c_or_d": str(sample.get("c_or_d", "")),
                "relative_region": str(sample.get("relative_region", "")),
                "resolved_image_path": str(image_path),
                "rollouts": rollout_records,
                "group_summary": build_group_summary(rollout_records),
                "generated_at": datetime.now().astimezone().isoformat(),
            }
            replace_or_append_group(groups, group)
            atomic_save_json(groups, rollout_json)

            gs = group["group_summary"]
            print(
                f"  GROUP: {gs['group_type']} | correct={gs['correct_count']}/{num_generations} | "
                f"correctness_std={gs['correctness_std']:.3f}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed groups...", file=sys.stderr)

    finally:
        atomic_save_json(groups, rollout_json)
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_generations": num_generations,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "do_sample": True,
        }
        summary = aggregate_summary(
            groups,
            model_name=resolved_base,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample stochastic MapWise rollouts and diagnose exact-match GRPO feasibility."
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="Optional PEFT/LoRA adapter; omit for baseline.")
    parser.add_argument("--evaluation-script", type=Path, default=None,
                        help="Optional path to mapwise_evaluation.py.")
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rollout-json", type=Path, default=DEFAULT_ROLLOUT_JSON)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                        help="Total questions; default 60 = 15 per L0-L3.")
    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> tuple[Path, Path]:
    args = parse_args()
    return run_diagnostic(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        evaluation_script=args.evaluation_script,
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
