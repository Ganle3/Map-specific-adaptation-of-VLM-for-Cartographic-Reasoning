#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRIEDA Qwen3-VL-8B-Thinking GRPO training with Unsloth.

Key design:
- QLoRA / 4-bit base model
- 8 rollouts per prompt by default
- temperature=0.9, top_p=0.95
- reward routing by answer_type
- cardinal: exact=1.0, adjacent 45°=0.9
- distance: <=20%=1.0, <=35%=0.5, <=50%=0.2
- textual lists: set-F1 partial reward
- ordinary textual answers: deterministic normalized exact match
- format reward: +0.1 for exactly one non-empty "Final answer:"
- behavior reward: -0.1 for no final answer, -0.05 for severe literal repetition
- periodic checkpoints plus an explicitly saved final adapter

Important:
The script intentionally does not select a "best validation adapter" during GRPO.
GRPO evaluation is sampled and expensive. Select the best checkpoint afterward
with the existing deterministic FRIEDA validation inference/evaluation pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

# Unsloth must be imported before transformers / trl.
from unsloth import FastVisionModel

import torch
from datasets import Dataset, Image as HFImage, Sequence
from PIL import Image
from trl import GRPOConfig, GRPOTrainer
import os
import wandb


# ============================================================
# Defaults
# ============================================================

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 16384
MAX_PROMPT_LENGTH = 8192
MAX_COMPLETION_LENGTH = 3072

NUM_GENERATIONS = 8
TEMPERATURE = 0.9
TOP_P = 0.95

LORA_RANK = 16
LORA_ALPHA = 16

LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1.0
SAVE_STEPS = 5
SEED = 3407

FORMAT_REWARD = 0.1
NO_FINAL_ANSWER_PENALTY = -0.1
SEVERE_REPETITION_PENALTY = -0.05
SEVERE_REPETITION_THRESHOLD = 0.45

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_QA_JSON = (
    PROJECT_ROOT / "Train_Val_data" / "FRIEDA" / "frieda_train.json"
)
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT / "Train_Val_data" / "FRIEDA" / "image"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "Training_outputs" / "FRIEDA_GRPO_Qwen3VL"
)

# ============================================================
# Prompt and dataset preparation
# ============================================================

def build_frieda_prompt(question: str) -> str:
    return f"""
This is a cartographic reasoning question from the FRIEDA dataset.

Use only the supplied map image or images. Carefully inspect all relevant map
labels, legends, symbols, boundaries, directions, distances, scales, insets,
and spatial relationships. When multiple maps are supplied, use all of them
and determine how their geographic extents correspond.

Question:
{question}

Reason through the problem carefully. Avoid repeating the same observation.
When an inset or enlarged panel is present, distinguish its page position from
its actual geographic extent.

At the end, provide the answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Training JSON does not exist:\n{path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Training JSON must contain a non-empty top-level list.")

    return data


def resolve_image_paths(
    sample: dict[str, Any],
    image_root: Path,
) -> list[Path]:
    urls = sample.get("image_urls")
    if not isinstance(urls, list) or not urls:
        raise ValueError(
            f"{sample.get('question_ref', 'unknown')} has no valid image_urls."
        )

    paths: list[Path] = []
    for item in urls:
        relative = Path(str(item).replace("\\", "/"))
        path = (image_root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing image for {sample.get('question_ref')}:\n{path}"
            )
        paths.append(path)

    return paths


def build_training_dataset(
    qa_json: Path,
    image_root: Path,
) -> Dataset:
    samples = load_json_list(qa_json)
    image_root = image_root.expanduser().resolve()

    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        question_ref = str(sample.get("question_ref", f"frieda_{index}"))
        question = str(sample.get("question_text", "")).strip()
        expected = str(sample.get("expected_answer", "")).strip()
        answer_type = str(sample.get("answer_type", "textual")).strip().lower()

        if not question or not expected:
            raise ValueError(
                f"{question_ref} has an empty question or expected answer."
            )

        image_paths = resolve_image_paths(sample, image_root)

        content: list[dict[str, Any]] = [
            {"type": "image"} for _ in image_paths
        ]
        content.append(
            {"type": "text", "text": build_frieda_prompt(question)}
        )

        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                # Store paths first. The datasets Image feature decodes to PIL.
                "images": [str(path) for path in image_paths],
                "expected_answer": expected,
                "answer_type": answer_type,
                "question_ref": question_ref,
                "spatial_relationship": str(
                    sample.get("spatial_relationship", "")
                ),
                "map_count": str(sample.get("map_count", "")),
            }
        )

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("images", Sequence(HFImage()))
    return dataset


# ============================================================
# Completion extraction
# ============================================================

def completion_to_text(completion: Any) -> str:
    if completion is None:
        return ""

    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(str(block.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()

    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return completion_to_text(content)

    return str(completion).strip()


def extract_final_answer(raw_response: Any) -> str:
    text = completion_to_text(raw_response)
    if not text:
        return ""

    matches = list(
        re.finditer(
            r"final\s+answer\s*:\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return ""

    answer = text[matches[-1].end():].strip()
    answer = re.split(
        r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    # Keep only the first non-empty answer line.
    for line in answer.splitlines():
        if line.strip():
            return line.strip()

    return ""


def has_exactly_one_final_answer(raw_response: Any) -> bool:
    text = completion_to_text(raw_response)
    matches = re.findall(
        r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$",
        text,
    )
    return len(matches) == 1 and bool(matches[0].strip())


# ============================================================
# Normalization
# ============================================================

def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.casefold()
    value = value.replace("–", "-").replace("—", "-")
    value = value.replace("&", " and ")
    value = re.sub(r"[\"'`´“”‘’]", "", value)
    value = re.sub(r"[^a-z0-9./+\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


CARDINAL_ALIASES = {
    "n": "north",
    "north": "north",
    "ne": "north east",
    "northeast": "north east",
    "north east": "north east",
    "e": "east",
    "east": "east",
    "se": "south east",
    "southeast": "south east",
    "south east": "south east",
    "s": "south",
    "south": "south",
    "sw": "south west",
    "southwest": "south west",
    "south west": "south west",
    "w": "west",
    "west": "west",
    "nw": "north west",
    "northwest": "north west",
    "north west": "north west",
}

CARDINAL_INDEX = {
    "north": 0,
    "north east": 1,
    "east": 2,
    "south east": 3,
    "south": 4,
    "south west": 5,
    "west": 6,
    "north west": 7,
}


def canonicalize_cardinal(text: Any) -> str:
    normalized = normalize_text(text).replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in CARDINAL_ALIASES:
        return CARDINAL_ALIASES[normalized]

    # Accept a direction embedded in a short answer.
    for alias in sorted(CARDINAL_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return CARDINAL_ALIASES[alias]

    return ""


def cardinal_reward_value(prediction: str, expected: str) -> float:
    pred = canonicalize_cardinal(prediction)
    gold = canonicalize_cardinal(expected)

    if not pred or not gold:
        return 0.0

    difference = abs(CARDINAL_INDEX[pred] - CARDINAL_INDEX[gold])
    cyclic_difference = min(difference, 8 - difference)

    if cyclic_difference == 0:
        return 1.0
    if cyclic_difference == 1:
        # Matches the existing FRIEDA evaluation protocol:
        # the gold direction and its two adjacent directions are accepted.
        return 0.9
    return 0.0


# ============================================================
# Distance reward
# ============================================================

DISTANCE_UNIT_TO_METERS = {
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


def parse_distance_to_meters(text: Any) -> Optional[float]:
    normalized = normalize_text(text)
    normalized = normalized.replace(",", "")

    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*"
        r"(kilometers?|kilometres?|km|meters?|metres?|m|"
        r"miles?|mi|feet|foot|ft)\b",
        normalized,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    factor = DISTANCE_UNIT_TO_METERS.get(unit)
    if factor is None:
        return None

    return value * factor


def distance_reward_value(prediction: str, expected: str) -> float:
    pred_m = parse_distance_to_meters(prediction)
    gold_m = parse_distance_to_meters(expected)

    if pred_m is None or gold_m is None or gold_m == 0:
        return 0.0

    relative_error = abs(pred_m - gold_m) / abs(gold_m)

    if relative_error <= 0.20:
        return 1.0
    if relative_error <= 0.35:
        return 0.5
    if relative_error <= 0.50:
        return 0.2
    return 0.0


# ============================================================
# Textual and list reward
# ============================================================

def split_semicolon_list(text: Any) -> list[str]:
    raw = str(text or "")
    if ";" not in raw:
        return []

    return [
        normalize_text(item)
        for item in raw.split(";")
        if normalize_text(item)
    ]


def multiset_f1(predicted: Iterable[str], expected: Iterable[str]) -> float:
    pred_counter = Counter(predicted)
    gold_counter = Counter(expected)

    pred_total = sum(pred_counter.values())
    gold_total = sum(gold_counter.values())

    if pred_total == 0 or gold_total == 0:
        return 0.0

    overlap = sum(
        min(pred_counter[item], gold_counter[item])
        for item in pred_counter.keys() | gold_counter.keys()
    )

    precision = overlap / pred_total
    recall = overlap / gold_total

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def textual_reward_value(prediction: str, expected: str) -> float:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(expected)

    if not pred_norm:
        return 0.0

    if pred_norm == gold_norm:
        return 1.0

    expected_items = split_semicolon_list(expected)
    if expected_items:
        predicted_items = split_semicolon_list(prediction)
        if not predicted_items:
            # Allow "A and B" only when the gold is a short semicolon list.
            candidate = re.split(r"\s+and\s+", pred_norm)
            predicted_items = [
                normalize_text(item)
                for item in candidate
                if normalize_text(item)
            ]

        score = multiset_f1(predicted_items, expected_items)
        # Preserve exact=1.0. Partial list overlap receives a shaped reward.
        return round(score, 4)

    # Ordinary single textual answers remain deterministic.
    # LLM-as-a-Judge should be used offline, not inside the online GRPO loop.
    return 0.0


def correctness_reward_value(
    prediction: str,
    expected: str,
    answer_type: str,
) -> float:
    kind = str(answer_type or "textual").strip().lower()

    if kind == "cardinal":
        return cardinal_reward_value(prediction, expected)

    if kind == "distance":
        return distance_reward_value(prediction, expected)

    return textual_reward_value(prediction, expected)


# ============================================================
# Format and behavior rewards
# ============================================================

def repeated_ngram_ratio(text: Any, n: int = 4) -> float:
    normalized = normalize_text(completion_to_text(text))
    tokens = normalized.split()

    if len(tokens) < n:
        return 0.0

    ngrams = [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]
    counts = Counter(ngrams)
    repeated_occurrences = sum(
        count - 1 for count in counts.values() if count > 1
    )
    return repeated_occurrences / max(len(ngrams), 1)


def frieda_correctness_reward(
    completions: list[Any],
    expected_answer: list[str],
    answer_type: list[str],
    **kwargs: Any,
) -> list[float]:
    rewards: list[float] = []

    for completion, expected, kind in zip(
        completions,
        expected_answer,
        answer_type,
    ):
        prediction = extract_final_answer(completion)
        rewards.append(
            correctness_reward_value(prediction, expected, kind)
        )

    return rewards


def final_answer_format_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    return [
        FORMAT_REWARD if has_exactly_one_final_answer(completion) else 0.0
        for completion in completions
    ]


def behavior_reward(
    completions: list[Any],
    **kwargs: Any,
) -> list[float]:
    rewards: list[float] = []

    for completion in completions:
        text = completion_to_text(completion)
        has_final = has_exactly_one_final_answer(text)

        if not has_final:
            rewards.append(NO_FINAL_ANSWER_PENALTY)
            continue

        ratio = repeated_ngram_ratio(text, n=4)
        if ratio >= SEVERE_REPETITION_THRESHOLD:
            rewards.append(SEVERE_REPETITION_PENALTY)
        else:
            rewards.append(0.0)

    return rewards


# ============================================================
# Model and training
# ============================================================

def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training script.")

    props = torch.cuda.get_device_properties(0)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA: {torch.version.cuda}")


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_size: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["qa_json"] = str(Path(args.qa_json).resolve())
    payload["image_root"] = str(Path(args.image_root).resolve())
    payload["output_dir"] = str(Path(args.output_dir).resolve())
    payload["dataset_size"] = dataset_size

    with (output_dir / "run_config.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def train(args: argparse.Namespace) -> Path:
    print_gpu_info()

    qa_json = Path(args.qa_json).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing FRIEDA GRPO dataset...")
    train_dataset = build_training_dataset(qa_json, image_root)
    print(f"Training prompts: {len(train_dataset)}")
    print(f"Dataset columns:  {train_dataset.column_names}")

    save_run_config(output_dir, args, len(train_dataset))

    print("\nLoading Qwen3-VL with Unsloth...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        use_gradient_checkpointing="unsloth",
    )

    # In the current Unsloth VLM GRPO implementation, the effective device
    # batch is expected to be a multiple of num_generations.
    per_device_batch = args.num_generations

    training_args = GRPOConfig(
        output_dir=str(output_dir),

        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=0.1,

        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        report_to="wandb",
        run_name=args.wandb_run_name,

        log_completions=args.log_completions,
        num_completions_to_print=2,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=False,

        loss_type="dr_grpo",
        mask_truncated_completions=True,
        scale_rewards="group",
        beta=args.beta,

        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,

        eval_strategy="no",
        load_best_model_at_end=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        reward_funcs=[
            frieda_correctness_reward,
            final_answer_format_reward,
            behavior_reward,
        ],
        train_dataset=train_dataset,
    )

    resume = args.resume_from_checkpoint
    if resume:
        resume = str(Path(resume).expanduser().resolve())

    print("\nStarting GRPO training...")
    print(f"num_generations:       {args.num_generations}")
    print(f"temperature:           {args.temperature}")
    print(f"top_p:                 {args.top_p}")
    print(f"max completion length: {args.max_completion_length}")
    print(f"epochs:                {args.num_train_epochs}")
    print(f"max_steps override:    {args.max_steps}")
    print(f"save every steps:      {args.save_steps}")

    result = trainer.train(resume_from_checkpoint=resume)

    trainer.save_state()

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    metrics = dict(result.metrics)
    metrics["final_adapter_dir"] = str(final_adapter_dir)

    with (output_dir / "train_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\nTraining complete.")
    print(f"Final adapter: {final_adapter_dir}")
    print(f"Checkpoints:   {output_dir / 'checkpoint-*'}")
    return final_adapter_dir


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsloth GRPO training for the FRIEDA train split."
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=MAX_PROMPT_LENGTH,
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=MAX_COMPLETION_LENGTH,
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
    )
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)

    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=NUM_TRAIN_EPOCHS,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value overrides num_train_epochs; use 1-3 for a smoke test.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
    )
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="VLM-Cartographic-GRPO",
        help="Weights & Biases project name.",
    )

    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="FRIEDA-GRPO-run1",
        help="Weights & Biases run name.",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B username or team name.",
    )

    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B logging mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_generations < 2:
        raise ValueError("num_generations must be at least 2.")
    if not 0 < args.temperature:
        raise ValueError("temperature must be greater than 0.")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    if args.max_completion_length < 128:
        raise ValueError("max_completion_length is unreasonably small.")

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ["WANDB_LOG_MODEL"] = "false"

    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    train(args)



if __name__ == "__main__":
    main()