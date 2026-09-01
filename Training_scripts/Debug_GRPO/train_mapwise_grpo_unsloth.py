#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MapWise Qwen3-VL-8B-Thinking GRPO training with Unsloth.

- 4-bit QLoRA base model.
- Only Vision LoRA scopes enabled at first.
- 4 rollouts per prompt, temperature=0.8, top_p=0.95.
- Correctness reward reuses mapwise_evaluation.py and ONLY strict_exact_match.
(- Format reward: +0.05 for exactly one non-empty ``Final answer:`` line.
(- Behavior reward: -0.10 for severe repeated 4-gram reasoning-loop behavior.
- No extra missing-final-answer penalty.
- W&B logging and checkpoint saving retained.
- Validation remains offline/deterministic; no best-checkpoint selection in GRPO.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import unicodedata

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

# Unsloth must be imported before transformers / trl.
from unsloth import FastVisionModel

import torch
from datasets import Dataset, Image as HFImage, Sequence
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 1. Defaults
# ============================================================

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"

MAX_SEQ_LENGTH = 16384
MAX_PROMPT_LENGTH = 8192
MAX_COMPLETION_LENGTH = 1536

NUM_GENERATIONS = 4
TEMPERATURE = 0.8
TOP_P = 0.95

LORA_RANK = 16
LORA_ALPHA = 16

LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1.0
SAVE_STEPS = 5
SEED = 3407

# FORMAT_REWARD = 0.05
# SEVERE_REPETITION_PENALTY = -0.10
# SEVERE_REPETITION_THRESHOLD = 0.45
# REPETITION_NGRAM = 4

SUPPORTED_COUNTRIES = {"china", "india", "usa"}
SUPPORTED_IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_QA_JSON = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation\Datasets\Processed_Mapwise\Train_Val\mapwise_grpo_train.json"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation\Training_outputs\MapWise_GRPO_Qwen3-VL-8B-Thinking_only_visLoRA"
)
DEFAULT_IMAGE_ROOT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation\Datasets\mapwise-dataset"
)

DEFAULT_EVALUATION_SCRIPT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation\Evaluation_scripts\mapwise_evaluation.py"
)


# ============================================================
# 2. MapWise evaluator
# ============================================================

def load_evaluation_module(evaluation_script: Path):
    evaluation_script = evaluation_script.expanduser().resolve()
    if not evaluation_script.is_file():
        raise FileNotFoundError(
            f"MapWise evaluation script not found:\n{evaluation_script}\n"
            "Use --evaluation-script to point to mapwise_evaluation.py."
        )

    spec = importlib.util.spec_from_file_location(
        "mapwise_evaluation_for_grpo", evaluation_script
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator: {evaluation_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_sample"):
        raise AttributeError("MapWise evaluator must expose evaluate_sample(record).")

    return module


# ============================================================
# 3. Prompt
# ============================================================

def build_mapwise_prompt(question: str) -> str:
    """Stable benchmark prompt. ground_truth_type is never shown to the model."""
    return f"""
This is a cartographic reasoning question from the MapWise dataset.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()


# ============================================================
# 4. Dataset and image resolution
# ============================================================

def load_json_list(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training JSON does not exist:\n{path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Training JSON must contain a non-empty top-level list.")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise TypeError(f"Sample {index} is not a dictionary.")

        row = dict(item)
        required = (
            "country", "map_no", "template_no", "question",
            "ground_truth", "ground_truth_type",
        )
        missing = [key for key in required if key not in row]
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


def resolve_mapwise_image(sample: Mapping[str, Any], image_root: Path) -> Path:
    """Resolve <root>/<country>/images/with_annotations/<map_no>.<suffix>."""
    country = str(sample.get("country", "")).strip().lower()
    map_no = str(sample.get("map_no", "")).strip()
    qa_id = str(sample.get("qa_id", "unknown"))

    country_root = (
        image_root.expanduser().resolve()
        / country / "images" / "with_annotations"
    )
    if not country_root.is_dir():
        raise FileNotFoundError(f"Country image directory does not exist:\n{country_root}")

    raw = Path(map_no)
    if raw.suffix:
        candidate = country_root / raw
        if candidate.is_file():
            return candidate.resolve()

    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = country_root / f"{map_no}{suffix}"
        if candidate.is_file():
            return candidate.resolve()

    target_stem = raw.stem.casefold()
    matches = [
        p for p in country_root.rglob("*")
        if p.is_file()
        and p.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        and p.stem.casefold() == target_stem
    ]

    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple images match qa_id={qa_id}, map_no={map_no!r}:\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        f"No image found for qa_id={qa_id}, country={country}, map_no={map_no!r} "
        f"under:\n{country_root}"
    )


def build_training_dataset(qa_json: Path, image_root: Path) -> Dataset:
    samples = load_json_list(qa_json)
    image_root = image_root.expanduser().resolve()
    rows: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        qa_id = str(sample["qa_id"])
        question = str(sample.get("question", "")).strip()
        ground_truth = str(sample.get("ground_truth", "")).strip()
        ground_truth_type = str(sample.get("ground_truth_type", "")).strip()

        if not question or not ground_truth:
            raise ValueError(f"{qa_id} has an empty question or ground_truth.")

        image_path = resolve_mapwise_image(sample, image_root)

        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": build_mapwise_prompt(question)},
                        ],
                    }
                ],
                "images": [str(image_path)],
                "qa_id": qa_id,
                "country": str(sample.get("country", "")),
                "map_no": str(sample.get("map_no", "")),
                "template_no": int(sample.get("template_no", -1)),
                "ground_truth": ground_truth,
                "ground_truth_type": ground_truth_type,
                "legend_style": str(
                    sample.get("legend_style", sample.get("c_or_d", ""))
                ),
                "c_or_d": str(sample.get("c_or_d", "")),
                "relative_region": str(sample.get("relative_region", "")),
                "corrected_template_no": sample.get("corrected_template_no"),
                "ability_level": str(sample.get("ability_level", "")),
                "map_family": str(sample.get("map_family", "")),
                "source_index": int(sample.get("source_index", index)),
            }
        )

    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("images", Sequence(HFImage()))
    return dataset


# ============================================================
# 5. Completion handling
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


def has_exactly_one_final_answer(raw_response: Any) -> bool:
    text = completion_to_text(raw_response)
    matches = re.findall(
        r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$",
        text,
    )
    return len(matches) == 1 and bool(matches[0].strip())


# ============================================================
# 6. Severe repetition / reasoning-loop detection
# ============================================================

# def normalize_for_repetition(text: Any) -> str:
#     value = unicodedata.normalize("NFKC", completion_to_text(text)).casefold()
#     value = value.replace("–", "-").replace("—", "-").replace("−", "-")
#     value = re.sub(r"[^a-z0-9<>=%+\-./]+", " ", value)
#     return re.sub(r"\s+", " ", value).strip()


# def repeated_ngram_ratio(text: Any, n: int = REPETITION_NGRAM) -> float:
#     normalized = normalize_for_repetition(text)
#     tokens = normalized.split()
#     if len(tokens) < n:
#         return 0.0

#     ngrams = [
#         tuple(tokens[i:i + n])
#         for i in range(len(tokens) - n + 1)
#     ]
#     counts = Counter(ngrams)
#     repeated_occurrences = sum(
#         count - 1 for count in counts.values() if count > 1
#     )
#     return repeated_occurrences / max(len(ngrams), 1)


# ============================================================
# 7. Reward functions
# ============================================================

MAPWISE_EVALUATOR = None
LAST_SAMPLE_TRACE_PATH = None


def _as_list(value: Any, expected_length: int) -> list[Any]:
    if isinstance(value, list):
        if len(value) == expected_length:
            return value
        if len(value) == 1:
            return value * expected_length
        return value
    return [value] * expected_length


def mapwise_correctness_reward(
    completions: list[Any],
    ground_truth: list[str],
    ground_truth_type: list[str],
    country: list[str],
    template_no: list[int],
    legend_style: Optional[list[str]] = None,
    c_or_d: Optional[list[str]] = None,
    qa_id: Optional[list[str]] = None,
    **kwargs: Any,
) -> list[float]:
    """Exact correctness reward using evaluator.strict_exact_match only."""
    if MAPWISE_EVALUATOR is None:
        raise RuntimeError("MapWise evaluator was not initialized.")

    n = len(completions)
    golds = _as_list(ground_truth, n)
    kinds = _as_list(ground_truth_type, n)
    countries = _as_list(country, n)
    templates = _as_list(template_no, n)

    if LAST_SAMPLE_TRACE_PATH is not None:
        map_nos = _as_list(kwargs.get("map_no", ""), n)
        source_indices = _as_list(kwargs.get("source_index", -1), n)

        trace = {
            "qa_id": str(_as_list(qa_id if qa_id is not None else "", n)[0]),
            "country": str(countries[0]),
            "map_no": str(map_nos[0]),
            "template_no": int(templates[0]),
            "source_index": int(source_indices[0]),
            "ground_truth": str(golds[0]),
            "ground_truth_type": str(kinds[0]),
        }

        with LAST_SAMPLE_TRACE_PATH.open("w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

    legends = _as_list(legend_style if legend_style is not None else "", n)
    cds = _as_list(c_or_d if c_or_d is not None else "", n)
    qa_ids = _as_list(qa_id if qa_id is not None else "", n)

    rewards: list[float] = []

    for completion, gold, kind, ctry, template, legend, cd, item_id in zip(
        completions, golds, kinds, countries, templates, legends, cds, qa_ids
    ):
        record = {
            "qa_id": str(item_id or ""),
            "country": str(ctry or ""),
            "ground_truth": str(gold or ""),
            "ground_truth_type": str(kind or ""),
            # IMPORTANT: preserve raw template_no exactly as stored.
            "template_no": int(template),
            "legend_style": str(legend or cd or ""),
            "c_or_d": str(cd or ""),
            "raw_response": completion_to_text(completion),
            "final_answer": "",
        }
        result = MAPWISE_EVALUATOR.evaluate_sample(record)
        rewards.append(
            1.0 if int(result.get("strict_exact_match", 0)) == 1 else 0.0
        )

    return rewards


# def final_answer_format_reward(
#     completions: list[Any],
#     **kwargs: Any,
# ) -> list[float]:
#     return [
#         FORMAT_REWARD if has_exactly_one_final_answer(c) else 0.0
#         for c in completions
#     ]


# def reasoning_behavior_reward(
#     completions: list[Any],
#     **kwargs: Any,
# ) -> list[float]:
#     rewards: list[float] = []
#     for completion in completions:
#         ratio = repeated_ngram_ratio(completion, n=REPETITION_NGRAM)
#         rewards.append(
#             SEVERE_REPETITION_PENALTY
#             if ratio >= SEVERE_REPETITION_THRESHOLD
#             else 0.0
#         )
#     return rewards


# Readable names in TRL/W&B reward logging.
mapwise_correctness_reward.__name__ = "correctness"
# final_answer_format_reward.__name__ = "final_answer_format"
# reasoning_behavior_reward.__name__ = "reasoning_behavior"


# ============================================================
# 8. Utilities
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

    for key in (
        "qa_json", "image_root", "output_dir",
        "evaluation_script", "resume_from_checkpoint",
    ):
        value = payload.get(key)
        if value is not None:
            payload[key] = str(Path(value).expanduser().resolve())

    payload["dataset_size"] = dataset_size
    payload["reward_design"] = {
        "strict_exact_correct": 1.0,
        # "valid_final_answer_format": FORMAT_REWARD,
        # "severe_reasoning_loop": SEVERE_REPETITION_PENALTY,
        # "repetition_threshold": SEVERE_REPETITION_THRESHOLD,
        # "repetition_ngram": REPETITION_NGRAM,
    }
    payload["lora_targets"] = {
        "finetune_vision_layers": True,
        "finetune_language_layers": False,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


# ============================================================
# 9. Training
# ============================================================

def train(args: argparse.Namespace) -> Path:
    global MAPWISE_EVALUATOR, LAST_SAMPLE_TRACE_PATH

    print_gpu_info()

    qa_json = Path(args.qa_json).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    evaluation_script = Path(args.evaluation_script).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LAST_SAMPLE_TRACE_PATH = output_dir / "last_grpo_sample.json"

    print("\nLoading MapWise evaluator...")
    print(f"Evaluator: {evaluation_script}")
    MAPWISE_EVALUATOR = load_evaluation_module(evaluation_script)

    print("\nPreparing MapWise GRPO dataset...")
    train_dataset = build_training_dataset(qa_json, image_root)
    print(f"Training prompts: {len(train_dataset)}")
    print(f"Dataset columns:  {train_dataset.column_names}")

    ability_counts = Counter(
        str(v) for v in train_dataset["ability_level"] if str(v).strip()
    )
    answer_type_counts = Counter(str(v) for v in train_dataset["ground_truth_type"])
    if ability_counts:
        print(f"Ability levels:   {dict(ability_counts)}")
    print(f"Answer types:     {dict(answer_type_counts)}")

    save_run_config(output_dir, args, len(train_dataset))

    print("\nLoading Qwen3-VL-8B-Thinking with Unsloth...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=False,
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

    # Keep a full rollout group in the device batch, matching the reference script.
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
            mapwise_correctness_reward,
            # final_answer_format_reward,
            # reasoning_behavior_reward,
        ],
        train_dataset=train_dataset,
    )

    resume = args.resume_from_checkpoint
    if resume is not None:
        resume = str(Path(resume).expanduser().resolve())

    print("\nStarting MapWise GRPO training...")
    print(f"Model:                 {args.model_name}")
    print(f"num_generations:       {args.num_generations}")
    print(f"temperature:           {args.temperature}")
    print(f"top_p:                 {args.top_p}")
    print(f"max completion length: {args.max_completion_length}")
    print(f"learning rate:         {args.learning_rate}")
    print(f"epochs:                {args.num_train_epochs}")
    print(f"max_steps override:    {args.max_steps}")
    print(f"save every steps:      {args.save_steps}")
    print("LoRA:                  vision=True, language=False, attention=True, mlp=True")
    print("Rewards:               exact=+1.00") #, format=+0.05, severe_loop=-0.10")

    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_state()

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    processor.save_pretrained(final_adapter_dir)

    metrics = dict(result.metrics)
    metrics["final_adapter_dir"] = str(final_adapter_dir)
    metrics["num_generations"] = args.num_generations
    metrics["temperature"] = args.temperature
    metrics["top_p"] = args.top_p

    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("\nTraining complete.")
    print(f"Final adapter: {final_adapter_dir}")
    print(f"Checkpoints:   {output_dir / 'checkpoint-*'}")
    return final_adapter_dir


# ============================================================
# 10. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsloth GRPO training for MapWise with Qwen3-VL-8B-Thinking."
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--qa-json", type=Path, default=DEFAULT_QA_JSON)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-script", type=Path, default=DEFAULT_EVALUATION_SCRIPT
    )

    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--max-prompt-length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument(
        "--max-completion-length", type=int, default=MAX_COMPLETION_LENGTH
    )

    parser.add_argument("--num-generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)

    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-train-epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value overrides num_train_epochs; use 1-3 for smoke tests.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--save-total-limit", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.0)

    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--log-completions", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)

    parser.add_argument(
        "--wandb-project", type=str, default="VLM-Cartographic-GRPO"
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default="MapWise-GRPO-Thinking-run1"
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_generations < 2:
        raise ValueError("num_generations must be at least 2.")
    if args.temperature <= 0:
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
    try:
        main()
    except Exception as error:
        print("\nProgram failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise
