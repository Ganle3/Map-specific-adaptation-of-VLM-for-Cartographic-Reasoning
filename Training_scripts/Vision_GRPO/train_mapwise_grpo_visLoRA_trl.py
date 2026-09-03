#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Formal MapWise Native Vision-only GRPO training.

Stack
-----
Qwen/Qwen3-VL-8B-Thinking
+ Hugging Face Transformers
+ bitsandbytes 4-bit NF4
+ PEFT Vision-only LoRA
+ TRL GRPOTrainer

Design goal
-----------
Keep the formal experiment as close as possible to the successful native
mini-pilot while changing only what is necessary for a full 1-epoch run.

Reward
------
Correctness only:
    strict exact match -> 1.0
    otherwise          -> 0.0

Important batching choice
-------------------------
num_generations = 4
per_device_train_batch_size = 1
gradient_accumulation_steps = 4

With current TRL, this gives an effective training batch of 4 completions,
which is divisible by num_generations=4. Therefore one 4-rollout GRPO group
is accumulated into one optimizer update instead of performing four optimizer
updates on the same generation batch.

For a dataset with N QA prompts, one epoch is therefore approximately N
optimizer steps, matching the intended "one QA group -> one optimizer step"
interpretation.

No Unsloth.
No vLLM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from datasets import Dataset, Image as HFImage, Sequence

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)


# ============================================================
# 1. Defaults
# ============================================================

MODEL_NAME = "Qwen/Qwen3-VL-8B-Thinking"

# LoRA
LORA_RANK = 16
LORA_ALPHA = 16

# GRPO
NUM_GENERATIONS = 4
MAX_COMPLETION_LENGTH = 1536
TEMPERATURE = 0.8
TOP_P = 0.95
BETA = 0.0
LOSS_TYPE = "dr_grpo"

# Training
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1.0
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.1
WARMUP_RATIO_FOR_STEP_CALCULATION = 0.10

SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LOGGING_STEPS = 1

SEED = 3407


# ============================================================
# 2. Paths
# ============================================================

DEFAULT_QA_JSON = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\Processed_Mapwise\Train_Val"
    r"\mapwise_grpo_train.json"
)

DEFAULT_IMAGE_ROOT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\mapwise-dataset"
)

DEFAULT_EVALUATION_SCRIPT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Evaluation_scripts"
    r"\mapwise_evaluation_exact.py"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Training_outputs"
    r"\MapWise_GRPO_Qwen3-VL-8B-Thinking_visLoRA_TRL"
)


# ============================================================
# 3. MapWise constants
# ============================================================

SUPPORTED_COUNTRIES = {
    "china",
    "india",
    "usa",
}

SUPPORTED_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".bmp",
)


# ============================================================
# 4. Vision-only LoRA scope
# ============================================================

# Verified native Qwen3-VL visual block structure:
#
# model.visual.blocks.0.attn.qkv
# model.visual.blocks.0.attn.proj
# model.visual.blocks.0.mlp.linear_fc1
# model.visual.blocks.0.mlp.linear_fc2
# ...
# model.visual.blocks.26.*
#
# Intentionally NOT targeting:
# model.visual.merger.*
# model.visual.deepstack_merger_list.*
#
# This preserves the same visual-block LoRA scope used in the diagnostic.

VISION_TARGET_REGEX = (
    r"^model\.visual\.blocks\.\d+\."
    r"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))$"
)

EXPECTED_VISION_BLOCKS = 27
EXPECTED_MODULES_PER_BLOCK = 4
EXPECTED_TARGET_MODULES = (
    EXPECTED_VISION_BLOCKS * EXPECTED_MODULES_PER_BLOCK
)
EXPECTED_LORA_TENSORS = EXPECTED_TARGET_MODULES * 2


# ============================================================
# 5. Globals
# ============================================================

MAPWISE_EVALUATOR = None
REWARD_CALL_COUNT = 0


# ============================================================
# 6. Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_info() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    props = torch.cuda.get_device_properties(0)

    print("=" * 88)
    print("MapWise Native Vision-only GRPO")
    print("=" * 88)
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print("=" * 88)


# ============================================================
# 7. MapWise evaluator
# ============================================================

def load_evaluation_module(evaluation_script: Path):
    evaluation_script = evaluation_script.expanduser().resolve()

    if not evaluation_script.is_file():
        raise FileNotFoundError(
            "MapWise evaluation script not found:\n"
            f"{evaluation_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "mapwise_evaluation_for_native_grpo",
        evaluation_script,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load evaluator:\n{evaluation_script}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "evaluate_sample"):
        raise AttributeError(
            "MapWise evaluator must expose evaluate_sample(record)."
        )

    return module


# ============================================================
# 8. Prompt
# ============================================================

def build_mapwise_prompt(question: str) -> str:
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
# 9. JSON loading
# ============================================================

def load_json_list(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Training JSON does not exist:\n{path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError(
            "Training JSON must contain a non-empty top-level list."
        )

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"Sample {index} is not a dictionary."
            )

        row = dict(item)

        required = (
            "country",
            "map_no",
            "template_no",
            "question",
            "ground_truth",
            "ground_truth_type",
        )

        missing = [
            key for key in required
            if key not in row
        ]

        if missing:
            raise KeyError(
                f"Sample {index} missing fields: {missing}"
            )

        country = str(row.get("country", "")).strip().lower()

        if country not in SUPPORTED_COUNTRIES:
            raise ValueError(
                f"Sample {index} has unsupported country={country!r}."
            )

        if not str(row.get("qa_id", "")).strip():
            source_index = int(
                row.get("source_index", index)
            )

            row["qa_id"] = (
                f"mapwise_{country}_"
                f"{row['map_no']}_"
                f"t{int(row['template_no'])}_"
                f"src{source_index:04d}"
            )

        rows.append(row)

    return rows


# ============================================================
# 10. Image resolution
# ============================================================

def resolve_mapwise_image(
    sample: Mapping[str, Any],
    image_root: Path,
) -> Path:

    country = str(
        sample.get("country", "")
    ).strip().lower()

    map_no = str(
        sample.get("map_no", "")
    ).strip()

    qa_id = str(
        sample.get("qa_id", "unknown")
    )

    country_root = (
        image_root.expanduser().resolve()
        / country
        / "images"
        / "with_annotations"
    )

    if not country_root.is_dir():
        raise FileNotFoundError(
            "Country image directory does not exist:\n"
            f"{country_root}"
        )

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
        p
        for p in country_root.rglob("*")
        if (
            p.is_file()
            and p.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
            and p.stem.casefold() == target_stem
        )
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple images match qa_id={qa_id}, "
            f"map_no={map_no!r}:\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        f"No image found for qa_id={qa_id}, "
        f"country={country}, map_no={map_no!r} under:\n"
        f"{country_root}"
    )


# ============================================================
# 11. Full training dataset
# ============================================================

def build_train_dataset(
    qa_json: Path,
    image_root: Path,
) -> Dataset:

    samples = load_json_list(qa_json)
    rows: list[dict[str, Any]] = []

    print("\nPreparing full MapWise GRPO training dataset...")

    for index, sample in enumerate(samples):
        qa_id = str(sample["qa_id"])

        question = str(
            sample.get("question", "")
        ).strip()

        ground_truth = str(
            sample.get("ground_truth", "")
        ).strip()

        ground_truth_type = str(
            sample.get("ground_truth_type", "")
        ).strip()

        if not question:
            raise ValueError(
                f"{qa_id} has an empty question."
            )

        if not ground_truth:
            raise ValueError(
                f"{qa_id} has an empty ground_truth."
            )

        image_path = resolve_mapwise_image(
            sample,
            image_root,
        )

        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": build_mapwise_prompt(
                                    question
                                ),
                            },
                        ],
                    }
                ],

                "images": [
                    str(image_path)
                ],

                "qa_id": qa_id,

                "country": str(
                    sample.get("country", "")
                ),

                "map_no": str(
                    sample.get("map_no", "")
                ),

                "template_no": int(
                    sample.get("template_no", -1)
                ),

                "ground_truth": ground_truth,

                "ground_truth_type": ground_truth_type,

                "legend_style": str(
                    sample.get(
                        "legend_style",
                        sample.get("c_or_d", ""),
                    )
                ),

                "c_or_d": str(
                    sample.get("c_or_d", "")
                ),

                "source_index": int(
                    sample.get("source_index", index)
                ),
            }
        )

    dataset = Dataset.from_list(rows)

    dataset = dataset.cast_column(
        "images",
        Sequence(HFImage()),
    )

    unique_maps = len(
        set(
            zip(
                dataset["country"],
                dataset["map_no"],
            )
        )
    )

    print(f"Training QA:   {len(dataset)}")
    print(f"Unique maps:   {unique_maps}")
    print(f"Columns:       {dataset.column_names}")

    return dataset


# ============================================================
# 12. Completion -> text
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
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            parts.append(
                                str(block.get("text", ""))
                            )
            else:
                parts.append(str(item))

        return "\n".join(
            part for part in parts if part
        ).strip()

    if isinstance(completion, dict):
        content = completion.get("content", "")

        if isinstance(content, str):
            return content.strip()

        return completion_to_text(content)

    return str(completion).strip()


def _as_list(
    value: Any,
    expected_length: int,
) -> list[Any]:

    if isinstance(value, list):
        if len(value) == expected_length:
            return value

        if len(value) == 1:
            return value * expected_length

        return value

    return [value] * expected_length


# ============================================================
# 13. Correctness-only reward
# ============================================================

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

    """
    Formal reward for this experiment.

    strict exact correct -> 1.0
    otherwise            -> 0.0

    No format bonus.
    No repetition penalty.
    No completion reward.
    """

    global MAPWISE_EVALUATOR, REWARD_CALL_COUNT

    if MAPWISE_EVALUATOR is None:
        raise RuntimeError(
            "MapWise evaluator was not initialized."
        )

    REWARD_CALL_COUNT += 1

    n = len(completions)

    golds = _as_list(ground_truth, n)
    kinds = _as_list(ground_truth_type, n)
    countries = _as_list(country, n)
    templates = _as_list(template_no, n)

    legends = _as_list(
        legend_style if legend_style is not None else "",
        n,
    )

    cds = _as_list(
        c_or_d if c_or_d is not None else "",
        n,
    )

    qa_ids = _as_list(
        qa_id if qa_id is not None else "",
        n,
    )

    rewards: list[float] = []

    for (
        completion,
        gold,
        kind,
        ctry,
        template,
        legend,
        cd,
        item_id,
    ) in zip(
        completions,
        golds,
        kinds,
        countries,
        templates,
        legends,
        cds,
        qa_ids,
    ):

        record = {
            "qa_id": str(item_id or ""),
            "country": str(ctry or ""),
            "ground_truth": str(gold or ""),
            "ground_truth_type": str(kind or ""),
            "template_no": int(template),
            "legend_style": str(legend or cd or ""),
            "c_or_d": str(cd or ""),
            "raw_response": completion_to_text(completion),
            "final_answer": "",
        }

        result = MAPWISE_EVALUATOR.evaluate_sample(
            record
        )

        reward = (
            1.0
            if int(
                result.get(
                    "strict_exact_match",
                    0,
                )
            ) == 1
            else 0.0
        )

        rewards.append(reward)

    first_qa = (
        str(qa_ids[0])
        if qa_ids
        else "unknown"
    )

    mean_reward = (
        sum(rewards) / max(len(rewards), 1)
    )

    has_variance = len(set(rewards)) > 1

    print(
        f"[REWARD {REWARD_CALL_COUNT:05d}] "
        f"qa={first_qa} | "
        f"rewards={rewards} | "
        f"mean={mean_reward:.2f} | "
        f"variance={'YES' if has_variance else 'NO'}"
    )

    return rewards


mapwise_correctness_reward.__name__ = "correctness"


# ============================================================
# 14. PEFT verification
# ============================================================

def verify_vision_only_lora(model) -> None:

    targeted = list(
        getattr(
            model,
            "targeted_module_names",
            [],
        )
    )

    trainable_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
    ]

    vision_trainable = [
        name
        for name in trainable_names
        if "visual" in name.lower()
    ]

    nonvision_trainable = [
        name
        for name in trainable_names
        if "visual" not in name.lower()
    ]

    print("\n")
    print("=" * 88)
    print("PEFT VISION-ONLY LoRA CHECK")
    print("=" * 88)
    print(f"Target modules:            {len(targeted)}")
    print(f"Trainable tensors:         {len(trainable_names)}")
    print(f"Vision trainable tensors:  {len(vision_trainable)}")
    print(f"Non-vision tensors:        {len(nonvision_trainable)}")

    if len(targeted) != EXPECTED_TARGET_MODULES:
        raise RuntimeError(
            "Unexpected LoRA target count: "
            f"expected {EXPECTED_TARGET_MODULES}, "
            f"got {len(targeted)}."
        )

    if len(trainable_names) != EXPECTED_LORA_TENSORS:
        raise RuntimeError(
            "Unexpected trainable tensor count: "
            f"expected {EXPECTED_LORA_TENSORS}, "
            f"got {len(trainable_names)}."
        )

    if nonvision_trainable:
        raise RuntimeError(
            "Non-vision trainable parameters found:\n"
            + "\n".join(nonvision_trainable)
        )

    bad_targets = [
        name
        for name in targeted
        if "visual" not in name.lower()
    ]

    if bad_targets:
        raise RuntimeError(
            "Non-vision LoRA targets found:\n"
            + "\n".join(bad_targets)
        )

    print("PASS: LoRA is strictly Vision-only.")
    print("=" * 88)
    print()


# ============================================================
# 15. Compact gradient / VRAM monitor
# ============================================================

class VisionTrainingMonitorCallback(
    TrainerCallback
):

    def __init__(
        self,
        model,
        output_dir: Path,
    ):
        self.model = model
        self.output_dir = output_dir
        self.metrics_path = (
            output_dir / "step_metrics.jsonl"
        )
        self.step_start_time: Optional[float] = None
        self.latest_gradient_stats: dict[str, Any] = {}

    def on_step_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        self.step_start_time = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        grad_none = 0
        grad_zero = 0
        grad_nonzero = 0
        squared_norm_sum = 0.0

        vision_trainable = 0
        nonvision_trainable = 0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "visual" in name.lower():
                vision_trainable += 1
            else:
                nonvision_trainable += 1

            if param.grad is None:
                grad_none += 1
                continue

            grad = param.grad.detach().float()
            grad_norm = grad.norm().item()

            if grad_norm == 0.0:
                grad_zero += 1
            else:
                grad_nonzero += 1
                squared_norm_sum += grad_norm * grad_norm

        self.latest_gradient_stats = {
            "vision_trainable": vision_trainable,
            "nonvision_trainable": nonvision_trainable,
            "grad_none": grad_none,
            "grad_zero": grad_zero,
            "grad_nonzero": grad_nonzero,
            "lora_grad_norm": squared_norm_sum ** 0.5,
        }

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        elapsed = None

        if self.step_start_time is not None:
            elapsed = (
                time.perf_counter()
                - self.step_start_time
            )

        payload: dict[str, Any] = {
            "global_step": int(state.global_step),
            "step_time_seconds": elapsed,
            **self.latest_gradient_stats,
        }

        if torch.cuda.is_available():
            gb = 1024 ** 3

            payload.update(
                {
                    "vram_allocated_gb":
                        torch.cuda.memory_allocated() / gb,
                    "vram_reserved_gb":
                        torch.cuda.memory_reserved() / gb,
                    "vram_peak_allocated_gb":
                        torch.cuda.max_memory_allocated() / gb,
                    "vram_peak_reserved_gb":
                        torch.cuda.max_memory_reserved() / gb,
                }
            )

        norm = payload.get(
            "lora_grad_norm",
            float("nan"),
        )

        print(
            f"\n[STEP {state.global_step:05d}] "
            f"grad none={payload.get('grad_none', '?')} "
            f"zero={payload.get('grad_zero', '?')} "
            f"nonzero={payload.get('grad_nonzero', '?')} "
            f"| LoRA norm={norm:.4e}"
        )

        if elapsed is not None:
            print(
                f"             time={elapsed:.2f}s"
            )

        if torch.cuda.is_available():
            print(
                "             VRAM "
                f"alloc={payload['vram_allocated_gb']:.2f} GB | "
                f"reserved={payload['vram_reserved_gb']:.2f} GB | "
                f"peak alloc={payload['vram_peak_allocated_gb']:.2f} GB | "
                f"peak reserved={payload['vram_peak_reserved_gb']:.2f} GB"
            )

        with self.metrics_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 16. Checkpoint helpers
# ============================================================

def find_latest_checkpoint(
    output_dir: Path,
) -> Optional[Path]:

    if not output_dir.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []

    pattern = re.compile(
        r"^checkpoint-(\d+)$"
    )

    for path in output_dir.iterdir():
        if not path.is_dir():
            continue

        match = pattern.match(path.name)

        if match:
            candidates.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[-1][1]


def resolve_resume_checkpoint(
    value: Optional[str],
    output_dir: Path,
) -> Optional[str]:

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    if value.lower() == "latest":
        latest = find_latest_checkpoint(
            output_dir
        )

        if latest is None:
            raise FileNotFoundError(
                "--resume-from-checkpoint latest was requested, "
                "but no checkpoint-* directory exists in:\n"
                f"{output_dir}"
            )

        return str(latest.resolve())

    path = Path(value).expanduser().resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory does not exist:\n{path}"
        )

    return str(path)


# ============================================================
# 17. Save run configuration
# ============================================================

def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_size: int,
    expected_optimizer_steps: int,
    warmup_steps: int,
) -> None:

    payload = vars(args).copy()

    for key in (
        "qa_json",
        "image_root",
        "evaluation_script",
        "output_dir",
    ):
        if key in payload:
            payload[key] = str(
                Path(payload[key])
                .expanduser()
                .resolve()
            )

    payload.update(
        {
            "dataset_size": dataset_size,
            "expected_optimizer_steps":
                expected_optimizer_steps,
            "warmup_steps": warmup_steps,
            "reward": "strict_exact_correctness_only",
            "reward_correct": 1.0,
            "reward_incorrect": 0.0,
            "format_reward": False,
            "loop_penalty": False,
            "vision_lora_target_modules":
                EXPECTED_TARGET_MODULES,
            "vision_lora_trainable_tensors":
                EXPECTED_LORA_TENSORS,
            "unsloth": False,
            "vllm": False,
        }
    )

    with (
        output_dir / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# 18. Main training
# ============================================================

def run_training(
    args: argparse.Namespace,
) -> Path:

    global MAPWISE_EVALUATOR

    set_seed(args.seed)
    print_gpu_info()

    qa_json = Path(
        args.qa_json
    ).expanduser().resolve()

    image_root = Path(
        args.image_root
    ).expanduser().resolve()

    evaluation_script = Path(
        args.evaluation_script
    ).expanduser().resolve()

    output_dir = Path(
        args.output_dir
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    print("\nLoading MapWise evaluator...")
    print(f"Evaluator: {evaluation_script}")

    MAPWISE_EVALUATOR = load_evaluation_module(
        evaluation_script
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = build_train_dataset(
        qa_json=qa_json,
        image_root=image_root,
    )

    # One optimizer step per original QA group under:
    #
    # per_device_train_batch_size = 1
    # gradient_accumulation_steps = num_generations = 4
    #
    # For one epoch, expected optimizer steps ~= dataset size.
    expected_optimizer_steps = math.ceil(
        (
            len(train_dataset)
            * args.num_train_epochs
        )
        / args.per_device_train_batch_size
    )

    warmup_steps = max(
        1,
        int(
            round(
                expected_optimizer_steps
                * args.warmup_fraction
            )
        ),
    )

    save_run_config(
        output_dir=output_dir,
        args=args,
        dataset_size=len(train_dataset),
        expected_optimizer_steps=expected_optimizer_steps,
        warmup_steps=warmup_steps,
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    print("\nLoading Qwen3-VL processor...")

    processor = AutoProcessor.from_pretrained(
        args.model_name
    )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = (
                processor.tokenizer.eos_token
            )

    # --------------------------------------------------------
    # 4-bit base model
    # --------------------------------------------------------

    print("\nLoading Qwen3-VL 4-bit base model...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = (
        AutoModelForImageTextToText
        .from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # QLoRA preparation
    # --------------------------------------------------------

    print("\nPreparing model for k-bit training...")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # --------------------------------------------------------
    # Vision-only LoRA
    # --------------------------------------------------------

    print("\nApplying Vision-only LoRA...")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=VISION_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    verify_vision_only_lora(model)

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    resume_checkpoint = (
        resolve_resume_checkpoint(
            args.resume_from_checkpoint,
            output_dir,
        )
    )

    if resume_checkpoint is not None:
        print(
            "\nResume checkpoint:\n"
            f"{resume_checkpoint}"
        )

    # --------------------------------------------------------
    # W&B
    # --------------------------------------------------------

    report_to = (
        "none"
        if args.no_wandb
        else "wandb"
    )

    # --------------------------------------------------------
    # GRPO config
    # --------------------------------------------------------

    training_args = GRPOConfig(
        output_dir=str(output_dir),

        # Optimizer
        learning_rate=args.learning_rate,
        adam_beta1=ADAM_BETA1,
        adam_beta2=ADAM_BETA2,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=MAX_GRAD_NORM,

        # Formal batching
        per_device_train_batch_size=(
            args.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        # IMPORTANT:
        # Do not explicitly set generation_batch_size here.
        #
        # Current TRL derives:
        # generation_batch_size =
        #   per_device_train_batch_size
        #   * world_size
        #   * steps_per_generation
        #
        # and steps_per_generation defaults to
        # gradient_accumulation_steps.
        #
        # On one GPU:
        # 1 * 4 = 4 completions,
        # exactly one complete 4-rollout group.
        num_generations=args.num_generations,

        # Generation
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=(
            args.max_completion_length
        ),
        use_vllm=False,

        # GRPO objective
        beta=args.beta,
        loss_type=LOSS_TYPE,
        scale_rewards="group",
        mask_truncated_completions=True,

        # Full training
        num_train_epochs=args.num_train_epochs,
        max_steps=-1,

        # Memory
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,

        # Logging
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        log_completions=False,
        report_to=report_to,
        run_name=args.run_name,

        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=False,

        # No validation inside GRPO for now
        eval_strategy="no",

        # Dataset/reward columns must be preserved
        remove_unused_columns=False,

        # Reproducibility
        seed=args.seed,
        data_seed=args.seed,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    print("\nCreating TRL GRPOTrainer...")

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        reward_funcs=[
            mapwise_correctness_reward
        ],
        train_dataset=train_dataset,
    )

    trainer.add_callback(
        VisionTrainingMonitorCallback(
            model=model,
            output_dir=output_dir,
        )
    )

    # --------------------------------------------------------
    # Pre-flight
    # --------------------------------------------------------

    print("\n")
    print("=" * 88)
    print("FORMAL TRAINING PRE-FLIGHT")
    print("=" * 88)
    print(f"Model:                         {args.model_name}")
    print(f"Training QA:                   {len(train_dataset)}")
    print(f"Epochs:                        {args.num_train_epochs}")
    print(f"Expected optimizer steps:      {expected_optimizer_steps}")
    print(f"num_generations:               {args.num_generations}")
    print(f"per_device_train_batch_size:   {args.per_device_train_batch_size}")
    print(f"gradient_accumulation_steps:   {args.gradient_accumulation_steps}")
    print(
        "Effective completion batch:    "
        f"{args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )
    print(f"max completion length:         {args.max_completion_length}")
    print(f"temperature:                   {args.temperature}")
    print(f"top_p:                         {args.top_p}")
    print(f"learning rate:                 {args.learning_rate}")
    print(f"warmup steps:                  {warmup_steps}")
    print(f"LoRA rank / alpha:             {args.lora_rank} / {args.lora_alpha}")
    print("LoRA scope:                    Vision only")
    print("Expected target modules:       108")
    print("Expected trainable tensors:    216")
    print("Reward:                        correctness only (0 / 1)")
    print("Format reward:                 False")
    print("Severe-loop penalty:           False")
    print(f"loss_type:                     {LOSS_TYPE}")
    print("scale_rewards:                 group")
    print("mask truncated completions:    True")
    print(f"beta:                          {args.beta}")
    print(f"save every optimizer steps:    {args.save_steps}")
    print(f"W&B:                           {not args.no_wandb}")
    print("vLLM:                          False")
    print("Unsloth:                       False")
    print("=" * 88)

    if resume_checkpoint is None:
        print("\nStarting formal Native Vision-GRPO training...\n")
    else:
        print("\nResuming formal Native Vision-GRPO training...\n")

    torch.cuda.empty_cache()

    overall_start = time.perf_counter()

    result = trainer.train(
        resume_from_checkpoint=resume_checkpoint
    )

    overall_time = (
        time.perf_counter()
        - overall_start
    )

    # --------------------------------------------------------
    # Final adapter
    # --------------------------------------------------------

    final_adapter_dir = (
        output_dir / "final_adapter"
    )

    final_adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        final_adapter_dir
    )

    processor.save_pretrained(
        final_adapter_dir
    )

    trainer.save_state()

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = dict(
        result.metrics
    )

    metrics.update(
        {
            "wall_time_seconds": overall_time,
            "wall_time_hours": (
                overall_time / 3600.0
            ),
            "dataset_size": len(train_dataset),
            "expected_optimizer_steps":
                expected_optimizer_steps,
            "warmup_steps": warmup_steps,
            "final_adapter_dir":
                str(final_adapter_dir),
        }
    )

    if torch.cuda.is_available():
        metrics.update(
            {
                "final_cuda_allocated_gb":
                    torch.cuda.memory_allocated()
                    / 1024**3,

                "final_cuda_reserved_gb":
                    torch.cuda.memory_reserved()
                    / 1024**3,
            }
        )

    with (
        output_dir
        / "training_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics,
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("\n")
    print("=" * 88)
    print("TRAINING COMPLETE")
    print("=" * 88)
    print(
        f"Runtime:       "
        f"{overall_time / 3600.0:.2f} h"
    )
    print(
        f"Final adapter:\n"
        f"{final_adapter_dir}"
    )
    print(
        f"\nStep metrics:\n"
        f"{output_dir / 'step_metrics.jsonl'}"
    )
    print(
        f"\nTraining metrics:\n"
        f"{output_dir / 'training_metrics.json'}"
    )
    print("=" * 88)

    return final_adapter_dir


# ============================================================
# 19. CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Formal MapWise Native HF "
            "Vision-only LoRA GRPO training."
        )
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--qa-json",
        type=Path,
        default=DEFAULT_QA_JSON,
    )

    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
    )

    parser.add_argument(
        "--evaluation-script",
        type=Path,
        default=DEFAULT_EVALUATION_SCRIPT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=NUM_TRAIN_EPOCHS,
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
    )

    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=PER_DEVICE_TRAIN_BATCH_SIZE,
    )

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )

    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=MAX_COMPLETION_LENGTH,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=LORA_RANK,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=LORA_ALPHA,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=WARMUP_RATIO_FOR_STEP_CALCULATION,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=BETA,
    )

    parser.add_argument(
        "--save-steps",
        type=int,
        default=SAVE_STEPS,
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Checkpoint path, or 'latest'. "
            "Example: --resume-from-checkpoint latest"
        ),
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=(
            "MapWise_Qwen3VL8B_"
            "Native_visLoRA_GRPO_correctness"
        ),
    )

    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


# ============================================================
# 20. Validation
# ============================================================

def validate_args(
    args: argparse.Namespace,
) -> None:

    if args.num_train_epochs <= 0:
        raise ValueError(
            "num_train_epochs must be > 0."
        )

    if args.num_generations < 2:
        raise ValueError(
            "num_generations must be >= 2."
        )

    if args.per_device_train_batch_size < 1:
        raise ValueError(
            "per_device_train_batch_size must be >= 1."
        )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps must be >= 1."
        )

    effective_batch = (
        args.per_device_train_batch_size
        * args.gradient_accumulation_steps
    )

    if (
        effective_batch
        % args.num_generations
        != 0
    ):
        raise ValueError(
            "Effective training batch must be divisible "
            "by num_generations.\n"
            f"per_device_train_batch_size="
            f"{args.per_device_train_batch_size}\n"
            f"gradient_accumulation_steps="
            f"{args.gradient_accumulation_steps}\n"
            f"effective_batch={effective_batch}\n"
            f"num_generations={args.num_generations}"
        )

    if args.max_completion_length < 128:
        raise ValueError(
            "max_completion_length is unreasonably small."
        )

    if args.temperature <= 0:
        raise ValueError(
            "temperature must be > 0."
        )

    if not 0 < args.top_p <= 1:
        raise ValueError(
            "top_p must be in (0, 1]."
        )

    if not 0 <= args.warmup_fraction < 1:
        raise ValueError(
            "warmup_fraction must be in [0, 1)."
        )

    if args.save_steps < 1:
        raise ValueError(
            "save_steps must be >= 1."
        )


# ============================================================
# 21. Main
# ============================================================

def main() -> None:
    args = parse_args()
    validate_args(args)
    run_training(args)


if __name__ == "__main__":

    try:
        main()

    except torch.OutOfMemoryError:
        print(
            "\n"
            + "=" * 88,
            file=sys.stderr,
        )
        print(
            "CUDA OUT OF MEMORY",
            file=sys.stderr,
        )
        print(
            "=" * 88,
            file=sys.stderr,
        )

        if torch.cuda.is_available():
            print(
                torch.cuda.memory_summary(
                    abbreviated=True
                ),
                file=sys.stderr,
            )

        print(
            "\nFirst recovery option:",
            file=sys.stderr,
        )
        print(
            "  --max-completion-length 1024",
            file=sys.stderr,
        )

        print(
            "\nDo not reduce LoRA rank as the "
            "first response to OOM.",
            file=sys.stderr,
        )

        raise

    except Exception as error:
        print(
            "\nProgram failed.",
            file=sys.stderr,
        )

        print(
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )

        raise
