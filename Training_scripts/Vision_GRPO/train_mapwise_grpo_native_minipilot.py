#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapWise Native Hugging Face Vision-only GRPO mini-pilot.

Stack
-----
Qwen/Qwen3-VL-8B-Thinking
+ Transformers
+ bitsandbytes 4-bit NF4
+ PEFT Vision-only LoRA
+ TRL GRPOTrainer

Purpose
-------
This is a short realistic pilot before full Native GRPO training.

Unlike test_native_vision_grpo_gradient.py:
- Uses the real MapWise strict exact-match reward.
- Uses 4 generations per prompt.
- Uses max_completion_length=1536.
- Uses the same MapWise image/prompt/reward logic as the Unsloth training run.
- Keeps ONLY Vision LoRA trainable.
- Compact gradient / VRAM diagnostics.
- Saves the final pilot adapter and metrics.

The purpose is to answer:

1. Does Vision-LoRA continue receiving real GRPO gradients?
2. Can RTX A4500 20 GB sustain the intended configuration?
3. What is the realistic Native HF step time / peak VRAM?
4. Does real MapWise reward produce useful reward variance?

This is NOT yet the final full training run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

# Windows currently warns that expandable_segments is unsupported.
# Keeping this does not hurt, but PyTorch may simply ignore it.

import numpy as np
import torch

from datasets import (
    Dataset,
    Image as HFImage,
    Sequence,
)

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

# ------------------------------------------------------------
# GRPO
# ------------------------------------------------------------

NUM_GENERATIONS = 4

MAX_COMPLETION_LENGTH = 1536

TEMPERATURE = 0.8
TOP_P = 0.95

BETA = 0.0

# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_RANK = 16
LORA_ALPHA = 16

# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LEARNING_RATE = 5e-6

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.99

WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1

MAX_GRAD_NORM = 0.1

# ------------------------------------------------------------
# Mini pilot
# ------------------------------------------------------------

NUM_SAMPLES = 20
MAX_STEPS = 20

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
    r"\mapwise_evaluation.py"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Training_outputs"
    r"\Native_Vision_GRPO_MiniPilot"
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
# 4. Vision LoRA target definition
# ============================================================

"""
Target exactly the same scope already verified in:

    test_native_vision_grpo_gradient.py

Qwen3-VL has 27 visual blocks:

    model.visual.blocks.0 ... 26

Within each block:

    attn.qkv
    attn.proj
    mlp.linear_fc1
    mlp.linear_fc2

27 * 4 = 108 Linear modules.

Each LoRA module has A + B:

108 * 2 = 216 trainable tensors.

Do NOT target:
    visual.merger
    visual.deepstack_merger_list

This preserves the controlled comparison with the previous
Unsloth Vision-only run.
"""

VISION_TARGET_REGEX = (
    r"^model\.visual\.blocks\.\d+\."
    r"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))$"
)

EXPECTED_VISION_BLOCKS = 27
EXPECTED_MODULES_PER_BLOCK = 4

EXPECTED_TARGET_MODULES = (
    EXPECTED_VISION_BLOCKS
    * EXPECTED_MODULES_PER_BLOCK
)

EXPECTED_LORA_TENSORS = (
    EXPECTED_TARGET_MODULES * 2
)


# ============================================================
# 5. Globals
# ============================================================

MAPWISE_EVALUATOR = None

REWARD_CALL_COUNT = 0


# ============================================================
# 6. Seed
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 7. MapWise evaluator
# ============================================================

def load_evaluation_module(
    evaluation_script: Path,
):

    evaluation_script = (
        evaluation_script
        .expanduser()
        .resolve()
    )

    if not evaluation_script.is_file():

        raise FileNotFoundError(
            "MapWise evaluation script "
            "not found:\n"
            f"{evaluation_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "mapwise_evaluation_for_native_grpo",
        evaluation_script,
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise ImportError(
            f"Could not load evaluator:\n"
            f"{evaluation_script}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "evaluate_sample",
    ):

        raise AttributeError(
            "MapWise evaluator must expose "
            "evaluate_sample(record)."
        )

    return module


# ============================================================
# 8. Prompt
# ============================================================

def build_mapwise_prompt(
    question: str,
) -> str:

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

def load_json_list(
    path: Path,
) -> list[dict[str, Any]]:

    path = (
        path
        .expanduser()
        .resolve()
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"Training JSON does not exist:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        data = json.load(handle)

    if (
        not isinstance(data, list)
        or not data
    ):

        raise ValueError(
            "Training JSON must contain "
            "a non-empty top-level list."
        )

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(data):

        if not isinstance(
            item,
            Mapping,
        ):

            raise TypeError(
                f"Sample {index} "
                "is not a dictionary."
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
            key
            for key in required
            if key not in row
        ]

        if missing:

            raise KeyError(
                f"Sample {index} "
                f"missing fields: {missing}"
            )

        country = str(
            row.get(
                "country",
                "",
            )
        ).strip().lower()

        if (
            country
            not in SUPPORTED_COUNTRIES
        ):

            raise ValueError(
                f"Sample {index} has "
                f"unsupported "
                f"country={country!r}."
            )

        if not str(
            row.get(
                "qa_id",
                "",
            )
        ).strip():

            source_index = int(
                row.get(
                    "source_index",
                    index,
                )
            )

            row["qa_id"] = (
                f"mapwise_{country}_"
                f"{row['map_no']}_"
                f"t{int(row['template_no'])}_"
                f"src{source_index:04d}"
            )

        rows.append(
            row
        )

    return rows


# ============================================================
# 10. Image resolution
# ============================================================

def resolve_mapwise_image(
    sample: Mapping[str, Any],
    image_root: Path,
) -> Path:

    """
    Resolve:

        <image_root>/
            <country>/
                images/
                    with_annotations/
                        <map_no>.<suffix>
    """

    country = str(
        sample.get(
            "country",
            "",
        )
    ).strip().lower()

    map_no = str(
        sample.get(
            "map_no",
            "",
        )
    ).strip()

    qa_id = str(
        sample.get(
            "qa_id",
            "unknown",
        )
    )

    country_root = (
        image_root
        .expanduser()
        .resolve()
        / country
        / "images"
        / "with_annotations"
    )

    if not country_root.is_dir():

        raise FileNotFoundError(
            "Country image directory "
            "does not exist:\n"
            f"{country_root}"
        )

    raw = Path(
        map_no
    )

    # --------------------------------------------------------
    # map_no already contains suffix
    # --------------------------------------------------------

    if raw.suffix:

        candidate = (
            country_root
            / raw
        )

        if candidate.is_file():

            return (
                candidate.resolve()
            )

    # --------------------------------------------------------
    # Try known suffixes
    # --------------------------------------------------------

    for suffix in (
        SUPPORTED_IMAGE_SUFFIXES
    ):

        candidate = (
            country_root
            / f"{map_no}{suffix}"
        )

        if candidate.is_file():

            return (
                candidate.resolve()
            )

    # --------------------------------------------------------
    # Fallback stem search
    # --------------------------------------------------------

    target_stem = (
        raw.stem.casefold()
    )

    matches = [
        p
        for p in country_root.rglob("*")
        if (
            p.is_file()
            and p.suffix.casefold()
            in SUPPORTED_IMAGE_SUFFIXES
            and p.stem.casefold()
            == target_stem
        )
    ]

    if len(matches) == 1:

        return (
            matches[0]
            .resolve()
        )

    if len(matches) > 1:

        raise RuntimeError(
            f"Multiple images match "
            f"qa_id={qa_id}, "
            f"map_no={map_no!r}:\n"
            + "\n".join(
                str(p)
                for p in matches
            )
        )

    raise FileNotFoundError(
        f"No image found for "
        f"qa_id={qa_id}, "
        f"country={country}, "
        f"map_no={map_no!r} "
        f"under:\n"
        f"{country_root}"
    )


# ============================================================
# 11. Pilot dataset
# ============================================================

def build_pilot_dataset(
    qa_json: Path,
    image_root: Path,
    num_samples: int,
    seed: int,
) -> Dataset:

    samples = (
        load_json_list(
            qa_json
        )
    )

    if num_samples < 1:

        raise ValueError(
            "num_samples must be >= 1."
        )

    if num_samples > len(samples):

        raise ValueError(
            f"num_samples={num_samples} "
            f"but dataset only has "
            f"{len(samples)} samples."
        )

    # --------------------------------------------------------
    # Deterministic random pilot subset
    #
    # Do not simply take samples[:20], because consecutive
    # MapWise rows often come from the same map.
    # --------------------------------------------------------

    rng = random.Random(
        seed
    )

    selected_indices = (
        rng.sample(
            range(
                len(samples)
            ),
            num_samples,
        )
    )

    selected_samples = [
        samples[i]
        for i in selected_indices
    ]

    rows: list[dict[str, Any]] = []

    print("\n")
    print("=" * 80)
    print("MAPWISE NATIVE GRPO MINI-PILOT DATASET")
    print("=" * 80)

    for local_index, sample in enumerate(
        selected_samples
    ):

        qa_id = str(
            sample["qa_id"]
        )

        question = str(
            sample.get(
                "question",
                "",
            )
        ).strip()

        ground_truth = str(
            sample.get(
                "ground_truth",
                "",
            )
        ).strip()

        ground_truth_type = str(
            sample.get(
                "ground_truth_type",
                "",
            )
        ).strip()

        if (
            not question
            or not ground_truth
        ):

            raise ValueError(
                f"{qa_id} has empty "
                "question or ground_truth."
            )

        image_path = (
            resolve_mapwise_image(
                sample,
                image_root,
            )
        )

        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image"
                            },
                            {
                                "type": "text",
                                "text": (
                                    build_mapwise_prompt(
                                        question
                                    )
                                ),
                            },
                        ],
                    }
                ],

                "images": [
                    str(
                        image_path
                    )
                ],

                "qa_id": qa_id,

                "country": str(
                    sample.get(
                        "country",
                        "",
                    )
                ),

                "map_no": str(
                    sample.get(
                        "map_no",
                        "",
                    )
                ),

                "template_no": int(
                    sample.get(
                        "template_no",
                        -1,
                    )
                ),

                "ground_truth": (
                    ground_truth
                ),

                "ground_truth_type": (
                    ground_truth_type
                ),

                "legend_style": str(
                    sample.get(
                        "legend_style",
                        sample.get(
                            "c_or_d",
                            "",
                        ),
                    )
                ),

                "c_or_d": str(
                    sample.get(
                        "c_or_d",
                        "",
                    )
                ),

                "relative_region": str(
                    sample.get(
                        "relative_region",
                        "",
                    )
                ),

                "corrected_template_no": (
                    sample.get(
                        "corrected_template_no"
                    )
                ),

                "ability_level": str(
                    sample.get(
                        "ability_level",
                        "",
                    )
                ),

                "map_family": str(
                    sample.get(
                        "map_family",
                        "",
                    )
                ),

                "source_index": int(
                    sample.get(
                        "source_index",
                        selected_indices[
                            local_index
                        ],
                    )
                ),
            }
        )

    dataset = (
        Dataset.from_list(
            rows
        )
    )

    dataset = (
        dataset.cast_column(
            "images",
            Sequence(
                HFImage()
            ),
        )
    )

    # --------------------------------------------------------
    # Compact distribution summary
    # --------------------------------------------------------

    countries = Counter(
        str(v)
        for v in dataset["country"]
    )

    answer_types = Counter(
        str(v)
        for v in dataset[
            "ground_truth_type"
        ]
    )

    maps = set(
        zip(
            dataset["country"],
            dataset["map_no"],
        )
    )

    print(
        f"Samples:      "
        f"{len(dataset)}"
    )

    print(
        f"Unique maps:  "
        f"{len(maps)}"
    )

    print(
        f"Countries:    "
        f"{dict(countries)}"
    )

    print(
        f"Answer types: "
        f"{dict(answer_types)}"
    )

    print("=" * 80)
    print()

    return dataset


# ============================================================
# 12. Completion handling
# ============================================================

def completion_to_text(
    completion: Any,
) -> str:

    if completion is None:
        return ""

    if isinstance(
        completion,
        str,
    ):

        return (
            completion.strip()
        )

    if isinstance(
        completion,
        list,
    ):

        parts: list[str] = []

        for item in completion:

            if isinstance(
                item,
                dict,
            ):

                content = item.get(
                    "content",
                    "",
                )

                if isinstance(
                    content,
                    str,
                ):

                    parts.append(
                        content
                    )

                elif isinstance(
                    content,
                    list,
                ):

                    for block in content:

                        if (
                            isinstance(
                                block,
                                dict,
                            )
                            and
                            block.get(
                                "type"
                            )
                            == "text"
                        ):

                            parts.append(
                                str(
                                    block.get(
                                        "text",
                                        "",
                                    )
                                )
                            )

            else:

                parts.append(
                    str(
                        item
                    )
                )

        return "\n".join(
            part
            for part in parts
            if part
        ).strip()

    if isinstance(
        completion,
        dict,
    ):

        content = (
            completion.get(
                "content",
                "",
            )
        )

        if isinstance(
            content,
            str,
        ):

            return (
                content.strip()
            )

        return (
            completion_to_text(
                content
            )
        )

    return (
        str(
            completion
        ).strip()
    )


# ============================================================
# 13. Reward utilities
# ============================================================

def _as_list(
    value: Any,
    expected_length: int,
) -> list[Any]:

    if isinstance(
        value,
        list,
    ):

        if (
            len(value)
            == expected_length
        ):

            return value

        if len(value) == 1:

            return (
                value
                * expected_length
            )

        return value

    return (
        [value]
        * expected_length
    )


# ============================================================
# 14. REAL MapWise reward
# ============================================================

def mapwise_correctness_reward(
    completions: list[Any],
    ground_truth: list[str],
    ground_truth_type: list[str],
    country: list[str],
    template_no: list[int],
    legend_style: Optional[
        list[str]
    ] = None,
    c_or_d: Optional[
        list[str]
    ] = None,
    qa_id: Optional[
        list[str]
    ] = None,
    **kwargs: Any,
) -> list[float]:

    """
    Real GRPO reward.

    Exactly the same logic as the existing Unsloth script:

        MapWise evaluator
            -> evaluate_sample(record)
            -> strict_exact_match

    Correct -> 1.0
    Wrong   -> 0.0
    """

    global MAPWISE_EVALUATOR, REWARD_CALL_COUNT

    if MAPWISE_EVALUATOR is None:

        raise RuntimeError(
            "MapWise evaluator "
            "was not initialized."
        )

    REWARD_CALL_COUNT += 1

    n = len(
        completions
    )

    golds = _as_list(
        ground_truth,
        n,
    )

    kinds = _as_list(
        ground_truth_type,
        n,
    )

    countries = _as_list(
        country,
        n,
    )

    templates = _as_list(
        template_no,
        n,
    )

    legends = _as_list(
        legend_style
        if legend_style is not None
        else "",
        n,
    )

    cds = _as_list(
        c_or_d
        if c_or_d is not None
        else "",
        n,
    )

    qa_ids = _as_list(
        qa_id
        if qa_id is not None
        else "",
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
            "qa_id": str(
                item_id or ""
            ),

            "country": str(
                ctry or ""
            ),

            "ground_truth": str(
                gold or ""
            ),

            "ground_truth_type": str(
                kind or ""
            ),

            # Preserve raw MapWise template number.
            "template_no": int(
                template
            ),

            "legend_style": str(
                legend
                or cd
                or ""
            ),

            "c_or_d": str(
                cd or ""
            ),

            "raw_response": (
                completion_to_text(
                    completion
                )
            ),

            "final_answer": "",
        }

        result = (
            MAPWISE_EVALUATOR
            .evaluate_sample(
                record
            )
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

        rewards.append(
            reward
        )

    # --------------------------------------------------------
    # Compact reward trace
    # --------------------------------------------------------

    qa = (
        str(
            qa_ids[0]
        )
        if qa_ids
        else "unknown"
    )

    mean_reward = (
        sum(rewards)
        / max(
            len(rewards),
            1,
        )
    )

    variance_exists = (
        len(
            set(
                rewards
            )
        )
        > 1
    )

    print(
        f"[REWARD {REWARD_CALL_COUNT:03d}] "
        f"qa={qa} | "
        f"rewards={rewards} | "
        f"mean={mean_reward:.2f} | "
        f"variance={'YES' if variance_exists else 'NO'}"
    )

    return rewards


mapwise_correctness_reward.__name__ = (
    "correctness"
)


# ============================================================
# 15. PEFT verification
# ============================================================

def verify_vision_only_lora(
    model,
) -> None:

    targeted = list(
        getattr(
            model,
            "targeted_module_names",
            [],
        )
    )

    trainable_names = [
        name
        for name, param
        in model.named_parameters()
        if param.requires_grad
    ]

    vision_trainable = [
        name
        for name
        in trainable_names
        if "visual"
        in name.lower()
    ]

    nonvision_trainable = [
        name
        for name
        in trainable_names
        if "visual"
        not in name.lower()
    ]

    print("\n")
    print("=" * 80)
    print("PEFT VISION-LoRA CHECK")
    print("=" * 80)

    print(
        f"Target modules:           "
        f"{len(targeted)}"
    )

    print(
        f"Trainable tensors:        "
        f"{len(trainable_names)}"
    )

    print(
        f"Vision trainable tensors: "
        f"{len(vision_trainable)}"
    )

    print(
        f"Non-vision tensors:       "
        f"{len(nonvision_trainable)}"
    )

    if (
        len(targeted)
        != EXPECTED_TARGET_MODULES
    ):

        raise RuntimeError(
            "Unexpected LoRA target count: "
            f"expected "
            f"{EXPECTED_TARGET_MODULES}, "
            f"got {len(targeted)}."
        )

    if (
        len(trainable_names)
        != EXPECTED_LORA_TENSORS
    ):

        raise RuntimeError(
            "Unexpected trainable tensor "
            f"count: expected "
            f"{EXPECTED_LORA_TENSORS}, "
            f"got "
            f"{len(trainable_names)}."
        )

    if nonvision_trainable:

        raise RuntimeError(
            "Non-vision trainable "
            "parameters found:\n"
            + "\n".join(
                nonvision_trainable
            )
        )

    bad_targets = [
        name
        for name in targeted
        if "visual"
        not in name.lower()
    ]

    if bad_targets:

        raise RuntimeError(
            "Non-vision LoRA targets found:\n"
            + "\n".join(
                bad_targets
            )
        )

    print(
        "PASS: Native PEFT is "
        "strictly Vision-only."
    )

    print("=" * 80)
    print()


# ============================================================
# 16. Compact gradient + VRAM callback
# ============================================================

class PilotMonitorCallback(
    TrainerCallback
):

    """
    Compact mini-pilot monitoring.

    Instead of printing all 216 tensors, prints one line:

    [STEP 001] grad none=0 zero=0 nonzero=216 |
               lora_grad_norm=... |
               VRAM alloc=... reserved=... peak=...

    Also writes JSONL metrics to disk.
    """

    def __init__(
        self,
        model,
        output_dir: Path,
    ):

        self.model = model

        self.output_dir = (
            output_dir
        )

        self.metrics_path = (
            output_dir
            / "pilot_step_metrics.jsonl"
        )

        self.step_start_time = None

        self.latest_gradient_stats = {}

    def on_step_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        self.step_start_time = (
            time.perf_counter()
        )

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

        for name, param in (
            self.model.named_parameters()
        ):

            if not param.requires_grad:
                continue

            if (
                "visual"
                in name.lower()
            ):

                vision_trainable += 1

            else:

                nonvision_trainable += 1

            if param.grad is None:

                grad_none += 1
                continue

            grad = (
                param.grad
                .detach()
                .float()
            )

            grad_norm = (
                grad.norm()
                .item()
            )

            if grad_norm == 0.0:

                grad_zero += 1

            else:

                grad_nonzero += 1

                squared_norm_sum += (
                    grad_norm
                    * grad_norm
                )

        total_lora_grad_norm = (
            squared_norm_sum
            ** 0.5
        )

        self.latest_gradient_stats = {
            "vision_trainable": (
                vision_trainable
            ),
            "nonvision_trainable": (
                nonvision_trainable
            ),
            "grad_none": (
                grad_none
            ),
            "grad_zero": (
                grad_zero
            ),
            "grad_nonzero": (
                grad_nonzero
            ),
            "lora_grad_norm": (
                total_lora_grad_norm
            ),
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

        payload = {
            "global_step": int(
                state.global_step
            ),

            "step_time_seconds": (
                elapsed
            ),

            **self.latest_gradient_stats,
        }

        # ----------------------------------------------------
        # CUDA memory
        # ----------------------------------------------------

        if torch.cuda.is_available():

            gb = (
                1024 ** 3
            )

            allocated = (
                torch.cuda
                .memory_allocated()
                / gb
            )

            reserved = (
                torch.cuda
                .memory_reserved()
                / gb
            )

            peak_allocated = (
                torch.cuda
                .max_memory_allocated()
                / gb
            )

            peak_reserved = (
                torch.cuda
                .max_memory_reserved()
                / gb
            )

            payload.update(
                {
                    "vram_allocated_gb": (
                        allocated
                    ),

                    "vram_reserved_gb": (
                        reserved
                    ),

                    "vram_peak_allocated_gb": (
                        peak_allocated
                    ),

                    "vram_peak_reserved_gb": (
                        peak_reserved
                    ),
                }
            )

        # ----------------------------------------------------
        # Console summary
        # ----------------------------------------------------

        print(
            "\n"
            f"[STEP {state.global_step:03d}] "
            f"grad: "
            f"none={payload.get('grad_none', '?')} "
            f"zero={payload.get('grad_zero', '?')} "
            f"nonzero={payload.get('grad_nonzero', '?')} "
            f"| norm="
            f"{payload.get('lora_grad_norm', float('nan')):.4e}"
            f" | time="
            f"{elapsed:.2f}s"
            if elapsed is not None
            else ""
        )

        if torch.cuda.is_available():

            print(
                f"           VRAM: "
                f"alloc="
                f"{payload['vram_allocated_gb']:.2f} GB | "
                f"reserved="
                f"{payload['vram_reserved_gb']:.2f} GB | "
                f"peak alloc="
                f"{payload['vram_peak_allocated_gb']:.2f} GB | "
                f"peak reserved="
                f"{payload['vram_peak_reserved_gb']:.2f} GB"
            )

        # ----------------------------------------------------
        # Persistent JSONL
        # ----------------------------------------------------

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
# 17. GPU info
# ============================================================

def print_gpu_info() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required."
        )

    props = (
        torch.cuda
        .get_device_properties(0)
    )

    print("=" * 80)

    print(
        "MapWise Native Vision-GRPO "
        "Mini-Pilot"
    )

    print("=" * 80)

    print(
        f"GPU:  "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{props.total_memory / 1024**3:.2f} GB"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    print("=" * 80)


# ============================================================
# 18. Save run configuration
# ============================================================

def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_size: int,
) -> None:

    payload = (
        vars(args)
        .copy()
    )

    for key in (
        "qa_json",
        "image_root",
        "evaluation_script",
        "output_dir",
    ):

        if key in payload:

            payload[key] = str(
                Path(
                    payload[key]
                )
                .expanduser()
                .resolve()
            )

    payload[
        "dataset_size"
    ] = dataset_size

    payload[
        "stack"
    ] = {
        "transformers": True,
        "peft": True,
        "trl": True,
        "unsloth": False,
        "vllm": False,
    }

    payload[
        "lora_scope"
    ] = {
        "vision": True,
        "language": False,
        "attention": True,
        "mlp": True,
        "target_modules": (
            EXPECTED_TARGET_MODULES
        ),
        "trainable_tensors": (
            EXPECTED_LORA_TENSORS
        ),
    }

    payload[
        "reward_design"
    ] = {
        "strict_exact_correct": 1.0,
        "incorrect": 0.0,
    }

    with (
        output_dir
        / "run_config.json"
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
# 19. Run mini-pilot
# ============================================================

def run_pilot(
    args: argparse.Namespace,
) -> Path:

    global MAPWISE_EVALUATOR

    set_seed(
        args.seed
    )

    print_gpu_info()

    qa_json = (
        Path(args.qa_json)
        .expanduser()
        .resolve()
    )

    image_root = (
        Path(args.image_root)
        .expanduser()
        .resolve()
    )

    evaluation_script = (
        Path(args.evaluation_script)
        .expanduser()
        .resolve()
    )

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove old per-step log if rerunning same output directory.
    step_metrics_path = (
        output_dir
        / "pilot_step_metrics.jsonl"
    )

    if step_metrics_path.exists():

        step_metrics_path.unlink()

    # ========================================================
    # Evaluator
    # ========================================================

    print(
        "\nLoading MapWise evaluator..."
    )

    print(
        f"Evaluator: "
        f"{evaluation_script}"
    )

    MAPWISE_EVALUATOR = (
        load_evaluation_module(
            evaluation_script
        )
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nPreparing pilot dataset..."
    )

    train_dataset = (
        build_pilot_dataset(
            qa_json=qa_json,
            image_root=image_root,
            num_samples=(
                args.num_samples
            ),
            seed=args.seed,
        )
    )

    save_run_config(
        output_dir=output_dir,
        args=args,
        dataset_size=len(
            train_dataset
        ),
    )

    # ========================================================
    # Processor
    # ========================================================

    print(
        "Loading Qwen3-VL processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_name
        )
    )

    if hasattr(
        processor,
        "tokenizer",
    ):

        processor.tokenizer.padding_side = (
            "left"
        )

        if (
            processor
            .tokenizer
            .pad_token_id
            is None
        ):

            processor.tokenizer.pad_token = (
                processor
                .tokenizer
                .eos_token
            )

    # ========================================================
    # Quantization
    # ========================================================

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type="nf4",

            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),

            bnb_4bit_use_double_quant=True,
        )
    )

    # ========================================================
    # Base model
    # ========================================================

    print(
        "\nLoading Qwen3-VL "
        "4-bit base model..."
    )

    model = (
        AutoModelForImageTextToText
        .from_pretrained(
            args.model_name,

            quantization_config=(
                quantization_config
            ),

            dtype=torch.bfloat16,

            device_map={
                "": 0
            },

            low_cpu_mem_usage=True,
        )
    )

    model.config.use_cache = False

    # ========================================================
    # QLoRA preparation
    # ========================================================

    print(
        "\nPreparing model "
        "for k-bit training..."
    )

    model = (
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    )

    # ========================================================
    # Vision-only LoRA
    # ========================================================

    print(
        "\nApplying Vision-only LoRA..."
    )

    lora_config = (
        LoraConfig(
            r=args.lora_rank,

            lora_alpha=(
                args.lora_alpha
            ),

            lora_dropout=0.0,

            bias="none",

            target_modules=(
                VISION_TARGET_REGEX
            ),

            task_type="CAUSAL_LM",
        )
    )

    model = (
        get_peft_model(
            model,
            lora_config,
        )
    )

    model.print_trainable_parameters()

    verify_vision_only_lora(
        model
    )

    # ========================================================
    # GRPO config
    # ========================================================

    training_args = (
        GRPOConfig(

            output_dir=str(
                output_dir
            ),

            # --------------------------------------------
            # Optimizer
            # --------------------------------------------

            learning_rate=(
                args.learning_rate
            ),

            adam_beta1=(
                ADAM_BETA1
            ),

            adam_beta2=(
                ADAM_BETA2
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),

            warmup_steps=2,

            lr_scheduler_type="cosine",

            # LoRA-only optimizer states are small;
            # adamw_8bit also mirrors your Unsloth run.
            optim="adamw_8bit",

            max_grad_norm=(
                MAX_GRAD_NORM
            ),

            # --------------------------------------------
            # Training batch
            # --------------------------------------------

            per_device_train_batch_size=1,

            gradient_accumulation_steps=1,

            # Current TRL requires generation_batch_size
            # divisible by num_generations.
            #
            # Keep optimizer-side batch=1, while forming
            # a complete 4-rollout GRPO group.
            generation_batch_size=(
                args.num_generations
            ),

            # --------------------------------------------
            # GRPO generation
            # --------------------------------------------

            num_generations=(
                args.num_generations
            ),

            temperature=(
                args.temperature
            ),

            top_p=(
                args.top_p
            ),

            max_completion_length=(
                args.max_completion_length
            ),

            use_vllm=False,

            # --------------------------------------------
            # Objective
            # --------------------------------------------

            beta=(
                args.beta
            ),

            loss_type="dr_grpo",

            scale_rewards="group",

            # Match the current formal Unsloth config.
            mask_truncated_completions=True,

            # --------------------------------------------
            # Mini-pilot length
            # --------------------------------------------

            max_steps=(
                args.max_steps
            ),

            # --------------------------------------------
            # Memory
            # --------------------------------------------

            gradient_checkpointing=True,

            bf16=True,
            fp16=False,

            # --------------------------------------------
            # Logging
            # --------------------------------------------

            logging_strategy="steps",

            logging_steps=1,

            logging_first_step=True,

            # Keep console output compact.
            log_completions=False,

            report_to="none",

            # --------------------------------------------
            # Saving
            # --------------------------------------------

            save_strategy="no",

            eval_strategy="no",

            load_best_model_at_end=False,

            # --------------------------------------------
            # Dataset
            # --------------------------------------------

            remove_unused_columns=False,

            # --------------------------------------------
            # Reproducibility
            # --------------------------------------------

            seed=args.seed,

            data_seed=args.seed,
        )
    )

    # ========================================================
    # Trainer
    # ========================================================

    print(
        "\nCreating TRL GRPOTrainer..."
    )

    trainer = (
        GRPOTrainer(
            model=model,

            args=training_args,

            processing_class=(
                processor
            ),

            reward_funcs=[
                mapwise_correctness_reward
            ],

            train_dataset=(
                train_dataset
            ),
        )
    )

    monitor = (
        PilotMonitorCallback(
            model=model,
            output_dir=output_dir,
        )
    )

    trainer.add_callback(
        monitor
    )

    # ========================================================
    # Pre-flight
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MINI-PILOT PRE-FLIGHT")
    print("=" * 80)

    print(
        f"Model:                   "
        f"{args.model_name}"
    )

    print(
        f"Pilot samples:           "
        f"{len(train_dataset)}"
    )

    print(
        f"Maximum steps:           "
        f"{args.max_steps}"
    )

    print(
        f"num_generations:         "
        f"{args.num_generations}"
    )

    print(
        f"generation_batch_size:   "
        f"{args.num_generations}"
    )

    print(
        f"optimizer batch size:    "
        f"1"
    )

    print(
        f"max completion length:   "
        f"{args.max_completion_length}"
    )

    print(
        f"temperature:             "
        f"{args.temperature}"
    )

    print(
        f"top_p:                   "
        f"{args.top_p}"
    )

    print(
        f"learning rate:           "
        f"{args.learning_rate}"
    )

    print(
        f"LoRA rank / alpha:       "
        f"{args.lora_rank} / "
        f"{args.lora_alpha}"
    )

    print(
        "LoRA:                    "
        "Vision only"
    )

    print(
        "Expected trainable:      "
        "7,699,968 params / "
        "216 tensors"
    )

    print(
        "Reward:                  "
        "MapWise strict exact "
        "(1 correct / 0 incorrect)"
    )

    print(
        f"loss_type:               "
        f"dr_grpo"
    )

    print(
        f"mask truncated:          "
        f"True"
    )

    print(
        f"beta:                    "
        f"{args.beta}"
    )

    print(
        "vLLM:                    "
        "False"
    )

    print(
        "Unsloth:                 "
        "False"
    )

    print("=" * 80)

    # ========================================================
    # Train
    # ========================================================

    print(
        "\nStarting Native MapWise "
        "Vision-GRPO mini-pilot...\n"
    )

    torch.cuda.empty_cache()

    overall_start = (
        time.perf_counter()
    )

    result = (
        trainer.train()
    )

    overall_time = (
        time.perf_counter()
        - overall_start
    )

    # ========================================================
    # Save adapter
    # ========================================================

    final_adapter_dir = (
        output_dir
        / "final_adapter"
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

    # ========================================================
    # Save metrics
    # ========================================================

    metrics = dict(
        result.metrics
    )

    metrics[
        "pilot_wall_time_seconds"
    ] = overall_time

    metrics[
        "pilot_wall_time_minutes"
    ] = (
        overall_time
        / 60.0
    )

    metrics[
        "final_adapter_dir"
    ] = str(
        final_adapter_dir
    )

    metrics[
        "num_generations"
    ] = (
        args.num_generations
    )

    metrics[
        "max_completion_length"
    ] = (
        args.max_completion_length
    )

    if torch.cuda.is_available():

        metrics[
            "final_cuda_allocated_gb"
        ] = (
            torch.cuda.memory_allocated()
            / 1024 ** 3
        )

        metrics[
            "final_cuda_reserved_gb"
        ] = (
            torch.cuda.memory_reserved()
            / 1024 ** 3
        )

    with (
        output_dir
        / "pilot_metrics.json"
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

    # ========================================================
    # Done
    # ========================================================

    print("\n")
    print("=" * 80)
    print("MINI-PILOT COMPLETE")
    print("=" * 80)

    print(
        f"Runtime: "
        f"{overall_time / 60:.2f} min"
    )

    print(
        f"Final adapter:\n"
        f"{final_adapter_dir}"
    )

    print(
        f"\nStep diagnostics:\n"
        f"{output_dir / 'pilot_step_metrics.jsonl'}"
    )

    print(
        f"\nTraining metrics:\n"
        f"{output_dir / 'pilot_metrics.json'}"
    )

    print(
        "\nWhat to check:"
    )

    print(
        "1. Reward vectors sometimes "
        "contain both 0 and 1."
    )

    print(
        "2. On useful-reward steps, "
        "grad_nonzero should be > 0."
    )

    print(
        "3. After the first optimizer "
        "update, up to all 216 LoRA "
        "tensors should receive gradients."
    )

    print(
        "4. Check peak VRAM against "
        "the 20 GB A4500 limit."
    )

    print(
        "5. Check seconds/step before "
        "deciding whether Native full "
        "training is practical."
    )

    print("=" * 80)

    return (
        final_adapter_dir
    )


# ============================================================
# 20. CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser(
            description=(
                "Native HF Vision-only "
                "GRPO mini-pilot for MapWise."
            )
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
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
    )

    parser.add_argument(
        "--num-generations",
        type=int,
        default=NUM_GENERATIONS,
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
        "--beta",
        type=float,
        default=BETA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return (
        parser.parse_args()
    )


# ============================================================
# 21. Main
# ============================================================

def main() -> None:

    args = parse_args()

    if (
        args.num_samples
        < 1
    ):

        raise ValueError(
            "num_samples must be >= 1."
        )

    if (
        args.max_steps
        < 1
    ):

        raise ValueError(
            "max_steps must be >= 1."
        )

    if (
        args.num_generations
        < 2
    ):

        raise ValueError(
            "num_generations must be >= 2."
        )

    if (
        args.max_completion_length
        < 128
    ):

        raise ValueError(
            "max_completion_length "
            "is unreasonably small."
        )

    if (
        args.temperature
        <= 0
    ):

        raise ValueError(
            "temperature must be > 0."
        )

    if not (
        0
        < args.top_p
        <= 1
    ):

        raise ValueError(
            "top_p must be in (0, 1]."
        )

    run_pilot(
        args
    )


if __name__ == "__main__":

    try:

        main()

    except (
        torch.OutOfMemoryError
    ) as error:

        print(
            "\n"
            + "=" * 80,
            file=sys.stderr,
        )

        print(
            "CUDA OUT OF MEMORY",
            file=sys.stderr,
        )

        print(
            "=" * 80,
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
            "\nDo NOT change LoRA rank "
            "as the first response.",
            file=sys.stderr,
        )

        print(
            "For this mini-pilot, first "
            "try lowering:",
            file=sys.stderr,
        )

        print(
            "  --max-completion-length 1024",
            file=sys.stderr,
        )

        print(
            "and only then consider "
            "image-resolution control.",
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