#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Native Hugging Face Qwen3-VL Vision-only LoRA GRPO gradient diagnostic.

Purpose
-------
Test whether:

    Transformers
    + PEFT Vision-only LoRA
    + TRL GRPOTrainer

can propagate GRPO gradients into the Qwen3-VL visual encoder LoRA parameters.

This is NOT a formal training script.

Diagnostic design
-----------------
- Qwen3-VL-8B-Thinking
- 4-bit NF4 base model
- Vision-only LoRA
- Exactly the same visual block scope as the Unsloth Vision-only experiment:
    visual.blocks.*.attn.qkv
    visual.blocks.*.attn.proj
    visual.blocks.*.mlp.linear_fc1
    visual.blocks.*.mlp.linear_fc2
- Expected:
    27 visual blocks
    108 LoRA target modules
    216 trainable LoRA tensors (A/B)
- Language model must remain frozen.
- num_generations = 2
- Synthetic rewards [0, 1] guarantee non-zero reward variance.
- Only 3 GRPO steps by default.
- No vLLM.
- No Unsloth.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

LORA_RANK = 16
LORA_ALPHA = 16

LEARNING_RATE = 5e-6

NUM_GENERATIONS = 2
MAX_COMPLETION_LENGTH = 256
MAX_STEPS = 3

TEMPERATURE = 0.8
TOP_P = 0.95

SEED = 3407


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


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DEFAULT_QA_JSON = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\Processed_Mapwise\Train_Val"
    r"\mapwise_grpo_validation.json"
)

DEFAULT_IMAGE_ROOT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\mapwise-dataset"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Training_outputs\Native_Vision_GRPO_Gradient_Debug"
)


# ============================================================
# 2. Vision LoRA target definition
# ============================================================

# Native Transformers Qwen3-VL structure:
#
# model.visual.blocks.0.attn.qkv
# model.visual.blocks.0.attn.proj
# model.visual.blocks.0.mlp.linear_fc1
# model.visual.blocks.0.mlp.linear_fc2
#
# ...
#
# model.visual.blocks.26.*
#
# Do NOT target:
#
# model.visual.merger.*
# model.visual.deepstack_merger_list.*
#
# This keeps the LoRA scope equivalent to the previous
# Unsloth Vision-only configuration.

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
# 3. Seed
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. Prompt
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
# 5. MapWise JSON loading
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
            f"QA JSON does not exist:\n{path}"
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
            "QA JSON must contain a "
            "non-empty top-level list."
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

        if country not in SUPPORTED_COUNTRIES:

            raise ValueError(
                f"Sample {index} "
                f"has unsupported "
                f"country={country!r}."
            )

        # Same fallback qa_id logic as
        # the Unsloth training script.

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

        rows.append(row)

    return rows


# ============================================================
# 6. MapWise image resolution
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

    Same logic as the formal Unsloth GRPO script.
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

    raw = Path(map_no)

    # --------------------------------------------------------
    # Case 1:
    # map_no already includes suffix
    # --------------------------------------------------------

    if raw.suffix:

        candidate = (
            country_root
            / raw
        )

        if candidate.is_file():
            return candidate.resolve()

    # --------------------------------------------------------
    # Case 2:
    # Try known suffixes
    # --------------------------------------------------------

    for suffix in SUPPORTED_IMAGE_SUFFIXES:

        candidate = (
            country_root
            / f"{map_no}{suffix}"
        )

        if candidate.is_file():
            return candidate.resolve()

    # --------------------------------------------------------
    # Case 3:
    # Recursive stem matching
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
        return matches[0].resolve()

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
# 7. Tiny diagnostic dataset
# ============================================================

def build_debug_dataset(
    qa_json: Path,
    image_root: Path,
    num_samples: int,
) -> Dataset:

    samples = load_json_list(
        qa_json
    )

    image_root = (
        image_root
        .expanduser()
        .resolve()
    )

    if num_samples < 1:
        raise ValueError(
            "num_samples must be >= 1."
        )

    samples = samples[:num_samples]

    rows = []

    print("\n")
    print("=" * 90)
    print("MAPWISE DIAGNOSTIC DATASET")
    print("=" * 90)

    for index, sample in enumerate(samples):

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

        if not question:

            raise ValueError(
                f"{qa_id} has "
                "an empty question."
            )

        image_path = (
            resolve_mapwise_image(
                sample,
                image_root,
            )
        )

        print(
            f"[{index}] "
            f"qa_id={qa_id}"
        )

        print(
            f"    country="
            f"{sample['country']}"
        )

        print(
            f"    map_no="
            f"{sample['map_no']}"
        )

        print(
            f"    image="
            f"{image_path}"
        )

        print(
            f"    question="
            f"{question[:120]}"
        )

        # Same multimodal prompt structure
        # as your Unsloth script.

        prompt = [
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
        ]

        rows.append(
            {
                "prompt": prompt,

                "images": [
                    str(image_path)
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

                "source_index": int(
                    sample.get(
                        "source_index",
                        index,
                    )
                ),
            }
        )

    dataset = Dataset.from_list(
        rows
    )

    dataset = dataset.cast_column(
        "images",
        Sequence(
            HFImage()
        ),
    )

    print("\n")
    print(
        f"Diagnostic samples: "
        f"{len(dataset)}"
    )

    print(
        f"Dataset columns: "
        f"{dataset.column_names}"
    )

    print("=" * 90)

    return dataset


# ============================================================
# 8. Completion -> text
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
        return completion.strip()

    if isinstance(
        completion,
        list,
    ):

        parts = []

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
                    str(item)
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

        content = completion.get(
            "content",
            "",
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        return completion_to_text(
            content
        )

    return str(
        completion
    ).strip()


# ============================================================
# 9. Synthetic diagnostic reward
# ============================================================

REWARD_CALL_COUNT = 0


def diagnostic_reward(
    completions,
    **kwargs,
):

    """
    Synthetic diagnostic reward.

    For each GRPO rollout group:

        rollout 0 -> reward 0
        rollout 1 -> reward 1

    This deliberately guarantees reward variance.

    We are testing gradient connectivity,
    NOT MapWise accuracy.
    """

    global REWARD_CALL_COUNT

    REWARD_CALL_COUNT += 1

    n = len(completions)

    if n < 2:

        raise RuntimeError(
            "Diagnostic requires "
            "at least two completions."
        )

    rewards = [
        float(i % 2)
        for i in range(n)
    ]

    print("\n")
    print("-" * 90)

    print(
        "DIAGNOSTIC REWARD "
        f"CALL {REWARD_CALL_COUNT}"
    )

    print("-" * 90)

    for index, completion in enumerate(
        completions
    ):

        text = completion_to_text(
            completion
        )

        preview = (
            text
            .replace(
                "\n",
                " ",
            )
            [:250]
        )

        print(
            f"rollout={index} "
            f"reward="
            f"{rewards[index]:.1f}"
        )

        print(
            f"completion: "
            f"{preview}"
        )

    print(
        f"Reward vector: "
        f"{rewards}"
    )

    print("-" * 90)

    return rewards


diagnostic_reward.__name__ = (
    "diagnostic_reward"
)


# ============================================================
# 10. Gradient debugger
# ============================================================

class VisionGradientDebugCallback(
    TrainerCallback
):

    def __init__(
        self,
        model,
        max_steps_to_print=3,
    ):

        self.model = model

        self.max_steps_to_print = (
            max_steps_to_print
        )

    def on_pre_optimizer_step(
        self,
        args,
        state,
        control,
        **kwargs,
    ):

        if (
            state.global_step
            >= self.max_steps_to_print
        ):
            return

        trainable = 0

        vision_trainable = 0
        language_trainable = 0
        other_trainable = 0

        grad_none = 0
        grad_zero = 0
        grad_nonzero = 0

        nonzero_norms = []

        print("\n")
        print("=" * 90)

        print(
            "NATIVE HF VISION-LoRA "
            "GRADIENT DEBUG "
            f"(global_step="
            f"{state.global_step})"
        )

        print("=" * 90)

        for name, param in (
            self.model.named_parameters()
        ):

            if not param.requires_grad:
                continue

            trainable += 1

            lname = (
                name.lower()
            )

            if "visual" in lname:

                vision_trainable += 1

            elif (
                "language_model"
                in lname
                or
                ".model.layers."
                in lname
            ):

                language_trainable += 1

            else:

                other_trainable += 1

            # ----------------------------------------
            # Gradient status
            # ----------------------------------------

            if param.grad is None:

                grad_none += 1

                status = (
                    "NONE"
                )

            else:

                grad = (
                    param
                    .grad
                    .detach()
                    .float()
                )

                norm = (
                    grad
                    .norm()
                    .item()
                )

                if norm == 0.0:

                    grad_zero += 1

                    status = (
                        "ZERO"
                    )

                else:

                    grad_nonzero += 1

                    nonzero_norms.append(
                        (
                            name,
                            norm,
                        )
                    )

                    status = (
                        f"NONZERO "
                        f"{norm:.8e}"
                    )

            print(
                f"[{status:25s}] "
                f"{name}"
            )

        # --------------------------------------------
        # Summary
        # --------------------------------------------

        print(
            "\n--------------------------------------------"
        )

        print(
            f"Trainable tensors : "
            f"{trainable}"
        )

        print(
            f"Vision trainable  : "
            f"{vision_trainable}"
        )

        print(
            f"Language trainable: "
            f"{language_trainable}"
        )

        print(
            f"Other trainable   : "
            f"{other_trainable}"
        )

        print(
            f"grad=None         : "
            f"{grad_none}"
        )

        print(
            f"grad=0            : "
            f"{grad_zero}"
        )

        print(
            f"grad!=0           : "
            f"{grad_nonzero}"
        )

        if nonzero_norms:

            nonzero_norms.sort(
                key=lambda x: x[1],
                reverse=True,
            )

            print(
                "\nTop 10 non-zero "
                "gradient norms:"
            )

            for (
                name,
                norm,
            ) in nonzero_norms[:10]:

                print(
                    f"{norm:.8e}  "
                    f"{name}"
                )

        print("=" * 90)
        print()


# ============================================================
# 11. PEFT verification
# ============================================================

def verify_vision_only_lora(
    model,
) -> None:

    print("\n")
    print("=" * 90)
    print(
        "VERIFYING PEFT "
        "VISION-ONLY LoRA"
    )
    print("=" * 90)

    targeted = list(
        getattr(
            model,
            "targeted_module_names",
            [],
        )
    )

    print(
        f"Targeted PEFT modules: "
        f"{len(targeted)}"
    )

    for name in targeted:

        print(
            f"[TARGET] {name}"
        )

    # --------------------------------------------------------
    # Target count
    # --------------------------------------------------------

    if (
        len(targeted)
        != EXPECTED_TARGET_MODULES
    ):

        raise RuntimeError(
            "\nUnexpected number "
            "of LoRA target modules.\n"
            f"Expected: "
            f"{EXPECTED_TARGET_MODULES}\n"
            f"Actual: "
            f"{len(targeted)}\n"
            "\nSTOPPING before GRPO."
        )

    # --------------------------------------------------------
    # All targets must be visual
    # --------------------------------------------------------

    bad_targets = [
        name
        for name in targeted
        if "visual" not in name.lower()
    ]

    if bad_targets:

        raise RuntimeError(
            "PEFT targeted "
            "NON-VISION modules:\n"
            + "\n".join(
                bad_targets
            )
        )

    # --------------------------------------------------------
    # Trainable parameters
    # --------------------------------------------------------

    trainable_names = [
        name
        for name, param
        in model.named_parameters()
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

    print(
        f"Trainable tensors: "
        f"{len(trainable_names)}"
    )

    print(
        f"Vision trainable tensors: "
        f"{len(vision_trainable)}"
    )

    print(
        f"Non-vision trainable tensors: "
        f"{len(nonvision_trainable)}"
    )

    # --------------------------------------------------------
    # Hard safety checks
    # --------------------------------------------------------

    if nonvision_trainable:

        raise RuntimeError(
            "\nNON-VISION parameters "
            "became trainable:\n"
            + "\n".join(
                nonvision_trainable
            )
        )

    if (
        len(trainable_names)
        != EXPECTED_LORA_TENSORS
    ):

        raise RuntimeError(
            "\nUnexpected number "
            "of trainable tensors.\n"
            f"Expected: "
            f"{EXPECTED_LORA_TENSORS}\n"
            f"Actual: "
            f"{len(trainable_names)}"
        )

    print("\n")
    print(
        "PASS: LoRA is strictly "
        "Vision-only."
    )

    print(
        "PASS: "
        f"{EXPECTED_TARGET_MODULES} "
        "target modules."
    )

    print(
        "PASS: "
        f"{EXPECTED_LORA_TENSORS} "
        "trainable LoRA tensors."
    )

    print("=" * 90)
    print()


# ============================================================
# 12. GPU information
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

    print("=" * 90)

    print(
        "Native HF Vision-LoRA "
        "GRPO Gradient Diagnostic"
    )

    print("=" * 90)

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


# ============================================================
# 13. Training
# ============================================================

def run_diagnostic(
    args: argparse.Namespace,
) -> None:

    set_seed(
        args.seed
    )

    print_gpu_info()

    output_dir = (
        Path(args.output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Dataset
    # ========================================================

    print(
        "\nPreparing MapWise "
        "diagnostic dataset..."
    )

    train_dataset = (
        build_debug_dataset(
            qa_json=Path(
                args.qa_json
            ),
            image_root=Path(
                args.image_root
            ),
            num_samples=(
                args.num_samples
            ),
        )
    )

    # ========================================================
    # Processor
    # ========================================================

    print(
        "\nLoading processor..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_name,
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
    # 4-bit model
    # ========================================================

    print(
        "\nLoading Qwen3-VL "
        "4-bit base model..."
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=(
                "nf4"
            ),

            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),

            bnb_4bit_use_double_quant=True,
        )
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
    # Prepare QLoRA base
    # ========================================================

    print(
        "\nPreparing quantized model "
        "for PEFT..."
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
        "\nApplying Vision-only "
        "PEFT LoRA..."
    )

    lora_config = LoraConfig(

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

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ========================================================
    # Verify Vision-only
    # ========================================================

    verify_vision_only_lora(
        model
    )

    # ========================================================
    # GRPO config
    # ========================================================

    print(
        "\nCreating GRPOConfig..."
    )

    training_args = GRPOConfig(

        output_dir=str(
            output_dir
        ),

        # --------------------------------------------
        # Optimizer
        # --------------------------------------------

        learning_rate=(
            args.learning_rate
        ),

        adam_beta1=0.9,
        adam_beta2=0.99,

        weight_decay=0.0,

        max_grad_norm=1.0,

        # --------------------------------------------
        # Tiny diagnostic batch
        # --------------------------------------------

        per_device_train_batch_size=1,

        gradient_accumulation_steps=1,

        # One prompt group contains 2 rollouts.
        # Keep training batch = 1 for VRAM,
        # but generate 2 completions together for GRPO.
        generation_batch_size=2,

        # --------------------------------------------
        # GRPO
        # --------------------------------------------

        num_generations=(
            NUM_GENERATIONS
        ),

        temperature=(
            TEMPERATURE
        ),

        top_p=(
            TOP_P
        ),

        max_completion_length=(
            args.max_completion_length
        ),

        # --------------------------------------------
        # Only 3 steps
        # --------------------------------------------

        max_steps=(
            args.max_steps
        ),

        # --------------------------------------------
        # Native HF generation
        # --------------------------------------------

        use_vllm=False,

        # --------------------------------------------
        # Remove reference model KL
        # --------------------------------------------

        beta=0.0,

        # --------------------------------------------
        # IMPORTANT:
        #
        # Do not mask truncated completions in this
        # diagnostic. max_completion_length=256 may
        # truncate Thinking responses.
        # --------------------------------------------

        mask_truncated_completions=False,

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

        report_to="none",

        log_completions=True,

        num_completions_to_print=2,

        # --------------------------------------------
        # No checkpoint needed
        # --------------------------------------------

        save_strategy="no",

        eval_strategy="no",

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

    # ========================================================
    # Trainer
    # ========================================================

    print(
        "\nCreating native "
        "TRL GRPOTrainer..."
    )

    trainer = GRPOTrainer(

        model=model,

        args=training_args,

        processing_class=(
            processor
        ),

        reward_funcs=[
            diagnostic_reward
        ],

        train_dataset=(
            train_dataset
        ),
    )

    # ========================================================
    # Gradient callback
    # ========================================================

    trainer.add_callback(

        VisionGradientDebugCallback(
            model=model,
            max_steps_to_print=(
                args.max_steps
            ),
        )
    )

    # ========================================================
    # Pre-flight summary
    # ========================================================

    print("\n")
    print("=" * 90)
    print("PRE-FLIGHT CHECK")
    print("=" * 90)

    print(
        f"Model:                  "
        f"{args.model_name}"
    )

    print(
        f"Samples:                "
        f"{len(train_dataset)}"
    )

    print(
        f"num_generations:        "
        f"{NUM_GENERATIONS}"
    )

    print(
        f"max_completion_length:  "
        f"{args.max_completion_length}"
    )

    print(
        f"max_steps:              "
        f"{args.max_steps}"
    )

    print(
        f"LoRA rank:              "
        f"{args.lora_rank}"
    )

    print(
        f"LoRA alpha:             "
        f"{args.lora_alpha}"
    )

    print(
        "LoRA scope:             "
        "Vision blocks only"
    )

    print(
        "Language LoRA:          "
        "False"
    )

    print(
        "Synthetic rewards:      "
        "[0, 1]"
    )

    print(
        "mask truncated:         "
        "False"
    )

    print(
        "vLLM:                   "
        "False"
    )

    print(
        "Unsloth:                "
        "False"
    )

    print("=" * 90)

    # ========================================================
    # GRPO
    # ========================================================

    print(
        "\nStarting native "
        "Vision-LoRA GRPO "
        "gradient diagnostic..."
    )

    trainer.train()

    # ========================================================
    # Done
    # ========================================================

    print("\n")
    print("=" * 90)
    print("DIAGNOSTIC FINISHED")
    print("=" * 90)

    print(
        "\nInterpretation:"
    )

    print(
        "\nSTRONG SUCCESS:"
    )

    print(
        "  Vision trainable   = 216"
    )

    print(
        "  Language trainable = 0"
    )

    print(
        "  grad=None          = 0"
    )

    print(
        "  grad!=0            > 0"
    )

    print(
        "\nGRAPH CONNECTED BUT ZERO SIGNAL:"
    )

    print(
        "  grad=None = 0"
    )

    print(
        "  grad=0    = 216"
    )

    print(
        "\nGRAPH DISCONNECTED / PROBLEM:"
    )

    print(
        "  grad=None = 216"
    )

    print(
        "\nIf Native HF produces "
        "Vision-LoRA gradients while "
        "the equivalent Unsloth run "
        "produced grad=None for all "
        "216 tensors, this strongly "
        "suggests an Unsloth-specific "
        "backpropagation/execution-path "
        "issue."
    )


# ============================================================
# 14. CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Native HF Qwen3-VL "
            "Vision-LoRA GRPO "
            "gradient diagnostic."
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
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=MAX_COMPLETION_LENGTH,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
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
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


# ============================================================
# 15. Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.num_samples < 1:

        raise ValueError(
            "num_samples must be >= 1."
        )

    if args.max_steps < 1:

        raise ValueError(
            "max_steps must be >= 1."
        )

    if (
        args.max_completion_length
        < 64
    ):

        raise ValueError(
            "max_completion_length "
            "is too small."
        )

    run_diagnostic(
        args
    )


if __name__ == "__main__":

    try:

        main()

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