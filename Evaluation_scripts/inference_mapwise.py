#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapWise inference for a base VLM plus an optional PEFT/LoRA adapter checkpoint.

Typical GRPO checkpoint usage:
    base model + --adapter-path .../checkpoint-15

Important:
- The inference prompt does NOT expose ground_truth_type.
- Every checkpoint is evaluated with the same deterministic decoding setup.
- If --adapter-path is supplied, the output directory automatically includes the
  adapter/checkpoint name so different checkpoints cannot silently reuse each
  other's prediction JSON.
- Existing predictions are validated against model_name, thinking_mode, and
  adapter identity before resume is allowed.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Unsloth must be imported before transformers / peft.
import unsloth  # noqa: F401

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# ============================================================
# 1. Default configuration
# ============================================================

# Keep this identical to the base model used for GRPO training.
MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
DO_SAMPLE = False
THINKING_MODE = "auto"  # auto | on | off
PRINT_EVERY = 1
SAVE_EVERY = 1

SUPPORTED_IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"
)
SUPPORTED_COUNTRIES = {"china", "india", "usa"}


# ============================================================
# 2. Default project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAPWISE_JSON = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\Processed_Mapwise\statistic_with_annotations"
    r"\mapwise_reasoning_test_no_list_rank.json"
)

MAPWISE_IMAGE_ROOT = Path(
    r"C:\Users\junyhuang\Thesis\VLM_adaptation"
    r"\Datasets\mapwise-dataset"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Evaluation_results" / "Mapwise_GRPO_checkpoints"
DEFAULT_OUTPUT_JSON = None


# ============================================================
# 3. JSON and image utilities
# ============================================================

def make_qa_id(sample: dict[str, Any], index: int) -> str:
    country = str(sample.get("country", "unknown")).strip().lower()
    map_no = str(sample.get("map_no", "unknown")).strip()
    template_no = int(sample.get("template_no", -1))
    return f"mapwise_{country}_{map_no}_t{template_no}_idx{index:04d}"


def load_json_list(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist:\n{json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            "The top level of the MapWise JSON must be a list, "
            f"but received {type(data).__name__}."
        )
    if not data:
        raise ValueError(f"The MapWise JSON is empty:\n{json_path}")

    required_fields = (
        "country", "map_no", "question", "ground_truth", "ground_truth_type"
    )

    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample {index} must be a dictionary, "
                f"but received {type(sample).__name__}."
            )
        missing = [field for field in required_fields if field not in sample]
        if missing:
            raise KeyError(f"Sample {index} is missing required fields: {missing}")

        country = str(sample.get("country", "")).strip().lower()
        if country not in SUPPORTED_COUNTRIES:
            raise ValueError(
                f"Sample {index} has unsupported country={country!r}. "
                f"Expected one of {sorted(SUPPORTED_COUNTRIES)}."
            )

        if not str(sample.get("qa_id", "")).strip():
            sample["qa_id"] = make_qa_id(sample, index)

    return data


def validate_image(image_path: Path) -> None:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist:\n{image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(
            f"Image cannot be opened:\n{image_path}\nOriginal error: {error}"
        ) from error


def resolve_mapwise_image(sample: dict[str, Any], image_root: Path) -> Path:
    country = str(sample.get("country", "")).strip().lower()
    map_no = str(sample.get("map_no", "")).strip()
    qa_id = str(sample.get("qa_id", "unknown"))

    if country not in SUPPORTED_COUNTRIES:
        raise ValueError(f"Unsupported country={country!r} for qa_id={qa_id}.")
    if not map_no:
        raise ValueError(f"Sample {qa_id} has an empty map_no.")

    country_image_root = image_root / country / "images" / "with_annotations"
    if not country_image_root.exists():
        raise FileNotFoundError(
            f"Country image directory does not exist:\n{country_image_root}"
        )

    raw_path = Path(map_no)
    if raw_path.suffix:
        direct_path = country_image_root / raw_path
        if direct_path.exists():
            validate_image(direct_path)
            return direct_path.resolve()

    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = country_image_root / f"{map_no}{suffix}"
        if candidate.exists():
            validate_image(candidate)
            return candidate.resolve()

    target_stem = raw_path.stem.casefold()
    matches = [
        path for path in country_image_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
        and path.stem.casefold() == target_stem
    ]

    if len(matches) == 1:
        validate_image(matches[0])
        return matches[0].resolve()
    if len(matches) > 1:
        formatted = "\n".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple images match country={country!r}, map_no={map_no!r}, "
            f"qa_id={qa_id}:\n{formatted}"
        )

    tried = "\n".join(
        str(country_image_root / f"{map_no}{suffix}")
        for suffix in SUPPORTED_IMAGE_SUFFIXES
    )
    raise FileNotFoundError(
        f"No image found for sample {qa_id}, country={country!r}, "
        f"map_no={map_no!r}.\nTried:\n{tried}"
    )


# ============================================================
# 4. Prompt
# ============================================================

def build_mapwise_prompt(question: str, template_no: int | None = None) -> str:
    prompt = f"""
This is a cartographic reasoning question from the MapWise dataset.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()

    if template_no == 43:
        prompt += """

For this ranking question, express the final ranking explicitly using
"<", ">", and "=" as appropriate between every item. Use "=" only for ties.
Do not replace the ranking symbols with commas or prose.

Example:
Final answer: A < B = C
"""
    return prompt.strip()


def build_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ============================================================
# 5. Final-answer extraction and generation status
# ============================================================

def extract_final_answer(raw_response: Any) -> str:
    if raw_response is None:
        return ""
    text = str(raw_response).strip()
    if not text:
        return ""

    matches = list(re.finditer(r"final\s+answer\s*:\s*", text, flags=re.IGNORECASE))
    if matches:
        answer = text[matches[-1].end():].strip()
    elif "</think>" in text.lower():
        marker_index = text.lower().rfind("</think>")
        answer = text[marker_index + len("</think>"):].strip()
    else:
        answer = text

    answer = re.split(
        r"(?:<\|im_end\|>|<\|endoftext\|>|</s>)",
        answer,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return answer


def classify_generation_status(
    raw_response: str,
    final_answer: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> str:
    if not raw_response.strip():
        return "empty"

    has_marker = bool(re.search(r"final\s+answer\s*:", raw_response, flags=re.IGNORECASE))
    reached_limit = generated_tokens >= max_new_tokens

    if has_marker and final_answer.strip():
        return "complete"
    if reached_limit:
        return "truncated"
    if final_answer.strip():
        return "fallback_extracted"
    return "not_extractable"


# ============================================================
# 6. Base model + adapter loading
# ============================================================

def print_gpu_information() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch did not detect a CUDA GPU.\n"
            f"torch version: {torch.__version__}\n"
            f"torch CUDA build: {torch.version.cuda}"
        )
    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU VRAM: {properties.total_memory / 1024**3:.2f} GB", flush=True)


def normalize_adapter_path(adapter_path: Optional[Path]) -> Optional[Path]:
    if adapter_path is None:
        return None

    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Adapter checkpoint directory not found:\n{path}")

    adapter_config = path / "adapter_config.json"
    if not adapter_config.is_file():
        raise FileNotFoundError(
            "The supplied --adapter-path does not look like a PEFT/LoRA checkpoint.\n"
            f"Missing: {adapter_config}"
        )
    return path


def load_model_and_processor(
    model_name: str = MODEL_NAME,
    adapter_path: Optional[Path] = None,
):
    """Load the exact GRPO base model and optionally attach one PEFT adapter."""
    adapter_path = normalize_adapter_path(adapter_path)

    print("\nLoading VLM...", flush=True)
    print(f"Base model: {model_name}", flush=True)
    print(f"Adapter:    {adapter_path if adapter_path else 'None (baseline)'}", flush=True)
    print_gpu_information()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

    model.eval()

    model_type = getattr(getattr(model, "config", None), "model_type", "unknown")
    print(f"Detected model type: {model_type}", flush=True)
    print("Model and processor loaded successfully.", flush=True)
    return model, processor, adapter_path


# ============================================================
# 7. Single-sample inference
# ============================================================

def move_inputs_to_model_device(inputs: dict[str, Any], model) -> dict[str, Any]:
    model_device = next(model.parameters()).device
    if hasattr(inputs, "to"):
        try:
            return inputs.to(model_device)
        except Exception:
            pass
    return {
        key: value.to(model_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def prepare_multimodal_inputs(
    processor,
    image: Image.Image,
    prompt: str,
    *,
    thinking_mode: str = THINKING_MODE,
):
    thinking_mode = str(thinking_mode).casefold().strip()
    if thinking_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"thinking_mode must be one of auto/on/off, got {thinking_mode!r}."
        )

    messages = build_messages(image=image, prompt=prompt)
    template_kwargs = {}
    if thinking_mode != "auto":
        template_kwargs["enable_thinking"] = thinking_mode == "on"

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
    except Exception as primary_error:
        placeholder_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            rendered_text = processor.apply_chat_template(
                placeholder_messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            try:
                return processor(
                    images=image,
                    text=rendered_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            except TypeError:
                return processor(
                    images=image,
                    text=rendered_text,
                    return_tensors="pt",
                )
        except Exception as fallback_error:
            raise RuntimeError(
                "The model processor could not prepare the multimodal input "
                "with either the standard or fallback chat-template path.\n"
                f"thinking_mode={thinking_mode!r}\n"
                f"Primary error: {type(primary_error).__name__}: {primary_error}\n"
                f"Fallback error: {type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error


def generate_response(
    model,
    processor,
    image_path: Path,
    prompt: str,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = DO_SAMPLE,
    thinking_mode: str = THINKING_MODE,
) -> tuple[str, int]:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        inputs = prepare_multimodal_inputs(
            processor=processor,
            image=image,
            prompt=prompt,
            thinking_mode=thinking_mode,
        )

    inputs = move_inputs_to_model_device(inputs, model)
    if "input_ids" not in inputs:
        raise KeyError(
            "Prepared model inputs do not contain input_ids. "
            f"Available keys: {list(inputs.keys())}"
        )

    input_length = inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
    }

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_token_ids = generated_ids[:, input_length:]
    generated_tokens = int(generated_token_ids.shape[1])
    output_text = processor.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_token_ids
    return output_text.strip(), generated_tokens


# ============================================================
# 8. Output naming and checkpoint-safe resume
# ============================================================

def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return value or "unnamed"


def model_output_dir_name(
    model_name: str,
    thinking_mode: str = "auto",
    adapter_path: Optional[Path] = None,
) -> str:
    short_name = safe_name(str(model_name).rstrip("/").split("/")[-1])
    mode = str(thinking_mode or "auto").strip().lower()
    if mode in {"on", "off"}:
        short_name += f"_thinking-{mode}"

    if adapter_path is None:
        return f"{short_name}_baseline"

    adapter_name = safe_name(Path(adapter_path).name)
    return f"{short_name}_{adapter_name}"


def resolve_output_json(
    *,
    model_name: str,
    thinking_mode: str,
    adapter_path: Optional[Path],
    output_json: Optional[Path],
) -> Path:
    if output_json is not None:
        return Path(output_json).resolve()

    return (
        DEFAULT_OUTPUT_DIR
        / model_output_dir_name(model_name, thinking_mode, adapter_path)
        / "mapwise_predictions.json"
    ).resolve()


def load_existing_predictions(output_json: Path) -> list[dict[str, Any]]:
    if not output_json.exists():
        return []
    with output_json.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise TypeError(
            f"Existing prediction file must contain a list:\n{output_json}"
        )
    return [record for record in data if isinstance(record, dict)]


def adapter_identity(adapter_path: Optional[Path]) -> str:
    return "" if adapter_path is None else str(Path(adapter_path).resolve())


def validate_prediction_configuration(
    predictions: list[dict[str, Any]],
    model_name: str,
    thinking_mode: str,
    adapter_path: Optional[Path],
) -> None:
    previous_models = {
        str(record.get("model_name", "")).strip()
        for record in predictions
        if str(record.get("model_name", "")).strip()
    }
    if previous_models and previous_models != {model_name}:
        raise ValueError(
            "Existing prediction JSON contains results from a different base model.\n"
            f"Existing model(s): {sorted(previous_models)}\n"
            f"Requested model:   {model_name}\n"
            "Use a different --output-json path or pass --overwrite."
        )

    previous_modes = {
        str(record.get("thinking_mode", "auto") or "auto").casefold().strip()
        for record in predictions
    }
    requested_mode = str(thinking_mode).casefold().strip()
    if predictions and previous_modes != {requested_mode}:
        raise ValueError(
            "Existing prediction JSON contains a different thinking mode.\n"
            f"Existing mode(s): {sorted(previous_modes)}\n"
            f"Requested mode:   {requested_mode}\n"
            "Use a different --output-json path or pass --overwrite."
        )

    requested_adapter = adapter_identity(adapter_path)
    previous_adapters = {
        str(record.get("adapter_path", "") or "").strip()
        for record in predictions
    }
    if predictions and previous_adapters != {requested_adapter}:
        raise ValueError(
            "Existing prediction JSON belongs to a different adapter checkpoint.\n"
            f"Existing adapter(s): {sorted(previous_adapters)}\n"
            f"Requested adapter:   {requested_adapter or 'None (baseline)'}\n"
            "Use a different --output-json path or pass --overwrite."
        )


def save_predictions(predictions: list[dict[str, Any]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_json.with_suffix(output_json.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_json)


def completed_qa_ids(predictions: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for record in predictions:
        qa_id = record.get("qa_id")
        raw_response = record.get("raw_response")
        if qa_id is not None and str(raw_response or "").strip():
            completed.add(str(qa_id))
    return completed


def replace_or_append_prediction(
    predictions: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> None:
    new_id = str(new_record["qa_id"])
    for index, record in enumerate(predictions):
        if str(record.get("qa_id", "")) == new_id:
            predictions[index] = new_record
            return
    predictions.append(new_record)


# ============================================================
# 9. Full MapWise inference
# ============================================================

def build_prediction_record(
    *,
    sample: dict[str, Any],
    sample_index: int,
    image_path: Path,
    raw_response: str,
    generated_tokens: int,
    model_name: str,
    adapter_path: Optional[Path],
    thinking_mode: str,
    max_new_tokens: int,
    inference_seconds: float,
) -> dict[str, Any]:
    qa_id = str(sample.get("qa_id", make_qa_id(sample, sample_index)))
    final_answer = extract_final_answer(raw_response)
    generation_status = classify_generation_status(
        raw_response=raw_response,
        final_answer=final_answer,
        generated_tokens=generated_tokens,
        max_new_tokens=max_new_tokens,
    )

    adapter_str = adapter_identity(adapter_path)
    return {
        "qa_id": qa_id,
        "sample_index": sample_index,
        "country": str(sample.get("country", "")).strip().lower(),
        "map_type": str(sample.get("map_type", "")).strip(),
        "map_no": str(sample.get("map_no", "")).strip(),
        "template_no": int(sample.get("template_no", -1)),
        "question": str(sample.get("question", "")).strip(),
        "ground_truth": str(sample.get("ground_truth", "")).strip(),
        "ground_truth_type": str(sample.get("ground_truth_type", "")).strip(),
        "c_or_d": str(sample.get("c_or_d", "")).strip(),
        "relative_region": str(sample.get("relative_region", "")).strip(),
        "source_index": sample.get("source_index"),
        "data_group_id": str(sample.get("data_group_id", "")).strip(),
        "legend_style": str(
            sample.get("legend_style", sample.get("c_or_d", ""))
        ).strip(),
        "reasoning_eligible": sample.get("reasoning_eligible"),
        "split": str(sample.get("split", "")).strip(),
        "resolved_image_path": str(image_path),
        "model_name": model_name,
        "adapter_path": adapter_str,
        "adapter_name": Path(adapter_str).name if adapter_str else "baseline",
        "thinking_mode": thinking_mode,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated_tokens,
        "generation_status": generation_status,
        "raw_response": raw_response,
        "final_answer": final_answer,
        "inference_seconds": round(inference_seconds, 3),
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def run_mapwise_inference(
    *,
    model_name: str = MODEL_NAME,
    adapter_path: Optional[Path] = None,
    qa_json: Path = MAPWISE_JSON,
    image_root: Path = MAPWISE_IMAGE_ROOT,
    output_json: Optional[Path] = DEFAULT_OUTPUT_JSON,
    max_new_tokens: int = MAX_NEW_TOKENS,
    thinking_mode: str = THINKING_MODE,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
    save_every: int = SAVE_EVERY,
    print_every: int = PRINT_EVERY,
) -> Path:
    qa_json = Path(qa_json).resolve()
    image_root = Path(image_root).resolve()
    adapter_path = normalize_adapter_path(adapter_path)

    output_json = resolve_output_json(
        model_name=model_name,
        thinking_mode=thinking_mode,
        adapter_path=adapter_path,
        output_json=output_json,
    )

    thinking_mode = str(thinking_mode).casefold().strip()
    if thinking_mode not in {"auto", "on", "off"}:
        raise ValueError(
            f"thinking_mode must be one of auto/on/off, got {thinking_mode!r}."
        )

    samples = load_json_list(qa_json)
    dataset_size = len(samples)
    if end_index is None:
        end_index = dataset_size

    if not 0 <= start_index <= dataset_size:
        raise IndexError(f"start_index={start_index} is outside 0..{dataset_size}.")
    if not start_index <= end_index <= dataset_size:
        raise IndexError(
            f"end_index={end_index} must be between start_index={start_index} "
            f"and {dataset_size}."
        )
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    if overwrite and output_json.exists():
        output_json.unlink()

    if resume and not overwrite:
        predictions = load_existing_predictions(output_json)
        validate_prediction_configuration(
            predictions,
            model_name,
            thinking_mode,
            adapter_path,
        )
    else:
        predictions = []

    completed = completed_qa_ids(predictions)
    selected_total = end_index - start_index

    print("=" * 80)
    print("MAPWISE GRPO CHECKPOINT INFERENCE")
    print("=" * 80)
    print(f"Base model:     {model_name}")
    print(f"Adapter:        {adapter_path if adapter_path else 'None (baseline)'}")
    print(f"QA JSON:        {qa_json}")
    print(f"Image root:     {image_root}")
    print(f"Output JSON:    {output_json}")
    print(f"Dataset size:   {dataset_size}")
    print(f"Selected:       [{start_index}, {end_index}) = {selected_total}")
    print(f"Resume:         {resume}")
    print(f"Already done:   {len(completed)}")
    print(f"Max tokens:     {max_new_tokens}")
    print(f"Thinking mode:  {thinking_mode}")
    print(f"Sampling:       {DO_SAMPLE}")

    model, processor, adapter_path = load_model_and_processor(
        model_name=model_name,
        adapter_path=adapter_path,
    )

    newly_completed = 0
    skipped = 0
    failed = 0
    run_start_time = time.perf_counter()

    try:
        for absolute_index in range(start_index, end_index):
            sample = samples[absolute_index]
            qa_id = str(sample["qa_id"])

            if resume and qa_id in completed:
                skipped += 1
                if print_every and (
                    (absolute_index - start_index + 1) % print_every == 0
                ):
                    print(
                        f"[{absolute_index + 1}/{dataset_size}] "
                        f"{qa_id}: skipped (already completed)",
                        flush=True,
                    )
                continue

            sample_start_time = time.perf_counter()
            try:
                question = str(sample["question"]).strip()
                image_path = resolve_mapwise_image(sample, image_root)
                prompt = build_mapwise_prompt(
                    question=question,
                    template_no=sample.get("template_no"),
                )

                raw_response, generated_tokens = generate_response(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                    thinking_mode=thinking_mode,
                )

                inference_seconds = time.perf_counter() - sample_start_time
                prediction_record = build_prediction_record(
                    sample=sample,
                    sample_index=absolute_index,
                    image_path=image_path,
                    raw_response=raw_response,
                    generated_tokens=generated_tokens,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    thinking_mode=thinking_mode,
                    max_new_tokens=max_new_tokens,
                    inference_seconds=inference_seconds,
                )

                replace_or_append_prediction(predictions, prediction_record)
                completed.add(qa_id)
                newly_completed += 1

                if newly_completed % save_every == 0:
                    save_predictions(predictions, output_json)

                if print_every and newly_completed % print_every == 0:
                    preview = prediction_record["final_answer"].replace("\n", " ")[:120]
                    print(
                        f"[{absolute_index + 1}/{dataset_size}] {qa_id} | "
                        f"country={sample.get('country')} | map={sample.get('map_no')} | "
                        f"tokens={generated_tokens} | "
                        f"status={prediction_record['generation_status']} | "
                        f"time={inference_seconds:.1f}s | answer={preview!r}",
                        flush=True,
                    )

            except KeyboardInterrupt:
                raise
            except Exception as error:
                failed += 1
                inference_seconds = time.perf_counter() - sample_start_time
                error_record = {
                    "qa_id": qa_id,
                    "sample_index": absolute_index,
                    "country": str(sample.get("country", "")).strip().lower(),
                    "map_type": str(sample.get("map_type", "")).strip(),
                    "map_no": str(sample.get("map_no", "")).strip(),
                    "template_no": sample.get("template_no"),
                    "question": str(sample.get("question", "")).strip(),
                    "ground_truth": str(sample.get("ground_truth", "")).strip(),
                    "ground_truth_type": str(
                        sample.get("ground_truth_type", "")
                    ).strip(),
                    "c_or_d": str(sample.get("c_or_d", "")).strip(),
                    "relative_region": str(
                        sample.get("relative_region", "")
                    ).strip(),
                    "model_name": model_name,
                    "adapter_path": adapter_identity(adapter_path),
                    "adapter_name": (
                        adapter_path.name if adapter_path is not None else "baseline"
                    ),
                    "thinking_mode": thinking_mode,
                    "max_new_tokens": max_new_tokens,
                    "generated_tokens": 0,
                    "generation_status": "error",
                    "raw_response": "",
                    "final_answer": "",
                    "inference_seconds": round(inference_seconds, 3),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "generated_at": datetime.now().astimezone().isoformat(),
                }
                replace_or_append_prediction(predictions, error_record)
                save_predictions(predictions, output_json)

                print(
                    f"[{absolute_index + 1}/{dataset_size}] {qa_id}: FAILED - "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print(
            "\nInference interrupted by user. Saving completed results...",
            file=sys.stderr,
            flush=True,
        )
    finally:
        save_predictions(predictions, output_json)
        total_seconds = time.perf_counter() - run_start_time
        print("\n" + "=" * 80)
        print("INFERENCE SUMMARY")
        print("=" * 80)
        print(f"Newly completed: {newly_completed}")
        print(f"Skipped:         {skipped}")
        print(f"Failed:          {failed}")
        print(f"Elapsed time:    {total_seconds / 60:.2f} minutes")
        print(f"Predictions:     {output_json}")

        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_json


# ============================================================
# 10. CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MapWise predictions for a base VLM plus an optional "
            "GRPO LoRA checkpoint."
        )
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help=(
            "Path to a PEFT/LoRA checkpoint directory, e.g. checkpoint-15. "
            "Omit for baseline inference."
        ),
    )
    parser.add_argument("--qa-json", type=Path, default=MAPWISE_JSON)
    parser.add_argument("--image-root", type=Path, default=MAPWISE_IMAGE_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default=THINKING_MODE,
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    return run_mapwise_inference(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        qa_json=args.qa_json,
        image_root=args.image_root,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
        thinking_mode=args.thinking,
        start_index=args.start_index,
        end_index=args.end_index,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        save_every=args.save_every,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nProgram failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise