# python3
# -*- coding: utf-8 -*-
"""
Run Qwen3-VL-8B-Thinking inference on the complete FRIEDA test set.

This module is intentionally responsible only for inference:

    FRIEDA test JSON + map images
                ↓
        Qwen3-VL generation
                ↓
       FRIEDA prediction JSON

It does not calculate accuracy. The generated JSON is designed to be consumed
by frieda_evaluation.py in the same directory.
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

# Unsloth must be imported before transformers.
import unsloth  # noqa: F401

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel


# ============================================================
# 1. Default configuration
# ============================================================

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
DO_SAMPLE = False
PRINT_EVERY = 1
SAVE_EVERY = 1


# ============================================================
# 2. Default project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

FRIEDA_ROOT = PROJECT_ROOT / "Test_data" / "FRIEDA_test"
FRIEDA_JSON = FRIEDA_ROOT / "frieda_test.json"
FRIEDA_IMAGE_ROOT = FRIEDA_ROOT / "image"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Evaluation_results" / "FRIEDA_Qwen3VL"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "frieda_predictions.json"


# ============================================================
# 3. JSON and image utilities
# ============================================================

def load_json_list(json_path: Path) -> list[dict[str, Any]]:
    """Load a JSON file whose top level must be a non-empty list."""
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist:\n{json_path}")

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            "The top level of the FRIEDA JSON must be a list, "
            f"but received {type(data).__name__}."
        )

    if not data:
        raise ValueError(f"The FRIEDA JSON is empty:\n{json_path}")

    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample {index} must be a dictionary, "
                f"but received {type(sample).__name__}."
            )

    return data


def validate_image(image_path: Path) -> None:
    """Verify that an image exists and can be opened by Pillow."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist:\n{image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image path is not a file:\n{image_path}")

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        raise RuntimeError(
            f"Image cannot be opened:\n{image_path}\n"
            f"Original error: {error}"
        ) from error


def resolve_frieda_images(
    sample: dict[str, Any],
    image_root: Path,
) -> list[Path]:
    """Resolve FRIEDA image_urls relative to FRIEDA_test/image/."""
    image_urls = sample.get("image_urls")

    if not isinstance(image_urls, list) or not image_urls:
        question_ref = sample.get("question_ref", "unknown")
        raise ValueError(
            f"Sample {question_ref} does not contain a valid image_urls list."
        )

    image_paths: list[Path] = []

    for relative_path in image_urls:
        normalized_relative_path = str(relative_path).replace("\\", "/")
        image_path = image_root / Path(normalized_relative_path)
        validate_image(image_path)
        image_paths.append(image_path.resolve())

    return image_paths


# ============================================================
# 4. Prompt and multimodal message construction
# ============================================================

def build_frieda_prompt(question: str) -> str:
    """Build a stable prompt for FRIEDA open-ended cartographic QA."""
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


def build_messages(
    image_paths: list[Path],
    prompt: str,
) -> list[dict[str, Any]]:
    """Insert images in image_urls order, followed by the text prompt."""
    content: list[dict[str, Any]] = []

    for image_path in image_paths:
        content.append({"type": "image", "image": str(image_path)})

    content.append({"type": "text", "text": prompt})

    return [{"role": "user", "content": content}]


# ============================================================
# 5. Final-answer extraction
# ============================================================

def extract_final_answer(raw_response: Any) -> str:
    """Extract text after the last Final answer: marker."""
    if raw_response is None:
        return ""

    text = str(raw_response).strip()
    if not text:
        return ""

    matches = list(
        re.finditer(r"final\s+answer\s*:\s*", text, flags=re.IGNORECASE)
    )

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


# ============================================================
# 6. Model loading
# ============================================================

def print_gpu_information() -> None:
    """Print the active CUDA device and memory capacity."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch did not detect a CUDA GPU.\n"
            f"torch version: {torch.__version__}\n"
            f"torch CUDA build: {torch.version.cuda}"
        )

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"GPU VRAM: {properties.total_memory / 1024**3:.2f} GB", flush=True)


def load_model_and_processor(
    model_name: str = MODEL_NAME,
    adapter_path: Optional[Path] = None,
):
    """Load the baseline model or the base model with a PEFT/LoRA adapter."""
    print("\nLoading Qwen3-VL model...", flush=True)
    print(f"Base model: {model_name}", flush=True)
    print_gpu_information()

    resolved_adapter: Optional[Path] = None
    if adapter_path is not None:
        resolved_adapter = Path(adapter_path).expanduser().resolve()
        if not resolved_adapter.exists():
            raise FileNotFoundError(
                f"Adapter directory does not exist:\n{resolved_adapter}"
            )
        if not (resolved_adapter / "adapter_config.json").exists():
            raise FileNotFoundError(
                "The adapter directory does not contain adapter_config.json:\n"
                f"{resolved_adapter}"
            )
        print(f"Adapter:    {resolved_adapter}", flush=True)
    else:
        print("Adapter:    None (baseline)", flush=True)

    # Use the base-model processor for both baseline and adapted inference so
    # preprocessing and chat-template behavior remain identical.
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

    if resolved_adapter is None:
        model = base_model
    else:
        model = PeftModel.from_pretrained(
            base_model,
            str(resolved_adapter),
            is_trainable=False,
        )

    model.eval()
    print("Model and processor loaded successfully.", flush=True)
    return model, processor


# ============================================================
# 7. Single-sample inference
# ============================================================

def move_inputs_to_model_device(
    inputs: dict[str, Any],
    model,
) -> dict[str, Any]:
    """Move tensor inputs to the first model device."""
    model_device = next(model.parameters()).device
    return {
        key: value.to(model_device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def generate_response(
    model,
    processor,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool = DO_SAMPLE,
) -> str:
    """Generate one complete Qwen3-VL response."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = move_inputs_to_model_device(inputs, model)
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
        )

    generated_tokens = generated_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    del inputs, generated_ids, generated_tokens
    return output_text.strip()


# ============================================================
# 8. Prediction persistence and resume support
# ============================================================

def load_existing_predictions(output_json: Path) -> list[dict[str, Any]]:
    """Load an existing prediction list for resume mode."""
    if not output_json.exists():
        return []

    with output_json.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            f"Existing prediction file must contain a list:\n{output_json}"
        )

    return [record for record in data if isinstance(record, dict)]


def save_predictions(
    predictions: list[dict[str, Any]],
    output_json: Path,
) -> None:
    """Atomically save predictions through a temporary file."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_json.with_suffix(output_json.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)

    temporary_path.replace(output_json)


def completed_question_refs(
    predictions: list[dict[str, Any]],
) -> set[str]:
    """Return IDs that already have a non-empty raw response."""
    completed: set[str] = set()

    for record in predictions:
        question_ref = record.get("question_ref")
        raw_response = record.get("raw_response")

        if question_ref is not None and str(raw_response or "").strip():
            completed.add(str(question_ref))

    return completed


def replace_or_append_prediction(
    predictions: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> None:
    """Replace a record with the same question_ref or append it."""
    new_id = str(new_record["question_ref"])

    for index, record in enumerate(predictions):
        if str(record.get("question_ref", "")) == new_id:
            predictions[index] = new_record
            return

    predictions.append(new_record)


# ============================================================
# 9. Full FRIEDA inference
# ============================================================

def build_prediction_record(
    *,
    sample: dict[str, Any],
    sample_index: int,
    image_paths: list[Path],
    raw_response: str,
    model_name: str,
    adapter_path: Optional[Path],
    max_new_tokens: int,
    inference_seconds: float,
) -> dict[str, Any]:
    """Create one self-contained prediction record."""
    question_ref = str(
        sample.get("question_ref", f"frieda_{sample_index}")
    )

    return {
        "question_ref": question_ref,
        "sample_index": sample_index,
        "question_text": str(sample.get("question_text", "")).strip(),
        "expected_answer": str(sample.get("expected_answer", "")).strip(),
        "answer_type": str(sample.get("answer_type", "textual")).strip(),
        "spatial_relationship": str(
            sample.get("spatial_relationship", "")
        ).strip(),
        "map_count": str(sample.get("map_count", "")).strip(),
        "domain": str(sample.get("domain", "")).strip(),
        "image_urls": list(sample.get("image_urls", [])),
        "resolved_image_paths": [str(path) for path in image_paths],
        "base_model_name": model_name,
        "adapter_path": (
            str(Path(adapter_path).expanduser().resolve())
            if adapter_path is not None
            else None
        ),
        "model_variant": "adapted" if adapter_path is not None else "baseline",
        "max_new_tokens": max_new_tokens,
        "raw_response": raw_response,
        "final_answer": extract_final_answer(raw_response),
        "inference_seconds": round(inference_seconds, 3),
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def run_frieda_inference(
    *,
    model_name: str = MODEL_NAME,
    adapter_path: Optional[Path] = None,
    qa_json: Path = FRIEDA_JSON,
    image_root: Path = FRIEDA_IMAGE_ROOT,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    max_new_tokens: int = MAX_NEW_TOKENS,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
    save_every: int = SAVE_EVERY,
    print_every: int = PRINT_EVERY,
) -> Path:
    """Run inference and return the saved prediction JSON path."""
    qa_json = Path(qa_json).resolve()
    image_root = Path(image_root).resolve()
    output_json = Path(output_json).resolve()

    samples = load_json_list(qa_json)
    dataset_size = len(samples)

    if end_index is None:
        end_index = dataset_size

    if not 0 <= start_index <= dataset_size:
        raise IndexError(
            f"start_index={start_index} is outside 0..{dataset_size}."
        )

    if not start_index <= end_index <= dataset_size:
        raise IndexError(
            f"end_index={end_index} must be between "
            f"start_index={start_index} and {dataset_size}."
        )

    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    if overwrite and output_json.exists():
        output_json.unlink()

    if resume and not overwrite:
        predictions = load_existing_predictions(output_json)
    else:
        predictions = []

    completed = completed_question_refs(predictions)
    selected_total = end_index - start_index

    print("=" * 80)
    print("FRIEDA QWEN3-VL INFERENCE")
    print("=" * 80)
    print(f"QA JSON:       {qa_json}")
    print(f"Image root:    {image_root}")
    print(f"Output JSON:   {output_json}")
    print(f"Dataset size:  {dataset_size}")
    print(f"Selected:      [{start_index}, {end_index}) = {selected_total}")
    print(f"Resume:        {resume}")
    print(f"Already done:  {len(completed)}")
    print(f"Max tokens:    {max_new_tokens}")
    print(
        "Adapter:       "
        + (str(Path(adapter_path).expanduser().resolve()) if adapter_path else "None (baseline)")
    )

    model, processor = load_model_and_processor(
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
            question_ref = str(
                sample.get("question_ref", f"frieda_{absolute_index}")
            )

            if resume and question_ref in completed:
                skipped += 1
                if print_every and (
                    (absolute_index - start_index + 1) % print_every == 0
                ):
                    print(
                        f"[{absolute_index + 1}/{dataset_size}] "
                        f"{question_ref}: skipped (already completed)",
                        flush=True,
                    )
                continue

            sample_start_time = time.perf_counter()

            try:
                question = str(sample["question_text"]).strip()
                image_paths = resolve_frieda_images(
                    sample=sample,
                    image_root=image_root,
                )

                prompt = build_frieda_prompt(question)
                messages = build_messages(
                    image_paths=image_paths,
                    prompt=prompt,
                )

                raw_response = generate_response(
                    model=model,
                    processor=processor,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=DO_SAMPLE,
                )

                inference_seconds = time.perf_counter() - sample_start_time

                prediction_record = build_prediction_record(
                    sample=sample,
                    sample_index=absolute_index,
                    image_paths=image_paths,
                    raw_response=raw_response,
                    model_name=model_name,
                    adapter_path=adapter_path,
                    max_new_tokens=max_new_tokens,
                    inference_seconds=inference_seconds,
                )

                replace_or_append_prediction(predictions, prediction_record)
                completed.add(question_ref)
                newly_completed += 1

                if newly_completed % save_every == 0:
                    save_predictions(predictions, output_json)

                if print_every and newly_completed % print_every == 0:
                    preview = prediction_record["final_answer"].replace("\n", " ")[:120]
                    print(
                        f"[{absolute_index + 1}/{dataset_size}] "
                        f"{question_ref} | images={len(image_paths)} | "
                        f"time={inference_seconds:.1f}s | answer={preview!r}",
                        flush=True,
                    )

            except KeyboardInterrupt:
                raise

            except Exception as error:
                failed += 1
                inference_seconds = time.perf_counter() - sample_start_time

                error_record = {
                    "question_ref": question_ref,
                    "sample_index": absolute_index,
                    "question_text": str(sample.get("question_text", "")).strip(),
                    "expected_answer": str(sample.get("expected_answer", "")).strip(),
                    "answer_type": str(sample.get("answer_type", "textual")).strip(),
                    "spatial_relationship": str(sample.get("spatial_relationship", "")).strip(),
                    "map_count": str(sample.get("map_count", "")).strip(),
                    "domain": str(sample.get("domain", "")).strip(),
                    "image_urls": list(sample.get("image_urls", [])),
                    "base_model_name": model_name,
                    "adapter_path": (
                        str(Path(adapter_path).expanduser().resolve())
                        if adapter_path is not None
                        else None
                    ),
                    "model_variant": (
                        "adapted" if adapter_path is not None else "baseline"
                    ),
                    "max_new_tokens": max_new_tokens,
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
                    f"[{absolute_index + 1}/{dataset_size}] "
                    f"{question_ref}: FAILED - "
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

    return output_json


# ============================================================
# 10. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3-VL-8B-Thinking predictions for the "
            "FRIEDA test set."
        )
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional PEFT/LoRA adapter directory. Omit for baseline inference.",
    )
    parser.add_argument("--qa-json", type=Path, default=FRIEDA_JSON)
    parser.add_argument("--image-root", type=Path, default=FRIEDA_IMAGE_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)

    return parser.parse_args()


def main() -> Path:
    args = parse_args()

    return run_frieda_inference(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        qa_json=args.qa_json,
        image_root=args.image_root,
        output_json=args.output_json,
        max_new_tokens=args.max_new_tokens,
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