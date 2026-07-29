# python3
# -*- coding: utf-8 -*-
"""
Run Qwen3-VL-8B-Thinking inference on the MapWise India test set.

This module is intentionally responsible only for inference:

    MapWise test JSON + map image
                ↓
        Qwen3-VL generation
                ↓
       MapWise prediction JSON

It does not calculate evaluation metrics. The generated JSON is designed to be
consumed by mapwise_evaluation.py in the same directory.
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


# ============================================================
# 1. Default configuration
# ============================================================

MODEL_NAME = "unsloth/Qwen3-VL-8B-Thinking-unsloth-bnb-4bit"
MAX_NEW_TOKENS = 3072
DO_SAMPLE = False
PRINT_EVERY = 1
SAVE_EVERY = 1

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
# 2. Default project paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

MAPWISE_ROOT = PROJECT_ROOT / "Test_data" / "Mapwise_india"
MAPWISE_JSON = MAPWISE_ROOT / "india_test_75_balanced.json"
MAPWISE_IMAGE_ROOT = MAPWISE_ROOT / "image"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Evaluation_results" / "Mapwise_Qwen3VL"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "mapwise_predictions.json"


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
            "The top level of the MapWise JSON must be a list, "
            f"but received {type(data).__name__}."
        )

    if not data:
        raise ValueError(f"The MapWise JSON is empty:\n{json_path}")

    for index, sample in enumerate(data):
        if not isinstance(sample, dict):
            raise TypeError(
                f"Sample {index} must be a dictionary, "
                f"but received {type(sample).__name__}."
            )

        required_fields = (
            "qa_id",
            "map_no",
            "question",
            "ground_truth",
            "ground_truth_type",
        )
        missing = [field for field in required_fields if field not in sample]
        if missing:
            raise KeyError(
                f"Sample {index} is missing required fields: {missing}"
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


def resolve_mapwise_image(
    sample: dict[str, Any],
    image_root: Path,
) -> Path:
    """
    Resolve one MapWise image from sample["map_no"].

    map_no normally omits the extension, for example:
        map101_2D

    The function first tries common image extensions directly under image_root.
    If no direct match is found, it performs a recursive, case-insensitive stem
    search.
    """
    map_no = str(sample.get("map_no", "")).strip()
    qa_id = str(sample.get("qa_id", "unknown"))

    if not map_no:
        raise ValueError(f"Sample {qa_id} has an empty map_no.")

    raw_path = Path(map_no)

    # Support map_no values that already contain an extension or subdirectory.
    if raw_path.suffix:
        direct_path = image_root / raw_path
        if direct_path.exists():
            validate_image(direct_path)
            return direct_path.resolve()

    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = image_root / f"{map_no}{suffix}"
        if candidate.exists():
            validate_image(candidate)
            return candidate.resolve()

    target_stem = raw_path.stem.casefold()
    matches = [
        path
        for path in image_root.rglob("*")
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
            f"Multiple image files match map_no={map_no!r} for {qa_id}:\n"
            f"{formatted}"
        )

    tried = "\n".join(
        str(image_root / f"{map_no}{suffix}")
        for suffix in SUPPORTED_IMAGE_SUFFIXES
    )
    raise FileNotFoundError(
        f"No image found for sample {qa_id}, map_no={map_no!r}.\n"
        f"Tried:\n{tried}"
    )


# ============================================================
# 4. Prompt and multimodal message construction
# ============================================================

def answer_format_instruction(
    ground_truth_type: str,
    template_no: int,
) -> str:
    """Return a type-specific final-answer formatting instruction."""
    answer_type = ground_truth_type.casefold().strip()

    if template_no == 43:
        return (
            "This is a ranking question. Return all requested states in ranked "
            "order using only '<', '>', and '=' to express relations. Preserve "
            "ties when states share the same rank."
        )

    if answer_type == "binary":
        return "Return only Yes or No in the final answer."

    if answer_type == "count":
        return "Return one integer in the final answer."

    if answer_type == "range":
        return (
            "Return the two range endpoints exactly as represented by the map "
            "legend, including k notation when appropriate."
        )

    if answer_type == "list":
        return (
            "Return every required state and no unrelated states. Separate "
            "state names with commas."
        )

    if answer_type == "single":
        return "Return only the requested state name or direct short answer."

    return "Return a concise direct answer."


def build_mapwise_prompt(
    question: str,
    ground_truth_type: str,
    template_no: int,
) -> str:
    """Build a stable prompt for MapWise choropleth-map reasoning."""
    format_instruction = answer_format_instruction(
        ground_truth_type=ground_truth_type,
        template_no=template_no,
    )

    return f"""
This is a cartographic reasoning question from the MapWise dataset.

Use only the supplied choropleth map. Carefully inspect the state boundaries,
state labels, legend values, color shades, spatial regions, coastlines,
international borders, and neighbouring-state relationships.

Question:
{question}

Answer type:
{ground_truth_type}

Reason through the problem carefully and systematically. Avoid repeating earlier
observations. Use no more than 15 concise reasoning steps.

Final-answer requirements:
- {format_instruction}
- Do not include explanation after the final answer.
- End with exactly one separate line in this format:

Final answer: <answer>
""".strip()


def build_messages(
    image_path: Path,
    prompt: str,
) -> list[dict[str, Any]]:
    """Build a one-image Qwen multimodal user message."""
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]
    return [{"role": "user", "content": content}]


# ============================================================
# 5. Final-answer extraction and generation status
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


def classify_generation_status(
    raw_response: str,
    final_answer: str,
    generated_tokens: int,
    max_new_tokens: int,
) -> str:
    """Classify output completion for later error analysis."""
    if not raw_response.strip():
        return "empty"

    has_marker = bool(
        re.search(r"final\s+answer\s*:", raw_response, flags=re.IGNORECASE)
    )
    reached_limit = generated_tokens >= max_new_tokens

    if has_marker and final_answer.strip():
        return "complete"

    if reached_limit:
        return "truncated"

    if final_answer.strip():
        return "fallback_extracted"

    return "not_extractable"


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


def load_model_and_processor(model_name: str = MODEL_NAME):
    """Load Qwen3-VL-8B-Thinking and its processor."""
    print("\nLoading Qwen3-VL model...", flush=True)
    print(f"Model: {model_name}", flush=True)
    print_gpu_information()

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
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
) -> tuple[str, int]:
    """Generate one complete Qwen3-VL response and token count."""
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


def completed_qa_ids(
    predictions: list[dict[str, Any]],
) -> set[str]:
    """Return QA IDs that already have a non-empty raw response."""
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
    """Replace a record with the same qa_id or append it."""
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
    max_new_tokens: int,
    inference_seconds: float,
) -> dict[str, Any]:
    """Create one self-contained MapWise prediction record."""
    qa_id = str(sample.get("qa_id", f"mapwise_{sample_index}"))
    final_answer = extract_final_answer(raw_response)
    generation_status = classify_generation_status(
        raw_response=raw_response,
        final_answer=final_answer,
        generated_tokens=generated_tokens,
        max_new_tokens=max_new_tokens,
    )

    return {
        "qa_id": qa_id,
        "sample_index": sample_index,
        "country": str(sample.get("country", "")).strip(),
        "map_type": str(sample.get("map_type", "")).strip(),
        "map_no": str(sample.get("map_no", "")).strip(),
        "template_no": int(sample.get("template_no", -1)),
        "question": str(sample.get("question", "")).strip(),
        "ground_truth": str(sample.get("ground_truth", "")).strip(),
        "ground_truth_type": str(
            sample.get("ground_truth_type", "")
        ).strip(),
        "c_or_d": str(sample.get("c_or_d", "")).strip(),
        "relative_region": str(
            sample.get("relative_region", "")
        ).strip(),
        "source_index": sample.get("source_index"),
        "data_group_id": str(sample.get("data_group_id", "")).strip(),
        "legend_style": str(sample.get("legend_style", "")).strip(),
        "reasoning_eligible": sample.get("reasoning_eligible"),
        "split": str(sample.get("split", "")).strip(),
        "resolved_image_path": str(image_path),
        "model_name": model_name,
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
    qa_json: Path = MAPWISE_JSON,
    image_root: Path = MAPWISE_IMAGE_ROOT,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    max_new_tokens: int = MAX_NEW_TOKENS,
    start_index: int = 0,
    end_index: Optional[int] = None,
    resume: bool = True,
    overwrite: bool = False,
    save_every: int = SAVE_EVERY,
    print_every: int = PRINT_EVERY,
) -> Path:
    """Run MapWise inference and return the saved prediction JSON path."""
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

    completed = completed_qa_ids(predictions)
    selected_total = end_index - start_index

    print("=" * 80)
    print("MAPWISE QWEN3-VL INFERENCE")
    print("=" * 80)
    print(f"QA JSON:       {qa_json}")
    print(f"Image root:    {image_root}")
    print(f"Output JSON:   {output_json}")
    print(f"Dataset size:  {dataset_size}")
    print(f"Selected:      [{start_index}, {end_index}) = {selected_total}")
    print(f"Resume:        {resume}")
    print(f"Already done:  {len(completed)}")
    print(f"Max tokens:    {max_new_tokens}")

    model, processor = load_model_and_processor(model_name)

    newly_completed = 0
    skipped = 0
    failed = 0
    run_start_time = time.perf_counter()

    try:
        for absolute_index in range(start_index, end_index):
            sample = samples[absolute_index]
            qa_id = str(sample.get("qa_id", f"mapwise_{absolute_index}"))

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
                ground_truth_type = str(
                    sample["ground_truth_type"]
                ).strip()
                template_no = int(sample.get("template_no", -1))

                image_path = resolve_mapwise_image(
                    sample=sample,
                    image_root=image_root,
                )

                prompt = build_mapwise_prompt(
                    question=question,
                    ground_truth_type=ground_truth_type,
                    template_no=template_no,
                )
                messages = build_messages(
                    image_path=image_path,
                    prompt=prompt,
                )

                raw_response, generated_tokens = generate_response(
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
                    image_path=image_path,
                    raw_response=raw_response,
                    generated_tokens=generated_tokens,
                    model_name=model_name,
                    max_new_tokens=max_new_tokens,
                    inference_seconds=inference_seconds,
                )

                replace_or_append_prediction(predictions, prediction_record)
                completed.add(qa_id)
                newly_completed += 1

                if newly_completed % save_every == 0:
                    save_predictions(predictions, output_json)

                if print_every and newly_completed % print_every == 0:
                    preview = (
                        prediction_record["final_answer"]
                        .replace("\n", " ")[:120]
                    )
                    print(
                        f"[{absolute_index + 1}/{dataset_size}] "
                        f"{qa_id} | map={sample.get('map_no')} | "
                        f"type={ground_truth_type} | "
                        f"tokens={generated_tokens} | "
                        f"status={prediction_record['generation_status']} | "
                        f"time={inference_seconds:.1f}s | "
                        f"answer={preview!r}",
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
                    "country": str(sample.get("country", "")).strip(),
                    "map_type": str(sample.get("map_type", "")).strip(),
                    "map_no": str(sample.get("map_no", "")).strip(),
                    "template_no": sample.get("template_no"),
                    "question": str(sample.get("question", "")).strip(),
                    "ground_truth": str(
                        sample.get("ground_truth", "")
                    ).strip(),
                    "ground_truth_type": str(
                        sample.get("ground_truth_type", "")
                    ).strip(),
                    "c_or_d": str(sample.get("c_or_d", "")).strip(),
                    "relative_region": str(
                        sample.get("relative_region", "")
                    ).strip(),
                    "source_index": sample.get("source_index"),
                    "data_group_id": str(
                        sample.get("data_group_id", "")
                    ).strip(),
                    "legend_style": str(
                        sample.get("legend_style", "")
                    ).strip(),
                    "reasoning_eligible": sample.get("reasoning_eligible"),
                    "split": str(sample.get("split", "")).strip(),
                    "model_name": model_name,
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
                    f"[{absolute_index + 1}/{dataset_size}] "
                    f"{qa_id}: FAILED - "
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

        # Explicitly release the inference model before a later evaluation
        # process loads any optional LLM extractor.
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_json


# ============================================================
# 10. Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3-VL-8B-Thinking predictions for the "
            "MapWise India test set."
        )
    )

    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--qa-json", type=Path, default=MAPWISE_JSON)
    parser.add_argument("--image-root", type=Path, default=MAPWISE_IMAGE_ROOT)
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

    return run_mapwise_inference(
        model_name=args.model_name,
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

