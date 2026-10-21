"""Test local Qwen3-VL Thinking tool calling with the official zoom protocol.

This is a minimal local equivalent of Qwen-Agent's image_zoom_in_tool loop:

    user image + question -> model -> <tool_call> -> crop original image
    -> <tool_response> with a new image -> model continuation -> final answer

It supports both a base model and a PEFT/LoRA adapter, with identical prompts
and generation settings.  The default question is the MapVerse question about
Russian military control in image 2.webp.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parents[1]
INFERENCE_DIR = VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"
sys.path.insert(0, str(INFERENCE_DIR))

import torch  # noqa: E402
import inference_mapwise_trl as inference  # noqa: E402
from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
DEFAULT_ADAPTER = (
    VLM_DIR / "Training_outputs" /
    "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500" /
    "checkpoint-2500"
)
DEFAULT_IMAGE = VLM_DIR.parent / "MapVerse" / "data" / "imgs" / "2.webp"
DEFAULT_QUESTION = "how many of the cities are in Russian military control?"

def make_zoom_tool(tool_name: str = "request_local_region") -> dict[str, Any]:
    """Build the paper-compatible absolute-pixel crop tool schema."""
    return {
    "type": "function",
    "function": {
        "name": tool_name,
        "description": (
            "Zoom in on a specific region of an image by cropping it based on "
            "a bounding box and an optional object label."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Absolute pixel bounding box in the current original "
                        "image as [x1, y1, x2, y2]."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Name or label of the object in the box.",
                },
                "img_idx": {
                    "type": "number",
                    "description": "Image index, starting from 0.",
                },
            },
            "required": ["bbox_2d"],
        },
    },
}


def smart_resize(height: int, width: int, factor: int = 32,
                 min_pixels: int = 256 * 32 * 32,
                 max_pixels: int = 12845056) -> tuple[int, int]:
    """Match Qwen-Agent's crop resize policy."""
    def round_factor(value: float) -> int:
        return round(value / factor) * factor

    def ceil_factor(value: float) -> int:
        return math.ceil(value / factor) * factor

    def floor_factor(value: float) -> int:
        return math.floor(value / factor) * factor

    h_bar = max(factor, round_factor(height))
    w_bar = max(factor, round_factor(width))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_factor(height / beta)
        w_bar = floor_factor(width / beta)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_factor(height * beta)
        w_bar = ceil_factor(width * beta)
    return h_bar, w_bar


def validate_bbox(arguments: dict[str, Any], image_sizes: list[tuple[int, int]]) -> dict[str, Any]:
    """Validate AdaptVision-style absolute pixel coordinates."""
    bbox = arguments.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("bbox_2d must be a list of four numbers")
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError) as error:
        raise ValueError("bbox_2d contains a non-numeric value") from error
    x1, y1, x2, y2 = values
    img_idx = int(arguments.get("img_idx", 0))
    if img_idx < 0 or img_idx >= len(image_sizes):
        raise ValueError(f"img_idx={img_idx} is outside [0, {len(image_sizes) - 1}]")
    image_width, image_height = image_sizes[img_idx]
    if not (0 <= x1 < image_width and 0 <= y1 < image_height and
            0 < x2 <= image_width and 0 < y2 <= image_height):
        raise ValueError(
            f"bbox_2d must be absolute pixels inside image {image_width}x{image_height}"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_2d must satisfy x2>x1 and y2>y1")
    label = arguments.get("label", "")
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    return {"bbox_2d": values, "img_idx": img_idx, "label": label,
            "image_size": [image_width, image_height]}


def crop_original_image(
    image_paths: list[Path], arguments: dict[str, Any], crop_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Crop the original full-resolution image using Qwen-Agent coordinates."""
    image_sizes = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_sizes.append(image.size)
    checked = validate_bbox(arguments, image_sizes)
    source_path = image_paths[checked["img_idx"]]
    with Image.open(source_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = checked["bbox_2d"]
        absolute = [x1, y1, x2, y2]
        left, top, right, bottom = absolute
        left, top = max(0, left), max(0, top)
        right, bottom = min(width, right), min(height, bottom)
        crop_width, crop_height = right - left, bottom - top
        if crop_width < 1 or crop_height < 1:
            raise ValueError("bbox becomes empty after conversion to pixels")

        # Preserve the original crop pixels.  The PNG is lossless and no
        # artificial upscale/downscale is performed here.  The model's image
        # processor may still apply its own model-required preprocessing later.
        cropped = image.crop((left, top, right, bottom))
        raw_crop_size = list(cropped.size)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"crop_{uuid.uuid4().hex}.png"
        cropped.save(crop_path)

    metadata = {
        "source_image": str(source_path),
        "source_size": [width, height],
        "bbox_2d_absolute_pixels": checked["bbox_2d"],
        "bbox_pixels_before_resize": [left, top, right, bottom],
        "crop_size_lossless": raw_crop_size,
        "img_idx": checked["img_idx"],
        "label": checked["label"],
        "crop_path": str(crop_path),
    }
    return crop_path, metadata


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse the official XML tool-call format plus common local variants."""
    calls: list[dict[str, Any]] = []
    for match in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S | re.I):
        raw = match.group(1).strip()
        try:
            value = json.loads(raw)
            if isinstance(value, dict):
                calls.append({"raw": raw, **value})
        except json.JSONDecodeError as error:
            calls.append({"raw": raw, "parse_error": str(error)})
    return calls


def tool_call_order(text: str) -> dict[str, Any]:
    """Diagnose whether a tool call happened before a candidate answer."""
    lower = text.lower()
    tool_position = lower.find("<tool_call>")
    markers = ("final answer:", "the answer should be", "so the answer is")
    answer_positions = [lower.find(marker) for marker in markers if lower.find(marker) >= 0]
    first_answer = min(answer_positions) if answer_positions else None
    return {
        "tool_call_position": tool_position if tool_position >= 0 else None,
        "candidate_answer_position": first_answer,
        "tool_call_before_candidate_answer": (
            tool_position >= 0 and (first_answer is None or tool_position < first_answer)
        ),
        "late_tool_call": (
            tool_position >= 0 and first_answer is not None and first_answer < tool_position
        ),
    }


class ToolCallEndCriteria(StoppingCriteria):
    """Stop generation immediately after the XML tool-call closing tag."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs) -> torch.Tensor:
        generated_text = self.tokenizer.decode(
            input_ids[0, self.prompt_length:].tolist(),
            skip_special_tokens=False,
        )
        return torch.tensor(
            ["</tool_call>" in generated_text.lower()],
            dtype=torch.bool,
            device=input_ids.device,
        )


def build_prompt(question: str, mode: str, image_size: tuple[int, int], tool_name: str) -> str:
    if mode == "sanity":
        instruction = (
            f"For this tool-calling syntax sanity test, you MUST call {tool_name} "
            "exactly once before answering. Do not state, count, or commit to any "
            "candidate answer before the tool call. You may briefly explain what "
            "you need to inspect, then end this reasoning step with the tool call. "
            "Choose a useful region yourself. Do not invent a crop; use the "
            "official tool format."
        )
    else:
        instruction = (
            f"You have access to {tool_name}. Decide autonomously whether "
            "zooming into the original map would improve your answer. If you use "
            "the tool, call it before giving any candidate or final answer. Never "
            "append a tool call after already answering. Do not call the tool "
            "unless you genuinely need a crop."
        )
    return (
        "You are solving a cartographic reasoning question. Use only the supplied "
        "map image. Think carefully about the legend, city markers, colors, and "
        "territorial regions.\n\n"
        + instruction + "\n"
        f"The original image size is {image_size[0]} by {image_size[1]} pixels. "
        "If you call the tool, use absolute pixel bbox_2d coordinates "
        "[x1, y1, x2, y2] from top-left to bottom-right, and img_idx=0.\n\n"
        f"Question: {question}\n\n"
        "After any tool response, continue your reasoning and finish with exactly "
        "one line: Final answer: <answer>"
    )


def make_messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]


def prepare_with_tools(processor, messages: list[dict[str, Any]], thinking: str,
                       tool_name: str):
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "tools": [make_zoom_tool(tool_name)],
    }
    if thinking != "auto":
        kwargs["enable_thinking"] = thinking == "on"
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except Exception as primary:
        rendered_kwargs = dict(kwargs)
        rendered_kwargs.pop("tokenize")
        rendered_kwargs.pop("return_dict")
        rendered_kwargs.pop("return_tensors")
        rendered = processor.apply_chat_template(messages, tokenize=False, **rendered_kwargs)
        # Fallback uses the actual PIL/path image content through processor.
        images = []
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and "image" in item:
                    images.append(item["image"])
        if not images:
            raise RuntimeError("No image content found for tool-call fallback") from primary
        return processor(images=images, text=rendered, return_tensors="pt")


def generate_turn(model, processor, messages, thinking: str, max_new_tokens: int,
                  temperature: float, top_p: float, seed: int,
                  tool_name: str) -> tuple[str, int]:
    inputs = prepare_with_tools(processor, messages, thinking, tool_name)
    inputs = inference.move_inputs_to_model_device(inputs, model)
    input_length = int(inputs["input_ids"].shape[1])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, use_cache=True,
            stopping_criteria=StoppingCriteriaList([
                ToolCallEndCriteria(processor.tokenizer, input_length),
            ]),
        )
    tokens = generated[:, input_length:]
    text = processor.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    del inputs, generated
    return text, int(tokens.shape[1])


def run_one(model, processor, image_path: Path, question: str, mode: str,
            crop_dir: Path, max_tool_calls: int, max_new_tokens: int,
            temperature: float, top_p: float, seed: int, thinking: str,
            tool_name: str) -> dict[str, Any]:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    image_size = original.size
    messages = make_messages(
        original, build_prompt(question, mode, image_size, tool_name),
    )
    started = time.perf_counter()
    turns = []
    crops = []
    tool_calls = 0
    status = "no_tool_call"
    final_text = ""
    try:
        for turn_index in range(max_tool_calls + 1):
            raw, token_count = generate_turn(
                model, processor, messages, thinking, max_new_tokens,
                temperature, top_p, seed + turn_index, tool_name,
            )
            parsed = extract_tool_calls(raw)
            ordering = tool_call_order(raw)
            turn_record = {
                "turn_index": turn_index,
                "raw_response": raw,
                "generated_tokens": token_count,
                "tool_calls": parsed,
                "tool_call_order": ordering,
            }
            turns.append(turn_record)
            if not parsed or tool_calls >= max_tool_calls:
                final_text = raw
                if parsed and tool_calls >= max_tool_calls:
                    status = "tool_call_limit_reached"
                break

            call = parsed[0]
            if call.get("name") != tool_name:
                status = "unsupported_tool_call"
                final_text = raw
                break
            if "arguments" not in call or not isinstance(call["arguments"], dict):
                status = "invalid_tool_arguments"
                final_text = raw
                break
            try:
                crop_path, crop_meta = crop_original_image(
                    [image_path], call["arguments"], crop_dir,
                )
            except Exception as error:
                status = "invalid_or_failed_crop"
                turn_record["tool_error"] = str(error)
                final_text = raw
                break

            tool_calls += 1
            crops.append(crop_meta)
            status = "late_tool_call" if ordering["late_tool_call"] else "tool_executed"
            # Preserve the assistant's reasoning/tool call, then return an
            # actual image content item through the local Qwen tool message.
            before_tool = re.split(r"<tool_call>", raw, maxsplit=1, flags=re.I)[0].strip()
            messages.append({
                "role": "assistant",
                "content": before_tool,
                "tool_calls": [{"name": tool_name, "arguments": call["arguments"]}],
            })
            with Image.open(crop_path) as crop_image:
                crop_for_message = crop_image.convert("RGB")
            messages.append({
                "role": "tool",
                "content": [
                    {"type": "image", "image": crop_for_message},
                    {"type": "text", "text": json.dumps(crop_meta, ensure_ascii=False)},
                ],
            })
        answer = inference.extract_final_answer(final_text)
        return {
            "mode": mode,
            "image_path": str(image_path),
            "question": question,
            "prompt": build_prompt(question, mode, image_size, tool_name),
            "tool_name": tool_name,
            "status": status,
            "tool_call_attempted": any(t["tool_calls"] for t in turns),
            "tool_call_count": tool_calls,
            "turns": turns,
            "crops": crops,
            "final_raw_response": final_text,
            "final_answer": answer,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        original.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run the base model without loading a PEFT adapter.")
    parser.add_argument("--mode", choices=("sanity", "autonomous", "both"), default="both")
    parser.add_argument("--tool-name", choices=("request_local_region", "image_zoom_in_tool"),
                        default="request_local_region",
                        help="Tool protocol name; request_local_region matches AdaptVision.")
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=VPD_DIR / "outputs")
    parser.add_argument("--no-quantization", action="store_true")
    args = parser.parse_args()
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if args.max_tool_calls < 1:
        raise ValueError("--max-tool-calls must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor, loaded_adapter = inference.load_model_and_processor(
        model_name=args.model_name,
        adapter_path=(None if args.no_adapter else args.adapter_path.expanduser().resolve()),
        load_in_4bit=not args.no_quantization,
    )
    modes = ("sanity", "autonomous") if args.mode == "both" else (args.mode,)
    results = []
    for index, mode in enumerate(modes):
        result = run_one(
            model, processor, image_path, args.question, mode,
            args.output_dir / mode / "crops", args.max_tool_calls,
            args.max_new_tokens, args.temperature, args.top_p,
            args.seed + index * 10000, args.thinking, args.tool_name,
        )
        result["model_name"] = args.model_name
        result["adapter_path"] = str(loaded_adapter) if loaded_adapter else None
        result["quantization"] = "4-bit NF4, BF16 compute" if not args.no_quantization else "BF16"
        results.append(result)
        print(json.dumps({
            "mode": mode, "status": result["status"],
            "tool_call_count": result["tool_call_count"],
            "final_answer": result["final_answer"],
        }, ensure_ascii=False), flush=True)

    payload = {
        "experiment": "local_qwen3vl_image_zoom_agent",
        "model_name": args.model_name,
        "adapter_path": str(loaded_adapter) if loaded_adapter else None,
        "quantization": "4-bit NF4, BF16 compute" if not args.no_quantization else "BF16",
        "generation": {
            "temperature": args.temperature, "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "max_tool_calls": args.max_tool_calls, "thinking": args.thinking,
        },
        "results": results,
    }
    output_json = args.output_dir / "agent_crop_results.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
