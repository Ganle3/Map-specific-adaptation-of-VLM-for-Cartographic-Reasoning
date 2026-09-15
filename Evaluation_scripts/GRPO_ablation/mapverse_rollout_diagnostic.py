"""
mapverse_rollout_diagnostic.py

Run N independent generations per MapVerse question and score each generation
with mapverse_evaluation_exact.py.

Default use case:
  - 50-question diagnostic CSV
  - 4 generations per question
  - Qwen3-VL Thinking model
  - binary correctness reward only

Outputs:
  1) detailed JSON with every generation
  2) flat CSV with one row per generation
  3) summary CSV with one row per question and reward pattern such as:
       0,0,0,0  -> all_wrong
       0,1,0,1  -> mixed
       1,1,1,1  -> all_correct

The script deliberately keeps generation sequential for reliability across
Qwen/Transformers versions and heterogeneous image sizes. It checkpoints after
every question, so an interrupted run can be resumed with --resume.
"""

from __future__ import annotations

import argparse
import gc
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


from mapverse_evaluation_exact import evaluate_exact


def build_mapverse_prompt(question: str) -> str:
    """Match the MapWise formal GRPO prompt structure as closely as possible."""
    return f"""
This is a cartographic reasoning question from the MapVerse dataset.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>
""".strip()



def load_image_safely(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def build_messages(image: Image.Image, question: str):
    prompt = build_mapverse_prompt(question)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def move_inputs_to_model_device(inputs, model):
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


def prepare_multimodal_inputs(processor, image: Image.Image, question: str):
    """
    Prepare Qwen3-VL multimodal inputs using the same processor-native path
    as inference_mapwise_trl.py. No qwen_vl_utils dependency is required.
    """
    messages = build_messages(image=image, question=question)

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as primary_error:
        # Fallback path, matching the MapWise inference script.
        prompt = build_mapverse_prompt(question)
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
                f"Primary error: {type(primary_error).__name__}: {primary_error}\n"
                f"Fallback error: {type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error


@torch.inference_mode()
def generate_once(
    model,
    processor,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = prepare_multimodal_inputs(
        processor=processor,
        image=image,
        question=question,
    )
    inputs = move_inputs_to_model_device(inputs, model)

    if "input_ids" not in inputs:
        raise KeyError(
            "Prepared model inputs do not contain input_ids. "
            f"Available keys: {list(inputs.keys())}"
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )

    generated_token_ids = generated_ids[:, input_length:]

    decoded = processor.batch_decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = decoded[0].strip()

    del inputs, generated_ids, generated_token_ids, decoded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_text


def pattern_label(rewards: List[int]) -> str:
    total = sum(rewards)
    if total == 0:
        return "all_wrong"
    if total == len(rewards):
        return "all_correct"
    return "mixed"


def save_outputs(results: List[Dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mapverse_rollout_diagnostic.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    flat_rows = []
    summary_rows = []

    for item in results:
        rewards = [int(g["evaluation"]["reward"]) for g in item["generations"]]
        reward_pattern = ",".join(str(x) for x in rewards)

        summary_rows.append({
            "selection_id": item["selection_id"],
            "source_row": item["source_row"],
            "image_name": item["image_name"],
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "answer_type": item["answer_type"],
            "question_type": item["question_type"],
            "map_type": item["map_type"],
            "geographic_level": item["geographic_level"],
            "reward_pattern": reward_pattern,
            "correct_generations": sum(rewards),
            "num_generations": len(rewards),
            "mean_reward": sum(rewards) / len(rewards),
            "diagnostic_class": pattern_label(rewards),
        })

        for g in item["generations"]:
            ev = g["evaluation"]
            flat_rows.append({
                "selection_id": item["selection_id"],
                "source_row": item["source_row"],
                "image_name": item["image_name"],
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "answer_type": item["answer_type"],
                "question_type": item["question_type"],
                "generation_id": g["generation_id"],
                "completion": g["completion"],
                "prediction_canonical": ev["prediction_canonical"],
                "ground_truth_canonical": ev["ground_truth_canonical"],
                "reward": int(ev["reward"]),
            })

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "mapverse_rollout_generations.csv", flat_rows)
    write_csv(output_dir / "mapverse_rollout_summary.csv", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="mapverse_diagnose_50.csv",
        help="Selected diagnostic CSV.",
    )
    parser.add_argument(
        "--image-base",
        default=r"C:\Users\junyhuang\Thesis\MapVerse\data\imgs",
        help="Directory containing MapVerse images.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Thinking",
        help="HF model name or local model path.",
    )
    parser.add_argument("--output-dir", default="mapverse_diagnostic_results")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSON in output-dir.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit, e.g. --limit 2.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    image_base = Path(args.image_base)
    output_dir = Path(args.output_dir)
    output_json = output_dir / "mapverse_rollout_diagnostic.json"

    rows = []
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        rows = rows[: args.limit]

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this diagnostic. "
            f"torch={torch.__version__}, cuda_build={torch.version.cuda}"
        )

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU VRAM: {props.total_memory / 1024**3:.2f} GB")

    print("Loading model in 4-bit NF4...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    results: List[Dict[str, Any]] = []
    completed_ids = set()

    if args.resume and output_json.exists():
        with output_json.open("r", encoding="utf-8") as f:
            results = json.load(f)
        completed_ids = {str(x["selection_id"]) for x in results}
        print(f"Resuming: {len(completed_ids)} questions already completed.")

    for pos, row in enumerate(rows, start=1):
        sid = str(row["selection_id"])
        if sid in completed_ids:
            continue

        image_path = image_base / row["image_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = load_image_safely(image_path)
        question = row["question"]
        ground_truth = row["correct_answer"]
        answer_type = row["answer_type"]

        print("\n" + "=" * 100)
        print(
            f"[{pos}/{len(rows)}] ID={sid} | {answer_type} | "
            f"{row['question_type']} | image={row['image_name']}"
        )
        print(f"Q:  {question}")
        print(f"GT: {ground_truth}")

        generations = []
        reward_vector = []

        for g in range(1, args.num_generations + 1):
            try:
                completion = generate_once(
                    model=model,
                    processor=processor,
                    image=image,
                    question=question,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                ev = evaluate_exact(
                    prediction=completion,
                    ground_truth=ground_truth,
                    answer_type=answer_type,
                )

                reward = int(ev["reward"])
                reward_vector.append(reward)

                generations.append({
                    "generation_id": g,
                    "completion": completion,
                    "evaluation": ev,
                })

                print(
                    f"  GEN {g:02d} | reward={reward} | "
                    f"pred={ev['prediction_canonical']!r}"
                )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        label = pattern_label(reward_vector)
        pattern = ",".join(str(x) for x in reward_vector)

        print(
            f"  => rewards=[{pattern}] | class={label} | "
            f"mean={sum(reward_vector)/len(reward_vector):.2f}"
        )

        results.append({
            "selection_id": row["selection_id"],
            "source_row": row["source_row"],
            "image_name": row["image_name"],
            "question": question,
            "correct_answer": ground_truth,
            "answer_type": answer_type,
            "question_type": row["question_type"],
            "map_type": row["map_type"],
            "geographic_level": row["geographic_level"],
            "generations": generations,
        })

        # Save after every QA so long runs are interruption-safe.
        save_outputs(results, output_dir)

        del image
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nFinished.")
    save_outputs(results, output_dir)

    all_rewards = [
        int(g["evaluation"]["reward"])
        for item in results
        for g in item["generations"]
    ]
    if all_rewards:
        print(
            f"Overall generation-level accuracy: "
            f"{sum(all_rewards)}/{len(all_rewards)} = "
            f"{sum(all_rewards)/len(all_rewards):.4f}"
        )

    classes = {"all_wrong": 0, "mixed": 0, "all_correct": 0}
    for item in results:
        rs = [int(g["evaluation"]["reward"]) for g in item["generations"]]
        classes[pattern_label(rs)] += 1

    print(f"Question-level classes: {classes}")
    print(
        "For the next 20-QA GRPO test, prioritize mixed questions, then use a "
        "small number of all_wrong/all_correct controls only if scientifically useful."
    )


if __name__ == "__main__":
    main()
