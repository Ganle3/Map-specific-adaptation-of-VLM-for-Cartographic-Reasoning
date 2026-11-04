"""Run requested MapVerse count questions with unadapted Qwen3-VL-8B-Instruct."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parent
sys.path.insert(0, str(VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"))
import inference_mapwise_trl as inference  # noqa: E402

MODEL_REPO = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3-VL-8B-Instruct-unsloth-bnb-4bit" / "snapshots"
IMAGE_ROOT = VLM_DIR.parent / "MapVerse" / "data" / "imgs"
OUTPUT = VPD_DIR / "mapverse_instruct_8b_count_results.json"
MAX_PIXELS = 1_000_000
QUESTIONS = [
    ("859.jpg", "How many toilets are shown on this map of the hospital?", "11"),
    ("15.jpg", "How many bars and pubs do you see?", "2"),
    ("15.jpg", "How many restrooms can a woman use here?", "8"),
    ("15.jpg", "How many of the restrooms can men use?", "8"),
    ("2.webp", "How many of the cities are in Russian military control?", "2"),
]


def restore_visual_mergers(model, snapshot: Path) -> list[str]:
    """Restore BF16 merger weights misclassified as uninitialized Linear4bit."""
    import bitsandbytes as bnb

    index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    repaired = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, bnb.nn.Linear4bit) or getattr(module.weight, "quant_state", None) is not None:
            continue
        if not (name.startswith("model.visual.merger.") or name.startswith("model.visual.deepstack_merger_list.")):
            raise RuntimeError(f"Unexpected uninitialized quantized layer: {name}")
        weight_key, bias_key = name + ".weight", name + ".bias"
        with safe_open(str(snapshot / index[weight_key]), framework="pt", device="cpu") as archive:
            weight = archive.get_tensor(weight_key)
        with safe_open(str(snapshot / index[bias_key]), framework="pt", device="cpu") as archive:
            bias = archive.get_tensor(bias_key)
        if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
            raise RuntimeError(f"Expected BF16 visual merger tensors: {name}")
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            linear.weight.copy_(weight.to("cuda"))
            linear.bias.copy_(bias.to("cuda"))
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, linear)
        repaired.append(name)
    if len(repaired) != 8:
        raise RuntimeError(f"Expected 8 visual merger layers to restore, found {len(repaired)}")
    return repaired


def main() -> None:
    snapshots = sorted(path for path in CACHE.iterdir() if (path / "config.json").is_file())
    if not snapshots:
        raise FileNotFoundError(f"No cached snapshot for {MODEL_REPO}: {CACHE}")
    snapshot = snapshots[-1]
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(snapshot), device_map={"": 0}, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    repaired_layers = restore_visual_mergers(model, snapshot)
    print(f"Restored {len(repaired_layers)} BF16 visual merger layers from checkpoint.", flush=True)
    results = []
    for index, (name, question, expected) in enumerate(QUESTIONS, 1):
        started = time.perf_counter()
        image_path = IMAGE_ROOT / name
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_size = image.size
        if image.width * image.height > MAX_PIXELS:
            scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
            smaller = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            image.close()
            image = smaller
        submitted_size = image.size
        try:
            inputs = inference.prepare_multimodal_inputs(
                processor, image, inference.build_mapwise_prompt(question), thinking_mode="auto"
            )
        finally:
            image.close()
        inputs = inference.move_inputs_to_model_device(inputs, model)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=500,
                num_beams=1, num_return_sequences=1, use_cache=True,
            )
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(
            tokens[None, :], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        answer = inference.extract_final_answer(raw)
        result = {
            "image_name": name, "image_path": str(image_path),
            "question": question, "correct_answer": expected,
            "raw_response": raw, "final_answer": answer,
            "original_size_wh": list(original_size), "submitted_size_wh": list(submitted_size),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": int(tokens.numel()),
            "generation_status": inference.classify_generation_status(raw, answer, int(tokens.numel()), 500),
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
        results.append(result)
        OUTPUT.write_text(json.dumps({"model_repo": MODEL_REPO, "snapshot": str(snapshot),
                                      "restored_bf16_visual_merger_layers": repaired_layers,
                                      "adapter": None, "max_pixels": MAX_PIXELS,
                                      "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(QUESTIONS)}] {name}: {answer!r} (expected {expected!r}; {result['generation_status']})", flush=True)
    print(f"Saved: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
