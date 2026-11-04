"""Run the nine requested MapVerse questions with the existing single-image path."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parent
sys.path.insert(0, str(VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"))
import inference_mapwise_trl as inference  # noqa: E402

ROOT = VLM_DIR.parent
IMAGE_ROOT = ROOT / "MapVerse" / "data" / "imgs"
ADAPTER = VLM_DIR / "Training_outputs" / "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500" / "checkpoint-3000"
BASE_MODEL = Path(r"C:\Users\junyhuang\.cache\huggingface\hub\models--Qwen--Qwen3-VL-8B-Thinking\snapshots\92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b")
OUTPUT = VPD_DIR / "requested_mapverse_checkpoint3000_results.json"
MAX_PIXELS = 1_000_000
QUESTIONS = [
    ("859.jpg", "How many toilets are shown on this map of the hospital?", "11"),
    ("874.jpg", "Is the helipad in the east or west part of the Titus Regional Medical Center?", "East"),
    ("648.png", "How many traffic lights are on the Ribaut Road?", "2"),
    ("15.jpg", "How many bars and pubs do you see?", "2"),
    ("15.jpg", "How many restrooms can a woman use here?", "8"),
    ("15.jpg", "How many of the restrooms can men use?", "8"),
    ("764.png", "Is Mansurah on the north or south side of the river?", "South"),
    ("764.png", "Which mixed territorial controlled city is south of Tel Abyad?", "Ayn Issa"),
    ("2.webp", "How many of the cities are in Russian military control?", "2"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explicit-symbol", action="store_true", help="Rerun the 764 and 2 questions on explicit_symbol PNGs.")
    parser.add_argument("--explicit-symbol-15", action="store_true", help="Rerun the three 15.jpg questions on explicit_symbol/15.jpg.")
    parser.add_argument("--explicit-symbol-859", action="store_true", help="Rerun the 859.jpg question on explicit_symbol/859.jpg.")
    args = parser.parse_args()
    if sum((args.explicit_symbol, args.explicit_symbol_15, args.explicit_symbol_859)) > 1:
        parser.error("Choose only one explicit-symbol option.")
    image_root = ROOT / "MapVerse" / "data" / "explicit_symbol" if (args.explicit_symbol or args.explicit_symbol_15 or args.explicit_symbol_859) else IMAGE_ROOT
    if args.explicit_symbol:
        questions = [
            ("2.png" if name == "2.webp" else name, question, expected)
            for name, question, expected in QUESTIONS if name in {"764.png", "2.webp"}
        ]
        output = VPD_DIR / "requested_mapverse_explicit_symbol_checkpoint3000_results.json"
    elif args.explicit_symbol_15:
        questions = [row for row in QUESTIONS if row[0] == "15.jpg"]
        output = VPD_DIR / "requested_mapverse_explicit_symbol_15_checkpoint3000_results.json"
    elif args.explicit_symbol_859:
        questions = [row for row in QUESTIONS if row[0] == "859.jpg"]
        output = VPD_DIR / "requested_mapverse_explicit_symbol_859_checkpoint3000_results.json"
    else:
        questions = QUESTIONS
        output = OUTPUT
    for name, _, _ in questions:
        if not (image_root / name).is_file():
            raise FileNotFoundError(image_root / name)
    model, processor, loaded_adapter = inference.load_model_and_processor(
        model_name=str(BASE_MODEL), adapter_path=ADAPTER, max_pixels=MAX_PIXELS
    )
    results = []
    for index, (name, question, expected) in enumerate(questions, 1):
        started = time.perf_counter()
        with Image.open(image_root / name) as source:
            image = source.convert("RGB")
            try:
                if image.width * image.height > MAX_PIXELS:
                    scale = math.sqrt(MAX_PIXELS / (image.width * image.height))
                    image = image.resize(
                        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                inputs = inference.prepare_multimodal_inputs(
                    processor, image, inference.build_mapwise_prompt(question), thinking_mode="auto"
                )
            finally:
                image.close()
        inputs = inference.move_inputs_to_model_device(inputs, model)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=1000,
                num_beams=1, num_return_sequences=1, use_cache=True,
            )
        tokens = generated[0, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(
            tokens[None, :], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        answer = inference.extract_final_answer(raw)
        result = {
            "image_name": name,
            "image_path": str(image_root / name),
            "question": question,
            "correct_answer": expected,
            "raw_response": raw,
            "final_answer": answer,
            "generated_tokens": int(tokens.numel()),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generation_status": inference.classify_generation_status(raw, answer, int(tokens.numel()), 1000),
            "inference_seconds": round(time.perf_counter() - started, 3),
        }
        results.append(result)
        output.write_text(json.dumps({"adapter_path": str(loaded_adapter), "max_pixels": MAX_PIXELS, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index}/{len(questions)}] {name}: {answer!r} (expected {expected!r}; {result['generation_status']})", flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
