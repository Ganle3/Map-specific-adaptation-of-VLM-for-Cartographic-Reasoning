"""Pilot inference with a full map plus visual detail crop(s).

Example:
  python pilot_multimage_inference.py --images "...\\2.webp" "...\\bakhmut_crop.png"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

VPD_DIR = Path(__file__).resolve().parent
VLM_DIR = VPD_DIR.parent
sys.path.insert(0, str(VLM_DIR / "Evaluation_scripts" / "GRPO_ablation"))
import inference_mapwise_trl as inference  # noqa: E402

DEFAULT_ADAPTER = VLM_DIR / "Training_outputs" / (
    "MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12"
) / "checkpoints" / "checkpoint-3000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True,
                        help="One full map, optionally followed by detail crop(s).")
    parser.add_argument("--question", required=True,
                        help="Question to send together with the images.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--thinking-mode", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    paths = [p.expanduser().resolve() for p in args.images]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    model, processor, loaded_adapter = inference.load_model_and_processor(
        adapter_path=args.adapter_path.expanduser().resolve()
    )
    images = []
    try:
        images = [Image.open(path).convert("RGB") for path in paths]
        prompt = (
            "You are given multiple views of the same map. Image 1 is the full map; "
            "the later image(s) are detail crops for visual verification. Use and only according to all "
            "images, especially the detail crop, before answering.\n\n"
            + inference.build_mapwise_prompt(args.question)
        )
        inputs = inference.prepare_multimodal_inputs_multi(
            processor, images, prompt, thinking_mode=args.thinking_mode
        )
    finally:
        for image in images:
            image.close()

    inputs = inference.move_inputs_to_model_device(inputs, model)
    import torch
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
            num_beams=1, num_return_sequences=1, use_cache=True,
        )
    tokens = generated[0, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(tokens[None, :], skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0].strip()
    result = {
        "images": [str(path) for path in paths],
        "question": args.question,
        "adapter_path": str(loaded_adapter) if loaded_adapter else None,
        "raw_response": raw,
        "final_answer": inference.extract_final_answer(raw),
        "generated_tokens": int(tokens.numel()),
        "inference_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.expanduser().resolve().write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
