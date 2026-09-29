"""Probe Florence-2 phrase grounding on one map image.

Example:
    python florence2_ground_map.py --image "C:\\Users\\junyhuang\\Thesis\\MapVerse\\data\\imgs\\2.webp"
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor,  Florence2ForConditionalGeneration

MODEL_ID = "florence-community/Florence-2-large-ft"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
DEFAULT_PROMPTS = [
    "black dots",
    "pink regions",
    "the black dot on the pink region",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("florence2_outputs"))
    parser.add_argument("--prompt", action="append", help="Repeat to try multiple phrases")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    image = Image.open(args.image).convert("RGB")

    for index, phrase in enumerate(args.prompt or DEFAULT_PROMPTS, start=1):
        inputs = processor(text=TASK + phrase, images=image, return_tensors="pt")
        inputs = inputs.to(device, dtype)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(decoded, task=TASK, image_size=image.size)
        result = parsed[TASK]

        prefix = args.output_dir / f"{args.image.stem}_prompt{index}"
        prefix.with_suffix(".json").write_text(
            json.dumps({"image": str(args.image), "prompt": phrase, "result": result},
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for box in result.get("bboxes", []):
            draw.rectangle(box, outline="red", width=3)
        annotated.save(prefix.with_suffix(".png"))
        print(f"[{index}] {phrase!r}: {len(result.get('bboxes', []))} box(es)")
        for label, box in zip(result.get("labels", []), result.get("bboxes", [])):
            print(f"    {label}: {box}")
        print(f"    Saved: {prefix.with_suffix('.png')} and {prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()