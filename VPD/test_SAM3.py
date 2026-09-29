"""Inspect SAM 3 masks for black dots inside irregular pink map regions."""

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import Sam3Model, Sam3Processor


# Change this to the map image on your machine.
IMAGE_PATH = Path(r"C:\Users\junyhuang\Thesis\MapVerse\data\imgs\2.webp")
MODEL_ID = "facebook/sam3"
DOT_PROMPT = "black dot"
REGION_PROMPT = "pink region"
DETECTION_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
MIN_DOT_OVERLAP = 0.30

# First run with None. Inspect pink_regions.png; if only mask 1 is the desired
# map region, set PINK_KEEP_INDICES = [1] and run again. Indices start at 0.
PINK_KEEP_INDICES = None


def segment(model, processor, image, prompt, device):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=DETECTION_THRESHOLD,
        mask_threshold=MASK_THRESHOLD,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]
    masks = result["masks"].detach().cpu().bool()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or tuple(masks.shape[1:]) != (image.height, image.width):
        raise ValueError(f"Unexpected mask shape {tuple(masks.shape)} for image {image.size}")
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    return masks, boxes, scores


def show_instances(image, masks, boxes, scores, color):
    canvas = image.convert("RGBA")
    for index, mask in enumerate(masks):
        alpha = Image.fromarray(mask.numpy().astype("uint8") * 110, mode="L")
        layer = Image.new("RGBA", image.size, color + (0,))
        layer.putalpha(alpha)
        canvas = Image.alpha_composite(canvas, layer)
    canvas = canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for index, (box, score) in enumerate(zip(boxes, scores)):
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1] - 13)), f"{index}: {score:.2f}",
                  fill="white", stroke_width=2, stroke_fill="black")
    return canvas


def main():
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    output_dir = IMAGE_PATH.parent / (IMAGE_PATH.stem + "_sam3_analysis")
    output_dir.mkdir(exist_ok=True)
    image = Image.open(IMAGE_PATH).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Image: {IMAGE_PATH}\nDevice: {device}")

    model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    processor = Sam3Processor.from_pretrained(MODEL_ID)

    dot_masks, dot_boxes, dot_scores = segment(model, processor, image, DOT_PROMPT, device)
    pink_masks, pink_boxes, pink_scores = segment(model, processor, image, REGION_PROMPT, device)

    show_instances(image, dot_masks, dot_boxes, dot_scores, (0, 190, 255)).save(
        output_dir / "black_dots.png"
    )
    show_instances(image, pink_masks, pink_boxes, pink_scores, (255, 0, 180)).save(
        output_dir / "pink_regions.png"
    )

    selected_pink = (list(range(len(pink_masks))) if PINK_KEEP_INDICES is None
                     else list(PINK_KEEP_INDICES))
    if any(i < 0 or i >= len(pink_masks) for i in selected_pink):
        raise ValueError(f"PINK_KEEP_INDICES must be within 0..{len(pink_masks) - 1}")

    # Union of the selected irregular masks; never use their bounding boxes.
    region_mask = torch.zeros((image.height, image.width), dtype=torch.bool)
    for i in selected_pink:
        region_mask |= pink_masks[i]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    dots = []
    for i, (mask, box, score) in enumerate(zip(dot_masks, dot_boxes, dot_scores)):
        pixels = int(mask.sum().item())
        overlap = int((mask & region_mask).sum().item())
        fraction = overlap / pixels if pixels else 0.0
        matched = fraction >= MIN_DOT_OVERLAP
        best_region = None
        if selected_pink and pixels:
            best_region = max(selected_pink, key=lambda j: int((mask & pink_masks[j]).sum()))
        draw.rectangle(box, outline="red" if matched else "cyan", width=3)
        draw.text((box[0], max(0, box[1] - 13)), f"{i}: {fraction:.2f}",
                  fill="white", stroke_width=2, stroke_fill="black")
        dots.append({"index": i, "box_xyxy": box, "score": score,
                     "dot_pixels": pixels, "overlap_pixels": overlap,
                     "overlap_fraction": fraction, "best_pink_mask_index": best_region,
                     "on_pink_region": matched})
        print(f"dot {i}: score={score:.3f}, overlap={fraction:.3f}, "
              f"on_pink_region={matched}, box={box}")

    annotated.save(output_dir / "selected_black_dots.png")
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump({"image": str(IMAGE_PATH), "dot_prompt": DOT_PROMPT,
                   "region_prompt": REGION_PROMPT,
                   "pink_region_count": len(pink_masks),
                   "pink_keep_indices": selected_pink,
                   "min_dot_overlap": MIN_DOT_OVERLAP, "dots": dots},
                  file, ensure_ascii=False, indent=2)

    print(f"Detected {len(dot_masks)} black dots and {len(pink_masks)} pink masks.")
    print(f"Saved diagnostics and results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
