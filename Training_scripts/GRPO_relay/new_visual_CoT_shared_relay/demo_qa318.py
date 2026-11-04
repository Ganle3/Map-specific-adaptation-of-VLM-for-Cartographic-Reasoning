#!/usr/bin/env python3
"""Run the Relay-4 lazy-HR demo on MapVerse question 318 / image 764.png."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
RELAY_ROOT = THIS_DIR.parent
VLM_ROOT = RELAY_ROOT.parents[1]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (RELAY_ROOT, SPARSE_ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import MapQuestion, build_cartographic_prompt, evaluate_answer
from rolling_sparse_qwen3vl import load_model
from utils import decoder_hidden_size, encode_global_image, qwen_lm

from dynamic_decode import generate_with_new_visual_relay
from policy import NewVisualCoTRelay
from tile_cache import CoarseImageGrid, OnDemandTileFeatureCache


QUESTION = "Which HTS controlled city is closest to Homs?"
GROUND_TRUTH = "Rastan"


def parse_args():
    default_adapter = (
        VLM_ROOT / "Training_outputs" /
        "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500" /
        "checkpoint-2500"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs/764.png")
    parser.add_argument("--adapter", type=Path, default=default_adapter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=(
        VLM_ROOT / "Training_outputs/GRPO_relay/relay4_demo_qa318"
    ))
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--local-max-pixels", type=int, default=500_000)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--lookup-start-token", type=int, default=10)
    parser.add_argument("--lookup-every", type=int, default=10)
    parser.add_argument("--relay-layer", type=int, default=16,
                        help="Zero-based decoder layer after which HR relay fusion occurs.")
    parser.add_argument("--num-relays", type=int, default=2)
    parser.add_argument("--window-side", type=int, default=5)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    parser.add_argument("--injection-scale", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=318)
    parser.add_argument("--sample-tiles", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.adapter.is_dir():
        parser.error(f"Adapter directory does not exist: {args.adapter}")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(
        args.model, args.adapter, load_in_4bit=args.load_in_4bit
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = qwen_lm(model).get_input_embeddings().weight.device

    with Image.open(args.image) as image_file:
        original = image_file.convert("RGB")
    global_submitted = _resize_to_pixel_cap(original, args.max_pixels)
    source = _content_inputs(
        processor, [global_submitted], build_cartographic_prompt(QUESTION),
        device, args.max_pixels,
    )
    global_features = encode_global_image(model, source)
    global_memory = global_features.pooler_output
    global_memory = (global_memory if isinstance(global_memory, torch.Tensor)
                     else torch.cat(tuple(global_memory), dim=0))
    spatial_merge_size = int(qwen_lm(model).config.vision_config.spatial_merge_size)
    coarse_grid = CoarseImageGrid(
        original_size=original.size,
        image_grid_thw=source["image_grid_thw"],
        spatial_merge_size=spatial_merge_size,
        expected_tokens=global_memory.shape[0],
        window_side=args.window_side,
        margin_fraction=args.margin_fraction,
    )
    policy = NewVisualCoTRelay(
        decoder_hidden_size(model), num_relays=args.num_relays,
        injection_scale=args.injection_scale,
    ).to(device)
    torch.save({
        "policy_kind": "new_visual_cot_shared_relay_demo",
        "seed": args.seed,
        "state_dict": policy.state_dict(),
    }, args.output_dir / "random_relay_initialization.pt")
    tile_cache = OnDemandTileFeatureCache(
        model=model, processor=processor, original_image=original,
        device=device, max_pixels=args.local_max_pixels,
        output_dir=args.output_dir / "selected_tiles",
    )

    completion, response, state = generate_with_new_visual_relay(
        model,
        source_inputs=source,
        global_features=global_features,
        policy=policy,
        coarse_grid=coarse_grid,
        tile_cache=tile_cache,
        tokenizer=processor.tokenizer,
        max_new_tokens=args.max_new_tokens,
        lookup_start_token=args.lookup_start_token,
        lookup_every=args.lookup_every,
        relay_layer=args.relay_layer,
        sample_tiles=args.sample_tiles,
    )
    item = MapQuestion(
        sample_id="318", image_path=args.image, question=QUESTION,
        answer=GROUND_TRUTH, answer_type="Single Entity",
        metadata={"dataset": "mapverse", "qa_id": "318"},
    )
    evaluation = evaluate_answer(response, item)
    result = {
        "qa_id": "318",
        "dataset": "mapverse",
        "image": str(args.image.resolve()),
        "question": QUESTION,
        "ground_truth": GROUND_TRUTH,
        "response": response,
        "evaluation": evaluation,
        "generated_tokens": int(completion.numel()),
        "random_relay_seed": args.seed,
        "num_relays": args.num_relays,
        "relay_parameter_count": int(policy.prototypes.numel()),
        "global_submitted_size": list(global_submitted.size),
        "global_visual_tokens": int(global_memory.shape[0]),
        "coarse_grid": [coarse_grid.height, coarse_grid.width],
        "lookup_start_token": args.lookup_start_token,
        "lookup_every": args.lookup_every,
        "relay_layer_zero_based": args.relay_layer,
        "tile_cache": tile_cache.summary(),
        "lookup_trace": state.trace,
    }
    output_file = args.output_dir / "qa318_relay4_demo.json"
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RAW RESPONSE ===\n", flush=True)
    print(response, flush=True)
    print("\n=== RESULT ===", flush=True)
    print(json.dumps({
        "reward": evaluation["reward"],
        "prediction": evaluation["prediction"],
        "ground_truth": GROUND_TRUTH,
        "lookups": len(state.trace),
        **tile_cache.summary(),
        "output": str(output_file),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

