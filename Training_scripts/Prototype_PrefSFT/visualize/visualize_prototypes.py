#!/usr/bin/env python3
"""Save routing collection/distribution maps and prototype diagnostics."""
from __future__ import annotations
import argparse, json, sys
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prototype_module import CachedVisionForward, PrototypeRoutingModule, VisionPrototypeHook
from utils import image_for_question, load_backbone, load_questions, make_trajectory_inputs, model_device, qwen_visual, vision_cache_key


def hidden_size(visual):
    for obj, name in ((getattr(visual, "config", None), "hidden_size"), (getattr(visual, "config", None), "embed_dim"), (getattr(getattr(visual, "merger", None), "linear_fc1", None), "in_features")):
        if getattr(obj, name, None): return int(getattr(obj, name))
    raise AttributeError("Cannot infer vision width")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("rollouts", "checkpoint", "adapter_path", "mapwise_image_root", "mapverse_image_root", "output_dir"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--qa-id", required=True); p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Thinking")
    p.add_argument("--full-precision", action="store_true"); p.add_argument("--vision-cache-dir", type=Path, default=None)
    p.add_argument("--min-pixels", type=int, default=65536); p.add_argument("--max-pixels", type=int, default=1000000)
    args = p.parse_args()
    question = next(q for q in load_questions(args.rollouts) if q["qa_id"] == args.qa_id)
    model, processor = load_backbone(args.model_name, args.adapter_path, load_in_4bit=not args.full_precision, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    device, visual = model_device(model), qwen_visual(model)
    state = torch.load(args.checkpoint, map_location=device); train_args = state.get("training_args", {})
    proto = PrototypeRoutingModule(hidden_size(visual), state["prototypes"].shape[0], alpha_init=float(state["alpha"]), tau1=float(train_args.get("tau1", .1)), tau2=float(train_args.get("tau2", .1))).to(device=device, dtype=torch.bfloat16)
    proto.prototypes.data.copy_(state["prototypes"].to(device)); proto.alpha.data.copy_(state["alpha"].to(device))
    hook = VisionPrototypeHook(visual, proto); cache = CachedVisionForward(visual, args.vision_cache_dir) if args.vision_cache_dir else None
    image = image_for_question(question, args.mapwise_image_root, args.mapverse_image_root)
    row = make_trajectory_inputs(processor, question, question["trajectories"][0], image, device)
    row.pop("labels"); grid = row["image_grid_thw"]; args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with torch.no_grad():
            cache_context = cache.use([vision_cache_key(question)]) if cache else nullcontext()
            with cache_context:
                with hook.context(grid, capture_attention=True): model(**row)
        collect, distribute = proto.last_collect_attention[0].float().cpu().numpy(), proto.last_distribute_attention[0].float().cpu().numpy()
        t, h, w = map(int, grid[0].cpu().tolist()); original = plt.imread(image)
        for pi in range(collect.shape[0]):
            for name, values in (("collect", collect[pi]), ("distribute", distribute[:, pi])):
                heat = np.asarray(values).reshape(t, h, w).mean(0)
                plt.figure(figsize=(8, 6)); plt.imshow(original); plt.imshow(heat, alpha=.48, cmap="magma", extent=(0, original.shape[1], original.shape[0], 0)); plt.axis("off"); plt.title(f"P{pi} {name}")
                plt.savefig(args.output_dir / f"prototype_{pi}_{name}.png", dpi=180, bbox_inches="tight"); plt.close()
        pvec = torch.nn.functional.normalize(proto.prototypes.float(), dim=-1); cos = (pvec @ pvec.T).cpu().numpy()
        collection_norm = collect / (np.linalg.norm(collect, axis=1, keepdims=True) + 1e-8)
        np.save(args.output_dir / "prototype_cosine_similarity.npy", cos); np.save(args.output_dir / "prototype_attention_overlap.npy", collection_norm @ collection_norm.T)
        (args.output_dir / "diagnostics.json").write_text(json.dumps({"residual_relative_norm": float(proto.last_residual_relative_norm.cpu()), "grid_thw": [t, h, w], "tau1": proto.tau1, "tau2": proto.tau2}, indent=2))
    finally:
        hook.remove()
        if cache: cache.remove()


if __name__ == "__main__": main()
