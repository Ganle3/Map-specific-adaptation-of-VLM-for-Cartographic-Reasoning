#!/usr/bin/env python3
"""Custom GRPO trainer: Qwen/LoRA frozen, only DynamicROIPolicy is updated."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from PIL import Image
from .data import evaluate_answer, load_mapverse_questions
from .grpo import GRPOConfig, ROIGrpoUpdater
from .roi_policy import DynamicROIPolicy, ROIPolicyConfig
from .rollout import RolloutConfig, collect_rollout
from .spatial_mask import SoftROIMask, VisualTokenGrid


def build_inputs(processor, image: Image.Image, question: str, device):
    # Keep the language instruction identical to Debug_GRPO's established
    # baseline. ROI experiments must not accidentally measure a new prompt.
    prompt = f"""This is a cartographic reasoning question from the MapWise dataset.

Use only the supplied map image to answer the question. Carefully inspect the
map legend, labels, colors, boundaries, spatial relationships, and other
relevant visual information.

Question:
{question}

After completing the model's native <think> section, provide only one answer
line using exactly this format:
Final answer: <answer>"""
    out = processor.apply_chat_template([{"role":"user", "content":[{"type":"image", "image":image}, {"type":"text", "text":prompt}]}],
                                        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    out.pop("token_type_ids", None)
    return out.to(device)


def save_checkpoint(path: Path, policy, updater, step: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "optimizer": updater.optimizer.state_dict(), "step": step,
                "config": vars(args), "torch_rng": torch.get_rng_state()}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--adapter")
    p.add_argument("--dataset", required=True); p.add_argument("--image-root", required=True); p.add_argument("--mapwise-root")
    p.add_argument("--output", required=True); p.add_argument("--limit", type=int, default=4)
    p.add_argument("--num-generations", type=int, default=4); p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--resume"); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    # Reuse the project's tested loader so PEFT adapter loading stays identical.
    sparse = Path(__file__).resolve().parents[2] / "SparseVLM_eliminateCONTEXT"; sys_path = __import__('sys').path
    if str(sparse) not in sys_path: sys_path.insert(0, str(sparse))
    from rolling_sparse_qwen3vl import load_model
    model, processor = load_model(args.model, Path(args.adapter) if args.adapter else None)
    device = model.get_input_embeddings().weight.device
    policy = DynamicROIPolicy(ROIPolicyConfig(hidden_size=model.config.text_config.hidden_size)).to(device)
    updater = ROIGrpoUpdater(policy, GRPOConfig(learning_rate=args.learning_rate)); start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device); policy.load_state_dict(state["policy"]); updater.optimizer.load_state_dict(state["optimizer"]); start = state["step"]
    items = load_mapverse_questions(Path(args.dataset), Path(args.image_root), mapwise_root=Path(args.mapwise_root) if args.mapwise_root else None, limit=args.limit)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    log = (output / "training.jsonl").open("a", encoding="utf-8")
    for step, item in enumerate(items[start:], start=start + 1):
        with Image.open(item.image_path) as im: image = im.convert("RGB")
        inputs = build_inputs(processor, image, item.question, device)
        merge = model.config.vision_config.spatial_merge_size; _, h, w = inputs["image_grid_thw"][0].tolist()
        mask = SoftROIMask(VisualTokenGrid(h // merge, w // merge))
        group = [collect_rollout(model, processor, inputs, policy, mask, lambda text: evaluate_answer(text, item)["reward"],
                                 RolloutConfig(args.max_new_tokens, eos_token_id=processor.tokenizer.eos_token_id)) for _ in range(args.num_generations)]
        metrics = updater.update_group(group)
        row = {"step": step, "sample_id": item.sample_id, **metrics, "rewards": [x["reward"] for x in group]}
        log.write(json.dumps(row) + "\n"); log.flush()
        if step % args.checkpoint_every == 0: save_checkpoint(output / f"checkpoint-{step}.pt", policy, updater, step, args)
    save_checkpoint(output / "final.pt", policy, updater, len(items), args)

if __name__ == "__main__": main()
