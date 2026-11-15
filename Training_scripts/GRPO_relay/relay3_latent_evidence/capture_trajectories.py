#!/usr/bin/env python3
"""Matched-prefix Relay3/adapter layer trajectories and visual-ablation evidence."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RELAY_ROOT = HERE.parent
VLM_ROOT = HERE.parents[2]
THESIS_ROOT = VLM_ROOT.parent
SPARSE_ROOT = VLM_ROOT / "SparseVLM_eliminateCONTEXT"
for path in (HERE, RELAY_ROOT, SPARSE_ROOT, RELAY_ROOT / "relay3_visualization"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Image_question_CoT_shared_relay.dynamic_decode import (
    _decode_pair, _decoder_layers, _group_position_ids, _set_custom_mask_attention,
)
from Image_question_CoT_shared_relay.layout import build_prefill as build_relay_prefill
from relay3_visualization.run_visualization import SelectedQuestion, prepare
from relay_module import RelayModule
from evaluate_core import load_evaluation_questions, load_policy
from rolling_sparse_qwen3vl import load_model
from utils import (
    _decoder_and_head, _vl_backbone, build_adapter_only_prefill,
    decoder_hidden_size, explicit_relay_decoder_inputs, qwen_lm,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures


DEFAULT_OUTPUT = VLM_ROOT / "Training_outputs/Prototype/relay3_latent_evidence"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=DEFAULT_OUTPUT / "selected_step4000.json")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "trajectories_step4000")
    p.add_argument("--mapverse-dataset", type=Path, default=VLM_ROOT / "Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json")
    p.add_argument("--mapwise-dataset", type=Path, default=VLM_ROOT / "Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json")
    p.add_argument("--mapverse-root", type=Path, default=THESIS_ROOT / "MapVerse/data/imgs")
    p.add_argument("--mapwise-root", type=Path, default=VLM_ROOT / "Datasets/mapwise-dataset")
    p.add_argument("--adapter-run-dir", type=Path, default=VLM_ROOT / "Training_outputs/Prototype/adapter_only/checkpoint")
    p.add_argument("--relay-run-dir", type=Path, default=VLM_ROOT / "Training_outputs/Prototype/relay3/checkpoint")
    p.add_argument("--step", type=int, default=4000)
    p.add_argument("--max-pixels", type=int, default=1_000_000)
    p.add_argument("--max-reference-tokens", type=int, default=40)
    p.add_argument("--capture-steps", type=int, nargs="+", default=[0, 4, 9, 19, 39])
    p.add_argument("--limit", type=int, help="Pilot on the first N selected questions")
    p.add_argument("--variants", nargs="+",
                   choices=("adapter_only", "relay3", "relay3_adapter_no_relay"),
                   default=["adapter_only", "relay3"],
                   help="Optional same-checkpoint relay-disabled control uses the Relay3 adapter.")
    p.add_argument("--model", default=None)
    p.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    p.set_defaults(load_in_4bit=True)
    a = p.parse_args()
    if not a.selection.is_file() or a.max_reference_tokens < 1 or min(a.capture_steps) < 0:
        p.error("Selection must exist and token limits must be positive")
    return a


def blank_features(features):
    return BaseModelOutputWithDeepstackFeatures(
        pooler_output=tuple(torch.zeros_like(x) for x in features.pooler_output),
        deepstack_features=[tuple(torch.zeros_like(x) for x in level)
                            for level in features.deepstack_features],
    )


class LayerCapture:
    def __init__(self, language_model, *, relay_count: int):
        self.relay_count = relay_count
        self.active = False
        self.language = []
        self.relay = []
        self.handles = [layer.register_forward_hook(self._hook) for layer in language_model.layers]

    def _hook(self, _layer, _inputs, output):
        if not self.active:
            return
        states = output[0] if isinstance(output, tuple) else output
        if states.ndim != 3 or states.shape[0] != 1 or states.shape[1] != 1 + self.relay_count:
            raise RuntimeError(f"Unexpected per-step decoder states: {tuple(states.shape)}")
        self.language.append(states[0, 0].detach().float().cpu().numpy())
        if self.relay_count:
            self.relay.append(states[0, 1:].detach().float().cpu().numpy())

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def replay(model, processor, source, features, policy, reference_ids, capture_steps, *, relay):
    augmented, _ = (build_relay_prefill(model, processor.tokenizer, source, features,
                                        policy, with_grad=False) if relay else
                    build_adapter_only_prefill(source, features))
    inputs = augmented["input_ids"]
    prompt_embeds, visual_mask, deepstack = explicit_relay_decoder_inputs(
        model, inputs, augmented["mm_encoder_outputs"]["image"]
    )
    positions, _ = _vl_backbone(model).get_rope_index(
        inputs, augmented["mm_token_type_ids"],
        image_grid_thw=augmented["image_grid_thw"],
        attention_mask=augmented["attention_mask"],
    )
    prefix_length = inputs.shape[1] - 1
    language_model, lm_head = _decoder_and_head(model)
    _set_custom_mask_attention(language_model)
    prefill = language_model(
        input_ids=None, inputs_embeds=prompt_embeds[:, :prefix_length],
        attention_mask=augmented["attention_mask"][:, :prefix_length],
        position_ids=positions[..., :prefix_length],
        visual_pos_masks=visual_mask[:, :prefix_length],
        deepstack_visual_embeds=deepstack, use_cache=True, return_dict=True,
    )
    past = prefill.past_key_values
    if past.get_seq_length() != prefix_length:
        raise RuntimeError("Prefill cache length mismatch")
    visible = (augmented["relay_visible_prompt_mask"][:prefix_length].to(inputs.device)
               if relay else None)
    capture = LayerCapture(language_model, relay_count=policy.prototypes.shape[0] if relay else 0)
    records = {}
    try:
        for t in range(min(len(reference_ids), max(capture_steps) + 1)):
            token = inputs[0, -1] if t == 0 else reference_ids[t - 1]
            position = (positions[..., -1:] if t == 0 else
                        torch.full_like(positions[..., -1:], int(positions.max()) + t))
            capture.active = t in capture_steps
            capture.language = []
            capture.relay = []
            if relay:
                logits, past = _decode_pair(
                    model, token=token, relay=policy.prototypes.detach(),
                    position=position, past=past, relay_visible_past=visible,
                )
                visible = torch.cat((visible, visible.new_zeros(1)))
            else:
                embed = qwen_lm(model).get_input_embeddings()(token.view(1, 1))
                mask = torch.zeros((1, 1, 1, past.get_seq_length() + 1),
                                   dtype=embed.dtype, device=embed.device)
                states = _decoder_layers(language_model, embed, position, mask,
                                         past_key_values=past)
                logits = lm_head(states[:, 0])
            if capture.active:
                if len(capture.language) != len(language_model.layers):
                    raise RuntimeError("Layer hooks missed a decoder layer")
                logp = torch.log_softmax(logits[0].float(), dim=-1)
                records[t] = {
                    "language": np.stack(capture.language),
                    "relay": (np.stack(capture.relay) if relay else None),
                    "reference_token_logp": float(logp[reference_ids[t]]),
                    "reference_token_id": int(reference_ids[t]),
                }
            capture.active = False
    finally:
        capture.close()
    return records


def cosine_distance(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - (a * b).sum(axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12
    )


def analyze_one(model, processor, policy, variant, record, item, args):
    selected = SelectedQuestion(record["dataset"], record["array_index_0based"], item,
                                f"{record['dataset']}_index{record['dataset_index']:04d}")
    source, features, *_ = prepare(model, processor, selected, args)
    reference_ids = processor.tokenizer.encode(record["relay_response"], add_special_tokens=False)
    reference_ids = torch.tensor(reference_ids[:args.max_reference_tokens],
                                 dtype=torch.long, device=source["input_ids"].device)
    if not reference_ids.numel():
        raise RuntimeError("Reference response tokenized to an empty sequence")
    steps = sorted(t for t in set(args.capture_steps) if t < len(reference_ids))
    normal = replay(model, processor, source, features, policy, reference_ids, steps,
                    relay=variant == "relay3")
    blank = replay(model, processor, source, blank_features(features), policy, reference_ids,
                   steps, relay=variant == "relay3")
    output = args.output_dir / selected.folder / variant
    output.mkdir(parents=True, exist_ok=True)
    save = {}
    visual_tokens = torch.cat(tuple(features.pooler_output), dim=0).float().cpu().numpy()
    save["image_merger_mean"] = visual_tokens.mean(axis=0).astype(np.float32)
    # A reproducible spatial subsample makes the exploratory latent plot manageable.
    positions = np.linspace(0, len(visual_tokens) - 1,
                            num=min(128, len(visual_tokens)), dtype=int)
    save["image_merger_sample"] = visual_tokens[positions].astype(np.float32)
    summary = []
    for t in steps:
        original, ablated = normal[t], blank[t]
        distances = cosine_distance(original["language"], ablated["language"])
        token = processor.tokenizer.decode([original["reference_token_id"]])
        logp_effect = original["reference_token_logp"] - ablated["reference_token_logp"]
        save[f"language_step{t}"] = original["language"].astype(np.float32)
        save[f"language_blank_step{t}"] = ablated["language"].astype(np.float32)
        if original["relay"] is not None:
            save[f"relay_step{t}"] = original["relay"].astype(np.float32)
        summary.append({
            "step": t, "reference_next_token": token,
            "reference_next_token_id": original["reference_token_id"],
            "logp_reference_real_image": original["reference_token_logp"],
            "logp_reference_blank_image": ablated["reference_token_logp"],
            "image_logp_effect": logp_effect,
            "hidden_image_cosine_distance_by_layer": distances.tolist(),
        })
    np.savez_compressed(output / "states.npz", **save)
    metadata = {
        "dataset": record["dataset"], "dataset_index": record["dataset_index"],
        "qa_id": record["qa_id"], "category": record["category"],
        "question": record["question"], "ground_truth": record["ground_truth"],
        "image_name": record["image_name"], "variant": variant,
        "reference": "Relay3 greedy response text from the saved step-4000 evaluation; "
                     "the same token IDs are teacher-forced in both models and image conditions",
        "ablation": "All post-ViT merger and DeepStack image embeddings set to zero, "
                    "preserving image token count, positions and text",
        "steps": summary,
    }
    (output / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[{variant}] {selected.folder} steps={steps}", flush=True)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL capture")
    manifest = json.loads(args.selection.read_text(encoding="utf-8"))
    if int(manifest["checkpoint_step"]) != args.step:
        raise ValueError("Selection/checkpoint step mismatch")
    rows = manifest["questions"][:args.limit]
    pools = {
        "mapverse": load_evaluation_questions(args.mapverse_dataset, args.mapverse_root, args.mapwise_root),
        "mapwise": load_evaluation_questions(args.mapwise_dataset, args.mapverse_root, args.mapwise_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_variant, run_dir in (("adapter_only", args.adapter_run_dir),
                                   ("relay3", args.relay_run_dir)):
        requested = (["adapter_only"] if model_variant == "adapter_only" else
                     ["relay3", "relay3_adapter_no_relay"])
        requested = [variant for variant in requested if variant in args.variants]
        if not requested:
            continue
        checkpoint = run_dir / f"checkpoint-{args.step}.pt"
        adapter = run_dir / f"checkpoint-{args.step}"
        if not checkpoint.is_file() or not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Incomplete {model_variant} checkpoint {checkpoint}")
        model, processor = load_model(args.model, adapter, load_in_4bit=args.load_in_4bit)
        model.eval()
        policy = None
        if "relay3" in requested:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            policy = RelayModule(decoder_hidden_size(model),
                                 num_relays=int(state["num_relay_tokens"])).to(
                                     model.get_input_embeddings().weight.device).eval()
            load_policy(policy, checkpoint, "dynamic_cot_relay", "image_question_cot_shared")
        try:
            for row in rows:
                item = pools[row["dataset"]][row["array_index_0based"]]
                if (item.sample_id != row["qa_id"] or item.question != row["question"]
                        or item.answer != row["ground_truth"]):
                    raise ValueError(f"Selection no longer matches dataset: {row['qa_id']}")
                for variant in requested:
                    analyze_one(model, processor, policy, variant, row, item, args)
        finally:
            del model, processor, policy
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Captured {args.variants} for {len(rows)} questions in {args.output_dir}")


if __name__ == "__main__":
    main()
