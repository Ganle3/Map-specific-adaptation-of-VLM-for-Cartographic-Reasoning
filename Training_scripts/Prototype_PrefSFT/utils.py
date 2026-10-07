"""Data, processor and frozen-backbone utilities for Prototype PrefSFT."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = ROOT / "Evaluation_scripts" / "GRPO_ablation"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))
import inference_mapwise_trl as inference  # noqa: E402


def load_questions(path: str | Path, cache_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load only complete mixed groups and merge immutable baseline cache by key."""
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = [q for q in source["questions"] if q.get("rollout_outcome") == "mixed"
                 and any(t.get("correct") for t in q["trajectories"])
                 and any(not t.get("correct") for t in q["trajectories"])]
    if cache_path is not None:
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        scores = cache.get("scores", cache)
        for q in questions:
            for t in q["trajectories"]:
                key = f"{q['dataset']}::{q['qa_id']}::{t['rollout_index']}"
                if key not in scores:
                    raise KeyError(f"Missing baseline score for {key}")
                t["baseline_normalized_logprob"] = float(scores[key]["baseline_normalized_logprob"])
    return questions


def trajectory_key(question: dict[str, Any], trajectory: dict[str, Any]) -> str:
    return f"{question['dataset']}::{question['qa_id']}::{trajectory['rollout_index']}"


def vision_cache_key(question: dict[str, Any]) -> str:
    """Question identity is safe: every rollout of a question has one image."""
    return f"vision-v1::{question['dataset']}::{question['qa_id']}::{question['image_name']}"


def image_for_question(question: dict[str, Any], mapwise_root: str | Path,
                       mapverse_root: str | Path) -> Path:
    root = Path(mapwise_root if question["dataset"] == "mapwise" else mapverse_root)
    sample = dict(question["source_metadata"])
    if question["dataset"] == "mapverse":
        sample.update(country="mapverse", map_no=question["image_name"],
                      image_name=question["image_name"], qa_id=question["qa_id"])
    else:
        sample["qa_id"] = question["qa_id"]
    return inference.resolve_mapwise_image(sample, root)


def prompt(question: dict[str, Any]) -> str:
    # Rollouts were generated with this exact shared prompt for both datasets.
    return inference.build_mapwise_prompt(question["question"])


def make_trajectory_inputs(processor, question: dict[str, Any], trajectory: dict[str, Any],
                           image_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Prompt + stored raw response, with labels only for stored response tokens."""
    with Image.open(image_path) as image:
        prompt_inputs = inference.prepare_multimodal_inputs(
            processor, image.convert("RGB"), prompt(question), thinking_mode="auto")
    prompt_ids = prompt_inputs["input_ids"][0]
    response_ids = processor.tokenizer(str(trajectory["raw_response"]), add_special_tokens=False,
                                       return_tensors="pt")["input_ids"][0]
    if response_ids.numel() == 0:
        raise ValueError(f"Empty raw_response for {trajectory_key(question, trajectory)}")
    input_ids = torch.cat((prompt_ids, response_ids)).unsqueeze(0)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.numel():] = response_ids
    result = {k: v for k, v in prompt_inputs.items() if torch.is_tensor(v)}
    result.update(input_ids=input_ids, labels=labels,
                  attention_mask=torch.ones_like(input_ids))
    return {k: v.to(device) for k, v in result.items()}


def collate_trajectory_inputs(rows: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad text; concatenate Qwen flattened visual tensors/grids."""
    maximum = max(x["input_ids"].shape[1] for x in rows)
    result: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "labels", "attention_mask"):
        chunks = []
        for row in rows:
            value = row[key]
            pad = maximum - value.shape[1]
            if pad:
                fill = pad_token_id if key == "input_ids" else (0 if key == "attention_mask" else -100)
                value = torch.nn.functional.pad(value, (0, pad), value=fill)
            chunks.append(value)
        result[key] = torch.cat(chunks, dim=0)
    for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask"):
        if key in rows[0]:
            result[key] = torch.cat([x[key] for x in rows], dim=0)
    return result


def normalized_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per sequence mean log p(token), excluding every -100 label."""
    target, logits = labels[:, 1:], logits[:, :-1]
    mask = target.ne(-100)
    safe = target.masked_fill(~mask, 0)
    token_logp = logits.log_softmax(dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def load_backbone(model_name: str, adapter_path: str | Path, *, device_map: str = "auto",
                  load_in_4bit: bool = True):
    """Load base+GRPO adapter and freeze absolutely every existing parameter."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    kwargs = dict(torch_dtype=torch.bfloat16, device_map=device_map, low_cpu_mem_usage=True)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def qwen_visual(model):
    """Resolve native and PEFT Qwen3-VL wrappers without relying on a name path."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "visual"):
        return base.visual
    if hasattr(base, "model") and hasattr(base.model, "visual"):
        return base.model.visual
    raise AttributeError("Could not find visual module on loaded Qwen3-VL/PEFT model.")


def cosine_metrics(prototypes: torch.Tensor, attention: torch.Tensor | None) -> dict[str, float]:
    p = torch.nn.functional.normalize(prototypes.detach().float(), dim=-1)[0]
    matrix = p @ p.T
    offdiag = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)]
    output = {"prototype/mean_cosine_similarity": float(offdiag.mean().cpu()) if offdiag.numel() else 1.0}
    if attention is not None:
        # attention is [images, heads, N+K, N+K]; compare P->Z maps.
        k = p.shape[0]
        a = attention[:, :, -k:, :-k].mean(dim=(0, 1)).float()
        a = torch.nn.functional.normalize(a, dim=-1)
        am = a @ a.T
        off = am[~torch.eye(k, dtype=torch.bool, device=am.device)]
        output["prototype/attention_overlap"] = float(off.mean().cpu()) if off.numel() else 1.0
    return output
