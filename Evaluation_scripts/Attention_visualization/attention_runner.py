"""Capture image attention during actual greedy Qwen3-VL generation.

No response replay, teacher forcing, or reasoning-to-region inference is used.
Batch inputs are processed sequentially to bound GPU memory usage.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER = ROOT / "Training_outputs/MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12/checkpoints/checkpoint-2500"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Thinking"


@dataclass(frozen=True)
class Question:
    qa_id: str
    question: str
    image: str
    prompt: str | None = None


@dataclass(frozen=True)
class Settings:
    layers: tuple[int, ...] = (8, 16, 24, 35)
    heads: tuple[int, ...] = tuple(range(32))
    max_new_tokens: int = 3072
    max_pixels: int = 1280 * 1280
    load_in_4bit: bool = True
    local_files_only: bool = True


def mapwise_prompt(question: str) -> str:
    return (
        "This is a cartographic reasoning question from the MapWise dataset.\n\n"
        "Use only the supplied map image to answer the question. Carefully inspect the\n"
        "map legend, labels, colors, boundaries, spatial relationships, and other\n"
        "relevant visual information.\n\n"
        f"Question:\n{question}\n\n"
        "Provide your answer on a separate line using exactly this format:\n"
        "Final answer: <answer>"
    )


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = []
    for row in data:
        row = dict(row)
        image = Path(row["image"])
        row["image"] = str((path.parent / image).resolve() if not image.is_absolute() else image)
        questions.append(Question(**row))
    validate_questions(questions)
    return questions


def validate_questions(questions: list[Question]) -> None:
    if not questions or len({q.qa_id for q in questions}) != len(questions):
        raise ValueError("Provide at least one question, with unique qa_id values.")
    for q in questions:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", q.qa_id):
            raise ValueError(f"Unsafe qa_id: {q.qa_id!r}")
        if not q.question.strip():
            raise ValueError(f"Empty question: {q.qa_id}")
        with Image.open(q.image) as image:
            image.verify()


class AttentionCapture:
    """Hook native eager attention outputs, retaining only image-key columns.

    Frame t is the last query row of the forward pass predicting output token t.
    Frame 0 uses the final prompt position; frame t>0 uses output token t-1.
    """

    def __init__(self, model, settings: Settings, image_positions, grid: tuple[int, int]):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        self.modules = {
            m.layer_idx: m for m in model.modules() if isinstance(m, Qwen3VLTextAttention)
        }
        count = len(self.modules)
        self.layers = tuple(i + count if i < 0 else i for i in settings.layers)
        self.heads = settings.heads
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Select unique layers.")
        if not self.heads or len(set(self.heads)) != len(self.heads):
            raise ValueError("Select unique heads.")
        for layer in self.layers:
            if layer not in self.modules:
                raise ValueError(f"Layer {layer} outside 0..{count - 1}")
            nheads = self.modules[layer].config.num_attention_heads
            if any(h < 0 or h >= nheads for h in self.heads):
                raise ValueError(f"Head outside 0..{nheads - 1}")
        self.positions = image_positions
        self.grid = grid
        self.rows = {layer: [] for layer in self.layers}
        self.handles = []
        self.original_configs = {}

    def __enter__(self):
        # The parent model remains eager so it builds an explicit causal mask.
        # Every decoder layer stays eager, whether captured or not. Otherwise
        # changing the selected layers also changes numerical kernels and can
        # alter the greedy generation trajectory of a quantized model.
        # Per-module config copies prevent shared config mutation between layers.
        for layer, module in self.modules.items():
            self.original_configs[layer] = module.config
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "eager"
            if layer in self.layers:
                self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        def capture(module, args, output):
            weights = output[1]
            if weights is None or weights.ndim != 4 or weights.shape[0] != 1:
                raise RuntimeError("Expected eager attention [1, heads, queries, keys].")
            positions = self.positions.to(weights.device)
            row = weights[0, self.heads, -1, :].index_select(-1, positions)
            self.rows[layer].append(row.detach().float().cpu().numpy().reshape(len(self.heads), *self.grid))
        return capture

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for layer, config in self.original_configs.items():
            self.modules[layer].config = config

    def array(self, steps: int) -> np.ndarray:
        if any(len(rows) != steps for rows in self.rows.values()):
            raise RuntimeError("Generation steps and captured forwards differ; refusing misaligned output.")
        return np.stack([np.stack(self.rows[layer]) for layer in self.layers], axis=1)


def generate_one(model, processor, question: Question, settings: Settings, destination: Path) -> dict:
    import torch
    import transformers

    image = Image.open(question.image).convert("RGB")
    prompt = question.prompt or mapwise_prompt(question.question)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    t, h, w = inputs["image_grid_thw"][0].tolist()
    merge = model.config.vision_config.spatial_merge_size
    grid = (h // merge, w // merge)
    if t != 1 or h % merge or w % merge or positions.numel() != grid[0] * grid[1]:
        raise ValueError("Expected one still image with a complete merged patch grid.")
    prompt_length = inputs["input_ids"].shape[1]
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.num_return_sequences = 1
    generation_config.max_new_tokens = settings.max_new_tokens
    generation_config.return_dict_in_generate = False
    generation_config.output_attentions = False
    generation_config.output_hidden_states = False
    generation_config.output_scores = False
    generation_config.use_cache = True
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.temperature = 1.0
    with AttentionCapture(model, settings, positions, grid) as capture, torch.inference_mode():
        sequences = model.generate(**inputs, generation_config=generation_config)
    ids = sequences[0, prompt_length:].tolist()
    attention = capture.array(len(ids))
    raw = processor.tokenizer.decode(ids, skip_special_tokens=False)
    text = processor.tokenizer.decode(ids, skip_special_tokens=True)
    reasoning, separator, answer = text.partition("</think>")
    eos = generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else (eos or [])
    result = {
        "qa_id": question.qa_id, "question": question.question, "image": str(Path(question.image).resolve()),
        "prompt": prompt, "rendered_prompt": processor.tokenizer.decode(inputs["input_ids"][0]),
        "text": text, "raw_text": raw,
        "reasoning": reasoning.removeprefix("<think>").strip() if separator else None,
        "answer": answer.strip() if separator else None,
        "status": "eos" if ids and ids[-1] in eos_ids else "token_limit",
        "token_ids": ids, "tokens": [processor.tokenizer.decode([i]) for i in ids],
        "layers": list(capture.layers), "heads": list(capture.heads),
        "image_grid_thw": [t, h, w], "merged_grid_hw": list(grid), "original_size_wh": list(image.size),
        "image_token_positions": positions.cpu().tolist(), "prompt_tokens": prompt_length,
        "frame_semantics": "Frame t: last query attention in the forward predicting token t (zero-based). Frame 0 is prefill; later frames query the preceding generated token.",
        "attention_axes": ["step", "layer", "head", "patch_y", "patch_x"],
        "settings": asdict(settings), "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "generation_config": generation_config.to_dict(),
        "model_commit": getattr(model.config, "_commit_hash", None),
    }
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "attention.npz", attention=attention)
    (destination / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "response.txt").write_text(text, encoding="utf-8")
    return result


def run_comparison(questions: list[Question], output_dir: str | Path,
                   adapter_path: str | Path = DEFAULT_ADAPTER,
                   model_name: str = DEFAULT_MODEL, settings: Settings = Settings()) -> list[Path]:
    """Public batch API. Returns one standalone comparison HTML path per question."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionAttention
    try:
        from .render import render_comparison
    except ImportError:
        from render import render_comparison

    validate_questions(questions)
    if settings.max_new_tokens < 1 or settings.max_pixels < 1024:
        raise ValueError("max_new_tokens must be positive and max_pixels >= 1024.")
    adapter_path, output_dir = Path(adapter_path).resolve(), Path(output_dir).resolve()
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config["base_model_name_or_path"] != model_name:
        raise ValueError("Baseline must match adapter base_model_name_or_path exactly.")
    for q in questions:
        for variant in ("baseline", "adapted"):
            if (output_dir / q.qa_id / variant).exists():
                raise FileExistsError(f"Use a fresh output directory: {output_dir / q.qa_id / variant}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this 8B experiment.")
    kwargs = dict(device_map="auto", dtype=torch.bfloat16, attn_implementation="eager",
                  local_files_only=settings.local_files_only)
    if settings.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    print(f"Loading {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=settings.local_files_only,
                                              max_pixels=settings.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs).eval()
    # Vision attention is not the question-conditioned decoder attention being inspected.
    for module in model.modules():
        if isinstance(module, Qwen3VLVisionAttention):
            module.config = copy.copy(module.config)
            module.config._attn_implementation = "sdpa"
    for variant in ("baseline", "adapted"):
        if variant == "adapted":
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False).eval()
        for q in questions:
            print(f"{variant}: {q.qa_id}", flush=True)
            dest = output_dir / q.qa_id / variant
            result = generate_one(model, processor, q, settings, dest)
            result.update(model=model_name, adapter=str(adapter_path) if variant == "adapted" else None,
                          variant=variant)
            (dest / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {len(result['token_ids'])} tokens, {result['status']}", flush=True)
    paths = [render_comparison(output_dir / q.qa_id) for q in questions]
    del model
    torch.cuda.empty_cache()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("example_questions.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24, 35])
    parser.add_argument("--heads", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--max-pixels", type=int, default=1280 * 1280)
    parser.add_argument("--bf16", action="store_true", help="Disable default NF4 quantization.")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    settings = Settings(tuple(args.layers), tuple(args.heads), args.max_new_tokens,
                        args.max_pixels, not args.bf16, not args.allow_download)
    for path in run_comparison(load_questions(args.questions), args.output, args.adapter, args.model, settings):
        print(path, flush=True)


if __name__ == "__main__":
    main()
