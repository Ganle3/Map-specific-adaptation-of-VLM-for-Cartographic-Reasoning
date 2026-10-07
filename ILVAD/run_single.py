"""Single-question, training-free ILVAD experiment for Qwen3-VL + PEFT LoRA."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from ilvad_qwen import AttentionRecorder, ILVADConfig, ILVADContext, saliency_from_records

DEFAULT_IMAGE = ROOT.parents[0] / "MapVerse" / "data" / "imgs" / "2.webp"
DEFAULT_ADAPTER = (
    ROOT / "Training_outputs" / "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500"
    / "checkpoint-3000"
)
QUESTION = "How many of the cities are in Russian military control?"

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    p.add_argument("--question", default=QUESTION)
    p.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    p.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking",
                   help="Official Hugging Face Qwen checkpoint; Unsloth is not used.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).with_name("outputs") / "mapverse_image2_checkpoint-3000")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--mode", choices=("rolling", "fixed"), default="rolling",
                   help="rolling is the default extension; fixed reproduces the paper's two-pass map.")
    p.add_argument("--fixed-t", type=int, default=10,
                   help="Fixed mode only: early decoding steps for the paper-reproduction map.")
    p.add_argument("--window", type=int, default=4, help="Rolling mode: latest native-attention steps per fresh map.")
    p.add_argument("--update-interval", type=int, default=4, help="Rolling mode: replace map every K steps.")
    p.add_argument("--warmup", type=int, default=4, help="Rolling mode: native-attention steps before first map.")
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--tau", type=float, default=5.0)
    p.add_argument("--layers", nargs="+", type=int, default=list(range(8, 36)))
    p.add_argument("--thinking-mode", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Optional bitsandbytes quantization; off means official BF16 loading.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--baseline-only", action="store_true")
    return p.parse_args()

def main():
    args = arguments()
    if (args.fixed_t < 1 or args.max_new_tokens < 1 or args.window < 1 or
            args.update_interval < 1 or args.warmup < args.window):
        raise ValueError("Invalid token-window parameters; require warmup >= window >= 1.")
    if not args.image.is_file(): raise FileNotFoundError(args.image)
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    adapter = args.adapter_path.expanduser().resolve()
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Not a PEFT adapter directory: {adapter}")
    model_source = resolve_model_source(args.model)
    # Passing a concrete snapshot path avoids a Transformers 5.5 tokenizer
    # metadata request even when all official Qwen files are already cached.
    source_is_local = Path(model_source).is_dir()
    local_only = args.local_files_only or source_is_local
    processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_only)
    model_kwargs = dict(device_map="auto", dtype=torch.bfloat16,
                        low_cpu_mem_usage=True, local_files_only=local_only,
                        attn_implementation="eager")
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    print(f"Loading official base model: {model_source}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)
    print(f"Attaching PEFT adapter: {adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    prompt = build_prompt(args.question)
    with Image.open(args.image) as im:
        inputs = prepare_inputs(processor, im.convert("RGB"), prompt, args.thinking_mode)
    inputs.pop("token_type_ids", None)
    inputs = move_to_model_device(inputs, model)
    positions = (inputs["input_ids"][0] == model.config.image_token_id).nonzero().flatten()
    if not len(positions): raise RuntimeError("No Qwen image tokens found.")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    def generate(label, maximum, context=None):
        started = time.perf_counter()
        with AttentionRecorder(model, positions) as recorder, (context or _Null()), torch.inference_mode():
            ids = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True, max_new_tokens=maximum)
        new = ids[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
        np = __import__("numpy")
        array = np.stack([[step[layer].numpy() for layer in sorted(step)] for step in recorder.records])
        np.savez_compressed(out / f"{label}_attention.npz", attention=array, layers=sorted(recorder.modules), image_positions=positions.cpu().numpy())
        record = {"label":label, "question":args.question, "image":str(args.image.resolve()), "adapter":str(adapter), "prompt":prompt,
                  "token_ids":new, "text":processor.tokenizer.decode(new, skip_special_tokens=True), "steps":len(recorder.records),
                  "seconds":round(time.perf_counter()-started, 3)}
        (out / f"{label}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        if isinstance(context, ILVADContext) and context.map_history:
            refresh_steps, maps = zip(*context.map_history)
            np.savez_compressed(out / f"{label}_rolling_maps.npz",
                                refresh_steps=np.asarray(refresh_steps), enhanced_maps=np.stack(maps))
        return recorder.records, record
    if not args.baseline_only:
        base_records, _ = generate("baseline", args.max_new_tokens)
        cfg = ILVADConfig(args.alpha, args.beta, args.tau, tuple(args.layers),
                          args.mode == "rolling", args.window, args.update_interval, args.warmup)
        if args.mode == "fixed":
            warm_records, _ = generate("saliency", args.fixed_t)
            saliency = saliency_from_records(warm_records, args.tau, args.alpha)
            __import__("numpy").save(out / "enhanced_map.npy", saliency.numpy())
        else:
            saliency = None
        context = ILVADContext(model, int(positions[0]), int(positions[-1]) + 1, saliency, cfg)
        ilvad_records, _ = generate("ilvad", args.max_new_tokens, context)
    (out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

class _Null:
    def __enter__(self): return self
    def __exit__(self, *_): return False

def build_prompt(question):
    return f"""This is a cartographic reasoning question. Use only the supplied map image.
Carefully inspect the map legend, labels, colors, boundaries, spatial relationships, and other relevant visual information.

Question:
{question}

Provide your answer on a separate line using exactly this format:
Final answer: <answer>"""

def prepare_inputs(processor, image, prompt, thinking_mode):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    options = {} if thinking_mode == "auto" else {"enable_thinking": thinking_mode == "on"}
    return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt", **options)

def move_to_model_device(inputs, model):
    device = next(model.parameters()).device
    try:
        return inputs.to(device)
    except AttributeError:
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

def resolve_model_source(model):
    """Prefer a complete local snapshot of the requested *official* HF model."""
    explicit = Path(model).expanduser()
    if explicit.is_dir():
        return str(explicit.resolve())
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model.replace("/", "--")
    snapshots = cache_root / cache_name / "snapshots"
    if snapshots.is_dir():
        candidates = sorted(
            (path for path in snapshots.iterdir()
             if (path / "config.json").is_file() and (path / "model.safetensors.index.json").is_file()),
            key=lambda path: path.stat().st_mtime, reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return model
if __name__ == "__main__": main()
