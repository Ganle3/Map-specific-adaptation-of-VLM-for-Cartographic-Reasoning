"""Qwen3-VL ROI attention-logit intervention (batch 1, greedy, dynamic KV cache).
Place in VLM_adaptation/VPD. Uses your existing inference_mapwise_trl loader.
No images, pixels, prompt tokens or visual-token budgets change across lambdas.
All decoder heads in selected layers receive log(lambda) on ROI visual keys,
only for queries predicting generated tokens. The vision encoder is not boosted.
ROI coordinates are normalized original-image coordinates; merged-token cells
intersecting that rectangle are selected. This is approximate spatial targeting:
visual token embeddings can already contain contextual information from elsewhere.
Outputs include a token-cell preview and observed post-softmax ROI attention mass.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = ROOT / 'Training_outputs' / 'MapWise_scaling1000_GRPO_Qwen3VL8B_BS2GA8_epoch12' / 'checkpoints' / 'checkpoint-3000'
CONTEXT = ('You are given multiple views of the same map. Image 1 is the full map; '
           'the later image(s) are detail crops for visual verification. Use and only according to all '
           'images, especially the detail crop, before answering.\n\n')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def roi_cells(box, rows, cols):
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('ROI must be x1 y1 x2 y2 in [0,1], with positive width/height')
    return [(r, c) for r in range(rows) for c in range(cols)
            if c / cols < x2 and (c + 1) / cols > x1
            and r / rows < y2 and (r + 1) / rows > y1]


def runs(ids, token):
    out, i = [], 0
    while i < len(ids):
        if ids[i] != token:
            i += 1
            continue
        a = i
        while i < len(ids) and ids[i] == token:
            i += 1
        out.append((a, i))
    return out


def select_roi(path):
    from PIL import Image, ImageTk
    import tkinter as tk
    with Image.open(path) as im:
        image = im.convert('RGB')
    window = tk.Tk()
    window.title('Drag ROI around black dot AND nearby map fill; exclude text. Enter = accept')
    scale = min(1, (window.winfo_screenwidth() - 100) / image.width,
                (window.winfo_screenheight() - 180) / image.height)
    w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    photo = ImageTk.PhotoImage(image.resize((w, h)))
    canvas = tk.Canvas(window, width=w, height=h)
    canvas.pack()
    canvas.create_image(0, 0, image=photo, anchor='nw')
    state = {}
    def start(event):
        state['start'] = (max(0, min(w, event.x)), max(0, min(h, event.y)))
        if 'rect' in state:
            canvas.delete(state['rect'])
        state['rect'] = canvas.create_rectangle(*state['start'], *state['start'], outline='cyan', width=2)
    def drag(event):
        if 'start' not in state:
            return
        x, y = max(0, min(w, event.x)), max(0, min(h, event.y))
        canvas.coords(state['rect'], *state['start'], x, y)
        a, b = state['start']
        state['box'] = [min(a, x)/w, min(b, y)/h, max(a, x)/w, max(b, y)/h]
    def accept(_):
        if 'box' in state and state['box'][0] < state['box'][2] and state['box'][1] < state['box'][3]:
            state['accepted'] = True
            window.destroy()
    canvas.bind('<Button-1>', start)
    canvas.bind('<B1-Motion>', drag)
    window.bind('<Return>', accept)
    window.mainloop()
    image.close()
    if not state.get('accepted'):
        raise RuntimeError('ROI selection cancelled; no inference performed')
    return state['box']


def add_bias(torch, mask, query, key, indices, strength):
    """Preserve additive causal mask. Only final query row is changed."""
    qlen, klen = query.shape[-2], key.shape[-2]
    if mask is not None and (mask.ndim != 4 or mask.dtype == torch.bool):
        raise RuntimeError('Expected a 4D additive eager causal mask; refusing to change mask semantics')
    if mask is None and qlen != 1:
        raise RuntimeError('Prefill has no causal mask; refusing unsafe attention intervention')
    if max(indices) >= klen:
        raise RuntimeError('ROI key positions missing (unsupported sliding/static cache)')
    bias = torch.zeros((1, 1, qlen, klen), dtype=query.dtype, device=query.device)
    bias[..., -1, indices] = math.log(strength)
    return bias if mask is None else mask + bias


class AttentionIntervention:
    def __init__(self, torch, modules, layers, indices, input_length, start, end, trace_every):
        self.torch = torch
        self.targets = {id(m): m.layer_idx for m in modules if m.layer_idx in layers}
        self.indices, self.input_length = indices, input_length
        self.start, self.end, self.trace_every = start, end, trace_every
        self.strength = 1.
        self.trace, self.calls, self.active_calls = [], 0, 0
        self.patches = []

    def install(self, modules):
        for name in sorted({m.__class__.__module__ for m in modules}):
            owner = importlib.import_module(name)
            original = getattr(owner, 'eager_attention_forward', None)
            if original is None:
                raise RuntimeError(f'{name} has no eager_attention_forward; unsupported Transformers implementation')
            signature = inspect.signature(original)
            if not {'module', 'query', 'key', 'attention_mask'} <= set(signature.parameters):
                raise RuntimeError(f'Unsupported eager signature: {signature}')
            def wrapped(*args, _original=original, _signature=signature, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                module = bound.arguments['module']
                if id(module) not in self.targets:
                    return _original(*args, **kwargs)
                self.calls += 1
                query, key = bound.arguments['query'], bound.arguments['key']
                qlen, klen = query.shape[-2], key.shape[-2]
                if query.shape[0] != 1 or klen < self.input_length:
                    raise RuntimeError('Only batch 1, full-prefix dynamic cache is supported')
                if klen != self.input_length and qlen != 1:
                    raise RuntimeError('Unexpected chunked decoding; intervention not applied')
                step = klen - self.input_length
                active = step >= self.start and (self.end is None or step <= self.end)
                if active:
                    self.active_calls += 1
                if active and self.strength != 1:
                    bound.arguments['attention_mask'] = add_bias(
                        self.torch, bound.arguments.get('attention_mask'), query, key,
                        self.indices, self.strength)
                output, weights = _original(*bound.args, **bound.kwargs)
                if active and step % self.trace_every == 0:
                    if weights is None:
                        raise RuntimeError('Eager attention returned no weights; cannot verify intervention')
                    after = weights[0, :, -1, :][:, self.indices].float().sum(-1)
                    # Analytic counterfactual at EXACT SAME Q/K, NOT a baseline trajectory.
                    before = after / (self.strength * (1 - after) + after)
                    self.trace.append({'step': step, 'layer': self.targets[id(module)],
                                       'roi_mass_after_per_head': after.detach().cpu().tolist(),
                                       'roi_mass_without_bias_same_qk_per_head': before.detach().cpu().tolist()})
                return output, weights
            self.patches.append((owner, original))
            owner.eager_attention_forward = wrapped

    def restore(self):
        for owner, original in reversed(self.patches):
            owner.eager_attention_forward = original
        self.patches.clear()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', nargs='+', type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--roi-image', type=int, default=2, help='1-based image number; default detail Image 2')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--select-roi', action='store_true')
    group.add_argument('--roi-file', type=Path)
    group.add_argument('--roi-box', nargs=4, type=float, metavar=('X1','Y1','X2','Y2'))
    p.add_argument('--select-only', action='store_true', help='Save ROI without loading model')
    p.add_argument('--lambdas', nargs='+', type=float, default=[1, 2, 5])
    p.add_argument('--layers', default='all', help='all or comma-separated decoder layer indices, e.g. 24,25,26,27')
    p.add_argument('--start-step', type=int, default=0, help='0 predicts first generated token')
    p.add_argument('--end-step', type=int, help='Inclusive final generated-token prediction step')
    p.add_argument('--trace-every', type=int, default=1)
    p.add_argument('--max-new-tokens', type=int, default=1000)
    p.add_argument('--thinking-mode', choices=['auto','on','off'], default='auto')
    p.add_argument('--adapter-path', type=Path, default=DEFAULT_ADAPTER)
    p.add_argument('--prompt-file', type=Path, help='Exact FULL prompt, UTF-8; overrides constructed prompt')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if any(not math.isfinite(x) or x < 1 for x in args.lambdas):
        raise ValueError('Lambdas must be finite and >= 1')
    if args.start_step < 0 or args.trace_every < 1 or args.max_new_tokens < 1:
        raise ValueError('Invalid step/trace/token arguments')
    if args.end_step is not None and args.end_step < args.start_step:
        raise ValueError('end-step must be >= start-step')
    paths = [x.expanduser().resolve() for x in args.images]
    target = args.roi_image - 1
    if not 0 <= target < len(paths):
        raise ValueError('roi-image does not identify an input image')
    hashes = [hashlib.sha256(x.read_bytes()).hexdigest() for x in paths]
    if args.roi_file:
        stored = json.loads(args.roi_file.read_text(encoding='utf-8-sig'))
        if stored['image_sha256'] != hashes[target]:
            raise ValueError('ROI belongs to a different image')
        box = stored['box_normalized']
    else:
        box = select_roi(paths[target]) if args.select_roi else args.roi_box
    roi_cells(box, 1, 1)  # validate normalized rectangle
    out = args.output_dir.expanduser().resolve()
    save(out / 'roi.json', {'image_path': str(paths[target]), 'image_number': target + 1,
                            'image_sha256': hashes[target], 'box_normalized': box})
    if args.select_only:
        print(f'Saved ROI: {out / "roi.json"}')
        return
    import torch
    import transformers
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(ROOT / 'Evaluation_scripts' / 'GRPO_ablation'))
    import inference_mapwise_trl as inference
    model, processor, loaded_adapter = inference.load_model_and_processor(adapter_path=args.adapter_path.resolve())
    model.eval()
    # Use eager for ALL branches including lambda=1; preserve standard HF causal masking.
    model.set_attn_implementation('eager')
    modules = [m for m in model.modules() if m.__class__.__name__ == 'Qwen3VLTextAttention']
    if not modules or any(m.config._attn_implementation != 'eager' for m in modules):
        raise RuntimeError('Expected Qwen3VLTextAttention using eager; intervention is not compatible with this model')
    layers = {m.layer_idx for m in modules} if args.layers == 'all' else {int(x) for x in args.layers.split(',')}
    if not layers or not layers <= {m.layer_idx for m in modules}:
        raise ValueError('Invalid decoder layer indices')
    prompt = args.prompt_file.read_text(encoding='utf-8-sig') if args.prompt_file else CONTEXT + inference.build_mapwise_prompt(args.question)
    images = []
    try:
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        inputs = inference.prepare_multimodal_inputs_multi(processor, images, prompt, thinking_mode=args.thinking_mode)
        if inputs['input_ids'].shape[0] != 1:
            raise ValueError('Batch size must be 1')
        if 'attention_mask' in inputs and not bool(inputs['attention_mask'].eq(1).all()):
            raise ValueError('Unpadded single-example input required')
        ids = inputs['input_ids'][0].detach().cpu().tolist()
        image_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        spans = runs(ids, image_id)
        grids = inputs['image_grid_thw'].detach().cpu().tolist()
        merge = int(processor.image_processor.merge_size)
        counts = [t*h*w//merge**2 for t,h,w in grids]
        if len(grids) != len(paths) or len(spans) != len(paths) or [b-a for a,b in spans] != counts:
            raise ValueError('Image-pad runs do not match image grids')
        t, h, w = grids[target]
        if t != 1 or h % merge or w % merge:
            raise ValueError('ROI mapping supports static images with merge-aligned grids only')
        rows, cols = h//merge, w//merge
        cells = roi_cells(box, rows, cols)
        indices = [spans[target][0] + r*cols + c for r,c in cells]
        preview = images[target].copy()
        draw = ImageDraw.Draw(preview)
        for r,c in cells:
            draw.rectangle((c*preview.width/cols,r*preview.height/rows,(c+1)*preview.width/cols,(r+1)*preview.height/rows), outline='cyan', width=2)
        draw.rectangle(tuple(x*d for x,d in zip(box,[preview.width,preview.height,preview.width,preview.height])),outline='yellow',width=2)
        preview.save(out / 'roi_token_cells.png')
        preview.close()
    finally:
        for image in images:
            image.close()
    report = {'question':args.question, 'prompt':prompt, 'image_paths':list(map(str,paths)),
              'image_sha256':hashes, 'image_grids':grids, 'visual_tokens_per_image':counts,
              'input_tokens':len(ids), 'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
              'roi_image':target+1, 'roi_box_normalized':box, 'merged_grid_rows_cols':[rows,cols],
              'roi_cells_row_col':cells, 'roi_key_indices':indices, 'layers':sorted(layers),
              'heads':'all', 'backend':'eager for all branches', 'adapter':str(loaded_adapter),
              'torch_version':torch.__version__, 'transformers_version':transformers.__version__,
              'start_step':args.start_step, 'end_step':args.end_step,
              'note':'Attention intervention is spatially approximate. Same-Q/K counterfactual mass is derived analytically, not independently measured. Trace is a manipulation check, not proof of perceptual correctness.',
              'runs':[]}
    print(f'Visual tokens: {counts}; selected ROI keys: {len(indices)}; preview: {out / "roi_token_cells.png"}',flush=True)
    inputs = inference.move_inputs_to_model_device(inputs, model)
    intervention = AttentionIntervention(torch,modules,layers,indices,len(ids),args.start_step,args.end_step,args.trace_every)
    try:
        intervention.install(modules)
        # Always include a lambda=1 reference, regardless of provided lambdas.
        for strength in sorted(set([1.] + args.lambdas)):
            intervention.strength = strength
            intervention.trace, intervention.calls, intervention.active_calls = [], 0, 0
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            for module in model.modules():
                if 'rope_deltas' in vars(module):
                    module.rope_deltas = None
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, num_beams=1,
                                           num_return_sequences=1, use_cache=True, cache_implementation='dynamic',
                                           max_new_tokens=args.max_new_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if not intervention.calls or not intervention.active_calls or not intervention.trace:
                raise RuntimeError('No verified active attention calls; do not interpret this run as an intervention')
            tokens = generated[0,len(ids):]
            raw = processor.batch_decode(tokens[None,:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            item = {'lambda':strength,'logit_bias':math.log(strength),'raw_response':raw,
                    'final_answer':inference.extract_final_answer(raw),'generated_tokens':int(tokens.numel()),
                    'generated_token_ids':tokens.detach().cpu().tolist(),'seconds':time.perf_counter()-started,
                    'decoder_calls':intervention.calls,'active_calls':intervention.active_calls,
                    'trace':intervention.trace}
            report['runs'].append(item)
            save(out / 'results.json', report)
            print(f'lambda={strength:g}: {item["final_answer"]}',flush=True)
    finally:
        intervention.restore()
    print(f'Saved: {out / "results.json"}')


if __name__ == '__main__':
    main()
