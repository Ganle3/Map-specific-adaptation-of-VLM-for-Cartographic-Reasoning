#!/usr/bin/env python3
"""Save P→patch attention overlays and prototype diagnostics for one question."""
from __future__ import annotations
import argparse, json, sys
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
from prototype_module import CachedVisionForward, PrototypeTransformer, VisionPrototypeHook
from utils import image_for_question, load_backbone, load_questions, make_trajectory_inputs, model_device, qwen_visual, vision_cache_key


def hidden_size(visual):
    for x, n in ((getattr(visual, "config", None), "hidden_size"), (getattr(visual, "config", None), "embed_dim"),
                 (getattr(getattr(visual, "merger", None), "linear_fc1", None), "in_features")):
        if getattr(x, n, None): return int(getattr(x, n))
    raise AttributeError("Cannot infer vision width")


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--rollouts",type=Path,required=True); p.add_argument("--qa-id",required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--adapter-path",type=Path,required=True); p.add_argument("--mapwise-image-root",type=Path,required=True); p.add_argument("--mapverse-image-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--model-name",default="Qwen/Qwen3-VL-8B-Thinking"); p.add_argument("--full-precision",action="store_true"); p.add_argument("--vision-cache-dir",type=Path,default=None); args=p.parse_args()
    question=next(q for q in load_questions(args.rollouts) if q["qa_id"]==args.qa_id)
    model,processor=load_backbone(args.model_name,args.adapter_path,load_in_4bit=not args.full_precision); device=model_device(model); visual=qwen_visual(model)
    state=torch.load(args.checkpoint,map_location=device); k=state["prototypes"].shape[1]
    proto=PrototypeTransformer(hidden_size(visual),k).to(device=device,dtype=torch.bfloat16)
    proto.prototypes.data.copy_(state["prototypes"].to(device)); proto.alpha.data.copy_(state["alpha"].to(device)); proto.load_state_dict({**state["t_pro"],"prototypes":proto.prototypes.data,"alpha":proto.alpha.data},strict=False)
    hook=VisionPrototypeHook(visual,proto); cache=CachedVisionForward(visual,args.vision_cache_dir) if args.vision_cache_dir else None; image=image_for_question(question,args.mapwise_image_root,args.mapverse_image_root)
    row=make_trajectory_inputs(processor,question,question["trajectories"][0],image,device); grid=row["image_grid_thw"]
    labels=row.pop("labels"); grid=row["image_grid_thw"]; args.output_dir.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        cache_context=cache.use([vision_cache_key(question)]) if cache else nullcontext()
        with cache_context:
            with hook.context(grid,capture_attention=True): model(**row)
    attn=proto.last_attention[0].float().cpu().numpy() # H,N+K,N+K
    n=int(np.prod(grid[0].cpu().numpy())); t,h,w=grid[0].cpu().tolist(); original=plt.imread(image)
    # Qwen vision grids may have temporal slices; save each P/head and head mean.
    for pi in range(k):
        maps=list(attn[:, n+pi, :n])+[attn[:, n+pi, :n].mean(0)]
        for hi, values in enumerate(maps):
            heat=np.asarray(values).reshape(t,h,w).mean(0)
            plt.figure(figsize=(8,6)); plt.imshow(original); plt.imshow(heat,alpha=.48,cmap="magma",extent=(0,original.shape[1],original.shape[0],0)); plt.axis("off"); plt.title(f"P{pi} {'head_mean' if hi==len(maps)-1 else 'head_'+str(hi)}"); plt.savefig(args.output_dir/f"prototype_{pi}_{'mean' if hi==len(maps)-1 else 'head'+str(hi)}.png",dpi=180,bbox_inches="tight"); plt.close()
    pvec=torch.nn.functional.normalize(proto.prototypes[0].float(),dim=-1); cos=(pvec@pvec.T).cpu().numpy()
    pa=attn[:,n:n+k,:n].mean(0); pa=pa/(np.linalg.norm(pa,axis=1,keepdims=True)+1e-8); overlap=pa@pa.T
    np.save(args.output_dir/"prototype_cosine_similarity.npy",cos); np.save(args.output_dir/"prototype_attention_overlap.npy",overlap)
    (args.output_dir/"diagnostics.json").write_text(json.dumps({"residual_relative_norm":float(proto.last_residual_relative_norm.cpu()),"grid_thw":[int(t),int(h),int(w)]},indent=2))
    hook.remove()
    if cache: cache.remove()
if __name__=="__main__": main()
