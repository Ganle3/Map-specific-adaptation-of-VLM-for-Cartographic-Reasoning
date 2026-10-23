#!/usr/bin/env python3
"""Render deterministic nine-grid decisions saved in a rollout trace."""
from __future__ import annotations
import argparse, json, sys, textwrap
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "Training_scripts" / "GRPO_lookbetter") not in sys.path:
    sys.path.insert(0, str(ROOT / "Training_scripts" / "GRPO_lookbetter"))
from data import load_mapverse_questions
from region_policy import nine_grid_boxes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True); p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--mapverse-root",type=Path,required=True); p.add_argument("--mapwise-root",type=Path,required=True)
    p.add_argument("--start-index",type=int,default=245); p.add_argument("--end-index",type=int,default=248)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    last={}
    for line in a.trace.read_text(encoding="utf-8").splitlines():
        row=json.loads(line)
        if row.get("phase")=="eval" and a.start_index <= row.get("dataset_index",-1) <= a.end_index:
            last[row["dataset_index"]]=row
    if len(last) != a.end_index-a.start_index+1: raise ValueError("Trace lacks deterministic eval rows for selected range")
    records=load_mapverse_questions(a.dataset,a.mapverse_root,mapwise_root=a.mapwise_root,limit=a.end_index)
    fig, axes=plt.subplots(len(last),2,figsize=(22,6.5*len(last)),gridspec_kw={"width_ratios":[1.45,1]})
    if len(last)==1: axes=np.array([axes])
    cmap=plt.get_cmap("tab10")
    for out_row,(dataset_index,row) in zip(axes,sorted(last.items())):
        ax, heat=out_row; item=records[dataset_index-1]; events=row["diagnostics"]
        image=np.asarray(Image.open(item.image_path).convert("RGB")); h,w=image.shape[:2]
        ax.imshow(image); cells=defaultdict(list)
        for look,event in enumerate(events,1): cells[event["selected_action"]].append(look)
        for index,(x0,y0,x1,y1) in enumerate(nine_grid_boxes(w,h)):
            col,row_i=index%3,index//3
            ax.add_patch(patches.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="white",linewidth=2.2))
            looks=cells[index]
            label=f"{index}\n"+("L"+",".join(map(str,looks)) if looks else "")
            ax.text((x0+x1)/2,(y0+y1)/2,label,ha="center",va="center",fontsize=11,fontweight="bold",
                    color="white",bbox=dict(facecolor=(cmap(looks[0]-1) if looks else "black"),alpha=.72,edgecolor="none",pad=3))
        question=textwrap.fill(item.question,width=100)
        ax.set_title(f"row {dataset_index} | qa_id={item.sample_id} | eval reward={row['reward']}\n{question}",fontsize=11)
        ax.axis("off")
        probs=np.asarray([event["probabilities"] for event in events])
        im=heat.imshow(probs,aspect="auto",vmin=0,vmax=max(.20,float(probs.max())),cmap="viridis")
        for look,event in enumerate(events):
            chosen=event["selected_action"]
            heat.add_patch(patches.Rectangle((chosen-.5,look-.5),1,1,fill=False,edgecolor="red",linewidth=2))
            heat.text(chosen,look,"★",color="white",ha="center",va="center",fontsize=9)
        heat.set_xticks(range(9)); heat.set_xlabel("nine-grid action (row-major)")
        heat.set_yticks(range(len(events))); heat.set_yticklabels([f"L{i+1} @ {e['reasoning_token_index']}" for i,e in enumerate(events)])
        heat.set_title("Conditional action probabilities; red/star = greedy selection")
        fig.colorbar(im,ax=heat,fraction=.046,pad=.04,label="π(action | h_t)")
    fig.suptitle("Nine-grid region policy: last deterministic eval saved in overfit trace",fontsize=16,y=.995)
    fig.tight_layout(rect=(0,0,1,.99)); a.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(a.output,dpi=180,bbox_inches="tight")
    print(a.output)
if __name__=="__main__": main()
