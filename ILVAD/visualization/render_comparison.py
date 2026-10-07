"""Render ILVAD-vs-baseline attention diagnostics from run_single.py artifacts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); args=p.parse_args(); run=args.run
    base=np.load(run/"baseline_attention.npz")["attention"]  # step, layer, head, patch
    ilvad=np.load(run/"ilvad_attention.npz")["attention"]
    fixed_map = run / "enhanced_map.npy"
    rolling_maps = run / "ilvad_rolling_maps.npz"
    if rolling_maps.is_file():
        rolling = np.load(rolling_maps)
        maps, refresh_steps = rolling["enhanced_maps"], rolling["refresh_steps"]
        saliency = maps[-1]
        map_caption = f"Final rolling map ({len(maps)} replacements; last at step {refresh_steps[-1]})"
    elif fixed_map.is_file():
        saliency = np.load(fixed_map)
        map_caption = "Fixed ILVAD enhanced-map multiplier"
    else:
        raise FileNotFoundError("Neither enhanced_map.npy nor ilvad_rolling_maps.npz exists.")
    n=min(len(base),len(ilvad)); base,ilvad=base[:n].mean(2),ilvad[:n].mean(2)
    # Mean across selected time/layer dimensions: visual evidence receives more mass if increased.
    fig,ax=plt.subplots(1,3,figsize=(15,4));
    ax[0].plot(base.mean((1,2)),label="baseline"); ax[0].plot(ilvad.mean((1,2)),label="ILVAD"); ax[0].set(title="Mean image attention per decoding step",xlabel="step",ylabel="attention mass / patch"); ax[0].legend()
    ax[1].plot(base.mean(2).mean(0),label="baseline"); ax[1].plot(ilvad.mean(2).mean(0),label="ILVAD"); ax[1].set(title="Mean image attention per decoder layer",xlabel="layer index",ylabel="attention mass / patch"); ax[1].legend()
    ax[2].plot(saliency); ax[2].set(title=map_caption,xlabel="image token",ylabel="multiplier")
    fig.tight_layout(); fig.savefig(run/"attention_comparison.png",dpi=180); plt.close(fig)
    early=max(1,n//3); late=slice(max(0,n-early),n)
    metrics={"steps_compared":int(n),"baseline_image_attention_early":float(base[:early].mean()),"baseline_image_attention_late":float(base[late].mean()),"ilvad_image_attention_early":float(ilvad[:early].mean()),"ilvad_image_attention_late":float(ilvad[late].mean()),"baseline_salient_attention":float((base*saliency).sum()/saliency.sum()/base.shape[0]/base.shape[1]),"ilvad_salient_attention":float((ilvad*saliency).sum()/saliency.sum()/ilvad.shape[0]/ilvad.shape[1])}
    metrics["late_attention_gain"] = metrics["ilvad_image_attention_late"]-metrics["baseline_image_attention_late"]
    if rolling_maps.is_file():
        metrics["rolling_map_replacements"] = int(len(maps))
        metrics["rolling_refresh_steps"] = refresh_steps.tolist()
    (run/"attention_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (run/"analysis.md").write_text("# ILVAD attention diagnostic\n\n"+json.dumps(metrics,indent=2)+"\n\nA positive `late_attention_gain` and higher `ilvad_salient_attention` are evidence consistent with reduced visual forgetting. They do **not** establish reduced hallucination alone; assess the two generated answers against the map.\n",encoding="utf-8")
if __name__=="__main__": main()
