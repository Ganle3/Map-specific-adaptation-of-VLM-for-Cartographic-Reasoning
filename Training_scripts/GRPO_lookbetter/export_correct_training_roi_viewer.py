#!/usr/bin/env python3
"""Export offline, per-question ROI players for correct GRPO training rollouts."""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path


from data import extract_final_answer, load_mapverse_questions


PLAYER = r"""<!doctype html><meta charset="utf-8"><title>Correct ROI trajectories</title>
<style>
body{margin:0;background:#f3f5f8;color:#172033;font:15px system-ui,sans-serif}.top{padding:22px max(24px,calc((100vw - 1450px)/2));background:#16243d;color:white}.top h1{font-size:20px;margin:0 0 9px}.top p{margin:5px 0;color:#d8e3f3}.layout{max-width:1450px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(500px,1.1fr) minmax(420px,.9fr);gap:18px}.card{background:#fff;border-radius:14px;padding:16px;box-shadow:0 2px 12px #0d1b2a12}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar select,button{padding:7px 10px;border:1px solid #bdc8d8;border-radius:7px;background:white}button{cursor:pointer;background:#16794a;color:#fff;border:0}canvas{width:100%;height:auto;display:block;border-radius:8px;background:#eee;margin-top:12px}.timeline{width:100%;margin:12px 0 4px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:12px}.stat{background:#f3f6fa;padding:8px;border-radius:7px}.reason{white-space:pre-wrap;line-height:1.55;max-height:580px;overflow:auto;background:#101827;color:#e4ecf8;padding:14px;border-radius:8px}.future{opacity:.32}.answer{border-left:4px solid #16803d;padding-left:9px;margin:9px 0}.note{font-size:12px;color:#556277}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:12px;color:#4d5b70}.swatch{width:12px;height:12px;border-radius:50%;display:inline-block}@media(max-width:940px){.layout{grid-template-columns:1fr}}
</style><body><section class="top"><h1 id="title"></h1><p id="question"></p><p id="gold"></p></section><main class="layout"><section class="card"><div class="toolbar"><label>Correct trajectory <select id="rollout"></select></label><button id="play">Play</button><label>Speed <select id="speed"><option value="800">Slow</option><option value="350" selected>Normal</option><option value="120">Fast</option></select></label></div><canvas id="map"></canvas><input class="timeline" id="step" type="range"><div id="stepLabel" class="note"></div><p class="legend"><span><i class="swatch" style="background:#94a3b8"></i> earlier LOOK ROI</span><span><i class="swatch" style="background:#f59e0b"></i> current ROI (+key-logit bias)</span></p><div id="stats" class="stats"></div></section><section class="card"><div id="result"></div><h3>Reasoning synchronized to LOOK token</h3><div id="reasoning" class="reason"></div><p class="note">The bright prefix is the assistant text already emitted when LOOK is selected; dim text is generated after the ROI-biased refresh. Legacy trajectories reconstruct the boundary with the Qwen tokenizer; new trajectories record it exactly.</p></section></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=x=>document.getElementById(x),sel=$('rollout'),slider=$('step'),canvas=$('map'),ctx=canvas.getContext('2d'),img=new Image();let current=0,timer=null;
D.runs.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`epoch ${r.epoch}, step ${r.step}, rollout ${r.rollout}; ${r.looks.length} LOOKs`;sel.append(o)});$('title').textContent=`Correct GRPO ROI trajectories — dataset index ${D.question.index}`;$('question').textContent=D.question.question;$('gold').textContent=`Gold answer: ${D.question.answer}`;
function run(){return D.runs[current]}function n(){return run().looks.length}function box(b,color,alpha,width){ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=Math.max(width,canvas.width/700);ctx.strokeRect(b.left,b.top,b.right-b.left,b.bottom-b.top);ctx.restore()}function prefix(text,token){let end=Math.floor(text.length*Math.min(1,token/Math.max(1,run().budget)));return[text.slice(0,end),text.slice(end)]}
function draw(){let r=run(),s=+slider.value,look=r.looks[s],parts=(look.prefix&&r.response.startsWith(look.prefix))?[look.prefix,r.response.slice(look.prefix.length)]:prefix(r.response,look.token);canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);r.looks.slice(0,s).forEach(x=>x.regions.forEach(b=>box(b,'#64748b',.45,2)));look.regions.forEach(b=>box(b,'#f59e0b',1,4));$('reasoning').replaceChildren(Object.assign(document.createElement('span'),{textContent:parts[0]}),Object.assign(document.createElement('span'),{className:'future',textContent:parts[1]}));$('result').innerHTML=`<div class=answer><b>Correct</b><br>Final: <code>${r.answer||'(missing)'}</code><br>Epoch / step: ${r.epoch} / ${r.step}<br>Natural EOS: ${r.eos?'yes':'no (budget)'}</div>`;$('stepLabel').textContent=`LOOK ${s+1} / ${n()} immediately before generating token ${look.token+1} (${look.prefix_source})`;$('stats').innerHTML=`<div class=stat>ROI regions<br><b>${look.regions.length}</b></div><div class=stat>Biased global visual tokens<br><b>${look.roi_tokens}</b></div><div class=stat>Key-logit bias<br><b>+${look.bias.toFixed(3)}</b></div>`}
function refresh(){if(!n())return;$('step').max=n()-1;$('step').value=0;draw()}sel.onchange=()=>{current=+sel.value;refresh()};slider.oninput=draw;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play';return}$('play').textContent='Pause';timer=setInterval(()=>{if(+slider.value>=+slider.max){clearInterval(timer);timer=null;$('play').textContent='Play';return}slider.value=+slider.value+1;draw()},+$('speed').value)};img.onload=refresh;img.src=D.image;
</script>"""


def _image_data(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--mapverse-image-root", type=Path, required=True)
    parser.add_argument("--mapwise-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-VL-8B-Thinking",
                        help="Tokenizer used to reconstruct exact-looking prefixes for legacy trajectories.")
    args = parser.parse_args()
    groups = json.loads(args.trajectories.read_text(encoding="utf-8"))
    records = load_mapverse_questions(args.dataset_json, args.mapverse_image_root, mapwise_root=args.mapwise_root)
    # Trajectories written before the explicit prefix field need a BPE-token
    # reconstruction.  This is substantially more faithful than the previous
    # character-fraction approximation.  New trajectories carry the exact
    # decoded boundary recorded by rollout.py and never use this fallback.
    needs_legacy_prefix = any(
        "visible_trace_before_look" not in event
        for group in groups for run in group["rollouts"] for event in run.get("look_events", [])
    )
    # The HTML-only exporter should work without a model environment for new
    # trajectories, which already record exact decoded LOOK boundaries.
    # Tokenizer loading is required only to reconstruct legacy records.
    if needs_legacy_prefix:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    else:
        tokenizer = None

    def look_prefix(run: dict, event: dict) -> tuple[str, str]:
        exact = event.get("visible_trace_before_look")
        if exact is not None:
            return exact, "recorded token boundary"
        ids = tokenizer.encode(run["response"], add_special_tokens=False)
        boundary = min(int(event["token"]), len(ids))
        return tokenizer.decode(ids[:boundary], skip_special_tokens=True), "tokenizer-reconstructed legacy boundary"

    selected: dict[int, list[dict]] = {}
    for group in groups:
        index = int(group["source_index"])
        for ordinal, run in enumerate(group["rollouts"]):
            if float(run["reward"]) != 1.0 or not run.get("look_events"):
                continue
            looks = []
            for event in run["look_events"]:
                prefix, prefix_source = look_prefix(run, event)
                looks.append({"token": int(event["token"]), "prefix": prefix,
                              "prefix_source": prefix_source, "regions": event["regions"],
                              "roi_tokens": int(event.get("roi_visual_tokens", 0)),
                              "bias": float(event.get("roi_key_bias", 0.0))})
            selected.setdefault(index, []).append({
                "epoch": int(group["epoch"]), "step": int(group["optimizer_step"]), "rollout": ordinal,
                "response": run["response"], "answer": extract_final_answer(run["response"]),
                "budget": int(run.get("generation_budget_tokens", 1)),
                "eos": bool(run.get("generation_terminated_by_eos", False)),
                "looks": looks,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for index, runs in sorted(selected.items()):
        item = records[index - 1]
        payload = {"question": {"index": index, "question": item.question, "answer": item.answer},
                   "image": _image_data(item.image_path), "runs": runs}
        name = f"question_{index:04d}_correct.html"
        page = PLAYER.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
        (args.output_dir / name).write_text(page, encoding="utf-8")
        links.append((name, item, len(runs)))
    lines = ["<!doctype html><meta charset=utf-8><style>body{font:16px system-ui;margin:32px;max-width:1100px}li{margin:12px 0}.meta{color:#566}</style>",
             "<h1>Correct GRPO ROI trajectory players</h1>", f"<p>{sum(n for _, _, n in links)} correct trajectories with at least one LOOK.</p><ul>"]
    for name, item, count in links:
        lines.append(f'<li><a href="{name}">Question {html.escape(item.sample_id)}: {html.escape(item.question)}</a> <span class="meta">— {count} correct trajectories</span></li>')
    lines.append("</ul>")
    (args.output_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {sum(n for _, _, n in links)} correct trajectories across {len(links)} questions to {args.output_dir}")


if __name__ == "__main__":
    main()
