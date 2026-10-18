#!/usr/bin/env python3
"""Create local, token-synchronised ROI viewers from a completed smoke JSONL.

The source JSONL can be multi-GB because it contains frozen hidden features.
This exporter streams it and retains only the visualisation fields, never
loading the entire experiment into memory.
"""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = ROOT / "VLM_adaptation" / "Training_outputs" / "GRPO_ROI_thinking_smoke_100"
DEFAULT_DATASET = ROOT / "VLM_adaptation" / "Datasets" / "When_Where_Crop" / "visual_hard_1000.json"
DEFAULT_MAPVERSE = ROOT / "MapVerse" / "data" / "imgs"
DEFAULT_MAPWISE = ROOT / "VLM_adaptation" / "Datasets" / "mapwise-dataset"
DEFAULT_TOKENIZER = ROOT / "VLM_adaptation" / "Training_outputs" / "MapVerse_scaling1000_GRPO_from_MapVerse5000_token500" / "checkpoint-2500"


def image_path(record: dict, mapverse: Path, mapwise: Path) -> Path:
    name = Path(record["image_name"])
    if record["dataset"].lower() == "mapverse":
        return mapverse / name
    country, stem = name.parts
    base = mapwise / country / "images" / str(record.get("map_type") or "with_annotations") / stem
    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = base.with_suffix(extension)
        if candidate.is_file(): return candidate
    raise FileNotFoundError(base)


def encoded_image(path: Path) -> str:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((1400, 1400))
        buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def compact_action(action: dict) -> dict:
    return {key: action.get(key) for key in ("geometry", "prior_geometry", "strengths", "bias_mean", "bias_max", "attention_summary_after")}


def token_pieces(rows: list[dict], tokenizer_path: Path | None) -> None:
    """Attach one decoded text piece per generated token, if tokenizer exists."""
    tokenizer = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as error:
            print(f"Tokenizer unavailable; reasoning sync will use character progress: {error}")
    for row in rows:
        if tokenizer:
            row["token_pieces"] = [tokenizer.decode([value], skip_special_tokens=True) for value in row["tokens"]]
        else:
            row["token_pieces"] = None


def slim(row: dict) -> dict:
    return {key: row.get(key) for key in ("question_index", "rollout_index", "sample_id", "dataset", "answer_type", "ability_level", "question", "ground_truth", "reward", "text", "tokens", "parsed_final_answer", "evaluation", "ended_with_eos") } | {
        "actions": [compact_action(action) for action in row.get("actions", [])]
    }


def page_payload(question: dict, rows: list[dict], image: str) -> dict:
    token_pieces(rows, question.pop("_tokenizer", None))
    return {"question": question, "image": image, "rollouts": rows}


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI reasoning player</title>
<style>
:root{color-scheme:light}body{margin:0;background:#f3f5f8;color:#172033;font:15px system-ui,sans-serif}.top{padding:22px max(24px,calc((100vw - 1450px)/2));background:#16243d;color:white}.top h1{font-size:20px;margin:0 0 10px}.top p{margin:5px 0;color:#d8e3f3}.layout{max-width:1450px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(500px,1.1fr) minmax(420px,.9fr);gap:18px}.card{background:white;border-radius:14px;padding:16px;box-shadow:0 2px 12px #0d1b2a12}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar select,button{padding:7px 10px;border:1px solid #bdc8d8;border-radius:7px;background:white}button{cursor:pointer;background:#1e5fa8;color:white;border:0}canvas{width:100%;height:auto;display:block;border-radius:8px;background:#eee;margin-top:12px}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:12px;color:#4d5b70}.swatch{width:12px;height:12px;border-radius:50%;display:inline-block}.timeline{width:100%;margin:12px 0 4px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:12px}.stat{background:#f3f6fa;padding:8px;border-radius:7px}.reason{white-space:pre-wrap;line-height:1.55;max-height:530px;overflow:auto;background:#101827;color:#e4ecf8;padding:14px;border-radius:8px}.future{opacity:.32}.answer{border-left:4px solid #3b82f6;padding-left:9px;margin:9px 0}.good{color:#087443;font-weight:700}.bad{color:#b42318;font-weight:700}.note{font-size:12px;color:#556277}.hidden{display:none}@media(max-width:940px){.layout{grid-template-columns:1fr}}
</style><body><section class="top"><h1 id="title"></h1><p id="meta"></p><p id="question"></p><p id="gold"></p></section><main class="layout"><section class="card"><div class="toolbar"><label>Rollout <select id="rollout"></select></label><button id="play">▶ Play</button><label>Speed <select id="speed"><option value="600">Slow</option><option value="250" selected>Normal</option><option value="80">Fast</option></select></label><label><input type="checkbox" id="prior"> Show attention prior</label></div><canvas id="map"></canvas><input class="timeline" id="step" type="range"><div id="stepLabel" class="note"></div><p class="legend"><span><i class="swatch" style="background:#fbbf24"></i> ROI box; opacity = strength</span><span><i class="swatch" style="background:#22c55e"></i> attention-guided prior</span><span>Each frame is an ROI action for the next decoding step.</span></p><div id="stats" class="stats"></div></section><section class="card"><div id="result"></div><h3>Reasoning synchronized to ROI token step</h3><div id="reasoning" class="reason"></div><p class="note">The highlighted prefix is exact token decoding when the tokenizer was available during export; otherwise it is a proportional character fallback.</p></section></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=x=>document.getElementById(x),sel=$('rollout'),slider=$('step'),canvas=$('map'),ctx=canvas.getContext('2d'),img=new Image();let current=0,timer=null;
D.rollouts.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`#${i+1} — ${r.reward?'correct':'wrong'}`;sel.append(o)});$('title').textContent=`Mixed-correctness ROI viewer — question ${D.question.question_index+1}`;$('meta').textContent=`sample ${D.question.sample_id} · ${D.question.dataset} · ${D.question.answer_type} · rewards: [${D.rollouts.map(x=>x.reward).join(', ')}]`;$('question').textContent=D.question.question;$('gold').textContent=`Gold answer: ${D.question.ground_truth}`;
function r(){return D.rollouts[current]}function n(){return r().actions.length}function textPrefix(run,step){if(run.token_pieces){return run.token_pieces.slice(0,Math.min(step+1,run.token_pieces.length)).join('')}let cut=Math.floor(run.text.length*Math.min(1,(step+1)/Math.max(1,n())));return run.text.slice(0,cut)}
function box(g,color,alpha,dash){let [cx,cy,w,h]=g;ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=Math.max(2,canvas.width/600);ctx.setLineDash(dash?[7,5]:[]);ctx.strokeRect((cx-w/2)*canvas.width,(cy-h/2)*canvas.height,w*canvas.width,h*canvas.height);ctx.restore()}
function draw(){let run=r(),step=+slider.value,action=run.actions[step],geo=action.geometry[0]||action.geometry,prior=action.prior_geometry;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0,canvas.width,canvas.height);if($('prior').checked&&prior)prior.forEach(g=>box(g,'#22c55e',.65,true));geo.forEach(g=>box(g,'#f59e0b',.15+.8*Math.min(1,g[4]/4),false));let prefix=textPrefix(run,step),remaining=run.text.slice(prefix.length);$('reasoning').replaceChildren(Object.assign(document.createElement('span'),{textContent:prefix}),Object.assign(document.createElement('span'),{className:'future',textContent:remaining}));let ev=run.evaluation||{},ok=run.reward?'good':'bad';$('result').innerHTML=`<div class="answer"><b class="${ok}">${run.reward?'Correct':'Wrong'}</b><br>Final: <code>${run.parsed_final_answer||'(missing)'}</code><br>Evaluator: ${ev.evaluator||'not recorded'} · ${ev.metric||''}<br>Canonical: ${ev.canonical_prediction||''} / ${ev.canonical_target||''}</div>`;$('stepLabel').textContent=`ROI action ${step+1} / ${n()} (token-aligned decoding step)`;let s=geo.map(x=>x[4]);$('stats').innerHTML=`<div class=stat>Bias max<br><b>${(+action.bias_max).toFixed(2)}</b></div><div class=stat>Mean strength<br><b>${(s.reduce((a,b)=>a+b,0)/s.length).toFixed(2)}</b></div><div class=stat>Max strength<br><b>${Math.max(...s).toFixed(2)}</b></div>`}
sel.onchange=()=>{current=+sel.value;slider.max=Math.max(0,n()-1);slider.value=0;draw()};slider.oninput=draw;$('prior').onchange=draw;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ Play';return}$('play').textContent='❚❚ Pause';timer=setInterval(()=>{if(+slider.value>=+slider.max){clearInterval(timer);timer=null;$('play').textContent='▶ Play';return}slider.value=+slider.value+1;draw()},+$('speed').value)};img.onload=()=>{slider.max=Math.max(0,n()-1);slider.value=0;draw()};img.src=D.image;
</script></body></html>'''


def resolve_record(question: dict, datasets: list[list[dict]]) -> dict:
    """Recover source image metadata even when smoke data was a reordered subset."""
    # Prefer a source file whose own index agrees with the saved question
    # metadata. This disambiguates repeated templates such as "Which class..."
    # that occur on many unrelated maps.
    index = question["question_index"]
    for dataset in datasets:
        if index >= len(dataset): continue
        row = dataset[index]
        if (row.get("dataset") == question["dataset"] and row.get("question") == question["question"]
                and str(row.get("correct_answer")) == str(question["ground_truth"])):
            return row
    matches = []
    for dataset in datasets:
        matches.extend(row for row in dataset if row.get("dataset") == question["dataset"]
                       and row.get("question") == question["question"]
                       and str(row.get("correct_answer")) == str(question["ground_truth"]))
    if not matches:
        raise LookupError(f"No source dataset record matches question {question['question_index'] + 1}.")
    images = {str(row["image_name"]) for row in matches}
    if len(images) != 1:
        raise LookupError(f"Ambiguous source images for question {question['question_index'] + 1}: {sorted(images)}")
    return matches[0]


def write_page(directory: Path, question: dict, rows: list[dict], datasets: list[list[dict]], mapverse: Path, mapwise: Path, tokenizer: Path | None) -> str:
    record = resolve_record(question, datasets)
    question["_tokenizer"] = tokenizer
    payload = page_payload(question, rows, encoded_image(image_path(record, mapverse, mapwise)))
    filename = f"question_{question['question_index'] + 1:03d}_{question['sample_id']}.html"
    (directory / filename).write_text(HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    return filename


def mixed_indices_from_readable(path: Path) -> set[int]:
    """Use the small Markdown summary to avoid JSON-decoding non-mixed rows."""
    selected, current, rewards = set(), None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        question = re.match(r"## Question (\d+):", line)
        if question:
            if current is not None and len(set(rewards)) > 1: selected.add(current)
            current, rewards = int(question.group(1)) - 1, []
        reward = re.match(r"### Rollout \d+ .* reward: ([01])", line)
        if reward: rewards.append(int(reward.group(1)))
    if current is not None and len(set(rewards)) > 1: selected.add(current)
    return selected


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,default=DEFAULT_RUN);p.add_argument("--dataset",type=Path,default=DEFAULT_DATASET);p.add_argument("--mapverse",type=Path,default=DEFAULT_MAPVERSE);p.add_argument("--mapwise",type=Path,default=DEFAULT_MAPWISE);p.add_argument("--tokenizer",type=Path,default=DEFAULT_TOKENIZER);p.add_argument("--output",type=Path,default=None);args=p.parse_args()
    output=args.output or args.run / "mixed_roi_viewer";output.mkdir(parents=True,exist_ok=True)
    dataset=json.loads(args.dataset.read_text(encoding="utf-8"))
    # A smoke run may use a reordered easy/hard subset. Load sibling manifests
    # solely as metadata lookup tables; never use question_index as image index.
    datasets=[dataset]+[json.loads(path.read_text(encoding="utf-8")) for path in args.dataset.parent.glob("*.json") if path != args.dataset]
    groups=defaultdict(list)
    readable = args.run / "readable_rollouts.md"
    selected = mixed_indices_from_readable(readable) if readable.is_file() else None
    if selected is not None: print(f"Markdown prefilter selected {len(selected)} mixed question groups", flush=True)
    with (args.run/"rollouts.jsonl").open(encoding="utf-8") as source:
        for line_number,line in enumerate(source,1):
            if not line.strip():continue
            match = re.search(r'"question_index"\s*:\s*(\d+)', line[:256])
            if not match: raise ValueError(f"Missing question_index in line {line_number}")
            if selected is not None and int(match.group(1)) not in selected: continue
            row=json.loads(line); groups[row["question_index"]].append(slim(row))
            if line_number%20==0:print(f"Read {line_number} rollout rows",flush=True)
    links=[]
    for index,rows in sorted(groups.items()):
        if len({row["reward"] for row in rows})<2:continue
        first=rows[0];question={key:first.get(key) for key in ("question_index","sample_id","dataset","answer_type","ability_level","question","ground_truth")}
        filename=write_page(output,question,rows,datasets,args.mapverse,args.mapwise,args.tokenizer if args.tokenizer.exists() else None)
        links.append((question,rows,filename));print(f"Wrote {filename}",flush=True)
    index_html="<h1>Mixed-correctness ROI viewers</h1><p>Each page has four rollouts, token-level ROI playback, and synchronized reasoning.</p><ul>"+"".join(f'<li><a href="{name}">Question {q["question_index"]+1}: {html.escape(q["question"])}</a> — rewards {[r["reward"] for r in rows]}</li>' for q,rows,name in links)+"</ul>"
    (output/"index.html").write_text("<!doctype html><meta charset=utf-8><style>body{font:16px system-ui;margin:32px;max-width:1000px}li{margin:12px 0}</style>"+index_html,encoding="utf-8")
    (output/"manifest.json").write_text(json.dumps({"source":str(args.run),"mixed_questions":len(links),"pages":[x[2] for x in links]},indent=2),encoding="utf-8")
    print(f"Created {len(links)} mixed-question viewers: {output/'index.html'}")
if __name__=="__main__":main()
