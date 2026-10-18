"""Export an offline HTML player for all-failed cold-start ROI trajectories.

The player needs no server: each question page embeds its source image and all
recorded LOOK events.  A frame is one successful ROI update; its boxes are
drawn on the original image and the accompanying reasoning panel is advanced
proportionally to the recorded reasoning-token boundary.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
from pathlib import Path

from data import extract_final_answer


PLAYER = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Cold-start ROI player</title>
<style>
:root{color-scheme:light}body{margin:0;background:#f3f5f8;color:#172033;font:15px system-ui,sans-serif}.top{padding:22px max(24px,calc((100vw - 1450px)/2));background:#16243d;color:white}.top h1{font-size:20px;margin:0 0 9px}.top p{margin:5px 0;color:#d8e3f3}.layout{max-width:1450px;margin:18px auto;padding:0 18px;display:grid;grid-template-columns:minmax(500px,1.1fr) minmax(420px,.9fr);gap:18px}.card{background:white;border-radius:14px;padding:16px;box-shadow:0 2px 12px #0d1b2a12}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar select,button{padding:7px 10px;border:1px solid #bdc8d8;border-radius:7px;background:white}button{cursor:pointer;background:#1e5fa8;color:white;border:0}canvas{width:100%;height:auto;display:block;border-radius:8px;background:#eee;margin-top:12px}.timeline{width:100%;margin:12px 0 4px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font-size:12px}.stat{background:#f3f6fa;padding:8px;border-radius:7px}.reason{white-space:pre-wrap;line-height:1.55;max-height:580px;overflow:auto;background:#101827;color:#e4ecf8;padding:14px;border-radius:8px}.future{opacity:.32}.answer{border-left:4px solid #b42318;padding-left:9px;margin:9px 0}.note{font-size:12px;color:#556277}.legend{display:flex;gap:13px;flex-wrap:wrap;font-size:12px;color:#4d5b70}.swatch{width:12px;height:12px;border-radius:50%;display:inline-block}@media(max-width:940px){.layout{grid-template-columns:1fr}}
</style><body><section class="top"><h1 id="title"></h1><p id="question"></p><p id="gold"></p></section><main class="layout"><section class="card"><div class="toolbar"><label>Rollout <select id="rollout"></select></label><button id="play">▶ Play</button><label>Speed <select id="speed"><option value="800">Slow</option><option value="350" selected>Normal</option><option value="120">Fast</option></select></label></div><canvas id="map"></canvas><input class="timeline" id="step" type="range"><div id="stepLabel" class="note"></div><p class="legend"><span><i class="swatch" style="background:#94a3b8"></i> earlier LOOK ROI</span><span><i class="swatch" style="background:#f59e0b"></i> current ROI (+2.079 key bias)</span></p><div id="stats" class="stats"></div></section><section class="card"><div id="result"></div><h3>Reasoning synchronized to LOOK token</h3><div id="reasoning" class="reason"></div><p class="note">The JSONL stores full decoded text rather than per-token strings; the highlighted text prefix is therefore a proportional display based on the recorded reasoning-token boundary.</p></section></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('payload').textContent),$=x=>document.getElementById(x),sel=$('rollout'),slider=$('step'),canvas=$('map'),ctx=canvas.getContext('2d'),img=new Image();let current=0,timer=null;
D.rollouts.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`#${i} — wrong; ${r.looks.length} LOOKs`;sel.append(o)});$('title').textContent=`All-wrong cold-start ROI viewer — dataset index ${D.question.dataset_index}`;$('question').textContent=D.question.question;$('gold').textContent=`Gold answer: ${D.question.answer}`;
function run(){return D.rollouts[current]}function n(){return run().looks.length}function box(b,color,alpha,width){ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=Math.max(width,canvas.width/700);ctx.strokeRect(b.left,b.top,b.right-b.left,b.bottom-b.top);ctx.restore()}
function prefix(text,token){let fraction=Math.min(1,token/Math.max(1,run().budget));let end=Math.floor(text.length*fraction);return [text.slice(0,end),text.slice(end)]}
function draw(){let r=run(),step=+slider.value,look=r.looks[step],parts=prefix(r.response,look.token);canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0,canvas.width,canvas.height);r.looks.slice(0,step).forEach(x=>x.regions.forEach(b=>box(b,'#64748b',.45,2)));look.regions.forEach(b=>box(b,'#f59e0b',1,4));$('reasoning').replaceChildren(Object.assign(document.createElement('span'),{textContent:parts[0]}),Object.assign(document.createElement('span'),{className:'future',textContent:parts[1]}));$('result').innerHTML=`<div class="answer"><b>Wrong</b><br>Parsed final: <code>${r.parsed_answer||'(missing)'}</code><br>Natural EOS: ${r.eos?'yes':'no (budget)'}<br>Elapsed: ${r.elapsed.toFixed(1)}s</div>`;$('stepLabel').textContent=`LOOK ${step+1} / ${n()} at reasoning token ${look.token}`;$('stats').innerHTML=`<div class=stat>ROI regions<br><b>${look.regions.length}</b></div><div class=stat>Global visual tokens biased<br><b>${look.roi_tokens}</b></div><div class=stat>Key-logit bias<br><b>+${look.bias.toFixed(3)}</b></div>`}
function refresh(){slider.max=Math.max(0,n()-1);slider.value=0;draw()}sel.onchange=()=>{current=+sel.value;refresh()};slider.oninput=draw;$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ Play';return}$('play').textContent='❚❚ Pause';timer=setInterval(()=>{if(+slider.value>=+slider.max){clearInterval(timer);timer=null;$('play').textContent='▶ Play';return}slider.value=+slider.value+1;draw()},+$('speed').value)};img.onload=refresh;img.src=D.image;
</script></body></html>"""


def _embedded_image(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _page_payload(row: dict) -> dict:
    runs = []
    for run in row["rollouts"]:
        looks = [{
            "token": int(event["token"]),
            "regions": event["regions"],
            "roi_tokens": int(event.get("roi_visual_tokens", 0)),
            "bias": float(event.get("roi_key_bias", 2.079442)),
        } for event in run.get("look_events", [])]
        runs.append({
            "response": run["response"], "parsed_answer": extract_final_answer(run["response"]),
            "looks": looks, "budget": int(run.get("generation_budget_tokens", 1)),
            "eos": bool(run.get("generation_terminated_by_eos", False)),
            "elapsed": float(run.get("elapsed_seconds", 0.0)),
        })
    return {"question": {"dataset_index": row["dataset_index"], "sample_id": row["sample_id"],
                         "question": row["question"], "answer": row["correct_answer"]},
            "image": _embedded_image(Path(row["image"])), "rollouts": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-all", action="store_true",
                        help="Export every completed question, including mixed-correctness groups, for diagnosis.")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = (rows if args.include_all else
              [row for row in rows if row["rollouts"] and all(float(run["reward"]) == 0.0 for run in row["rollouts"])])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for row in failed:
        name = f"question_{int(row['dataset_index']):04d}.html"
        page = PLAYER.replace("__PAYLOAD__", json.dumps(_page_payload(row), ensure_ascii=False).replace("</", "<\\/"))
        (args.output_dir / name).write_text(page, encoding="utf-8")
        links.append((name, row))
    index = ["<!doctype html><meta charset=utf-8><style>body{font:16px system-ui;margin:32px;max-width:1100px}li{margin:12px 0}.meta{color:#566}</style>",
             "<h1>Cold-start ROI viewers</h1>",
             (f"<p>Exported all {len(failed)} completed questions.</p><ul>" if args.include_all else
              f"<p>{len(failed)} / {len(rows)} completed questions had zero correct rollouts.</p><ul>")]
    for name, row in links:
        rewards = [run["reward"] for run in row["rollouts"]]
        index.append(f'<li><a href="{name}">Question {row["dataset_index"]}: {html.escape(row["question"])}</a> '
                     f'<span class="meta">— rewards {rewards}</span></li>')
    index.append("</ul>")
    (args.output_dir / "index.html").write_text("\n".join(index), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({"source": str(args.input_jsonl), "completed_questions": len(rows),
        "all_wrong_questions": len(failed), "pages": [name for name, _ in links]}, indent=2), encoding="utf-8")
    print(f"Exported {len(failed)} all-wrong question viewers to {args.output_dir}")


if __name__ == "__main__":
    main()
