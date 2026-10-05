#!/usr/bin/env python3
"""Summarize per-response inference_seconds by ability level and termination."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--evaluation-dir',type=Path,required=True); args=p.parse_args()
    rows=[]
    for path in args.evaluation_dir.rglob('responses.jsonl'):
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                r=json.loads(line)
                if r.get('inference_seconds') is not None: rows.append(r)
    groups=defaultdict(list)
    for r in rows:
        level=r.get('ability_level') or 'unknown'
        status=r.get('generation_status') or ('truncated' if not r.get('terminated',True) else 'complete')
        groups[(level,status)].append(float(r['inference_seconds']))
    out=args.evaluation_dir/'inference_time_by_ability_and_status.csv'
    with out.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['ability_level','generation_status','samples','average_inference_seconds'])
        for (level,status), values in sorted(groups.items()): w.writerow([level,status,len(values),sum(values)/len(values)])
    print(f'Wrote {out}')

if __name__=='__main__': main()
