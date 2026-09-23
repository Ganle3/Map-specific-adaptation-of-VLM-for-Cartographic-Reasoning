#!/usr/bin/env python3
"""Aggregate MapWise benchmark runs by model, token budget, level and answer type."""
import argparse, csv, json
from pathlib import Path
from analyze_binary_classification import metrics as binary_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',type=Path,required=True)
    p.add_argument('--qa-json',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args(); root=a.benchmark_root.resolve(); out=(a.output_dir or root/'analysis').resolve();out.mkdir(parents=True,exist_ok=True)
    qa=json.loads(a.qa_json.read_text(encoding='utf-8-sig'))
    summaries=[]; groups=[]
    for path in sorted(root.glob('*/tokens*/evaluation/evaluation_details.json')):
        rows=json.loads(path.read_text(encoding='utf-8'))
        model=path.parents[2].name; tokens=int(path.parents[1].name.removeprefix('tokens'))
        for r in rows:
            meta=qa[int(r['sample_index'])]
            r['ability_level']=meta['ability_level']
        correct=sum(int(r.get('strict_exact_match',0)) for r in rows)
        binary=[r for r in rows if r.get('ground_truth_type')=='Binary']
        summaries.append(dict(model=model,max_new_tokens=tokens,total=len(rows),correct=correct,
                              accuracy_percent=100*correct/len(rows),**binary_metrics(binary)))
        for field,theme in [('ability_level','ability_level'),('ground_truth_type','answer_type')]:
            for value in sorted({str(r[field]) for r in rows}):
                part=[r for r in rows if str(r[field])==value]; c=sum(int(r.get('strict_exact_match',0)) for r in part)
                groups.append(dict(model=model,max_new_tokens=tokens,theme=theme,category=value,
                                   total=len(part),correct=c,accuracy_percent=100*c/len(part)))
    if not summaries: raise FileNotFoundError(f'No evaluation_details.json under {root}')
    for name,data in [('benchmark_summary.csv',summaries),('grouped_accuracy.csv',groups)]:
        with (out/name).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(out)
if __name__=='__main__': main()
