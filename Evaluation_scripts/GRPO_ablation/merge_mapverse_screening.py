#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results-root',required=True,type=Path); p.add_argument('--output',required=True,type=Path); a=p.parse_args()
    files=sorted(a.results_root.glob('shard_*/mapverse_rollout_summary.csv'))
    if not files: raise FileNotFoundError(f'No shard summaries found under {a.results_root}')
    frames=[]
    for f in files:
        d=pd.read_csv(f); d['shard']=f.parent.name; frames.append(d)
    m=pd.concat(frames,ignore_index=True)
    key='source_row' if 'source_row' in m.columns else ('screening_id' if 'screening_id' in m.columns else None)
    if key and m.duplicated([key],keep=False).any(): raise ValueError('Duplicate questions found across shards')
    a.output.parent.mkdir(parents=True,exist_ok=True); m.to_csv(a.output,index=False)
    print(f'Questions merged: {len(m)}')
    if 'diagnostic_class' in m:
        c=m.diagnostic_class.value_counts(dropna=False); print(c.to_string()); print((c/len(m)).round(4).to_string())
    if {'answer_type','diagnostic_class'}.issubset(m.columns): print(pd.crosstab(m.answer_type,m.diagnostic_class,margins=True).to_string())
    mixed=m[m.diagnostic_class=='mixed'].copy() if 'diagnostic_class' in m else m.iloc[0:0].copy()
    mixed_out=a.output.with_name(a.output.stem+'_mixed_only.csv'); mixed.to_csv(mixed_out,index=False)
    print(f'Merged CSV: {a.output.resolve()}'); print(f'Mixed-only CSV: {mixed_out.resolve()}')

if __name__=='__main__': main()
