#!/usr/bin/env python3
"""Visualize both placements of shared Relay2 tokens for selected questions."""
from __future__ import annotations

import argparse, gc, json, shutil, sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE=Path(__file__).resolve().parent; RELAY_ROOT=HERE.parent; VLM_ROOT=HERE.parents[2]; THESIS_ROOT=VLM_ROOT.parent
for path in (HERE, RELAY_ROOT, VLM_ROOT/"SparseVLM_eliminateCONTEXT"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

from continuous_cache_qwen3vl import _content_inputs, _resize_to_pixel_cap
from data import build_cartographic_prompt
from evaluate_core import load_evaluation_questions, load_policy
from Image_question_shared_relay.layout import build_prefill
from relay1_visualization.efficient_capture import EfficientRelayPrefillCapture
from relay_module import RelayModule
from render import render_question
from rolling_sparse_qwen3vl import load_model
from utils import decoder_hidden_size, encode_global_image, generate_from_relays, qwen_lm

@dataclass(frozen=True)
class Selected: dataset:str; index:int; item:object; folder:str

def args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--selection',type=Path,default=HERE/'questions.json')
    p.add_argument('--mapverse-dataset',type=Path,default=VLM_ROOT/'Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json')
    p.add_argument('--mapwise-dataset',type=Path,default=VLM_ROOT/'Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json')
    p.add_argument('--mapverse-root',type=Path,default=THESIS_ROOT/'MapVerse/data/imgs'); p.add_argument('--mapwise-root',type=Path,default=VLM_ROOT/'Datasets/mapwise-dataset')
    p.add_argument('--run-dir',type=Path,default=VLM_ROOT/'Training_outputs/Prototype/relay2/checkpoint'); p.add_argument('--checkpoint-step',type=int,default=2000)
    p.add_argument('--output',type=Path); p.add_argument('--model'); p.add_argument('--max-pixels',type=int,default=1_000_000); p.add_argument('--overwrite',action='store_true')
    p.add_argument('--no-load-in-4bit',action='store_false',dest='load_in_4bit'); p.set_defaults(load_in_4bit=True); return p.parse_args()

def main():
    a=args(); a.output=(a.output or VLM_ROOT/f'Training_outputs/Prototype/relay2_visualization_step{a.checkpoint_step}').resolve()
    checkpoint=a.run_dir/f'checkpoint-{a.checkpoint_step}.pt'; adapter=a.run_dir/f'checkpoint-{a.checkpoint_step}'
    if not checkpoint.is_file() or not (adapter/'adapter_config.json').is_file(): raise FileNotFoundError(f'Missing matched checkpoint/adapter: {checkpoint}, {adapter}')
    if a.output.exists() and a.overwrite: shutil.rmtree(a.output)
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    specs=json.loads(a.selection.read_text(encoding='utf-8')); pools={
      'mapverse':load_evaluation_questions(a.mapverse_dataset,a.mapverse_root,a.mapwise_root),
      'mapwise':load_evaluation_questions(a.mapwise_dataset,a.mapverse_root,a.mapwise_root)}
    rows=[Selected(s['dataset'],int(s['array_index_0based']),pools[s['dataset']][int(s['array_index_0based'])],f"{s['dataset']}_qa{int(s['array_index_0based']):03d}") for s in specs]
    model,processor=load_model(a.model,adapter,load_in_4bit=a.load_in_4bit); model.eval()
    for p in model.parameters():p.requires_grad_(False)
    policy=RelayModule(decoder_hidden_size(model),num_relays=2).to(model.get_input_embeddings().weight.device).eval()
    state=load_policy(policy,checkpoint,'shared_dual_placement','image_question_shared')
    if int(state.get('step',-1))!=a.checkpoint_step: raise ValueError('checkpoint step mismatch')
    pages=[]
    try:
      for row in rows:
        with Image.open(row.item.image_path) as h: original=h.convert('RGB'); submitted=_resize_to_pixel_cap(original,a.max_pixels)
        source=_content_inputs(processor,[submitted],build_cartographic_prompt(row.item.question),model.get_input_embeddings().weight.device,a.max_pixels)
        features=encode_global_image(model,source); augmented,_=build_prefill(model,processor.tokenizer,source,features,policy,with_grad=False)
        qwen=qwen_lm(model); image_pos=augmented['input_ids'][0].eq(int(qwen.config.image_token_id)).nonzero().flatten(); positions=augmented['relay_positions']
        t,h,w=map(int,augmented['image_grid_thw'][0].tolist()); merge=int(qwen.config.vision_config.spatial_merge_size); grid=(h//merge,w//merge)
        with EfficientRelayPrefillCapture(model,image_pos,positions.flatten(),grid) as capture:
            generate_from_relays(model,**augmented,do_sample=False,num_beams=1,max_new_tokens=1,use_cache=True)
        flat_maps=capture.arrays()
        maps=flat_maps.reshape(2,2,flat_maps.shape[1],*grid)
        destination=a.output/row.folder; destination.mkdir(); np.savez_compressed(destination/'attention.npz',relay_attention=maps)
        record={'checkpoint_step':a.checkpoint_step,'checkpoint':str(checkpoint.resolve()),'dataset':row.dataset,'array_index_0based':row.index,'qa_id':row.item.sample_id,'question':row.item.question,'ground_truth':row.item.answer,'image':str(row.item.image_path.resolve()),'relay_positions':positions.tolist(),'map_axes':['placement','relay_token','decoder_layer','patch_y','patch_x'],'placement_names':['after_vision_end','after_question'],'shared_parameter':True,'layers':int(maps.shape[2]),'merged_grid_hw':list(grid)}
        (destination/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8'); pages.append(render_question(destination)); print(row.folder,maps.shape,positions.tolist(),flush=True)
    finally:
      del policy,model,processor; gc.collect(); torch.cuda.empty_cache()
    links=''.join(f'<li><a href="{p.relative_to(a.output).as_posix()}">{p.parent.name}</a></li>' for p in pages)
    (a.output/'index.html').write_text('<meta charset="utf-8"><h1>Relay2 dual-placement attention</h1><ul>'+links+'</ul>',encoding='utf-8')
    print(a.output/'index.html')

if __name__=='__main__':main()
