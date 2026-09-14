"""Four-QA E2 greedy trajectory and 100-draw milestone evaluations."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'Evaluation_scripts/GRPO_ablation'))
    import torch
    from PIL import Image
    from transformers import set_seed
    import inference_mapwise_trl as inference
    from inference_mapwise_trl_e2 import install
    import mapwise_evaluation_exact as scorer
    from validate_joint4 import validate_dataset
    from evaluate_single_qa_sampling_extended import paired_summary
    qa=repo/'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_joint_debug4_src2051.json'
    validate_dataset(qa)
    samples=inference.load_json_list(qa)
    run=args.run_dir.resolve()
    state=json.loads((run/'trainer_state.json').read_text())
    if state['global_step']!=120 or not (run/'final_adapter/adapter_config.json').is_file():
        raise ValueError('Require completed 120-update training with final adapter')
    targets=[('baseline',None),*[(f'checkpoint-{s}',run/f'checkpoint-{s}') for s in range(20,121,20)]]
    for _,cp in targets[1:]:
        if not (cp/'adapter_model.safetensors').is_file():raise FileNotFoundError(cp)
    out=run/'evaluation_e2_joint4'
    out.mkdir(exist_ok=True)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(protocol='joint4_E2_v1',qa_sha256=sha(qa),max_new_tokens=1536,
        temperature=.8,top_p=.95,top_k=0,seeds_per_qa=100,seed_start=900000,
        source_sha256=sha(Path(__file__)),
        dependency_hashes={name:sha(repo/'Evaluation_scripts/GRPO_ablation'/name) for name in
            ['inference_mapwise_trl.py','inference_mapwise_trl_e2.py','mapwise_evaluation_exact.py']},
        checkpoint_hashes={label:sha(cp/'adapter_model.safetensors') for label,cp in targets if cp})
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Evaluation manifest changed')
    write(mp,manifest)
    original_load=inference.load_model_and_processor
    summary=[]
    baseline_scores={}
    for label,cp in targets:
        dest=out/label;dest.mkdir(exist_ok=True)
        done=dest/'completed.json'
        if done.exists():
            result=json.loads(done.read_text())
        else:
            if (dest/'responses.jsonl').exists():
                (dest/'responses.jsonl').rename(dest/f'responses_incomplete_{time.time_ns()}.jsonl')
            inference.load_model_and_processor=original_load
            install(inference,dest)
            model,processor,loaded=inference.load_model_and_processor(adapter_path=cp)
            perqa=[]
            for qi,sample in enumerate(samples):
                path=inference.resolve_mapwise_image(sample,repo/'Datasets/mapwise-dataset')
                with Image.open(path) as im:
                    inputs=inference.prepare_multimodal_inputs(processor,im.convert('RGB'),
                        inference.build_mapwise_prompt(sample['question']),thinking_mode='auto')
                inputs=inference.move_inputs_to_model_device(inputs,model)
                draws=100 if label in ['baseline','checkpoint-40','checkpoint-80','checkpoint-120'] else 0
                scores=[];greedy=None;truncated=0
                for index in range(-1,draws):
                    seed=900000+qi*1000+max(index,0)
                    set_seed(seed)
                    settings=dict(do_sample=index>=0,max_new_tokens=1536,num_beams=1,num_return_sequences=1,
                                  repetition_penalty=1.,use_cache=True)
                    if index>=0:settings.update(temperature=.8,top_p=.95,top_k=0,min_p=None)
                    start=time.perf_counter()
                    with torch.inference_mode():generated=model.generate(**inputs,**settings)
                    tokens=generated[0,inputs['input_ids'].shape[1]:]
                    eos=model.generation_config.eos_token_id
                    eos=eos if isinstance(eos,list) else [eos]
                    terminated=bool(tokens.numel() and int(tokens[-1]) in eos)
                    raw=processor.batch_decode(tokens[None,:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
                    record=inference.build_prediction_record(sample=sample,sample_index=qi,image_path=path,
                        raw_response=raw,generated_tokens=int(tokens.numel()),model_name='Qwen/Qwen3-VL-8B-Thinking',
                        adapter_path=loaded,thinking_mode='auto',max_new_tokens=1536,inference_seconds=time.perf_counter()-start)
                    record.update(decoding='greedy' if index<0 else 'sampling',sampling_seed=seed,
                                  terminated=terminated,generation_status='complete' if terminated else 'truncated')
                    record=scorer.evaluate_sample(record)
                    with (dest/'responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if index<0:greedy=record['strict_exact_match']
                    else:
                        scores.append(record['strict_exact_match']);truncated+=not terminated
                    print(f'{label} src{sample["source_index"]} draw={index} correct={record["strict_exact_match"]}',flush=True)
                    del generated,tokens
                perqa.append(dict(source_index=sample['source_index'],country=sample['country'],question=sample['question'],
                                  greedy_correct=greedy,sampling_scores=scores,sampling_truncated=truncated))
                del inputs
            result=dict(checkpoint=label,per_qa=perqa)
            write(done,result)
            del model,processor
            gc.collect();torch.cuda.empty_cache()
        if label=='baseline':baseline_scores={r['source_index']:r['sampling_scores'] for r in result['per_qa']}
        comparisons={}
        for r in result['per_qa']:
            scores=r['sampling_scores']
            summary.append(dict(checkpoint=label,source_index=r['source_index'],greedy_correct=r['greedy_correct'],
                                samples=len(scores),sampled_correct=sum(scores) if scores else '',
                                sampled_accuracy_percent=100*sum(scores)/len(scores) if scores else '',
                                truncated=r['sampling_truncated']))
            if cp and scores:comparisons[str(r['source_index'])]=paired_summary(baseline_scores[r['source_index']],scores)
        write(dest/'paired_vs_baseline.json',comparisons)
        with (out/'per_qa_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('COMPLETE:',out/'per_qa_accuracy.csv',flush=True)


if __name__=='__main__':main()
