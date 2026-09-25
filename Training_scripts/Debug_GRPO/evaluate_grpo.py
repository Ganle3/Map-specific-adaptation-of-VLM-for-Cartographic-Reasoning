"""Standard greedy evaluation for any completed joint GRPO run."""
import argparse
import csv
import gc
import hashlib
import json
import importlib.util
from pathlib import Path
import sys
import time


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--qa-json',type=Path,required=True,
                        help='QA dataset used by the training run.')
    parser.add_argument('--image-root',type=Path,required=True)
    parser.add_argument('--evaluation-script',type=Path,required=True)
    parser.add_argument('--max-new-tokens',type=int,default=1536)
    parser.add_argument('--checkpoint-steps',type=int,nargs='+',default=None,
                        help='Evaluate only these checkpoint step numbers.')
    parser.add_argument('--skip-baseline',action='store_true',
                        help='Evaluate only selected checkpoints, without the base model.')
    parser.add_argument('--baseline-only',action='store_true',
                        help='Evaluate only the untouched base model, without checkpoints.')
    parser.add_argument('--checkpoint-stride',type=int,default=None,
                        help='Evaluate checkpoints at this step interval.')
    parser.add_argument('--include-final',action='store_true',
                        help='Include the highest available checkpoint.')
    parser.add_argument('--sampling-draws',type=int,default=100,
                        help='Optional sampled diagnostic draws for selected checkpoints.')
    parser.add_argument('--skip-greedy',action='store_true',
                        help='Run only matched stochastic draws; useful for rollout-group diagnostics.')
    parser.add_argument('--output-subdir',default='evaluation_greedy',
                        help='Directory under run-dir for this evaluation manifest and CSVs.')
    parser.add_argument('--min-pixels',type=int,default=None)
    parser.add_argument('--max-pixels',type=int,default=None)
    args=parser.parse_args()
    if args.skip_baseline and args.baseline_only:
        parser.error('--skip-baseline and --baseline-only are mutually exclusive')
    if args.baseline_only and (args.checkpoint_steps is not None or args.checkpoint_stride is not None):
        parser.error('--baseline-only cannot be combined with checkpoint selection')
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'Evaluation_scripts/GRPO_ablation'))
    import torch
    from PIL import Image
    from transformers import set_seed
    import inference_mapwise_trl as inference
    from inference_mapwise_trl_e2 import install
    spec = importlib.util.spec_from_file_location('scorer', args.evaluation_script.expanduser().resolve())
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    from evaluate_single_qa_sampling_extended import paired_summary
    qa=args.qa_json.expanduser().resolve()
    # Applied after each processor load below; values are also recorded in the manifest.
    if not qa.is_file():raise FileNotFoundError(qa)
    samples=inference.load_json_list(qa)
    for index, sample in enumerate(samples):
        if 'correct_answer' in sample and 'ground_truth' not in sample:
            sample['ground_truth'] = sample['correct_answer']
            sample['ground_truth_type'] = sample.get('answer_type', '')
            sample['qa_id'] = str(sample.get('qa_id', f"mapverse_{sample.get('sample_id', index)}"))
    run=args.run_dir.resolve()
    state=json.loads((run/'trainer_state.json').read_text())
    if not (run/'final_adapter/adapter_config.json').is_file():
        raise ValueError('Require completed training with final adapter')
    steps=sorted(int(p.name.split('-')[1]) for p in run.glob('checkpoint-*')
                 if p.is_dir() and p.name.split('-')[1].isdigit())
    if args.baseline_only:
        steps=[]
    elif args.checkpoint_steps is not None:
        requested=set(args.checkpoint_steps)
        missing=requested-set(steps)
        if missing: raise FileNotFoundError(f'Missing checkpoints: {sorted(missing)}')
        steps=[step for step in steps if step in requested]
    elif args.checkpoint_stride is not None:
        if args.checkpoint_stride < 1: raise ValueError('--checkpoint-stride must be positive')
        selected=[step for step in steps if step % args.checkpoint_stride == 0]
        if args.include_final and steps and steps[-1] not in selected:
            selected.append(steps[-1])
        steps=sorted(set(selected))
    targets=[] if args.skip_baseline else [('baseline',None)]
    targets.extend((f'checkpoint-{step}',run/f'checkpoint-{step}') for step in steps)
    if not targets:
        raise ValueError('No evaluation targets selected')
    for _,cp in targets:
        if cp is not None and not (cp/'adapter_model.safetensors').is_file():
            raise FileNotFoundError(cp)
    out=run/args.output_subdir
    out.mkdir(exist_ok=True)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(protocol='standard_greedy_v1',qa_sha256=sha(qa),max_new_tokens=args.max_new_tokens,
        temperature=.8,top_p=.95,top_k=0,seeds_per_qa=args.sampling_draws,seed_start=900000,
        include_greedy=not args.skip_greedy,
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
            if args.max_pixels is not None: processor.image_processor.max_pixels=args.max_pixels
            if args.min_pixels is not None: processor.image_processor.min_pixels=args.min_pixels
            perqa=[]
            for qi,sample in enumerate(samples):
                path=inference.resolve_mapwise_image(sample,args.image_root)
                with Image.open(path) as im:
                    inputs=inference.prepare_multimodal_inputs(processor,im.convert('RGB'),
                        inference.build_mapwise_prompt(sample['question']),thinking_mode='auto')
                inputs=inference.move_inputs_to_model_device(inputs,model)
                draws=args.sampling_draws if label == 'baseline' or label == f'checkpoint-{steps[-1]}' else 0
                scores=[];greedy=None;truncated=0
                first_draw = 0 if args.skip_greedy else -1
                for index in range(first_draw,draws):
                    seed=900000+qi*1000+max(index,0)
                    set_seed(seed)
                    settings=dict(do_sample=index>=0,max_new_tokens=args.max_new_tokens,num_beams=1,num_return_sequences=1,
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
                        adapter_path=loaded,thinking_mode='auto',max_new_tokens=args.max_new_tokens,inference_seconds=time.perf_counter()-start)
                    record.update(decoding='greedy' if index<0 else 'sampling',sampling_seed=seed,
                                  terminated=terminated,generation_status='complete' if terminated else 'truncated')
                    if hasattr(scorer, 'evaluate_exact'):
                        scored = scorer.evaluate_exact(
                            prediction=raw, ground_truth=sample.get('correct_answer', sample.get('ground_truth', '')),
                            answer_type=sample.get('answer_type', sample.get('ground_truth_type', '')))
                        record['strict_exact_match'] = int(bool(scored.get('correct', scored.get('reward', 0))))
                    else:
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
            if cp and scores and baseline_scores:
                comparisons[str(r['source_index'])]=paired_summary(
                    baseline_scores[r['source_index']],scores)
        write(dest/'paired_vs_baseline.json',comparisons)
        with (out/'per_qa_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        overall=[]
        for ckpt in sorted({r['checkpoint'] for r in summary}):
            part=[r for r in summary if r['checkpoint']==ckpt]
            greedy_available=any(r['greedy_correct'] is not None for r in part)
            if greedy_available:
                correct=sum(bool(r['greedy_correct']) for r in part)
                total=len(part)
            else:
                correct=sum(int(r['sampled_correct']) for r in part if r['sampled_correct'] != '')
                total=sum(int(r['samples']) for r in part)
            overall.append(dict(checkpoint=ckpt,status='ok',total=total,correct=correct,
                                accuracy_percent=100*correct/total))
        with (out/'checkpoint_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(overall[0]));w.writeheader();w.writerows(overall)
    print('COMPLETE:',out/'per_qa_accuracy.csv',flush=True)


if __name__=='__main__':main()
