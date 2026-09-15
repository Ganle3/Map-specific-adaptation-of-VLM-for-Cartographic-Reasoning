"""Fixed baseline/checkpoint comparison: independent stochastic generations per seed."""
import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time


def wilson(correct, total):
    if total < 1:
        raise ValueError('No observations')
    z = 1.959963984540054
    p = correct / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [max(0., center-half), min(1., center+half)]


def paired_summary(before, after):
    if len(before) != len(after) or not before:
        raise ValueError('Equal nonempty seed lists required')
    diffs = [b-a for a,b in zip(before, after)]
    rng = random.Random(1729)
    boot = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(10000))
    return dict(accuracy_difference=sum(diffs)/len(diffs),
                paired_bootstrap_95_percentile=[boot[249],boot[9749]],
                wrong_to_correct=sum(a==0 and b==1 for a,b in zip(before,after)),
                correct_to_wrong=sum(a==1 and b==0 for a,b in zip(before,after)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--samples', type=int, default=100)
    p.add_argument('--seed-start', type=int, default=900000)
    args = p.parse_args()
    if args.samples < 2 or args.seed_start < 0:
        p.error('Require samples>=2 and nonnegative seed start')
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo/'Evaluation_scripts/GRPO_ablation'))
    import inference_mapwise_trl as inference
    import mapwise_evaluation_exact as scorer
    from train_mapwise_grpo_single_qa import validate_qa
    from PIL import Image
    import torch
    from transformers import set_seed
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint/'adapter_config.json').is_file():
        raise FileNotFoundError(checkpoint)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory; partial runs are not silently reused')
    out.mkdir(parents=True, exist_ok=True)
    qa = repo/'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_debug_single_src2051.json'
    validate_qa(json.loads(qa.read_text()))
    sample = inference.load_json_list(qa)[0]
    image_path = inference.resolve_mapwise_image(sample, repo/'Datasets/mapwise-dataset')
    seeds = list(range(args.seed_start,args.seed_start+args.samples))
    settings = dict(do_sample=True, temperature=.8, top_p=.95, top_k=0,
                    min_p=None, repetition_penalty=1., max_new_tokens=1536,
                    num_beams=1, num_return_sequences=1, use_cache=True)
    def write(name, value):
        (out/name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    write('manifest.json', dict(checkpoint=str(checkpoint), samples=args.samples, seeds=seeds,
        model='Qwen/Qwen3-VL-8B-Thinking', thinking='auto', generation=settings,
        qa_sha256=hashlib.sha256(qa.read_bytes()).hexdigest(),
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256(Path(scorer.__file__).read_bytes()).hexdigest(),
        versions={k: importlib.metadata.version(k) for k in ('torch','transformers','peft')},
        note='Fixed checkpoints, no training. Same seeds couple random draws, not reasoning paths.'))
    summaries, outcomes = [], {}
    for label, adapter in [('baseline',None), ('checkpoint-40',checkpoint)]:
        model, processor, loaded = inference.load_model_and_processor(adapter_path=adapter)
        write(label+'_generation_config.json', model.generation_config.to_dict())
        with Image.open(image_path) as im:
            inputs = inference.prepare_multimodal_inputs(processor, im.convert('RGB'),
                         inference.build_mapwise_prompt(sample['question']), thinking_mode='auto')
        inputs = inference.move_inputs_to_model_device(inputs, model)
        prompt_length = inputs['input_ids'].shape[1]
        records = []
        for index, seed in enumerate(seeds):
            set_seed(seed)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, **settings)
            tokens = generated[0,prompt_length:]
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos,list) else [eos]
            terminated = bool(tokens.numel() and int(tokens[-1]) in eos)
            raw = processor.batch_decode(tokens[None,:], skip_special_tokens=True,
                         clean_up_tokenization_spaces=False)[0].strip()
            record = inference.build_prediction_record(sample=sample, sample_index=0,
                image_path=image_path, raw_response=raw, generated_tokens=int(tokens.numel()),
                model_name='Qwen/Qwen3-VL-8B-Thinking', adapter_path=loaded, thinking_mode='auto',
                max_new_tokens=1536, inference_seconds=time.perf_counter()-started)
            record.update(sample_index=index, sampling_seed=seed, terminated=terminated,
                          generation_status='complete' if terminated else 'truncated')
            record = scorer.evaluate_sample(record)
            record['terminated_correct'] = int(terminated and record['strict_exact_match']==1)
            records.append(record)
            with (out/(label+'_samples.jsonl')).open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(f"{label} {index+1}/{args.samples}: correct={record['strict_exact_match']} "
                  f"terminated={terminated} answer={record['final_answer']!r}",flush=True)
            del generated, tokens
        correct = sum(r['strict_exact_match'] for r in records)
        outcomes[label] = [r['strict_exact_match'] for r in records]
        summaries.append(dict(model=label,total=len(records),correct=correct,accuracy=correct/len(records),
            wilson_95_low=wilson(correct,len(records))[0],wilson_95_high=wilson(correct,len(records))[1],
            terminated_correct=sum(r['terminated_correct'] for r in records),
            truncated=sum(not r['terminated'] for r in records),
            mean_tokens=sum(r['generated_tokens'] for r in records)/len(records)))
        with (out/'sampling_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        del model,processor,inputs
        gc.collect();torch.cuda.empty_cache()
    write('comparison.json',dict(status='complete',models=summaries,
        comparison=paired_summary(outcomes['baseline'],outcomes['checkpoint-40']),
        interpretation='Intervals describe sampling uncertainty on this one QA, not training-seed or dataset generalization uncertainty.'))
    print('COMPLETE:',out/'comparison.json',flush=True)


if __name__ == '__main__':
    main()
