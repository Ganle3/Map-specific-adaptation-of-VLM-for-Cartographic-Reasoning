"""Fresh-rollout single-QA GRPO diagnostic; frozen Vision r16 baseline loop."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

QA_ID = 'mapwise_usa_map_15117_t3_src2051'


def validate_qa(rows):
    if len(rows) != 1 or rows[0]['country'] != 'usa' or rows[0]['source_index'] != 2051 or rows[0]['map_no'] != 'map_15117' or rows[0]['template_no'] != 3:
        raise ValueError('Expected exactly the QA selected by diagnostic 14037996')


def summarize(groups):
    blocks = []
    for start in range(0, len(groups), 5):
        window = groups[start:start+5]
        rewards = [r for g in window for r in g['rewards']]
        blocks.append(dict(first_step=window[0]['update'], last_step=window[-1]['update'],
            groups=len(window), correct=sum(rewards), responses=len(rewards),
            accuracy=sum(rewards)/len(rewards),
            mixed_groups=sum(len(set(g['rewards']))>1 for g in window),
            all_wrong_groups=sum(sum(g['rewards'])==0 for g in window),
            all_correct_groups=sum(sum(g['rewards'])==4 for g in window),
            truncated=sum(g['truncated'] for g in window)))
    return blocks


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--max-steps', type=int, default=40)
    extra, remaining = p.parse_known_args()
    import _vislora_grpo_baseline_snapshot as base
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    base.validate_args(args)
    if importlib.metadata.version('trl') != '1.12.0':
        raise RuntimeError('Requires TRL 1.12.0')
    if extra.max_steps < 5 or extra.max_steps % 5:
        raise ValueError('Use a positive multiple of 5 optimizer updates')
    if args.init_adapter_path or args.resume_from_checkpoint or int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise ValueError('Raw base, fresh adapter, one GPU; no resume or init adapter')
    validate_qa(json.loads(Path(args.qa_json).read_text(encoding='utf-8-sig')))
    args.lora_rank = args.lora_alpha = 16
    args.per_device_train_batch_size = 2
    args.gradient_accumulation_steps = 2
    args.num_generations = 4
    args.beta = 0
    args.warmup_fraction = 0
    args.preserve_qa_order = True
    args.save_steps = 5
    args.save_total_limit = extra.max_steps // 5 + 1
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'run_config.json').exists():
        raise ValueError('Use a fresh output directory')
    Config = base.GRPOConfig
    def config(**kw):
        kw.update(max_steps=extra.max_steps, num_iterations=1, warmup_steps=0,
                  warmup_ratio=0, lr_scheduler_type='constant')
        result = Config(**kw)
        (out/'effective_grpo_config.json').write_text(json.dumps(result.to_dict(), indent=2), encoding='utf-8')
        return result
    base.GRPOConfig = config
    save = base.save_run_config
    def save_config(output_dir, args, dataset_size, **unused):
        save(output_dir, args, dataset_size, extra.max_steps, 0)
        path = Path(output_dir)/'run_config.json'
        data = json.loads(path.read_text())
        data.update(diagnostic='single_qa_fresh_rollouts', qa_id=QA_ID, max_steps=extra.max_steps,
                    num_iterations=1, lr_scheduler_type='constant', replay=False,
                    versions={k: importlib.metadata.version(k) for k in ('torch','trl','peft','transformers')})
        path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    base.save_run_config = save_config
    Native = base.GRPOTrainer
    groups = []
    class SingleQA(Native):
        def _calculate_rewards(self, *a, **kw):
            result = super()._calculate_rewards(*a, **kw)
            if result.shape != (4, 1):
                raise RuntimeError('Expected exactly four responses and one reward')
            self.group_rewards = result[:, 0].detach().tolist()
            if any(r not in (0, 1) for r in self.group_rewards):
                raise ValueError('Invalid binary rewards')
            return result

        def _generate_and_score_completions(self, inputs):
            if len(inputs) != 4 or any(row['qa_id'] != QA_ID for row in inputs):
                raise RuntimeError('Unexpected QA or group size')
            if len(groups) != self.state.global_step:
                raise RuntimeError('Expected one fresh generation group per optimizer update')
            result = super()._generate_and_score_completions(inputs)
            group = dict(update=self.state.global_step+1, qa_id=QA_ID,
                rewards=self.group_rewards,
                advantages=result['advantages'].tolist(),
                truncated=int((result['completion_mask'].sum(1)==0).sum()))
            groups.append(group)
            with (out/'fresh_group_history.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(group)+'\n')
            (out/'sampling_windows.json').write_text(json.dumps(summarize(groups), indent=2), encoding='utf-8')
            return result
    base.GRPOTrainer = SingleQA
    base.run_training(args)
    if len(groups) != extra.max_steps:
        raise RuntimeError('Fresh generation count does not match update budget')
    path = out/'training_metrics.json'
    metrics = json.loads(path.read_text())
    metrics.update(expected_optimizer_steps=extra.max_steps, warmup_steps=0,
                   step_counts_are_estimates=False, fresh_groups=len(groups), fresh_responses=4*len(groups))
    path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
