"""One native TRL update on four fixed, mixed-reward, terminated trajectories.

Not an overfitting run. No replay/SFT; uses the frozen baseline model/reward.
Requires TRL 1.12.0. Outputs probabilities at the training temperature, not
top-p sampling probabilities. A reload uses a second adapter on the same base.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys


def clipped_objective(torch, logps, old, advantages, mask, max_length, low, high):
    ratio = (logps - old).exp()
    term = torch.minimum(ratio * advantages[:, None],
                         ratio.clamp(1-low, 1+high) * advantages[:, None])
    return (term * mask).sum() / (logps.shape[0] * max_length)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate-attempts', type=int, default=12)
    parser.add_argument('--candidate-start', type=int, default=0)
    extra, remaining = parser.parse_known_args()
    import _vislora_grpo_baseline_snapshot as base
    import torch
    from trl.models.utils import disable_gradient_checkpointing
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    base.validate_args(args)
    if importlib.metadata.version('trl') != '1.12.0':
        raise RuntimeError('Requires TRL 1.12.0')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1 or args.init_adapter_path or args.resume_from_checkpoint:
        raise ValueError('Fresh base model, one GPU only')
    if extra.candidate_attempts < 1 or extra.candidate_start < 0:
        raise ValueError('Invalid candidate search bounds')
    args.per_device_train_batch_size = 2
    args.gradient_accumulation_steps = 2
    args.num_generations = 4
    args.warmup_fraction = 0
    args.beta = 0
    args.preserve_qa_order = True
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'run_config.json').exists():
        raise ValueError('Use a fresh output directory')

    def write(name, data):
        (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    Config = base.GRPOConfig
    def config(**kw):
        kw.update(max_steps=1, num_iterations=1, warmup_steps=0, warmup_ratio=0,
                  lr_scheduler_type='constant', save_strategy='no', report_to=[])
        result = Config(**kw)
        write('effective_grpo_config.json', result.to_dict())
        return result
    base.GRPOConfig = config
    save_config = base.save_run_config
    def save_probe_config(output_dir, args, dataset_size, **unused):
        save_config(output_dir, args, dataset_size, 1, 0)
        path = Path(output_dir) / 'run_config.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload.update(diagnostic='fixed_group_one_update', max_steps=1, num_iterations=1,
                       lr_scheduler_type='constant', candidate_attempts=extra.candidate_attempts,
                       candidate_start=extra.candidate_start)
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    base.save_run_config = save_probe_config
    Native = base.GRPOTrainer
    keys = ('pixel_values', 'image_grid_thw', 'num_images', 'pixel_attention_mask',
            'spatial_shapes', 'num_tiles', 'image_sizes', 'token_type_ids',
            'mm_token_type_ids', 'image_position_ids')

    class Probe(Native):
        def _calculate_rewards(self, *a, **kw):
            result = super()._calculate_rewards(*a, **kw)
            if result.shape != (4, 1):
                raise RuntimeError('Expected four responses and one correctness reward')
            self.latest_rewards = result[:, 0].detach().clone()
            return result

        def measure(self):
            data = self.fixed
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad(), disable_gradient_checkpointing(self.model), self.compute_loss_context_manager():
                    lp, _, _ = self._get_per_token_logps_and_entropies(
                        self.model, torch.cat([data['prompt_ids'], data['completion_ids']], 1),
                        torch.cat([data['prompt_mask'], data['completion_mask']], 1),
                        data['completion_ids'].shape[1], batch_size=2,
                        **{k: data[k] for k in keys if k in data})
                return lp.detach().float()
            finally:
                self.model.train(was_training)

        def _generate_and_score_completions(self, inputs):
            if hasattr(self, 'fixed'):
                raise RuntimeError('Unexpected second generation after selecting fixed trajectories')
            history = []
            for attempt in range(extra.candidate_attempts):
                idx = (extra.candidate_start + attempt) % len(self.train_dataset)
                row = self.train_dataset[idx]
                data = super()._generate_and_score_completions([dict(row) for _ in range(4)])
                adv = data['advantages']
                mask = data['completion_mask']
                # Native truncation masking zeros the entire truncated response.
                rewards = self.latest_rewards
                if not bool(((rewards == 0) | (rewards == 1)).all()):
                    raise RuntimeError('Expected finite binary correctness rewards')
                valid = bool((mask.sum(1) > 0).all() and (adv > 0).any() and (adv < 0).any())
                history.append(dict(attempt=attempt, qa_id=row['qa_id'], accepted=valid,
                                    rewards=rewards.tolist(), advantages=adv.tolist(), valid_tokens=mask.sum(1).tolist()))
                write('candidate_search.json', history)
                if not valid:
                    continue
                self.fixed = data
                self.before = self.measure()
                self.repeat_before = self.measure()
                data['old_per_token_logps'] = self.before.clone()
                self.initial = {n: v.detach().cpu().clone() for n, v in self.model.named_parameters() if v.requires_grad}
                self.dtypes = {n: str(v.dtype) for n, v in self.model.named_parameters() if v.requires_grad}
                tokenizer = getattr(self.processing_class, 'tokenizer', self.processing_class)
                self.rewards = rewards.int().tolist()
                write('fixed_rollouts.json', dict(qa_id=row['qa_id'], question=row.get('question'),
                    rewards=self.rewards, advantages=adv.tolist(), completion_ids=data['completion_ids'].tolist(),
                    completion_mask=mask.tolist(), prompt_ids=data['prompt_ids'].tolist(),
                    answers=tokenizer.batch_decode(data['completion_ids'], skip_special_tokens=False)))
                torch.save({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in data.items()}, out / 'fixed_batch.pt')
                print('SELECTED:', row['qa_id'], 'rewards=', self.rewards, flush=True)
                # TRL subsequently splits/shuffles multimodal fields; preserve the
                # original unsplit order for every probability measurement.
                return copy.deepcopy(data)
            write('result.json', dict(status='no_eligible_group', optimizer_updates=0))
            raise RuntimeError('No mixed, untruncated group found; no optimizer update performed')

    class Audit(base.TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
            ids = {id(p) for group in optimizer.param_groups for p in group['params']}
            missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in ids]
            if missing:
                raise RuntimeError(f'Parameters missing from optimizer: {missing}')

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kw):
            trainer.grad = {n: float(p.grad.detach().float().norm()) if p.grad is not None else None
                            for n, p in model.named_parameters() if p.requires_grad}
            trainer.actual_lr = [group['lr'] for group in optimizer.param_groups]
            if any(value is not None and not math.isfinite(value) for value in trainer.grad.values()):
                raise FloatingPointError('Nonfinite gradient before optimizer update')

        def on_step_end(self, args, state, control, **kw):
            trainer.after = trainer.measure()

    trainer = None
    class RunProbe(Probe):
        def __init__(self, *a, **kw):
            nonlocal trainer
            super().__init__(*a, **kw)
            trainer = self
            self.add_callback(Audit())
    base.GRPOTrainer = RunProbe
    base.run_training(args)
    if trainer.state.global_step != 1:
        raise RuntimeError('Expected exactly one optimizer update')
    changed = []
    for n, p in trainer.model.named_parameters():
        if n in trainer.initial:
            delta = p.detach().cpu().float() - trainer.initial[n].float()
            changed.append(dict(name=n, dtype=trainer.dtypes[n], grad_norm=trainer.grad[n],
                                changed_elements=int((delta != 0).sum()), max_abs_delta=float(delta.abs().max())))
    # Load saved weights into a separate adapter; match dtypes to avoid silently
    # comparing BF16 trained weights against PEFT's promoted FP32 reload.
    model = trainer.model
    model.load_adapter(str(out / 'final_adapter'), adapter_name='probe_reload', is_trainable=True,
                       autocast_adapter_dtype=False)
    for n, p in model.named_parameters():
        if '.probe_reload.' in n:
            original = n.replace('.probe_reload.', '.default.')
            p.data = p.data.to(trainer.initial[original].dtype)
    model.set_adapter('probe_reload')
    try:
        reloaded = trainer.measure()
    finally:
        model.set_adapter('default')
        model.delete_adapter('probe_reload')
    mask = trainer.fixed['completion_mask'].float()
    adv = trainer.fixed['advantages'].float()
    def stats(lp):
        return dict(sum_logp=(lp * mask).sum(1).tolist(),
                    mean_logp=((lp * mask).sum(1) / mask.sum(1)).tolist())
    delta = trainer.after - trainer.before
    # First-order directional statistic and exact fixed-old-policy clipped objective.
    def objective(lp):
        return float(clipped_objective(torch, lp, trainer.before, adv, mask,
                     trainer.max_completion_length, trainer.epsilon_low, trainer.epsilon_high))
    result = dict(status='completed', optimizer_updates=1, learning_rates=trainer.actual_lr,
        before=stats(trainer.before), after=stats(trainer.after), reloaded=stats(reloaded),
        positive_advantage_direction=float((delta * adv[:, None] * mask).sum() / (4 * trainer.max_completion_length)),
        clipped_objective_before=objective(trainer.before), clipped_objective_after=objective(trainer.after),
        repeated_forward_max_abs_logp=float(((trainer.before-trainer.repeat_before).abs()*mask).max()),
        reload_max_abs_logp=float(((reloaded-trainer.after).abs()*mask).max()), parameters=changed,
        limitations=['Selected one mixed group; not evidence of dataset accuracy improvement.',
                     'BS2/GA2, no warmup, one update; other loss/optimizer settings inherited.',
                     'Reloads adapter on the same base instance, not a new process.',
                     'No forced pass threshold; compare improvement with repeated-forward numerical noise.'],
        versions={k: importlib.metadata.version(k) for k in ('torch','trl','peft','transformers')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write('result.json', result)
    torch.save(dict(before=trainer.before.cpu(), after=trainer.after.cpu(), reloaded=reloaded.cpu()), out / 'token_logps.pt')
    print('DIAGNOSTIC COMPLETE:', out / 'result.json', flush=True)


if __name__ == '__main__':
    main()
