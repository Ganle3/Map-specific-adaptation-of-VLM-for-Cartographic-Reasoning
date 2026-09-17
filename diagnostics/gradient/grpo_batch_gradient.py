"""Random-order joint20/joint44 training with read-only online gradient logging."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diagnostics.gradient.analyze_batch_gradients import TrajectoryRecorder


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str)+'\n', encoding='utf-8')


def copy_accumulated_gradient(named_parameters):
    """Read .grad without modifying it; never concatenate gradients on GPU."""
    result = np.zeros(sum(p.numel() for _, p in named_parameters), dtype=np.float32)
    offset = 0
    for name, parameter in named_parameters:
        size = parameter.numel()
        if parameter.grad is not None:
            values = parameter.grad.detach().reshape(-1).float().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite gradient: {name}')
            result[offset:offset+size] = values
        offset += size
    return result


def make_online_trainer(base, options, metadata, config):
    """Native sampling/loss/backward/clipping/update remain unchanged."""
    torch = base.torch
    from grpo_joint_debug import parameter_inventory, dtype_changes, expected_bias_cast

    class CompletedWithoutSaving(Exception):
        """Stop monolithic baseline before its final adapter/state save."""

    class EndStepAudit(base.TrainerCallback):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, **kwargs):
            if state.max_steps != options.epochs*(len(metadata)//4):
                raise RuntimeError('Unexpected optimizer update budget')

        def on_step_end(self, args, state, control, **kwargs):
            trainer = self.trainer
            if trainer.accelerator.optimizer_step_was_skipped:
                raise RuntimeError('Optimizer skipped an update; inspect partial CSVs')
            if trainer.recorder.step != state.global_step:
                raise RuntimeError('Gradient measurements and optimizer updates are misaligned')
            if options.wandb:
                trainer.log({f'diagnostic/{key}': trainer.latest_batch[key] for key in
                    ('gradient_norm', 'signal_fraction', 'adjacent_cosine', 'negative_cosine_fraction')
                    if math.isfinite(trainer.latest_batch[key])})

    class OnlineTrainer(base.GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (self.accelerator.num_processes != 1 or self.accelerator.scaler is not None
                    or self.is_deepspeed_enabled or self.is_fsdp_enabled or self.use_vllm
                    or self.use_liger_kernel or self.args.use_transformers_continuous_batching):
                raise ValueError('Requires one GPU, native BF16 TRL, no scaler/DeepSpeed/FSDP/Liger/vLLM')
            self.named_lora = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            if not self.named_lora or any('.lora_A.' not in n and '.lora_B.' not in n for n, _ in self.named_lora):
                raise ValueError('Unexpected trainable tensors outside LoRA A/B')
            self.recorder = TrajectoryRecorder(options.output_dir, metadata, options.history_steps, options.eps)
            self.micro_count, self.loss_sum, self.generation_calls = 0, 0.0, 0
            self.groups = None
            self.add_callback(EndStepAudit(self))
            from bitsandbytes.nn import Linear4bit
            self.allowed_bias_casts = {n+'.bias' for n, m in self.model.named_modules()
                if isinstance(m, Linear4bit) and '.visual.' in n and m.bias is not None and not m.bias.requires_grad}
            self.seen_bias_casts = set()
            config.update(effective_grpo_config=self.args.to_dict(),
                num_trainable_params=sum(p.numel() for _, p in self.named_lora),
                lora_config=self.model.peft_config['default'].to_dict(),
                trainable_parameters=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n, p in self.named_lora],
                history_cpu_bytes_estimate=(options.history_steps+1)*4*sum(p.numel() for _, p in self.named_lora))
            write_json(options.output_dir/'config.json', config)

        def _calculate_rewards(self, *args, **kwargs):
            rewards = super()._calculate_rewards(*args, **kwargs)
            if rewards.shape != (16, 1):
                raise RuntimeError('Expected one correctness reward for each of 16 rollouts')
            self.observed_rewards = rewards[:, 0].detach().float().cpu().numpy()
            return rewards

        def _generate_and_score_completions(self, inputs):
            if self.micro_count or self.generation_calls:
                raise RuntimeError('Unexpected generation within an accumulation window')
            ids = [r['qa_id'] for r in inputs]
            if len(ids) != 16 or len(set(ids)) != 4 or any(len(set(ids[i:i+4])) != 1 for i in range(0, 16, 4)):
                raise RuntimeError('Expected four contiguous groups of four rollouts')
            before = parameter_inventory(self.model)
            result = super()._generate_and_score_completions(inputs)
            changes = dtype_changes(before, parameter_inventory(self.model))
            if any(not expected_bias_cast(c, self.allowed_bias_casts, self.seen_bias_casts) for c in changes):
                raise RuntimeError(f'Unexpected parameter dtype changes: {changes}')
            self.seen_bias_casts.update(c['name'] for c in changes)
            if changes:
                config['first_use_bias_casts'] = sorted(self.seen_bias_casts)
                write_json(options.output_dir/'config.json', config)
            self.groups = [dict(qa_id=ids[i], rewards=self.observed_rewards[i:i+4].tolist()) for i in range(0, 16, 4)]
            self.generation_calls += 1
            return result

        def training_step(self, model, inputs, num_items_in_batch=None):
            # Native call performs backward and advances TRL's microstep counter.
            # Outer Transformers loop clips/steps only AFTER this method returns.
            loss = super().training_step(model, inputs, num_items_in_batch)
            self.micro_count += 1
            self.loss_sum += float(loss.detach())
            if self.current_gradient_accumulation_steps != 8:
                raise RuntimeError('Expected a complete eight-microbatch accumulation window')
            if self.micro_count == 8:
                if not self.accelerator.sync_gradients or self.generation_calls != 1 or self.groups is None:
                    raise RuntimeError('Invalid optimizer-batch boundary')
                started = time.perf_counter()
                vector = copy_accumulated_gradient(self.named_lora)
                self.latest_batch = self.recorder.record(self.state.global_step+1, vector, self.groups, self.loss_sum,
                    cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                elapsed = time.perf_counter()-started
                print(f"GRAD step={self.recorder.step} norm={self.latest_batch['gradient_norm']:.5g} "
                      f"active={self.latest_batch['signal_group_count']}/4 readout_seconds={elapsed:.2f}", flush=True)
                self.micro_count, self.loss_sum, self.generation_calls = 0, 0.0, 0
                self.groups = None
            return loss

        def train(self, *args, **kwargs):
            base.REWARD_GROUPS_PATH = None  # qa_metrics.csv already records rewards.
            config['status'] = 'running'
            write_json(options.output_dir/'config.json', config)
            try:
                result = super().train(*args, **kwargs)
                if (self.recorder.step != options.epochs*len(metadata)//4 or self.micro_count
                        or any(v != options.epochs for v in self.recorder.visits.values())):
                    raise RuntimeError('Incomplete epoch/QA coverage')
                config.update(status='complete', optimizer_updates=self.state.global_step,
                              training_metrics=result.metrics, saved_weights=False)
                write_json(options.output_dir/'config.json', config)
            except Exception as error:
                config.update(status='failed', error=repr(error), completed_optimizer_updates=self.state.global_step)
                write_json(options.output_dir/'config.json', config)
                raise
            finally:
                self.recorder.close()
            raise CompletedWithoutSaving

    return OnlineTrainer, CompletedWithoutSaving


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-size', type=int, choices=(20, 44), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--history-steps', type=int, default=11)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--base-model', default='Qwen/Qwen3-VL-8B-Thinking')
    parser.add_argument('--qa-json', type=Path)
    parser.add_argument('--image-root', type=Path, default=ROOT/'Datasets/mapwise-dataset')
    parser.add_argument('--wandb', action='store_true', help='Optional native training and diagnostic scalar logging')
    options = parser.parse_args()
    if min(options.epochs, options.history_steps, options.rank) < 1 or not 0 < options.eps < float('inf'):
        parser.error('epochs/history/rank/eps must be positive, eps finite')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Single process/GPU only')
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        parser.error('Use a new empty output directory; weights/resume are intentionally disabled')
    if options.qa_json is None:
        filename = 'mapwise_grpo_joint44_improved_20.json' if options.dataset_size == 20 else 'mapwise_grpo_debug_filtered_44.json'
        options.qa_json = ROOT/'Datasets/Processed_Mapwise/Train_Val'/filename
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'trl', 'transformers', 'accelerate', 'peft', 'bitsandbytes')}
    if versions['trl'] != '1.12.0' or versions['transformers'] != '5.5.0':
        raise RuntimeError('Use original TRL 1.12.0 / Transformers 5.5.0 container; hook order is version-specific')
    sys.path.insert(0, str(ROOT/'Training_scripts/Debug_GRPO'))
    import _vislora_grpo_baseline_snapshot as base
    from grpo_joint_debug import preset
    rows = base.load_json_list(options.qa_json)
    metadata = {r['qa_id']: r for r in rows}
    if len(rows) != options.dataset_size or len(metadata) != options.dataset_size:
        raise ValueError('Dataset size/unique IDs do not match --dataset-size')
    options.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(status='initializing', diagnostic='online_optimizer_batch_trajectory',
                  timestamp=datetime.now(timezone.utc).isoformat(), **vars(options), versions=versions,
                  expected_optimizer_steps=options.epochs*options.dataset_size//4,
                  gradient_definition='actual accumulated LoRA .grad after backward, before clipping and optimizer.step',
                  comparison='different batches at different parameter states; not a fixed-state conflict/forgetting test',
                  source_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (Path(__file__), Path(base.__file__), Path(inspect.getfile(base.GRPOTrainer)),
                              options.qa_json, base.DEFAULT_EVALUATION_SCRIPT)})
    try:
        config['git_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
    write_json(options.output_dir/'config.json', config)
    with patch.object(sys, 'argv', [sys.argv[0]]):
        args = base.parse_args()
    args.qa_json, args.image_root, args.output_dir = options.qa_json, options.image_root, options.output_dir
    args.model_name, args.seed, args.no_wandb = options.base_model, options.seed, not options.wandb
    args.run_name = f'joint{options.dataset_size}_online_{options.output_dir.name}'
    args.num_train_epochs, args.preserve_qa_order = options.epochs, False
    args.lora_rank = args.lora_alpha = options.rank
    args.num_generations, args.per_device_train_batch_size, args.gradient_accumulation_steps = 4, 2, 8
    args.learning_rate, args.warmup_fraction = 5e-6, 0.0
    _, targets = preset('joint')
    regex = '^(?:'+'|'.join(re.escape(n) for n in sorted(targets))+')$'

    def verify(model):
        if set(model.targeted_module_names) != targets:
            raise ValueError('Joint LoRA targets do not match the training preset')

    Config = base.GRPOConfig

    def training_config(**kwargs):
        kwargs.update(num_train_epochs=options.epochs, max_steps=-1, num_iterations=1,
                      lr_scheduler_type='constant', warmup_steps=0, warmup_ratio=0,
                      shuffle_dataset=True, save_strategy='no', report_to='wandb' if options.wandb else [],
                      steps_per_generation=8, generation_batch_size=None)
        result = Config(**kwargs)
        if result.generation_batch_size != 16 or result.loss_type != 'dr_grpo' or result.beta != 0:
            raise RuntimeError('Unexpected GRPO batch/objective configuration')
        return result

    Trainer, Completed = make_online_trainer(base, options, metadata, config)
    base.validate_args(args)
    # Process-local substitutions reuse the loader; no existing training files change.
    with patch.multiple(base, GRPOTrainer=Trainer, GRPOConfig=training_config,
                        VISION_TARGET_REGEX=regex, verify_vision_only_lora=verify,
                        save_run_config=lambda **kwargs: None):
        try:
            base.run_training(args)
        except Completed:
            pass


if __name__ == '__main__':
    main()
