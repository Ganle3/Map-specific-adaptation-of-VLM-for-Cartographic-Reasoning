"""Native TRL 1.12 fixed20 gradient measurement, without an optimizer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diagnostics.configs.fixed20_batches import FIXED_BATCHES, index_fixed20
from diagnostics.gradient.analyze_batch_gradients import write_results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def state_digest(model):
    """Hash parameters and buffers sequentially; never duplicate the base model on GPU."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def measure(trainer, options, base):
    import torch
    from bitsandbytes.nn import Linear4bit

    # Trainer.train normally installs Accelerate's mixed-precision forward wrapper.
    # Prepare only the model here: never create/prepare an optimizer or scheduler.
    trainer.model = trainer.accelerator.prepare_model(trainer.model)
    trainer.model_wrapped = trainer.model
    model = trainer.model
    if trainer.accelerator.num_processes != 1 or trainer.use_vllm or trainer.use_liger_kernel:
        raise ValueError("Only single-GPU native TRL without Liger is supported")
    if trainer.args.use_transformers_continuous_batching:
        raise ValueError("Continuous batching is not supported")
    index = index_fixed20(trainer.train_dataset)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any('.lora_A.' not in n and '.lora_B.' not in n for n, _ in named):
        raise ValueError("Only LoRA A/B parameters may be trainable")
    params = [p for _, p in named]
    count = sum(p.numel() for p in params)
    # bitsandbytes casts frozen visual biases on first forward in the training run.
    # Normalize these once BEFORE defining theta, so every measured batch shares theta.
    casts = []
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit) and '.visual.' in name and module.bias is not None:
            if not module.bias.requires_grad and module.bias.dtype != torch.bfloat16:
                casts.append(name + '.bias')
                module.bias.data = module.bias.data.to(torch.bfloat16)
    # Non-reentrant checkpointing supports autograd.grad; no loss/objective change.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainer.current_gradient_accumulation_steps = 8
    baseline_hash = state_digest(model)
    config = dict(checkpoint=str(options.checkpoint), base_model=options.base_model,
                  seed=options.seed, num_repeats=options.num_repeats, num_generations=4,
                  per_device_batch_size=2, gradient_accumulation_steps=8,
                  max_completion_length=trainer.args.max_completion_length,
                  temperature=trainer.args.temperature, top_p=trainer.args.top_p,
                  lora_config=model.peft_config['default'].to_dict(),
                  num_trainable_params=count, timestamp=datetime.now(timezone.utc).isoformat(),
                  eps=options.eps, fixed_batches=FIXED_BATCHES,
                  gradient_definition="sum of 8 native dr_grpo microbatch gradients; native loss already divides by 8; before clipping/optimizer",
                  frozen_visual_bias_casts=casts, fixed_state_sha256=baseline_hash,
                  gradient_checkpointing_use_reentrant=False,
                  effective_grpo_config=trainer.args.to_dict(),
                  training_config=str(options.training_config) if options.training_config else None,
                  qa_json=str(options.qa_json), qa_json_sha256=hashlib.sha256(options.qa_json.read_bytes()).hexdigest(),
                  trainable_parameters=[dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n, p in named],
                  versions={p: importlib.metadata.version(p) for p in ('torch', 'trl', 'transformers', 'peft', 'accelerate', 'bitsandbytes')},
                  source_sha256={str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in (base.__file__, __file__, base.DEFAULT_EVALUATION_SCRIPT,
                                           inspect.getfile(type(trainer).__mro__[1]))})
    try:
        config['git_commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        config['git_commit'] = None
    config['status'] = 'running'
    write_json(options.output_dir / 'config.json', config)
    for repeat in range(options.num_repeats):
        output = options.output_dir if options.num_repeats == 1 else options.output_dir / f'repeat_{repeat:03d}'
        output.mkdir(parents=True, exist_ok=True)
        vectors, metrics = [], []
        for batch_number, (batch_id, ids) in enumerate(FIXED_BATCHES.items()):
            seed = options.seed + repeat * 5 + batch_number
            base.set_seed(seed)
            trainer._step = 0
            trainer._buffered_inputs = None
            trainer.generation_calls = 0
            model.zero_grad(set_to_none=True)
            rows = [dict(trainer.train_dataset[index[qid]]) for qid in ids for _ in range(4)]
            gradient = torch.zeros(count, dtype=torch.float32, device='cpu')
            loss_sum = 0.0
            for micro in range(8):
                # Native path handles image patch splitting, rollout shuffling and buffering.
                with trainer.compute_loss_context_manager():
                    prepared = trainer._prepare_inputs(rows)
                    if prepared['completion_ids'].shape[0] != 2:
                        raise RuntimeError("Expected two rollouts per microbatch")
                    loss = trainer.compute_loss(model, prepared)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss")
                pieces = torch.autograd.grad(loss, params, allow_unused=True)
                offset = 0
                for parameter, piece in zip(params, pieces):
                    size = parameter.numel()
                    if piece is not None:
                        if not torch.isfinite(piece).all():
                            raise FloatingPointError("Nonfinite LoRA gradient")
                        gradient[offset:offset + size].add_(piece.detach().reshape(-1).float().cpu())
                    offset += size
                loss_sum += float(loss.detach())
                trainer._step += 1
                del pieces, piece, loss, prepared
            if trainer.generation_calls != 1:
                raise RuntimeError("Expected exactly one generation call per optimizer batch")
            if any(p.grad is not None for p in params):
                raise RuntimeError("autograd.grad unexpectedly populated parameter.grad")
            rewards = trainer.latest_rewards.reshape(4, 4)
            signal = int((rewards.std(dim=1, unbiased=False) > 0).sum())
            metrics.append(dict(batch_id=batch_id, qa_ids=json.dumps(ids), seed=seed,
                                loss=loss_sum, mean_reward=float(rewards.mean()),
                                reward_std=float(rewards.std(unbiased=False)), signal_group_count=signal,
                                signal_fraction=signal / 4, num_trainable_params=count))
            vectors.append(gradient.numpy())
            trainer._buffered_inputs = None
            model.zero_grad(set_to_none=True)
            del rows, gradient
            gc.collect()
            torch.cuda.empty_cache()
            if state_digest(model) != baseline_hash:
                raise RuntimeError("Model parameters/buffers changed during measurement")
            print(f"repeat={repeat} {batch_id}: loss={loss_sum:.6g}, signal_groups={signal}/4", flush=True)
        write_results(output, list(FIXED_BATCHES), vectors, metrics, options.eps)
        del vectors
    config.update(status='complete', model_state_unchanged=True, optimizer_updates=0)
    write_json(options.output_dir / 'config.json', config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--base-model', default='Qwen/Qwen3-VL-8B-Thinking')
    parser.add_argument('--qa-json', type=Path, default=ROOT / 'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_joint44_improved_20.json')
    parser.add_argument('--image-root', type=Path, default=ROOT / 'Datasets/mapwise-dataset')
    parser.add_argument('--training-config', type=Path, help='Original effective_grpo_config.json; auto-detected beside checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-repeats', type=int, default=1)
    parser.add_argument('--eps', type=float, default=1e-12)
    options = parser.parse_args()
    if options.num_repeats < 1 or not 0 < options.eps < float('inf'):
        parser.error('num-repeats must be positive; eps must be finite and positive')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Single GPU/process only')
    if not (options.checkpoint / 'adapter_config.json').is_file():
        parser.error('--checkpoint must be a PEFT adapter/checkpoint directory')
    if options.output_dir.exists() and any(options.output_dir.iterdir()):
        parser.error('Use an empty output directory')
    if importlib.metadata.version('trl') != '1.12.0':
        raise RuntimeError('Requires the original TRL 1.12.0 training environment')
    sys.path.insert(0, str(ROOT / 'Training_scripts/Debug_GRPO'))
    import _vislora_grpo_baseline_snapshot as base
    from train_mapwise_grpo_joint4 import preset

    # Reuse the monolithic loader via a process-local Trainer.train replacement.
    # It measures and exits before baseline's training/save continuation. No source edits.
    class MeasurementComplete(Exception):
        pass

    class DiagnosticTrainer(base.GRPOTrainer):
        def _calculate_rewards(self, *args, **kwargs):
            rewards = super()._calculate_rewards(*args, **kwargs)
            if rewards.shape != (16, 1) or not ((rewards == 0) | (rewards == 1)).all():
                raise RuntimeError('Expected 16 binary correctness rewards')
            self.latest_rewards = rewards[:, 0].detach().float().cpu()
            return rewards

        def _generate_and_score_completions(self, inputs):
            self.generation_calls += 1
            return super()._generate_and_score_completions(inputs)

        def train(self, *args, **kwargs):
            base.REWARD_GROUPS_PATH = None
            measure(self, options, base)
            raise MeasurementComplete

    with patch.object(sys, 'argv', [sys.argv[0]]):
        args = base.parse_args()
    args.model_name = options.base_model
    args.qa_json, args.image_root = options.qa_json, options.image_root
    args.output_dir, args.init_adapter_path = options.output_dir, options.checkpoint
    args.no_wandb, args.preserve_qa_order = True, True
    args.seed = options.seed
    args.num_generations, args.per_device_train_batch_size, args.gradient_accumulation_steps = 4, 2, 8
    args.lora_rank = args.lora_alpha = 16
    _, targets = preset('joint_r16')
    index_fixed20({'qa_id': [row['qa_id'] for row in base.load_json_list(options.qa_json)]})

    def verify_joint(model):
        actual = set(model.targeted_module_names)
        if actual != targets:
            raise ValueError('Checkpoint does not match fixed20 joint_r16 module scope')
        adapter = model.peft_config['default']
        if adapter.lora_dropout != 0 or adapter.bias != 'none' or adapter.use_dora:
            raise ValueError('Expected standard joint LoRA: dropout=0, bias=none, no DoRA')

    if options.training_config is None:
        candidate = options.checkpoint.parent / 'effective_grpo_config.json'
        if candidate.is_file():
            options.training_config = candidate
    saved = json.loads(options.training_config.read_text(encoding='utf-8')) if options.training_config else {}
    required = dict(num_generations=4, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    loss_type='dr_grpo', beta=0.0, scale_rewards='group', mask_truncated_completions=True,
                    num_iterations=1, use_vllm=False, use_liger_kernel=False)
    for key, value in required.items():
        if key in saved and saved[key] != value:
            raise ValueError(f'Original config {key}={saved[key]!r} differs from required {value!r}')
    Config = base.GRPOConfig

    def config(**kwargs):
        if saved:
            kwargs.update(saved)
        kwargs.update(required, output_dir=str(options.output_dir), report_to=[], save_strategy='no',
                      seed=options.seed, data_seed=options.seed, shuffle_dataset=False,
                      steps_per_generation=8, generation_batch_size=None,
                      gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False})
        result = Config(**kwargs)
        if result.generation_batch_size != 16 or result.steps_per_generation != 8:
            raise RuntimeError('TRL must derive exactly 16 rollouts and eight microbatches')
        return result

    base.validate_args(args)
    with patch.multiple(base, GRPOTrainer=DiagnosticTrainer, GRPOConfig=config,
                        verify_vision_only_lora=verify_joint, save_run_config=lambda **kwargs: None):
        try:
            base.run_training(args)
        except MeasurementComplete:
            pass


if __name__ == '__main__':
    main()
