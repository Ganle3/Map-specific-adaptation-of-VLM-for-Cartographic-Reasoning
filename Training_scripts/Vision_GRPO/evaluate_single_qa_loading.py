"""Inference-only precision controls; reuse the existing fixed-seed evaluator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loading-mode', choices=['E0', 'E1', 'E2'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'Evaluation_scripts/GRPO_ablation'))
    import torch
    from peft import prepare_model_for_kbit_training
    import inference_mapwise_trl as inference
    import evaluate_single_qa_sampling_extended as sampling

    original_load = inference.load_model_and_processor
    original_inputs = inference.move_inputs_to_model_device
    current_label = None

    def write(name, value):
        (args.output_dir / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

    def load(*positional, **keywords):
        nonlocal current_label
        model, processor, adapter = original_load(*positional, **keywords)
        current_label = Path(adapter).name if adapter else 'baseline'
        # Applied AFTER adapter loading deliberately: kbit preparation casts all
        # non-quantized half parameters to FP32; adapter dtype is then restored.
        # No optimizer/backward/gradient checkpointing is used in this experiment.
        if args.loading_mode == 'E2':
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        adapter_count = 0
        for name, parameter in model.named_parameters():
            if 'lora_A.' in name or 'lora_B.' in name:
                adapter_count += 1
                if args.loading_mode != 'E0':
                    parameter.data = parameter.data.to(dtype=torch.bfloat16)
        if adapter and not adapter_count:
            raise RuntimeError('Checkpoint loaded without any LoRA A/B parameters')
        model.eval()
        inventory = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype),
                          parameter_class=type(p).__name__, requires_grad=p.requires_grad)
                     for n, p in model.named_parameters()]
        checkpoint_hashes = {}
        if adapter:
            for file in sorted(Path(adapter).glob('adapter*')):
                if file.is_file():
                    digest = hashlib.sha256()
                    with file.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024*1024), b''):
                            digest.update(block)
                    checkpoint_hashes[file.name] = digest.hexdigest()
        write(current_label + '_loading_audit.json', dict(
            mode=args.loading_mode, checkpoint=str(adapter),
            checkpoint_hashes=checkpoint_hashes,
            active_adapters=getattr(model, 'active_adapters', None),
            adapter_tensor_count=adapter_count, training=model.training,
            autocast_enabled=torch.is_autocast_enabled(),
            dtype_tensor_counts=dict(Counter(p['dtype'] for p in inventory)),
            parameters=inventory,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            note='E2 matches parameter-storage preparation, not the entire Trainer execution context.'))
        return model, processor, adapter

    def inputs(*positional, **keywords):
        result = original_inputs(*positional, **keywords)
        audit = {}
        for key, value in result.items():
            if torch.is_tensor(value):
                raw = value.detach().cpu().contiguous()
                audit[key] = dict(shape=list(raw.shape), dtype=str(raw.dtype),
                    sha256=hashlib.sha256(raw.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        write(current_label + '_input_audit.json', audit)
        return result

    inference.load_model_and_processor = load
    inference.move_inputs_to_model_device = inputs
    sys.argv = [sys.argv[0], '--output-dir', str(args.output_dir), *remaining]
    sampling.main()


if __name__ == '__main__':
    main()
