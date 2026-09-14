"""Versioned E2 inference: native NF4 loader + kbit preparation + BF16 LoRA.

Preserves the legacy entry point. Only use for adapters trained by the matching
prepared NF4 pipeline; no optimizer or backward pass is created here.
"""
import hashlib
import json
from pathlib import Path
from collections import Counter


def install(inference, audit_dir):
    import torch
    from peft import prepare_model_for_kbit_training
    original = inference.load_model_and_processor

    def load(*args, **kwargs):
        model, processor, adapter = original(*args, **kwargs)
        # Keep the exact order validated by experiment E2 (14110807).
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        count = 0
        for name, param in model.named_parameters():
            if 'lora_A.' in name or 'lora_B.' in name:
                param.data = param.data.to(torch.bfloat16)
                count += 1
        if adapter is not None and count == 0:
            raise RuntimeError('No LoRA tensors found in adapted model')
        model.eval()
        active = []
        if adapter is not None:
            active = getattr(model, 'active_adapters', None)
            active = active() if callable(active) else active
            active = [active] if isinstance(active, str) else active
        inventory = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype),
                          parameter_class=type(p).__name__, requires_grad=p.requires_grad)
                     for n,p in model.named_parameters()]
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir/'loading_audit_e2.json').write_text(json.dumps(dict(
            loading_protocol='E2_v1', adapter=str(adapter), active_adapters=active,
            adapter_tensor_count=count, training=model.training,
            autocast_at_load=torch.is_autocast_enabled(), parameters=inventory,
            dtype_tensor_counts=dict(Counter(p['dtype'] for p in inventory)),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2), encoding='utf-8')
        return model, processor, adapter

    inference.load_model_and_processor = load


def main():
    import inference_mapwise_trl as inference
    args = inference.parse_args()
    install(inference, args.output_json.expanduser().resolve().parent)
    inference.main()


if __name__ == '__main__':
    main()
