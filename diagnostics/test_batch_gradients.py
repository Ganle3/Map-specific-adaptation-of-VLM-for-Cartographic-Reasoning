"""CPU tests, including a toy training loop checking the logging hook is read-only."""
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'Training_scripts/Debug_GRPO'))
from diagnostics.gradient.analyze_batch_gradients import gradient_pair, TrajectoryRecorder
from diagnostics.gradient.grpo_batch_gradient import copy_accumulated_gradient, make_online_trainer
from diagnostics.visualization.plot_batch_gradient_matrix import plot_results, plot_comparison


def metadata(size=20):
    return {f'q{i}': dict(template_no=i % 3, ability_level=f'L{i%2}', country='china', ground_truth_type='Count') for i in range(size)}


def groups(ids):
    return [dict(qa_id=q, rewards=[0, 1, 0, 1]) for q in ids]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_geometry():
    assert gradient_pair([1., 0], [-1., 0])['cosine'] == -1
    assert gradient_pair([1., 0], [0., 2])['cosine'] == 0
    assert gradient_pair([1., 0], [-1., 0])['distance'] == 2
    assert gradient_pair([1., 0], [-1., 0])['dot_product'] == -1
    for v in ([0., 0], [1e-14, 0]):
        pair = gradient_pair(v, [1., 0])
        assert not pair['valid_pair'] and np.isnan(pair['cosine'])


@pytest.mark.parametrize('a,b', [([np.nan], [0]), ([np.inf], [0]), ([1], [1, 2])])
def test_invalid_geometry(a, b):
    with pytest.raises(ValueError):
        gradient_pair(a, b)


def test_history_and_visit_gaps(tmp_path):
    logger = TrajectoryRecorder(tmp_path, metadata(), history_steps=2)
    ids = list(metadata())
    for step in range(1, 7):
        batch = ids[((step-1)%5)*4:((step-1)%5+1)*4]
        logger.record(step, np.array([(-1.)**step, 0]), groups(batch), 0)
    logger.close()
    pairs = read(tmp_path/'gradient_pairs.csv')
    assert len(pairs) == 9
    assert {int(p['lag']) for p in pairs} == {1, 2}
    assert all(float(p['cosine']) == (-1.)**int(p['lag']) for p in pairs)
    visits = [r for r in read(tmp_path/'qa_metrics.csv') if r['step']=='6']
    assert all(r['steps_since_previous']=='5' and r['visit']=='2' for r in visits)
    assert len(logger.history) == 2


def test_wrong_epoch_coverage_fails(tmp_path):
    logger = TrajectoryRecorder(tmp_path, metadata())
    try:
        for step in range(1, 5):
            logger.record(step, np.ones(2), groups([f'q{i}' for i in range(4)]), 0)
        with pytest.raises(RuntimeError, match='once'):
            logger.record(5, np.ones(2), groups([f'q{i}' for i in range(4)]), 0)
    finally:
        logger.close()


class Tensor:
    """Small tensor protocol for testing logger integration without a torch install."""
    def __init__(self, value):
        self.value = np.asarray(value, dtype=np.float32)

    def detach(self):
        return self

    def reshape(self, *shape):
        return Tensor(self.value.reshape(*shape))

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __float__(self):
        return float(self.value)


class Parameter:
    def __init__(self, value, grad=None):
        self.value = np.asarray(value, dtype=np.float32)
        self.grad = grad

    def numel(self):
        return self.value.size


def test_gradient_copy_does_not_alias_or_modify():
    p = Parameter([1, 2], Tensor([3, 4]))
    q = Parameter([5])
    result = copy_accumulated_gradient([('a', p), ('unused', q)])
    np.testing.assert_array_equal(result, [3, 4, 0])
    result[:] = 9
    np.testing.assert_array_equal(p.grad.value, [3, 4])
    assert q.grad is None


def test_online_hook_accumulates_before_clipping_without_changing_updates(tmp_path):
    class Native:
        def training_step(self, model, inputs, num_items_in_batch=None):
            # Toy differentiable objective, already normalized for accumulation.
            derivative = (model.value-inputs)/8
            if model.grad is None:
                model.grad = Tensor(np.zeros_like(model.value))
            model.grad.value += derivative
            return Tensor(np.sum((model.value-inputs)**2)/16)

    base = SimpleNamespace(GRPOTrainer=Native, TrainerCallback=object,
        torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)))
    options = SimpleNamespace(wandb=False)
    Trainer, _ = make_online_trainer(base, options, metadata(), {})
    trainer = Trainer.__new__(Trainer)
    parameter = Parameter([.7, -.3])
    reference = Parameter(parameter.value.copy())
    trainer.named_lora = [('lora_A', parameter)]
    trainer.recorder = TrajectoryRecorder(tmp_path, metadata())
    trainer.micro_count, trainer.loss_sum, trainer.generation_calls = 0, 0., 0
    trainer.current_gradient_accumulation_steps = 8
    trainer.accelerator = SimpleNamespace(sync_gradients=False)
    trainer.state = SimpleNamespace(global_step=0)
    native = Native()
    try:
        for step in range(5):
            trainer.groups = groups([f'q{i}' for i in range(step*4, step*4+4)])
            trainer.generation_calls = 1
            for micro in range(8):
                trainer.accelerator.sync_gradients = micro == 7
                inputs = np.array([step*.2, micro*.1])
                trainer.training_step(parameter, inputs)
                native.training_step(reference, inputs)
                np.testing.assert_array_equal(parameter.grad.value, reference.grad.value)
                assert trainer.recorder.step == step+(micro == 7)
            np.testing.assert_array_equal(trainer.recorder.history[-1][1], reference.grad.value)
            # Simulated native outer-loop clipping and update happen after the hook.
            for p in (parameter, reference):
                p.grad.value *= min(1., .1/np.linalg.norm(p.grad.value))
                p.value -= .01*p.grad.value
                p.grad = None
            np.testing.assert_array_equal(parameter.value, reference.value)
            trainer.state.global_step += 1
    finally:
        trainer.recorder.close()


def test_zero_gradient_csv_and_plots(tmp_path):
    roots = []
    for size in (20, 44):
        root = tmp_path/str(size)
        roots.append(root)
        data = metadata(size)
        recorder = TrajectoryRecorder(root, data)
        for step in range(1, size//4+1):
            batch = groups(list(data)[(step-1)*4:step*4])
            for g in batch:
                g['rewards'] = [1]*4
            recorder.record(step, np.zeros(3), batch, 0)
        recorder.close()
        (root/'config.json').write_text(json.dumps(dict(dataset_size=size, seed=42)))
        plot_results(root)
        assert all(r['valid_gradient']=='False' for r in read(root/'batch_metrics.csv'))
        assert all(r['valid_pairs']=='0' for r in read(root/'lag_summary.csv'))
        assert len(list((root/'plots').glob('*.png'))) == 5
    plot_comparison(roots, tmp_path/'comparison')
    assert (tmp_path/'comparison/joint20_vs_joint44.png').stat().st_size > 1000


def test_actual_dataset_membership():
    directory = ROOT/'Datasets/Processed_Mapwise/Train_Val'
    small = json.loads((directory/'mapwise_grpo_joint44_improved_20.json').read_text())
    large = json.loads((directory/'mapwise_grpo_debug_filtered_44.json').read_text())
    keys = lambda rows: {(r['country'], r['source_index']) for r in rows}
    assert len(keys(small)) == 20 and len(keys(large)) == 44
    assert keys(small) <= keys(large)
