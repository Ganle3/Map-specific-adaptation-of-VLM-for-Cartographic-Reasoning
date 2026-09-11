"""CPU checks; tensor/autograd checks also run automatically in the Euler job.

python Training_scripts/Vision_GRPO/test_vislora_reuse.py
No model download. These checks do not substitute for a real VLM smoke run.
"""
import importlib.util
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from _vislora_replay_buffer import SuccessBuffer
from _vislora_reuse_trainers import make_trainers, replay_correction

HAS_TORCH = importlib.util.find_spec("torch") is not None


class BufferChecks(unittest.TestCase):
    def test_success_and_termination_both_required(self):
        b = SuccessBuffer()
        self.assertFalse(b.add("q", [1], [2], [-0.2], 0, True, 0))
        self.assertFalse(b.add("q", [1], [2], [-0.2], 1, False, 0))
        self.assertEqual(b.items, {})

    def test_capacity_and_duplicate_refresh(self):
        b = SuccessBuffer(2)
        for step in range(4):
            b.add("q", [1], [step], [-0.2], 1, True, step)
        self.assertEqual([x["tokens"] for x in b.items["q"]], [[2], [3]])
        b.add("q", [1], [3], [-0.7], 1, True, 8)
        self.assertEqual(len(b.items["q"]), 2)
        self.assertEqual(b.items["q"][-1]["step"], 8)

    def test_prompt_identity_and_age(self):
        b = SuccessBuffer()
        b.add("q", [1, 2], [3, 9], [-0.3, -0.4], 1, True, 3)
        self.assertIsNone(b.choose("q", [1, 2], 3, 4))
        self.assertIsNone(b.choose("q", [2, 1], 4, 4))
        self.assertIsNone(b.choose("other", [1, 2], 4, 4))
        self.assertIsNone(b.choose("q", [1, 2], 8, 4))
        self.assertIsNotNone(b.choose("q", [1, 2], 7, 4))

    def test_invalid_probability_alignment(self):
        for logps in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                SuccessBuffer().add("q", [1], [2], logps, 1, True, 0)


class NativeFake:
    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, **kwargs):
        return [[i + 30, 9] for i in range(len(prompt_ids))], None


class ReplayRoutingChecks(unittest.TestCase):
    def trainer(self, step=11, calls=0):
        options = SimpleNamespace(replay_start_step=11, replay_max_age=44)
        _, Replay = make_trainers(SimpleNamespace(torch=None, GRPOTrainer=NativeFake), options)
        t = Replay.__new__(Replay)
        t.num_generations = 4
        t.state = SimpleNamespace(global_step=step)
        t.reuse_calls = calls
        t.train_dataset = [None] * 44
        t._replay_inputs = [{"qa_id": f"q{i // 4}"} for i in range(16)]
        t._pending_tokens = None
        t._pending_replays = {}
        t.successes = SuccessBuffer()
        for i in range(4):
            t.successes.add(f"q{i}", [i], [100 + i, 9], [-0.1, -0.2], 1, True, 0)
        t.fresh_count = t.fresh_tokens = t.replay_count = t.discarded_count = 0
        return t

    def test_same_question_fixed_slot_and_fresh_budget(self):
        t = self.trainer()
        tokens, _ = t._generate_single_turn([[i // 4] for i in range(16)], None, {})
        self.assertEqual(set(t._pending_replays), {3, 11})
        self.assertEqual(tokens[3], [100, 9])
        self.assertEqual(tokens[11], [102, 9])
        self.assertEqual(tokens[7], [37, 9])
        self.assertEqual((t.fresh_count, t.replay_count, t.discarded_count), (16, 2, 2))

    def test_selection_rotates_by_pass(self):
        t = self.trainer(calls=11)
        t._generate_single_turn([[i // 4] for i in range(16)], None, {})
        self.assertEqual(set(t._pending_replays), {7, 15})

    def test_no_replay_during_initial_pass(self):
        t = self.trainer(step=10)
        t._generate_single_turn([[i // 4] for i in range(16)], None, {})
        self.assertEqual(t._pending_replays, {})

    def test_cross_qa_group_rejected(self):
        t = self.trainer()
        t._replay_inputs[1]["qa_id"] = "other"
        with self.assertRaisesRegex(RuntimeError, "cross-question"):
            t._generate_single_turn([[i // 4] for i in range(16)], None, {})

    def test_baseline_snapshot_frozen(self):
        root = Path(__file__).parent
        content = (root / "_vislora_grpo_baseline_snapshot.py").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(),
                         "a9a7ac339065adf6ee4cc22970693b37740ea8bc1aa9ace2cf94141f57169641")


@unittest.skipUnless(HAS_TORCH, "torch unavailable locally; these run in the Euler container")
class TensorChecks(unittest.TestCase):
    def test_loss_and_gradient_equal_direct_mixed_objective(self):
        import torch
        for polarity in (-1.0, 0.0, 1.0):
            logps = torch.tensor([[-1.0, -2.0], [-2.0, -3.0]], requires_grad=True)
            old = torch.tensor([[-1.2, -1.9], [-2.1, -2.7]])
            adv = torch.tensor([polarity, -0.5])
            replay = torch.tensor([True, False])
            mask = torch.tensor([[1, 1], [1, 0]])
            ratio = (logps - old).exp()
            clipped = torch.minimum(ratio * adv[:, None], ratio.clamp(.8, 1.2) * adv[:, None])
            native = -(clipped * mask).sum() / (2 * 8 * 4)
            corrected = native + replay_correction(torch, logps, old, adv, replay, mask, .2, .2, .1, 8, 4)
            shaped = torch.sigmoid(logps - old - math.log(.1)) * adv[:, None]
            expected = -(torch.where(replay[:, None], shaped, clipped) * mask).sum() / (2 * 8 * 4)
            torch.testing.assert_close(corrected, expected)
            g1 = torch.autograd.grad(corrected, logps, retain_graph=True)[0]
            g2 = torch.autograd.grad(expected, logps)[0]
            torch.testing.assert_close(g1, g2)
            self.assertEqual(g1[1, 1].item(), 0.0)  # masked token cannot update

    def test_zero_advantage_group_has_zero_gradient(self):
        import torch
        x = torch.zeros((4, 3), requires_grad=True)
        correction = replay_correction(torch, x, torch.zeros_like(x), torch.zeros(4),
                                       torch.tensor([True, False, False, False]), torch.ones_like(x),
                                       .2, .2, .1, 3, 8)
        correction.backward()
        self.assertEqual(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
