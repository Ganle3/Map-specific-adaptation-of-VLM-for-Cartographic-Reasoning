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

from _vislora_replay_buffer import SuccessBuffer, minimum_entropy_index
from _vislora_reuse_trainers import make_trainers, replay_correction

HAS_TORCH = importlib.util.find_spec("torch") is not None


class BufferChecks(unittest.TestCase):
    def test_entropy_selection_and_ties(self):
        self.assertEqual(minimum_entropy_index([.9, .2]), 1)
        self.assertEqual(minimum_entropy_index([.2, .2]), 0)
        for scores in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                minimum_entropy_index(scores)

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
        t._candidate_entropies = lambda row, prompt, entries: [float(i) for i in range(len(entries))]
        t._event = lambda *args, **kwargs: None
        return t

    def test_current_entropy_selects_without_changing_old_probabilities(self):
        t = self.trainer()
        t.successes.add("q0", [0], [200, 9], [-.8, -.9], 1, True, 1)
        t._candidate_entropies = lambda row, prompt, entries: [.9, .1][:len(entries)]
        tokens, _ = t._generate_single_turn([[i // 4] for i in range(16)], None, {})
        self.assertEqual(tokens[3], [200, 9])
        self.assertEqual(t._pending_replays[3]["logps"], [-.8, -.9])
        self.assertEqual(t._pending_replays[3]["step"], 1)

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
    def test_entropy_scoring_alignment_image_and_mode(self):
        import torch
        from contextlib import nullcontext
        from unittest.mock import patch
        calls = []
        class ScoringNative(NativeFake):
            def _get_per_token_logps_and_entropies(self, model, ids, mask, length, **kwargs):
                self_test.assertFalse(torch.is_grad_enabled())
                self_test.assertFalse(model.training)
                self_test.assertTrue(kwargs["compute_entropy"])
                self_test.assertEqual(kwargs["num_images"], [1])
                self_test.assertEqual(kwargs["pixel_values"].tolist(), [[7.0]])
                self_test.assertEqual(ids.shape, (1, 2 + length))
                self_test.assertEqual(mask.sum().item(), 2 + length)
                self_test.assertEqual(kwargs["mm_token_type_ids"].tolist(), [[1, 0] + [0] * length])
                calls.append(ids.tolist())
                ent = torch.tensor([[2., 4.]]) if length == 2 else torch.ones((1, length))
                return torch.zeros_like(ent), ent, None
        self_test = self
        _, Replay = make_trainers(SimpleNamespace(torch=torch, GRPOTrainer=ScoringNative), SimpleNamespace())
        t = Replay.__new__(Replay)
        t.model = torch.nn.Linear(1, 1).train()
        t.accelerator = SimpleNamespace(device="cpu")
        t.args = SimpleNamespace(gradient_checkpointing_kwargs=None)
        t._entropy_prompts = ["original-image-prompt"]
        def tokenize(prompts):
            self.assertEqual(prompts, ["original-image-prompt"])
            return [[10, 11]], [["original-image"]], {
                "pixel_values": [[7.]], "image_grid_thw": [[1, 1, 1]],
                "mm_token_type_ids": [[1, 0]]}
        t._tokenize_prompts = tokenize
        utils = SimpleNamespace(disable_gradient_checkpointing=lambda *args: nullcontext())
        with patch.dict("sys.modules", {"trl.models.utils": utils}):
            scores = t._candidate_entropies(0, [10, 11], [{"tokens": [20, 9]}, {"tokens": [21, 22, 9]}])
        self.assertEqual(scores, [3., 1.])
        self.assertTrue(t.model.training)
        self.assertEqual(calls, [[[10, 11, 20, 9]], [[10, 11, 21, 22, 9]]])

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
