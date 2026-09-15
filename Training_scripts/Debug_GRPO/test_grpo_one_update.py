import unittest
import torch
from diagnose_grpo_one_update import clipped_objective


class ObjectiveTests(unittest.TestCase):
    def test_gradient_and_mask(self):
        lp = torch.zeros(2, 3, requires_grad=True)
        old = lp.detach().clone()
        adv = torch.tensor([1., -1.])
        mask = torch.tensor([[1., 1., 0.], [1., 0., 0.]])
        objective = clipped_objective(torch, lp, old, adv, mask, 3, .2, .2)
        objective.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([[1., 1., 0.], [-1., 0., 0.]]) / 6)
        self.assertGreater(float(clipped_objective(torch, lp.detach()+.01*lp.grad,
                           old, adv, mask, 3, .2, .2)), float(objective))

    def test_clipping_and_zero_signal(self):
        old = torch.zeros(2, 1)
        lp = torch.tensor([[1.], [-1.]], requires_grad=True)
        mask = torch.ones_like(old)
        objective = clipped_objective(torch, lp, old, torch.tensor([1., -1.]), mask, 1, .2, .2)
        objective.backward()
        torch.testing.assert_close(lp.grad, torch.zeros_like(lp))
        self.assertAlmostEqual(float(objective), .2, places=6)
        self.assertEqual(float(clipped_objective(torch, lp, old, torch.zeros(2), mask, 1, .2, .2)), 0)


if __name__ == '__main__':
    unittest.main()
