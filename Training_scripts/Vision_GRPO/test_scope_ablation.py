"""Dependency-free scope isolation checks; GPU runtime validates actual modules."""
import ast
from pathlib import Path
import re
import unittest
from train_mapwise_grpo_scope_ablation import preset


class ScopeTests(unittest.TestCase):
    def test_counts_and_rank(self):
        for name, rank, count in (("vision_r64", 64, 108), ("vision_merger_r16", 16, 116),
                                  ("language_r16", 16, 252)):
            actual_rank, targets = preset(name)
            self.assertEqual(actual_rank, rank)
            self.assertEqual(len(targets), count)

    def test_merger_is_exact_addition(self):
        blocks = preset("vision_r64")[1]
        expanded = preset("vision_merger_r16")[1]
        self.assertTrue(blocks < expanded)
        self.assertEqual(len(expanded - blocks), 8)
        self.assertTrue(all("merger" in n and "linear_fc" in n for n in expanded - blocks))

    def test_language_matches_original_script_regex(self):
        path = Path(__file__).with_name("train_mapwise_grpo_LanLoRA_trl.py")
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        regex = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "LANGUAGE_TARGET_REGEX" for t in n.targets))
        self.assertTrue(all(re.fullmatch(regex, n) for n in preset("language_r16")[1]))
        self.assertFalse(preset("language_r16")[1] & preset("vision_merger_r16")[1])

    def test_no_norms_head_embeddings(self):
        for name in ("vision_r64", "vision_merger_r16", "language_r16"):
            self.assertFalse(any(any(x in n for x in ("norm", "lm_head", "embed")) for n in preset(name)[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
