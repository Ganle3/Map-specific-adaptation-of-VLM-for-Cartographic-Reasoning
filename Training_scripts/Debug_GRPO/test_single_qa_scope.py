import unittest
from train_single_qa_scope import targets_for
from train_mapwise_grpo_scope_ablation import preset

class ScopeTests(unittest.TestCase):
    def test_exact_union(self):
        language=targets_for('language_r16')
        joint=targets_for('joint_r16')
        visual=preset('vision_merger_r16')[1]
        self.assertEqual(len(language),252)
        self.assertEqual(len(joint),368)
        self.assertFalse(language & visual)
        self.assertEqual(joint, language | visual)
        self.assertEqual(sum('merger' in n for n in joint),8)
        self.assertFalse(any('norm' in n or 'lm_head' in n or 'embed' in n for n in joint))

if __name__=='__main__':
    unittest.main()
