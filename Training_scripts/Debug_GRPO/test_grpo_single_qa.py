import json
from pathlib import Path
import unittest
from train_mapwise_grpo_single_qa import validate_qa, summarize


class SingleQATests(unittest.TestCase):
    def test_exact_subset(self):
        root = Path(__file__).resolve().parents[2]/'Datasets/Processed_Mapwise/Train_Val'
        selected = json.loads((root/'mapwise_grpo_debug_single_src2051.json').read_text())
        source = json.loads((root/'mapwise_grpo_debug_filtered_44_mixed_order.json').read_text())
        validate_qa(selected)
        self.assertIn(selected[0], source)
        with self.assertRaises(ValueError):
            validate_qa(selected*2)

    def test_sampling_windows_keep_zero_signal(self):
        groups = [dict(update=i+1, rewards=r, truncated=t) for i,(r,t) in enumerate([
            ([0,0,0,0], 2), ([1,1,1,1], 0), ([1,0,0,0], 0),
            ([1,1,0,0], 0), ([1,1,1,0], 1), ([1,1,1,1], 0)])]
        a,b = summarize(groups)
        self.assertEqual(a['responses'],20)
        self.assertEqual(a['accuracy'],.5)
        self.assertEqual(a['all_wrong_groups'],1)
        self.assertEqual(a['all_correct_groups'],1)
        self.assertEqual(a['mixed_groups'],3)
        self.assertEqual(a['truncated'],3)
        self.assertEqual(b['first_step'],6)
        self.assertEqual(b['accuracy'],1)


if __name__ == '__main__':
    unittest.main()
