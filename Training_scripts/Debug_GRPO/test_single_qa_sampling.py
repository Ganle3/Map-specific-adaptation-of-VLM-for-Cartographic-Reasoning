import unittest
from evaluate_single_qa_sampling import wilson, paired_summary


class SamplingTests(unittest.TestCase):
    def test_wilson_boundaries(self):
        self.assertAlmostEqual(wilson(0,100)[0],0)
        self.assertAlmostEqual(wilson(100,100)[1],1)
        lo,hi=wilson(50,100)
        self.assertAlmostEqual(lo,1-hi)
        self.assertLess(lo,.5)
        self.assertGreater(hi,.5)

    def test_pair_alignment(self):
        r=paired_summary([0,0,1,1],[1,1,0,1])
        self.assertEqual(r['accuracy_difference'],.25)
        self.assertEqual(r['wrong_to_correct'],2)
        self.assertEqual(r['correct_to_wrong'],1)
        with self.assertRaises(ValueError):
            paired_summary([1],[])


if __name__ == '__main__':
    unittest.main()
