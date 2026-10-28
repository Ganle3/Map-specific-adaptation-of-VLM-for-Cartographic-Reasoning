"""CPU checks for selected data and fixed diagnostic settings."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from grpo_joint_debug import (EXPECTED_DEBUG_QA, validate_debug_dataset, preset,
                               dtype_changes, expected_bias_cast)

FOLDER=Path(__file__).resolve().parent
DATA=FOLDER.parents[1]/'Datasets/Processed_Mapwise/Train_Val'

class Checks(unittest.TestCase):
    def test_only_first_frozen_bias_cast_is_allowed(self):
        b=dict(dtype='torch.float32',trainable=False,shape=[4],parameter_class='Parameter')
        a=dict(b,dtype='torch.bfloat16')
        c=dict(name='visual.bias',before=b,after=a)
        self.assertTrue(expected_bias_cast(c,{'visual.bias'},set()))
        self.assertFalse(expected_bias_cast(c,{'visual.bias'},{'visual.bias'}))
        self.assertFalse(expected_bias_cast(c,set(),set()))
        self.assertFalse(expected_bias_cast(dict(c,after=dict(a,trainable=True)),{'visual.bias'},set()))
        self.assertFalse(expected_bias_cast(dict(c,before=a,after=b),{'visual.bias'},set()))

    def test_dtype_changes_identify_parameters(self):
        before={'base':{'dtype':'torch.float32'}, 'adapter':{'dtype':'torch.bfloat16'}}
        self.assertEqual(dtype_changes(before, dict(before)), [])
        after={**before, 'base':{'dtype':'torch.bfloat16'}}
        changed=dtype_changes(before,after)
        self.assertEqual([r['name'] for r in changed], ['base'])
        self.assertEqual(changed[0]['before']['dtype'], 'torch.float32')
        self.assertEqual(changed[0]['after']['dtype'], 'torch.bfloat16')
        self.assertEqual(len(dtype_changes(before,{})),2)

    def test_selection_and_anchor(self):
        rows=validate_debug_dataset(DATA/'mapwise_grpo_joint_debug4_src2051.json')
        improved=json.loads((DATA/'mapwise_grpo_joint44_improved_20.json').read_text(encoding='utf-8'))
        keys={(r['country'],r['source_index']) for r in improved}
        self.assertEqual(len(keys),20)
        self.assertNotIn(EXPECTED_DEBUG_QA[0],keys)
        self.assertTrue(set(EXPECTED_DEBUG_QA[1:])<=keys)
        self.assertEqual(len(rows),4)

    def test_reject_wrong_dataset(self):
        bad=[dict(country='usa',source_index=2051)]*4
        with patch.object(Path,'read_text',return_value=json.dumps(bad)):
            with self.assertRaises(ValueError):validate_debug_dataset('fake.json')

    def test_scope_and_schedule(self):
        rank,targets=preset('joint')
        self.assertEqual(rank,16);self.assertEqual(len(targets),368)
        source=(FOLDER/'grpo_joint_debug.py').read_text()
        # Defaults are declared by argparse and then passed dynamically into
        # GRPOConfig; inspecting kwargs.update for literal values is stale.
        self.assertIn('default="constant"',source)
        self.assertIn('default=0.0',source)
        self.assertIn('default=1,',source)

if __name__=='__main__':unittest.main()
