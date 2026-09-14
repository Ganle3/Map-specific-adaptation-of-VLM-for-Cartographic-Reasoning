"""CPU checks for selected data and fixed diagnostic settings."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from validate_joint4 import validate_dataset, EXPECTED
from train_mapwise_grpo_joint4 import preset

FOLDER=Path(__file__).resolve().parent
DATA=FOLDER.parents[1]/'Datasets/Processed_Mapwise/Train_Val'

class Checks(unittest.TestCase):
    def test_selection_and_anchor(self):
        rows=validate_dataset(DATA/'mapwise_grpo_joint_debug4_src2051.json')
        improved=json.loads((DATA/'mapwise_grpo_joint44_improved_20.json').read_text(encoding='utf-8'))
        keys={(r['country'],r['source_index']) for r in improved}
        self.assertEqual(len(keys),20)
        self.assertNotIn(EXPECTED[0],keys)
        self.assertTrue(set(EXPECTED[1:])<=keys)
        self.assertEqual(len(rows),4)

    def test_reject_wrong_dataset(self):
        bad=[dict(country='usa',source_index=2051)]*4
        with patch.object(Path,'read_text',return_value=json.dumps(bad)):
            with self.assertRaises(ValueError):validate_dataset('fake.json')

    def test_scope_and_schedule(self):
        rank,targets=preset('joint_r16')
        self.assertEqual(rank,16);self.assertEqual(len(targets),368)
        tree=ast.parse((FOLDER/'train_mapwise_grpo_joint4.py').read_text())
        updates=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                 and isinstance(n.func.value,ast.Name) and n.func.value.id=='kwargs' and n.func.attr=='update']
        kw={k.arg:ast.literal_eval(k.value) for k in updates[0].keywords if isinstance(k.value,ast.Constant)}
        self.assertEqual(kw['lr_scheduler_type'],'constant')
        self.assertEqual(kw['warmup_steps'],0)
        self.assertEqual(kw['num_iterations'],1)

if __name__=='__main__':unittest.main()
