"""Reject accidental dataset substitutions in the four-QA diagnostic."""
import json
from pathlib import Path

EXPECTED=[('usa',2051),('china',525),('usa',2021),('china',646)]

def validate_dataset(path):
    rows=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if [(r['country'],r['source_index']) for r in rows]!=EXPECTED:
        raise ValueError('Expected source2051 plus the three selected stability controls, in fixed order')
    canonical=Path(__file__).resolve().parents[2]/'Datasets/Processed_Mapwise/Train_Val/mapwise_grpo_joint_debug4_src2051.json'
    if rows!=json.loads(canonical.read_text(encoding='utf-8-sig')):
        raise ValueError('QA metadata differs from the selected diagnostic dataset')
    return rows
