"""Experiment A: ExGRPO-inspired success replay with current-policy entropy selection.

Not a full ExGRPO reproduction: keeps fixed QA exposure and baseline scaling.
See README_vislora_reuse.md before interpreting the reward curves.
"""
from _vislora_reuse_common import run_experiment

if __name__ == "__main__":
    run_experiment("replay")
