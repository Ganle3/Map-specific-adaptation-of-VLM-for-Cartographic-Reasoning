"""Experiment B: native TRL rollout reuse (default two optimizer iterations).

Use --num-iterations 1 for the shared control. The original trainer is untouched.
See README_vislora_reuse.md for budgets, method definitions and Euler workflow.
"""
from _vislora_reuse_common import run_experiment

if __name__ == "__main__":
    run_experiment("iterations")
