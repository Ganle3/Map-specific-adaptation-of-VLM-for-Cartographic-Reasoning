"""Canonical joint-debug entry point.

The shared implementation serves 4-, 20-, and 44-QA diagnostics; dataset and
optimizer-update budget are supplied by each sbatch file. The old module name
remains available for historical jobs and reproducibility.
"""
from train_mapwise_grpo_joint4 import main

if __name__ == "__main__":
    main()
