"""Training-only When--Where visual observation controller.

The frozen Qwen3-VL / reasoning-adapter pipeline deliberately lives outside
this package.  This package owns only data handling, stochastic controller
actions, trajectory bookkeeping, and the GRPO update.
"""

