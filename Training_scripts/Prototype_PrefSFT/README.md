# Prototype PrefSFT

First run `precompute_vision_embeddings.py --cache-dir <DIR>` once, then pass the same directory as `--vision-cache-dir` to both `precompute_baseline_scores.py` and `train_prototype_prefsft.py`. This persists final frozen-ViT Z per question and on cache hits skips all ViT blocks; only T_pro, residual and merger/LLM run. The cache must be regenerated after changing the adapter or image preprocessing. `question-batch-size` is the number of questions per optimizer step; `trajectory-micro-batch-size` controls GPU activation memory without changing the question-level pairwise loss.

The model backbone (including the checkpoint-3000 LoRA adapter) is frozen. Checkpoints contain only P, T_pro, alpha, optimizer/scheduler state, args, step and epoch. `visualize/visualize_prototypes.py` saves every prototype/head P-to-patch overlay, head means, prototype cosine and attention-overlap matrices, and residual magnitude.
