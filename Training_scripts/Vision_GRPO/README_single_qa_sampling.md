# Fixed-model stochastic correctness comparison

`evaluate_single_qa_sampling.py` evaluates raw baseline and checkpoint-40 from
single-QA training 14040058. Same source2051 question, same image/prompt, 100
separate batch-size-one generations per model, seeds 900000–900099 reset before
each generation. No training and no correctness-based filtering or retries.
Only one model is resident at a time. Uses the existing evaluation loader
(NF4 base, BF16 compute, PEFT inference adapter behavior), not an altered training
precision. Generation: sample=true, temperature=.8, top_p=.95, top_k=0,
min_p=None, repetition_penalty=1, max_new_tokens=1536, thinking=auto, beams=1.
Saved generation configurations document other model defaults.

Each JSONL row preserves the actual response, seed, exact scoring result, length,
and termination based on generated EOS tokens. `strict_exact_match` follows
the existing independent evaluator; `terminated_correct` additionally requires
EOS termination. Truncated samples remain in the denominator. Runtime errors
abort the job instead of silently dropping or resampling observations.

`sampling_accuracy.csv` has one row per completed model. `comparison.json` is
written only after both models complete. Reports Wilson 95% intervals per model,
absolute accuracy difference and a seeded paired percentile bootstrap interval
(10,000 resamples). Same seeds couple random draws, not identical reasoning paths.
These intervals describe sampling uncertainty for this QA, not variation across
training seeds, generalization, or proof of a universal training effect.
No partial-run resume; use a fresh output directory after a failure. Checkpoint
choice is predeclared (step40), not selected from this sampling evaluation.

Repository files sync through GitHub. Local uploader
`Euler_results/VisionLoRA/debug50/jobs/upload_single_qa_sampling.ps1` uploads only
the `.sbatch` to scratch. Output directory:
`$SCRATCH/thesis/evaluations/single_src2051_sampling_JOBID/`.
The A100 job reserves up to 8 hours for 200 sequential responses and model loads.

Run CPU statistics checks with `python test_single_qa_sampling.py`.
GPU inference needs validation in the Euler container.
