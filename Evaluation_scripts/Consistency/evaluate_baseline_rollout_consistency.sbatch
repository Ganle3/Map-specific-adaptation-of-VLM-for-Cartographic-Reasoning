#!/bin/bash
# GRPO-style baseline consistency on the held-out MapWise-400 and MapVerse-600 tests.
#SBATCH --account=es_ikg
#SBATCH --partition=gpupr.120
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --job-name=baseline_consistency
#SBATCH --output=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out
#SBATCH --error=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out

set -euo pipefail
REPO="${REPO:-$HOME/VLM_adaptation}"
MAPWISE_QA="${MAPWISE_QA:-$REPO/Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json}"
MAPVERSE_QA="${MAPVERSE_QA:-$REPO/Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json}"
IMG_MAPWISE="${IMG_MAPWISE:-$REPO/Datasets/mapwise-dataset}"
IMG_MAPVERSE="${IMG_MAPVERSE:-$HOME/MapVerse/data/imgs}"
MAPWISE_EVAL="$REPO/Evaluation_scripts/GRPO_ablation/mapwise_evaluation_exact.py"
MAPVERSE_EVAL="$REPO/Evaluation_scripts/GRPO_ablation/mapverse_evaluation_exact.py"
EVALUATE="$REPO/Training_scripts/Debug_GRPO/evaluate_grpo.py"
ANALYZE="$REPO/Evaluation_scripts/Consistency/analyze_grpo_rollout_consistency.py"
SIF="${SIF:-$SCRATCH/containers/ablationgrpo.sif}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
RUN_DIR="${OUTPUT_ROOT:-$SCRATCH/thesis/runs}/baseline_test_consistency_${SLURM_JOB_ID}"

for required in "$MAPWISE_QA" "$MAPVERSE_QA" "$IMG_MAPWISE" "$IMG_MAPVERSE" \
  "$MAPWISE_EVAL" "$MAPVERSE_EVAL" "$EVALUATE" "$ANALYZE" "$SIF"; do
  test -e "$required" || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$RUN_DIR" "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp"

CONTAINER=(apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind "$HOME:$HOME"
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface" --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1 --env TMPDIR="$SCRATCH/thesis/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$SIF")

run_test() {
  local name="$1" qa="$2" images="$3" evaluator="$4"
  local evaluation_dir="rollouts_${name}_baseline_4x_tokens${MAX_NEW_TOKENS}"
  srun "${CONTAINER[@]}" python -u "$EVALUATE" \
    --run-dir "$RUN_DIR" --qa-json "$qa" --image-root "$images" \
    --evaluation-script "$evaluator" --max-new-tokens "$MAX_NEW_TOKENS" \
    --min-pixels 65536 --max-pixels 1000000 \
    --sampling-draws 4 --skip-greedy --baseline-only \
    --baseline-label baseline \
    --output-subdir "$evaluation_dir"
  srun "${CONTAINER[@]}" python -u "$ANALYZE" \
    --responses-jsonl "$RUN_DIR/$evaluation_dir/baseline/responses.jsonl" \
    --evaluation-script "$evaluator" \
    --output-dir "$RUN_DIR/consistency_${name}_baseline_4x_tokens${MAX_NEW_TOKENS}" \
    --expected-draws 4
}

run_test mapwise_test400 "$MAPWISE_QA" "$IMG_MAPWISE" "$MAPWISE_EVAL"
run_test mapverse_test600 "$MAPVERSE_QA" "$IMG_MAPVERSE" "$MAPVERSE_EVAL"
echo "COMPLETE: $RUN_DIR"
