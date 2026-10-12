#!/bin/bash
# MapWise test-400, greedy Accuracy at a 10,000-token cap.
# Submit independently: sbatch evaluate_mapverse_checkpoint3000_test400_unlimited.sbatch
#SBATCH --account=es_ikg
#SBATCH --partition=gpupr.120
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --job-name=mapverse3000_mw400_unlimited
#SBATCH --output=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out
#SBATCH --error=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out

set -euo pipefail
REPO="${REPO:-$HOME/VLM_adaptation}"
QA="${MAPWISE_QA:-$REPO/Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json}"
IMG="${IMG_MAPWISE:-$REPO/Datasets/mapwise-dataset}"
EVAL="$REPO/Evaluation_scripts/GRPO_ablation/mapwise_evaluation_exact.py"
SUMMARIZE="$REPO/Evaluation_scripts/GRPO_ablation/summarize_final_answer_evaluation.py"
EVALUATE="$REPO/Training_scripts/Debug_GRPO/evaluate_grpo.py"
ADAPTER="${ADAPTER:-$SCRATCH/thesis/runs/mapverse_train1000_reviewed_from_baseline1500_5000_tokens500_16399855/checkpoint-3000}"
SIF="${SIF:-$SCRATCH/containers/ablationgrpo.sif}"
RUN_DIR="${OUTPUT_ROOT:-$SCRATCH/thesis/runs}/mapverse_checkpoint3000_test400_unlimited_${SLURM_JOB_ID}"
OUTPUT_SUBDIR="evaluation_test400_checkpoint3000_unlimited_tokens"

for required in "$QA" "$IMG" "$EVAL" "$SUMMARIZE" "$EVALUATE" "$SIF" \
  "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors"; do
  test -e "$required" || { echo "Missing required path: $required" >&2; exit 2; }
done
mkdir -p "$RUN_DIR" "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp"
CONTAINER=(apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind "$HOME:$HOME"
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface" --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1 --env TMPDIR="$SCRATCH/thesis/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$SIF")

srun "${CONTAINER[@]}" python -u "$EVALUATE" \
  --run-dir "$RUN_DIR" --qa-json "$QA" --image-root "$IMG" \
  --evaluation-script "$EVAL" --max-new-tokens 10000 \
  --min-pixels 65536 --max-pixels 1000000 --sampling-draws 0 \
  --baseline-only --baseline-label checkpoint-3000 --baseline-adapter-path "$ADAPTER" \
  --output-subdir "$OUTPUT_SUBDIR"

srun "${CONTAINER[@]}" python -u "$SUMMARIZE" \
  --responses-jsonl "$RUN_DIR/$OUTPUT_SUBDIR/checkpoint-3000/responses.jsonl" \
  --output-dir "$RUN_DIR/$OUTPUT_SUBDIR/checkpoint-3000"
echo "COMPLETE: $RUN_DIR/$OUTPUT_SUBDIR/checkpoint-3000"
