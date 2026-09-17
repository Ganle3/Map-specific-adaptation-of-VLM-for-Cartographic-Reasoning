#!/bin/bash
#SBATCH --account=es_ikg
#SBATCH --job-name=joint-online-grad
#SBATCH --array=0-1
#SBATCH --gpus=rtx_4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=joint-gradient-%A_%a.out
set -euo pipefail

# Run from anywhere; upload the complete repository to this path first.
REPO="${REPO:-$HOME/VLM_adaptation}"
CONTAINER="${CONTAINER:-$SCRATCH/containers/ablationgrpo.sif}"
SIZES=(20 44)
SIZE="${SIZES[$SLURM_ARRAY_TASK_ID]}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-3407}"
RESULT_ROOT="${RESULT_ROOT:-$SCRATCH/thesis/diagnostics/joint_gradient_${SLURM_ARRAY_JOB_ID}}"
RUN_DIR="$RESULT_ROOT/joint${SIZE}"
mkdir -p "$RESULT_ROOT" "$SCRATCH/thesis/cache/tmp" "$RESULT_ROOT/wandb"
export OMP_NUM_THREADS=8
export WANDB_DIR="$RESULT_ROOT/wandb"
export WANDB_PROJECT="${WANDB_PROJECT:-MapWise-Online-Gradient}"
EXTRA=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then EXTRA+=(--wandb); fi

run_container() {
  srun apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind "$HOME:$HOME" \
    --env HF_HOME="$SCRATCH/thesis/cache/huggingface" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    --env TMPDIR="$SCRATCH/thesis/cache/tmp" \
    "$CONTAINER" "$@"
}

# Fail before spending GPU time if the container lacks plotting dependencies.
run_container python -c "import matplotlib, numpy"
status=0
run_container python -u "$REPO/diagnostics/gradient/grpo_batch_gradient.py" \
  --dataset-size "$SIZE" --epochs "$EPOCHS" --seed "$SEED" \
  --history-steps 11 --output-dir "$RUN_DIR" \
  --image-root "${IMAGE_ROOT:-$REPO/Datasets/mapwise-dataset}" "${EXTRA[@]}" || status=$?

# Also plot available partial CSVs after a Python failure; preserve failure exit code.
if [[ -f "$RUN_DIR/batch_metrics.csv" ]]; then
  run_container python "$REPO/diagnostics/visualization/plot_batch_gradient_matrix.py" \
    --results-dir "$RUN_DIR" || { if [[ "$status" == "0" ]]; then status=1; fi; }
fi
echo "Results: $RUN_DIR (exit $status)"
exit "$status"
