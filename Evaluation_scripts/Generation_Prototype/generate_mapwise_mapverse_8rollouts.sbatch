#!/bin/bash
#SBATCH --account=es_ikg
#SBATCH --partition=gpupr.120
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --job-name=unified_8rollouts
#SBATCH --output=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out
#SBATCH --error=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out

set -euo pipefail

REPO="$HOME/VLM_adaptation"
SCRIPT="$REPO/Evaluation_scripts/Generation_Prototype/generate_unified_trajectories.py"
MAPWISE_QA="$REPO/Datasets/Processed_Mapwise/scaling_1000/mapwise_scale1000_train.json"
MAPVERSE_QA="$REPO/Datasets/Subset_MapVerse/scaling_1500/MapVerse_train_1000_reviewed.json"
MAPWISE_IMAGES="$REPO/Datasets/mapwise-dataset"
MAPVERSE_IMAGES="$HOME/MapVerse/data/imgs"
ADAPTER="$SCRATCH/thesis/runs/mapverse_train1000_reviewed_from_mapverse1500_5000_tokens500_15683546/checkpoint-3000"
OUTPUT="$SCRATCH/thesis/runs/unified_mapwise1000_mapverse1000_checkpoint3000_rollouts8"
SIF="$SCRATCH/containers/ablationgrpo.sif"

for required in "$SCRIPT" "$MAPWISE_QA" "$MAPVERSE_QA" "$MAPWISE_IMAGES" \
  "$MAPVERSE_IMAGES" "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" "$SIF"; do
  test -e "$required" || { echo "Missing required path: $required" >&2; exit 2; }
done

mkdir -p "$OUTPUT" "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp"

CONTAINER=(
  apptainer exec --nv
  --bind "$SCRATCH:$SCRATCH"
  --bind "$HOME:$HOME"
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env TMPDIR="$SCRATCH/thesis/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "$SIF"
)

srun "${CONTAINER[@]}" python -u "$SCRIPT" \
  --mapwise-json "$MAPWISE_QA" \
  --mapwise-image-root "$MAPWISE_IMAGES" \
  --mapverse-json "$MAPVERSE_QA" \
  --mapverse-image-root "$MAPVERSE_IMAGES" \
  --adapter-path "$ADAPTER" \
  --output-dir "$OUTPUT" \
  --num-rollouts 8 \
  --temperature 0.8 \
  --top-p 0.95 \
  --top-k 0 \
  --max-new-tokens 1536 \
  --seed 3407 \
  --min-pixels 65536 \
  --max-pixels 1000000

echo "COMPLETE: $OUTPUT/unified_trajectories.json"
