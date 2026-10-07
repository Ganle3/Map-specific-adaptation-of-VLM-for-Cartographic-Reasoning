#!/bin/bash
#SBATCH --account=es_ikg
#SBATCH --partition=gpupr.120
#SBATCH --gpus=a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --job-name=mapwise_consistency
#SBATCH --output=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out
#SBATCH --error=/cluster/home/junyhuang/VLM_adaptation/slurm-%j.out

# Example (all experiment-specific locations are supplied at submission time):
# TEST_QA=/path/to/test.json MAPWISE_IMAGES=/path/to/mapwise-dataset \
# ADAPTER=/path/to/checkpoint-3000 OUTPUT=/path/to/output \
# SIF=/path/to/container.sif sbatch evaluate_mapwise_rollout_consistency.sbatch
set -euo pipefail

: "${TEST_QA:?Set TEST_QA to the Test JSON path}"
: "${MAPWISE_IMAGES:?Set MAPWISE_IMAGES to the MapWise image root}"
: "${ADAPTER:?Set ADAPTER to the adapted PEFT checkpoint}"
: "${OUTPUT:?Set OUTPUT to an empty or resumable output directory}"
: "${SIF:?Set SIF to the Apptainer image}"

REPO="${REPO:-$HOME/VLM_adaptation}"
SCRIPT="$REPO/Evaluation_scripts/Consistency/evaluate_mapwise_rollout_consistency.py"

for required in "$SCRIPT" "$TEST_QA" "$MAPWISE_IMAGES" "$ADAPTER/adapter_config.json" "$SIF"; do
  test -e "$required" || { echo "Missing required path: $required" >&2; exit 2; }
done

mkdir -p "$OUTPUT" "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp"

apptainer exec --nv \
  --bind "$SCRATCH:$SCRATCH" \
  --bind "$HOME:$HOME" \
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TMPDIR="$SCRATCH/thesis/cache/tmp" \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$SIF" python -u "$SCRIPT" \
    --qa-json "$TEST_QA" \
    --image-root "$MAPWISE_IMAGES" \
    --adapter-path "$ADAPTER" \
    --output-dir "$OUTPUT" \
    --num-rollouts 4 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 0 \
    --max-new-tokens 1536 \
    --seed 3407 \
    --min-pixels 65536 \
    --max-pixels 1000000

echo "COMPLETE: $OUTPUT/summary.json"
