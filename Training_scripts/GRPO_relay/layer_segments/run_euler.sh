#!/bin/bash
# Called by one of the three matched layer-segment sbatch files.
set -euo pipefail

MODE="${RELAY_LAYER_MODE:?Set RELAY_LAYER_MODE in the sbatch wrapper}"
case "$MODE" in
  continuous|reset_shared4|reset_stage1) ;;
  *) echo "Unsupported RELAY_LAYER_MODE=$MODE" >&2; exit 2 ;;
esac

REPO="$HOME/VLM_adaptation"
TRAIN="$REPO/Training_scripts/GRPO_relay/Image_question_CoT_shared_relay/train.py"
EVALUATE="$REPO/Training_scripts/GRPO_relay/Image_question_CoT_shared_relay/evaluate.py"
TRAIN_DATA="$REPO/Datasets/When_Where_Crop/visual_easier_1000.json"
MAPWISE_TEST="$REPO/Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json"
MAPVERSE_TEST="$REPO/Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json"
MAPVERSE_IMAGES="$HOME/MapVerse/data/imgs"
MAPWISE_IMAGES="$REPO/Datasets/mapwise-dataset"
SIF="$SCRATCH/containers/ablationgrpo.sif"
ADAPTER="${ADAPTER:-$SCRATCH/thesis/runs/mapverse_train1000_reviewed_from_mapverse1500_5000_tokens500_15683546/checkpoint-2500}"
RUN_ID="relay3_r4_${MODE}_easier1000_q4r4_ppo2_${SLURM_JOB_ID}"
RUN_DIR="$SCRATCH/thesis/runs/$RUN_ID"

for required in "$TRAIN" "$EVALUATE" "$TRAIN_DATA" "$MAPWISE_TEST" \
  "$MAPVERSE_TEST" "$SIF" "$ADAPTER/adapter_config.json"; do
  test -f "$required" || { echo "Missing required file: $required" >&2; exit 2; }
done
test -d "$MAPVERSE_IMAGES" || { echo "Missing MapVerse images: $MAPVERSE_IMAGES" >&2; exit 2; }
test -d "$MAPWISE_IMAGES" || { echo "Missing MapWise images: $MAPWISE_IMAGES" >&2; exit 2; }

mkdir -p "$RUN_DIR/wandb" "$SCRATCH/thesis/logs" \
  "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp" \
  "$SCRATCH/thesis/cache/wandb" "$SCRATCH/thesis/cache/wandb-data"

CONTAINER=(
  apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind "$HOME:$HOME"
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env TMPDIR="$SCRATCH/thesis/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env WANDB_PROJECT=MapQA-Relay-Ablation
  --env WANDB_RUN_ID="$RUN_ID" --env WANDB_RESUME=never --env WANDB_MODE=online
  --env WANDB_DIR="$RUN_DIR/wandb"
  --env WANDB_CACHE_DIR="$SCRATCH/thesis/cache/wandb"
  --env WANDB_DATA_DIR="$SCRATCH/thesis/cache/wandb-data"
  "$SIF"
)

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo "RELAY_LAYER_MODE=$MODE"
echo "RELAY_PARAMETER_SHAPE=[4,4096]"
echo "TRAIN=1000 easier questions; 4 questions x 4 rollouts; 2 policy updates per generation"

srun "${CONTAINER[@]}" python -u "$TRAIN" \
  --dataset "$TRAIN_DATA" --mapverse-root "$MAPVERSE_IMAGES" --mapwise-root "$MAPWISE_IMAGES" \
  --adapter "$ADAPTER" --output-dir "$RUN_DIR" \
  --max-steps 4000 --group-size 4 --questions-per-update 4 \
  --max-new-tokens 500 --max-pixels 1000000 --num-relay-tokens 4 \
  --relay-layer-mode "$MODE" \
  --learning-rate 0.002 --adapter-learning-rate 5e-6 --adapter-weight-decay 0.1 \
  --clip-ratio 0.2 --num-policy-updates 2 --off-policy-mask-threshold 0.02 \
  --lr-scheduler-type constant_then_cosine --decay-start-step 3000 --decay-final-factor 0.6 \
  --save-steps 1000 --seed 1234 \
  --wandb-project MapQA-Relay-Ablation --wandb-run-name "$RUN_ID" --wandb-mode online

COMMON=(
  --mapverse-root "$MAPVERSE_IMAGES" --mapwise-root "$MAPWISE_IMAGES"
  --adapter "$ADAPTER" --run-dir "$RUN_DIR" --max-new-tokens 500
  --max-pixels 1000000 --num-relay-tokens 4 --relay-layer-mode "$MODE"
)

echo "Training complete; evaluating greedy train initial/final"
srun "${CONTAINER[@]}" python -u "$EVALUATE" \
  --dataset "$TRAIN_DATA" "${COMMON[@]}" --checkpoint-steps 0 4000 \
  --output-subdir evaluation_train_initial_without_relay_0_4000

echo "Evaluating greedy MapWise test400"
srun "${CONTAINER[@]}" python -u "$EVALUATE" \
  --dataset "$MAPWISE_TEST" "${COMMON[@]}" --checkpoint-steps 0 2000 4000 \
  --output-subdir evaluation_mapwise_test400_initial_without_relay_2000_4000

echo "Evaluating greedy MapVerse test600"
srun "${CONTAINER[@]}" python -u "$EVALUATE" \
  --dataset "$MAPVERSE_TEST" "${COMMON[@]}" --checkpoint-steps 0 2000 4000 \
  --output-subdir evaluation_mapverse_test600_initial_without_relay_2000_4000

echo "COMPLETE RUN_DIR=$RUN_DIR"

