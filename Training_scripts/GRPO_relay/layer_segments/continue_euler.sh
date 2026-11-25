#!/bin/bash
# Shared bounded continuation for the timed-out layer-segment runs.
set -euo pipefail

KIND="${RELAY_CONTINUE_KIND:?Set RELAY_CONTINUE_KIND in the sbatch wrapper}"
case "$KIND" in
  shared4)
    MODE=reset_shared4
    ORIGINAL_RUN_ID=relay3_r4_reset_shared4_mapwise1058_xavier_rank32_q4r4_ppo2_21656514
    INITIAL_STAGE=train
    SELF="$SCRATCH/thesis/jobs/resume_relay3_r4_reset_shared4_mapwise1058_4090.sbatch"
    ;;
  stage1)
    MODE=reset_stage1
    ORIGINAL_RUN_ID=relay3_r4_reset_stage1_mapwise1058_xavier_rank32_q4r4_ppo2_21656577
    INITIAL_STAGE=eval_mapverse
    SELF="$SCRATCH/thesis/jobs/resume_relay3_r4_reset_stage1_mapverse_4090.sbatch"
    ;;
  *) echo "Unsupported RELAY_CONTINUE_KIND=$KIND" >&2; exit 2 ;;
esac

STAGE="${RELAY_RESUME_STAGE:-$INITIAL_STAGE}"
CHUNK="${RELAY_RESUME_CHUNK:-0}"
[[ "$CHUNK" =~ ^[0-9]+$ ]] || { echo "Invalid RELAY_RESUME_CHUNK=$CHUNK" >&2; exit 2; }
(( CHUNK < 20 )) || { echo "Continuation limit reached" >&2; exit 2; }
if [[ "$KIND" == stage1 && "$STAGE" != eval_mapverse ]]; then
  echo "Stage1 continuation only supports eval_mapverse" >&2
  exit 2
fi

REPO="$HOME/VLM_adaptation"
TRAIN="$REPO/Training_scripts/GRPO_relay/Image_question_CoT_shared_relay/train.py"
EVALUATE="$REPO/Training_scripts/GRPO_relay/Image_question_CoT_shared_relay/evaluate.py"
TRAIN_DATA="$REPO/Datasets/Processed_Mapwise/mapwise_filtered_1058.json"
MAPWISE_TEST="$REPO/Datasets/Processed_Mapwise/Test/mapwise_grpo_test_400.json"
MAPVERSE_TEST="$REPO/Datasets/Subset_MapVerse/scaling_1500/MapVerse_test_600_quality_screened.json"
MAPVERSE_IMAGES="$HOME/MapVerse/data/imgs"
MAPWISE_IMAGES="$REPO/Datasets/mapwise-dataset"
SIF="$SCRATCH/containers/ablationgrpo.sif"
RUN_DIR="$SCRATCH/thesis/runs/$ORIGINAL_RUN_ID"
INIT_ADAPTER="$RUN_DIR/initial_adapter_rank32"
TRAIN_OUTPUT=evaluation_train_initial_without_relay_0_4000
MAPWISE_OUTPUT=evaluation_mapwise_test400_initial_without_relay_2000_4000
MAPVERSE_OUTPUT=evaluation_mapverse_test600_initial_without_relay_2000_4000

for required in "$SELF" "$TRAIN" "$EVALUATE" "$TRAIN_DATA" "$MAPWISE_TEST" \
  "$MAPVERSE_TEST" "$SIF" "$INIT_ADAPTER/adapter_config.json" \
  "$INIT_ADAPTER/adapter_model.safetensors"; do
  test -f "$required" || { echo "Missing required file: $required" >&2; exit 2; }
done
if [[ "$KIND" == shared4 ]]; then
  for required in "$RUN_DIR/checkpoint-3000.pt" \
    "$RUN_DIR/checkpoint-3000/adapter_config.json" \
    "$RUN_DIR/checkpoint-3000/adapter_model.safetensors" \
    "$RUN_DIR/checkpoint-2000.pt" \
    "$RUN_DIR/checkpoint-2000/adapter_config.json" \
    "$RUN_DIR/checkpoint-2000/adapter_model.safetensors"; do
    test -f "$required" || { echo "Missing required file: $required" >&2; exit 2; }
  done
else
  for required in "$RUN_DIR/checkpoint-2000.pt" \
    "$RUN_DIR/checkpoint-4000.pt" \
    "$RUN_DIR/checkpoint-2000/adapter_config.json" \
    "$RUN_DIR/checkpoint-2000/adapter_model.safetensors" \
    "$RUN_DIR/checkpoint-4000/adapter_config.json" \
    "$RUN_DIR/checkpoint-4000/adapter_model.safetensors"; do
    test -f "$required" || { echo "Missing required file: $required" >&2; exit 2; }
  done
fi
test -d "$MAPVERSE_IMAGES" || { echo "Missing MapVerse images" >&2; exit 2; }
test -d "$MAPWISE_IMAGES" || { echo "Missing MapWise images" >&2; exit 2; }

mkdir -p "$RUN_DIR/wandb-resume-$SLURM_JOB_ID" \
  "$SCRATCH/thesis/cache/huggingface" "$SCRATCH/thesis/cache/tmp" \
  "$SCRATCH/thesis/cache/wandb" "$SCRATCH/thesis/cache/wandb-data"

RESUME_RUN_ID="${ORIGINAL_RUN_ID}_resume_${SLURM_JOB_ID}"
CONTAINER=(
  apptainer exec --nv --bind "$SCRATCH:$SCRATCH" --bind "$HOME:$HOME"
  --env HF_HOME="$SCRATCH/thesis/cache/huggingface"
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env TMPDIR="$SCRATCH/thesis/cache/tmp"
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env WANDB_PROJECT=MapQA-Relay-Ablation
  --env WANDB_RUN_ID="$RESUME_RUN_ID" --env WANDB_RESUME=never --env WANDB_MODE=online
  --env WANDB_DIR="$RUN_DIR/wandb-resume-$SLURM_JOB_ID"
  --env WANDB_CACHE_DIR="$SCRATCH/thesis/cache/wandb"
  --env WANDB_DATA_DIR="$SCRATCH/thesis/cache/wandb-data"
  "$SIF"
)

submit_next() {
  local next_stage="$1" next_chunk=$((CHUNK + 1)) job_id
  (( next_chunk < 20 )) || { echo "Continuation limit reached" >&2; exit 3; }
  job_id=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,RELAY_RESUME_STAGE=${next_stage},RELAY_RESUME_CHUNK=${next_chunk}" "$SELF")
  echo "QUEUED_NEXT stage=$next_stage job_id=$job_id after=$SLURM_JOB_ID"
}

# Leave 12 hours before Slurm's 120-hour limit to queue the next allocation.
run_bounded() {
  local status
  set +e
  timeout --signal=TERM --kill-after=120s 108h "$@"
  status=$?
  set -e
  if (( status == 124 )); then return 124; fi
  if (( status != 0 )); then
    echo "Command failed with exit code $status; continuation stopped" >&2
    exit "$status"
  fi
  return 0
}

run_eval() {
  local dataset="$1" output="$2" next_stage="$3" summary
  summary="$RUN_DIR/$output/accuracy.json"
  shift 3
  if [[ -f "$summary" ]]; then
    echo "Evaluation already complete: $summary"
  elif run_bounded srun "${CONTAINER[@]}" python -u "$EVALUATE" \
      --dataset "$dataset" --mapverse-root "$MAPVERSE_IMAGES" --mapwise-root "$MAPWISE_IMAGES" \
      --adapter "$INIT_ADAPTER" --run-dir "$RUN_DIR" --checkpoint-steps "$@" \
      --output-subdir "$output" --max-new-tokens 500 --max-pixels 1000000 \
      --num-relay-tokens 4 --relay-layer-mode "$MODE"; then
    test -f "$summary" || { echo "Evaluator exited without $summary" >&2; exit 4; }
  else
    echo "Evaluation time budget reached; partial responses remain in $RUN_DIR/$output/responses.jsonl"
    submit_next "$STAGE"
    exit 0
  fi
  if [[ "$next_stage" == done ]]; then
    echo "COMPLETE RUN_DIR=$RUN_DIR"
  else
    submit_next "$next_stage"
  fi
}

echo "KIND=$KIND STAGE=$STAGE CHUNK=$CHUNK RUN_DIR=$RUN_DIR"
echo "Each allocation stops at 108 hours; evaluator resumes from existing responses.jsonl"

case "$STAGE" in
  train)
    [[ "$KIND" == shared4 ]] || { echo "Train stage is only for shared4" >&2; exit 2; }
    if [[ -f "$RUN_DIR/checkpoint-4000.pt" && \
          -f "$RUN_DIR/checkpoint-4000/adapter_config.json" && \
          -f "$RUN_DIR/checkpoint-4000/adapter_model.safetensors" ]]; then
      echo "Final checkpoint already present; skipping training"
      submit_next eval_train
      exit 0
    fi
    if run_bounded srun "${CONTAINER[@]}" python -u "$TRAIN" \
      --dataset "$TRAIN_DATA" --mapverse-root "$MAPVERSE_IMAGES" --mapwise-root "$MAPWISE_IMAGES" \
      --adapter "$RUN_DIR/checkpoint-3000" --expected-adapter-rank 32 \
      --output-dir "$RUN_DIR" --resume-from-checkpoint "$RUN_DIR/checkpoint-3000.pt" \
      --max-steps 4000 --group-size 4 --questions-per-update 4 \
      --max-new-tokens 500 --max-pixels 1000000 --num-relay-tokens 4 \
      --relay-layer-mode "$MODE" --relay-init xavier_normal \
      --learning-rate 0.002 --adapter-learning-rate 5e-6 --adapter-weight-decay 0.1 \
      --clip-ratio 0.2 --num-policy-updates 2 --off-policy-mask-threshold 0.02 \
      --lr-scheduler-type constant_then_cosine --decay-start-step 3000 --decay-final-factor 0.6 \
      --save-steps 1000 --seed 1234 \
      --wandb-project MapQA-Relay-Ablation --wandb-run-name "$RESUME_RUN_ID" --wandb-mode online; then
      test -f "$RUN_DIR/checkpoint-4000.pt" || {
        echo "Training exited without checkpoint-4000.pt" >&2; exit 4;
      }
      test -f "$RUN_DIR/checkpoint-4000/adapter_model.safetensors" || {
        echo "Training exited without checkpoint-4000 adapter" >&2; exit 4;
      }
      submit_next eval_train
    else
      echo "Training reached its time budget before checkpoint-4000"
      exit 5
    fi
    ;;
  eval_train)
    [[ "$KIND" == shared4 ]] || { echo "Train evaluation is only for shared4" >&2; exit 2; }
    run_eval "$TRAIN_DATA" "$TRAIN_OUTPUT" eval_mapwise 0 4000
    ;;
  eval_mapwise)
    [[ "$KIND" == shared4 ]] || { echo "MapWise evaluation is only for shared4" >&2; exit 2; }
    run_eval "$MAPWISE_TEST" "$MAPWISE_OUTPUT" eval_mapverse 0 2000 4000
    ;;
  eval_mapverse)
    run_eval "$MAPVERSE_TEST" "$MAPVERSE_OUTPUT" done 0 2000 4000
    ;;
  *) echo "Unknown RELAY_RESUME_STAGE=$STAGE" >&2; exit 2 ;;
esac
