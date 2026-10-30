param(
    [int]$Epochs = 50,
    [int]$GroupSize = 4,
    [int]$MaxNewTokens = 500,
    [int]$NumRelayTokens = 2,
    [double]$RelayLearningRate = 0.002,
    [double]$AdapterLearningRate = 0.000005,
    [string]$WandbProject = "mapqa-image-question-shared-relay-overfit",
    [ValidateSet("online", "offline", "disabled")][string]$WandbMode = "online"
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$output = Join-Path $vlmRoot "Training_outputs\GRPO_image_question_shared_relay_overfit_0245_0248"

& $python (Join-Path $PSScriptRoot "train.py") `
    --dataset (Join-Path $vlmRoot "Datasets\When_Where_Crop\visual_easier_1000.json") `
    --mapverse-root (Join-Path $repoRoot "MapVerse\data\imgs") `
    --mapwise-root (Join-Path $vlmRoot "Datasets\mapwise-dataset") `
    --adapter $adapter --output-dir $output `
    --start-index 245 --end-index 248 `
    --epochs $Epochs --group-size $GroupSize --questions-per-update 4 `
    --max-new-tokens $MaxNewTokens --max-pixels 1000000 `
    --num-relay-tokens $NumRelayTokens `
    --learning-rate $RelayLearningRate `
    --adapter-learning-rate $AdapterLearningRate --adapter-weight-decay 0.1 `
    --num-policy-updates 2 --off-policy-mask-threshold 0.02 `
    --eval-every-epochs 10 `
    --save-every-epoch --wandb-project $WandbProject `
    --wandb-run-name "image-question-shared-relay-r$NumRelayTokens" --wandb-mode $WandbMode
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }


