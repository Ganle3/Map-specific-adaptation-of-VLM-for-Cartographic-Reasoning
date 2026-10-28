param(
    [int]$Epochs = 10,
    [int]$GroupSize = 16,
    [int]$MaxNewTokens = 800,
    [int]$PolicyHiddenLayer = -1,
    [string]$WandbProject = "mapqa-region-direct-visual-overfit",
    [ValidateSet("online", "offline", "disabled")][string]$WandbMode = "online"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$output = Join-Path $vlmRoot "Training_outputs\GRPO_prototype_direct_visual_overfit_0245_0248"

& $python (Join-Path $PSScriptRoot "train_direct_visual.py") `
    --dataset (Join-Path $vlmRoot "Datasets\When_Where_Crop\visual_easier_1000.json") `
    --mapverse-root (Join-Path $repoRoot "MapVerse\data\imgs") `
    --mapwise-root (Join-Path $vlmRoot "Datasets\mapwise-dataset") `
    --adapter $adapter `
    --output-dir $output `
    --start-index 245 --end-index 248 `
    --epochs $Epochs --group-size $GroupSize --questions-per-update 4 --eval-after-epoch --save-every-epoch `
    --max-new-tokens $MaxNewTokens --look-every 20 --max-looks 9 `
    --policy-hidden-layer $PolicyHiddenLayer `
    --global-max-pixels 1000000 --crop-max-pixels 1000000 `
    --print-policy-update-diagnostics `
    --wandb-project $WandbProject --wandb-mode $WandbMode

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
