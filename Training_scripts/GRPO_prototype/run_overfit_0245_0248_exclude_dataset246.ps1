param(
    [int]$Epochs = 10,
    [int]$GroupSize = 16,
    [int]$MaxNewTokens = 800,
    [int]$PolicyHiddenLayer = -1,
    [switch]$Resume,
    [string]$WandbProject = "mapqa-region-overfit",
    [ValidateSet("online", "offline", "disabled")][string]$WandbMode = "online"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$outputLeaf = if ($PolicyHiddenLayer -eq -1) {
    "GRPO_prototype_overfit_0245_0248_exclude_dataset246"
} else {
    "GRPO_prototype_overfit_0245_0248_exclude_dataset246_layer$PolicyHiddenLayer"
}
$output = Join-Path $vlmRoot "Training_outputs\$outputLeaf"
$resumeArgs = @()
if ($Resume) {
    $checkpoint = Get-ChildItem -LiteralPath $output -Filter 'region_policy_epoch_*.pt' -File |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $checkpoint) {
        throw "Resume requested but no epoch checkpoint exists below: $output"
    }
    $resumeArgs = @("--resume-from-checkpoint", $checkpoint.FullName)
    Write-Host "Resuming from $($checkpoint.FullName); -Epochs means additional epochs."
}

& $python (Join-Path $PSScriptRoot "train.py") `
    --dataset (Join-Path $vlmRoot "Datasets\When_Where_Crop\visual_easier_1000.json") `
    --mapverse-root (Join-Path $repoRoot "MapVerse\data\imgs") `
    --mapwise-root (Join-Path $vlmRoot "Datasets\mapwise-dataset") `
    --adapter $adapter `
    --output-dir $output `
    --start-index 245 --end-index 248 --exclude-indices 246 `
    --epochs $Epochs --group-size $GroupSize --questions-per-update 4 --eval-after-epoch --save-every-epoch `
    --stop-at-eval-accuracy 1.0 `
    --max-new-tokens $MaxNewTokens --look-every 20 --max-looks 9 `
    --policy-hidden-layer $PolicyHiddenLayer `
    --global-max-pixels 1000000 --crop-max-pixels 1000000 `
    --print-policy-update-diagnostics `
    --wandb-project $WandbProject --wandb-mode $WandbMode `
    @resumeArgs

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
