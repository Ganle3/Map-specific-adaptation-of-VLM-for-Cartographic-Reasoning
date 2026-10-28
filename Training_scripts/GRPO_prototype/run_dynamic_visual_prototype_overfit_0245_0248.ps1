param(
    [int]$Epochs = 10,
    [int]$GroupSize = 16,
    [int]$MaxNewTokens = 800,
    [int]$NumPrototypes = 9,
    [switch]$Resume,
    [string]$WandbProject = "mapqa-dynamic-visual-prototype-overfit",
    [ValidateSet("online", "offline", "disabled")][string]$WandbMode = "online"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$output = Join-Path $vlmRoot "Training_outputs\GRPO_dynamic_visual_prototype_overfit_0245_0248"
$resumeArgs = @()
if ($Resume) {
    $checkpoint = Get-ChildItem -LiteralPath $output -Filter 'dynamic_visual_policy_epoch_*.pt' -File |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($null -eq $checkpoint) { throw "Resume requested but no epoch checkpoint exists below: $output" }
    $resumeArgs = @("--resume-from-checkpoint", $checkpoint.FullName)
}

& $python (Join-Path $PSScriptRoot "train_dynamic_visual_prototype_append.py") `
    --dataset (Join-Path $vlmRoot "Datasets\When_Where_Crop\visual_easier_1000.json") `
    --mapverse-root (Join-Path $repoRoot "MapVerse\data\imgs") `
    --mapwise-root (Join-Path $vlmRoot "Datasets\mapwise-dataset") `
    --adapter $adapter --output-dir $output --start-index 245 --end-index 248 `
    --epochs $Epochs --group-size $GroupSize --questions-per-update 4 --eval-after-epoch --save-every-epoch `
    --max-new-tokens $MaxNewTokens --look-every 20 --max-looks 9 --global-max-pixels 1000000 `
    --num-prototypes $NumPrototypes --print-policy-update-diagnostics `
    --wandb-project $WandbProject --wandb-mode $WandbMode `
    @resumeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
