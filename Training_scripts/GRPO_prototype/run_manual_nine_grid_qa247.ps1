param(
    [int]$MaxNewTokens = 500,
    [int]$LookEvery = 15,
    [int]$MaxLooks = 9,
    [int]$ReplayTailTokens = 0,
    [ValidateSet("chat-turn", "visual-embeds")][string]$AppendMode = "visual-embeds"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$output = Join-Path $vlmRoot "Training_outputs\manual_nine_grid_dataset247.json"

& $python (Join-Path $PSScriptRoot "manual_nine_grid_qa.py") `
    --adapter $adapter `
    --dataset-index 247 `
    --max-new-tokens $MaxNewTokens `
    --look-every $LookEvery `
    --max-looks $MaxLooks `
    --replay-tail-tokens $ReplayTailTokens `
    --append-mode $AppendMode `
    --global-max-pixels 1000000 `
    --crop-max-pixels 1000000 `
    --output $output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
