param(
    [int]$Epochs = 50,
    [int]$GroupSize = 16,
    [int]$MaxNewTokens = 500,
    [string]$WandbProject = "mapqa-prototype-attention-overfit",
    [ValidateSet("online", "offline", "disabled")][string]$WandbMode = "online"
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$vlmRoot = Join-Path $repoRoot "VLM_adaptation"
$python = Join-Path $env:USERPROFILE ".conda\envs\VisionGRPO\python.exe"
$adapter = Join-Path $vlmRoot "Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500"
$output = Join-Path $vlmRoot "Training_outputs\GRPO_prototype_attention_after_image_overfit_0245_0248"
& $python (Join-Path $PSScriptRoot "train.py") `
    --placement after_image `
    --dataset (Join-Path $vlmRoot "Datasets\When_Where_Crop\visual_easier_1000.json") `
    --mapverse-root (Join-Path $repoRoot "MapVerse\data\imgs") `
    --mapwise-root (Join-Path $vlmRoot "Datasets\mapwise-dataset") `
    --adapter $adapter --output-dir $output `
    --start-index 245 --end-index 248 `
    --epochs $Epochs --group-size $GroupSize --questions-per-update 4 `
    --max-new-tokens $MaxNewTokens --max-pixels 1000000 `
    --num-prototypes 9 --attention-heads 8 --learning-rate 0.002 `
    --save-every-epoch --wandb-project $WandbProject `
    --wandb-run-name "prototype-after-image" --wandb-mode $WandbMode
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
