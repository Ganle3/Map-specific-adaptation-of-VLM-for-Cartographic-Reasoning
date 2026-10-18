param(
    [int]$TrainingPid,
    [string]$TrainingOutputDir,
    [string]$PythonExe,
    [string]$WorkingDirectory
)

Set-Location -LiteralPath $WorkingDirectory
$log = Join-Path $TrainingOutputDir 'fixed_evaluation_watcher.log'
"Waiting for training PID $TrainingPid at $(Get-Date -Format o)" | Set-Content -LiteralPath $log -Encoding utf8
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
$checkpoint = Join-Path $TrainingOutputDir 'controller_final.pt'
if (-not (Test-Path -LiteralPath $checkpoint)) {
    "Training stopped without controller_final.pt; evaluation not started." | Add-Content -LiteralPath $log -Encoding utf8
    exit 2
}
"Training finished. Starting fixed evaluation at $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
$evaluationDir = Join-Path $TrainingOutputDir 'fixed_evaluation_32rollouts'
& $PythonExe .\Training_scripts\GRPO_lookbetter\evaluate_overfit_checkpoints.py `
    --dataset-json .\Datasets\When_Where_Crop\visual_easier_1000.json `
    --mapverse-image-root ..\MapVerse\data\imgs `
    --mapwise-root .\Datasets\mapwise-dataset `
    --adapter .\Training_outputs\MapVerse_scaling1000_GRPO_from_MapVerse5000_token500\checkpoint-2500 `
    --checkpoint $checkpoint `
    --output-dir $evaluationDir `
    --start-index 245 --end-index 248 --rollouts-per-question 32 `
    --tau0 0.091112 --temperature 0.05 --max-reasoning-tokens 500 `
    --max-looks 10 --cooldown-tokens 20 --eligible-after-tokens 12 --eligible-stride 4 `
    --revision-tokens 0 --attention-mass 0.75 --initial-size-scale 1.5 `
    --roi-key-bias 3.079442 --global-max-pixels 1000000 --seed 20261017 *>> $log
"Evaluation exit code: $LASTEXITCODE at $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
