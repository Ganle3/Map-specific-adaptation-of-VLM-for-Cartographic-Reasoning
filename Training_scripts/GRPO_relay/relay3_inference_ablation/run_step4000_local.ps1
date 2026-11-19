param(
    [string]$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe",
    [string]$Output = ".\VLM_adaptation\Training_outputs\Prototype\relay3_inference_ablation_step4000",
    [int]$StartIndex = 0,
    [int]$EndIndex = 0
)

$ErrorActionPreference = "Stop"

$Evaluate = ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_inference_ablation\evaluate_ablation.py"
$Analyze = ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_inference_ablation\analyze_ablation.py"
$MapVerseTest = ".\VLM_adaptation\Datasets\Subset_MapVerse\scaling_1500\MapVerse_test_600_quality_screened.json"
$MapWiseTest = ".\VLM_adaptation\Datasets\Processed_Mapwise\Test\mapwise_grpo_test_400.json"
$MapVerseImages = ".\MapVerse\data\imgs"
$MapWiseImages = ".\VLM_adaptation\Datasets\mapwise-dataset"
$RunDir = ".\VLM_adaptation\Training_outputs\Prototype\relay3\checkpoint"

$Required = @(
    $Python, $Evaluate, $Analyze, $MapVerseTest, $MapWiseTest,
    "$RunDir\checkpoint-4000.pt", "$RunDir\checkpoint-4000\adapter_config.json"
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
}

$Common = @(
    "--mapverse-root", $MapVerseImages,
    "--mapwise-root", $MapWiseImages,
    "--run-dir", $RunDir,
    "--checkpoint-step", "4000",
    "--output-dir", $Output,
    "--num-relay-tokens", "1",
    "--max-new-tokens", "500",
    "--max-pixels", "1000000",
    "--variants", "full", "relay_off", "no_image_read", "no_current_x_read", "static_relay",
    "--start-index", "$StartIndex"
)
if ($EndIndex -gt 0) { $Common += @("--end-index", "$EndIndex") }

Write-Host "`n=== MapVerse test600 ===" -ForegroundColor Cyan
& $Python $Evaluate --dataset-name mapverse --dataset $MapVerseTest @Common
if ($LASTEXITCODE -ne 0) { throw "MapVerse ablation failed" }

Write-Host "`n=== MapWise test400 ===" -ForegroundColor Cyan
& $Python $Evaluate --dataset-name mapwise --dataset $MapWiseTest @Common
if ($LASTEXITCODE -ne 0) { throw "MapWise ablation failed" }

if ($EndIndex -eq 0) {
    Write-Host "`n=== Paired ablation analysis ===" -ForegroundColor Cyan
    & $Python $Analyze --input $Output --output "$Output\analysis" --bootstrap-samples 5000
    if ($LASTEXITCODE -ne 0) { throw "Ablation analysis failed" }
    Write-Host "Complete: $Output\analysis\report.md" -ForegroundColor Green
} else {
    Write-Host "Partial interval complete; analysis requires complete test sets." -ForegroundColor Yellow
}

