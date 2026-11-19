param(
    [string]$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe",
    [string]$Output = ".\VLM_adaptation\Training_outputs\Prototype\relay3_global_qk_step4000"
)

$ErrorActionPreference = "Stop"

$Capture = ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_global_qk_analysis\capture_population.py"
$Analyze = ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_global_qk_analysis\analyze_population.py"
$MapVerseTest = ".\VLM_adaptation\Datasets\Subset_MapVerse\scaling_1500\MapVerse_test_600_quality_screened.json"
$MapWiseTest = ".\VLM_adaptation\Datasets\Processed_Mapwise\Test\mapwise_grpo_test_400.json"
$MapVerseImages = ".\MapVerse\data\imgs"
$MapWiseImages = ".\VLM_adaptation\Datasets\mapwise-dataset"
$AdapterRun = ".\VLM_adaptation\Training_outputs\Prototype\adapter_only\checkpoint"
$RelayRun = ".\VLM_adaptation\Training_outputs\Prototype\relay3\checkpoint"
$AdapterMapVerse = ".\VLM_adaptation\Training_outputs\Prototype\adapter_only\evaluation_mapverse_test600_checkpoint_2000_4000\responses.jsonl"
$AdapterMapWise = ".\VLM_adaptation\Training_outputs\Prototype\adapter_only\evaluation_mapwise_test400_checkpoint_2000_4000\responses.jsonl"
$RelayMapVerse = ".\VLM_adaptation\Training_outputs\Prototype\relay3\evaluation_mapverse_test600_initial_without_relay_2000_4000_5000\responses.jsonl"
$RelayMapWise = ".\VLM_adaptation\Training_outputs\Prototype\relay3\evaluation_mapwise_test400_initial_without_relay_2000_4000_5000\responses.jsonl"

$Required = @(
    $Python, $Capture, $Analyze, $MapVerseTest, $MapWiseTest,
    $AdapterMapVerse, $AdapterMapWise, $RelayMapVerse, $RelayMapWise,
    "$AdapterRun\checkpoint-4000.pt", "$AdapterRun\checkpoint-4000\adapter_config.json",
    "$RelayRun\checkpoint-4000.pt", "$RelayRun\checkpoint-4000\adapter_config.json"
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required path: $Path" }
}

$Common = @(
    "--mapverse-root", $MapVerseImages,
    "--mapwise-root", $MapWiseImages,
    "--checkpoint-step", "4000",
    "--output", $Output,
    "--max-new-tokens", "500",
    "--max-pixels", "1000000",
    "--projection-dim", "64",
    "--key-samples", "32",
    "--projection-seed", "20261117",
    "--projection-steps", "0", "4", "9", "19", "39", "79", "159", "319"
)

function Invoke-Capture {
    param(
        [string]$Variant,
        [string]$DatasetName,
        [string]$Dataset,
        [string]$RunDir,
        [string]$Evaluation,
        [switch]$Relay
    )
    $Arguments = @(
        $Capture,
        "--variant", $Variant,
        "--dataset-name", $DatasetName,
        "--dataset", $Dataset,
        "--run-dir", $RunDir,
        "--evaluation", $Evaluation
    ) + $Common
    if ($Relay) { $Arguments += @("--num-relay-tokens", "1") }
    Write-Host "`n=== $Variant / $DatasetName ===" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Capture failed: $Variant / $DatasetName" }
}

Invoke-Capture -Variant "adapter_only" -DatasetName "mapverse" `
    -Dataset $MapVerseTest -RunDir $AdapterRun -Evaluation $AdapterMapVerse
Invoke-Capture -Variant "adapter_only" -DatasetName "mapwise" `
    -Dataset $MapWiseTest -RunDir $AdapterRun -Evaluation $AdapterMapWise
Invoke-Capture -Variant "relay3" -DatasetName "mapverse" `
    -Dataset $MapVerseTest -RunDir $RelayRun -Evaluation $RelayMapVerse -Relay
Invoke-Capture -Variant "relay3" -DatasetName "mapwise" `
    -Dataset $MapWiseTest -RunDir $RelayRun -Evaluation $RelayMapWise -Relay

Write-Host "`n=== Paired population analysis ===" -ForegroundColor Cyan
& $Python $Analyze `
    --input $Output `
    --output "$Output\analysis" `
    --bootstrap-samples 2000 `
    --early-steps 20 `
    --early-layer-start 8 `
    --early-layer-end 32 `
    --cluster-layers 8 16 24 35
if ($LASTEXITCODE -ne 0) { throw "Population analysis failed" }

Write-Host "`nComplete: $Output\analysis\report.md" -ForegroundColor Green

