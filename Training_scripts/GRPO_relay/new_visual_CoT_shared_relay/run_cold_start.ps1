param(
    [int]$Epochs = 30,
    [string]$TeacherRoot = ".\VLM_adaptation\Training_outputs\gradattn_teacher",
    [string]$OutputDir = ".\VLM_adaptation\Training_outputs\GRPO_relay\relay4_cold_start"
)
$ErrorActionPreference = "Stop"
$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

& $Python "$Here\prepare_cold_start_cache.py" `
    --teacher-root $TeacherRoot --correct-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$Here\train_cold_start.py" `
    --teacher-root $TeacherRoot --output-dir $OutputDir --epochs $Epochs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$Here\demo_qa318.py" `
    --policy-checkpoint "$OutputDir\relay4_cold_start_best.pt" `
    --output-dir "$OutputDir\qa318_cold_start_test"
exit $LASTEXITCODE
