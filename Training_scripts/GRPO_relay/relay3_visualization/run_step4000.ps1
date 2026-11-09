param(
    [string]$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe",
    [string]$Output = ".\VLM_adaptation\Training_outputs\Prototype\relay3_visualization_step4000",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$arguments = @(
    ".\VLM_adaptation\Training_scripts\GRPO_relay\relay3_visualization\run_visualization.py",
    "--output", $Output,
    "--checkpoint-step", "4000",
    "--max-new-tokens", "500",
    "--max-pixels", "1000000",
    "--num-relay-tokens", "1"
)
if ($Overwrite) { $arguments += "--overwrite" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

