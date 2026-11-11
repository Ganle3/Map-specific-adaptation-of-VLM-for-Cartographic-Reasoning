$ErrorActionPreference = "Stop"
$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python "$Here\run_visualization.py" --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

