$ErrorActionPreference = "Stop"
$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
& $Python "$Here\analyze_test_maps.py" --mapverse-images 100 --mapwise-images 100 --overwrite
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

