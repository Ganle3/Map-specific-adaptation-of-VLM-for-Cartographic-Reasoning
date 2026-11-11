$ErrorActionPreference="Stop"
& "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe" "$PSScriptRoot\run_visualization.py" --checkpoint-step 2000 --overwrite
if($LASTEXITCODE-ne 0){exit $LASTEXITCODE}
