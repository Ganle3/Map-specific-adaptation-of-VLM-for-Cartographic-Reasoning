$ErrorActionPreference = "Stop"
$Python = "C:\Users\junyhuang\.conda\envs\VisionGRPO\python.exe"
$Script = Join-Path $PSScriptRoot "analyze_test_maps.py"

& $Python $Script `
  --checkpoint-step 2000 `
  --mapverse-images 100 `
  --mapwise-images 100 `
  --top-patches 6 `
  --clusters 12 `
  --overwrite

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
